"""
training.train.py
Main entry point for training all model types.

Usage:
    python -m training.train --model_type polychain --fold 0 --person person1
    python -m training.train --model_type xgb --fold 0
    python -m training.train --model_type gcn --fold 0
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError:
    PyGDataLoader = None

from .train_utils import (
    set_seed, save_checkpoint, load_checkpoint,
    MetricTracker, rmse, mae, r2_score, spearman,
)
from sklearn.preprocessing import StandardScaler


def _build_ckpt_meta(model_type, fold, epoch, val_rmse, cfg, extra=None):
    """Build the standard metadata dict stored in every checkpoint."""
    meta = {
        "model_type": model_type,
        "fold": fold,
        "epoch": epoch,
        "val_rmse": val_rmse,
        "config": cfg,
    }
    if extra:
        meta.update(extra)
    return meta


# ----------------------------------------------------------------------------
# Model factories
# ----------------------------------------------------------------------------
def build_model(model_type: str, cfg: dict, in_dim: int = None, edge_dim: int = None,
                n_features: int = None):
    """Return (model, is_pytorch_graph) for a given model_type."""
    if model_type == "ridge":
        from models.baselines import get_linear_model
        return get_linear_model("ridge", alpha=cfg.get("alpha", 1.0)), False
    if model_type == "xgb":
        from models.tree_models import get_tree_model
        return get_tree_model("xgb", **cfg), False
    if model_type == "lgb":
        from models.tree_models import get_tree_model
        return get_tree_model("lgb", **cfg), False
    if model_type == "catboost":
        from models.tree_models import get_tree_model
        return get_tree_model("catboost", **cfg), False
    if model_type == "rf":
        from models.tree_models import get_tree_model
        return get_tree_model("rf", **cfg), False
    if model_type == "mlp":
        from models.mlp import FingerprintMLP
        return FingerprintMLP(in_dim=n_features, out_dim=1, dropout=cfg.get("dropout", 0.3)), True
    if model_type in ("gcn", "gat", "mpnn"):
        from models.gnn import get_gnn
        return get_gnn(model_type, in_dim, edge_dim, hidden_dim=cfg.get("hidden_dim", 128),
                       n_layers=cfg.get("n_layers", 3), dropout=cfg.get("dropout", 0.2)), True
    if model_type == "graph_transformer":
        from models.graph_transformer import GraphTransformerRegressor
        return GraphTransformerRegressor(in_dim, edge_dim, hidden_dim=cfg.get("hidden_dim", 128),
                                         n_layers=cfg.get("n_layers", 4)), True
    if model_type == "fusionnet":
        from models.fusionnet import PolymerFusionNet
        return PolymerFusionNet(n_modalities=cfg.get("n_modalities", 5),
                                dim=cfg.get("dim", 256),
                                n_layers=cfg.get("n_layers", 2)), True
    if model_type == "polychain":
        from models.polychain import PolyChain
        return PolyChain(in_atom_dim=in_dim, in_edge_dim=edge_dim,
                         hidden_dim=cfg.get("hidden_dim", 256),
                         n_backbone_layers=cfg.get("n_backbone_layers", 4),
                         n_hamf_layers=cfg.get("n_hamf_layers", 2),
                         dropout=cfg.get("dropout", 0.2)), True
    raise ValueError(f"Unknown model_type: {model_type}")


# ----------------------------------------------------------------------------
# Tabular trainer (baselines, tree models, MLPs on features)
# ----------------------------------------------------------------------------
def train_tabular(model, X_train, y_train, X_val, y_val, cfg, model_type, device):
    """Train sklearn-style model on tabular features."""
    if model_type == "xgb":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif model_type == "lgb":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[])
    elif model_type == "catboost":
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    else:
        model.fit(X_train, y_train)
    pred_val = model.predict(X_val)
    return pred_val


# Models that require feature scaling (sensitive to feature magnitude)
MODELS_NEED_SCALING = {"ridge", "mlp"}


# ----------------------------------------------------------------------------
# Graph trainer (PyTorch models)
# ----------------------------------------------------------------------------
def train_graph(model, train_loader, val_loader, cfg, device, model_type="polychain",
                cst_mean=None, cst_std=None, cst_dim=32,
                ckpt_dir=None, fold=0, person="anon", full_cfg=None,
                model_ckpt_cfg=None, resume=False, auto_save_every=0):
    """Train a PyTorch graph model.

    Saves checkpoints when ckpt_dir is provided:
        - {tag}_best.pt      (best val_rmse, model only)
        - {tag}_final.pt     (last epoch, model only)
        - {tag}_recovery.pt  (every N epochs, includes opt+sched state)

    If resume=True, loads recovery checkpoint (preferred) or final checkpoint
    and restores model + optimizer + scheduler state.
    """
    epochs = cfg.get("epochs", 200)
    lr = cfg.get("lr", 1e-4)
    weight_decay = cfg.get("weight_decay", 1e-5)
    grad_clip = cfg.get("grad_clip", 1.0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    patience = cfg.get("early_stopping", {}).get("patience", 30)
    bad = 0
    start_epoch = 1

    ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_tag = f"{person}_{model_type}_fold{fold}"

    # Auto-resume: prefer recovery over final (recovery has opt+sched)
    if resume and ckpt_dir:
        resume_candidates = [
            ckpt_dir / f"{ckpt_tag}_recovery.pt",
            ckpt_dir / f"{ckpt_tag}_final.pt",
        ]
        resume_path = next((p for p in resume_candidates if p.exists()), None)
        if resume_path is not None:
            print(f"  Resuming from {resume_path}")
            ckpt_data = load_checkpoint(resume_path)
            if "model_state" in ckpt_data:
                model.load_state_dict(ckpt_data["model_state"])
            if "optimizer_state" in ckpt_data:
                opt.load_state_dict(ckpt_data["optimizer_state"])
            if "scheduler_state" in ckpt_data:
                sched.load_state_dict(ckpt_data["scheduler_state"])
            start_epoch = ckpt_data.get("epoch", 0) + 1
            best_val_rmse = ckpt_data.get("val_rmse", float("inf"))
            bad = ckpt_data.get("bad_epochs", 0)
            print(f"  Resumed from epoch {start_epoch - 1}, "
                  f"best_val_rmse={best_val_rmse:.4f}, bad_epochs={bad}")
        else:
            print("  No checkpoint found for resume. Starting from scratch.")

    def _save_full(tag):
        if ckpt_dir is None:
            return
        extra = {"model_state": model.state_dict()}
        if tag == "recovery":
            extra["optimizer_state"] = opt.state_dict()
            extra["scheduler_state"] = sched.state_dict()
            extra["bad_epochs"] = bad
        if cst_mean is not None:
            extra["cst_mean"] = cst_mean.tolist() if hasattr(cst_mean, "tolist") else list(cst_mean)
        if cst_std is not None:
            extra["cst_std"] = cst_std.tolist() if hasattr(cst_std, "tolist") else list(cst_std)
        ckpt_cfg = model_ckpt_cfg if model_ckpt_cfg is not None else (full_cfg or cfg)
        meta = _build_ckpt_meta(model_type, fold, epoch, val_rmse, ckpt_cfg, extra=extra)
        path = ckpt_dir / f"{ckpt_tag}_{tag}.pt"
        save_checkpoint(meta, path)
        print(f"  Checkpoint saved -> {path}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        for batch_dict in train_loader:
            opt.zero_grad()
            if model_type == "polychain":
                batch_dict = move_to_device(batch_dict, device)
                pred = model(batch_dict)
                y = batch_dict["y"]
                loss = criterion(pred, y.view(-1))
            else:
                batch = batch_dict.to(device)
                pred = model(batch)
                y = batch.y.view(-1)
                loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        sched.step()

        # Auto-save recovery checkpoint
        if auto_save_every > 0 and epoch % auto_save_every == 0:
            _save_full("recovery")

        # Validation
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch_dict in val_loader:
                if model_type == "polychain":
                    batch_dict = move_to_device(batch_dict, device)
                    pred = model(batch_dict)
                    y = batch_dict["y"]
                else:
                    batch = batch_dict.to(device)
                    pred = model(batch)
                    y = batch.y
                preds.append(pred.cpu().numpy())
                gts.append(y.view(-1).cpu().numpy())
        preds = np.concatenate(preds)
        gts = np.concatenate(gts)
        val_rmse = rmse(gts, preds)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            _save_full("best")
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                _save_full("final")
                break

    _save_full("final")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_rmse


def move_to_device(batch_dict, device):
    """Move all tensors in a PolyChain batch dict to device."""
    out = {}
    for k, v in batch_dict.items():
        if k == "smiles":
            out[k] = v
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            # PyG Batch objects – use .to
            try:
                out[k] = v.to(device)
            except Exception:
                out[k] = v
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model_config", default=None,
                        help="Per-model YAML config (e.g., training/configs/polychain_finetune.yaml)")
    parser.add_argument("--person", default="anon",
                        help="Team member name (used in output filename).")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to N samples (for smoke testing).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs.")
    parser.add_argument("--auto_save_every", type=int, default=5,
                        help="Save recovery checkpoint every N epochs (0 to disable)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.model_config and os.path.exists(args.model_config):
        with open(args.model_config) as f:
            model_cfg = yaml.safe_load(f)
    else:
        model_cfg = {}

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and
                          cfg.get("device", {}).get("use_cuda", True) else "cpu")
    target = cfg["target"]["column"]
    ckpt_dir = Path(cfg["paths"].get("checkpoints_dir", "outputs/checkpoints/"))

    # Load features and splits
    data_dir = Path(cfg["paths"]["data_dir"])
    train = pd.read_parquet(data_dir / "processed" / "train_features.parquet")
    with open(data_dir / "splits.pkl", "rb") as f:
        splits = pickle.load(f)

    fold = splits[args.fold]
    train_idx, val_idx = fold["train"], fold["val"]
    tr_df = train.iloc[train_idx].reset_index(drop=True)
    va_df = train.iloc[val_idx].reset_index(drop=True)

    # Apply max_samples limit for smoke testing
    if args.max_samples:
        n = min(args.max_samples, len(tr_df))
        tr_df = tr_df.iloc[:n].reset_index(drop=True)
        n_val = min(n // 4, len(va_df))
        va_df = va_df.iloc[:n_val].reset_index(drop=True)
        print(f"Limited to {len(tr_df)} train / {len(va_df)} val samples")

    # Override epochs if specified
    if args.epochs:
        model_cfg["epochs"] = args.epochs

    # Build model
    feature_cols = [c for c in train.columns
                    if c not in ("SMILES", "id", target)]

    if args.model_type in ("ridge", "xgb", "lgb", "catboost", "rf"):
        X_tr = tr_df[feature_cols].values
        y_tr = tr_df[target].values
        X_va = va_df[feature_cols].values
        y_va = va_df[target].values

        # Apply StandardScaler for linear models (Ridge)
        scaler = None
        if args.model_type in MODELS_NEED_SCALING:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)

        model = build_model(args.model_type, model_cfg, n_features=len(feature_cols))[0]
        pred_va = train_tabular(model, X_tr, y_tr, X_va, y_va, model_cfg, args.model_type, device)
        # Save sklearn-style model checkpoint
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_tag = f"{args.person}_{args.model_type}_fold{args.fold}"
        best_path = ckpt_dir / f"{ckpt_tag}_best.pt"
        final_path = ckpt_dir / f"{ckpt_tag}_final.pt"
        ckpt_payload = {
            "model_state": model,
            "model_type": args.model_type,
            "fold": args.fold,
            "epoch": 0,
            "val_rmse": rmse(y_va, pred_va),
            "config": model_cfg,
            "feature_cols": feature_cols,
        }
        if scaler is not None:
            ckpt_payload["scaler"] = scaler
        save_checkpoint(ckpt_payload, best_path)
        save_checkpoint(ckpt_payload, final_path)
        print(f"  Checkpoint saved -> {best_path}")
    elif args.model_type in ("mlp",):
        from models.mlp import FingerprintMLP
        X_tr = tr_df[feature_cols].values.astype(np.float32)
        y_tr = tr_df[target].values.astype(np.float32)
        X_va = va_df[feature_cols].values.astype(np.float32)
        y_va = va_df[target].values.astype(np.float32)

        # Apply StandardScaler for MLP
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_va = scaler.transform(X_va).astype(np.float32)

        model = FingerprintMLP(in_dim=X_tr.shape[1])
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        crit = nn.MSELoss()

        best_val_rmse = float("inf")
        best_state = None
        patience = 20
        bad = 0

        for ep in range(1, 101):
            model.train()
            idx = np.random.permutation(len(X_tr))
            for i in range(0, len(X_tr), 64):
                b = idx[i:i+64]
                xb = torch.from_numpy(X_tr[b]).to(device)
                yb = torch.from_numpy(y_tr[b]).to(device)
                opt.zero_grad()
                pred = model(xb).squeeze(-1)
                loss = crit(pred, yb)
                loss.backward()
                opt.step()
            sched.step()

            # Validation
            model.eval()
            with torch.no_grad():
                pred_va = model(torch.from_numpy(X_va).to(device)).squeeze(-1).cpu().numpy()
            val_rmse = float(np.sqrt(np.mean((y_va - pred_va) ** 2)))
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred_va = model(torch.from_numpy(X_va).to(device)).squeeze(-1).cpu().numpy()

        # Save MLP checkpoint
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_tag = f"{args.person}_{args.model_type}_fold{args.fold}"
        best_path = ckpt_dir / f"{ckpt_tag}_best.pt"
        final_path = ckpt_dir / f"{ckpt_tag}_final.pt"
        ckpt_payload = {
            "model_state": model.state_dict(),
            "model_type": args.model_type,
            "fold": args.fold,
            "epoch": ep,
            "val_rmse": best_val_rmse,
            "config": {"in_dim": X_tr.shape[1], "out_dim": 1, **model_cfg},
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }
        save_checkpoint(ckpt_payload, best_path)
        save_checkpoint(ckpt_payload, final_path)
        print(f"  Checkpoint saved -> {best_path}")
    elif args.model_type == "polychain":
        from features.graph_utils import build_multiscale
        from models.polychain.cst import compute_cst_batch

        # Build multi-scale graphs
        train_samples = [build_multiscale(s, y=y) for s, y in
                         zip(tr_df["SMILES"].tolist(), tr_df[target].tolist())]
        val_samples = [build_multiscale(s, y=y) for s, y in
                       zip(va_df["SMILES"].tolist(), va_df[target].tolist())]
        train_samples = [s for s in train_samples if s is not None]
        val_samples = [s for s in val_samples if s is not None]

        # CST calibration
        cst_train = compute_cst_batch([s.smiles for s in train_samples])
        cst_mean = cst_train.mean(axis=0)
        cst_std = cst_train.std(axis=0) + 1e-6

        def collate(samples):
            from features.graph_utils import collate_multiscale
            batch = collate_multiscale(samples)
            from models.polychain.cst import compute_cst_batch
            batch["cst"] = torch.tensor(compute_cst_batch([s.smiles for s in samples]),
                                        dtype=torch.float)
            return batch

        train_loader = DataLoader(train_samples, batch_size=model_cfg.get("batch_size", 32),
                                  shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_samples, batch_size=model_cfg.get("batch_size", 32),
                                shuffle=False, collate_fn=collate)

        # Inspect first batch for dims
        first = next(iter(train_loader))
        in_dim = first["monomer"].x.size(1)
        edge_dim = first["monomer"].edge_attr.size(1)
        cst_dim = first["cst"].size(1)

        model, _ = build_model(args.model_type, model_cfg, in_dim=in_dim, edge_dim=edge_dim)
        model = model.to(device)
        # Inject calibration stats
        model.cst_norm.mean.data = torch.tensor(cst_mean, dtype=torch.float).to(device)
        model.cst_norm.std.data = torch.tensor(cst_std, dtype=torch.float).to(device)

        # Build model-specific config for checkpoint (predictor.py needs in_atom_dim/in_edge_dim)
        model_ckpt_cfg = {
            "in_atom_dim": in_dim,
            "in_edge_dim": edge_dim,
            "hidden_dim": model_cfg.get("hidden_dim", 256),
            "n_backbone_layers": model_cfg.get("n_backbone_layers", 4),
            "n_hamf_layers": model_cfg.get("n_hamf_layers", 2),
        }

        model, best_val_rmse = train_graph(model, train_loader, val_loader,
                                            model_cfg, device, model_type="polychain",
                                            cst_mean=cst_mean, cst_std=cst_std,
                                            ckpt_dir=ckpt_dir, fold=args.fold,
                                            person=args.person, full_cfg=cfg,
                                            model_ckpt_cfg=model_ckpt_cfg,
                                            resume=args.resume,
                                            auto_save_every=args.auto_save_every)
        model.eval()
        preds = []
        with torch.no_grad():
            for batch_dict in val_loader:
                batch_dict = move_to_device(batch_dict, device)
                pred = model(batch_dict)
                preds.append(pred.cpu().numpy())
        pred_va = np.concatenate(preds)
    else:
        # GNN baselines (GCN/GAT/MPNN/GraphTransformer/FusionNet)
        from features.graphs import smiles_to_graph
        train_graphs = [smiles_to_graph(s, y=y) for s, y in
                        zip(tr_df["SMILES"].tolist(), tr_df[target].tolist())]
        val_graphs = [smiles_to_graph(s, y=y) for s, y in
                      zip(va_df["SMILES"].tolist(), va_df[target].tolist())]
        train_graphs = [g for g in train_graphs if g is not None]
        val_graphs = [g for g in val_graphs if g is not None]

        in_dim = train_graphs[0].x.size(1)
        edge_dim = train_graphs[0].edge_attr.size(1)

        PyGDL = PyGDataLoader if PyGDataLoader is not None else DataLoader
        train_loader = PyGDL(train_graphs, batch_size=64, shuffle=True)
        val_loader = PyGDL(val_graphs, batch_size=64, shuffle=False)

        model, _ = build_model(args.model_type, model_cfg, in_dim=in_dim, edge_dim=edge_dim)
        model = model.to(device)
        # Build model-specific config for checkpoint
        model_ckpt_cfg = {
            "in_atom_dim": in_dim,
            "in_edge_dim": edge_dim,
            "hidden_dim": model_cfg.get("hidden_dim", 128),
            "n_layers": model_cfg.get("n_layers", 3),
        }
        model, best_val_rmse = train_graph(model, train_loader, val_loader,
                                            model_cfg, device, model_type=args.model_type,
                                            ckpt_dir=ckpt_dir, fold=args.fold,
                                            person=args.person, full_cfg=cfg,
                                            model_ckpt_cfg=model_ckpt_cfg,
                                            resume=args.resume,
                                            auto_save_every=args.auto_save_every)
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                preds.append(pred.cpu().numpy())
                gts.append(batch.y.view(-1).cpu().numpy())
        pred_va = np.concatenate(preds)

    # Metrics
    y_va = va_df[target].values
    metrics = {
        "rmse": rmse(y_va, pred_va),
        "mae": mae(y_va, pred_va),
        "r2": r2_score(y_va, pred_va),
        "spearman": spearman(y_va, pred_va),
    }
    print(f"[{args.model_type} | fold {args.fold}] " + json.dumps(metrics))

    # Save OOF predictions
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_file = pred_dir / f"{args.person}_{args.model_type}_fold{args.fold}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump({"val_idx": val_idx, "pred": pred_va, "y": y_va,
                     "metrics": metrics, "model_type": args.model_type,
                     "fold": args.fold, "person": args.person}, f)
    print(f"Saved predictions -> {out_file}")


if __name__ == "__main__":
    main()
