"""
training/train_gin_conditioned.py
Conditioned GIN training with XGB predictions as auxiliary input.
Per-fold XGB training to avoid data leakage.

Variants:
  B: GIN embedding + XGB prediction → concat → head
  C: FiLM modulation: γ(XGB) * h + β(XGB) → head
  D: GIN + XGB prediction + uncertainty → concat → head

Usage:
    python -m training.train_gin_conditioned --target tg --variant C --fold 0
    python -m training.train_gin_conditioned --target tg --variant C --all_folds
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import yaml
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch as PyGBatch

from features.graphs import smiles_to_graph
from models.gnn import (
    GINRegressor,
    ConditionedGINRegressor,
    FiLMGINRegressor,
)
from training.train import set_seed


BATCH_SIZE = 64
XGB_PARAMS_BASE = {
    "n_estimators": 1500, "max_depth": 7, "learning_rate": 0.03,
    "subsample": 0.85, "colsample_bytree": 0.5, "min_child_weight": 3,
    "gamma": 0.1, "reg_alpha": 1.0, "reg_lambda": 2.0,
    "n_jobs": -1, "verbosity": 0,
}


class GraphAuxDS(Dataset):
    def __init__(self, graphs, aux, y=None):
        self.graphs = graphs
        self.aux = torch.from_numpy(aux).float()
        self.y = y
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        if self.y is not None:
            return self.graphs[idx], self.aux[idx], self.y[idx]
        return self.graphs[idx], self.aux[idx]


def collate_aux(batch):
    graphs = [b[0] for b in batch]
    aux = torch.stack([b[1] for b in batch])
    if len(batch[0]) == 3:
        ys = torch.tensor([b[2] for b in batch], dtype=torch.float)
        return PyGBatch.from_data_list(graphs), aux, ys
    return PyGBatch.from_data_list(graphs), aux


def build_graphs(smiles_list):
    graphs = []
    for s in smiles_list:
        g = smiles_to_graph(s)
        if g is not None:
            graphs.append(g)
    return graphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="tg")
    parser.add_argument("--variant", default="B", choices=["B", "C", "D"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all_folds", action="store_true")
    parser.add_argument("--residual", action="store_true",
                       help="Train GIN on residual of XGB (y - xgb_pred)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="outputs/gin_conditioned")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    use_cuda = cfg.get("device", {}).get("use_cuda", True) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}, Variant: {args.variant}, Target: {args.target}")
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir) / f"variant_{args.variant}" / args.target
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    tr = pd.read_csv(data_dir / "train.csv")
    mask = tr["target_type"].values == args.target
    y = tr["target"].values[mask].astype(np.float32)
    smiles = tr["smiles"].values[mask]

    te = pd.read_csv(data_dir / "test.csv")
    tmask = te["target_type"].values == args.target
    te_ids = te["id"].values[tmask]
    te_smiles = te["smiles"].values[tmask]

    # Load tabular features
    X_tr = pd.read_parquet(data_dir / "processed" / "features_train.parquet")
    X_te = pd.read_parquet(data_dir / "processed" / "features_test.parquet")
    exclude = {"id", "canon_smiles", "SMILES"}
    feat_cols = [c for c in X_tr.columns if c not in exclude]

    X_arr = X_tr[feat_cols].values.astype(np.float32)[mask]
    X_te_arr = X_te[feat_cols].values.astype(np.float32)[tmask]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_arr)
    X_te_s = scaler.transform(X_te_arr)

    with open(data_dir / f"splits_{args.target}.pkl", "rb") as f:
        splits = pickle.load(f)

    n = len(y)

    # Build graphs
    tr_graphs = build_graphs(smiles)
    te_graphs = build_graphs(te_smiles)
    in_dim = tr_graphs[0].x.size(1)
    edge_dim = tr_graphs[0].edge_attr.size(1)
    print(f"  Graphs: {len(tr_graphs)} train, {len(te_graphs)} test")

    # Determine model class
    if args.variant in ("B", "D"):
        model_cls = ConditionedGINRegressor
    elif args.variant == "C":
        model_cls = FiLMGINRegressor

    folds_to_run = range(5) if args.all_folds else [args.fold]
    all_oof = np.zeros(n, dtype=np.float32)
    all_oof_count = np.zeros(n, dtype=np.int32)
    test_preds_list = []

    for fold in folds_to_run:
        print(f"\n--- Fold {fold} ---")
        tr_idx = [i for i in splits[fold]["train"] if i < n]
        va_idx = [i for i in splits[fold]["val"] if i < n]

        # Per-fold XGB: train only on GIN training folds, predict all
        xgbp = {**XGB_PARAMS_BASE, "random_state": 42}
        xgb_fold = xgb.XGBRegressor(**xgbp, early_stopping_rounds=50)
        xgb_fold.fit(Xs[tr_idx], y[tr_idx],
                     eval_set=[(Xs[va_idx], y[va_idx])], verbose=False)

        xgb_tr_pred = xgb_fold.predict(Xs[tr_idx])
        xgb_va_pred = xgb_fold.predict(Xs[va_idx])
        xgb_te_pred = xgb_fold.predict(X_te_s)

        tr_va_r2 = r2_score(y[va_idx], xgb_va_pred)
        print(f"  XGB val R2={tr_va_r2:.4f}")

        # Per-fold uncertainty: train 4 additional seeds on GIN training folds
        if args.variant == "D":
            seed_preds_tr = np.zeros((5, len(tr_idx)), dtype=np.float32)
            seed_preds_va = np.zeros((5, len(va_idx)), dtype=np.float32)
            seed_preds_te = np.zeros((5, len(te_graphs)), dtype=np.float32)
            seed_preds_tr[0] = xgb_tr_pred
            seed_preds_va[0] = xgb_va_pred
            seed_preds_te[0] = xgb_te_pred
            for si, seed in enumerate((43, 44, 45, 46), start=1):
                xgbp_s = {**xgbp, "random_state": seed}
                m = xgb.XGBRegressor(**xgbp_s, early_stopping_rounds=50)
                m.fit(Xs[tr_idx], y[tr_idx],
                      eval_set=[(Xs[va_idx], y[va_idx])], verbose=False)
                seed_preds_tr[si] = m.predict(Xs[tr_idx])
                seed_preds_va[si] = m.predict(Xs[va_idx])
                seed_preds_te[si] = m.predict(X_te_s)

        # Build auxiliary features
        if args.variant == "B":
            aux_dim = 1
            tr_aux = xgb_tr_pred.reshape(-1, 1).astype(np.float32)
            va_aux = xgb_va_pred.reshape(-1, 1).astype(np.float32)
            te_aux = xgb_te_pred.reshape(-1, 1).astype(np.float32)
        elif args.variant == "C":
            aux_dim = 1
            tr_aux = xgb_tr_pred.reshape(-1, 1).astype(np.float32)
            va_aux = xgb_va_pred.reshape(-1, 1).astype(np.float32)
            te_aux = xgb_te_pred.reshape(-1, 1).astype(np.float32)
        elif args.variant == "D":
            aux_dim = 2
            tr_uncert = seed_preds_tr.std(axis=0, keepdims=True).T
            va_uncert = seed_preds_va.std(axis=0, keepdims=True).T
            te_uncert = seed_preds_te.std(axis=0, keepdims=True).T
            tr_aux = np.column_stack([xgb_tr_pred, tr_uncert]).astype(np.float32)
            va_aux = np.column_stack([xgb_va_pred, va_uncert]).astype(np.float32)
            te_aux = np.column_stack([xgb_te_pred, te_uncert]).astype(np.float32)

        if args.residual:
            train_target = y[tr_idx] - xgb_tr_pred
            val_target = y[va_idx] - xgb_va_pred
        else:
            train_target = y[tr_idx]
            val_target = y[va_idx]

        train_ds = GraphAuxDS(
            [tr_graphs[i] for i in tr_idx], tr_aux, train_target,
        )
        val_ds = GraphAuxDS(
            [tr_graphs[i] for i in va_idx], va_aux, val_target,
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=0, collate_fn=collate_aux)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0, collate_fn=collate_aux)

        # Build model
        model = model_cls(
            in_dim=in_dim, edge_dim=edge_dim,
            aux_dim=aux_dim,
            hidden_dim=args.hidden_dim, embed_dim=args.embed_dim,
            n_layers=3, dropout=0.2,
        ).to(device)

        # Stage 1: freeze GIN encoder
        for p in model.encoder.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        patience = 20
        bad = 0
        stage1_epochs = min(30, args.epochs // 3)

        for epoch in range(1, args.epochs + 1):
            if epoch == stage1_epochs + 1:
                for p in model.encoder.parameters():
                    p.requires_grad = True
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=args.lr / 10, weight_decay=1e-5,
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - stage1_epochs,
                )

            model.train()
            epoch_loss = 0.0
            for graph_batch, aux_batch, y_batch in train_loader:
                graph_batch = graph_batch.to(device)
                aux_batch = aux_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                pred = model(graph_batch, aux_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for graph_batch, aux_batch, y_batch in val_loader:
                    graph_batch = graph_batch.to(device)
                    aux_batch = aux_batch.to(device)
                    y_batch = y_batch.to(device)
                    pred = model(graph_batch, aux_batch)
                    val_loss += criterion(pred, y_batch).item()
            val_loss /= max(len(val_loader), 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"  Early stop at epoch {epoch}")
                    break

            if epoch % 20 == 0 or epoch == 1:
                vp = []
                vy = []
                with torch.no_grad():
                    for gb, ab, yb in val_loader:
                        vp.append(model(gb.to(device), ab.to(device)).cpu().numpy())
                        vy.append(yb.numpy())
                vr2 = r2_score(np.concatenate(vy), np.concatenate(vp))
                print(f"  Epoch {epoch:3d}: train_loss={epoch_loss/len(train_loader):.3f}, val_R2={vr2:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        # OOF
        vp = []
        vy = []
        with torch.no_grad():
            for gb, ab, yb in val_loader:
                vp.append(model(gb.to(device), ab.to(device)).cpu().numpy())
                vy.append(yb.numpy())
        va_residuals = np.concatenate(vp)
        va_preds = xgb_va_pred + va_residuals if args.residual else va_residuals
        va_r2 = r2_score(y[va_idx], va_preds)
        print(f"  Fold {fold} Val R2={va_r2:.4f}")

        all_oof[va_idx] = va_preds
        all_oof_count[va_idx] = 1

        # Test
        te_ds = GraphAuxDS(te_graphs, te_aux)
        te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, collate_fn=collate_aux)
        te_residuals = []
        with torch.no_grad():
            for gb, ab in te_loader:
                te_residuals.append(model(gb.to(device), ab.to(device)).cpu().numpy())
        te_preds = xgb_te_pred + np.concatenate(te_residuals) if args.residual else np.concatenate(te_residuals)
        test_preds_list.append(te_preds)

        # Save per-fold model
        torch.save(best_state, out_dir / f"checkpoint_fold{fold}.pt")

    mask = all_oof_count > 0
    oof_r2 = r2_score(y[mask], all_oof[mask])
    print(f"\nOverall OOF R2: {oof_r2:.4f}")

    oof_data = {
        "val_idx": np.where(mask)[0], "pred": all_oof[mask],
        "y": y[mask], "model_type": f"gin_conditioned_{args.variant}", "target": args.target,
    }
    with open(out_dir / f"oof_{args.target}.pkl", "wb") as f:
        pickle.dump(oof_data, f)

    test_preds_avg = np.mean(test_preds_list, axis=0)
    te_sub = pd.DataFrame({"id": te_ids, "target": test_preds_avg})
    te_sub.to_csv(out_dir / f"submission_{args.target}.csv", index=False)

    print(f"Saved: {out_dir / f'oof_{args.target}.pkl'}")
    print(f"Saved: {out_dir / f'submission_{args.target}.csv'}")


if __name__ == "__main__":
    main()
