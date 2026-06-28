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
import subprocess
import threading
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


# ----------------------------------------------------------------------------
# Optuna hyperparameter tuning (used by run_tree_models.py)
# ----------------------------------------------------------------------------
def tune_model_optuna(model_type: str, X: np.ndarray, y: np.ndarray,
                      n_trials: int = 30, seed: int = 42) -> dict:
    """Find best hyperparameters via Optuna TPE sampler."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        if model_type == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": seed,
            }
            from xgboost import XGBRegressor
            model = XGBRegressor(**params)
        elif model_type == "lgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": seed,
                "verbose": -1,
            }
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(**params)
        elif model_type == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 200, 1000),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                "random_seed": seed,
                "verbose": 0,
            }
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(**params)
        else:
            raise ValueError(f"Unsupported model_type for tuning: {model_type}")

        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _collate_polychain(samples):
    """Module-level collate for polychain (needed for Windows multiprocessing)."""
    from features.graph_utils import collate_multiscale
    from models.polychain.cst import compute_cst_batch
    batch = collate_multiscale(samples)
    batch["cst"] = torch.tensor(compute_cst_batch([s.smiles for s in samples]),
                                dtype=torch.float)
    return batch


def _gpu_monitor(interval: float = 30.0, log_path: str = "outputs/logs/gpu_util.csv"):
    import csv
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "util_pct", "memory_mb"])
    while getattr(_gpu_monitor, "_running", True):
        try:
            result = subprocess.check_output(
                "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
                shell=True
            ).decode().strip()
            parts = result.replace("%", "").replace(" MiB", "").split(", ")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([time.time(), parts[0], parts[1]])
        except Exception:
            pass
        time.sleep(interval)


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
    if model_type == "polychain" or model_type.startswith("polychain_"):
        from models.polychain import PolyChain
        return PolyChain(in_atom_dim=in_dim, in_edge_dim=edge_dim,
                         hidden_dim=cfg.get("hidden_dim", 256),
                         n_backbone_layers=cfg.get("n_backbone_layers", 4),
                         n_hamf_layers=cfg.get("n_hamf_layers", 2),
                         cst_dim=cfg.get("cst_dim", 32),
                         dropout=cfg.get("dropout", 0.2),
                         grad_checkpoint=cfg.get("gradient_checkpointing", True)), True
    raise ValueError(f"Unknown model_type: {model_type}")


# ----------------------------------------------------------------------------
# Multi-task training
# ----------------------------------------------------------------------------
def train_multitask_per_fold(cfg: dict, fold: int, exp_ver: str,
                              ckpt_dir: Path, pred_dir: Path,
                              data_dir: Path, seed: int,
                              device: torch.device):
    """Run one fold of multi-task training and write OOF + test pickles.

    The fold structure comes from ``splits_tg.pkl`` (the larger dataset). Tg
    rows in the val split contribute to the Tg OOF pickle; Egc rows in the
    Egc val split contribute to the Egc OOF pickle. Train rows from both
    targets are concatenated and trained jointly with masks.

    Pickle filenames match the ``weight_optimizer`` glob pattern:

        {exp_ver}_{target}_multitask_fold{k}.pkl
        {exp_ver}_{target}_multitask_fold{k}_test.pkl
    """
    from models.multitask import MultiTaskModel

    mt_cfg = cfg.get("multitask", {})
    epochs = mt_cfg.get("epochs", 100)
    log = {"fold": fold, "target": "joint"}

    splits_tg_path = data_dir / "splits_tg.pkl"
    splits_egc_path = data_dir / "splits_egc.pkl"
    with open(splits_tg_path, "rb") as f:
        splits_tg = pickle.load(f)
    with open(splits_egc_path, "rb") as f:
        splits_egc = pickle.load(f)

    if fold >= len(splits_tg) or fold >= len(splits_egc):
        raise ValueError(f"fold {fold} not present in tg/egc splits")

    target_col = cfg["data"]["target_col"]
    # Re-load with target_type annotation so we can split by target cleanly.
    # Matches the join logic in main() when --target is passed.
    train_feat = pd.read_parquet(data_dir / "processed" / "features_train.parquet")
    full_train = pd.read_csv(data_dir / "train.csv")
    if len(train_feat) == len(full_train):
        train_feat[target_col] = full_train[target_col].values
        train_feat["target_type"] = full_train["target_type"].values
    else:
        from rdkit import Chem
        def _canon(s):
            mol = Chem.MolFromSmiles(s)
            return Chem.MolToSmiles(mol, canonical=True) if mol else None
        canon_map = {s: _canon(s) for s in full_train["smiles"].unique()}
        full_train["canon_smiles"] = full_train["smiles"].map(canon_map)
        full_train = full_train.dropna(subset=["canon_smiles"])
        train_feat = full_train[["canon_smiles", target_col, "target_type"]].merge(
            train_feat.drop(columns=["canon_smiles"], errors="ignore"),
            left_on="canon_smiles", right_on="SMILES", how="left",
        )
        train_feat = train_feat.dropna(subset=[target_col]).reset_index(drop=True)

    # Tg fold: train on Tg-only rows; predict on the *full* val set so the
    # OOF pickle lines up with the other v27/v28 model OOFs (same val_idx).
    tg_train_idx, tg_val_idx = splits_tg[fold]["train"], splits_tg[fold]["val"]
    tg_train_df = train_feat.iloc[tg_train_idx].reset_index(drop=True)
    tg_val_df = train_feat.iloc[tg_val_idx].reset_index(drop=True)
    tg_train_df = tg_train_df[tg_train_df["target_type"] == "tg"].reset_index(drop=True)

    # Egc fold: same pattern.
    egc_train_idx, egc_val_idx = splits_egc[fold]["train"], splits_egc[fold]["val"]
    egc_train_df = train_feat.iloc[egc_train_idx].reset_index(drop=True)
    egc_val_df = train_feat.iloc[egc_val_idx].reset_index(drop=True)
    egc_train_df = egc_train_df[egc_train_df["target_type"] == "egc"].reset_index(drop=True)

    feature_cols = [c for c in train_feat.columns
                    if c not in ("SMILES", "id", "canon_smiles", "target_type", target_col)]

    X_tg_tr = tg_train_df[feature_cols]
    y_tg_tr = tg_train_df[target_col].values.astype(np.float32)
    X_egc_tr = egc_train_df[feature_cols]
    y_egc_tr = egc_train_df[target_col].values.astype(np.float32)

    model, common_features, oof_train = train_multitask(
        X_tg_tr, y_tg_tr, X_egc_tr, y_egc_tr, cfg, device=device, return_oof=True,
    )

    # Inference on the FULL tg val set (matches other v27/v28 model OOFs).
    model.eval()
    X_val_tg = tg_val_df[common_features].values.astype(np.float32)
    if X_val_tg.size > 0:
        X_val_tg_scaled = model._scaler_X.transform(X_val_tg).astype(np.float32)
        with torch.no_grad():
            tg_pred_n, _ = model(torch.from_numpy(X_val_tg_scaled).to(device))
        tg_val_pred = model._y_scaler_tg.inverse_transform(
            tg_pred_n.cpu().numpy().reshape(-1, 1)).flatten()
        y_val_tg = tg_val_df[target_col].values.astype(np.float32)
    else:
        tg_val_pred = np.zeros(0)
        y_val_tg = np.zeros(0)

    # Inference on the FULL egc val set.
    X_val_egc = egc_val_df[common_features].values.astype(np.float32)
    if X_val_egc.size > 0:
        X_val_egc_scaled = model._scaler_X.transform(X_val_egc).astype(np.float32)
        with torch.no_grad():
            _, egc_pred_n = model(torch.from_numpy(X_val_egc_scaled).to(device))
        egc_val_pred = model._y_scaler_egc.inverse_transform(
            egc_pred_n.cpu().numpy().reshape(-1, 1)).flatten()
        y_val_egc = egc_val_df[target_col].values.astype(np.float32)
    else:
        egc_val_pred = np.zeros(0)
        y_val_egc = np.zeros(0)

    pred_dir.mkdir(parents=True, exist_ok=True)
    if len(tg_val_pred) > 0:
        out_tg = pred_dir / f"{exp_ver}_tg_multitask_fold{fold}.pkl"
        with open(out_tg, "wb") as f:
            pickle.dump({"val_idx": list(tg_val_idx[:len(tg_val_pred)]),
                         "pred": tg_val_pred, "y": y_val_tg,
                         "model_type": "multitask", "fold": fold, "target": "tg"}, f)
        print(f"Saved tg multitask OOF -> {out_tg}")
    if len(egc_val_pred) > 0:
        out_egc = pred_dir / f"{exp_ver}_egc_multitask_fold{fold}.pkl"
        with open(out_egc, "wb") as f:
            pickle.dump({"val_idx": list(egc_val_idx[:len(egc_val_pred)]),
                         "pred": egc_val_pred, "y": y_val_egc,
                         "model_type": "multitask", "fold": fold, "target": "egc"}, f)
        print(f"Saved egc multitask OOF -> {out_egc}")

    # Test inference — both targets (multitask predicts both heads)
    test_feat_full = pd.read_parquet(data_dir / "processed" / "features_test.parquet")
    # Filter to common feature columns (same as training)
    test_feat_cols = [c for c in test_feat_full.columns if c in common_features]
    X_test = test_feat_full[test_feat_cols].values.astype(np.float32)
    # Apply the same X-scaler used during training
    if model._scaler_X is not None:
        X_test = model._scaler_X.transform(X_test)
    model.eval()
    with torch.no_grad():
        tg_test_pred_t, egc_test_pred_t = model(
            torch.from_numpy(X_test).to(device)
        )
        tg_test_pred = tg_test_pred_t.cpu().numpy()
        egc_test_pred = egc_test_pred_t.cpu().numpy()

    # Inverse-transform each target's predictions
    if model._y_scaler_tg is not None:
        tg_test_pred = model._y_scaler_tg.inverse_transform(
            tg_test_pred.reshape(-1, 1)
        ).ravel()
    if model._y_scaler_egc is not None:
        egc_test_pred = model._y_scaler_egc.inverse_transform(
            egc_test_pred.reshape(-1, 1)
        ).ravel()

    # Per-target test ids (tg test set is smaller than the combined test set;
    # egc test set is also a disjoint subset).
    all_test_ids = test_feat_full["id"].values
    pred_by_target = {"tg": tg_test_pred, "egc": egc_test_pred}
    for target, all_preds in pred_by_target.items():
        target_test_csv = data_dir / target / "test.csv"
        target_ids = pd.read_csv(target_test_csv)["id"].values
        # Reindex predictions to target-specific test ids
        id_to_pred = dict(zip(all_test_ids.tolist(), all_preds.tolist()))
        target_preds = [id_to_pred[i] for i in target_ids.tolist()]
        test_out = pred_dir / f"{exp_ver}_{target}_multitask_fold{fold}_test.pkl"
        with open(test_out, "wb") as f:
            pickle.dump({
                "id": target_ids.tolist(),
                "pred": target_preds,
                "model_type": "multitask",
                "fold": fold,
                "target": target,
            }, f)
        print(f"Saved {target} multitask test preds -> {test_out}")

    # Persist checkpoint with learned uncertainty parameters
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_tag = f"{exp_ver}_multitask_fold{fold}"
    payload = {
        "model_state": model.state_dict(),
        "model_type": "multitask",
        "fold": fold,
        "epoch": epochs,
        "val_rmse": float("nan"),
        "config": cfg.get("multitask", {}),
        "common_features": common_features,
        "log_var_tg": float(model.log_var_tg.detach().cpu().item()),
        "log_var_egc": float(model.log_var_egc.detach().cpu().item()),
        "scaler_X": model._scaler_X,
        "y_scaler_tg": model._y_scaler_tg,
        "y_scaler_egc": model._y_scaler_egc,
    }
    save_checkpoint(payload, ckpt_dir / f"{ckpt_tag}_best.pt")
    save_checkpoint(payload, ckpt_dir / f"{ckpt_tag}_final.pt")

    return {
        "log_var_tg": payload["log_var_tg"],
        "log_var_egc": payload["log_var_egc"],
        "n_tg_train": len(y_tg_tr),
        "n_egc_train": len(y_egc_tr),
        "n_tg_val_pred": len(tg_val_pred),
        "n_egc_val_pred": len(egc_val_pred),
    }


def train_multitask(X_tg, y_tg, X_egc, y_egc, config,
                    device: torch.device = None,
                    return_oof: bool = False):
    """Train the multi-task MLP on (Tg ∪ Egc) samples with uncertainty weighting.

    The Tg dataset (~4143 samples) and the Egc dataset (~2028 samples) are
    concatenated and the smaller subset is NOT zero-padded: each row carries a
    boolean mask saying which targets it contributes to. The Kendall-style
    multi-task loss in :class:`MultiTaskModel` consumes those masks and only
    back-propagates the per-head loss on present samples.

    Args:
        X_tg: Tg feature matrix (DataFrame).
        y_tg: Tg targets (1-D array).
        X_egc: Egc feature matrix (DataFrame).
        y_egc: Egc targets (1-D array).
        config: Training config dict. Recognized keys:
            - ``multitask.hidden_dims`` (default ``[256, 128, 64]``)
            - ``multitask.dropout`` (default ``0.2``)
            - ``multitask.epochs`` (default ``100``)
            - ``multitask.batch_size`` (default ``64``)
            - ``multitask.lr`` (default ``1e-3``)
            - ``multitask.weight_decay`` (default ``1e-5``)
            - ``multitask.patience`` (default ``20``)
            - ``multitask.seed`` (default ``42``)
        device: Optional torch device. Defaults to CUDA if available else CPU.
        return_oof: If True, also return per-target predictions for the
            provided samples (in original order).

    Returns:
        Tuple ``(model, common_features)`` by default. When ``return_oof=True``,
        additionally returns ``{"tg_pred": ..., "egc_pred": ..., "y_tg": y_tg,
        "y_egc": y_egc}`` as a fourth element.
    """
    from torch.utils.data import DataLoader, TensorDataset
    from models.multitask import MultiTaskModel
    from sklearn.preprocessing import StandardScaler

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mt_cfg = config.get("multitask", {}) if isinstance(config, dict) else {}
    hidden_dims = mt_cfg.get("hidden_dims", [256, 128, 64])
    dropout = mt_cfg.get("dropout", 0.2)
    epochs = mt_cfg.get("epochs", 100)
    batch_size = mt_cfg.get("batch_size", 64)
    lr = mt_cfg.get("lr", 1e-3)
    weight_decay = mt_cfg.get("weight_decay", 1e-5)
    patience = mt_cfg.get("patience", 20)
    seed = mt_cfg.get("seed", 42)

    common_features = sorted(set(X_tg.columns) & set(X_egc.columns))
    if not common_features:
        raise ValueError("No common features between Tg and Egc feature sets.")

    X_tg_arr = X_tg[common_features].values.astype(np.float32)
    X_egc_arr = X_egc[common_features].values.astype(np.float32)

    # Standardise features jointly so both targets live in the same space.
    scaler = StandardScaler()
    X_all = np.vstack([X_tg_arr, X_egc_arr])
    scaler.fit(X_all)
    X_tg_scaled = scaler.transform(X_tg_arr).astype(np.float32)
    X_egc_scaled = scaler.transform(X_egc_arr).astype(np.float32)

    # Build a single concatenated tensor of features with two boolean masks.
    # Tg samples: tg_mask=True,  egc_mask=False
    # Egc samples: tg_mask=False, egc_mask=True
    n_tg, n_egc = len(X_tg_scaled), len(X_egc_scaled)
    X_concat = np.vstack([X_tg_scaled, X_egc_scaled]).astype(np.float32)
    tg_mask = np.zeros(n_tg + n_egc, dtype=bool)
    egc_mask = np.zeros(n_tg + n_egc, dtype=bool)
    tg_mask[:n_tg] = True
    egc_mask[n_tg:] = True

    # Standardise targets per-head so the shared encoder and the
    # uncertainty-weighted loss live on comparable scales.
    y_scaler_tg = StandardScaler()
    y_scaler_egc = StandardScaler()
    y_tg_norm = y_scaler_tg.fit_transform(y_tg.reshape(-1, 1)).flatten().astype(np.float32)
    y_egc_norm = y_scaler_egc.fit_transform(y_egc.reshape(-1, 1)).flatten().astype(np.float32)
    y_norm = np.zeros(n_tg + n_egc, dtype=np.float32)
    y_norm[:n_tg] = y_tg_norm
    y_norm[n_tg:] = y_egc_norm

    X_tensor = torch.from_numpy(X_concat)
    y_tensor = torch.from_numpy(y_norm)
    tg_mask_tensor = torch.from_numpy(tg_mask)
    egc_mask_tensor = torch.from_numpy(egc_mask)

    dataset = TensorDataset(X_tensor, y_tensor, tg_mask_tensor, egc_mask_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=False)

    model = MultiTaskModel(n_features=X_concat.shape[1],
                           hidden_dims=hidden_dims,
                           dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb, tgb, egb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            tgb = tgb.to(device)
            egb = egb.to(device)
            optimizer.zero_grad()
            tg_pred, egc_pred = model(xb)
            loss, _logs = model.loss(tg_pred, egc_pred, yb, yb, tgb, egb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Attach the scalers + mask bookkeeping to the model so callers can run
    # inference on new data without re-deriving the schema.
    model._scaler_X = scaler
    model._y_scaler_tg = y_scaler_tg
    model._y_scaler_egc = y_scaler_egc
    model._common_features = common_features

    if return_oof:
        with torch.no_grad():
            tg_pred_n, egc_pred_n = model(X_tensor.to(device))
        tg_pred_n = tg_pred_n.cpu().numpy()
        egc_pred_n = egc_pred_n.cpu().numpy()
        tg_pred = y_scaler_tg.inverse_transform(tg_pred_n[:n_tg].reshape(-1, 1)).flatten()
        egc_pred = y_scaler_egc.inverse_transform(egc_pred_n[n_tg:].reshape(-1, 1)).flatten()
        oof = {
            "tg_pred": tg_pred,
            "egc_pred": egc_pred,
            "y_tg": np.asarray(y_tg),
            "y_egc": np.asarray(y_egc),
        }
        return model, common_features, oof
    return model, common_features


def apply_target_transform(y, config):
    """
    Apply target transformation if configured.

    Args:
        y: Target values
        config: Configuration dictionary

    Returns:
        Tuple of (transformed_y, inverse_func_or_None, transform_name_or_None)
    """
    enabled = (
        config.get('use_target_transform', False) or
        config.get('training', {}).get('target_transforms', {}).get('enabled', False)
    )
    if not enabled:
        return y, None, None

    from features.target_transforms import select_best_transform

    y_transformed, inv_func, transform_name = select_best_transform(y)
    print(f"Applied {transform_name} transformation to targets")
    return y_transformed, inv_func, transform_name


def train_model(model_type, config, fold=0, target="tg", **kwargs):
    """
    Convenience wrapper to train a model by type.

    This function can be imported and called directly (no argparse needed).

    Args:
        model_type: Model type string (e.g. 'xgb', 'polychain', 'multitask')
        config: Path to config YAML or dict
        fold: Fold index (default 0)
        target: Target name (default 'tg')
        **kwargs: Additional arguments passed to main training logic

    Returns:
        dict with metrics and trained model
    """
    import sys

    if isinstance(config, (str, Path)):
        with open(config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = config

    seed = cfg.get("seed", {}).get("global", 42)
    set_seed(seed)

    use_cuda = torch.cuda.is_available() and cfg.get("device", {}).get("use_cuda", True)
    if use_cuda:
        try:
            torch.tensor([1.0], device="cuda").sum()
        except Exception:
            use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    data_dir = Path(cfg["paths"]["data_dir"])
    target_col = cfg["data"]["target_col"]
    exp_ver = cfg.get("experiment", {}).get("version", "v1")
    ckpt_dir = Path(cfg["paths"].get("checkpoints_dir", "outputs/checkpoints/"))

    train_df = pd.read_parquet(data_dir / "processed" / "features_train.parquet")
    splits_path = data_dir / f"splits_{target}.pkl"

    try:
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
    except FileNotFoundError:
        from data.generate_splits import generate_splits as _make_splits
        splits = _make_splits(str(splits_path), str(splits_path),
                              n_folds=cfg["cv"]["n_folds"], seed=seed,
                              target_col=target_col, smiles_col="SMILES")

    fold_data = splits[fold]
    train_idx, val_idx = fold_data["train"], fold_data["val"]
    tr_df = train_df.iloc[train_idx].reset_index(drop=True)
    va_df = train_df.iloc[val_idx].reset_index(drop=True)

    feature_cols = [c for c in train_df.columns
                    if c not in ("SMILES", "id", "canon_smiles", "target_type", target_col)]

    model_cfg = cfg.get("model_types", {}).get(model_type, {})
    if isinstance(model_cfg, dict):
        model_cfg.update(cfg.get("training", {}))

    if model_type in ("ridge", "xgb", "lgb", "catboost", "rf"):
        X_tr = tr_df[feature_cols].values
        y_tr = tr_df[target_col].values
        X_va = va_df[feature_cols].values
        y_va = va_df[target_col].values

        scaler = None
        if model_type in MODELS_NEED_SCALING:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)

        model = build_model(model_type, model_cfg, n_features=len(feature_cols))[0]
        pred_va = train_tabular(model, X_tr, y_tr, X_va, y_va, model_cfg, model_type, device)

        metrics = {
            "rmse": rmse(y_va, pred_va),
            "mae": mae(y_va, pred_va),
            "r2": r2_score(y_va, pred_va),
            "spearman": spearman(y_va, pred_va),
        }
        print(f"[{model_type} | fold {fold}] {metrics}")
        return {"metrics": metrics, "model": model, "scaler": scaler}

    elif model_type in ("mlp",):
        from models.mlp import FingerprintMLP
        X_tr = tr_df[feature_cols].values.astype(np.float32)
        y_tr = tr_df[target_col].values.astype(np.float32)
        X_va = va_df[feature_cols].values.astype(np.float32)
        y_va = va_df[target_col].values.astype(np.float32)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_va = scaler.transform(X_va).astype(np.float32)

        model = FingerprintMLP(in_dim=X_tr.shape[1]).to(device)
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
                loss = crit(model(xb).squeeze(-1), yb)
                loss.backward()
                opt.step()
            sched.step()

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

        metrics = {
            "rmse": rmse(y_va, pred_va),
            "mae": mae(y_va, pred_va),
            "r2": r2_score(y_va, pred_va),
            "spearman": spearman(y_va, pred_va),
        }
        print(f"[{model_type} | fold {fold}] {metrics}")
        return {"metrics": metrics, "model": model, "scaler": scaler}

    else:
        raise ValueError(f"train_model not supported for model_type={model_type}. "
                         f"Use CLI entry point (python -m training.train) for graph models.")


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

    use_amp = device.type == "cuda" and cfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

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
            if use_amp:
                with torch.amp.autocast("cuda"):
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
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
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
            except Exception as e:
                import logging
                log = logging.getLogger(__name__)
                log.warning("Failed to move '%s' to device: %s", k, e)
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
    parser.add_argument("--target", default=None,
                        help="Target name (tg/egc). Loads target-specific features and splits.")
    parser.add_argument("--multitask", action="store_true",
                        help="Train the multi-task MLP jointly on Tg+Egc with "
                             "uncertainty weighting. Writes OOF for both targets.")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip test inference after training.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.model_config and os.path.exists(args.model_config):
        with open(args.model_config) as f:
            model_cfg = yaml.safe_load(f)
    else:
        # Fallback: read model config from config.yaml's model_types section
        model_types = cfg.get("model_types", {})
        mt_entry = model_types.get(args.model_type, {})
        if isinstance(mt_entry, dict) and "config" in mt_entry:
            config_path = mt_entry["config"]
            if os.path.exists(config_path):
                with open(config_path) as f:
                    raw_cfg = yaml.safe_load(f)
                # Flatten structured config sections (model, optimizer, scheduler, regularization)
                # to top-level flat dict expected by training code
                model_cfg = {}
                for section in ("model", "optimizer", "scheduler", "regularization"):
                    section_cfg = raw_cfg.get(section, {})
                    if isinstance(section_cfg, dict):
                        model_cfg.update(section_cfg)
            else:
                print(f"WARNING: model config path '{config_path}' not found. Using defaults.")
                model_cfg = {}
        elif isinstance(mt_entry, dict) and mt_entry.get("extends"):
            base_path = mt_entry["extends"]
            if os.path.exists(base_path):
                with open(base_path) as f:
                    raw_cfg = yaml.safe_load(f)
                model_cfg = {}
                for section in ("model", "optimizer", "scheduler", "regularization"):
                    section_cfg = raw_cfg.get(section, {})
                    if isinstance(section_cfg, dict):
                        model_cfg.update(section_cfg)
                overrides = mt_entry.get("overrides", {})
                model_cfg.update(overrides)
            else:
                print(f"WARNING: base config path '{base_path}' not found for variant. Using defaults.")
                model_cfg = {}
        elif isinstance(mt_entry, dict):
            model_cfg = mt_entry
        else:
            model_cfg = {}

    # Thread training-level config into model_cfg
    train_cfg = cfg.get("training", {})
    model_cfg["amp"] = train_cfg.get("amp", True)
    model_cfg["gradient_checkpointing"] = train_cfg.get("gradient_checkpointing", True)
    model_cfg["num_workers"] = train_cfg.get("num_workers", 2)
    model_cfg["pin_memory"] = train_cfg.get("pin_memory", True)
    model_cfg["prefetch_factor"] = train_cfg.get("prefetch_factor", 2)

    seed = cfg.get("seed", {}).get("global", 42)
    t_start = time.time()
    set_seed(seed)
    use_cuda = torch.cuda.is_available() and cfg.get("device", {}).get("use_cuda", True)
    if use_cuda:
        try:
            torch.tensor([1.0], device="cuda").sum()
        except Exception:
            print("WARNING: CUDA device found but not usable. Falling back to CPU.")
            use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    _gpu_monitor._running = True
    monitor_thread = threading.Thread(target=_gpu_monitor, daemon=True)
    monitor_thread.start()

    target_col = cfg["data"]["target_col"]
    exp_ver = cfg.get("experiment", {}).get("version", "v1")
    ckpt_dir = Path(cfg["paths"].get("checkpoints_dir", "outputs/checkpoints/"))

    target = args.target or list(cfg.get("targets", {"tg": {}}).keys())[0]

    data_dir = Path(cfg["paths"]["data_dir"])

    if args.target:
        train = pd.read_parquet(data_dir / "processed" / "features_train.parquet")
        full_train = pd.read_csv(data_dir / "train.csv")
        if len(train) == len(full_train):
            # Same row count — use positional alignment (preserves fold split indices)
            train[target_col] = full_train[target_col].values
            train["target_type"] = full_train["target_type"].values
        else:
            # Row count mismatch due to canonicalization failures — use SMILES merge
            from rdkit import Chem
            def _canon(s):
                mol = Chem.MolFromSmiles(s)
                return Chem.MolToSmiles(mol, canonical=True) if mol else None
            canon_map = {s: _canon(s) for s in full_train["smiles"].unique()}
            full_train["canon_smiles"] = full_train["smiles"].map(canon_map)
            full_train = full_train.dropna(subset=["canon_smiles"])
            train = full_train[["canon_smiles", target_col, "target_type"]].merge(
                train.drop(columns=["canon_smiles"], errors="ignore"),
                left_on="canon_smiles", right_on="SMILES", how="left",
            )
            train = train.dropna(subset=[target_col]).reset_index(drop=True)
        train = train[train["target_type"] == target].reset_index(drop=True)
        splits_path = data_dir / f"splits_{target}.pkl"
    else:
        train = pd.read_parquet(data_dir / "processed" / "train_features.parquet")
        splits_path = data_dir / "splits.pkl"

    try:
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
        print(f"Loaded {len(splits)} fold splits from {splits_path}")
    except FileNotFoundError:
        print(f"{splits_path} not found. Generating splits on-the-fly ...")
        from data.generate_splits import generate_splits as _make_splits
        splits = _make_splits(str(splits_path), str(splits_path),
                              n_folds=cfg["cv"]["n_folds"], seed=seed,
                              target_col=target_col, smiles_col="SMILES")

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
                    if c not in ("SMILES", "id", "canon_smiles", "target_type", target_col)]
    scaler = None

    # Apply target transform
    y_tr_raw = tr_df[target_col].values
    y_tr_raw, inv_func, transform_name = apply_target_transform(y_tr_raw, cfg)
    if transform_name is not None:
        tr_df[target_col] = y_tr_raw
        print(f"Applied target transform: {transform_name}")

    if args.multitask or args.model_type == "multitask":
        # Multi-task training produces OOF for both tg and egc in one pass.
        pred_dir = Path(cfg["paths"]["predictions_dir"])
        info = train_multitask_per_fold(
            cfg=cfg, fold=args.fold, exp_ver=exp_ver,
            ckpt_dir=ckpt_dir, pred_dir=pred_dir, data_dir=data_dir,
            seed=seed, device=device,
        )
        print(f"[multitask | fold {args.fold}] {json.dumps(info)}")

        from experiments.manifest import record_run
        record_run(
            experiment=exp_ver, target="joint", model_type="multitask",
            fold=args.fold, score=None,
            checkpoint=str(ckpt_dir / f"{exp_ver}_multitask_fold{args.fold}_best.pt"),
            duration_sec=int(time.time() - t_start), seed=seed,
            config_path=args.config,
        )

        _gpu_monitor._running = False
        try:
            monitor_thread.join(timeout=5)
        except Exception:
            pass
        return

    if args.model_type in ("ridge", "xgb", "lgb", "catboost", "rf"):
        X_tr = tr_df[feature_cols].values
        y_tr = tr_df[target_col].values
        X_va = va_df[feature_cols].values
        y_va = va_df[target_col].values

        # Apply StandardScaler for linear models (Ridge)
        if args.model_type in MODELS_NEED_SCALING:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)

        model = build_model(args.model_type, model_cfg, n_features=len(feature_cols))[0]
        pred_va = train_tabular(model, X_tr, y_tr, X_va, y_va, model_cfg, args.model_type, device)
        # Save sklearn-style model checkpoint
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_tag = f"{exp_ver}_{target}_{args.model_type}_fold{args.fold}"
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
        y_tr = tr_df[target_col].values.astype(np.float32)
        X_va = va_df[feature_cols].values.astype(np.float32)
        y_va = va_df[target_col].values.astype(np.float32)

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
        ckpt_tag = f"{exp_ver}_{target}_{args.model_type}_fold{args.fold}"
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
    elif args.model_type == "polychain" or args.model_type.startswith("polychain_"):
        from features.graph_utils import build_multiscale
        from models.polychain.cst import compute_cst_batch

        # Build multi-scale graphs
        train_samples = [build_multiscale(s, y=y) for s, y in
                         zip(tr_df["SMILES"].tolist(), tr_df[target_col].tolist())]
        val_samples = [build_multiscale(s, y=y) for s, y in
                       zip(va_df["SMILES"].tolist(), va_df[target_col].tolist())]
        train_samples = [s for s in train_samples if s is not None]
        val_samples = [s for s in val_samples if s is not None]

        # CST calibration
        cst_train = compute_cst_batch([s.smiles for s in train_samples])
        cst_mean = cst_train.mean(axis=0)
        cst_std = cst_train.std(axis=0) + 1e-6

        train_loader = DataLoader(train_samples, batch_size=model_cfg.get("batch_size", 32),
                                  shuffle=True, collate_fn=_collate_polychain,
                                  num_workers=2, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=2)
        val_loader = DataLoader(val_samples, batch_size=model_cfg.get("batch_size", 32),
                                shuffle=False, collate_fn=_collate_polychain,
                                num_workers=2, pin_memory=True,
                                persistent_workers=True, prefetch_factor=2)

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
            "cst_dim": model_cfg.get("cst_dim", 32),
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
        # Save checkpoint for manifest recording
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_tag = f"{exp_ver}_{target}_{args.model_type}_fold{args.fold}"
        best_path = ckpt_dir / f"{ckpt_tag}_best.pt"
        final_path = ckpt_dir / f"{ckpt_tag}_final.pt"
        ckpt_payload = {
            "model_state": model.state_dict(),
            "model_type": args.model_type,
            "fold": args.fold,
            "epoch": 0,
            "val_rmse": best_val_rmse,
            "config": model_ckpt_cfg,
        }
        save_checkpoint(ckpt_payload, best_path)
        save_checkpoint(ckpt_payload, final_path)
        print(f"  Checkpoint saved -> {best_path}")
    else:
        # GNN baselines (GCN/GAT/MPNN/GraphTransformer/FusionNet)
        from features.graphs import smiles_to_graph
        train_graphs = [smiles_to_graph(s, y=y) for s, y in
                        zip(tr_df["SMILES"].tolist(), tr_df[target_col].tolist())]
        val_graphs = [smiles_to_graph(s, y=y) for s, y in
                      zip(va_df["SMILES"].tolist(), va_df[target_col].tolist())]
        train_graphs = [g for g in train_graphs if g is not None]
        val_graphs = [g for g in val_graphs if g is not None]

        in_dim = train_graphs[0].x.size(1)
        edge_dim = train_graphs[0].edge_attr.size(1)

        PyGDL = PyGDataLoader if PyGDataLoader is not None else DataLoader
        train_loader = PyGDL(train_graphs, batch_size=64, shuffle=True,
                             num_workers=2, pin_memory=True,
                             persistent_workers=True, prefetch_factor=2)
        val_loader = PyGDL(val_graphs, batch_size=64, shuffle=False,
                           num_workers=2, pin_memory=True,
                           persistent_workers=True, prefetch_factor=2)

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
        ckpt_tag = f"{exp_ver}_{target}_{args.model_type}_fold{args.fold}"

    # Metrics
    y_va = va_df[target_col].values
    # Inverse-transform predictions if target transform was applied
    if transform_name is not None:
        pred_va = inv_func(pred_va)
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
    out_file = pred_dir / f"{exp_ver}_{target}_{args.model_type}_fold{args.fold}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump({"val_idx": val_idx, "pred": pred_va, "y": y_va,
                     "metrics": metrics, "model_type": args.model_type,
                     "fold": args.fold, "target": target}, f)
    print(f"Saved predictions -> {out_file}")

    # Record run in experiment manifest
    from experiments.manifest import record_run
    ckpt_path = ckpt_dir / f"{ckpt_tag}_best.pt"
    record_run(
        experiment=exp_ver,
        target=target,
        model_type=args.model_type,
        fold=args.fold,
        score=metrics.get("r2"),
        checkpoint=str(ckpt_path) if ckpt_path.exists() else None,
        duration_sec=int(time.time() - t_start),
        seed=seed,
        config_path=args.config,
    )

    # Test inference
    if not args.skip_inference and args.target:
        test_feat = pd.read_parquet(data_dir / "processed" / "features_test.parquet")
        target_test_csv = data_dir / target / "test.csv"
        test_ids = pd.read_csv(target_test_csv)["id"].tolist()
        test_feat = test_feat[test_feat["id"].isin(test_ids)].reset_index(drop=True)
        test_feat_cols = [c for c in test_feat.columns if c in feature_cols]
        X_test = test_feat[test_feat_cols].values

        if args.model_type in ("ridge", "xgb", "lgb", "catboost", "rf"):
            if scaler is not None:
                X_test = scaler.transform(X_test)
            test_preds = model.predict(X_test)
        elif args.model_type in ("mlp",):
            if scaler is not None:
                X_test = scaler.transform(X_test).astype(np.float32)
            model.eval()
            with torch.no_grad():
                test_preds = model(torch.from_numpy(X_test).to(device)).squeeze(-1).cpu().numpy()
        elif args.model_type == "polychain" or args.model_type.startswith("polychain_"):
            from features.graph_utils import build_multiscale
            from models.polychain.cst import compute_cst_batch
            test_samples = [build_multiscale(s) for s in test_feat["SMILES"].tolist()]
            test_samples = [s for s in test_samples if s is not None]

            test_loader = DataLoader(test_samples, batch_size=64, shuffle=False, collate_fn=_collate_polychain,
                                      num_workers=2, pin_memory=True,
                                      persistent_workers=True, prefetch_factor=2)
            model.eval()
            test_preds = []
            with torch.no_grad():
                for batch_dict in test_loader:
                    batch_dict = move_to_device(batch_dict, device)
                    pred = model(batch_dict)
                    test_preds.append(pred.cpu().numpy())
            test_preds = np.concatenate(test_preds)
        else:
            from features.graphs import smiles_to_graph
            test_graphs = [smiles_to_graph(s) for s in test_feat["SMILES"].tolist()]
            test_graphs = [g for g in test_graphs if g is not None]
            from torch_geometric.loader import DataLoader as PyGDL
            test_loader = PyGDL(test_graphs, batch_size=64, shuffle=False,
                                 num_workers=2, pin_memory=True,
                                 persistent_workers=True, prefetch_factor=2)
            model.eval()
            test_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    test_preds.append(pred.cpu().numpy())
            test_preds = np.concatenate(test_preds)

        # Inverse-transform test predictions if target transform was applied
        if transform_name is not None:
            test_preds = inv_func(test_preds)

        test_out = pred_dir / f"{exp_ver}_{target}_{args.model_type}_fold{args.fold}_test.pkl"
        with open(test_out, "wb") as f:
            pickle.dump({
                "id": test_feat["id"].values.tolist(),
                "pred": test_preds.tolist(),
                "model_type": args.model_type,
                "fold": args.fold,
                "target": target,
            }, f)
        print(f"Test predictions saved -> {test_out}")


    _gpu_monitor._running = False
    monitor_thread.join(timeout=5)
    if torch.cuda.is_available():
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
