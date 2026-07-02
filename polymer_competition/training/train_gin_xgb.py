"""
training/train_gin_xgb.py
Stage 2: GIN embeddings (128-dim) + 6394 tabular features → XGBoost.

Extracts GIN encoder embeddings per fold (OOF), concatenates with
precomputed tabular features, then trains XGBoost with 5-fold CV.

Usage:
    python -m training.train_gin_xgb --target tg
    python -m training.train_gin_xgb --target egc
"""
from __future__ import annotations

import argparse
import pickle
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import xgboost as xgb
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch as PyGBatch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from features.graphs import smiles_to_graph
from models.gnn import GINRegressor
from training.train import set_seed


class GraphDataset(Dataset):
    def __init__(self, graphs, y=None):
        self.graphs = graphs
        self.y = y
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        if self.y is not None:
            return self.graphs[idx], self.y[idx]
        return self.graphs[idx]


def collate_graph(batch):
    if len(batch[0]) == 2:
        graphs = [b[0] for b in batch]
        ys = torch.tensor([b[1] for b in batch], dtype=torch.float)
        return PyGBatch.from_data_list(graphs), ys
    return PyGBatch.from_data_list(batch)


def build_graphs(smiles_list):
    graphs = []
    for s in smiles_list:
        g = smiles_to_graph(s)
        if g is not None:
            graphs.append(g)
    return graphs


def load_features(data_dir, target):
    """Load 6394 tabular features and return with metadata."""
    tr_feat = pd.read_parquet(data_dir / "processed" / "features_train.parquet")
    te_feat = pd.read_parquet(data_dir / "processed" / "features_test.parquet")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    exclude = {"id", "canon_smiles", "SMILES"}
    feat_cols = [c for c in tr_feat.columns if c not in exclude]
    X_tr = tr_feat[feat_cols].values.astype(np.float32)
    X_te = te_feat[feat_cols].values.astype(np.float32)

    tmask = train["target_type"].values == target
    X_tr = X_tr[tmask]
    y_tr = train["target"].values[tmask].astype(np.float32)
    tr_smiles = train["smiles"].values[tmask]

    # Test filtered by target
    tmask_te = test["target_type"].values == target
    X_te = X_te[tmask_te]
    te_smiles = test["smiles"].values[tmask_te]
    te_ids = test["id"].values[tmask_te]

    return X_tr, y_tr, tr_smiles, X_te, te_smiles, te_ids


def extract_embeddings(model, loader, device):
    model.eval()
    all_embs = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, list):
                batch = batch[0] if isinstance(batch[0], PyGBatch.__class__) else PyGBatch.from_data_list(batch)
            batch = batch.to(device)
            emb = model.get_embedding(batch)
            all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="tg")
    parser.add_argument("--gin_hidden", type=int, default=512)
    parser.add_argument("--gin_embed", type=int, default=128)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--gin_out", default="outputs/gin")
    parser.add_argument("--out", default="outputs/gin_xgb")
    parser.add_argument("--no_save", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(args.seed)
    use_cuda = torch.cuda.is_available() and cfg.get("device", {}).get("use_cuda", True)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    gin_dir = Path(args.gin_out) / args.target
    out_dir = Path(args.out) / args.target
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tabular features for {args.target}...")
    X_tr_tab, y_tr, tr_smiles, X_te_tab, te_smiles, te_ids = load_features(data_dir, args.target)
    print(f"  Train: {X_tr_tab.shape}, Test: {X_te_tab.shape}")

    print("Building graphs...")
    tr_graphs = build_graphs(tr_smiles.tolist())
    te_graphs = build_graphs(te_smiles.tolist())
    print(f"  Train graphs: {len(tr_graphs)}, Test graphs: {len(te_graphs)}")

    in_dim = tr_graphs[0].x.size(1)
    edge_dim = tr_graphs[0].edge_attr.size(1)

    # Load splits
    splits_path = data_dir / f"splits_{args.target}.pkl"
    with open(splits_path, "rb") as f:
        splits = pickle.load(f)

    n_folds = cfg["cv"]["n_folds"]
    n_train = len(tr_graphs)

    # Storage
    emb_dim = args.gin_embed
    oof_embeddings = np.zeros((n_train, emb_dim), dtype=np.float32)
    oof_count = np.zeros(n_train, dtype=np.int32)
    test_embeddings = np.zeros((len(te_graphs), emb_dim), dtype=np.float32)

    print(f"\nExtracting GIN embeddings ({n_folds} folds)...")
    for fold in range(n_folds):
        ckpt_path = gin_dir / "checkpoints" / f"gin_gin_fold{fold}_best.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: checkpoint {ckpt_path} not found, skipping fold {fold}")
            continue

        tr_idx, va_idx = splits[fold]["train"], splits[fold]["val"]
        tr_idx = [i for i in tr_idx if i < n_train]
        va_idx = [i for i in va_idx if i < n_train]

        # Load checkpoint
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Infer hidden_dim from checkpoint to handle mixed-dim checkpoints
        ckpt_hidden = state["model_state"]["encoder.atom_encoder.weight"].shape[0]
        ckpt_embed = state["model_state"]["encoder.output_proj.weight"].shape[0]

        model = GINRegressor(
            in_dim=in_dim, edge_dim=edge_dim,
            hidden_dim=ckpt_hidden, embed_dim=ckpt_embed,
            n_layers=3, dropout=0.0,
        )
        model.load_state_dict(state["model_state"])
        model = model.to(device)

        # Extract OOF (val) embeddings
        val_ds = GraphDataset([tr_graphs[i] for i in va_idx])
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                                num_workers=0, collate_fn=collate_graph)
        va_emb = extract_embeddings(model, val_loader, device)
        oof_embeddings[va_idx[:len(va_emb)]] += va_emb
        oof_count[va_idx[:len(va_emb)]] += 1

        # Extract test embeddings (all folds, will average)
        te_ds = GraphDataset(te_graphs)
        te_loader = DataLoader(te_ds, batch_size=64, shuffle=False,
                               num_workers=0, collate_fn=collate_graph)
        te_emb = extract_embeddings(model, te_loader, device)
        test_embeddings += te_emb

        print(f"  Fold {fold}: done (val={len(va_emb)}, test={len(te_emb)})")

        # Cleanup
        del model, val_loader, te_loader, val_ds, te_ds
        gc.collect()
        torch.cuda.empty_cache()

    # Average OOF and test embeddings
    mask = oof_count > 0
    oof_embeddings[mask] /= oof_count[mask, np.newaxis]
    test_embeddings /= n_folds

    # Combine embeddings with tabular features
    X_tr_combined = np.concatenate([X_tr_tab, oof_embeddings], axis=1)
    X_te_combined = np.concatenate([X_te_tab, test_embeddings], axis=1)

    print(f"\nCombined feature space: {X_tr_combined.shape}")

    # Standardize
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_combined)
    X_te_scaled = scaler.transform(X_te_combined)

    # XGBoost with 5-fold CV
    print(f"\nTraining XGBoost on combined features...")
    xgb_params = {
        "n_estimators": 1500,
        "max_depth": 7,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.5,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "random_state": args.seed,
        "n_jobs": -1,
        "verbosity": 0,
    }

    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds_list = []

    for fold in range(n_folds):
        tr_idx, va_idx = splits[fold]["train"], splits[fold]["val"]
        tr_idx = [i for i in tr_idx if i < n_train]
        va_idx = [i for i in va_idx if i < n_train]

        X_tr_fold = X_tr_scaled[tr_idx]
        y_tr_fold = y_tr[tr_idx]
        X_va_fold = X_tr_scaled[va_idx]
        y_va_fold = y_tr[va_idx]

        model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
        model.fit(
            X_tr_fold, y_tr_fold,
            eval_set=[(X_va_fold, y_va_fold)],
            verbose=False,
        )

        va_pred = model.predict(X_va_fold)
        oof_preds[va_idx] = va_pred

        te_pred = model.predict(X_te_scaled)
        test_preds_list.append(te_pred)

        fold_r2 = r2_score(y_va_fold, va_pred)
        print(f"  Fold {fold}: Val R²={fold_r2:.4f}")

    # OOF score
    oof_r2 = r2_score(y_tr[mask], oof_preds[mask])
    print(f"\nOverall OOF R²: {oof_r2:.4f}")

    if args.no_save:
        return

    # Save submission
    test_preds_avg = np.mean(test_preds_list, axis=0)
    te_sub = pd.DataFrame({"id": te_ids, "target": test_preds_avg})
    te_sub.to_csv(out_dir / f"submission_gin_xgb_{args.target}.csv", index=False)
    print(f"Saved -> {out_dir / f'submission_gin_xgb_{args.target}.csv'}")

    # Save OOF
    oof_data = {
        "val_idx": np.where(mask)[0],
        "pred": oof_preds[mask],
        "y": y_tr[mask],
        "model_type": "gin_xgb",
        "target": args.target,
    }
    with open(out_dir / f"oof_{args.target}_gin_xgb.pkl", "wb") as f:
        pickle.dump(oof_data, f)
    print(f"Saved -> {out_dir / f'oof_{args.target}_gin_xgb.pkl'}")

    return oof_r2


if __name__ == "__main__":
    main()
