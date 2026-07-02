"""
training/layer2_embeddings.py
Extract GIN + Hybrid fusion embeddings, combine with 6394 features, train XGBoost.

Usage:
    python -m training.layer2_embeddings
"""
from __future__ import annotations

import pickle
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch as PyGBatch

from features.graphs import smiles_to_graph
from models.gnn import GINRegressor
from models.hybrid import HybridNet
from training.train import set_seed


TARGETS = ["tg", "egc"]
XGB_PARAMS = {
    "n_estimators": 1500, "max_depth": 7, "learning_rate": 0.03,
    "subsample": 0.85, "colsample_bytree": 0.5, "min_child_weight": 3,
    "gamma": 0.1, "reg_alpha": 1.0, "reg_lambda": 2.0,
    "random_state": 42, "n_jobs": -1, "verbosity": 0,
}
BATCH_SIZE = 64


class GraphDS(Dataset):
    def __init__(self, gs): self.gs = gs
    def __len__(self): return len(self.gs)
    def __getitem__(self, i): return self.gs[i]


def collate(batch):
    return PyGBatch.from_data_list(batch)


class TabDS(Dataset):
    def __init__(self, tab, gs=None):
        self.tab = torch.from_numpy(tab).float()
        self.gs = gs
    def __len__(self):
        return len(self.tab)
    def __getitem__(self, i):
        if self.gs is not None:
            return self.gs[i], self.tab[i]
        return self.tab[i]


def collate_tab(batch):
    if isinstance(batch[0], tuple):
        gs = [b[0] for b in batch]
        tabs = torch.stack([b[1] for b in batch])
        return PyGBatch.from_data_list(gs), tabs
    return torch.stack(batch)


def build_graphs(smiles_list):
    graphs = []
    for s in smiles_list:
        g = smiles_to_graph(s)
        if g is not None:
            graphs.append(g)
    return graphs


def extract_gin_embeddings(gin_dir, target, data_dir, device, n_folds=5):
    """Extract GIN encoder embeddings OOF per fold."""
    tr = pd.read_csv(data_dir / "train.csv")
    mask = tr["target_type"].values == target
    y = tr["target"].values[mask].astype(np.float32)
    smiles = tr["smiles"].values[mask]
    graphs = build_graphs(smiles)

    with open(data_dir / f"splits_{target}.pkl", "rb") as f:
        splits = pickle.load(f)

    in_dim = graphs[0].x.size(1)
    edge_dim = graphs[0].edge_attr.size(1)
    n = len(graphs)
    emb_dim = 128
    oof_emb = np.zeros((n, emb_dim), dtype=np.float32)

    for fold in range(n_folds):
        ckpt = torch.load(
            gin_dir / target / "checkpoints" / f"gin_gin_fold{fold}_best.pt",
            map_location="cpu", weights_only=False,
        )
        ms = ckpt["model_state"]
        ch = ms["encoder.atom_encoder.weight"].shape[0]
        ce = ms["encoder.output_proj.weight"].shape[0]

        model = GINRegressor(in_dim, edge_dim, hidden_dim=ch, embed_dim=ce,
                             n_layers=3, dropout=0.0)
        model.load_state_dict(ms)
        model = model.to(device)
        model.eval()

        _, va_idx = splits[fold]["train"], splits[fold]["val"]
        va_idx = [i for i in va_idx if i < n]

        loader = DataLoader(GraphDS([graphs[i] for i in va_idx]),
                           batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=0, collate_fn=collate)
        embs = []
        with torch.no_grad():
            for batch in loader:
                embs.append(model.get_embedding(batch.to(device)).cpu().numpy())
        oof_emb[va_idx] = np.concatenate(embs)

    return oof_emb, y, graphs


def extract_hybrid_embeddings(hybrid_dir, target, data_dir, device, n_folds=5):
    """Extract HybridNet fusion embeddings OOF per fold."""
    tr = pd.read_csv(data_dir / "train.csv")
    mask = tr["target_type"].values == target
    y = tr["target"].values[mask].astype(np.float32)
    smiles = tr["smiles"].values[mask]
    graphs = build_graphs(smiles)

    X_tr = pd.read_parquet(data_dir / "processed" / "features_train.parquet")
    exclude = {"id", "canon_smiles", "SMILES"}
    feat_cols = [c for c in X_tr.columns if c not in exclude]
    X_arr = X_tr[feat_cols].values.astype(np.float32)[mask]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_arr)

    with open(data_dir / f"splits_{target}.pkl", "rb") as f:
        splits = pickle.load(f)

    in_dim = graphs[0].x.size(1)
    edge_dim = graphs[0].edge_attr.size(1)
    n_features = X_arr.shape[1]
    n = len(graphs)

    # HybridNet uses graph_hidden=512, embed_dim=128, fusion_proj=256
    oof_emb = np.zeros((n, 512), dtype=np.float32)  # 256+256 from fusion

    for fold in range(n_folds):
        ckpt_path = hybrid_dir / target / "checkpoints" / f"hybrid_fold{fold}_best.pt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ms = state
        else:
            print(f"  WARNING: {ckpt_path} not found, using final checkpoint")
            ckpt_path = hybrid_dir / target / "checkpoints" / f"hybrid_fold{fold}_final.pt"
            if not ckpt_path.exists():
                print(f"  SKIP: no checkpoint for fold {fold}")
                continue
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ms = state

        model = HybridNet(
            in_dim=in_dim, edge_dim=edge_dim, n_features=n_features,
            graph_hidden=512, graph_embed=128,
            tab_hidden=1024, tab_embed=512,
            fusion_proj=256, n_layers=3, dropout=0.0,
        )
        if isinstance(ms, dict) and "model_state" in ms:
            model.load_state_dict(ms["model_state"])
        else:
            model.load_state_dict(ms)
        model = model.to(device)
        model.eval()

        _, va_idx = splits[fold]["train"], splits[fold]["val"]
        va_idx = [i for i in va_idx if i < n]

        gs_va = [graphs[i] for i in va_idx]
        tab_va = Xs[va_idx]
        va_ds = TabDS(tab_va, gs_va)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, collate_fn=collate_tab)

        embs = []
        with torch.no_grad():
            for gb, tb in va_loader:
                embs.append(model.get_fusion_embedding(
                    gb.to(device), tb.to(device)).cpu().numpy())
        oof_emb[va_idx] = np.concatenate(embs)

    return oof_emb, Xs, y, scaler


def main():
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    use_cuda = cfg.get("device", {}).get("use_cuda", True) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    set_seed(42)
    data_dir = Path("data")
    gin_dir = Path("outputs/gin")
    hybrid_dir = Path("outputs/hybrid")
    out_dir = Path("outputs/layer2")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"  {target.upper()}")
        print(f"{'='*60}")

        tr = pd.read_csv(data_dir / "train.csv")
        mask = tr["target_type"].values == target
        y = tr["target"].values[mask].astype(np.float32)
        n = len(y)

        # Load tabular features
        X_tr = pd.read_parquet(data_dir / "processed" / "features_train.parquet")
        exclude = {"id", "canon_smiles", "SMILES"}
        feat_cols = [c for c in X_tr.columns if c not in exclude]
        X_arr = X_tr[feat_cols].values.astype(np.float32)[mask]
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_arr)

        with open(data_dir / f"splits_{target}.pkl", "rb") as f:
            splits = pickle.load(f)

        # 1. Extract GIN embeddings
        print("\nExtracting GIN embeddings...")
        gin_emb, y_g, graphs = extract_gin_embeddings(gin_dir, target, data_dir, device)

        # 2. Extract Hybrid fusion embeddings
        print("Extracting Hybrid fusion embeddings...")
        try:
            hybrid_emb, Xs_hyb, y_h, sc = extract_hybrid_embeddings(
                hybrid_dir, target, data_dir, device)
        except Exception as e:
            print(f"  Hybrid extraction failed: {e}")
            hybrid_emb = None

        # 3. Combined feature experiments
        print("\n--- XGBoost on combined features ---")
        configs = {
            "tabular_only": Xs,
            "tabular+gin_emb": np.concatenate([Xs, gin_emb], axis=1),
        }
        if hybrid_emb is not None:
            configs["tabular+hybrid_emb"] = np.concatenate([Xs, hybrid_emb], axis=1)
            configs["tabular+gin+hybrid_emb"] = np.concatenate([Xs, gin_emb, hybrid_emb], axis=1)
            configs["gin+hybrid_emb"] = np.concatenate([gin_emb, hybrid_emb], axis=1)

        for name, X_comb in configs.items():
            X_sc = StandardScaler().fit_transform(X_comb)
            oof = np.zeros(n, dtype=np.float32)
            for fold in range(5):
                ti, vi = splits[fold]["train"], splits[fold]["val"]
                ti = [i for i in ti if i < n]
                vi = [i for i in vi if i < n]

                m = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=50)
                m.fit(X_sc[ti], y[ti], eval_set=[(X_sc[vi], y[vi])], verbose=False)
                oof[vi] = m.predict(X_sc[vi])
            r2 = r2_score(y, oof)
            print(f"  {name:35s}: R2={r2:.4f}")
            results[(target, name)] = r2

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for target in TARGETS:
        print(f"\n  {target.upper()}:")
        for (t, name), r2 in results.items():
            if t == target:
                print(f"    {name:35s}: R2={r2:.4f}")


if __name__ == "__main__":
    main()
