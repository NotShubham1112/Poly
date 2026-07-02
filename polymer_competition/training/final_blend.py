"""
training/final_blend.py
Compute optimal blend of GIN, XGBoost, and Hybrid models.
Generates OOF predictions, finds optimal weights, and creates submission.

Usage:
    python -m training.final_blend
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch as PyGBatch

from features.graphs import smiles_to_graph
from models.gnn import GINRegressor
from training.train import set_seed


TARGETS = ["tg", "egc"]
XGB_PARAMS = {
    "n_estimators": 1500, "max_depth": 7, "learning_rate": 0.03,
    "subsample": 0.85, "colsample_bytree": 0.5, "min_child_weight": 3,
    "gamma": 0.1, "reg_alpha": 1.0, "reg_lambda": 2.0,
    "random_state": 42, "n_jobs": -1, "verbosity": 0,
}


class GraphDS(Dataset):
    def __init__(self, gs): self.gs = gs
    def __len__(self): return len(self.gs)
    def __getitem__(self, i): return self.gs[i]


def collate(batch):
    return PyGBatch.from_data_list(batch)


def build_graphs(smiles_list):
    graphs = []
    for s in smiles_list:
        g = smiles_to_graph(s)
        if g is not None:
            graphs.append(g)
    return graphs


def compute_gin_oof(gin_dir, target, data_dir, device="cpu"):
    """Compute GIN OOF predictions for all folds."""
    tr = pd.read_csv(data_dir / "train.csv")
    mask = tr["target_type"].values == target
    y = tr["target"].values[mask].astype(np.float32)
    smiles = tr["smiles"].values[mask]
    graphs = build_graphs(smiles)
    in_dim = graphs[0].x.size(1)
    edge_dim = graphs[0].edge_attr.size(1)

    with open(data_dir / f"splits_{target}.pkl", "rb") as f:
        splits = pickle.load(f)

    n = len(graphs)
    preds = np.zeros(n, dtype=np.float32)

    for fold in range(5):
        ckpt_path = gin_dir / target / "checkpoints" / f"gin_gin_fold{fold}_best.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: {ckpt_path} not found")
            continue

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ms = state["model_state"]
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
                           batch_size=64, shuffle=False, num_workers=0,
                           collate_fn=collate)
        fp = []
        with torch.no_grad():
            for batch in loader:
                fp.append(model(batch.to(device)).cpu().numpy())
        preds[va_idx] = np.concatenate(fp)

    return preds, y, graphs


def compute_xgb_oof(X_arr, y, splits, n):
    """Compute XGBoost OOF predictions."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_arr)
    preds = np.zeros(n, dtype=np.float32)

    for fold in range(5):
        ti, vi = splits[fold]["train"], splits[fold]["val"]
        ti = [i for i in ti if i < n]
        vi = [i for i in vi if i < n]

        m = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=50)
        m.fit(Xs[ti], y[ti], eval_set=[(Xs[vi], y[vi])], verbose=False)
        preds[vi] = m.predict(Xs[vi])
    return preds, scaler





def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    use_cuda = cfg.get("device", {}).get("use_cuda", True) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    set_seed(42)
    data_dir = Path("data")
    gin_dir = Path("outputs/gin")
    out_dir = Path("outputs/final_blend")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    test_preds = {}

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"  {target.upper()}")
        print(f"{'='*60}")

        # Load tabular features
        X_tr = pd.read_parquet(data_dir / "processed/features_train.parquet")
        exclude = {"id", "canon_smiles", "SMILES"}
        feat_cols = [c for c in X_tr.columns if c not in exclude]

        tr = pd.read_csv(data_dir / "train.csv")
        mask = tr["target_type"].values == target
        y = tr["target"].values[mask].astype(np.float32)
        X_arr = X_tr[feat_cols].values.astype(np.float32)[mask]

        with open(data_dir / f"splits_{target}.pkl", "rb") as f:
            splits = pickle.load(f)

        n = len(y)

        # 1. GIN OOF
        print("Computing GIN OOF...")
        gin_preds, y_g, _ = compute_gin_oof(gin_dir, target, data_dir, device)
        gin_r2 = r2_score(y, gin_preds)
        print(f"  GIN: R2={gin_r2:.4f}")

        # 2. XGBoost OOF
        print("Computing XGBoost OOF...")
        Xs_global = None
        xgb_preds = np.zeros(n, dtype=np.float32)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_arr)
        for fold in range(5):
            ti, vi = splits[fold]["train"], splits[fold]["val"]
            ti = [i for i in ti if i < n]
            vi = [i for i in vi if i < n]
            m = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=50)
            m.fit(Xs[ti], y[ti], eval_set=[(Xs[vi], y[vi])], verbose=False)
            xgb_preds[vi] = m.predict(Xs[vi])
        xgb_r2 = r2_score(y, xgb_preds)
        print(f"  XGB: R2={xgb_r2:.4f}")

        # 3. Hybrid OOF
        hybrid_path = f"outputs/hybrid/{target}/oof_{target}_hybrid.pkl"
        if Path(hybrid_path).exists():
            d = pickle.load(open(hybrid_path, "rb"))
            hybrid_preds = d["pred"]
            hybrid_r2 = r2_score(y, hybrid_preds)
            print(f"  Hybrid: R2={hybrid_r2:.4f}")
        else:
            hybrid_preds = None
            hybrid_r2 = None

        # Model dict
        models = {"GIN": gin_preds, "XGB": xgb_preds}
        if hybrid_preds is not None:
            models["Hybrid"] = hybrid_preds

        # Correlations
        names = list(models.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                c, _ = pearsonr(models[names[i]], models[names[j]])
                print(f"  r({names[i]}, {names[j]}) = {c:.4f}")

        # 2-way blends
        print("\n  2-way blends:")
        best_r2 = -1
        best_weights = None
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                n1, n2 = names[i], names[j]
                def nr2(w):
                    return -r2_score(y, w[0] * models[n1] + (1-w[0]) * models[n2])
                res = minimize(nr2, [0.5], bounds=[(0, 1)], method="L-BFGS-B")
                w = res.x[0]
                bl = w * models[n1] + (1-w) * models[n2]
                br2 = -res.fun
                print(f"    {n1}({w:.3f}) + {n2}({1-w:.3f}): R2={br2:.4f}")
                if br2 > best_r2:
                    best_r2 = br2
                    best_weights = {n1: w, n2: 1-w}

        # 3-way blend
        if len(names) >= 3:
            print("  3-way blend:")
            def nr3(w):
                w = np.clip(w, 0, 1)
                w = w / w.sum()
                pred = sum(w[i] * models[names[i]] for i in range(len(names)))
                return -r2_score(y, pred)
            res = minimize(nr3, np.ones(len(names))/len(names),
                          bounds=[(0, 1)]*len(names), method="L-BFGS-B")
            w3 = res.x / res.x.sum()
            parts = " + ".join(f"{names[i]}({w3[i]:.3f})" for i in range(len(names)))
            r3 = -res.fun
            print(f"    {parts}: R2={r3:.4f}")
            if r3 > best_r2:
                best_r2 = r3
                best_weights = dict(zip(names, w3))

        results[target] = {
            "GIN": gin_r2,
            "XGB": xgb_r2,
            "Hybrid": hybrid_r2,
            "best_blend": best_r2,
            "best_weights": best_weights,
        }

    # Mean R²
    mean_r2 = np.mean([results[t]["best_blend"] for t in TARGETS])
    print(f"\n{'='*60}")
    print(f"  FINAL: Mean R² = {mean_r2:.4f}")
    print(f"{'='*60}")

    # Save results
    with open(out_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"Saved -> {out_dir / 'results.pkl'}")

    # Print summary table
    print(f"\n  Summary:")
    print(f"  {'Model':<15} {'TG':>10} {'EGC':>10} {'Mean':>10}")
    print(f"  {'-'*45}")
    for model in ["GIN", "XGB", "Hybrid", "best_blend"]:
        tg = results["tg"].get(model, 0)
        egc = results["egc"].get(model, 0)
        mean = np.mean([v for v in [tg, egc] if v is not None])
        print(f"  {model:<15} {tg if tg else 0:>10.4f} {egc if egc else 0:>10.4f} {mean:>10.4f}")


if __name__ == "__main__":
    main()
