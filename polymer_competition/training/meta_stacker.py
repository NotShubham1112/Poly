"""
training/meta_stacker.py
Stack OOF predictions from GIN, XGB, Hybrid using XGBoost meta-learner.

Usage:
    python -m training.meta_stacker
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
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch as PyGBatch

from features.graphs import smiles_to_graph
from models.gnn import GINRegressor
from training.train import set_seed


TARGETS = ["tg", "egc"]


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


def compute_gin_oof(gin_dir, target, data_dir, device, n_folds=5):
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
    preds = np.zeros(n, dtype=np.float32)

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
                           batch_size=64, shuffle=False, num_workers=0, collate_fn=collate)
        fp = []
        with torch.no_grad():
            for batch in loader:
                fp.append(model(batch.to(device)).cpu().numpy())
        preds[va_idx] = np.concatenate(fp)

    return preds, y


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    use_cuda = cfg.get("device", {}).get("use_cuda", True) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    set_seed(42)
    data_dir = Path("data")
    gin_dir = Path("outputs/gin")
    out_dir = Path("outputs/meta_stacker")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"  {target.upper()}")
        print(f"{'='*60}")

        tr = pd.read_csv(data_dir / "train.csv")
        mask = tr["target_type"].values == target
        y = tr["target"].values[mask].astype(np.float32)
        n = len(y)

        with open(data_dir / f"splits_{target}.pkl", "rb") as f:
            splits = pickle.load(f)

        # Collect OOF predictions from all available models
        models = {}

        # GIN
        gin_preds, y_gin = compute_gin_oof(gin_dir, target, data_dir, device)
        models["GIN"] = gin_preds
        print(f"  GIN: R2={r2_score(y, gin_preds):.4f}")

        # XGB (compute fresh)
        from sklearn.preprocessing import StandardScaler
        X_tr = pd.read_parquet(data_dir / "processed" / "features_train.parquet")
        exclude = {"id", "canon_smiles", "SMILES"}
        feat_cols = [c for c in X_tr.columns if c not in exclude]
        X_arr = X_tr[feat_cols].values.astype(np.float32)[mask]
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_arr)

        xgbp = {"n_estimators": 1500, "max_depth": 7, "learning_rate": 0.03,
                "subsample": 0.85, "colsample_bytree": 0.5, "min_child_weight": 3,
                "gamma": 0.1, "reg_alpha": 1.0, "reg_lambda": 2.0,
                "random_state": 42, "n_jobs": -1, "verbosity": 0}

        xgb_preds = np.zeros(n, dtype=np.float32)
        for fold in range(5):
            ti, vi = splits[fold]["train"], splits[fold]["val"]
            ti = [i for i in ti if i < n]
            vi = [i for i in vi if i < n]
            m = xgb.XGBRegressor(**xgbp, early_stopping_rounds=50)
            m.fit(Xs[ti], y[ti], eval_set=[(Xs[vi], y[vi])], verbose=False)
            xgb_preds[vi] = m.predict(Xs[vi])
        models["XGB"] = xgb_preds
        print(f"  XGB: R2={r2_score(y, xgb_preds):.4f}")

        # Hybrid (load from saved OOF)
        hybrid_path = f"outputs/hybrid/{target}/oof_{target}_hybrid.pkl"
        if Path(hybrid_path).exists():
            d = pickle.load(open(hybrid_path, "rb"))
            models["Hybrid"] = d["pred"]
            print(f"  Hybrid: R2={r2_score(y, models['Hybrid']):.4f}")

        model_names = list(models.keys())

        # Residual correlation
        print(f"\n  --- Residual Analysis ---")
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                n1, n2 = model_names[i], model_names[j]
                e1 = y - models[n1]
                e2 = y - models[n2]
                r_err, _ = pearsonr(e1, e2)
                r_pred, _ = pearsonr(models[n1], models[n2])
                print(f"  {n1} vs {n2}: r(pred)={r_pred:.4f}, r(error)={r_err:.4f}")

        # --- Meta-stacker with XGBoost on OOF predictions ---
        print(f"\n  --- Meta-Stacker ---")

        # Build meta features: OOF predictions from each model
        meta_X = np.column_stack([models[n] for n in model_names])
        meta_scaler = StandardScaler()
        meta_Xs = meta_scaler.fit_transform(meta_X)

        meta_preds = np.zeros(n, dtype=np.float32)
        for fold in range(5):
            ti, vi = splits[fold]["train"], splits[fold]["val"]
            ti = [i for i in ti if i < n]
            vi = [i for i in vi if i < n]

            # XGBoost on meta features
            m_xgb = xgb.XGBRegressor(
                n_estimators=500, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, n_jobs=-1, verbosity=0,
                early_stopping_rounds=30,
            )
            m_xgb.fit(meta_Xs[ti], y[ti], eval_set=[(meta_Xs[vi], y[vi])], verbose=False)
            meta_preds[vi] = m_xgb.predict(meta_Xs[vi])

        meta_r2 = r2_score(y, meta_preds)
        print(f"  XGB(meta on {model_names}): R2={meta_r2:.4f}")

        # RidgeCV on meta features
        m_ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        ridge_preds = np.zeros(n, dtype=np.float32)
        for fold in range(5):
            ti, vi = splits[fold]["train"], splits[fold]["val"]
            ti = [i for i in ti if i < n]
            vi = [i for i in vi if i < n]
            m_ridge.fit(meta_Xs[ti], y[ti])
            ridge_preds[vi] = m_ridge.predict(meta_Xs[vi])
        ridge_r2 = r2_score(y, ridge_preds)
        print(f"  Ridge(meta on {model_names}): R2={ridge_r2:.4f}")

        # Best 2-way blend for comparison
        from scipy.optimize import minimize
        best_r2 = -1
        best_w = None
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                def nr2(w):
                    return -r2_score(y, w[0]*models[model_names[i]] + (1-w[0])*models[model_names[j]])
                res = minimize(nr2, [0.5], bounds=[(0,1)], method="L-BFGS-B")
                bl = res.x[0]*models[model_names[i]] + (1-res.x[0])*models[model_names[j]]
                br2 = -res.fun
                if br2 > best_r2:
                    best_r2 = br2
                    best_w = (model_names[i], model_names[j], res.x[0])

        print(f"  Best 2-way blend ({best_w[0]}:{best_w[2]:.3f}, {best_w[1]}:{1-best_w[2]:.3f}): R2={best_r2:.4f}")

        all_results[target] = {
            "individual": {n: r2_score(y, models[n]) for n in model_names},
            "best_2way_blend": best_r2,
            "xgb_meta": meta_r2,
            "ridge_meta": ridge_r2,
            "residual_corrs": {},
        }

        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                n1, n2 = model_names[i], model_names[j]
                e1 = y - models[n1]
                e2 = y - models[n2]
                all_results[target]["residual_corrs"][f"{n1}_vs_{n2}"] = pearsonr(e1, e2)[0]

    # Summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'TG':>10} {'EGC':>10} {'Mean':>10}")
    print(f"  {'-'*55}")

    models_combined = set()
    for t in TARGETS:
        models_combined.update(all_results[t]["individual"].keys())
    models_combined = sorted(models_combined)

    for model in models_combined:
        tg = all_results["tg"]["individual"].get(model, 0)
        egc = all_results["egc"]["individual"].get(model, 0)
        mean = (tg + egc) / 2
        print(f"  {model:<25} {tg:>10.4f} {egc:>10.4f} {mean:>10.4f}")

    # Meta methods
    for method in ["best_2way_blend", "xgb_meta", "ridge_meta"]:
        tg = all_results["tg"].get(method, 0)
        egc = all_results["egc"].get(method, 0)
        mean = (tg + egc) / 2
        print(f"  {method:<25} {tg:>10.4f} {egc:>10.4f} {mean:>10.4f}")

    # Residual correlation summary
    print(f"\n  --- Residual Correlations ---")
    for t in TARGETS:
        for pair, r in all_results[t]["residual_corrs"].items():
            print(f"  {t} {pair}: r(error)={r:.4f}")

    # Save results
    with open(out_dir / "results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved -> {out_dir / 'results.pkl'}")


if __name__ == "__main__":
    main()
