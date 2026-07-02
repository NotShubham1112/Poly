"""
training/generate_final_submission.py
Generate final submission using Ridge on GIN + XGB + Hybrid OOF predictions.

Usage:
    python -m training.generate_final_submission
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch as PyGBatch

from features.graphs import smiles_to_graph
from models.gnn import GINRegressor
from models.hybrid import HybridNet
from training.train import set_seed
from training.train_utils import rmse


BATCH_SIZE = 64
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


class TabGraphDS(Dataset):
    def __init__(self, tab, gs):
        self.tab = torch.from_numpy(tab).float()
        self.gs = gs
    def __len__(self): return len(self.gs)
    def __getitem__(self, i): return self.gs[i], self.tab[i]


def collate_graph(batch):
    return PyGBatch.from_data_list(batch)


def collate_tab_graph(batch):
    gs = [b[0] for b in batch]
    tabs = torch.stack([b[1] for b in batch])
    return PyGBatch.from_data_list(gs), tabs


def build_graphs(smiles_list):
    graphs = []
    for s in smiles_list:
        g = smiles_to_graph(s)
        if g is not None:
            graphs.append(g)
    return graphs


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    use_cuda = cfg.get("device", {}).get("use_cuda", True) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    set_seed(42)

    data_dir = Path("data")
    gin_dir = Path("outputs/gin")
    hybrid_dir = Path("outputs/hybrid")
    out_dir = Path("outputs/final_submission")
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude = {"id", "canon_smiles", "SMILES"}

    final_parts = []
    all_info = {}

    for target in ["tg", "egc"]:
        print(f"\n{'='*60}")
        print(f"  {target.upper()}")
        print(f"{'='*60}")

        # Data
        tr = pd.read_csv(data_dir / "train.csv")
        mask = tr["target_type"].values == target
        y = tr["target"].values[mask].astype(np.float32)
        n = len(y)

        te = pd.read_csv(data_dir / "test.csv")
        tmask = te["target_type"].values == target
        te_ids = te["id"].values[tmask]

        X_tr = pd.read_parquet(data_dir / "processed" / "features_train.parquet")
        X_te = pd.read_parquet(data_dir / "processed" / "features_test.parquet")
        feat_cols = [c for c in X_tr.columns if c not in exclude]
        X_arr = X_tr[feat_cols].values.astype(np.float32)[mask]
        X_te_arr = X_te[feat_cols].values.astype(np.float32)[tmask]
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X_arr)
        X_te_s = scaler.transform(X_te_arr)

        tr_smiles = tr["smiles"].values[mask]
        te_smiles = te["smiles"].values[tmask]

        with open(data_dir / f"splits_{target}.pkl", "rb") as f:
            splits = pickle.load(f)

        # Graphs
        tr_graphs = build_graphs(tr_smiles)
        te_graphs = build_graphs(te_smiles)
        in_dim = tr_graphs[0].x.size(1)
        edge_dim = tr_graphs[0].edge_attr.size(1)

        has_hybrid_ckpt = (hybrid_dir / target / "checkpoints" / "hybrid_fold0_best.pt").exists()

        # --- GIN ---
        print("  GIN...")
        gin_oof = np.zeros(n, dtype=np.float32)
        gin_te = np.zeros(len(te_graphs), dtype=np.float32)
        for fold in range(5):
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
            loader = DataLoader(GraphDS([tr_graphs[i] for i in va_idx]),
                               batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=0, collate_fn=collate_graph)
            fp = []
            with torch.no_grad():
                for batch in loader:
                    fp.append(model(batch.to(device)).cpu().numpy())
            gin_oof[va_idx] = np.concatenate(fp)

            te_loader = DataLoader(GraphDS(te_graphs), batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0, collate_fn=collate_graph)
            fp_te = []
            with torch.no_grad():
                for batch in te_loader:
                    fp_te.append(model(batch.to(device)).cpu().numpy())
            gin_te += np.concatenate(fp_te)
        gin_te /= 5
        print(f"    OOF R2={r2_score(y, gin_oof):.4f}")

        # --- XGB ---
        print("  XGB...")
        xgb_oof = np.zeros(n, dtype=np.float32)
        xgb_te = np.zeros(X_te_s.shape[0], dtype=np.float32)
        for fold in range(5):
            ti, vi = splits[fold]["train"], splits[fold]["val"]
            ti = [i for i in ti if i < n]
            vi = [i for i in vi if i < n]
            m = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=50)
            m.fit(Xs[ti], y[ti], eval_set=[(Xs[vi], y[vi])], verbose=False)
            xgb_oof[vi] = m.predict(Xs[vi])
            xgb_te += m.predict(X_te_s)
        xgb_te /= 5
        print(f"    OOF R2={r2_score(y, xgb_oof):.4f}")

        # --- Hybrid (OOF only, test only if checkpoints exist) ---
        hybrid_oof = None
        hybrid_te = None
        if has_hybrid_ckpt:
            print("  Hybrid (with test predictions)...")
            hybrid_oof = np.zeros(n, dtype=np.float32)
            hybrid_te = np.zeros(len(te_graphs), dtype=np.float32)
            for fold in range(5):
                ckpt = torch.load(
                    hybrid_dir / target / "checkpoints" / f"hybrid_fold{fold}_best.pt",
                    map_location="cpu", weights_only=False,
                )
                ms = ckpt["model_state"] if "model_state" in ckpt else ckpt
                model = HybridNet(in_dim=in_dim, edge_dim=edge_dim,
                                  n_features=X_arr.shape[1],
                                  graph_hidden=512, graph_embed=128,
                                  tab_hidden=1024, tab_embed=512,
                                  fusion_proj=256, n_layers=3, dropout=0.0)
                model.load_state_dict(ms)
                model = model.to(device)
                model.eval()

                _, va_idx = splits[fold]["train"], splits[fold]["val"]
                va_idx = [i for i in va_idx if i < n]
                ds = TabGraphDS(Xs[va_idx], [tr_graphs[i] for i in va_idx])
                loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=0, collate_fn=collate_tab_graph)
                fp = []
                with torch.no_grad():
                    for gb, tb in loader:
                        fp.append(model(gb.to(device), tb.to(device)).cpu().numpy())
                hybrid_oof[va_idx] = np.concatenate(fp)

                te_ds = TabGraphDS(X_te_s, te_graphs)
                te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=0, collate_fn=collate_tab_graph)
                fp_te = []
                with torch.no_grad():
                    for gb, tb in te_loader:
                        fp_te.append(model(gb.to(device), tb.to(device)).cpu().numpy())
                hybrid_te += np.concatenate(fp_te)
            hybrid_te /= 5
            print(f"    OOF R2={r2_score(y, hybrid_oof):.4f}")
        else:
            print("  Hybrid (OOF only — no checkpoints for test)...")
            hybrid_oof = pickle.load(open(f"outputs/hybrid/{target}/oof_{target}_hybrid.pkl", "rb"))["pred"]
            print(f"    OOF R2={r2_score(y, hybrid_oof):.4f}")

        # --- Ridge meta-stacker ---
        oof_dict = {"GIN": gin_oof, "XGB": xgb_oof}
        if hybrid_te is not None:
            oof_dict["Hybrid"] = hybrid_oof

        names = list(oof_dict.keys())
        meta_X = np.column_stack([oof_dict[n] for n in names])
        meta_scaler = StandardScaler()
        meta_Xs = meta_scaler.fit_transform(meta_X)

        ridge = Ridge(alpha=1.0)
        ridge.fit(meta_Xs, y)
        ridge_oof = ridge.predict(meta_Xs)
        print(f"  Ridge OOF R2={r2_score(y, ridge_oof):.4f}")
        print(f"  Ridge weights: {dict(zip(names, ridge.coef_))}")

        # Test predictions
        te_dict = {"GIN": gin_te, "XGB": xgb_te}
        if hybrid_te is not None:
            te_dict["Hybrid"] = hybrid_te

        # Only use models available for test
        test_names = [n for n in names if n in te_dict]
        test_meta_X = np.column_stack([te_dict[n] for n in test_names])

        if len(test_names) < len(names):
            # Re-fit Ridge with only available test models
            oof_sub = np.column_stack([oof_dict[n] for n in test_names])
            ss = StandardScaler()
            oof_sub_s = ss.fit_transform(oof_sub)
            ridge2 = Ridge(alpha=1.0)
            ridge2.fit(oof_sub_s, y)
            test_meta_s = ss.transform(test_meta_X)
            final_preds = ridge2.predict(test_meta_s)
            print(f"  Re-fit Ridge on {test_names} for test: w={dict(zip(test_names, ridge2.coef_))}")
        else:
            test_meta_s = meta_scaler.transform(test_meta_X)
            final_preds = ridge.predict(test_meta_s)

        sub = pd.DataFrame({"id": te_ids, "target": final_preds})
        sub.to_csv(out_dir / f"submission_{target}.csv", index=False)
        print(f"  Saved {len(sub)} rows -> {out_dir / f'submission_{target}.csv'}")
        final_parts.append(sub)
        all_info[target] = {"GIN": r2_score(y, gin_oof), "XGB": r2_score(y, xgb_oof),
                           "Ridge": r2_score(y, ridge_oof)}

    full = pd.concat(final_parts, ignore_index=True)
    full.to_csv(out_dir / "submission_final.csv", index=False)
    print(f"\nFull submission -> {out_dir / 'submission_final.csv'} ({len(full)} rows)")

    # Mean R2
    tgs = all_info["tg"]["Ridge"]
    egcs = all_info["egc"]["Ridge"]
    print(f"\nMean R² (Ridge meta-stacker): {(tgs + egcs) / 2:.4f}")
    print(f"  TG: {tgs:.4f}, EGC: {egcs:.4f}")


if __name__ == "__main__":
    main()
