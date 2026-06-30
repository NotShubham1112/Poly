"""Residual boosting: train GNNs to predict ensemble residuals.

Pipeline:
1. Load pre-computed residuals (y_true - ensemble_oof) for each sample.
2. Train GCN/GAT/MPNN on these residuals for 5 folds.
3. Generate test predictions for residuals.
4. Final prediction = ensemble_test + gnn_residual_test.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import time
import yaml

DATA_DIR = Path("data")
PRED_DIR = Path("predictions")
CKPT_DIR = Path("outputs/checkpoints")

# We'll use the existing GNN infrastructure
import torch
import torch.nn as nn
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def train_gnn_residual(target, model_type="gat", n_epochs=150, patience=15,
                        lr=5e-4, hidden_dim=128, n_layers=3):
    """Train a GNN to predict residuals for one target."""
    print(f"\n{'='*60}")
    print(f"  RESIDUAL GNN: {target.upper()} ({model_type})")
    print(f"{'='*60}")

    # Load residuals
    with open(DATA_DIR / f"residuals_{target}.pkl", "rb") as f:
        resid_data = pickle.load(f)
    residuals = resid_data["residuals"]
    y_true = resid_data["y"]

    # Load splits
    with open(DATA_DIR / f"splits_{target}.pkl", "rb") as f:
        splits = pickle.load(f)

    # Load features to get SMILES
    feat_train = pd.read_parquet(DATA_DIR / "processed" / "features_train.parquet")
    raw_train = pd.read_csv(DATA_DIR / "train.csv")
    if len(feat_train) == len(raw_train):
        feat_train["target"] = raw_train["target"].values
        feat_train["target_type"] = raw_train["target_type"].values
    feat_train = feat_train[feat_train["target_type"] == target].reset_index(drop=True)

    smiles_list = feat_train["SMILES"].tolist()

    # Load test features to get test SMILES
    feat_test = pd.read_parquet(DATA_DIR / "processed" / "features_test.parquet")
    test_smiles = feat_test["SMILES"].tolist()
    test_ids = feat_test["id"].values

    print(f"  Train samples: {len(smiles_list)}, Test samples: {len(test_smiles)}")
    print(f"  Residual std: {residuals.std():.4f}")

    # Import GNN building blocks
    from features.graphs import smiles_to_graph
    from models.gnn import get_gnn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Build all training graphs (with residual as y)
    print("  Building training graphs...")
    all_graphs = []
    valid_indices = []
    for i, (s, r) in enumerate(zip(smiles_list, residuals)):
        g = smiles_to_graph(s, y=float(r))
        if g is not None:
            all_graphs.append(g)
            valid_indices.append(i)

    if not all_graphs:
        print("  ERROR: No valid graphs"); return None

    in_dim = all_graphs[0].x.size(1)
    edge_dim = all_graphs[0].edge_attr.size(1)
    print(f"  Valid train graphs: {len(all_graphs)}, in_dim={in_dim}, edge_dim={edge_dim}")

    # Build test graphs
    print("  Building test graphs...")
    test_graphs = []
    test_valid_indices = []
    for i, s in enumerate(test_smiles):
        g = smiles_to_graph(s)
        if g is not None:
            test_graphs.append(g)
            test_valid_indices.append(i)

    print(f"  Valid test graphs: {len(test_graphs)}")

    from torch_geometric.loader import DataLoader as PyGDL

    valid_indices = np.array(valid_indices)
    valid_residuals = residuals[valid_indices]

    # 5-fold training
    all_test_residuals = np.zeros(len(test_smiles))
    all_test_counts = np.zeros(len(test_smiles))
    fold_r2s = []

    for fold in range(5):
        print(f"\n  --- Fold {fold} ---")
        train_idx = splits[fold]["train"]
        val_idx = splits[fold]["val"]

        # Map splits indices to valid_indices
        # splits indices are into feat_train; valid_indices tells us which ones became graphs
        train_valid_mask = np.isin(valid_indices, train_idx)
        val_valid_mask = np.isin(valid_indices, val_idx)

        train_graphs = [all_graphs[i] for i in range(len(all_graphs)) if train_valid_mask[i]]
        val_graphs_list = [all_graphs[i] for i in range(len(all_graphs)) if val_valid_mask[i]]

        if not train_graphs or not val_graphs_list:
            print(f"  Skip fold {fold}: no valid graphs"); continue

        train_loader = PyGDL(train_graphs, batch_size=64, shuffle=True,
                             num_workers=0, pin_memory=True)
        val_loader = PyGDL(val_graphs_list, batch_size=64, shuffle=False,
                           num_workers=0)

        # Build model
        model = get_gnn(model_type, in_dim, edge_dim,
                        hidden_dim=hidden_dim, n_layers=n_layers, dropout=0.2)
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        bad = 0

        for ep in range(1, n_epochs + 1):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = model(batch).squeeze(-1)
                loss = criterion(pred, batch.y.view(-1))
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_preds = []
                val_targets = []
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = model(batch).squeeze(-1)
                    val_preds.append(pred.cpu().numpy())
                    val_targets.append(batch.y.view(-1).cpu().numpy())
                val_preds = np.concatenate(val_preds)
                val_targets = np.concatenate(val_targets)
                val_loss = np.sqrt(np.mean((val_preds - val_targets) ** 2))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
            model = model.to(device)

        # Evaluate on val fold
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_targets = []
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch).squeeze(-1)
                val_preds.append(pred.cpu().numpy())
                val_targets.append(batch.y.view(-1).cpu().numpy())
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)
            val_r2 = r2_score(val_targets, val_preds)
            fold_r2s.append(val_r2)
            print(f"    Val residual R²: {val_r2:.4f}, RMSE: {best_val_loss:.4f}")

        # Predict on test set
        model.eval()
        test_loader = PyGDL(test_graphs, batch_size=64, shuffle=False, num_workers=0)
        with torch.no_grad():
            test_preds = []
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch).squeeze(-1)
                test_preds.append(pred.cpu().numpy())
            test_preds = np.concatenate(test_preds)

        # Accumulate test predictions (each fold predicts on all test graphs)
        for i, gi in enumerate(test_valid_indices):
            all_test_residuals[gi] += test_preds[i]
            all_test_counts[gi] += 1

    # Average test predictions across folds
    mask = all_test_counts > 0
    all_test_residuals[mask] /= all_test_counts[mask]

    mean_r2 = np.mean(fold_r2s) if fold_r2s else 0
    print(f"\n  Mean val residual R²: {mean_r2:.4f}")

    # Save results
    result = {
        "test_residuals": all_test_residuals,
        "test_ids": test_ids,
        "val_r2": mean_r2,
        "fold_r2s": fold_r2s,
        "model_type": model_type,
    }
    out_path = PRED_DIR / f"residual_gnn_{target}_{model_type}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  Saved -> {out_path}")

    return result


def build_residual_submission(targets=("tg", "egc")):
    """Combine ensemble test predictions + GNN residual test predictions."""
    print(f"\n{'='*60}")
    print(f"  BUILDING RESIDUAL BOOSTED SUBMISSION")
    print(f"{'='*60}")

    test_feat = pd.read_parquet(DATA_DIR / "processed" / "features_test.parquet")
    raw_test = pd.read_csv(DATA_DIR / "test.csv")

    rows = []
    for target in targets:
        ttype = "tg" if target == "tg" else "egc"
        test_mask = raw_test["target_type"] == ttype

        # Load ensemble test predictions
        ensemble_preds = np.zeros(len(test_feat))
        ensemble_counts = np.zeros(len(test_feat))
        for model_type in ["xgb", "lgb", "catboost", "rf"]:
            for fold in range(5):
                path = PRED_DIR / f"v27_{target}_{model_type}_fold{fold}_test.pkl"
                if not path.exists():
                    continue
                with open(path, "rb") as f:
                    d = pickle.load(f)
                pred = np.array(d["pred"])
                ids = np.array(d["id"])
                id_to_idx = {id_val: idx for idx, id_val in enumerate(test_feat["id"].values)}
                for j, (id_val, p) in enumerate(zip(ids, pred)):
                    if id_val in id_to_idx:
                        idx = id_to_idx[id_val]
                        ensemble_preds[idx] += p
                        ensemble_counts[idx] += 1
        mask = ensemble_counts > 0
        ensemble_preds[mask] /= ensemble_counts[mask]

        # Load GNN residual predictions (best model type)
        best_gnn = None
        best_r2 = -999
        for gnn_type in ["gat", "gcn", "mpnn"]:
            path = PRED_DIR / f"residual_gnn_{target}_{gnn_type}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    d = pickle.load(f)
                if d["val_r2"] > best_r2:
                    best_r2 = d["val_r2"]
                    best_gnn = d
                    best_gnn_name = gnn_type

        if best_gnn is None:
            print(f"  WARNING: No GNN residual for {target}, using ensemble only")
            residual_preds = np.zeros(len(test_feat))
        else:
            print(f"  {target.upper()}: best GNN = {best_gnn_name} (R²={best_r2:.4f})")
            residual_preds = best_gnn["test_residuals"]

        # Final prediction = ensemble + residual
        final_preds = ensemble_preds + residual_preds

        # Clip to training range
        raw_train = pd.read_csv(DATA_DIR / "train.csv")
        feat_train = pd.read_parquet(DATA_DIR / "processed" / "features_train.parquet")
        if len(feat_train) == len(raw_train):
            feat_train["target"] = raw_train["target"].values
            feat_train["target_type"] = raw_train["target_type"].values
        train_target = feat_train[feat_train["target_type"] == ttype]["target"]
        lower = float(np.percentile(train_target, 0.5))
        upper = float(np.percentile(train_target, 99.5))
        final_preds = np.clip(final_preds, lower, upper)

        # Map to test samples with correct target_type
        for _, row in raw_test[test_mask].iterrows():
            sample_id = row["id"]
            id_to_idx = {id_val: idx for idx, id_val in enumerate(test_feat["id"].values)}
            if sample_id in id_to_idx:
                pred_val = float(final_preds[id_to_idx[sample_id]])
            else:
                pred_val = 0.0
            rows.append({"id": sample_id, "target": pred_val})

    sub = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    sub.to_csv("outputs/submissions/submission.csv", index=False)
    print(f"\n  Submission: {len(sub)} rows, NaN={sub['target'].isna().sum()}")
    print(f"  Saved -> outputs/submissions/submission.csv")

    # Copy to Material
    import shutil
    shutil.copy2("outputs/submissions/submission.csv", "D:/Parth/Poly/Material/submission.csv")
    print(f"  Copied -> D:/Parth/Poly/Material/submission.csv")

    return sub


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "submit":
        build_residual_submission()
    else:
        # Train GNNs on residuals for both targets
        for target in ["tg", "egc"]:
            for gnn_type in ["gat", "gcn", "mpnn"]:
                train_gnn_residual(target, model_type=gnn_type,
                                   n_epochs=150, patience=15, lr=5e-4)
        # Build submission
        build_residual_submission()
