"""Residual MLP: train a small MLP on top-100 XGB features to predict ensemble residuals.
The key difference from previous MLP attempts:
- Target is RESIDUALS (low-variance, zero-centered) not raw Y
- Only 100 features instead of 6394 (fixes the aspect ratio)
- Tiny architecture [128, 64] with high dropout (0.5)
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn

DATA_DIR = Path("data")
PRED_DIR = Path("predictions")


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims=[128, 64], dropout=0.5):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_residual_mlp(target, n_epochs=200, patience=20, lr=5e-4, n_seeds=5):
    print(f"\n{'='*60}")
    print(f"  RESIDUAL MLP: {target.upper()} (top-100 features, {n_seeds} seeds)")
    print(f"{'='*60}")

    # Load residuals
    with open(DATA_DIR / f"residuals_{target}.pkl", "rb") as f:
        resid_data = pickle.load(f)
    residuals = resid_data["residuals"]
    y_true = resid_data["y"]

    # Load top features
    with open(DATA_DIR / "top_features.pkl", "rb") as f:
        top_feats = pickle.load(f)
    feature_cols = top_feats[target][:100]

    # Load features
    feat = pd.read_parquet(DATA_DIR / "processed" / "features_train.parquet")
    raw = pd.read_csv(DATA_DIR / "train.csv")
    if len(feat) == len(raw):
        feat["target"] = raw["target"].values
        feat["target_type"] = raw["target_type"].values
    feat = feat[feat["target_type"] == target].reset_index(drop=True)

    X_all = feat[feature_cols].values.astype(np.float32)

    # Standardize features
    from sklearn.preprocessing import StandardScaler

    # Load splits
    with open(DATA_DIR / f"splits_{target}.pkl", "rb") as f:
        splits = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}, Features: {len(feature_cols)}, Samples: {len(X_all)}")
    print(f"  Residual std: {residuals.std():.4f}, target std: {y_true.std():.4f}")

    # 5-fold cross-validation with multiple seeds
    all_oof_preds = np.full(len(X_all), np.nan)
    fold_r2s = []

    for fold in range(5):
        train_idx = splits[fold]["train"]
        val_idx = splits[fold]["val"]

        X_tr = X_all[train_idx]
        y_tr = residuals[train_idx]
        X_va = X_all[val_idx]
        y_va = residuals[val_idx]

        # Standardize
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_va = scaler.transform(X_va).astype(np.float32)

        seed_preds = []
        for s in range(n_seeds):
            set_seed(42 + s)
            model = ResidualMLP(len(feature_cols), hidden_dims=[128, 64], dropout=0.5).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

            best_val_loss = float("inf")
            best_state = None
            bad = 0

            X_tr_t = torch.from_numpy(X_tr).to(device)
            y_tr_t = torch.from_numpy(y_tr).to(device)
            X_va_t = torch.from_numpy(X_va).to(device)

            for ep in range(1, n_epochs + 1):
                model.train()
                idx = np.random.permutation(len(X_tr))
                for i in range(0, len(X_tr), 128):
                    b = idx[i:i+128]
                    xb = X_tr_t[b]
                    yb = y_tr_t[b]
                    opt.zero_grad()
                    pred = model(xb)
                    loss = nn.MSELoss()(pred, yb)
                    loss.backward()
                    opt.step()
                sched.step()

                model.eval()
                with torch.no_grad():
                    val_pred = model(X_va_t).cpu().numpy()
                val_loss = np.mean((val_pred - y_va) ** 2)
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
            model.eval()
            with torch.no_grad():
                val_pred = model(X_va_t).cpu().numpy()
            seed_preds.append(val_pred)

        # Average across seeds
        mean_pred = np.mean(seed_preds, axis=0)
        val_r2 = r2_score(y_va, mean_pred)
        fold_r2s.append(val_r2)
        all_oof_preds[val_idx] = mean_pred
        print(f"    Fold {fold}: residual R²={val_r2:.4f} (n_seeds={n_seeds})")

    # Overall OOF metrics
    valid_mask = ~np.isnan(all_oof_preds)
    oof_r2 = r2_score(residuals[valid_mask], all_oof_preds[valid_mask])
    mean_fold_r2 = np.mean(fold_r2s)
    print(f"\n  Mean fold R²: {mean_fold_r2:.4f}, OOF R²: {oof_r2:.4f}")

    # Now train on FULL data for test predictions
    print(f"\n  --- Training on full data for test predictions ---")
    scaler_full = StandardScaler()
    X_full = scaler_full.fit_transform(X_all).astype(np.float32)

    # Load test features
    feat_test = pd.read_parquet(DATA_DIR / "processed" / "features_test.parquet")
    X_test = feat_test[feature_cols].values.astype(np.float32)
    X_test = scaler_full.transform(X_test).astype(np.float32)
    test_ids = feat_test["id"].values

    test_preds_seeds = []
    for s in range(n_seeds):
        set_seed(42 + s)
        model = ResidualMLP(len(feature_cols), hidden_dims=[128, 64], dropout=0.5).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

        X_full_t = torch.from_numpy(X_full).to(device)
        y_full_t = torch.from_numpy(residuals).to(device)
        X_test_t = torch.from_numpy(X_test).to(device)

        best_val_loss = float("inf")
        best_state = None
        bad = 0

        for ep in range(1, n_epochs + 1):
            model.train()
            idx = np.random.permutation(len(X_full))
            for i in range(0, len(X_full), 128):
                b = idx[i:i+128]
                xb = X_full_t[b]
                yb = y_full_t[b]
                opt.zero_grad()
                pred = model(xb)
                loss = nn.MSELoss()(pred, yb)
                loss.backward()
                opt.step()
            sched.step()

            # Use last 20% as internal val for early stopping
            n_val = len(X_full) // 5
            model.eval()
            with torch.no_grad():
                val_pred = model(X_full_t[-n_val:]).cpu().numpy()
            val_loss = np.mean((val_pred - residuals[-n_val:]) ** 2)
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
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t).cpu().numpy()
        test_preds_seeds.append(test_pred)
        print(f"    Seed {s}: {ep} epochs")

    test_residuals = np.mean(test_preds_seeds, axis=0)

    # Save
    result = {
        "test_residuals": test_residuals,
        "test_ids": test_ids,
        "oof_r2": oof_r2,
        "mean_fold_r2": mean_fold_r2,
        "fold_r2s": fold_r2s,
    }
    out_path = PRED_DIR / f"residual_mlp_{target}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  Saved -> {out_path}")

    return result


if __name__ == "__main__":
    results = {}
    for target in ["tg", "egc"]:
        results[target] = train_residual_mlp(target, n_epochs=200, patience=20,
                                              lr=5e-4, n_seeds=5)

    print(f"\n{'='*60}")
    print("  RESIDUAL MLP RESULTS")
    print(f"{'='*60}")
    for target in ["tg", "egc"]:
        r = results[target]
        print(f"  {target.upper()}: OOF R²={r['oof_r2']:.4f}, fold mean={r['mean_fold_r2']:.4f}")
