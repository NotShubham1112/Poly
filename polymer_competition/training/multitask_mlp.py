"""Multi-Task Auxiliary MLP: predict target + RDKit auxiliary properties simultaneously.

The auxiliary tasks provide a strong, low-noise training signal that stabilizes
the shared encoder's gradients. After training, we discard the auxiliary heads
and use only the target head for predictions.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data")
PRED_DIR = Path("predictions")

# Auxiliary targets: polymer-relevant RDKit properties with good variance
AUX_COLS = [
    "sp3_c_frac", "rotatable_bonds", "ring_count", "aromatic_rings",
    "hbd", "hba", "polymer_mw", "polymer_logp", "polymer_tpsa",
    "chain_flexibility", "hansen_dp", "hansen_dP", "hansen_dH",
]


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


class MultiTaskMLP(nn.Module):
    """Shared encoder with separate output heads for target + auxiliary tasks."""

    def __init__(self, in_dim, n_aux, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        # Shared encoder
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
        self.encoder = nn.Sequential(*layers)

        # Target head (main prediction)
        self.target_head = nn.Sequential(
            nn.Linear(prev, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Auxiliary heads (separate for each property)
        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            for _ in range(n_aux)
        ])

    def forward(self, x):
        shared = self.encoder(x)
        target = self.target_head(shared).squeeze(-1)
        aux = [head(shared).squeeze(-1) for head in self.aux_heads]
        return target, aux


def train_fold(target, fold, feat, feature_cols, aux_cols, splits, device,
               n_epochs=200, patience=20, lr=5e-4, aux_weight=0.5):
    """Train multi-task MLP on one fold. Returns (oof_pred, test_pred, val_r2)."""
    train_idx = splits[fold]["train"]
    val_idx = splits[fold]["val"]

    tr_df = feat.iloc[train_idx]
    va_df = feat.iloc[val_idx]

    X_tr = tr_df[feature_cols].values.astype(np.float32)
    y_tr = tr_df["target"].values.astype(np.float32)
    X_va = va_df[feature_cols].values.astype(np.float32)
    y_va = va_df["target"].values.astype(np.float32)

    # Auxiliary targets
    aux_tr = tr_df[aux_cols].values.astype(np.float32)
    aux_va = va_df[aux_cols].values.astype(np.float32)

    # Standardize
    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_tr).astype(np.float32)
    X_va = scaler_X.transform(X_va).astype(np.float32)

    scaler_aux = StandardScaler()
    aux_tr = scaler_aux.fit_transform(aux_tr).astype(np.float32)
    aux_va = scaler_aux.transform(aux_va).astype(np.float32)

    # To tensors
    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    aux_tr_t = torch.from_numpy(aux_tr).to(device)
    X_va_t = torch.from_numpy(X_va).to(device)

    # Build model
    n_aux = len(aux_cols)
    model = MultiTaskMLP(len(feature_cols), n_aux, hidden_dims=[256, 128, 64],
                         dropout=0.3).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    target_loss_fn = nn.MSELoss()
    aux_loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, n_epochs + 1):
        model.train()
        idx = np.random.permutation(len(X_tr))
        for i in range(0, len(X_tr), 128):
            b = idx[i:i+128]
            xb = X_tr_t[b]
            yb = y_tr_t[b]
            aux_b = aux_tr_t[b]

            opt.zero_grad()
            pred_target, pred_aux = model(xb)

            # Combined loss: target + weighted auxiliary
            loss_target = target_loss_fn(pred_target, yb)
            loss_aux = sum(aux_loss_fn(pred_aux[j], aux_b[:, j]) for j in range(n_aux))
            loss = loss_target + aux_weight * loss_aux / n_aux
            loss.backward()
            opt.step()
        sched.step()

        # Validate
        model.eval()
        with torch.no_grad():
            pred_va, _ = model(X_va_t)
            pred_va_np = pred_va.cpu().numpy()
        val_r2 = r2_score(y_va, pred_va_np)
        val_loss = np.mean((pred_va_np - y_va) ** 2)

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
        pred_va, _ = model(X_va_t)
        pred_va_np = pred_va.cpu().numpy()
    val_r2 = r2_score(y_va, pred_va_np)

    return pred_va_np, val_r2, model, scaler_X, scaler_aux, best_state


def run(target):
    print(f"\n{'='*60}")
    print(f"  MULTI-TASK MLP: {target.upper()}")
    print(f"  Aux tasks: {len(AUX_COLS)} RDKit properties")
    print(f"{'='*60}")

    # Load features
    feat = pd.read_parquet(DATA_DIR / "processed" / "features_train.parquet")
    raw = pd.read_csv(DATA_DIR / "train.csv")
    if len(feat) == len(raw):
        feat["target"] = raw["target"].values
        feat["target_type"] = raw["target_type"].values
    feat = feat[feat["target_type"] == target].reset_index(drop=True)

    # Check which aux cols exist
    aux_cols = [c for c in AUX_COLS if c in feat.columns]
    print(f"  Available aux cols: {len(aux_cols)}: {aux_cols}")

    feature_cols = [c for c in feat.columns
                    if c not in ("id", "SMILES", "target", "target_type", "canon_smiles")
                    and c not in aux_cols]

    print(f"  Features: {len(feature_cols)}, Samples: {len(feat)}")

    with open(DATA_DIR / f"splits_{target}.pkl", "rb") as f:
        splits = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # 5-fold with 3 seeds
    n_seeds = 3
    all_oof = np.full(len(feat), np.nan)
    fold_r2s = []

    for fold in range(5):
        fold_preds = []
        for s in range(n_seeds):
            set_seed(42 + s)
            pred_va, val_r2, model, scaler_X, scaler_aux, state = train_fold(
                target, fold, feat, feature_cols, aux_cols, splits, device,
                n_epochs=200, patience=20, lr=5e-4, aux_weight=0.5
            )
            fold_preds.append(pred_va)
        mean_pred = np.mean(fold_preds, axis=0)
        val_idx = splits[fold]["val"]
        y_va = feat.iloc[val_idx]["target"].values
        r2 = r2_score(y_va, mean_pred)
        fold_r2s.append(r2)
        all_oof[val_idx] = mean_pred
        print(f"    Fold {fold}: R²={r2:.4f} ({n_seeds} seeds)")

    valid_mask = ~np.isnan(all_oof)
    oof_r2 = r2_score(feat["target"].values[valid_mask], all_oof[valid_mask])
    mean_fold_r2 = np.mean(fold_r2s)
    print(f"\n  Mean fold R²: {mean_fold_r2:.4f}, OOF R²: {oof_r2:.4f}")

    # Compare with original MLP
    orig_r2s = []
    for fold in range(5):
        with open(PRED_DIR / f"v27_{target}_mlp_fold{fold}.pkl", "rb") as f:
            orig_r2s.append(pickle.load(f)["metrics"]["r2"])
    orig_mean = np.mean(orig_r2s)
    print(f"  Original MLP: {orig_mean:.4f}")
    print(f"  Multi-task MLP: {oof_r2:.4f} (delta={oof_r2 - orig_mean:+.4f})")

    # Train on full data for test predictions
    print(f"\n  --- Training on full data for test predictions ---")
    feat_test = pd.read_parquet(DATA_DIR / "processed" / "features_test.parquet")
    X_test = feat_test[feature_cols].values.astype(np.float32)

    all_test_preds = []
    for s in range(n_seeds):
        set_seed(42 + s)

        X_full = feat[feature_cols].values.astype(np.float32)
        y_full = feat["target"].values.astype(np.float32)
        aux_full = feat[aux_cols].values.astype(np.float32)

        scaler_X = StandardScaler()
        X_full = scaler_X.fit_transform(X_full).astype(np.float32)
        X_test_scaled = scaler_X.transform(X_test).astype(np.float32)

        scaler_aux = StandardScaler()
        aux_full = scaler_aux.fit_transform(aux_full).astype(np.float32)

        X_full_t = torch.from_numpy(X_full).to(device)
        y_full_t = torch.from_numpy(y_full).to(device)
        aux_full_t = torch.from_numpy(aux_full).to(device)
        X_test_t = torch.from_numpy(X_test_scaled).to(device)

        n_aux = len(aux_cols)
        model = MultiTaskMLP(len(feature_cols), n_aux, hidden_dims=[256, 128, 64],
                             dropout=0.3).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

        target_loss_fn = nn.MSELoss()
        aux_loss_fn = nn.MSELoss()

        best_loss = float("inf")
        best_state = None
        bad = 0

        for ep in range(1, 201):
            model.train()
            idx = np.random.permutation(len(X_full))
            for i in range(0, len(X_full), 128):
                b = idx[i:i+128]
                xb = X_full_t[b]
                yb = y_full_t[b]
                aux_b = aux_full_t[b]
                opt.zero_grad()
                pred_target, pred_aux = model(xb)
                loss = target_loss_fn(pred_target, yb) + 0.5 * sum(
                    aux_loss_fn(pred_aux[j], aux_b[:, j]) for j in range(n_aux)) / n_aux
                loss.backward()
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                pred, _ = model(X_full_t[-len(X_full)//5:])
            val_loss = nn.MSELoss()(pred, y_full_t[-len(X_full)//5:]).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= 20:
                    break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_pred, _ = model(X_test_t)
        all_test_preds.append(test_pred.cpu().numpy())

    test_preds = np.mean(all_test_preds, axis=0)
    test_ids = feat_test["id"].values

    # Save
    out_path = PRED_DIR / f"multitask_mlp_{target}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "test_residuals": test_preds,
            "test_ids": test_ids,
            "oof_r2": oof_r2,
            "mean_fold_r2": mean_fold_r2,
            "fold_r2s": fold_r2s,
            "orig_mlp_r2": orig_mean,
        }, f)
    print(f"  Saved -> {out_path}")

    return {"oof_r2": oof_r2, "orig_mlp_r2": orig_mean}


if __name__ == "__main__":
    results = {}
    for target in ["tg", "egc"]:
        results[target] = run(target)

    print(f"\n{'='*60}")
    print("  MULTI-TASK RESULTS")
    print(f"{'='*60}")
    for target in ["tg", "egc"]:
        r = results[target]
        delta = r["oof_r2"] - r["orig_mlp_r2"]
        print(f"  {target.upper()}: orig MLP={r['orig_mlp_r2']:.4f}, "
              f"multi-task={r['oof_r2']:.4f} (delta={delta:+.4f})")
