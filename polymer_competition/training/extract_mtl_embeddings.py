"""Extract 64-dim latent embeddings from Multi-Task MLP encoder.

Saves OOF training embeddings and test embeddings for both targets,
to be concatenated with existing fingerprint features for tree retraining.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data") / "processed"
PRED_DIR = Path("predictions")
OUT_DIR = Path("data") / "processed"

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


class MultiTaskMLP(nn.Module):
    def __init__(self, in_dim, n_aux, hidden_dims=[256, 128, 64], dropout=0.3):
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
        self.encoder = nn.Sequential(*layers)
        self.target_head = nn.Sequential(
            nn.Linear(prev, 32), nn.ReLU(), nn.Linear(32, 1),
        )
        self.aux_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(prev, 32), nn.ReLU(), nn.Linear(32, 1))
            for _ in range(n_aux)
        ])

    def forward(self, x):
        shared = self.encoder(x)
        target = self.target_head(shared).squeeze(-1)
        aux = [head(shared).squeeze(-1) for head in self.aux_heads]
        return target, aux

    def get_embedding(self, x):
        return self.encoder(x)


def train_fold(fold, target, feat, feature_cols, aux_cols, splits, device,
               n_epochs=200, patience=20, lr=5e-4, aux_weight=0.5):
    """Train multi-task MLP on one fold, return model + scalers."""
    train_idx = splits[fold]["train"]
    val_idx = splits[fold]["val"]

    tr_df = feat.iloc[train_idx]
    va_df = feat.iloc[val_idx]

    X_tr = tr_df[feature_cols].values.astype(np.float32)
    y_tr = tr_df["target"].values.astype(np.float32)
    aux_tr = tr_df[aux_cols].values.astype(np.float32)
    X_va = va_df[feature_cols].values.astype(np.float32)
    y_va = va_df["target"].values.astype(np.float32)

    scaler_X = StandardScaler()
    X_tr = scaler_X.fit_transform(X_tr).astype(np.float32)
    X_va = scaler_X.transform(X_va).astype(np.float32)

    scaler_aux = StandardScaler()
    aux_tr = scaler_aux.fit_transform(aux_tr).astype(np.float32)

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    aux_tr_t = torch.from_numpy(aux_tr).to(device)
    X_va_t = torch.from_numpy(X_va).to(device)

    n_aux = len(aux_cols)
    model = MultiTaskMLP(len(feature_cols), n_aux, dropout=0.3).to(device)
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
            loss_target = target_loss_fn(pred_target, yb)
            loss_aux = sum(aux_loss_fn(pred_aux[j], aux_b[:, j]) for j in range(n_aux))
            loss = loss_target + aux_weight * loss_aux / n_aux
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pred_va, _ = model(X_va_t)
            pred_va_np = pred_va.cpu().numpy()
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
        val_emb = model.get_embedding(X_va_t).cpu().numpy()
    return model, scaler_X, scaler_aux, best_state, val_emb


def train_full(feature_cols, aux_cols, feat, device, n_epochs=200, patience=20, lr=5e-4, aux_weight=0.5):
    """Train on all data, return model with shared encoder for test embedding extraction."""
    X_full = feat[feature_cols].values.astype(np.float32)
    y_full = feat["target"].values.astype(np.float32)
    aux_full = feat[aux_cols].values.astype(np.float32)

    scaler_X = StandardScaler()
    X_full = scaler_X.fit_transform(X_full).astype(np.float32)
    scaler_aux = StandardScaler()
    aux_full = scaler_aux.fit_transform(aux_full).astype(np.float32)

    X_full_t = torch.from_numpy(X_full).to(device)
    y_full_t = torch.from_numpy(y_full).to(device)
    aux_full_t = torch.from_numpy(aux_full).to(device)

    n_aux = len(aux_cols)
    model = MultiTaskMLP(len(feature_cols), n_aux, dropout=0.3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    target_loss_fn = nn.MSELoss()
    aux_loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, n_epochs + 1):
        model.train()
        idx = np.random.permutation(len(X_full))
        for i in range(0, len(X_full), 128):
            b = idx[i:i+128]
            xb = X_full_t[b]
            yb = y_full_t[b]
            aux_b = aux_full_t[b]
            opt.zero_grad()
            pred_target, pred_aux = model(xb)
            loss = target_loss_fn(pred_target, yb) + aux_weight * sum(
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

    return model, scaler_X


def extract(target, n_seeds=3):
    print(f"\n{'='*60}")
    print(f"  EXTRACT MTL EMBEDDINGS: {target.upper()}")
    print(f"{'='*60}")

    feat = pd.read_parquet(DATA_DIR / "features_train.parquet")
    raw = pd.read_csv(Path("data") / "train.csv")
    if len(feat) == len(raw):
        feat["target"] = raw["target"].values
        feat["target_type"] = raw["target_type"].values
    feat = feat[feat["target_type"] == target].reset_index(drop=True)
    print(f"  Samples: {len(feat)}")

    aux_cols = [c for c in AUX_COLS if c in feat.columns]
    feature_cols = [c for c in feat.columns
                    if c not in ("id", "SMILES", "target", "target_type", "canon_smiles")
                    and c not in aux_cols]
    print(f"  Features: {len(feature_cols)}, Aux: {len(aux_cols)}")

    with open(Path("data") / f"splits_{target}.pkl", "rb") as f:
        splits = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # 1. OOF embeddings (from fold models)
    all_oof_emb = np.full((len(feat), 64), np.nan)
    all_emb_folds = []

    for fold in range(5):
        fold_embs = []
        for s in range(n_seeds):
            set_seed(42 + s)
            model, scaler_X, scaler_aux, state, val_emb = train_fold(
                fold, target, feat, feature_cols, aux_cols, splits, device,
            )
            if s == 0:
                fold_embs.append(val_emb)
            else:
                fold_embs[0] += val_emb
        mean_emb = fold_embs[0] / n_seeds
        val_idx = splits[fold]["val"]
        all_oof_emb[val_idx] = mean_emb
        all_emb_folds.append(mean_emb)
        print(f"    Fold {fold}: val_emb shape {mean_emb.shape}")

    # 2. Test embeddings (from full-data model)
    set_seed(42)
    full_model, _ = train_full(feature_cols, aux_cols, feat, device)
    feat_test = pd.read_parquet(DATA_DIR / "features_test.parquet")
    X_test = feat_test[feature_cols].values.astype(np.float32)
    scaler_X = StandardScaler()
    scaler_X.fit(feat[feature_cols].values.astype(np.float32))
    X_test_scaled = scaler_X.transform(X_test).astype(np.float32)
    full_model.eval()
    with torch.no_grad():
        test_emb = full_model.get_embedding(torch.from_numpy(X_test_scaled).to(device)).cpu().numpy()

    # 3. Save
    emb_cols = [f"MTL_{i}" for i in range(64)]

    oof_df = pd.DataFrame(all_oof_emb, columns=emb_cols)
    oof_df["target_type"] = target.upper()
    train_out = OUT_DIR / f"mtl_embeddings_{target}_train.parquet"
    oof_df.to_parquet(train_out)
    print(f"  Saved OOF embeddings -> {train_out} ({len(oof_df)} rows)")

    test_df = pd.DataFrame(test_emb, columns=emb_cols)
    test_df["id"] = feat_test["id"].values
    test_out = OUT_DIR / f"mtl_embeddings_{target}_test.parquet"
    test_df.to_parquet(test_out)
    print(f"  Saved test embeddings -> {test_out} ({len(test_df)} rows)")

    # Sanity: check for NaNs
    nan_frac = np.isnan(all_oof_emb).mean()
    print(f"  OOF NaN fraction: {nan_frac:.6f}")

    return oof_df, test_df


if __name__ == "__main__":
    for target in ["tg", "egc"]:
        extract(target, n_seeds=3)
    print("\nDone!")
