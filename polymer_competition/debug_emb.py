import pickle, numpy as np
from sklearn.metrics import r2_score
import xgboost as xgb

data = pickle.load(open("outputs/gin_xgb/tg/oof_tg_gin_xgb.pkl", "rb"))
y = data["y"]
p = data["pred"]
print(f"OOF R2: {r2_score(y, p):.4f}")
print(f"y: mean={y.mean():.2f}, std={y.std():.2f}")
print(f"p: mean={p.mean():.2f}, std={p.std():.2f}")

# Load full data
import pandas as pd
import torch
from pathlib import Path
from torch_geometric.data import Batch as PyGBatch
from torch.utils.data import Dataset, DataLoader
from features.graphs import smiles_to_graph
from models.gnn import GINRegressor

data_dir = Path("data")
tr = pd.read_csv(data_dir / "train.csv")
tmask = tr["target_type"].values == "tg"
y_tr = tr["target"].values[tmask].astype(np.float32)
tr_smiles = tr["smiles"].values[tmask]

# Build graphs
graphs = []
for s in tr_smiles:
    g = smiles_to_graph(s)
    if g is not None:
        graphs.append(g)

# Load splits
with open(data_dir / "splits_tg.pkl", "rb") as f:
    splits = pickle.load(f)

in_dim = graphs[0].x.size(1)
edge_dim = graphs[0].edge_attr.size(1)

n_train = len(graphs)
emb_dim = 128
oof_embeddings = np.zeros((n_train, emb_dim), dtype=np.float32)
oof_count = np.zeros(n_train, dtype=np.int32)

for fold in range(5):
    ckpt = torch.load(f"outputs/gin/tg/checkpoints/gin_gin_fold{fold}_best.pt", map_location="cpu", weights_only=False)
    ms = ckpt["model_state"]
    ch = ms["encoder.atom_encoder.weight"].shape[0]
    ce = ms["encoder.output_proj.weight"].shape[0]
    
    model = GINRegressor(in_dim=in_dim, edge_dim=edge_dim, hidden_dim=ch, embed_dim=ce, n_layers=3, dropout=0.0)
    model.load_state_dict(ms)
    model.eval()

    tr_idx, va_idx = splits[fold]["train"], splits[fold]["val"]
    va_idx = [i for i in va_idx if i < n_train]
    
    class GD(Dataset):
        def __init__(self, gs): self.gs = gs
        def __len__(self): return len(self.gs)
        def __getitem__(self, i): return self.gs[i]
    
    va_ds = GD([graphs[i] for i in va_idx])
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=0,
                          collate_fn=lambda b: PyGBatch.from_data_list(b))
    
    all_embs = []
    with torch.no_grad():
        for batch in va_loader:
            emb = model.get_embedding(batch)
            all_embs.append(emb.numpy())
    va_emb = np.concatenate(all_embs)
    
    oof_embeddings[va_idx[:len(va_emb)]] += va_emb
    oof_count[va_idx[:len(va_emb)]] += 1

mask = oof_count > 0
oof_embeddings[mask] /= oof_count[mask, np.newaxis]

# Test XGBoost on embeddings only
xgb_params = {"n_estimators": 1500, "max_depth": 7, "learning_rate": 0.03,
              "subsample": 0.85, "colsample_bytree": 0.5, "min_child_weight": 3,
              "gamma": 0.1, "reg_alpha": 1.0, "reg_lambda": 2.0,
              "random_state": 42, "n_jobs": -1, "verbosity": 0}

print("\n=== XGBoost on GIN embeddings only ===")
oof_preds = np.zeros(n_train, dtype=np.float32)
for fold in range(5):
    tr_idx, va_idx = splits[fold]["train"], splits[fold]["val"]
    tr_idx = [i for i in tr_idx if i < n_train]
    va_idx = [i for i in va_idx if i < n_train]
    
    model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    model.fit(oof_embeddings[tr_idx], y_tr[tr_idx],
              eval_set=[(oof_embeddings[va_idx], y_tr[va_idx])], verbose=False)
    va_pred = model.predict(oof_embeddings[va_idx])
    oof_preds[va_idx] = va_pred
    print(f"  Fold {fold}: R2={r2_score(y_tr[va_idx], va_pred):.4f}")

print(f"Overall OOF R2: {r2_score(y_tr[mask], oof_preds[mask]):.4f}")

# Also test: XGBoost on original tabular features
from sklearn.preprocessing import StandardScaler
X_tr = pd.read_parquet(data_dir / "processed/features_train.parquet")
exc = {"id", "canon_smiles", "SMILES"}
fc = [c for c in X_tr.columns if c not in exc]
X_arr = X_tr[fc].values.astype(np.float32)[tmask]

scaler = StandardScaler()
Xs = scaler.fit_transform(X_arr)

print("\n=== XGBoost on tabular features only ===")
oof_preds2 = np.zeros(n_train, dtype=np.float32)
for fold in range(5):
    tr_idx, va_idx = splits[fold]["train"], splits[fold]["val"]
    tr_idx = [i for i in tr_idx if i < n_train]
    va_idx = [i for i in va_idx if i < n_train]
    
    model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    model.fit(Xs[tr_idx], y_tr[tr_idx],
              eval_set=[(Xs[va_idx], y_tr[va_idx])], verbose=False)
    va_pred = model.predict(Xs[va_idx])
    oof_preds2[va_idx] = va_pred
    print(f"  Fold {fold}: R2={r2_score(y_tr[va_idx], va_pred):.4f}")

print(f"Overall OOF R2: {r2_score(y_tr[mask], oof_preds2[mask]):.4f}")

# Combined
print("\n=== XGBoost on combined ===")
X_comb = np.concatenate([scaler.transform(X_arr), oof_embeddings], axis=1)
Xcs = StandardScaler().fit_transform(X_comb)

oof_preds3 = np.zeros(n_train, dtype=np.float32)
for fold in range(5):
    tr_idx, va_idx = splits[fold]["train"], splits[fold]["val"]
    tr_idx = [i for i in tr_idx if i < n_train]
    va_idx = [i for i in va_idx if i < n_train]
    
    model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    model.fit(Xcs[tr_idx], y_tr[tr_idx],
              eval_set=[(Xcs[va_idx], y_tr[va_idx])], verbose=False)
    va_pred = model.predict(Xcs[va_idx])
    oof_preds3[va_idx] = va_pred
    print(f"  Fold {fold}: R2={r2_score(y_tr[va_idx], va_pred):.4f}")

print(f"Overall OOF R2: {r2_score(y_tr[mask], oof_preds3[mask]):.4f}")
