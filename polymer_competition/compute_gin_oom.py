import pickle, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
import torch
from torch_geometric.data import Batch as PyGBatch
from torch.utils.data import Dataset, DataLoader
from features.graphs import smiles_to_graph
from models.gnn import GINRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

data_dir = Path("data")
tr = pd.read_csv(data_dir / "train.csv")
tmask = tr["target_type"].values == "tg"
y_tr = tr["target"].values[tmask].astype(np.float32)
tr_smiles = tr["smiles"].values[tmask]

graphs = []
for s in tr_smiles:
    g = smiles_to_graph(s)
    if g is not None:
        graphs.append(g)

in_dim = graphs[0].x.size(1)
edge_dim = graphs[0].edge_attr.size(1)

with open(data_dir / "splits_tg.pkl", "rb") as f:
    splits = pickle.load(f)

n_train = len(graphs)
all_gin_preds = np.zeros(n_train, dtype=np.float32)

class GraphDS(Dataset):
    def __init__(self, gs, ys=None):
        self.gs = gs
        self.ys = ys
    def __len__(self):
        return len(self.gs)
    def __getitem__(self, i):
        if self.ys is not None:
            return self.gs[i], self.ys[i]
        return self.gs[i]

def collate(batch):
    if isinstance(batch[0], tuple):
        gs = [b[0] for b in batch]
        ys = torch.tensor([b[1] for b in batch], dtype=torch.float)
        return PyGBatch.from_data_list(gs), ys
    return PyGBatch.from_data_list(batch)

for fold in range(5):
    ckpt = torch.load(f"outputs/gin/tg/checkpoints/gin_gin_fold{fold}_best.pt",
                      map_location="cpu", weights_only=False)
    ms = ckpt["model_state"]
    ch = ms["encoder.atom_encoder.weight"].shape[0]
    ce = ms["encoder.output_proj.weight"].shape[0]
    
    model = GINRegressor(in_dim=in_dim, edge_dim=edge_dim,
                         hidden_dim=ch, embed_dim=ce,
                         n_layers=3, dropout=0.0)
    model.load_state_dict(ms)
    model.eval()
    
    _, va_idx = splits[fold]["train"], splits[fold]["val"]
    va_idx = [i for i in va_idx if i < n_train]
    
    va_ds = GraphDS([graphs[i] for i in va_idx])
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False,
                          num_workers=0, collate_fn=collate)
    
    preds = []
    with torch.no_grad():
        for batch in va_loader:
            p = model(batch)
            preds.append(p.numpy())
    va_preds = np.concatenate(preds)
    all_gin_preds[va_idx] = va_preds
    
    fold_r2 = r2_score(y_tr[va_idx], va_preds)
    print(f"Fold {fold}: R2={fold_r2:.4f}")

gin_oom_r2 = r2_score(y_tr, all_gin_preds)
print(f"\nGIN OOF R2: {gin_oom_r2:.4f}")

# Now compare with tabular XGBoost
X_tr = pd.read_parquet(data_dir / "processed/features_train.parquet")
exc = {"id", "canon_smiles", "SMILES"}
fc = [c for c in X_tr.columns if c not in exc]
X_arr = X_tr[fc].values.astype(np.float32)[tmask]
scaler = StandardScaler()
Xs = scaler.fit_transform(X_arr)

xgbp = {"n_estimators": 1500, "max_depth": 7, "learning_rate": 0.03,
        "subsample": 0.85, "colsample_bytree": 0.5, "min_child_weight": 3,
        "gamma": 0.1, "reg_alpha": 1.0, "reg_lambda": 2.0,
        "random_state": 42, "n_jobs": -1, "verbosity": 0}
all_xgb_preds = np.zeros(n_train, dtype=np.float32)
for fold in range(5):
    ti, vi = splits[fold]["train"], splits[fold]["val"]
    ti = [i for i in ti if i < n_train]
    vi = [i for i in vi if i < n_train]
    m = xgb.XGBRegressor(**xgbp, early_stopping_rounds=50)
    m.fit(Xs[ti], y_tr[ti], eval_set=[(Xs[vi], y_tr[vi])], verbose=False)
    all_xgb_preds[vi] = m.predict(Xs[vi])
    print(f"XGB Fold {fold}: R2={r2_score(y_tr[vi], all_xgb_preds[vi]):.4f}")
print(f"XGB OOF R2: {r2_score(y_tr, all_xgb_preds):.4f}")

# Correlations
from scipy.stats import pearsonr
c, _ = pearsonr(all_gin_preds, all_xgb_preds)
print(f"\nPearson r between GIN and XGB: {c:.4f}")
print(f"R2 of GIN: {r2_score(y_tr, all_gin_preds):.4f}")
print(f"R2 of XGB: {r2_score(y_tr, all_xgb_preds):.4f}")

# Simple blend
for w in [0.3, 0.5, 0.7]:
    bl = w * all_xgb_preds + (1-w) * all_gin_preds
    br2 = r2_score(y_tr, bl)
    print(f"Blend w(xgb)={w}: R2={br2:.4f}")

# Optimal blend
from scipy.optimize import minimize_scalar
def nr2(w):
    return -r2_score(y_tr, w * all_xgb_preds + (1-w) * all_gin_preds)
res = minimize_scalar(nr2, bounds=(0, 1), method="bounded")
print(f"Optimal w(xgb)={res.x:.4f}, R2={-res.fun:.4f}")

# Stack: GIN pred as feature
X_plus = np.concatenate([Xs, all_gin_preds.reshape(-1, 1)], axis=1)
X_plus_s = StandardScaler().fit_transform(X_plus)
all_stack_preds = np.zeros(n_train, dtype=np.float32)
for fold in range(5):
    ti, vi = splits[fold]["train"], splits[fold]["val"]
    ti = [i for i in ti if i < n_train]
    vi = [i for i in vi if i < n_train]
    m = xgb.XGBRegressor(**xgbp, early_stopping_rounds=50)
    m.fit(X_plus_s[ti], y_tr[ti], eval_set=[(X_plus_s[vi], y_tr[vi])], verbose=False)
    all_stack_preds[vi] = m.predict(X_plus_s[vi])
print(f"Stack (tabular+GINpred) OOF R2: {r2_score(y_tr, all_stack_preds):.4f}")
