import pickle, numpy as np
from sklearn.metrics import r2_score

# Load GIN OOF predictions
gin_oof = pickle.load(open("outputs/gin/tg/oof_tg_gin.pkl", "rb"))
gin_pred = gin_oof["pred"]
gin_y = gin_oof["y"]
print(f"GIN OOF: R2={r2_score(gin_y, gin_pred):.4f}")

# Load tabular XGBoost OOF (we need to generate this)
# For now, let me test: GIN pred as feature for XGBoost
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

data_dir = Path("data")
tr = pd.read_csv(data_dir / "train.csv")
tmask = tr["target_type"].values == "tg"
y_tr = tr["target"].values[tmask].astype(np.float32)

X_tr = pd.read_parquet(data_dir / "processed/features_train.parquet")
exc = {"id", "canon_smiles", "SMILES"}
fc = [c for c in X_tr.columns if c not in exc]
X_arr = X_tr[fc].values.astype(np.float32)[tmask]

scaler = StandardScaler()
Xs = scaler.fit_transform(X_arr)

with open(data_dir / "splits_tg.pkl", "rb") as f:
    splits = pickle.load(f)

n_train = len(y_tr)
xgb_params = {"n_estimators": 1500, "max_depth": 7, "learning_rate": 0.03,
              "subsample": 0.85, "colsample_bytree": 0.5, "min_child_weight": 3,
              "gamma": 0.1, "reg_alpha": 1.0, "reg_lambda": 2.0,
              "random_state": 42, "n_jobs": -1, "verbosity": 0}

# Tabular only
print("\n=== Tabular only ===")
tab_oom = np.zeros(n_train, dtype=np.float32)
for fold in range(5):
    ti, vi = splits[fold]["train"], splits[fold]["val"]
    ti = [i for i in ti if i < n_train]
    vi = [i for i in vi if i < n_train]
    m = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    m.fit(Xs[ti], y_tr[ti], eval_set=[(Xs[vi], y_tr[vi])], verbose=False)
    tab_oom[vi] = m.predict(Xs[vi])
    print(f"  Fold {fold}: R2={r2_score(y_tr[vi], tab_oom[vi]):.4f}")
print(f"Overall: {r2_score(y_tr, tab_oom):.4f}")

# Tabular + GIN prediction as feature
print("\n=== Tabular + GIN pred feature ===")
X_with_gin = np.concatenate([Xs, gin_pred.reshape(-1, 1)], axis=1)
X_with_gin_s = StandardScaler().fit_transform(X_with_gin)
mix_oom = np.zeros(n_train, dtype=np.float32)
for fold in range(5):
    ti, vi = splits[fold]["train"], splits[fold]["val"]
    ti = [i for i in ti if i < n_train]
    vi = [i for i in vi if i < n_train]
    m = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    m.fit(X_with_gin_s[ti], y_tr[ti], eval_set=[(X_with_gin_s[vi], y_tr[vi])], verbose=False)
    mix_oom[vi] = m.predict(X_with_gin_s[vi])
    print(f"  Fold {fold}: R2={r2_score(y_tr[vi], mix_oom[vi]):.4f}")
print(f"Overall: {r2_score(y_tr, mix_oom):.4f}")

# Blending
print("\n=== Blend (0.5 * tabular + 0.5 * GIN) ===")
blend = 0.5 * tab_oom + 0.5 * gin_pred
print(f"Blend R2: {r2_score(y_tr, blend):.4f}")

# Optimal blend weight
from scipy.optimize import minimize
def negr2(w):
    b = w * tab_oom + (1-w) * gin_pred
    return -r2_score(y_tr, b)
res = minimize(negr2, 0.5, bounds=[(0,1)])
print(f"Optimal w(tabular)={res.x[0]:.4f}, R2={-res.fun:.4f}")
