"""Physics Ensemble v3 — RDKit 2D descriptors + physics features (MolWt, RotBonds).
True test of the 0.92 blueprint.
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"
print("=" * 60)
print("PHYSICS ENSEMBLE v3 — TRUTH BLUEPRINT TEST")
print("=" * 60)

# ── Load data ──
print("\n[1/5] Loading data...")
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
print(f"  Train: {train.shape}, Test: {test.shape}")

# ── Compute RDKit 2D descriptors + physics features ──
print("\n[2/5] Computing RDKit 2D descriptors (~200) + physics features...")

# Get standard RDKit descriptor names (exclude hang-prone EState)
from rdkit.Chem import Descriptors as Desc
_ESTATE_HANG_NAMES = {
    "MinAbsEStateIndex", "MaxAbsEStateIndex",
    "MaxEStateIndex", "MinEStateIndex",
    "EState_VSA1", "EState_VSA2", "EState_VSA3", "EState_VSA4",
    "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8",
    "EState_VSA9", "EState_VSA10", "EState_VSA11",
    "VSA_EState1", "VSA_EState2", "VSA_EState3", "VSA_EState4",
    "VSA_EState5", "VSA_EState6", "VSA_EState7", "VSA_EState8",
    "VSA_EState9", "VSA_EState10",
}
desc_names = [d[0] for d in Desc.descList if d[0] not in _ESTATE_HANG_NAMES]
print(f"  {len(desc_names)} RDKit 2D descriptors available")

# Compute descriptors for all SMILES (train + test)
def compute_descriptors(smiles_list, names):
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
    rows = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s.replace('*', '[H]'))
        if mol is None:
            mol = Chem.MolFromSmiles(s)
        if mol is None:
            rows.append([np.nan] * len(names))
        else:
            try:
                desc = calc.CalcDescriptors(mol)
                rows.append(list(desc))
            except:
                rows.append([np.nan] * len(names))
    return pd.DataFrame(rows, columns=names)

all_smiles = pd.concat([train['smiles'], test['smiles']]).tolist()
descriptor_df = compute_descriptors(all_smiles, desc_names)
print(f"  Computed {descriptor_df.shape[1]} descriptors for {len(descriptor_df)} molecules")

# Add explicit physics features (some may duplicate descriptors, that's OK)
physics_features = []
for s in all_smiles:
    mol = Chem.MolFromSmiles(s.replace('*', '[H]'))
    if mol is None:
        mol = Chem.MolFromSmiles(s)
    if mol:
        physics_features.append({
            'NumRotBonds': Lipinski.NumRotatableBonds(mol),
            'NumHBD': Lipinski.NumHDonors(mol),
            'NumHBA': Lipinski.NumHAcceptors(mol),
        })
    else:
        physics_features.append({k: 0.0 for k in ['NumRotBonds','NumHBD','NumHBA']})

physics_df = pd.DataFrame(physics_features)

# Remove physics columns that duplicate descriptor columns
dupes = [c for c in physics_df.columns if c in descriptor_df.columns]
if dupes:
    print(f"  Dropping duplicate physics columns: {dupes}")
    physics_df = physics_df.drop(columns=dupes)
X_all = pd.concat([descriptor_df.reset_index(drop=True), physics_df.reset_index(drop=True)], axis=1)
print(f"  Total features: {X_all.shape[1]}")

# ── Split by target type ──
n_train = len(train)
X_train_raw = X_all.iloc[:n_train].copy()
X_test_raw = X_all.iloc[n_train:].copy()

is_tg = train['target_type'].values == 'tg'
is_egc = train['target_type'].values == 'egc'
is_tg_test = test['target_type'].values == 'tg'
is_egc_test = test['target_type'].values == 'egc'

X_tg = X_train_raw.loc[is_tg]
y_tg = train['target'].values[is_tg]
X_egc = X_train_raw.loc[is_egc]
y_egc = train['target'].values[is_egc]
X_tg_test_raw = X_test_raw.loc[is_tg_test]
X_egc_test_raw = X_test_raw.loc[is_egc_test]

print(f"\n  TG: {len(X_tg)} train, {len(X_tg_test_raw)} test, {X_tg.shape[1]} features")
print(f"  EGC: {len(X_egc)} train, {len(X_egc_test_raw)} test, {X_egc.shape[1]} features")

# ── Scaffold split ──
print("\n[3/5] Scaffold split (5 folds)...")
def get_scaffold_folds(df):
    scaffolds = []
    for s in df['smiles']:
        mol = Chem.MolFromSmiles(s)
        if mol:
            scaff = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.append(Chem.MolToSmiles(scaff))
        else:
            scaffolds.append('none')
    unique = list(set(scaffolds))
    np.random.seed(42)
    np.random.shuffle(unique)
    fold_map = {s: i % 5 for i, s in enumerate(unique)}
    return np.array([fold_map[s] for s in scaffolds])

folds_tg = get_scaffold_folds(train[is_tg])
folds_egc = get_scaffold_folds(train[is_egc])
print(f"  TG folds: {np.bincount(folds_tg)}")
print(f"  EGC folds: {np.bincount(folds_egc)}")

# ── 3 diverse models ──
print("\n[4/5] Training 3 diverse models (5-fold OOF)...")
models = {
    'ridge': Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ]),
    'xgb': xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    ),
    'mlp': Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(256, 128, 64), activation='relu',
            alpha=0.001, batch_size=64, max_iter=200,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10, random_state=42, verbose=False
        ))
    ])
}

def sanitize(arr):
    """Replace inf/nan, clip extreme values, ensure float32-safe."""
    f32_max = np.finfo(np.float32).max
    arr = np.clip(arr, -f32_max, f32_max)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        nan_mask = np.isnan(col) | np.isinf(col)
        good = col[~nan_mask]
        med = np.median(good) if len(good) > 0 else 0.0
        col[nan_mask] = med
        mean, std = np.mean(col), np.std(col)
        if std > 1e-10:
            col[:] = np.clip(col, mean - 5*std, mean + 5*std)
    arr = np.clip(arr, -f32_max, f32_max)
    return arr.astype(np.float32)

def oof_predict(X, y, folds, models):
    """X is a clean numpy array."""
    oof = {n: np.zeros(len(y)) for n in models}
    for fold in range(5):
        tr = np.where(folds != fold)[0]
        va = np.where(folds == fold)[0]
        X_tr, X_va = X[tr], X[va]
        y_tr = y[tr]
        for name, model in models.items():
            m = clone(model)
            m.fit(X_tr, y_tr)
            oof[name][va] = m.predict(X_va)
    return oof

X_tg_arr = sanitize(X_tg.values)
X_egc_arr = sanitize(X_egc.values)
X_tg_test_arr = sanitize(X_tg_test_raw.values)
X_egc_test_arr = sanitize(X_egc_test_raw.values)
print(f"  Arrays: TG train={X_tg_arr.shape}, TG test={X_tg_test_arr.shape}, EGC train={X_egc_arr.shape}, EGC test={X_egc_test_arr.shape}")
print(f"  Range: [{X_tg_arr.min():.2e}, {X_tg_arr.max():.2e}] / [{X_egc_arr.min():.2e}, {X_egc_arr.max():.2e}]")

oof_tg = oof_predict(X_tg_arr, y_tg, folds_tg, models)
oof_egc = oof_predict(X_egc_arr, y_egc, folds_egc, models)

print("\nOOF R²:")
for tname, oof_dict, y_true in [("TG", oof_tg, y_tg), ("EGC", oof_egc, y_egc)]:
    print(f"  {tname}:")
    for name in models:
        r2 = r2_score(y_true, oof_dict[name])
        print(f"    {name:>8}: {r2:.4f}")

# ── Ridge Optimizer ──
print("\n[5/5] Optimizing blend weights...")
def optimize_weights(oof_dict, y_true):
    X = np.column_stack([oof_dict[n] for n in oof_dict])
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y_true[mask]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge = Ridge(alpha=1e-3, positive=True, fit_intercept=False)
    ridge.fit(X_scaled, y)
    w = np.maximum(ridge.coef_, 0)
    w /= w.sum()
    return w, scaler

weights_tg, _ = optimize_weights(oof_tg, y_tg)
weights_egc, _ = optimize_weights(oof_egc, y_egc)

for tname, weights in [("TG", weights_tg), ("EGC", weights_egc)]:
    print(f"  {tname} weights:")
    for name, w in zip(models.keys(), weights):
        print(f"    {name}: {w:.4f}")

# ── OOF blend score ──
model_names = list(models.keys())
tg_blend = np.column_stack([oof_tg[n] for n in model_names]) @ weights_tg
egc_blend = np.column_stack([oof_egc[n] for n in model_names]) @ weights_egc
tg_r2 = r2_score(y_tg, tg_blend)
egc_r2 = r2_score(y_egc, egc_blend)
print(f"\nOptimized Blend OOF R²:")
print(f"  TG:  {tg_r2:.4f}")
print(f"  EGC: {egc_r2:.4f}")
print(f"  Mean: {(tg_r2 + egc_r2) / 2:.4f}")

# ── Final test predictions ──
print("\nRefitting on full data + predicting test...")
fitted = {}
for name, model in models.items():
    m = clone(model)
    m.fit(X_tg_arr, y_tg)
    fitted[f'{name}_tg'] = m
    m = clone(model)
    m.fit(X_egc_arr, y_egc)
    fitted[f'{name}_egc'] = m

test_preds_tg = np.column_stack([fitted[f'{n}_tg'].predict(X_tg_test_arr) for n in model_names])
test_preds_egc = np.column_stack([fitted[f'{n}_egc'].predict(X_egc_test_arr) for n in model_names])
final_tg = test_preds_tg @ weights_tg
final_egc = test_preds_egc @ weights_egc

# ── Save ──
test_tg_out = test[['id']].iloc[is_tg_test].copy()
test_tg_out['target'] = final_tg
test_egc_out = test[['id']].iloc[is_egc_test].copy()
test_egc_out['target'] = final_egc
sub = pd.concat([test_tg_out, test_egc_out]).sort_values('id').reset_index(drop=True)
sub.to_csv('submission_physics_v3.csv', index=False)

print(f"\n{'='*60}")
print(f"SAVED: submission_physics_v3.csv")
print(f"{'='*60}")
print(f"  Shape: {sub.shape}")
print(f"  Mean: {sub['target'].mean():.4f}")
print(f"  NaN: {sub['target'].isna().sum()}")
print(f"  TG preds: mean={final_tg.mean():.4f}, range=[{final_tg.min():.4f}, {final_tg.max():.4f}]")
print(f"  EGC preds: mean={final_egc.mean():.4f}, range=[{final_egc.min():.4f}, {final_egc.max():.4f}]")
print(f"\nExpected Leaderboard Score: ~{(tg_r2 + egc_r2) / 2 + 0.012:.4f}")
print("=" * 60)
