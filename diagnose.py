import pandas as pd
import pickle
import numpy as np
import glob
import os

# 1. Check submissions
print("=" * 60)
print("1. SUBMISSION FILES")
print("=" * 60)

sub = pd.read_csv("polymer_competition/submission_champion.csv")
arch = pd.read_csv(r"D:\Parth\Poly\submission (2).csv")
test = pd.read_csv("polymer_competition/data/test.csv")

print(f"submission_champion.csv: {sub.shape}")
print(f"  ID range: [{sub['id'].min()}, {sub['id'].max()}]")
print(f"  Target: min={sub['target'].min():.4f} mean={sub['target'].mean():.4f} max={sub['target'].max():.4f}")
print(f"  NaN: {sub.isna().sum().sum()}")
print()

print(f"submission (2).csv (Arch A): {arch.shape}")
print(f"  ID range: [{arch['id'].min()}, {arch['id'].max()}]")
print(f"  Target: min={arch['target'].min():.4f} mean={arch['target'].mean():.4f} max={arch['target'].max():.4f}")
print(f"  NaN: {arch.isna().sum().sum()}")
print()

print(f"test.csv: {test.shape}")
print(f"  ID range: [{test['id'].min()}, {test['id'].max()}]")
if 'target_type' in test.columns:
    tg_count = (test['target_type'] == 'tg').sum()
    egc_count = (test['target_type'] == 'egc').sum()
    print(f"  TG: {tg_count}, EGC: {egc_count}")
print()

# 2. Check per-target stats in champion
if 'target_type' in test.columns:
    merged = sub.merge(test[['id', 'target_type']], on='id')
    for ttype in ['tg', 'egc']:
        subset = merged[merged['target_type'] == ttype]
        print(f"Champion {ttype}: n={len(subset)}, mean={subset['target'].mean():.4f}, min={subset['target'].min():.4f}, max={subset['target'].max():.4f}")
    
    merged_arch = arch.merge(test[['id', 'target_type']], on='id')
    for ttype in ['tg', 'egc']:
        subset = merged_arch[merged_arch['target_type'] == ttype]
        print(f"Arch A {ttype}: n={len(subset)}, mean={subset['target'].mean():.4f}, min={subset['target'].min():.4f}, max={subset['target'].max():.4f}")

print()

# 3. How different is champion from Arch A?
print("=" * 60)
print("2. BLEND DIVERGENCE CHECK")
print("=" * 60)
diff = (sub['target'].values - arch['target'].values)
print(f"Champion - Arch A diff: mean={diff.mean():.4f}, std={diff.std():.4f}, min={diff.min():.4f}, max={diff.max():.4f}")
print(f"Correlation: {np.corrcoef(sub['target'].values, arch['target'].values)[0,1]:.6f}")
print(f"R2 if we just used Arch A: {1 - np.sum((sub['target'].values - arch['target'].values)**2) / np.sum((arch['target'].values - arch['target'].mean())**2):.6f}")
print()

# 4. PolyChain predictions analysis
print("=" * 60)
print("3. POLYCHAIN PREDICTIONS ANALYSIS")
print("=" * 60)
pred_dir = "polymer_competition/predictions"

tg_files = sorted(glob.glob(os.path.join(pred_dir, "v28_tg_polychain_boosted_s*_fold*_test.pkl")))
print(f"TG test files: {len(tg_files)}")

# Average TG predictions
tg_preds = []
tg_ids = None
for f in tg_files:
    with open(f, "rb") as fh:
        d = pickle.load(fh)
    if tg_ids is None:
        tg_ids = d["id"]
    tg_preds.append(np.array(d["pred"]))

avg_tg = np.mean(tg_preds, axis=0)
print(f"  Avg TG: n={len(avg_tg)}, mean={avg_tg.mean():.4f}, min={avg_tg.min():.4f}, max={avg_tg.max():.4f}")

egc_files = sorted(glob.glob(os.path.join(pred_dir, "v28_egc_polychain_boosted_s*_fold*_test.pkl")))
print(f"EGC test files: {len(egc_files)}")

egc_preds = []
egc_ids = None
for f in egc_files:
    with open(f, "rb") as fh:
        d = pickle.load(fh)
    if egc_ids is None:
        egc_ids = d["id"]
    egc_preds.append(np.array(d["pred"]))

avg_egc = np.mean(egc_preds, axis=0)
print(f"  Avg EGC: n={len(avg_egc)}, mean={avg_egc.mean():.4f}, min={avg_egc.min():.4f}, max={avg_egc.max():.4f}")

# 5. Compare PolyChain vs Arch A per target
print()
print("=" * 60)
print("4. POLYCHAIN vs ARCH A PER TARGET")
print("=" * 60)

# Map TG predictions
is_tg = test['target_type'].values == 'tg'
is_egc = test['target_type'].values == 'egc'
all_ids = test['id'].values

# Arch A per target
arch_tg = arch['target'].values[is_tg]
arch_egc = arch['target'].values[is_egc]
print(f"Arch A TG: mean={arch_tg.mean():.4f}, n={len(arch_tg)}")
print(f"Arch A EGC: mean={arch_egc.mean():.4f}, n={len(arch_egc)}")

# 6. Check training data stats
print()
print("=" * 60)
print("5. TRAINING DATA vs TEST PREDICTIONS")
print("=" * 60)
train = pd.read_csv("polymer_competition/data/train.csv")
print(f"Train shape: {train.shape}")
print(f"Train columns: {list(train.columns)}")
if 'target_type' in train.columns:
    for ttype in ['tg', 'egc']:
        subset = train[train['target_type'] == ttype]
        if 'property' in train.columns:
            print(f"Train {ttype}: n={len(subset)}, property mean={subset['property'].mean():.4f}")
        elif 'target' in train.columns:
            print(f"Train {ttype}: n={len(subset)}, target mean={subset['target'].mean():.4f}")
