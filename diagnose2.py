import pickle
import numpy as np
import pandas as pd
import glob
import os
from sklearn.metrics import r2_score

pred_dir = "polymer_competition/predictions"

print("=" * 70)
print("ROOT CAUSE ANALYSIS: Why 0.876 instead of ~0.912?")
print("=" * 70)

# 1. Compute OOF R² for each PolyChain seed
print("\n--- A. POLYCHAIN OOF R² per seed ---")
seeds = [42, 123, 456, 789, 101112]
for target in ['tg', 'egc']:
    print(f"\n  Target: {target.upper()}")
    for seed in seeds:
        oof_preds = []
        oof_ys = []
        for fold in range(5):
            pattern = os.path.join(pred_dir, f"v28_{target}_polychain_boosted_s{seed}_fold{fold}.pkl")
            if os.path.exists(pattern):
                with open(pattern, 'rb') as f:
                    d = pickle.load(f)
                oof_preds.extend(d['pred'])
                oof_ys.extend(d.get('y_true', d.get('y', [])))
        if oof_preds:
            r2 = r2_score(oof_ys, oof_preds)
            print(f"    Seed {seed}: OOF R² = {r2:.6f} (n={len(oof_preds)})")
        else:
            print(f"    Seed {seed}: NO OOF DATA")

# 2. Compute OOF R² for the v27 ensemble models
print("\n--- B. V27 ENSEMBLE MODELS OOF R² ---")
models = ['xgb', 'lgb', 'catboost', 'rf', 'mlp', 'gcn', 'gat', 'mpnn']
for target in ['tg', 'egc']:
    print(f"\n  Target: {target.upper()}")
    for model in models:
        oof_preds = []
        oof_ys = []
        for fold in range(5):
            pattern = os.path.join(pred_dir, f"v27_{target}_{model}_fold{fold}.pkl")
            if os.path.exists(pattern):
                with open(pattern, 'rb') as f:
                    d = pickle.load(f)
                oof_preds.extend(d['pred'])
                oof_ys.extend(d.get('y_true', d.get('y', [])))
        if oof_preds:
            r2 = r2_score(oof_ys, oof_preds)
            print(f"    {model}: OOF R² = {r2:.6f} (n={len(oof_preds)})")

# 3. Compute OOF R² for the v28 non-polychain models
print("\n--- C. V28 MODELS OOF R² ---")
models_v28 = ['xgb', 'lgb', 'catboost', 'rf', 'mlp', 'gcn', 'multitask']
for target in ['tg', 'egc']:
    print(f"\n  Target: {target.upper()}")
    for model in models_v28:
        oof_preds = []
        oof_ys = []
        for fold in range(5):
            pattern = os.path.join(pred_dir, f"v28_{target}_{model}_fold{fold}.pkl")
            if os.path.exists(pattern):
                with open(pattern, 'rb') as f:
                    d = pickle.load(f)
                oof_preds.extend(d['pred'])
                oof_ys.extend(d.get('y_true', d.get('y', [])))
        if oof_preds:
            r2 = r2_score(oof_ys, oof_preds)
            print(f"    {model}: OOF R² = {r2:.6f} (n={len(oof_preds)})")

# 4. Best single PolyChain seed test predictions vs Arch A
print("\n--- D. POLYCHAIN TEST PREDICTIONS ANALYSIS ---")
test = pd.read_csv("polymer_competition/data/test.csv")
is_tg = test['target_type'].values == 'tg'
is_egc = test['target_type'].values == 'egc'
all_ids = test['id'].values

archA = pd.read_csv(r"D:\Parth\Poly\submission (2).csv")
arch_preds = archA['target'].values

# Per-seed PolyChain test prediction stats
for seed in seeds:
    tg_files = sorted(glob.glob(os.path.join(pred_dir, f"v28_tg_polychain_boosted_s{seed}_fold*_test.pkl")))
    if tg_files:
        tg_preds = []
        for f in tg_files:
            with open(f, 'rb') as fh:
                d = pickle.load(fh)
            tg_preds.append(np.array(d['pred']))
        avg_tg = np.mean(tg_preds, axis=0)
        
        # Compare to Arch A TG
        arch_tg = arch_preds[is_tg]
        tg_corr = np.corrcoef(avg_tg, arch_tg)[0, 1]
        tg_rmse = np.sqrt(np.mean((avg_tg - arch_tg) ** 2))
        print(f"  Seed {seed} TG: mean={avg_tg.mean():.2f} (Arch A: {arch_tg.mean():.2f}), corr={tg_corr:.4f}, RMSE_vs_ArchA={tg_rmse:.2f}")

    egc_files = sorted(glob.glob(os.path.join(pred_dir, f"v28_egc_polychain_boosted_s{seed}_fold*_test.pkl")))
    if egc_files:
        egc_preds = []
        for f in egc_files:
            with open(f, 'rb') as fh:
                d = pickle.load(fh)
            egc_preds.append(np.array(d['pred']))
        avg_egc = np.mean(egc_preds, axis=0)
        
        arch_egc = arch_preds[is_egc]
        egc_corr = np.corrcoef(avg_egc, arch_egc)[0, 1]
        egc_rmse = np.sqrt(np.mean((avg_egc - arch_egc) ** 2))
        print(f"  Seed {seed} EGC: mean={avg_egc.mean():.4f} (Arch A: {arch_egc.mean():.4f}), corr={egc_corr:.4f}, RMSE_vs_ArchA={egc_rmse:.4f}")

# 5. What did the previous submissions score?
print("\n--- E. PREVIOUS SUBMISSION FILE SIZES ---")
for f in sorted(glob.glob("polymer_competition/outputs/submissions/*.csv")):
    df = pd.read_csv(f)
    print(f"  {os.path.basename(f)}: {df.shape}, mean={df['target'].mean():.4f}")

# 6. Check if Arch A is actually the full ensemble
print("\n--- F. ARCH A vs ENSEMBLE SUBMISSION COMPARISON ---")
try:
    ens_sub = pd.read_csv("polymer_competition/outputs/submissions/submission.csv")
    print(f"  ensemble submission.csv: mean={ens_sub['target'].mean():.4f}, shape={ens_sub.shape}")
    print(f"  Arch A submission (2).csv: mean={archA['target'].mean():.4f}, shape={archA.shape}")
    diff = archA['target'].values - ens_sub['target'].values
    print(f"  Diff: mean={diff.mean():.4f}, std={diff.std():.4f}, max_abs={np.abs(diff).max():.4f}")
except:
    print("  Could not load ensemble submission.csv")
