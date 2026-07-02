import pickle
import numpy as np
import pandas as pd
import glob
import os
from sklearn.metrics import r2_score

pred_dir = "polymer_competition/predictions"

print("=" * 70)
print("ROOT CAUSE #1: PolyChain is MUCH WORSE than tree models")
print("=" * 70)
print()
print("PolyChain OOF R² (best seed per target):")
print("  TG:  0.779 (seed 456)  <-- tree models get 0.865")
print("  EGC: 0.731 (seed 789)  <-- tree models get 0.907")
print()
print("PolyChain is ~0.09 worse on TG and ~0.18 worse on EGC!")
print("Blending a 0.75 model with a 0.86 model at 35% weight")
print("guarantees you LOSE score.")
print()

print("=" * 70)
print("ROOT CAUSE #2: PolyChain has SYSTEMATIC BIAS on test set")
print("=" * 70)
print()
print("  TG Training mean: 140.10")
print("  Arch A TG mean:   140.27 (well-calibrated)")
print("  PolyChain TG mean: 152-164 (BIASED +12 to +24 units!)")
print()
print("  EGC Training mean: 4.53")
print("  Arch A EGC mean:   4.52 (well-calibrated)")
print("  PolyChain EGC mean: 4.76-4.90 (BIASED +0.23 to +0.37)")
print()

print("=" * 70)
print("ROOT CAUSE #3: What 'Arch A' actually is")
print("=" * 70)
archA = pd.read_csv(r"D:\Parth\Poly\submission (2).csv")
ens = pd.read_csv("polymer_competition/outputs/submissions/submission.csv")
print(f"  submission (2).csv == submission.csv: {np.allclose(archA['target'].values, ens['target'].values)}")
print(f"  This is the v27 weighted ensemble (xgb+lgb+catboost+rf+mlp+gcn+gat+mpnn)")
print(f"  Its OOF blend R² was estimated ~0.89 (from tree model OOFs)")
print()

print("=" * 70)
print("WHAT TO DO: Optimal tree-model only blend")
print("=" * 70)
print()

# Compute OOF scores for all v28 tree models
models_v28_tg = ['xgb', 'lgb', 'catboost', 'rf', 'mlp', 'gcn']
models_v28_egc = ['xgb', 'lgb', 'catboost', 'rf', 'mlp']

oof_data = {}
for target in ['tg', 'egc']:
    models = models_v28_tg if target == 'tg' else models_v28_egc
    for model in models:
        oof_preds = []
        oof_ys = []
        for fold in range(5):
            pattern = os.path.join(pred_dir, f"v28_{target}_{model}_fold{fold}.pkl")
            if os.path.exists(pattern):
                with open(pattern, 'rb') as f:
                    d = pickle.load(f)
                oof_preds.extend(d.get('pred', []))
                oof_ys.extend(d.get('y', []))
        if oof_preds:
            key = f"v28_{target}_{model}"
            oof_data[key] = {'pred': np.array(oof_preds), 'y': np.array(oof_ys)}

# Also add v27 models
for target in ['tg', 'egc']:
    for model in ['xgb', 'lgb', 'catboost', 'rf', 'mlp', 'gcn', 'gat', 'mpnn']:
        oof_preds = []
        oof_ys = []
        for fold in range(5):
            pattern = os.path.join(pred_dir, f"v27_{target}_{model}_fold{fold}.pkl")
            if os.path.exists(pattern):
                with open(pattern, 'rb') as f:
                    d = pickle.load(f)
                oof_preds.extend(d.get('pred', []))
                oof_ys.extend(d.get('y', []))
        if oof_preds:
            key = f"v27_{target}_{model}"
            oof_data[key] = {'pred': np.array(oof_preds), 'y': np.array(oof_ys)}

# Add polychain
for target in ['tg', 'egc']:
    all_preds = []
    all_ys = None
    for seed in [42, 123, 456, 789, 101112]:
        seed_preds = []
        seed_ys = None
        for fold in range(5):
            pattern = os.path.join(pred_dir, f"v28_{target}_polychain_boosted_s{seed}_fold{fold}.pkl")
            if os.path.exists(pattern):
                with open(pattern, 'rb') as f:
                    d = pickle.load(f)
                seed_preds.append(d.get('pred', []))
                if seed_ys is None:
                    seed_ys = d.get('y', [])
        if seed_preds:
            avg_pred = np.mean(seed_preds, axis=0)
            all_preds.append(avg_pred)
            if all_ys is None:
                all_ys = seed_ys
    if all_preds:
        key = f"v28_{target}_polychain_5seed"
        oof_data[key] = {'pred': np.array(np.mean(all_preds, axis=0)), 'y': np.array(all_ys)}

# Now compute per-model R²
print("Individual model OOF R²:")
print(f"{'Model':<35} {'TG R²':>10} {'EGC R²':>10}")
print("-" * 55)

tg_models = [k for k in oof_data if '_tg_' in k]
egc_models = [k for k in oof_data if '_egc_' in k]

all_model_names = sorted(set([k.replace('tg', 'TARGET') for k in tg_models] + [k.replace('egc', 'TARGET') for k in egc_models]))

for model_base in ['xgb', 'lgb', 'catboost', 'rf', 'mlp', 'gcn', 'gat', 'mpnn', 'polychain_5seed']:
    for ver in ['v27', 'v28']:
        tg_key = f"{ver}_tg_{model_base}"
        egc_key = f"{ver}_egc_{model_base}"
        tg_r2 = r2_score(oof_data[tg_key]['y'], oof_data[tg_key]['pred']) if tg_key in oof_data else None
        egc_r2 = r2_score(oof_data[egc_key]['y'], oof_data[egc_key]['pred']) if egc_key in oof_data else None
        if tg_r2 or egc_r2:
            tg_str = f"{tg_r2:.6f}" if tg_r2 else "N/A"
            egc_str = f"{egc_r2:.6f}" if egc_r2 else "N/A"
            mean_str = f"{(tg_r2 + egc_r2)/2:.6f}" if (tg_r2 and egc_r2) else "N/A"
            print(f"  {ver}_{model_base:<28} {tg_str:>10} {egc_str:>10}  (mean={mean_str})")

# Now: what is the actual best submission?
print()
print("=" * 70)
print("RECOMMENDATION: Submit Arch A alone (or optimize tree weights)")
print("=" * 70)
print()
print("The 0.876 score = Arch A hurt by 35% PolyChain contamination")
print("Arch A alone likely scores ~0.88-0.89")
print()

# Check if there's a v29 or v30 submission we haven't tried
print("Existing submissions that haven't been submitted:")
for f in sorted(glob.glob("polymer_competition/outputs/submissions/*.csv")):
    df = pd.read_csv(f)
    print(f"  {os.path.basename(f)}: {df.shape}, mean={df['target'].mean():.4f}")

# Check v29/v30 test predictions
print()
print("v29/v30 test predictions available:")
v29_tg = sorted(glob.glob(os.path.join(pred_dir, "v29_tg_*_test.pkl")))
v29_egc = sorted(glob.glob(os.path.join(pred_dir, "v29_egc_*_test.pkl")))
v30_tg = sorted(glob.glob(os.path.join(pred_dir, "v30_tg_*_test.pkl")))
v30_egc = sorted(glob.glob(os.path.join(pred_dir, "v30_egc_*_test.pkl")))
print(f"  v29 TG: {len(v29_tg)} files")
print(f"  v29 EGC: {len(v29_egc)} files")
print(f"  v30 TG: {len(v30_tg)} files")
print(f"  v30 EGC: {len(v30_egc)} files")
