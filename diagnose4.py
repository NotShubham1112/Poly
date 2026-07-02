import pickle
import numpy as np
import pandas as pd
import glob
import os
from sklearn.metrics import r2_score
from scipy.optimize import minimize

pred_dir = "polymer_competition/predictions"

# Load all OOF data
oof_data = {}
for target in ['tg', 'egc']:
    for model in ['xgb', 'lgb', 'catboost', 'rf', 'mlp', 'gcn', 'gat', 'mpnn']:
        for ver in ['v27', 'v28']:
            oof_preds = []
            oof_ys = []
            for fold in range(5):
                pattern = os.path.join(pred_dir, f"{ver}_{target}_{model}_fold{fold}.pkl")
                if os.path.exists(pattern):
                    with open(pattern, 'rb') as f:
                        d = pickle.load(f)
                    preds = d.get('pred', [])
                    ys = d.get('y', [])
                    if len(preds) == len(ys):
                        oof_preds.extend(preds)
                        oof_ys.extend(ys)
            if oof_preds:
                key = f"{ver}_{target}_{model}"
                oof_data[key] = {'pred': np.array(oof_preds), 'y': np.array(oof_ys)}

# Add polychain per-seed and 5-seed avg
for target in ['tg', 'egc']:
    seed_data = {}
    for seed in [42, 123, 456, 789, 101112]:
        for fold in range(5):
            pattern = os.path.join(pred_dir, f"v28_{target}_polychain_boosted_s{seed}_fold{fold}.pkl")
            if os.path.exists(pattern):
                with open(pattern, 'rb') as f:
                    d = pickle.load(f)
                idx = d.get('val_idx', list(range(len(d['pred']))))
                preds = np.array(d['pred'])
                ys = np.array(d['y'])
                if seed not in seed_data:
                    seed_data[seed] = []
                seed_data[seed].append({'idx': np.array(idx), 'pred': preds, 'y': ys})

    # For 5-seed average, align by val_idx
    if seed_data:
        # Collect all unique val indices
        all_idx = set()
        for seed in seed_data:
            for fold_data in seed_data[seed]:
                all_idx.update(fold_data['idx'].tolist())
        all_idx = sorted(all_idx)
        
        # Average predictions per val_idx across all seeds and folds
        pred_matrix = np.full((len(all_idx), len(seed_data)), np.nan)
        y_vals = np.zeros(len(all_idx))
        idx_map = {idx: i for i, idx in enumerate(all_idx)}
        
        for s, seed in enumerate(sorted(seed_data.keys())):
            seed_preds = np.zeros(len(all_idx))
            seed_counts = np.zeros(len(all_idx))
            for fold_data in seed_data[seed]:
                for j, idx in enumerate(fold_data['idx']):
                    ii = idx_map[idx]
                    seed_preds[ii] += fold_data['pred'][j]
                    seed_counts[ii] += 1
                    y_vals[ii] = fold_data['y'][j]
            mask = seed_counts > 0
            pred_matrix[mask, s] = seed_preds[mask] / seed_counts[mask]
        
        avg_pred = np.nanmean(pred_matrix, axis=1)
        valid = ~np.isnan(avg_pred)
        key = f"v28_{target}_polychain_5seed"
        oof_data[key] = {'pred': avg_pred[valid], 'y': y_vals[valid]}

# Compute individual R²
print("=" * 70)
print("INDIVIDUAL MODEL OOF R²")
print("=" * 70)

tg_models = {}
egc_models = {}
for key in sorted(oof_data.keys()):
    d = oof_data[key]
    r2 = r2_score(d['y'], d['pred'])
    parts = key.split('_', 2)
    ver = parts[0]
    target = parts[1]
    model = parts[2]
    if target == 'tg':
        tg_models[key] = r2
    else:
        egc_models[key] = r2

print(f"\n{'Key':<40} {'R²':>10} {'Mean R²':>10}")
print("-" * 60)

# For each model, find best version per target
best_models = {}
for model in ['xgb', 'lgb', 'catboost', 'rf', 'mlp', 'gcn', 'gat', 'mpnn', 'polychain_5seed']:
    tg_key = f"v28_tg_{model}"
    egc_key = f"v28_egc_{model}"
    tg_r2 = tg_models.get(tg_key)
    egc_r2 = egc_models.get(egc_key)
    
    # Also check v27
    tg_key27 = f"v27_tg_{model}"
    egc_key27 = f"v27_egc_{model}"
    tg_r2_27 = tg_models.get(tg_key27)
    egc_r2_27 = egc_models.get(egc_key27)
    
    # Use best version
    if tg_r2_27 and (not tg_r2 or tg_r2_27 > tg_r2):
        tg_r2 = tg_r2_27
        tg_key = tg_key27
    if egc_r2_27 and (not egc_r2 or egc_r2_27 > egc_r2):
        egc_r2 = egc_r2_27
        egc_key = egc_key27
    
    if tg_r2 or egc_r2:
        tg_str = f"{tg_r2:.6f}" if tg_r2 else "N/A"
        egc_str = f"{egc_r2:.6f}" if egc_r2 else "N/A"
        mean_str = f"{(tg_r2 + egc_r2)/2:.6f}" if (tg_r2 and egc_r2) else "N/A"
        print(f"  {model:<38} {tg_str:>10} {egc_str:>10}  ({mean_str})")
        if tg_r2 and egc_r2:
            best_models[model] = {'tg': tg_r2, 'egc': egc_r2, 'mean': (tg_r2 + egc_r2) / 2,
                                   'tg_key': tg_key, 'egc_key': egc_key}

# Optimize weights
print()
print("=" * 70)
print("OPTIMAL BLEND WEIGHTS (tree models only)")
print("=" * 70)

for target in ['tg', 'egc']:
    models_with_oof = {}
    for key in oof_data:
        parts = key.split('_', 2)
        if parts[1] == target and 'polychain' not in key:
            models_with_oof[key] = oof_data[key]
    
    model_names = sorted(set([k.split('_', 2)[2] for k in models_with_oof]))
    
    # Find which version to use per model (prefer v27)
    best_ver = {}
    for m in model_names:
        v27_key = f"v27_{target}_{m}"
        v28_key = f"v28_{target}_{m}"
        if v27_key in models_with_oof and v28_key in models_with_oof:
            best_ver[m] = 'v27'  # v27 had more models, use those
        elif v27_key in models_with_oof:
            best_ver[m] = 'v27'
        else:
            best_ver[m] = 'v28'
    
    # Build matrix
    active_models = []
    for m in model_names:
        key = f"{best_ver[m]}_{target}_{m}"
        if key in models_with_oof:
            active_models.append((m, key))
    
    if not active_models:
        continue
    
    y = models_with_oof[active_models[0][1]]['y']
    n = len(y)
    pred_matrix = np.zeros((n, len(active_models)))
    for i, (m, key) in enumerate(active_models):
        pred_matrix[:, i] = models_with_oof[key]['pred']
    
    # Uniform blend
    uniform_blend = pred_matrix.mean(axis=1)
    uniform_r2 = r2_score(y, uniform_blend)
    
    # Optimize weights
    def neg_r2(w):
        w = np.abs(w)
        w = w / w.sum()
        blended = pred_matrix @ w
        return -r2_score(y, blended)
    
    x0 = np.ones(len(active_models)) / len(active_models)
    res = minimize(neg_r2, x0, method='Nelder-Mead', options={'maxiter': 10000})
    opt_w = np.abs(res.x)
    opt_w = opt_w / opt_w.sum()
    opt_blend = pred_matrix @ opt_w
    opt_r2 = r2_score(y, opt_blend)
    
    print(f"\n  {target.upper()} (n={n}, models={len(active_models)}):")
    print(f"    Uniform blend R²: {uniform_r2:.6f}")
    print(f"    Optimized blend R²: {opt_r2:.6f}")
    print(f"    Weights:")
    for i, (m, key) in enumerate(active_models):
        if opt_w[i] > 0.01:
            print(f"      {m}: {opt_w[i]:.4f}")

# What would the best approach score?
print()
print("=" * 70)
print("BOTTOM LINE")
print("=" * 70)
print()
print("The champion scored 0.876 because:")
print("  1. PolyChain OOF R² (TG=0.75, EGC=0.70) is much worse than")
print("     tree models (TG=0.865, EGC=0.907)")
print("  2. PolyChain has systematic positive bias on test predictions")
print("  3. 35% weight to a worse model = guaranteed score drop")
print()
print("RECOMMENDATIONS:")
print("  Option A: Submit Arch A alone (= submission.csv, no PolyChain)")
print("  Option B: Optimize tree-only blend weights with run_submission.py")
print("  Option C: Try v29/v30 tree-model blend if those are better")
