import pickle
import numpy as np
import pandas as pd
import glob
import os
from sklearn.metrics import r2_score
from scipy.optimize import minimize

pred_dir = "polymer_competition/predictions"
test = pd.read_csv("polymer_competition/data/test.csv")

# Build a proper tree-model-only submission using run_submission.py approach
# But first check what the existing best submission (submission.csv) actually contains

# Check if Arch A was actually optimal
# Compute v28 tree-only blend on test set

# Load v28 test predictions for tree models
models = ['xgb', 'lgb', 'catboost', 'rf', 'mlp', 'gcn']
test_preds = {}
for target in ['tg', 'egc']:
    test_preds[target] = {}
    for model in models:
        key = f"v28_{target}_{model}"
        per_fold = []
        for fold in range(5):
            path = os.path.join(pred_dir, f"v28_{target}_{model}_fold{fold}_test.pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    d = pickle.load(f)
                per_fold.append(np.array(d['pred']))
        if per_fold:
            test_preds[target][model] = np.mean(per_fold, axis=0)

# Also load v27 test preds
for target in ['tg', 'egc']:
    for model in ['xgb', 'lgb', 'catboost', 'rf', 'mlp', 'gcn', 'gat', 'mpnn']:
        key = f"v27_{target}_{model}"
        per_fold = []
        for fold in range(5):
            path = os.path.join(pred_dir, f"v27_{target}_{model}_fold{fold}_test.pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    d = pickle.load(f)
                per_fold.append(np.array(d['pred']))
        if per_fold:
            test_preds[target][model] = np.mean(per_fold, axis=0)

print("Available test predictions:")
for target in ['tg', 'egc']:
    print(f"\n  {target.upper()}:")
    for model, preds in test_preds[target].items():
        print(f"    {model}: n={len(preds)}, mean={preds.mean():.4f}")

# Use OOF-optimized weights to build test predictions
# TG weights: lgb=0.4074, mlp=0.2443, catboost=0.2223, xgb=0.0702, gcn=0.0559
# EGC weights: xgb=0.3176, lgb=0.2515, mlp=0.1989, catboost=0.1981, rf=0.0316

# But these were trained on different OOF sizes. Let me use the BEST available
# version per model.

# Actually, let me build the optimal test submission using the v27 ensemble weights
# from run_submission.py

# Load OOF for proper optimization
def load_oof(target, model, version):
    path = os.path.join(pred_dir, f"{version}_{target}_{model}_fold0.pkl")
    if not os.path.exists(path):
        return None, None
    all_preds = []
    all_ys = []
    for fold in range(5):
        path = os.path.join(pred_dir, f"{version}_{target}_{model}_fold{fold}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                d = pickle.load(f)
            all_preds.extend(d.get('pred', []))
            all_ys.extend(d.get('y', []))
    return np.array(all_preds), np.array(all_ys)

# For each model, get the best OOF R²
best_info = {}
for target in ['tg', 'egc']:
    for model in ['xgb', 'lgb', 'catboost', 'rf', 'mlp', 'gcn', 'gat', 'mpnn']:
        for ver in ['v28', 'v27']:
            preds, ys = load_oof(target, model, ver)
            if preds is not None and len(preds) > 0:
                r2 = r2_score(ys, preds)
                key = f"{ver}_{target}_{model}"
                if target not in best_info:
                    best_info[target] = {}
                if model not in best_info[target] or r2 > best_info[target][model]['r2']:
                    best_info[target][model] = {'r2': r2, 'ver': ver, 'key': key}

print("\n\nBest version per model per target:")
for target in ['tg', 'egc']:
    print(f"\n  {target.upper()}:")
    for model in ['xgb', 'lgb', 'catboost', 'rf', 'mlp', 'gcn', 'gat', 'mpnn']:
        if model in best_info[target]:
            info = best_info[target][model]
            print(f"    {model}: {info['ver']} R²={info['r2']:.6f}")

# Build the optimal tree-only test submission
print("\n\nBuilding optimal tree-only submission...")
final_preds = np.zeros(len(test))

for target in ['tg', 'egc']:
    is_target = test['target_type'].values == target
    n_target = is_target.sum()
    
    # Get test predictions from the best version of each model
    model_preds = {}
    for model in ['xgb', 'lgb', 'catboost', 'rf', 'mlp', 'gcn', 'gat', 'mpnn']:
        if model in best_info[target]:
            ver = best_info[target][model]['ver']
            path = os.path.join(pred_dir, f"{ver}_{target}_{model}_fold0_test.pkl")
            if os.path.exists(path):
                per_fold = []
                for fold in range(5):
                    p = os.path.join(pred_dir, f"{ver}_{target}_{model}_fold{fold}_test.pkl")
                    if os.path.exists(p):
                        with open(p, 'rb') as f:
                            d = pickle.load(f)
                        per_fold.append(np.array(d['pred']))
                if per_fold:
                    model_preds[model] = np.mean(per_fold, axis=0)
    
    print(f"  {target.upper()}: {len(model_preds)} models with test preds")
    
    # Use OOF-based optimized weights
    # Build OOF matrix for optimization
    oof_matrix = {}
    oof_y = None
    for model in model_preds:
        preds, ys = load_oof(target, model, best_info[target][model]['ver'])
        if preds is not None:
            oof_matrix[model] = preds
            if oof_y is None:
                oof_y = ys
    
    if oof_matrix and oof_y is not None:
        model_list = list(oof_matrix.keys())
        oof_mat = np.column_stack([oof_matrix[m] for m in model_list])
        
        def neg_r2(w):
            w = np.abs(w)
            w = w / w.sum()
            return -r2_score(oof_y, oof_mat @ w)
        
        x0 = np.ones(len(model_list)) / len(model_list)
        res = minimize(neg_r2, x0, method='Nelder-Mead', options={'maxiter': 10000})
        opt_w = np.abs(res.x)
        opt_w = opt_w / opt_w.sum()
        
        # Also compute uniform for comparison
        uniform_w = np.ones(len(model_list)) / len(model_list)
        uniform_r2 = r2_score(oof_y, oof_mat @ uniform_w)
        opt_r2 = r2_score(oof_y, oof_mat @ opt_w)
        
        print(f"    Uniform OOF R²: {uniform_r2:.6f}")
        print(f"    Optimized OOF R²: {opt_r2:.6f}")
        
        # Apply to test
        test_mat = np.column_stack([model_preds[m] for m in model_list])
        blended = test_mat @ opt_w
        final_preds[is_target] = blended
        
        for i, m in enumerate(model_list):
            if opt_w[i] > 0.01:
                print(f"    {m}: {opt_w[i]:.4f}")

# Save
output = pd.DataFrame({'id': test['id'].values, 'target': final_preds})
output.to_csv("polymer_competition/submission_tree_optimal.csv", index=False)
print(f"\nSaved submission_tree_optimal.csv: {output.shape}")
print(f"TG mean: {output[test['target_type']=='tg']['target'].mean():.4f}")
print(f"EGC mean: {output[test['target_type']=='egc']['target'].mean():.4f}")

# Compare with Arch A
archA = pd.read_csv(r"D:\Parth\Poly\submission (2).csv")
diff = final_preds - archA['target'].values
print(f"\nvs Arch A: mean diff={diff.mean():.4f}, std={diff.std():.4f}")
