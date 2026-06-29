#!/usr/bin/env python
"""reports/diversity_analysis.py
Analyze ensemble diversity: prediction correlations and residual correlations.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def load_oof_matrix(pred_dir, exp, target):
    """Load OOF predictions and build (n_samples, n_models) matrix."""
    rows = []
    for pkl_file in pred_dir.glob(f"{exp}_{target}_*_fold*.pkl"):
        if '_test' in pkl_file.stem:
            continue
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        for idx, pred, y in zip(data['val_idx'], data['pred'], data['y']):
            rows.append({
                'idx': int(idx), 'pred': float(pred), 'y': float(y),
                'model': data.get('model_type', 'unknown'),
                'fold': int(data.get('fold', 0)),
            })
    df = pd.DataFrame(rows)
    grouped = df.groupby(['idx', 'model'])['pred'].mean().unstack()
    y = df.groupby('idx')['y'].first().reindex(grouped.index)
    return grouped, y


def analyze_diversity(exp, target, output_dir='reports/plots'):
    pred_dir = Path('predictions')
    oof_df, y = load_oof_matrix(pred_dir, exp, target)
    
    # Remove models with too few predictions
    min_preds = len(oof_df) * 0.5
    oof_df = oof_df.dropna(axis=1, thresh=int(min_preds))
    
    print(f"\n{'='*60}")
    print(f"  Diversity Analysis: {target.upper()}")
    print(f"{'='*60}")
    print(f"\nModels: {list(oof_df.columns)}")
    print(f"Samples: {len(oof_df)}")
    
    # Prediction correlation
    pred_corr = oof_df.corr()
    print(f"\n--- Prediction Correlation Matrix ---")
    print(pred_corr.round(3).to_string())
    
    # Residual correlation
    residuals = pd.DataFrame(index=oof_df.index)
    for col in oof_df.columns:
        residuals[col] = y - oof_df[col]
    res_corr = residuals.corr()
    print(f"\n--- Residual Correlation Matrix ---")
    print(res_corr.round(3).to_string())
    
    # Identify highly correlated model pairs
    print(f"\n--- Highly Correlated Pairs (|r| > 0.95) ---")
    found = False
    for i in range(len(pred_corr.columns)):
        for j in range(i+1, len(pred_corr.columns)):
            r = pred_corr.iloc[i, j]
            if abs(r) > 0.95:
                print(f"  {pred_corr.columns[i]} <-> {pred_corr.columns[j]}: {r:.3f}")
                found = True
    if not found:
        print("  None found")
    
    # Identify diverse model pairs (low residual correlation)
    print(f"\n--- Most Diverse Pairs (lowest residual correlation) ---")
    pairs = []
    for i in range(len(res_corr.columns)):
        for j in range(i+1, len(res_corr.columns)):
            pairs.append((res_corr.columns[i], res_corr.columns[j], res_corr.iloc[i, j]))
    pairs.sort(key=lambda x: x[2])
    for m1, m2, r in pairs[:5]:
        print(f"  {m1} <-> {m2}: {r:.3f}")
    
    # Individual model performance
    print(f"\n--- Individual Model R2 ---")
    for col in oof_df.columns:
        r2 = 1 - np.sum((y - oof_df[col]) ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"  {col:<15} R2 = {r2:.4f}")
    
    # Plot correlation heatmaps
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(pred_corr, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                ax=axes[0], vmin=0, vmax=1)
    axes[0].set_title(f'Prediction Correlation ({target.upper()})')
    
    sns.heatmap(res_corr, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0,
                ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title(f'Residual Correlation ({target.upper()})')
    
    plt.tight_layout()
    out_path = f'{output_dir}/diversity_{target}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")
    
    return pred_corr, res_corr


if __name__ == '__main__':
    for t in ['tg', 'egc']:
        analyze_diversity('v27', t)
