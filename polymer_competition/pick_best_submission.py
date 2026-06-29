import pandas as pd
import yaml
from pathlib import Path

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
ver = cfg['experiment']['version']
sub_dir = Path(cfg['paths']['submissions_dir'])

# Compare all submissions
for target in ['tg', 'egc']:
    print(f'\n=== {target.upper()} ===')
    
    w_path = sub_dir / f'{target}_preds.csv'
    s_path = sub_dir / f'{ver}_{target}_stacking.csv'
    
    w = pd.read_csv(w_path) if w_path.exists() else None
    s = pd.read_csv(s_path) if s_path.exists() else None
    
    if w is not None:
        print(f'  Weighted: {len(w)} rows, mean={w["target"].mean():.2f}')
    if s is not None:
        print(f'  Stacking: {len(s)} rows, mean={s["target"].mean():.2f}')
    
    # Pick best based on CV R2
    if target == 'egc' and s is not None:
        best = s
        print(f'  -> Using Stacking (CV R2=0.9118)')
    elif w is not None:
        best = w
        print(f'  -> Using Weighted')
    else:
        best = s
    
    best.to_csv(sub_dir / f'{target}_best.csv', index=False)

# Merge final submission
tg = pd.read_csv(sub_dir / 'tg_best.csv')
egc = pd.read_csv(sub_dir / 'egc_best.csv')
combined = pd.concat([tg, egc]).sort_values('id').reset_index(drop=True)
combined.to_csv(sub_dir / 'submission.csv', index=False)
print(f'\nFinal submission: {len(combined)} rows')
print(f'ID range: [{combined["id"].min()}, {combined["id"].max()}]')
