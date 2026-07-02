import pickle, numpy as np, glob
from sklearn.metrics import r2_score

files = glob.glob('predictions/v28_*polychain_boosted_s*_fold*.pkl')
files = [f for f in files if '_test' not in f]

seeds_data = {}
for f in sorted(files):
    parts = f.replace('.pkl','').split('_')
    target = parts[1]
    seed = None
    for p in parts:
        if p.startswith('s') and p[1:].isdigit():
            seed = int(p[1:])
            break
    fold = int([p for p in parts if p.startswith('fold')][0].replace('fold',''))
    with open(f, 'rb') as fh:
        d = pickle.load(fh)
    key = (target, seed)
    if key not in seeds_data:
        seeds_data[key] = {'preds': [], 'y': [], 'folds': 0, 'r2s': []}
    seeds_data[key]['preds'].extend(d['pred'])
    seeds_data[key]['y'].extend(d['y'])
    seeds_data[key]['folds'] += 1
    seeds_data[key]['r2s'].append(d['metrics']['r2'])

print('=== POLYCHAIN BOOSTED RESULTS ===')
for (target, seed), data in sorted(seeds_data.items()):
    r2 = r2_score(data['y'], data['preds'])
    print("{} seed={:<6} folds={}  CV R2={:.4f}+-{:.4f}  OOF R2={:.4f}".format(
        target.upper(), seed, data['folds'], np.mean(data['r2s']), np.std(data['r2s']), r2))

for target in ['tg','egc']:
    all_p, all_y = [], []
    for (t,s), data in seeds_data.items():
        if t == target:
            all_p.extend(data['preds'])
            all_y.extend(data['y'])
    if all_p:
        print("{} ALL SEEDS OOF R2={:.4f}".format(target.upper(), r2_score(all_y, all_p)))

# Load existing v28 8-model predictions for comparison
existing_files = [f for f in glob.glob('predictions/v28_*_oof.pkl') if 'polychain' not in f.lower()]
if existing_files:
    print("\n=== EXISTING 8-MODEL OOF (for reference) ===")
    for f in sorted(existing_files):
        with open(f, 'rb') as fh:
            d = pickle.load(fh)
        r2 = r2_score(d['targets'], d['predictions']) if 'targets' in d else r2_score(d['y'], d['pred'])
        print("  {} R2={:.4f}".format(f.split('\\')[-1], r2))
