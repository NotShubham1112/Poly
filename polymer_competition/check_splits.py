import pickle, pathlib
for f in ['data/splits_tg.pkl', 'data/splits_egc.pkl']:
    if pathlib.Path(f).exists():
        s = pickle.load(open(f, 'rb'))
        print(f'{f}: {len(s)} folds')
        for k, v in s.items():
            print(f'  Fold {k}: {len(v[\"train\"])} train, {len(v[\"val\"])} val')
    else:
        print(f'{f}: NOT FOUND')
