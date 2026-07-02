import pandas as pd, numpy as np

test = pd.read_csv('polymer_competition/data/test.csv')

files = {
    'champion_0.876': 'polymer_competition/submission_champion.csv',
    'archA_only': r'D:\Parth\Poly\submission (2).csv',
    'final_blend': 'polymer_competition/submission_final_blend.csv',
    'exhaustive': 'polymer_competition/submission_exhaustive.csv',
}

print("COMPARISON OF ALL SUBMISSION CANDIDATES")
print("=" * 70)
for name, path in files.items():
    df = pd.read_csv(path)
    merged = df.merge(test[['id','target_type']], on='id')
    tg_mean = merged[merged['target_type']=='tg']['target'].mean()
    egc_mean = merged[merged['target_type']=='egc']['target'].mean()
    overall_mean = df['target'].mean()
    print(f"  {name:25s}: overall={overall_mean:.4f}  TG={tg_mean:.4f}  EGC={egc_mean:.4f}")

print("\nCorrelation matrix:")
dfs = {}
for name, path in files.items():
    df = pd.read_csv(path)
    dfs[name] = df.set_index('id')['target']

names = list(dfs.keys())
for i, n1 in enumerate(names):
    for j, n2 in enumerate(names):
        if j > i:
            common = dfs[n1].index.intersection(dfs[n2].index)
            corr = np.corrcoef(dfs[n1][common], dfs[n2][common])[0,1]
            print(f"  {n1} x {n2}: {corr:.6f}")

print("\n" + "=" * 70)
print("OOF SCORES (from optimization)")
print("=" * 70)
print("  champion (Arch A + 35% PolyChain): score=0.876 (actual Kaggle)")
print("  final_blend (v27+v30 trees only):  TG OOF=0.882, EGC OOF=0.917")
print("                                     Mean OOF=0.899, expected test ~0.911")
print()
print("  The final_blend removes PolyChain and adds v30 xgb+lgb.")
print("  Expected improvement over champion: +0.02 to +0.035")
