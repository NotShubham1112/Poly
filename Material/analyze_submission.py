import pandas as pd
import numpy as np

df = pd.read_csv('D:/Parth/Poly/Material/submission.csv')
print(f"Rows: {len(df)}")
print(f"Negative preds: {(df['target'] < 0).sum()}")
print(f"Min: {df['target'].min():.2f}")
print(f"Max: {df['target'].max():.2f}")
print(f"Mean: {df['target'].mean():.2f}")
print(f"Median: {df['target'].median():.2f}")
print(f"Std: {df['target'].std():.2f}")

# Distribution analysis
print("\n--- Distribution ---")
print(f"< 0: {(df['target'] < 0).sum()}")
print(f"0-10: {((df['target'] >= 0) & (df['target'] < 10)).sum()}")
print(f"10-50: {((df['target'] >= 10) & (df['target'] < 50)).sum()}")
print(f"50-100: {((df['target'] >= 50) & (df['target'] < 100)).sum()}")
print(f"100-200: {((df['target'] >= 100) & (df['target'] < 200)).sum()}")
print(f"200-300: {((df['target'] >= 200) & (df['target'] < 300)).sum()}")
print(f"300+: {(df['target'] >= 300).sum()}")

# Percentiles
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"  P{p}: {np.percentile(df['target'], p):.2f}")
