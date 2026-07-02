import pandas as pd
t = pd.read_csv("data/train.csv")
tg = t[t["target_type"]=="tg"].reset_index(drop=True)
tg2 = pd.read_csv("data/tg/train.csv")
print(f"Filtered: {len(tg)}, Target-specific: {len(tg2)}")
match = (tg["smiles"] == tg2["smiles"]).mean()
print(f"SMILES match: {match*100:.1f}%")
