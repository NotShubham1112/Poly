"""Check what target transforms are auto-selected for TG and EGC."""
import pandas as pd
import numpy as np
from features.target_transforms import select_best_transform

train = pd.read_csv("data/train.csv")
for target_type in ["tg", "egc"]:
    mask = train["target_type"] == target_type
    y = train.loc[mask, "target"].values
    print(f"\n{target_type.upper()}: n={len(y)}, mean={y.mean():.2f}, std={y.std():.2f}, "
          f"min={y.min():.2f}, max={y.max():.2f}")
    y_trans, inv_func, name = select_best_transform(y)
    print(f"  Auto-selected transform: {name}")
    print(f"  Transformed: mean={y_trans.mean():.2f}, std={y_trans.std():.2f}")
