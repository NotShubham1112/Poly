"""
features/build_features.py

Master feature-building pipeline.

Loads train/test CSVs, computes fingerprints + descriptors + polymer-specific
features, merges them, applies imputation, and saves:
    data/processed/train_features.parquet
    data/processed/test_features.parquet
    data/splits.pkl

Run:
    python -m features.build_features --config config.yaml
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold

from .fingerprints import all_fingerprints
from .descriptors import compute_descriptors, select_descriptors_by_variance
from .custom_polymer import compute_all_custom_features


# ----------------------------------------------------------------------------
# Cross-validation split generation
# ----------------------------------------------------------------------------
def _smiles_scaffold(smiles: str) -> str:
    """Compute a coarse SMILES scaffold for grouping."""
    # Strip polymer asterisks and ring markers to get a simple key
    s = smiles.replace("*", "")
    s = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    return s[:min(20, len(s))]  # use prefix as proxy scaffold


def make_splits(df: pd.DataFrame, n_folds: int, seed: int, target: str) -> dict:
    """Generate Group K-Fold splits by SMILES scaffold."""
    scaffolds = df["SMILES"].apply(_smiles_scaffold).values
    gkf = GroupKFold(n_splits=n_folds)
    splits = {}
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df[target], groups=scaffolds)):
        splits[fold] = {"train": train_idx.tolist(), "val": val_idx.tolist()}
    return splits


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def build_features(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    out_dir = data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    target = cfg["target"]["column"]
    n_folds = cfg["cv"]["n_folds"]
    seed = cfg["seed"]

    print(f"Loading data from {data_dir} ...")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    print(f"  train: {len(train)} rows, test: {len(test)} rows, target: {target}")

    # 1. Fingerprints
    print("Computing fingerprints ...")
    train_smiles = train["SMILES"].tolist()
    test_smiles = test["SMILES"].tolist()
    all_smiles = train_smiles + test_smiles
    fps = all_fingerprints(all_smiles)
    fp_dfs = {name: pd.DataFrame(arr,
                                 columns=[f"{name}_{i}" for i in range(arr.shape[1])])
              for name, arr in fps.items()}
    train_fp = {name: df.iloc[: len(train)].reset_index(drop=True)
                for name, df in fp_dfs.items()}
    test_fp = {name: df.iloc[len(train):].reset_index(drop=True)
               for name, df in fp_dfs.items()}

    # 2. RDKit descriptors
    print("Computing RDKit descriptors ...")
    train_desc = compute_descriptors(train_smiles)
    test_desc = compute_descriptors(test_smiles)
    train_desc = select_descriptors_by_variance(train_desc)
    test_desc = test_desc[train_desc.columns]  # align columns
    desc_cols = [c for c in train_desc.columns if c != "SMILES"]

    # 3. Custom polymer features
    print("Computing polymer-specific features ...")
    train_cust = compute_all_custom_features(train_smiles)
    test_cust = compute_all_custom_features(test_smiles)
    cust_cols = [c for c in train_cust.columns if c != "SMILES"]

    # 4. Merge
    print("Merging feature matrices ...")
    train_X = pd.concat(
        [train["SMILES"], train["id"], train[target]] +
        [df for df in train_fp.values()] +
        [train_desc[desc_cols]] +
        [train_cust[cust_cols]],
        axis=1,
    )
    test_X = pd.concat(
        [test["SMILES"], test["id"]] +
        [df for df in test_fp.values()] +
        [test_desc[desc_cols]] +
        [test_cust[cust_cols]],
        axis=1,
    )

    # 5. Imputation (median for everything numeric)
    print("Imputing missing values ...")
    num_cols = [c for c in train_X.columns
                if c not in ("SMILES", "id", target) and train_X[c].dtype != object]
    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_X[num_cols])
    train_X[num_cols] = imputer.transform(train_X[num_cols])
    test_X[num_cols] = imputer.transform(test_X[num_cols])

    # 6. Save
    print(f"Saving to {out_dir} ...")
    train_X.to_parquet(out_dir / "train_features.parquet", index=False)
    test_X.to_parquet(out_dir / "test_features.parquet", index=False)

    # 7. Cross-validation splits
    print("Generating cross-validation splits ...")
    splits = make_splits(train_X, n_folds=n_folds, seed=seed, target=target)
    with open(data_dir / "splits.pkl", "wb") as f:
        pickle.dump(splits, f)

    print(f"Done. Train shape: {train_X.shape}, Test shape: {test_X.shape}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    build_features(args.config)
