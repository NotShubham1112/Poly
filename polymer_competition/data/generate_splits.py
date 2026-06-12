"""
data/generate_splits.py

Generate shared 5-fold cross-validation splits and save to splits.pkl.
Uses GroupKFold by SMILES scaffold (or plain KFold if scaffolds are
unavailable). The same splits.pkl is consumed by training/train.py
and ensemble/build_ensemble.py.

Usage:
    python -m data.generate_splits --config config.yaml
    python data/generate_splits.py                      # standalone
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running standalone (outside package context)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def smiles_scaffold(smiles: str) -> str:
    """Compute a coarse scaffold string for grouping."""
    s = smiles.replace("*", "")
    s = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    return s[:min(20, len(s))]


def generate_splits(
    train_path: str | Path,
    output_path: str | Path,
    n_folds: int = 5,
    seed: int = 42,
    target_col: str = "property",
    strategy: str = "group",
) -> dict:
    """Create n_folds cross-validation splits and save as .pkl.

    Parameters
    ----------
    train_path  : path to train.csv
    output_path : where to write splits.pkl
    n_folds     : number of CV folds
    seed        : random state
    target_col  : name of the target column
    strategy    : 'group' (GroupKFold by scaffold) or 'random' (KFold)

    Returns
    -------
    splits : dict  {fold_id: {'train': [...], 'val': [...]}}
    """
    train = pd.read_csv(train_path)
    print(f"Loaded {len(train)} rows from {train_path}")

    splits = {}

    if strategy == "group":
        from sklearn.model_selection import GroupKFold

        scaffolds = train["SMILES"].apply(smiles_scaffold).values
        gkf = GroupKFold(n_splits=n_folds)
        y = train[target_col].values if target_col in train.columns else np.zeros(len(train))
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y, groups=scaffolds)):
            splits[fold] = {"train": tr_idx.tolist(), "val": va_idx.tolist()}
    else:
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(train)):
            splits[fold] = {"train": tr_idx.tolist(), "val": va_idx.tolist()}

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(splits, f)
    print(f"Saved {n_folds}-fold splits → {output_path}")

    # Summary
    for fold_id, idx in splits.items():
        print(f"  Fold {fold_id}: train={len(idx['train'])}, val={len(idx['val'])}")

    return splits


def main():
    parser = argparse.ArgumentParser(description="Generate CV splits")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to global config.yaml")
    parser.add_argument("--train", default=None,
                        help="Override path to train.csv")
    parser.add_argument("--output", default=None,
                        help="Override output path for splits.pkl")
    parser.add_argument("--strategy", default=None,
                        choices=["group", "random"],
                        help="Override split strategy from config")
    args = parser.parse_args()

    # Load config
    try:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        cfg = {}

    data_dir = Path(cfg.get("paths", {}).get("data_dir", "data/"))
    train_path = args.train or str(data_dir / "train.csv")
    output_path = args.output or str(data_dir / "splits.pkl")
    n_folds = cfg.get("cv", {}).get("n_folds", 5)
    seed = cfg.get("seed", 42)
    target_col = cfg.get("target", {}).get("column", "property")
    strategy = args.strategy or cfg.get("cv", {}).get("split_type", "group")

    generate_splits(train_path, output_path, n_folds, seed, target_col, strategy)


if __name__ == "__main__":
    main()
