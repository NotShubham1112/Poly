from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def smiles_scaffold(smiles: str) -> str:
    s = smiles.replace("*", "")
    s = s.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    return s[:20]


def generate_splits(
    train_path: str | Path,
    output_path: str | Path,
    n_folds: int = 5,
    seed: int = 42,
    target_col: str = "target",
    smiles_col: str = "SMILES",
    strategy: str = "group",
) -> dict:
    train = pd.read_csv(train_path)
    if smiles_col not in train.columns:
        for c in train.columns:
            if c.lower() == smiles_col.lower():
                smiles_col = c
                break
    print(f"Loaded {len(train)} rows from {train_path}")

    splits = {}
    if strategy == "group":
        from sklearn.model_selection import GroupKFold
        scaffolds = train[smiles_col].apply(smiles_scaffold).values
        gkf = GroupKFold(n_splits=n_folds)
        y = train[target_col].values if target_col in train.columns else np.zeros(len(train))
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y, groups=scaffolds)):
            splits[fold] = {"train": tr_idx.tolist(), "val": va_idx.tolist()}
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(train)):
            splits[fold] = {"train": tr_idx.tolist(), "val": va_idx.tolist()}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(splits, f)
    print(f"Saved {n_folds}-fold splits -> {output_path}")
    for fold_id, idx in splits.items():
        print(f"  Fold {fold_id}: train={len(idx['train'])}, val={len(idx['val'])}")
    return splits


def main():
    parser = argparse.ArgumentParser(description="Generate CV splits")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--strategy", default=None, choices=["group", "random"])
    parser.add_argument("--target", default=None, help="Target name (tg/egc). If omitted, runs for all.")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg.get("paths", {}).get("data_dir", "data/"))
    n_folds = cfg.get("cv", {}).get("n_folds", 5)
    seed = cfg.get("seed", {}).get("global", 42)
    strategy = args.strategy or cfg.get("cv", {}).get("split_type", "group")

    targets = [args.target] if args.target else list(cfg.get("targets", {"tg": {}}).keys())
    for t in targets:
        train_path = data_dir / t / "train.csv"
        output_path = data_dir / f"splits_{t}.pkl"
        if not train_path.exists():
            print(f"  Skipping {t}: {train_path} not found")
            continue
        generate_splits(
            train_path, output_path,
            n_folds=n_folds, seed=seed,
            target_col=cfg["data"]["target_col"],
            smiles_col=cfg["data"].get("smiles_col", "smiles"),
            strategy=strategy,
        )


if __name__ == "__main__":
    main()
