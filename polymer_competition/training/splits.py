"""training/splits.py

Scaffold-aware 5-fold CV split generation using Murcko scaffolds.
Produces data/splits_{target}_scaffold.pkl files consumed by train.py.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import Scaffolds
from sklearn.model_selection import GroupKFold


def murcko_scaffold(smiles: str) -> str:
    """Compute Murcko scaffold from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        scaffold = Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return smiles


def generate_scaffold_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    smiles_col: str = "smiles",
    target_col: str = "target",
    seed: int = 42,
) -> dict[int, dict[str, np.ndarray]]:
    """Generate scaffold-aware 5-fold CV splits.

    Molecules sharing the same Murcko scaffold are kept in the same fold
    to prevent scaffold leakage. Folds are stratified by target quantiles.
    """
    scaffolds = df[smiles_col].apply(murcko_scaffold)
    target_bins = pd.qcut(df[target_col].rank(method="first"), q=5, labels=False)

    scaffold_groups = pd.DataFrame({"scaffold": scaffolds, "bin": target_bins})
    group_df = scaffold_groups.groupby("scaffold").agg(
        bin=("bin", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0),
        count=("bin", "count"),
    ).reset_index()

    gkf = GroupKFold(n_splits=n_folds)
    splits = {}
    rng = np.random.RandomState(seed)
    shuffled_idx = rng.permutation(len(group_df))
    group_df_shuffled = group_df.iloc[shuffled_idx].reset_index(drop=True)

    for fold, (train_group_idx, val_group_idx) in enumerate(
        gkf.split(group_df_shuffled, group_df_shuffled["bin"], groups=group_df_shuffled["scaffold"])
    ):
        val_scaffolds = set(group_df_shuffled.iloc[val_group_idx]["scaffold"])
        val_idx = df[scaffolds.isin(val_scaffolds)].index.values
        train_idx = df[~scaffolds.isin(val_scaffolds)].index.values
        splits[fold] = {"train_idx": train_idx, "val_idx": val_idx}

    return splits


def main():
    parser = argparse.ArgumentParser(description="Generate scaffold-aware CV splits")
    parser.add_argument("--data", default=None, help="Path to train.csv")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--n_folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg.get("paths", {}).get("data_dir", "data/"))
    n_folds = args.n_folds or cfg.get("cv", {}).get("n_folds", 5)
    seed = args.seed or cfg.get("seed", {}).get("global", 42)
    smiles_col = cfg.get("data", {}).get("smiles_col", "smiles")
    target_col = cfg.get("data", {}).get("target_col", "target")
    target_type_col = cfg.get("data", {}).get("target_type", "target_type")
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    data_path = Path(args.data) if args.data else data_dir / "train.csv"

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")

    targets = list(cfg.get("targets", {"tg": {}, "egc": {}}).keys())
    for target in targets:
        out_path = output_dir / f"splits_{target}_scaffold.pkl"
        if out_path.exists():
            print(f"Skipping {target} — split file already exists: {out_path}")
            continue

        tdf = df[df[target_type_col] == target].reset_index(drop=True) if target_type_col in df.columns else df
        print(f"{target}: {len(tdf)} samples")
        splits = generate_scaffold_splits(
            tdf, n_folds=n_folds, smiles_col=smiles_col, target_col=target_col, seed=seed
        )
        with open(out_path, "wb") as f:
            pickle.dump(splits, f)
        fold_sizes = [len(v["val_idx"]) for v in splits.values()]
        print(f"{target}: {len(splits)} folds, val sizes={fold_sizes}, mean={np.mean(fold_sizes):.0f}")


if __name__ == "__main__":
    main()
