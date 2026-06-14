"""
data/download_qm9.py

Download QM9 dataset from HuggingFace and save as parquet.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROPERTY_COLUMNS = [
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
    "u0", "u", "h", "g", "cv",
]

ALL_TARGET_COLUMNS = PROPERTY_COLUMNS + ["A", "B", "C"]


def download_qm9(save_dir: str | Path = "data/") -> pd.DataFrame:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "qm9_raw.parquet"

    if out_path.exists():
        print(f"QM9 already downloaded: {out_path}")
        return pd.read_parquet(out_path)

    from datasets import load_dataset
    ds = load_dataset("yairschiff/qm9", split="train")
    df = ds.to_pandas()
    assert "smiles" in df.columns, f"Missing 'smiles' column, got {list(df.columns)}"
    assert "canonical_smiles" in df.columns, f"Missing 'canonical_smiles' column"

    missing = [c for c in ALL_TARGET_COLUMNS if c not in df.columns]
    if missing:
        print(f"Warning: missing target columns: {missing}")

    df.to_parquet(out_path, index=False)
    print(f"Saved QM9 ({len(df):,} molecules, {len(df.columns)} columns) -> {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Download QM9 dataset")
    parser.add_argument("--save_dir", default="data/")
    args = parser.parse_args()
    df = download_qm9(args.save_dir)
    print(f"Columns: {list(df.columns)}")
    print(f"Target columns: {ALL_TARGET_COLUMNS}")


if __name__ == "__main__":
    main()
