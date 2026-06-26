"""data/split_by_target.py

Split train/test CSVs by target_type into per-target subdirectories.

Usage:
    python -m data.split_by_target --config config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def split_by_target(
    train_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    targets: list[str] | None = None,
) -> None:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    available = train["target_type"].unique().tolist()
    if targets is None:
        targets = available
    for t in targets:
        t_dir = Path(output_dir) / t
        t_dir.mkdir(parents=True, exist_ok=True)
        train_subset = train[train["target_type"] == t].copy()
        test_subset = test[test["target_type"] == t].copy()
        train_subset.to_csv(t_dir / "train.csv", index=False)
        test_subset.to_csv(t_dir / "test.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--targets", default=None, help="Comma-separated target types")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_dir = Path(cfg.get("paths", {}).get("data_dir", "data/"))
    targets = args.targets.split(",") if args.targets else None
    split_by_target(
        data_dir / "train.csv",
        data_dir / "test.csv",
        data_dir,
        targets=targets,
    )


if __name__ == "__main__":
    main()
