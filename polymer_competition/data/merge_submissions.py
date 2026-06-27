from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def merge_submissions(
    tg_csv: str | Path,
    egc_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    parts = []
    for label, path in [("tg", tg_csv), ("egc", egc_csv)]:
        p = Path(path)
        if p.exists():
            parts.append(pd.read_csv(p))
        else:
            print(f"WARNING: {label} submission not found at {p}. Skipping.")
    if not parts:
        raise FileNotFoundError("No submission files found to merge.")
    combined = pd.concat(parts, axis=0)
    combined = combined.sort_values("id").reset_index(drop=True)
    combined = combined[["id", "target"]]
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    print(f"Merged submission ({len(combined)} rows) -> {output_csv}")
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tg", default=None)
    parser.add_argument("--egc", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sub_dir = Path(cfg["paths"].get("submissions_dir", "outputs/submissions/"))
    tg_path = args.tg or sub_dir / "tg_preds.csv"
    egc_path = args.egc or sub_dir / "egc_preds.csv"
    out_path = args.output or sub_dir / "submission.csv"
    merge_submissions(tg_path, egc_path, out_path)


if __name__ == "__main__":
    main()
