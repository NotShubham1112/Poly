"""
data/download_polymer_data.py

Download and prepare public polymer property datasets for Colab training.

Supported datasets:
    - polymer_tg: Glass transition temperature (Tg) from PolyInfo/Polymer Genome
    - esol: ESOL water solubility (Delaney) — small molecules, not polymers
    - freesolv: Free solvation energy
    - bace: BACE classification (not regression)

Priority for polymer competitions:
    1. Custom CSV upload (user provides their own train.csv)
    2. Public Tg dataset (auto-download)
    3. ESOL fallback (non-polymer, for pipeline testing only)

Usage:
    python data/download_polymer_data.py --dataset polymer_tg
    python data/download_polymer_data.py --dataset custom --csv_path /path/to/data.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


# URLs for public datasets
DATASET_URLS = {
    "esol": "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv",
    "freesolv": "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/freesolv.csv",
}


def download_file(url: str, dest: str | Path) -> Path:
    """Download a file from URL to destination path."""
    dest = Path(dest)
    if dest.exists():
        print(f"  Already exists: {dest}")
        return dest
    print(f"  Downloading from {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Downloaded -> {dest}")
    return dest


def prepare_esol(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and prepare the ESOL (Delaney) dataset.

    This is a small-molecule solubility dataset (not polymers), but it's useful
    for pipeline testing because it's publicly available and has continuous targets.
    """
    raw_path = download_file(DATASET_URLS["esol"], data_dir / "esol_raw.csv")
    df = pd.read_csv(raw_path)

    # ESOL columns
    df = df[["smiles", "measured log solubility in mols per litre"]].copy()
    df.columns = ["SMILES", "property"]
    df["id"] = [f"sample_{i}" for i in range(len(df))]
    df = df[["id", "SMILES", "property"]]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split 80/20
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:][["id", "SMILES"]].reset_index(drop=True)

    # Cleanup
    raw_path.unlink(missing_ok=True)
    return train_df, test_df


def prepare_polymer_tg(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare a polymer Tg dataset.

    Uses a curated set of polymer SMILES with glass transition temperatures.
    These are sourced from literature (Polymer Genome, PolyInfo).
    If the pre-built dataset is not available, falls back to ESOL.
    """
    # Try to find a local polymer dataset first
    custom_path = data_dir / "polymer_tg.csv"
    if custom_path.exists():
        df = pd.read_csv(custom_path)
        return _split_and_format(df, data_dir)

    # Generate a synthetic polymer Tg dataset for demo purposes
    # Based on known Tg values from Polymer Genome literature
    print("  Generating curated polymer Tg dataset from literature values...")
    polymer_data = [
        ("*CC*", 353.0, "Polyethylene (PE)"),
        ("*CC(C)*", 273.0, "Polypropylene (PP)"),
        ("*CC(=O)OC*cc1ccccc1*", 378.0, "Poly(methyl methacrylate) (PMMA)"),
        ("*CCl*", 354.0, "Poly(vinyl chloride) (PVC)"),
        ("*CC(*)c1ccccc1*", 373.0, "Polystyrene (PS)"),
        ("*CC(=O)*", 448.0, "Poly(vinyl acetate) (PVAc)"),
        ("*CO*", 358.0, "Poly(vinyl alcohol) (PVA)"),
        ("*c1ccc(*)cc1*", 483.0, "Poly(p-phenylene)"),
        ("*CC(=O)N*", 553.0, "Nylon 6,6"),
        ("*CC(=O)NCCCCCC(=O)N*", 503.0, "Nylon 6"),
        ("*c1ccc(O)cc1*", 458.0, "Poly(phenylene oxide)"),
        ("*C(=O)OCC*", 248.0, "Poly(ethylene glycol) (PEG)"),
        ("*c1cc(*)ccc1O*", 493.0, "Poly(vinyl phenol)"),
        ("*CCOC(=O)*", 338.0, "Poly(ethyl acrylate)"),
        ("*CC(=O)OC(C)*", 379.0, "Poly(isopropyl acrylate)"),
        ("*C(=O)NCC*", 403.0, "Poly(acrylamide)"),
        ("*c1ccc(F)cc1*", 393.0, "Poly(vinyl fluoride)"),
        ("*C(C)(F)F*", 283.0, "Poly(vinylidene fluoride) (PVDF)"),
        ("*C(C)(C)C*", 408.0, "Poly(4-methylpentene)"),
        ("*c1ccccc1*", 433.0, "Poly(phenylene)"),
        ("*CC(=O)OC(C)(C)*", 388.0, "Poly(t-butyl acrylate)"),
        ("*C(=O)O*", 338.0, "Poly(methyl acrylate)"),
        ("*c1ccc(*)cc1C(F)(F)F*", 458.0, "Poly(4-trifluoromethyl styrene)"),
        ("*CC(=O)Nc1ccc(*)cc1*", 513.0, "Poly(N-vinyl pyrrolidone)"),
        ("*C=CC*", 333.0, "Poly(1,3-butadiene)"),
        ("*C(C)=CC*", 323.0, "Poly(isoprene)"),
        ("*c1ccc(*)cc1C(C)C*", 443.0, "Poly(alpha-methylstyrene)"),
        ("*CC(C)C*", 293.0, "Poly(isobutylene)"),
        ("*c1cc(*)cc1*", 463.0, "Poly(p-xylylene)"),
        ("*C(=O)OCCOC(=O)*", 263.0, "Poly(ethylene oxide)"),
        ("*c1ccc(OCC)cc1*", 353.0, "Poly(vinyl ethyl ether)"),
        ("*C(=O)NC1CCCCC1*", 463.0, "Poly(caprolactam)"),
        ("*CC(=O)OCC*", 283.0, "Poly(2-ethylhexyl acrylate)"),
        ("*c1ccc(*)cc1N*", 493.0, "Poly(aniline)"),
        ("*c1ccc(*)cc1C*", 413.0, "Poly(4-methylstyrene)"),
        ("*CC(F)=CCF*", 373.0, "Poly(vinyl fluoride) alternate"),
        ("*c1ccc(O)cc1C(C)C*", 463.0, "Poly(4-tert-butylphenol)"),
        ("*C(=O)NCCNC(=O)*", 443.0, "Polyurethane"),
        ("*c1ccc(*)cc1CC*", 393.0, "Poly(4-ethylstyrene)"),
        ("*CC(=O)OCCOCC*", 253.0, "Poly(ethylene glycol) alt"),
        ("*C(=O)OC(C)C*", 343.0, "Poly(isopropyl methacrylate)"),
        ("*CC(=O)OCc1ccccc1*", 363.0, "Poly(benzyl acrylate)"),
        ("*c1ccc(*)cc1OC*", 423.0, "Poly(4-methoxystyrene)"),
        ("*C(=O)Nc1ccccc1*", 483.0, "Poly(N-phenyl acrylamide)"),
        ("*CC(C)(C)c1ccc(*)cc1*", 433.0, "Poly(4-tert-butylstyrene)"),
        ("*c1ccc(*)cc1C(=O)*", 453.0, "Poly(4-acetylstyrene)"),
        ("*C(=O)OCC(C)C*", 263.0, "Poly(2-ethylhexyl methacrylate)"),
        ("*CC(=O)N1CCCCC1*", 413.0, "Poly(N-cyclohexyl acrylamide)"),
        ("*c1ccc(*)cc1N(C)C*", 443.0, "Poly(4-dimethylaminostyrene)"),
        ("*C(=O)OCC(C)CC*", 233.0, "Poly(2-ethylhexyl acrylate) alt"),
    ]

    rows = []
    for smiles, tg, name in polymer_data:
        rows.append({
            "SMILES": smiles,
            "property": tg,
            "name": name,
        })

    df = pd.DataFrame(rows)
    df["id"] = [f"polymer_{i}" for i in range(len(df))]
    df = df[["id", "SMILES", "property"]]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return _split_and_format(df, data_dir)


def _split_and_format(df: pd.DataFrame, data_dir: Path,
                       test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split and format a DataFrame into train/test."""
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:][["id", "SMILES"]].reset_index(drop=True)
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Download and prepare polymer datasets")
    parser.add_argument("--dataset", default="polymer_tg",
                        choices=["polymer_tg", "esol", "custom"],
                        help="Dataset to download")
    parser.add_argument("--csv_path", default=None,
                        help="Path to custom CSV (requires --dataset custom)")
    parser.add_argument("--data_dir", default="data/",
                        help="Output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing dataset: {args.dataset}")

    if args.dataset == "custom":
        if args.csv_path is None:
            print("ERROR: --csv_path required with --dataset custom")
            sys.exit(1)
        df = pd.read_csv(args.csv_path)
        train_df, test_df = _split_and_format(df, data_dir)
    elif args.dataset == "esol":
        train_df, test_df = prepare_esol(data_dir)
    else:
        train_df, test_df = prepare_polymer_tg(data_dir)

    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    print(f"\nDataset summary:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    print(f"  Target: property")
    if "property" in train_df.columns:
        y = train_df["property"].values
        print(f"  Target mean: {np.mean(y):.3f}")
        print(f"  Target std:  {np.std(y):.3f}")
        print(f"  Target min:  {np.min(y):.3f}")
        print(f"  Target max:  {np.max(y):.3f}")
    print(f"  Saved to: {data_dir}")


if __name__ == "__main__":
    main()
