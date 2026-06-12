"""
notebooks/eda_report.py

Automated Exploratory Data Analysis for the polymer competition.
Generates an HTML report (or prints to stdout) covering:
    1. Target distribution (histogram + stats)
    2. Missing values summary
    3. Duplicate SMILES detection
    4. Train/test distribution drift (fingerprint PCA)
    5. Leakage checks
    6. Feature correlation heatmap

Usage:
    python notebooks/eda_report.py --config config.yaml
    python notebooks/eda_report.py --train data/train.csv --test data/test.csv

This can also be used inside a Jupyter/Colab notebook:
    from notebooks.eda_report import run_eda
    run_eda("data/train.csv", "data/test.csv", target_col="property")
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_eda(
    train_path: str,
    test_path: str | None = None,
    target_col: str = "property",
    output_dir: str = "reports",
) -> dict:
    """Run full EDA and return a summary dict."""
    results = {}

    print("=" * 70)
    print("  POLYMER COMPETITION — AUTOMATED EDA REPORT")
    print("=" * 70)

    # ── Load data ──
    train = pd.read_csv(train_path)
    print(f"\n[Data] Train: {train.shape[0]} rows, {train.shape[1]} columns")
    print(f"  Columns: {list(train.columns)}")
    results["train_shape"] = train.shape

    test = None
    if test_path and Path(test_path).exists():
        test = pd.read_csv(test_path)
        print(f"[Data] Test:  {test.shape[0]} rows, {test.shape[1]} columns")
        results["test_shape"] = test.shape

    # ── 1. Target distribution ──
    print(f"\n{'─' * 50}")
    print("1. TARGET DISTRIBUTION")
    print(f"{'─' * 50}")
    if target_col in train.columns:
        y = train[target_col]
        stats = {
            "count": int(y.count()),
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "25%": float(y.quantile(0.25)),
            "50%": float(y.median()),
            "75%": float(y.quantile(0.75)),
            "max": float(y.max()),
            "skewness": float(y.skew()),
            "kurtosis": float(y.kurtosis()),
            "n_missing": int(y.isna().sum()),
        }
        for k, v in stats.items():
            print(f"  {k:>12}: {v:.4f}" if isinstance(v, float) else f"  {k:>12}: {v}")
        results["target_stats"] = stats

        # Save histogram
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(y.dropna(), bins=50, edgecolor="black", alpha=0.7, color="#4C78A8")
            axes[0].set_xlabel(target_col)
            axes[0].set_ylabel("Count")
            axes[0].set_title(f"Target Distribution: {target_col}")
            axes[1].boxplot(y.dropna(), vert=True)
            axes[1].set_ylabel(target_col)
            axes[1].set_title("Box Plot")
            plt.tight_layout()
            plt.savefig(out_dir / "eda_target_distribution.png", dpi=120)
            plt.close()
            print(f"  → Saved: {out_dir / 'eda_target_distribution.png'}")
        except ImportError:
            print("  (matplotlib not available — skipping plots)")
    else:
        print(f"  ⚠ Target column '{target_col}' not found in train data.")

    # ── 2. Missing values ──
    print(f"\n{'─' * 50}")
    print("2. MISSING VALUES")
    print(f"{'─' * 50}")
    missing = train.isnull().sum()
    missing_pct = (missing / len(train) * 100).round(2)
    missing_df = pd.DataFrame({"missing": missing, "pct": missing_pct})
    missing_df = missing_df[missing_df["missing"] > 0].sort_values("missing", ascending=False)
    if missing_df.empty:
        print("  ✓ No missing values in training data.")
    else:
        print(missing_df.to_string())
    results["missing"] = missing_df.to_dict()

    # ── 3. Duplicate SMILES ──
    print(f"\n{'─' * 50}")
    print("3. DUPLICATE SMILES")
    print(f"{'─' * 50}")
    if "SMILES" in train.columns:
        n_unique = train["SMILES"].nunique()
        n_total = len(train)
        n_dupes = n_total - n_unique
        print(f"  Total SMILES:  {n_total}")
        print(f"  Unique SMILES: {n_unique}")
        print(f"  Duplicates:    {n_dupes} ({n_dupes/n_total*100:.1f}%)")

        if n_dupes > 0 and target_col in train.columns:
            dupe_smiles = train[train.duplicated("SMILES", keep=False)]
            dupe_var = dupe_smiles.groupby("SMILES")[target_col].std()
            high_var = dupe_var[dupe_var > dupe_var.quantile(0.9)]
            if len(high_var) > 0:
                print(f"  ⚠ {len(high_var)} duplicate groups have high target variance (>p90).")
                print(f"    Max std within duplicates: {dupe_var.max():.4f}")
        results["duplicates"] = {"total": n_total, "unique": n_unique, "dupes": n_dupes}
    else:
        print("  ⚠ No 'SMILES' column found.")

    # ── 4. Train/test overlap & distribution drift ──
    print(f"\n{'─' * 50}")
    print("4. TRAIN/TEST DISTRIBUTION")
    print(f"{'─' * 50}")
    if test is not None and "SMILES" in train.columns and "SMILES" in test.columns:
        overlap = set(train["SMILES"]) & set(test["SMILES"])
        print(f"  SMILES overlap: {len(overlap)} molecules")
        if len(overlap) > 0:
            print(f"  ⚠ LEAKAGE RISK: {len(overlap)} SMILES appear in both train and test!")

        # Fingerprint-based drift check
        try:
            from features.fingerprints import morgan_fingerprints
            from sklearn.decomposition import PCA

            all_smi = train["SMILES"].tolist() + test["SMILES"].tolist()
            fps = morgan_fingerprints(all_smi, radius=2, n_bits=512)  # smaller for speed
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(fps.astype(float))
            train_coords = coords[:len(train)]
            test_coords = coords[len(train):]

            # Simple drift metric: Wasserstein distance on PC1
            from scipy.stats import wasserstein_distance
            drift_pc1 = wasserstein_distance(train_coords[:, 0], test_coords[:, 0])
            drift_pc2 = wasserstein_distance(train_coords[:, 1], test_coords[:, 1])
            print(f"  FP-PCA drift (Wasserstein): PC1={drift_pc1:.4f}, PC2={drift_pc2:.4f}")
            results["drift"] = {"pc1": drift_pc1, "pc2": drift_pc2}

            # Save PCA scatter
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                out_dir = Path(output_dir)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(train_coords[:, 0], train_coords[:, 1],
                          alpha=0.3, s=10, label="Train", color="#4C78A8")
                ax.scatter(test_coords[:, 0], test_coords[:, 1],
                          alpha=0.3, s=10, label="Test", color="#E45756")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title("Train/Test Distribution (Morgan FP PCA)")
                ax.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "eda_train_test_drift.png", dpi=120)
                plt.close()
                print(f"  → Saved: {out_dir / 'eda_train_test_drift.png'}")
            except ImportError:
                pass
        except ImportError:
            print("  (sklearn/rdkit not available — skipping drift analysis)")
    elif test is None:
        print("  No test file provided.")

    # ── 5. Leakage check ──
    print(f"\n{'─' * 50}")
    print("5. LEAKAGE CHECKS")
    print(f"{'─' * 50}")
    if "id" in train.columns and target_col in train.columns:
        corr = train["id"].corr(train[target_col])
        print(f"  id ↔ target correlation: {corr:.4f}")
        if abs(corr) > 0.1:
            print(f"  ⚠ Possible target leakage through ID column!")
        else:
            print(f"  ✓ No obvious leakage from ID.")
    # Check if SMILES length correlates with target
    if "SMILES" in train.columns and target_col in train.columns:
        train["_smi_len"] = train["SMILES"].str.len()
        corr_len = train["_smi_len"].corr(train[target_col])
        print(f"  SMILES length ↔ target correlation: {corr_len:.4f}")
        train.drop(columns=["_smi_len"], inplace=True)

    # ── 6. Summary ──
    print(f"\n{'=' * 70}")
    print("  EDA COMPLETE")
    print(f"{'=' * 70}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Automated EDA Report")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train", default=None)
    parser.add_argument("--test", default=None)
    args = parser.parse_args()

    try:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        cfg = {}

    data_dir = Path(cfg.get("paths", {}).get("data_dir", "data/"))
    train_path = args.train or str(data_dir / "train.csv")
    test_path = args.test or str(data_dir / "test.csv")
    target_col = cfg.get("target", {}).get("column", "property")

    run_eda(train_path, test_path, target_col)


if __name__ == "__main__":
    main()
