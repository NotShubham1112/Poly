"""
training/run_all_folds.py

Run 5-fold cross-validation for all model types and collect metrics.

Produces:
    results/fold_metrics.csv  — per-fold metrics for all models
    results/summary.csv       — mean +/- std per model
    reports/plots/            — all visualization plots

Usage:
    python -m training.run_all_folds --config config.yaml
    python -m training.run_all_folds --models ridge,xgb,gcn,polychain
    python -m training.run_all_folds --folds 0,1,2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ALL_MODELS = [
    "ridge", "rf", "xgb", "lgb", "catboost",
    "mlp", "gcn", "gat", "polychain",
]


def run_fold(model_type: str, fold: int, config: str, person: str) -> dict | None:
    """Train a single model on a single fold. Return metrics dict or None on failure."""
    # Find model-specific config
    model_cfg_candidates = [
        f"training/configs/{model_type}.yaml",
        f"training/configs/{model_type}_finetune.yaml",
        f"models/polychain/configs/finetune.yaml",
    ]
    model_cfg = None
    for c in model_cfg_candidates:
        if (PROJECT_ROOT / c).exists():
            model_cfg = c
            break

    cmd = [
        sys.executable, "-m", "training.train",
        "--model_type", model_type,
        "--fold", str(fold),
        "--config", config,
        "--person", person,
    ]
    if model_cfg:
        cmd += ["--model_config", model_cfg]

    print(f"\n{'='*60}")
    print(f"  Training {model_type} fold {fold}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  FAILED: {model_type} fold {fold}")
        return None

    # Load prediction file to get metrics
    pred_file = PROJECT_ROOT / "predictions" / f"{person}_{model_type}_fold{fold}.pkl"
    if pred_file.exists():
        import pickle
        with open(pred_file, "rb") as f:
            data = pickle.load(f)
        metrics = data.get("metrics", {})
        metrics["fold"] = fold
        metrics["model_type"] = model_type
        return metrics
    return None


def main():
    parser = argparse.ArgumentParser(description="Run full 5-fold cross-validation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model types (default: all)")
    parser.add_argument("--folds", default=None,
                        help="Comma-separated fold indices (default: 0-4)")
    parser.add_argument("--person", default="cv_run",
                        help="Person name for prediction files")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    models = args.models.split(",") if args.models else ALL_MODELS
    n_folds = cfg.get("cv", {}).get("n_folds", 5)
    folds = [int(x) for x in args.folds.split(",")] if args.folds else list(range(n_folds))

    print(f"Running {len(models)} models x {len(folds)} folds = {len(models) * len(folds)} jobs")

    # Run all combinations
    all_metrics = []
    for model_type in models:
        for fold in folds:
            metrics = run_fold(model_type, fold, args.config, args.person)
            if metrics:
                all_metrics.append(metrics)

    if not all_metrics:
        print("No successful training runs. Check errors above.")
        return

    # Build results DataFrame
    df = pd.DataFrame(all_metrics)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # Save per-fold metrics
    df.to_csv(results_dir / "fold_metrics.csv", index=False)
    print(f"\nPer-fold metrics -> {results_dir / 'fold_metrics.csv'}")

    # Compute summary (mean +/- std)
    summary_rows = []
    for model in df["model_type"].unique():
        subset = df[df["model_type"] == model]
        row = {"model": model}
        for metric in ["rmse", "mae", "r2", "spearman"]:
            if metric in subset.columns:
                vals = subset[metric].values
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals))
        row["n_folds"] = len(subset)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("rmse_mean")
    summary.to_csv(results_dir / "summary.csv", index=False)
    print(f"Summary -> {results_dir / 'summary.csv'}")
    print("\n" + summary.to_string(index=False))

    # Generate comparison plots
    try:
        from reports.visualizations import ReportGenerator
        gen = ReportGenerator(results_dir.parent / "reports" / "plots")

        model_rmse = {row["model"]: row["rmse_mean"] for row in summary_rows}
        gen.plot_model_comparison(model_rmse, metric_name="rmse")

        # Per-model CV plots
        for model in df["model_type"].unique():
            subset = df[df["model_type"] == model]
            fold_metrics = subset.to_dict("records")
            gen.plot_cv_rmse(fold_metrics, model_name=model)

        print(f"\nAll plots saved to {results_dir.parent / 'reports' / 'plots'}")
    except Exception as e:
        print(f"\nWarning: Could not generate plots: {e}")

    print("\n=== 5-fold CV complete ===")


if __name__ == "__main__":
    main()
