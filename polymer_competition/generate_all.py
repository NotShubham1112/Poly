"""
generate_all.py

Master script that orchestrates the full pipeline:
    1. Generate CV splits
    2. Build feature matrix
    3. Train all model types (one at a time)
    4. Build ensemble and produce submission.csv
    5. Generate reports

Usage:
    python generate_all.py                     # full pipeline
    python generate_all.py --steps 1,2         # just splits + features
    python generate_all.py --steps 3 --models xgb,lgb,polychain
    python generate_all.py --steps 4,5         # ensemble + reports only

Designed for Kaggle/Colab: auto-resumes after disconnections via checkpoints.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# All model types available for training
ALL_MODEL_TYPES = [
    "ridge",
    "xgb",
    "lgb",
    "catboost",
    "rf",
    "mlp",
    "gcn",
    "gat",
    "mpnn",
    "graph_transformer",
    "polychain",
]

# Default person name (for prediction file naming)
DEFAULT_PERSON = "team"


def run_cmd(cmd: list[str], desc: str = "") -> int:
    """Run a command and stream output. Return exit code."""
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  ⚠ Command failed with exit code {result.returncode}")
    return result.returncode


def step_1_splits(config: str):
    """Generate cross-validation splits."""
    run_cmd(
        [sys.executable, "-m", "data.generate_splits", "--config", config],
        desc="Step 1: Generate CV Splits",
    )


def step_2_features(config: str):
    """Build the feature matrix (fingerprints + descriptors + custom)."""
    run_cmd(
        [sys.executable, "-m", "features.build_features", "--config", config],
        desc="Step 2: Build Feature Matrix",
    )


def step_3_train(config: str, models: list[str], person: str, n_folds: int):
    """Train all specified model types across all folds."""
    for model_type in models:
        for fold in range(n_folds):
            # Check if prediction already exists (resume support)
            pred_file = PROJECT_ROOT / "predictions" / f"{person}_{model_type}_fold{fold}.pkl"
            if pred_file.exists():
                print(f"  ✓ {pred_file.name} already exists — skipping.")
                continue

            # Find model-specific config if it exists
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

            rc = run_cmd(
                cmd,
                desc=f"Step 3: Train {model_type} (fold {fold}/{n_folds - 1})",
            )
            if rc != 0:
                print(f"  ⚠ Training {model_type} fold {fold} failed. Continuing...")


def step_4_ensemble(config: str):
    """Build the ensemble blend and produce submission.csv."""
    run_cmd(
        [sys.executable, "-m", "ensemble.build_ensemble", "--config", config],
        desc="Step 4: Build Ensemble → submission.csv",
    )


def step_5_reports(config: str):
    """Generate evaluation reports (SHAP, error analysis, summary)."""
    run_cmd(
        [sys.executable, "reports/generate_reports.py", "--config", config],
        desc="Step 5: Generate Reports",
    )
    # Also run EDA if train.csv exists
    if (PROJECT_ROOT / "data" / "train.csv").exists():
        run_cmd(
            [sys.executable, "notebooks/eda_report.py", "--config", config],
            desc="Step 5b: Run Automated EDA",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Master pipeline: splits → features → train → ensemble → reports"
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Path to global config")
    parser.add_argument("--steps", default="1,2,3,4,5",
                        help="Comma-separated step numbers to run (e.g., '1,2,3')")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model types for step 3 "
                             f"(default: all = {','.join(ALL_MODEL_TYPES)})")
    parser.add_argument("--person", default=DEFAULT_PERSON,
                        help="Team member name for prediction files")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds")
    args = parser.parse_args()

    steps = set(int(s.strip()) for s in args.steps.split(","))
    models = args.models.split(",") if args.models else ALL_MODEL_TYPES

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  POLYMER COMPETITION — FULL PIPELINE                       ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Config:  {args.config:<50}║")
    print(f"║  Steps:   {args.steps:<50}║")
    print(f"║  Models:  {','.join(models)[:50]:<50}║")
    print(f"║  Person:  {args.person:<50}║")
    print(f"║  Folds:   {args.n_folds:<50}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if 1 in steps:
        step_1_splits(args.config)
    if 2 in steps:
        step_2_features(args.config)
    if 3 in steps:
        step_3_train(args.config, models, args.person, args.n_folds)
    if 4 in steps:
        step_4_ensemble(args.config)
    if 5 in steps:
        step_5_reports(args.config)

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
