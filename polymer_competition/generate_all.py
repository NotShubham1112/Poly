from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent

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

ALL_TARGETS = ["tg", "egc"]
DEFAULT_PERSON = "team"


def load_exp_ver(config: str) -> str:
    with open(config) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("experiment", {}).get("version", "v1")


def run_cmd(cmd: list[str], desc: str = "") -> int:
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  WARNING: Command failed with exit code {result.returncode}")
    return result.returncode


def step_1_splits(config: str, targets: list[str]):
    for target in targets:
        run_cmd(
            [sys.executable, "-m", "data.generate_splits",
             "--config", config, "--target", target],
            desc=f"Step 1: Generate CV Splits for {target}",
        )


def step_2_features(config: str):
    run_cmd(
        [sys.executable, "-m", "features.build_features", "--config", config],
        desc="Step 2: Build Feature Matrix",
    )


def step_3_train(config: str, models: list[str], person: str, n_folds: int, targets: list[str]):
    exp_ver = load_exp_ver(config)
    for target in targets:
        for model_type in models:
            for fold in range(n_folds):
                pred_file = PROJECT_ROOT / "predictions" / f"{exp_ver}_{target}_{model_type}_fold{fold}.pkl"
                if pred_file.exists():
                    print(f"  SKIP: {pred_file.name} already exists.")
                    continue

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
                    "--target", target,
                ]
                if model_cfg:
                    cmd += ["--model_config", model_cfg]

                rc = run_cmd(
                    cmd,
                    desc=f"Step 3: Train {model_type} {target} (fold {fold}/{n_folds - 1})",
                )
                if rc != 0:
                    print(f"  WARNING: Training {model_type} {target} fold {fold} failed.")


def step_4_ensemble(config: str, targets: list[str]):
    for target in targets:
        run_cmd(
            [sys.executable, "-m", "ensemble.build_ensemble",
             "--config", config, "--target", target],
            desc=f"Step 4: Build Ensemble for {target}",
        )
    run_cmd(
        [sys.executable, "-m", "data.merge_submissions",
         "--config", config],
        desc="Step 4b: Merge submissions",
    )


def step_5_reports(config: str):
    run_cmd(
        [sys.executable, "reports/generate_reports.py", "--config", config],
        desc="Step 5: Generate Reports",
    )
    if (PROJECT_ROOT / "data" / "train.csv").exists():
        run_cmd(
            [sys.executable, "notebooks/eda_report.py", "--config", config],
            desc="Step 5b: Run Automated EDA",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Master pipeline: splits -> features -> train -> ensemble -> reports"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--steps", default="1,2,3,4,5",
                        help="Comma-separated step numbers")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model types "
                             f"(default: all = {','.join(ALL_MODEL_TYPES)})")
    parser.add_argument("--person", default=DEFAULT_PERSON)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--targets", default="tg,egc",
                        help="Comma-separated target types")
    args = parser.parse_args()

    steps = set(int(s.strip()) for s in args.steps.split(","))
    models = args.models.split(",") if args.models else ALL_MODEL_TYPES
    targets = args.targets.split(",") if args.targets else ALL_TARGETS

    print("============================================================")
    print("  POLYMER COMPETITION - FULL PIPELINE")
    print("============================================================")
    print(f"  Config:  {args.config}")
    print(f"  Steps:   {args.steps}")
    print(f"  Models:  {','.join(models)}")
    print(f"  Person:  {args.person}")
    print(f"  Folds:   {args.n_folds}")
    print(f"  Targets: {','.join(targets)}")
    print("============================================================")

    if 1 in steps:
        step_1_splits(args.config, targets)
    if 2 in steps:
        step_2_features(args.config)
    if 3 in steps:
        step_3_train(args.config, models, args.person, args.n_folds, targets)
    if 4 in steps:
        step_4_ensemble(args.config, targets)
    if 5 in steps:
        step_5_reports(args.config)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
