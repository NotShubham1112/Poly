"""Drive multi-task training across all 5 folds.

The ``--multitask`` CLI flag wired in 3.2 trains one fold per invocation.
This wrapper shells out five times so the full OOF + test pickles are
produced with proper epochs (default 80, configurable).

Re-runs are idempotent: any fold whose OOF pickle already exists is skipped.
This makes interrupted runs safe to retry.

Usage:
    python run_multitask.py                # all 5 folds, default 80 epochs
    python run_multitask.py --epochs 30    # shorter run for ablation studies
    python run_multitask.py --fold 0       # single fold
    python run_multitask.py --force        # overwrite existing OOFs
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.yaml"
LOG_DIR = REPO_ROOT / "outputs" / "logs"
LOG_PATH = LOG_DIR / "multitask_run.log"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _oof_pickle_path(predictions_dir: Path, exp_ver: str, target: str, fold: int) -> Path:
    return predictions_dir / f"{exp_ver}_{target}_multitask_fold{fold}.pkl"


def _run_fold(cfg: dict, fold: int, epochs: int, force: bool) -> bool:
    """Train multitask on one fold. Returns True if a run actually happened."""
    predictions_dir = REPO_ROOT / cfg["paths"]["predictions_dir"]
    exp_ver = cfg.get("experiment", {}).get("version", "v1")
    target = cfg["data"]["target_col"]  # unused for routing; multitask writes both targets

    # Skip if both tg and egc OOFs already exist for this fold.
    have_tg = _oof_pickle_path(predictions_dir, exp_ver, "tg", fold).exists()
    have_egc = _oof_pickle_path(predictions_dir, exp_ver, "egc", fold).exists()
    if not force and have_tg and have_egc:
        print(f"[fold {fold}] OOFs already exist, skipping (use --force to overwrite)")
        return False

    cmd = [
        sys.executable, "-m", "training.train",
        "--model_type", "multitask",
        "--fold", str(fold),
        "--target", "tg",
        "--epochs", str(epochs),
        "--config", str(CONFIG_PATH),
        "--person", "multitask_runner",
        "--auto_save_every", "0",
    ]
    print(f"[fold {fold}] running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[fold {fold}] FAILED after {elapsed:.1f}s (exit={result.returncode})")
        return False
    print(f"[fold {fold}] OK in {elapsed:.1f}s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs (default: config.yaml multitask.epochs or 80).")
    parser.add_argument("--fold", type=int, default=None,
                        help="Single fold to run (default: all 5).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing OOF pickles.")
    args = parser.parse_args()

    cfg = _load_config()
    epochs = (args.epochs
              or cfg.get("multitask", {}).get("epochs")
              or 80)
    n_folds = cfg.get("cv", {}).get("n_folds", 5)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Multitask training driver — {n_folds} folds × {epochs} epochs each")
    print(f"Log file: {LOG_PATH}")

    folds = [args.fold] if args.fold is not None else list(range(n_folds))
    n_ran = 0
    for fold in folds:
        if _run_fold(cfg, fold, epochs, args.force):
            n_ran += 1
    print(f"\nDone. Ran {n_ran}/{len(folds)} folds.")


if __name__ == "__main__":
    main()