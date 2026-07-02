"""
training/run_seeded_ensemble.py

Train PolyChain with N seeds + XGBoost, build weighted ensemble.
Supports resume — re-run to skip completed folds.

Usage:
    python -m training.run_seeded_ensemble --config config.yaml --n_seeds 5
    python -m training.run_seeded_ensemble --n_seeds 3 --folds 1 --targets tg --skip_ensemble
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEEDS = [42, 123, 456, 789, 101112]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gpu_memory_status() -> str:
    """Return GPU memory usage string, or empty if no CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return ""
        i = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        res = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        return f"GPU: {alloc:.2f}/{total:.2f} GB allocated, {res:.2f} GB cached"
    except Exception:
        return ""


def prediction_exists(pred_dir: Path, exp_ver: str, target: str,
                       model_type_key: str, fold: int) -> bool:
    pattern = f"{exp_ver}_{target}_{model_type_key}_fold{fold}.pkl"
    return any(pred_dir.glob(pattern))


def checkpoint_exists(ckpt_dir: Path, exp_ver: str, target: str,
                       seed_suffix: str, model_type: str, fold: int,
                       person: str) -> bool:
    ckpt_tag = f"{exp_ver}_{target}{seed_suffix}_{model_type}_fold{fold}"
    return (ckpt_dir / f"{ckpt_tag}_final.pt").exists()


def run_cmd(cmd: list[str], desc: str = "") -> int:
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    mem = gpu_memory_status()
    if mem:
        print(f"  {mem}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"  WARNING: Command failed with exit code {result.returncode}")
    return result.returncode


def build_commands(args, cfg, targets, seeds):
    """Build list of (cmd, description) tuples, skipping completed runs."""
    exp_ver = cfg.get("experiment", {}).get("version", "v28")
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    ckpt_dir = Path(cfg["paths"].get("checkpoints_dir", "outputs/checkpoints/"))

    commands = []
    total = 0
    skipped = 0

    if not args.skip_polychain:
        for target in targets:
            for seed in seeds:
                seed_suffix = f"_s{seed}"
                model_type_key = f"polychain_boosted{seed_suffix}"
                all_done = True
                for fold in range(args.folds):
                    if prediction_exists(pred_dir, exp_ver, target, model_type_key, fold):
                        skipped += 1
                        continue
                    all_done = False
                    cmd = [
                        sys.executable, "-m", "training.train",
                        "--model_type", "polychain_boosted",
                        "--fold", str(fold),
                        "--config", args.config,
                        "--target", target,
                        "--seed", str(seed),
                        "--epochs", str(args.epochs),
                        "--person", f"boosted_s{seed}",
                    ]
                    if checkpoint_exists(ckpt_dir, exp_ver, target, seed_suffix,
                                          "polychain_boosted", fold, f"boosted_s{seed}"):
                        cmd += ["--resume"]
                    commands.append((cmd, f"PolyChain boosted seed={seed} {target} fold {fold}/{args.folds - 1}"))
                    total += 1
                if all_done:
                    pass  # progress bar will handle this

    if not args.skip_xgb:
        for target in targets:
            for fold in range(args.folds):
                if prediction_exists(pred_dir, exp_ver, target, "xgb", fold):
                    skipped += 1
                    continue
                cmd = [
                    sys.executable, "-m", "training.train",
                    "--model_type", "xgb",
                    "--fold", str(fold),
                    "--config", args.config,
                    "--target", target,
                    "--person", "boosted_xgb",
                ]
                commands.append((cmd, f"XGBoost {target} fold {fold}/{args.folds - 1}"))
                total += 1

    return commands, total, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed PolyChain + XGBoost ensemble training with progress bar"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Number of PolyChain seeds to train")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--targets", default="tg,egc",
                        help="Comma-separated target types")
    parser.add_argument("--skip_xgb", action="store_true",
                        help="Skip XGBoost training")
    parser.add_argument("--skip_polychain", action="store_true",
                        help="Skip PolyChain training")
    parser.add_argument("--skip_ensemble", action="store_true",
                        help="Skip ensemble building")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Training epochs per model")
    args = parser.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    targets = args.targets.split(",") if args.targets else ["tg", "egc"]
    seeds = SEEDS[:args.n_seeds]

    print("=" * 60)
    print("  POLYCHAIN SEEDED ENSEMBLE")
    print(f"  Seeds:    {seeds}")
    print(f"  Targets:  {targets}")
    print(f"  Folds:    {args.folds}")
    print(f"  Config:   {args.config}")
    mem = gpu_memory_status()
    if mem:
        print(f"  {mem}")
    print("=" * 60)

    commands, total, skipped = build_commands(args, cfg, targets, seeds)

    if args.dry_run:
        print(f"\n=== DRY RUN: {total} pending + {skipped} already done ===")
        for cmd, desc in commands:
            print(f"\n  [{desc}]")
            print(f"  $ {' '.join(cmd)}")
        return

    if total == 0:
        print(f"\n✓ All {total + skipped} runs already complete. Skipping to ensemble.")
    else:
        if skipped > 0:
            print(f"\n{skipped} runs already complete, {total} remaining\n")

        if tqdm is not None:
            pbar = tqdm(total=total, desc="Ensemble", unit="run", smoothing=0.1)
        else:
            pbar = None
            print(f"  Tip: pip install tqdm for a live progress bar")

        start_time = time.time()
        failures = 0

        for i, (cmd, desc) in enumerate(commands):
            if pbar is not None:
                pbar.set_postfix_str(desc[-60:], refresh=False)
                pbar.update(0)  # refresh display

            rc = run_cmd(cmd, desc=desc)

            if rc != 0:
                failures += 1
                print(f"  !! FAILED: {desc}")

            if pbar is not None:
                elapsed = time.time() - start_time
                done = i + 1
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (total - done) / rate if rate > 0 else 0
                pbar.set_postfix({
                    "done": f"{done}/{total}",
                    "fail": failures,
                    "ETA": f"{remaining/60:.0f}m" if remaining < 3600 else f"{remaining/3600:.1f}h",
                })
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        elapsed = time.time() - start_time
        print(f"\n  Training complete: {total} runs in {elapsed/60:.0f}m"
              f" ({failures} failed)")

    # --- Ensemble ---
    if not args.skip_ensemble:
        for target in targets:
            cmd = [
                sys.executable, "-m", "ensemble.build_ensemble",
                "--config", args.config,
                "--target", target,
            ]
            run_cmd(cmd, desc=f"Build Ensemble for {target}")

        run_cmd(
            [sys.executable, "-m", "data.merge_submissions",
             "--config", args.config],
            desc="Merge tg + egc submissions",
        )

    print("\n" + "=" * 60)
    print("  SEEDED ENSEMBLE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Re-run the same command to resume — completed folds will be skipped.")
        sys.exit(1)
