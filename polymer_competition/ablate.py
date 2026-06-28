"""Ablation harness for the v28 ensemble.

Measures the contribution of each base model to the OOF blend by repeatedly
removing one model group at a time and re-optimizing the weights. The
resulting delta-R² is the marginal contribution of that group.

Output: a CSV table sorted by impact (largest negative delta = most
important model; positive delta = the model is hurting the ensemble).

This is *measurement*, not estimation. Run it before adding new models so
you don't compound noise.

Usage:
    python ablate.py --target tg
    python ablate.py --target egc
    python ablate.py --target all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ensemble import weight_optimizer as wo


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _stack_for_models(oof_dict, models, n_folds):
    """Restrict the OOF dict to ``models`` and run _stack_oof."""
    sub = {m: oof_dict[m] for m in models if m in oof_dict}
    return wo._stack_oof(sub, list(sub.keys()), n_folds)


def _r2_for_subset(oof_dict, models, n_folds):
    """Compute the optimized-weights R² for a subset of models."""
    all_preds, all_y, active = _stack_for_models(oof_dict, models, n_folds)
    if all_preds is None or len(active) < 2:
        return None
    weights, score = wo.optimize_weights(
        {m: oof_dict[m] for m in active}, n_folds=n_folds,
    )
    return float(score) if weights else None


def ablate_target(oof_dict, n_folds: int) -> pd.DataFrame:
    """Return a DataFrame of one-row-per-dropped-model ablation."""
    models = list(oof_dict.keys())
    baseline = _r2_for_subset(oof_dict, models, n_folds)
    rows = []
    for m in models:
        # Drop just this model
        keep = [k for k in models if k != m]
        r2_drop_one = _r2_for_subset(oof_dict, keep, n_folds)
        # Also: drop only this model (force 0 weight) — same effect as above
        # so we don't need a separate column.
        if r2_drop_one is None or baseline is None:
            delta = None
        else:
            delta = r2_drop_one - baseline
        rows.append({
            "model": m,
            "r2_full": round(baseline, 4) if baseline is not None else None,
            "r2_without": round(r2_drop_one, 4) if r2_drop_one is not None else None,
            "delta_r2": round(delta, 4) if delta is not None else None,
        })
    df = pd.DataFrame(rows).sort_values("delta_r2", ascending=False, na_position="last")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["tg", "egc", "all"], default="all")
    parser.add_argument("--exp_ver", default=None,
                        help="Override experiment version (default from config.yaml).")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--out", default="reports/ablation_v28.csv",
                        help="Output CSV path (single-target) or base path.")
    args = parser.parse_args()

    cfg = _load_config()
    exp_ver = args.exp_ver or cfg.get("experiment", {}).get("version", "v1")
    pred_dir = REPO_ROOT / cfg["paths"]["predictions_dir"]
    out_path = REPO_ROOT / args.out

    targets = ["tg", "egc"] if args.target == "all" else [args.target]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for target in targets:
        oof = wo.load_oof_predictions(pred_dir, target, exp_ver=exp_ver)
        if not oof:
            print(f"[{target}] no OOF predictions found under {pred_dir}")
            continue
        print(f"\n=== Ablating {target} ({len(oof)} models) ===")
        df = ablate_target(oof, args.n_folds)
        if df.empty:
            print(f"[{target}] no consistent models to ablate")
            continue
        print(df.to_string(index=False))

        # Per-target output path when --target all
        if args.target == "all":
            target_out = out_path.with_name(
                f"{out_path.stem}_{target}{out_path.suffix}"
            )
        else:
            target_out = out_path
        df.to_csv(target_out, index=False)
        print(f"  Saved -> {target_out}")


if __name__ == "__main__":
    main()