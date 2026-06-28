"""End-to-end local submission producer.

Combines OOF + test pickles into a Kaggle-format ``submission.csv`` with
columns ``id,target``. Steps per target:

  1. Load OOF predictions across every model × fold.
  2. Optimize blending weights with ``weight_optimizer.optimize_weights``
     (the same function used by ``weight_optimizer``'s CLI).
  3. Load matching test pickles for every surviving model.
  4. For each model, average predictions across folds (mean of 5 fold
     test predictions, since the test set is the same across folds for
     a given model).
  5. Blend per-model test predictions using the OOF-derived weights.
  6. Write per-target CSV and the combined ``submission.csv``.

Idempotent and seed-controlled: re-running with the same inputs produces
byte-identical output.

Usage:
    python run_submission.py                # both targets, optimize strategy
    python run_submission.py --target tg    # one target only
    python run_submission.py --strategy uncertainty
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ensemble import weight_optimizer as wo


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_test_predictions(pred_dir: Path, exp_ver: str, target: str,
                             models: list[str]) -> dict:
    """Load test pickles for ``models`` matching ``{exp_ver}_{target}_{m}_fold*_test.pkl``.

    Returns ``{model: {"ids": np.ndarray, "preds_mean": np.ndarray}}`` where
    ``preds_mean`` is the per-sample mean across folds for that model.
    Models with no test pickles are silently skipped.
    """
    out: dict[str, dict] = {}
    for model in models:
        per_fold_ids = []
        per_fold_preds = []
        for p in sorted(pred_dir.glob(f"{exp_ver}_{target}_{model}_fold*_test.pkl")):
            with open(p, "rb") as f:
                d = pickle.load(f)
            per_fold_ids.append(np.asarray(d["id"]))
            per_fold_preds.append(np.asarray(d["pred"], dtype=float))
        if not per_fold_preds:
            continue
        # All folds should predict on the same test set in the same id order;
        # verify and average across folds for robustness.
        ref_ids = per_fold_ids[0]
        consistent = all(np.array_equal(ref_ids, ids) for ids in per_fold_ids[1:])
        if not consistent:
            print(f"  WARNING: {model} test ids disagree across folds; "
                  f"using fold 0 ordering and averaging on aligned rows")
            # Align by id — for a single test set this is usually identical
            # across folds, but if not, intersection-average.
            common = set(map(int, ref_ids.tolist()))
            for ids in per_fold_ids[1:]:
                common &= set(map(int, ids.tolist()))
            common_ids = np.array(sorted(common))
            preds_aligned = []
            for ids, preds in zip(per_fold_ids, per_fold_preds):
                m = pd.Series(preds, index=ids).reindex(common_ids).values
                preds_aligned.append(m)
            mean_pred = np.nanmean(np.column_stack(preds_aligned), axis=1)
            out[model] = {"ids": common_ids, "preds_mean": mean_pred}
        else:
            stacked = np.column_stack(per_fold_preds)
            mean_pred = stacked.mean(axis=1)
            out[model] = {"ids": ref_ids, "preds_mean": mean_pred}
    return out


def _stack_test_preds(test_dict: dict, models: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-model mean test predictions into (ids, matrix) shape.

    Restricts to models present in ``test_dict`` and uses the union of
    ids (preserves order from the first model). Rows missing in any
    model get NaN — the caller decides how to handle them.
    """
    available = [m for m in models if m in test_dict]
    if not available:
        return np.zeros(0), np.zeros((0, 0))
    ids = test_dict[available[0]]["ids"]
    cols = [test_dict[m]["preds_mean"] for m in available]
    matrix = np.column_stack(cols)
    return ids, matrix


# ---------------------------------------------------------------------------
# Per-target blend
# ---------------------------------------------------------------------------
def blend_target(*, target: str, exp_ver: str, pred_dir: Path,
                 strategy: str) -> tuple[pd.DataFrame, dict]:
    """Produce per-target predictions for the full test set.

    Returns ``(df, info)`` where ``df`` has columns ``id,target`` (sorted by
    id) and ``info`` is a small dict with R², weights, and dropped models.
    """
    oof = wo.load_oof_predictions(pred_dir, target, exp_ver=exp_ver)
    if not oof:
        raise RuntimeError(f"No OOF predictions found for {target}")

    models_all = list(oof.keys())
    all_preds, all_y, active = wo._stack_oof(oof, models_all, n_folds=5)
    if all_preds is None:
        raise RuntimeError(f"No consistent OOF folds for {target}")

    # Get the weights (with the chosen strategy)
    w_vec = wo.get_weights(strategy, all_preds, all_y)
    weights = dict(zip(active, [float(v) for v in w_vec]))
    blend_r2 = float(wo.r2_score(all_y, all_preds @ w_vec))

    # Load matching test predictions (only for models that survived the OOF stack)
    test_dict = _load_test_predictions(pred_dir, exp_ver, target, list(weights.keys()))
    missing_in_test = [m for m in weights if m not in test_dict]
    if missing_in_test:
        print(f"  WARNING: OOF models {missing_in_test} have no test pickles; "
              f"redistributing their weight uniformly to surviving models")
        surviving = [m for m in weights if m in test_dict]
        if not surviving:
            raise RuntimeError(f"No test predictions available for {target}")
        # Move the missing model's weight into the others, then drop it
        extra = sum(weights[m] for m in missing_in_test)
        per_extra = extra / len(surviving)
        for m in surviving:
            weights[m] += per_extra
        for m in missing_in_test:
            del weights[m]

    # Build the test matrix and blend
    ids, matrix = _stack_test_preds(test_dict, list(weights.keys()))
    active_for_test = [m for m in weights if m in test_dict]
    w_for_test = np.array([weights[m] for m in active_for_test], dtype=float)
    # If some active models lack test preds but are present in weights dict,
    # _stack_test_preds has already skipped them — re-derive w_for_test
    # against the actual column count.
    if matrix.shape[1] != len(w_for_test):
        w_for_test = np.array([weights[m] for m in active_for_test], dtype=float)

    blended = matrix @ w_for_test

    df = pd.DataFrame({"id": ids.astype(int), "target": blended})
    df = df.sort_values("id").reset_index(drop=True)
    info = {
        "target": target,
        "n_models": len(weights),
        "weights": weights,
        "blend_val_r2": blend_r2,
        "n_test_rows": len(df),
        "missing_test_models": missing_in_test,
    }
    return df, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["tg", "egc", "all"], default="all")
    parser.add_argument("--strategy", default="optimize",
                        choices=["uniform", "optimize", "uncertainty"])
    parser.add_argument("--exp_ver", default=None)
    parser.add_argument("--out_dir", default=None,
                        help="Override submissions dir (default from config.yaml).")
    args = parser.parse_args()

    cfg = _load_config()
    exp_ver = args.exp_ver or cfg.get("experiment", {}).get("version", "v1")
    pred_dir = REPO_ROOT / cfg["paths"]["predictions_dir"]
    out_dir = (REPO_ROOT / args.out_dir) if args.out_dir else (
        REPO_ROOT / cfg["paths"]["submissions_dir"]
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = ["tg", "egc"] if args.target == "all" else [args.target]
    parts: list[pd.DataFrame] = []
    for t in targets:
        print(f"\n=== Blending {t} (strategy={args.strategy}) ===")
        df, info = blend_target(
            target=t, exp_ver=exp_ver, pred_dir=pred_dir, strategy=args.strategy,
        )
        out_csv = out_dir / f"{t}_blend.csv"
        df.to_csv(out_csv, index=False)
        print(f"  Saved {out_csv}  (n={info['n_test_rows']}, val_R²={info['blend_val_r2']:.4f})")
        for m, w in sorted(info["weights"].items(), key=lambda x: -x[1]):
            print(f"    {m}: {w:.3f}")
        parts.append(df)

    if args.target == "all":
        combined = pd.concat(parts, ignore_index=True).sort_values("id").reset_index(drop=True)
        # Final sanity checks
        if combined["target"].isna().any():
            n_nan = int(combined["target"].isna().sum())
            raise RuntimeError(f"submission has {n_nan} NaN predictions — refusing to write")
        if combined["id"].duplicated().any():
            n_dup = int(combined["id"].duplicated().sum())
            raise RuntimeError(f"submission has {n_dup} duplicate ids — refusing to write")
        sub_path = out_dir / "submission.csv"
        combined.to_csv(sub_path, index=False)
        print(f"\nFinal submission: {sub_path}  (n={len(combined)}, "
              f"id=[{combined['id'].min()}, {combined['id'].max()}])")


if __name__ == "__main__":
    main()