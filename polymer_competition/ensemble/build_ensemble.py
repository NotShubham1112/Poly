"""
ensemble.build_ensemble.py

Collect all per-fold OOF .pkl files from predictions/, blend them,
and produce a final submission.csv.

Usage:
    python -m ensemble.build_ensemble --config config.yaml
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .weight_optimizer import get_weights


def load_predictions(pred_dir: Path) -> pd.DataFrame:
    """Read all .pkl files in pred_dir and concatenate into a wide DataFrame.

    Returns a DataFrame with columns: idx, y, model, pred, fold, person, metrics.
    """
    rows = []
    for pkl_file in pred_dir.glob("*.pkl"):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        val_idx = np.asarray(data["val_idx"])
        preds = np.asarray(data["pred"])
        y = np.asarray(data["y"])
        for i, (idx, p, t) in enumerate(zip(val_idx, preds, y)):
            rows.append({
                "idx": int(idx),
                "y": float(t),
                "pred": float(p),
                "model_type": data.get("model_type", "unknown"),
                "fold": int(data.get("fold", 0)),
                "person": data.get("person", "anon"),
                "file": pkl_file.name,
            })
    return pd.DataFrame(rows)


def build_oof_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pivot to (n_samples, n_models) matrices.

    Returns
    -------
    oof_matrix : (n_samples, n_models) – mean prediction per (idx, model).
    y          : (n_samples,)          – ground truth.
    model_names : list[str]           – column names.
    """
    grouped = df.groupby(["idx", "model_type"])["pred"].mean().unstack()
    y = df.groupby("idx")["y"].first().reindex(grouped.index)
    return grouped.values, y.values, list(grouped.columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--strategy", default=None,
                        help="Override ensemble strategy from config.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pred_dir = Path(cfg["paths"]["predictions_dir"])
    sub_dir = Path(cfg["paths"]["submissions_dir"])
    sub_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from {pred_dir} ...")
    df = load_predictions(pred_dir)
    print(f"  Found {len(df)} rows from {df['file'].nunique()} files, "
          f"{df['model_type'].nunique()} model types.")

    # Pivot to OOF matrix
    oof, y, model_names = build_oof_matrix(df)
    print(f"  OOF matrix shape: {oof.shape}")

    # Compute weights
    strategy = args.strategy or cfg.get("ensemble", {}).get("strategy", "inverse_rmse")
    print(f"Optimizing weights with strategy='{strategy}' ...")
    w = get_weights(strategy, oof, y)
    print(f"  Weights: {dict(zip(model_names, w.round(4)))}")

    # Blended OOF score
    blended = oof @ w
    rmse = float(np.sqrt(np.mean((blended - y) ** 2)))
    mae = float(np.mean(np.abs(blended - y)))
    print(f"  Blended OOF RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    # Per-model OOF metrics for the report
    per_model = {}
    for j, name in enumerate(model_names):
        p = oof[:, j]
        per_model[name] = {
            "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "mae":  float(np.mean(np.abs(p - y))),
        }

    # ------------------------------------------------------------------
    # Generate test predictions
    # ------------------------------------------------------------------
    test_pred_files = list(pred_dir.glob("*test*.pkl"))
    if not test_pred_files:
        print("No test prediction files found (*test*.pkl). Skipping submission.")
    else:
        # Average across folds per (model, person)
        test_rows = []
        for pkl_file in test_pred_files:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            for i, p in enumerate(np.asarray(data["pred"])):
                test_rows.append({
                    "id": int(np.asarray(data["id"])[i]),
                    "pred": float(p),
                    "model_type": data.get("model_type", "unknown"),
                    "person": data.get("person", "anon"),
                })
        test_df = pd.DataFrame(test_rows)
        test_pivot = test_df.groupby(["id", "model_type"])["pred"].mean().unstack()
        test_blend = test_pivot.values @ w
        submission = pd.DataFrame({
            "id": test_pivot.index,
            cfg["target"]["column"]: test_blend,
        })
        sub_path = sub_dir / "submission.csv"
        submission.to_csv(sub_path, index=False)
        print(f"  Submission saved -> {sub_path}")

    # ------------------------------------------------------------------
    # Save model summary
    # ------------------------------------------------------------------
    summary_path = Path("reports/model_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, m in per_model.items():
        rows.append({"model": name, "rmse": m["rmse"], "mae": m["mae"],
                     "weight": float(w[model_names.index(name)])})
    rows.append({"model": "BLEND", "rmse": rmse, "mae": mae, "weight": 1.0})
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"  Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
