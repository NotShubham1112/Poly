from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .weight_optimizer import get_weights


def load_predictions(pred_dir: Path, exp: str, target: str) -> pd.DataFrame:
    rows = []
    pattern = f"{exp}_{target}_*.pkl"
    for pkl_file in pred_dir.glob(pattern):
        if pkl_file.stem.endswith("_test"):
            continue
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
            })
    return pd.DataFrame(rows)


def build_oof_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    grouped = df.groupby(["idx", "model_type"])["pred"].mean().unstack()
    y = df.groupby("idx")["y"].first().reindex(grouped.index)
    return grouped.values, y.values, list(grouped.columns)


def _drop_nan_models(oof: np.ndarray, y: np.ndarray, model_names: list[str]
                     ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    valid = ~np.any(np.isnan(oof), axis=0)
    if not np.any(valid):
        return oof, y, model_names
    return oof[:, valid], y, [m for m, v in zip(model_names, valid) if v]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--target", required=True, help="Target name (tg/egc)")
    parser.add_argument("--strategy", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pred_dir = Path(cfg["paths"]["predictions_dir"])
    sub_dir = Path(cfg["paths"]["submissions_dir"])
    sub_dir.mkdir(parents=True, exist_ok=True)

    exp = cfg.get("experiment", {}).get("version", "v1")
    target = args.target

    df = load_predictions(pred_dir, exp, target)
    if len(df) == 0:
        print(f"No OOF predictions found for {exp}_{target}_*.pkl. Skipping.")
        return
    print(f"Loaded {len(df)} rows from {df['model_type'].nunique()} model types for target={target}")

    oof, y, model_names = build_oof_matrix(df)
    oof, y, model_names = _drop_nan_models(oof, y, model_names)
    if len(model_names) == 0:
        print(f"No models with complete OOF predictions for target={target}. Skipping.")
        return
    print(f"Using {len(model_names)} models with complete predictions: {model_names}")

    w = get_weights(args.strategy or cfg["ensemble"]["strategy"], oof, y)
    print(f"Weights ({target}): {dict(zip(model_names, w.round(4)))}")

    weight_dir = Path("ensembles")
    weight_dir.mkdir(exist_ok=True)
    weight_path = weight_dir / f"{exp}_{target}_weights.json"
    with open(weight_path, "w") as f:
        json.dump({
            "experiment": exp,
            "target": target,
            "strategy": args.strategy or cfg["ensemble"]["strategy"],
            "weights": dict(zip(model_names, w.round(4))),
            "cv_score": float(np.sqrt(np.mean((oof @ w - y) ** 2))),
        }, f, indent=2)
    print(f"Weights saved -> {weight_path}")

    test_pattern = f"{exp}_{target}_*_test.pkl"
    test_rows = []
    for pkl_file in pred_dir.glob(test_pattern):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        for i, p in enumerate(np.asarray(data["pred"])):
            test_rows.append({
                "id": int(np.asarray(data["id"])[i]),
                "pred": float(p),
                "model_type": data.get("model_type", "unknown"),
            })
    test_df = pd.DataFrame(test_rows)
    test_pivot = test_df.groupby(["id", "model_type"])["pred"].mean().unstack()
    # Align test columns with the models used in OOF weight optimization
    test_pivot = test_pivot[[m for m in model_names if m in test_pivot.columns]]
    if len(test_pivot.columns) == 0:
        print(f"No test predictions for models {model_names}. Skipping submission.")
        return
    test_blend = test_pivot.values @ w
    submission = pd.DataFrame({
        "id": test_pivot.index,
        "target": test_blend,
    })
    sub_path = sub_dir / f"{target}_preds.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Submission for {target} -> {sub_path}")


if __name__ == "__main__":
    main()
