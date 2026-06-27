from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score


def load_oof_predictions(pred_dir: Path, exp: str, target: str) -> pd.DataFrame:
    rows = []
    for pkl_file in pred_dir.glob(f"{exp}_{target}_*_fold*.pkl"):
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


def handle_nan(oof: np.ndarray, y: np.ndarray, model_names: list[str],
               impute: bool = True) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if impute:
        col_mean = np.nanmean(oof, axis=0)
        inds = np.where(np.isnan(oof))
        oof = oof.copy()
        oof[inds] = np.take(col_mean, inds[1])
        return oof, y, model_names
    valid = ~np.any(np.isnan(oof), axis=0)
    if not np.any(valid):
        return oof, y, model_names
    return oof[:, valid], y, [m for m, v in zip(model_names, valid) if v]


def load_test_predictions(pred_dir: Path, exp: str, target: str) -> pd.DataFrame:
    rows = []
    for pkl_file in pred_dir.glob(f"{exp}_{target}_*_test.pkl"):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        ids = np.asarray(data["id"])
        preds = np.asarray(data["pred"])
        for i, p in enumerate(preds):
            rows.append({
                "id": int(ids[i]),
                "pred": float(p),
                "model_type": data.get("model_type", "unknown"),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--target", required=True, help="Target name (tg/egc)")
    parser.add_argument("--meta", default="xgb", choices=["xgb", "ridge"],
                        help="Meta-model type")
    parser.add_argument("--exp", default=None, help="Experiment version override")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pred_dir = Path(cfg["paths"]["predictions_dir"])
    sub_dir = Path(cfg["paths"]["submissions_dir"])
    sub_dir.mkdir(parents=True, exist_ok=True)
    ensemble_dir = Path("ensembles")
    ensemble_dir.mkdir(exist_ok=True)

    exp = args.exp or cfg.get("experiment", {}).get("version", "v1")
    target = args.target

    df = load_oof_predictions(pred_dir, exp, target)
    if len(df) == 0:
        print(f"No OOF predictions found for {exp}_{target}_*_fold*.pkl. Skipping.")
        return
    print(f"Loaded {len(df)} rows from {df['model_type'].nunique()} model types for target={target}")

    oof, y, model_names = build_oof_matrix(df)
    oof, y, model_names = handle_nan(oof, y, model_names, impute=True)
    if len(model_names) == 0:
        print(f"No models with complete OOF predictions for target={target}. Skipping.")
        return
    print(f"OOF matrix: {oof.shape} with models: {model_names}")

    if args.meta == "xgb":
        from xgboost import XGBRegressor
        meta = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    else:
        from sklearn.linear_model import RidgeCV
        meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(meta, oof, y, cv=cv, scoring="r2")
    for i, score in enumerate(cv_scores):
        print(f"  Fold {i} R²: {score:.4f}")
    print(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    meta.fit(oof, y)
    train_r2 = r2_score(y, meta.predict(oof))
    print(f"Train R² (full OOF): {train_r2:.4f}")

    model_path = ensemble_dir / f"stacking_{target}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"meta": meta, "model_names": model_names, "meta_type": args.meta}, f)
    print(f"Meta-model saved -> {model_path}")

    test_df = load_test_predictions(pred_dir, exp, target)
    if len(test_df) == 0:
        print(f"No test predictions found for {exp}_{target}_*_test.pkl. Skipping submission.")
        return

    test_pivot = test_df.groupby(["id", "model_type"])["pred"].mean().unstack()
    test_pivot = test_pivot[[m for m in model_names if m in test_pivot.columns]]
    if len(test_pivot.columns) == 0:
        print(f"No test predictions for models {model_names}. Skipping submission.")
        return

    if test_pivot.isna().any().any():
        print("Warning: NaN values in test predictions; filling with column mean.")
        test_pivot = test_pivot.fillna(test_pivot.mean())

    test_preds = meta.predict(test_pivot.values)
    submission = pd.DataFrame({"id": test_pivot.index, "target": test_preds})
    sub_path = sub_dir / f"{exp}_{target}_stacking.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Submission for {target} -> {sub_path}")
    print(f"CV R²: {cv_scores.mean():.4f}")


if __name__ == "__main__":
    main()
