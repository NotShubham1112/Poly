from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


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


def select_best_meta(oof: np.ndarray, y: np.ndarray,
                     model_names: list[str]) -> tuple[Any, str, float]:
    from catboost import CatBoostRegressor
    from sklearn.linear_model import RidgeCV, ElasticNetCV
    from xgboost import XGBRegressor

    candidates = {
        "ridge": RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
        "xgb": XGBRegressor(n_estimators=300, max_depth=3,
                            learning_rate=0.1, random_state=42,
                            n_jobs=-1),
        "catboost": CatBoostRegressor(iterations=500, depth=3,
                                       learning_rate=0.1, verbose=0,
                                       random_seed=42),
        "elasticnet": ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1.0],
                                    alphas=[0.01, 0.1, 1.0, 10.0],
                                    max_iter=5000, random_state=42),
    }

    best_score, best_name, best_model = -np.inf, None, None
    results = {}
    for name, meta in candidates.items():
        scores = cross_val_score(meta, oof, y, cv=min(5, len(oof) // 2),
                                  scoring="r2", n_jobs=-1)
        mean_score = scores.mean()
        results[name] = mean_score
        print(f"  {name}: CV R² = {mean_score:.4f} (std={scores.std():.4f})")
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_model = meta

    print(f"  Best: {best_name} (R² = {best_score:.4f})")
    best_model.fit(oof, y)
    return best_model, best_name, best_score


def try_stage2_stacking(oof: np.ndarray, y: np.ndarray,
                         stage1_model, stage1_score: float,
                         model_names: list[str]) -> tuple[Any, float]:
    """Train a second-level model if it improves CV by >0.002."""
    from sklearn.linear_model import RidgeCV
    from xgboost import XGBRegressor

    # Level 1 predictions
    oof_l1 = stage1_model.predict(oof).reshape(-1, 1)

    # Level 2: Ridge on L1 predictions + original features
    oof_l2 = np.concatenate([oof, oof_l1], axis=1)
    l2_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    l2_score = cross_val_score(l2_model, oof_l2, y, cv=min(5, len(oof) // 2),
                                scoring="r2", n_jobs=-1).mean()

    print(f"  Stage-2 CV R² = {l2_score:.4f} (improvement: {l2_score - stage1_score:.4f})")
    if l2_score > stage1_score + 0.002:
        print("  -> Using Stage-2 stacking")
        l2_model.fit(oof_l2, y)
        return l2_model, l2_score

    print("  -> Stage-2 not beneficial, keeping Stage-1")
    return None, stage1_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--target", required=True, help="Target name (tg/egc)")
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

    meta_model, meta_name, meta_score = select_best_meta(oof, y, model_names)
    train_r2 = r2_score(y, meta_model.predict(oof))
    print(f"Train R² (full OOF): {train_r2:.4f}")

    # Stage-2 stacking (conditional)
    stage2_model, final_score = try_stage2_stacking(oof, y, meta_model, meta_score, model_names)

    # Save meta-model info
    model_path = ensemble_dir / f"stacking_{target}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "meta": meta_model,
            "meta_name": meta_name,
            "stage2": stage2_model,
            "cv_score": final_score,
            "model_names": model_names,
        }, f)
    print(f"Meta-model saved -> {model_path}")

    test_df = load_test_predictions(pred_dir, exp, target)
    if len(test_df) == 0:
        print(f"No test predictions found for {exp}_{target}_*_test.pkl. Skipping submission.")
        return

    test_pivot = test_df.groupby(["id", "model_type"])["pred"].mean().unstack()
    available_models = [m for m in model_names if m in test_pivot.columns]
    if len(available_models) == 0:
        print(f"No test predictions for models {model_names}. Skipping submission.")
        return
    if len(available_models) < len(model_names):
        missing = set(model_names) - set(available_models)
        print(f"Warning: missing test predictions for {missing}. Retraining OOF with {available_models}")
        # Retrain meta-model with only the models available in test
        avail_idx = [i for i, m in enumerate(model_names) if m in available_models]
        oof = oof[:, avail_idx]
        model_names = available_models
        meta_model, meta_name, meta_score = select_best_meta(oof, y, model_names)
        stage2_model, final_score = try_stage2_stacking(oof, y, meta_model, meta_score, model_names)
    test_pivot = test_pivot[available_models]

    if test_pivot.isna().any().any():
        print("Warning: NaN values in test predictions; filling with column mean.")
        test_pivot = test_pivot.fillna(test_pivot.mean())

    if stage2_model is not None:
        # Stage-2: L1 predictions + original features
        l1_preds = meta_model.predict(test_pivot.values).reshape(-1, 1)
        test_input = np.concatenate([test_pivot.values, l1_preds], axis=1)
        test_preds = stage2_model.predict(test_input)
    else:
        test_preds = meta_model.predict(test_pivot.values)

    submission = pd.DataFrame({"id": test_pivot.index, "target": test_preds})
    sub_path = sub_dir / f"{exp}_{target}_stacking.csv"
    submission.to_csv(sub_path, index=False)
    print(f"Submission for {target} -> {sub_path}")
    print(f"CV R²: {final_score:.4f}")


if __name__ == "__main__":
    main()
