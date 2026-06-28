"""training/run_tree_models.py

Train XGB, LGB, CatBoost with Optuna tuning on competition data.
Produces OOF predictions and test predictions per fold.

Usage:
    cd polymer_competition
    python -m training.run_tree_models --config config.yaml --target tg
    python -m training.run_tree_models --config config.yaml --target egc
    python -m training.run_tree_models --config config.yaml --all
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score


def tune_model_optuna(model_type: str, X: np.ndarray, y: np.ndarray,
                      n_trials: int = 30, seed: int = 42) -> dict:
    """Find best hyperparameters via Optuna."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        if model_type == "xgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": seed,
            }
            from xgboost import XGBRegressor
            model = XGBRegressor(**params)
        elif model_type == "lgb":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": seed,
                "verbose": -1,
            }
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(**params)
        elif model_type == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 200, 1000),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                "random_seed": seed,
                "verbose": 0,
            }
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(**params)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_tree_fold(model_type: str, X_tr: np.ndarray, y_tr: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    best_params: dict, seed: int = 42):
    """Train a single tree model fold and return predictions."""
    if model_type == "xgb":
        from xgboost import XGBRegressor
        model = XGBRegressor(**best_params, random_state=seed)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    elif model_type == "lgb":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(**best_params, random_state=seed, verbose=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[])
    elif model_type == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(**best_params, random_seed=seed, verbose=0)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    else:
        raise ValueError(f"Unsupported: {model_type}")

    pred_val = model.predict(X_val)
    return model, pred_val


def run_tree_models(config_path: str = "config.yaml", target: str = None,
                    n_trials: int = 30, seed: int = 42):
    """Run tree models for a target (tg/egc or all)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_ver = cfg.get("experiment", {}).get("version", "v27")
    pred_dir = Path(cfg["paths"]["predictions_dir"])
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Load features (drop non-numeric columns)
    train_features = pd.read_parquet("data/processed/features_train.parquet")
    test_features = pd.read_parquet("data/processed/features_test.parquet")
    exclude_cols = {"canon_smiles", "SMILES"}
    feature_cols = [c for c in train_features.columns
                    if c not in exclude_cols and train_features[c].dtype != object]
    test_feature_cols = [c for c in feature_cols if c in test_features.columns]
    train_features_num = train_features[feature_cols]
    test_features_num = test_features[test_feature_cols]

    # Load train data for target splitting
    train_df = pd.read_csv("data/train.csv")

    # Load splits
    data_dir = Path(cfg["paths"]["data_dir"])
    targets = [target] if target else ["tg", "egc"]

    for tgt in targets:
        print(f"\n{'='*60}")
        print(f"  Target: {tgt}")
        print(f"{'='*60}")

        # Filter train by target
        mask = train_df["target_type"] == tgt
        tgt_indices = mask.values
        y_all = train_df.loc[mask, "target"].values
        X_all = train_features_num.loc[tgt_indices].values

        # Load test for this target
        target_test_csv = data_dir / tgt / "test.csv"
        if target_test_csv.exists():
            test_ids = pd.read_csv(target_test_csv)["id"].tolist()
            test_mask = test_features["id"].isin(test_ids).values
        else:
            test_mask = np.ones(len(test_features_num), dtype=bool)
        X_test = test_features_num.loc[test_mask].values
        n_test = X_test.shape[0]

        # Load CV splits
        splits_path = data_dir / f"splits_{tgt}.pkl"
        if not splits_path.exists():
            print(f"  Splits not found at {splits_path}. Generating...")
            from data.split_by_target import split_by_target
            split_by_target(data_dir / "train.csv", data_dir / "test.csv", data_dir, targets=[tgt])
            # Also generate scaffold splits
            from features.build_features import _smiles_scaffold
            from sklearn.model_selection import GroupKFold
            smiles_col = cfg.get("data", {}).get("smiles_col", "smiles")
            tgt_df = train_df[mask].reset_index(drop=True)
            scaffolds = tgt_df[smiles_col].apply(_smiles_scaffold).values
            gkf = GroupKFold(n_splits=cfg["cv"]["n_folds"])
            splits = {}
            for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(tgt_df, groups=scaffolds)):
                splits[fold_idx] = {"train": tr_idx.tolist(), "val": va_idx.tolist()}
            with open(splits_path, "wb") as f:
                pickle.dump(splits, f)
            print(f"  Generated {len(splits)} fold splits")

        with open(splits_path, "rb") as f:
            splits = pickle.load(f)

        models = ["xgb", "lgb", "catboost"]

        for model_type in models:
            print(f"\n  Training {model_type}...")
            t0 = time.time()

            oof_preds = np.zeros(len(y_all))
            test_preds = np.zeros(n_test)
            fold_r2s = []
            fold_params = []

            for fold_idx in range(len(splits)):
                train_idx = splits[fold_idx]["train"]
                val_idx = splits[fold_idx]["val"]

                X_tr, X_val = X_all[train_idx], X_all[val_idx]
                y_tr, y_val = y_all[train_idx], y_all[val_idx]

                # Optuna tuning
                print(f"    Fold {fold_idx}: tuning {n_trials} trials...", end=" ", flush=True)
                best_params = tune_model_optuna(model_type, X_tr, y_tr,
                                                n_trials=n_trials, seed=seed)
                fold_params.append(best_params)

                # Train with best params
                model, pred_val = train_tree_fold(model_type, X_tr, y_tr,
                                                  X_val, y_val, best_params, seed=seed)

                oof_preds[val_idx] = pred_val
                test_preds += model.predict(X_test) / len(splits)

                fold_r2 = r2_score(y_val, pred_val)
                fold_r2s.append(fold_r2)
                print(f"R² = {fold_r2:.4f}")

                # Save fold OOF prediction
                fold_pred = {
                    "target": tgt,
                    "model_type": model_type,
                    "fold": fold_idx,
                    "val_r2": fold_r2,
                    "pred": pred_val,
                    "y": y_val,
                    "val_idx": val_idx,
                }
                with open(pred_dir / f"{exp_ver}_{tgt}_{model_type}_fold{fold_idx}.pkl", "wb") as f:
                    pickle.dump(fold_pred, f)

            # Save final averaged test predictions (after all folds)
            test_pred = {
                "target": tgt,
                "model_type": model_type,
                "fold": -1,
                "pred": test_preds.tolist(),
                "id": test_features.loc[test_mask, "id"].values.tolist() if "id" in test_features.columns else list(range(n_test)),
            }
            with open(pred_dir / f"{exp_ver}_{tgt}_{model_type}_test.pkl", "wb") as f:
                pickle.dump(test_pred, f)

            # Summary
            mean_r2 = np.mean(fold_r2s)
            std_r2 = np.std(fold_r2s)
            elapsed = time.time() - t0
            print(f"\n  {model_type} Mean R²: {mean_r2:.4f} ± {std_r2:.4f} ({elapsed:.1f}s)")

            summary = {
                "target": tgt,
                "model_type": model_type,
                "mean_r2": float(mean_r2),
                "std_r2": float(std_r2),
                "fold_r2s": [float(r) for r in fold_r2s],
                "n_trials": n_trials,
                "best_params": fold_params,
            }
            with open(pred_dir / f"{exp_ver}_{tgt}_{model_type}_summary.json", "w") as f:
                json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--target", default=None, help="tg or egc (default: both)")
    parser.add_argument("--all", action="store_true", help="Train all targets")
    parser.add_argument("--n_trials", type=int, default=30, help="Optuna trials per fold")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    target = args.target if not args.all else None
    run_tree_models(args.config, target=target, n_trials=args.n_trials, seed=args.seed)
