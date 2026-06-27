"""
training/tune.py
Hyperparameter optimization for tree-based models using Optuna.

Usage:
    python -m training.tune --model xgb --target tg --n_trials 20
    python -m training.tune --all  # tune all models for all targets
"""

import argparse
import json
import yaml
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def load_data(target=None):
    """Load features and merge target values."""
    train_feat = pd.read_parquet(DATA_DIR / "processed" / "features_train.parquet")
    full_train = pd.read_csv(DATA_DIR / "train.csv")
    if "target" not in train_feat.columns and len(train_feat) == len(full_train):
        train_feat["target"] = full_train["target"].values
        train_feat["target_type"] = full_train["target_type"].values
    if target:
        train_feat = train_feat[train_feat["target_type"] == target].reset_index(drop=True)
    return train_feat

def get_X_y(df, target_col="target"):
    feature_cols = [c for c in df.columns if c not in ("SMILES", "id", "canon_smiles", "target_type", "target")]
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values
    return X, y, feature_cols


def tune_xgb(X, y, n_trials=20):
    import optuna
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBRegressor

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            "random_state": 42,
        }
        model = XGBRegressor(**params, verbosity=0)
        scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name="xgb")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def tune_lgb(X, y, n_trials=20):
    import optuna
    from sklearn.model_selection import cross_val_score
    from lightgbm import LGBMRegressor

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            "random_state": 42,
            "verbosity": -1,
        }
        model = LGBMRegressor(**params)
        scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name="lgb")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def tune_catboost(X, y, n_trials=20):
    import optuna
    from sklearn.model_selection import cross_val_score
    from catboost import CatBoostRegressor

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000, step=50),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0, 10),
            "random_seed": 42,
            "verbose": False,
        }
        model = CatBoostRegressor(**params)
        scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name="catboost")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


TUNERS = {
    "xgb": tune_xgb,
    "lgb": tune_lgb,
    "catboost": tune_catboost,
}


def save_params(model, target, params, score):
    out_dir = PROJECT_ROOT / "training" / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model}_tuned.yaml"
    existing = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = yaml.safe_load(f) or {}
    existing[target] = {"params": params, "cv_r2": float(f"{score:.4f}")}
    with open(out_path, "w") as f:
        yaml.dump(existing, f, default_flow_style=False)
    print(f"  Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Model to tune: xgb, lgb, catboost")
    parser.add_argument("--target", default=None, help="Target: tg, egc")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--all", action="store_true", help="Tune all models for all targets")
    args = parser.parse_args()

    targets = [args.target] if args.target else ["tg", "egc"]
    models = list(TUNERS.keys()) if args.all else ([args.model] if args.model else list(TUNERS.keys()))

    for target in targets:
        print(f"\n=== Target: {target} ===")
        df = load_data(target)
        X, y, _ = get_X_y(df)
        print(f"  Samples: {len(df)}, Features: {X.shape[1]}")
        for model_name in models:
            print(f"\n  Tuning {model_name} ({args.n_trials} trials)...")
            try:
                best_params, best_score = TUNERS[model_name](X, y, n_trials=args.n_trials)
                print(f"  Best CV R2: {best_score:.4f}")
                print(f"  Best params: {best_params}")
                save_params(model_name, target, best_params, best_score)
            except Exception as e:
                print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
