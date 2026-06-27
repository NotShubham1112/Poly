"""
models/tree_models.py
Tree-based regressors: RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost.
"""
from __future__ import annotations

import numpy as np


def get_tree_model(model_type: str, **kwargs):
    """Factory for tree-based models.

    Parameters
    ----------
    model_type : 'rf', 'et', 'xgb', 'lgb', 'catboost'
    """
    if model_type in ("rf", "randomforest"):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 500),
            max_depth=kwargs.get("max_depth", None),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            n_jobs=kwargs.get("n_jobs", -1),
            random_state=kwargs.get("random_state", 42),
        )
    if model_type in ("et", "extratrees"):
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(
            n_estimators=kwargs.get("n_estimators", 500),
            max_depth=kwargs.get("max_depth", None),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            n_jobs=kwargs.get("n_jobs", -1),
            random_state=kwargs.get("random_state", 42),
        )
    if model_type == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=kwargs.get("n_estimators", 2000),
            learning_rate=kwargs.get("learning_rate", 0.05),
            max_depth=kwargs.get("max_depth", 6),
            subsample=kwargs.get("subsample", 0.8),
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),
            reg_alpha=kwargs.get("reg_alpha", 0.0),
            reg_lambda=kwargs.get("reg_lambda", 1.0),
            tree_method=kwargs.get("tree_method", "hist"),
            device=kwargs.get("device", "cuda"),
            random_state=kwargs.get("random_state", 42),
            early_stopping_rounds=kwargs.get("early_stopping_rounds", 50),
            eval_metric=kwargs.get("eval_metric", "rmse"),
        )
    if model_type == "lgb":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=kwargs.get("n_estimators", 2000),
            learning_rate=kwargs.get("learning_rate", 0.05),
            num_leaves=kwargs.get("num_leaves", 31),
            max_depth=kwargs.get("max_depth", -1),
            subsample=kwargs.get("subsample", 0.8),
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),
            reg_alpha=kwargs.get("reg_alpha", 0.0),
            reg_lambda=kwargs.get("reg_lambda", 1.0),
            random_state=kwargs.get("random_state", 42),
            n_jobs=kwargs.get("n_jobs", -1),
        )
    if model_type == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=kwargs.get("iterations", 2000),
            learning_rate=kwargs.get("learning_rate", 0.05),
            depth=kwargs.get("depth", 6),
            l2_leaf_reg=kwargs.get("l2_leaf_reg", 3.0),
            random_seed=kwargs.get("random_state", 42),
            loss_function=kwargs.get("loss_function", "RMSE"),
            early_stopping_rounds=kwargs.get("early_stopping_rounds", 50),
            task_type=kwargs.get("task_type", "CPU"),
            verbose=False,
        )
    raise ValueError(f"Unknown tree model: {model_type}")
