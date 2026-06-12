"""
models/baselines.py
Linear / Ridge / Lasso regression wrappers.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


def get_linear_model(model_type: str = "ridge", **kwargs):
    """Factory for linear baselines.

    Parameters
    ----------
    model_type : 'linear', 'ridge', 'lasso', 'elasticnet'
    """
    if model_type == "linear":
        return LinearRegression(**kwargs)
    if model_type == "ridge":
        return Ridge(alpha=kwargs.pop("alpha", 1.0), **kwargs)
    if model_type == "lasso":
        return Lasso(alpha=kwargs.pop("alpha", 1e-3), **kwargs)
    if model_type == "elasticnet":
        return ElasticNet(alpha=kwargs.pop("alpha", 1e-3),
                          l1_ratio=kwargs.pop("l1_ratio", 0.5),
                          **kwargs)
    raise ValueError(f"Unknown linear model: {model_type}")
