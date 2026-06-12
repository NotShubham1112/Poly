"""
ensemble.weight_optimizer.py
Strategies for combining OOF predictions from multiple models.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge


def inverse_rmse_weights(oof_preds: np.ndarray, y_true: np.ndarray,
                          min_w: float = 0.0, max_w: float = 1.0) -> np.ndarray:
    """Weights proportional to 1/RMSE per model, then normalized.

    Parameters
    ----------
    oof_preds : (n_samples, n_models)
    y_true    : (n_samples,)
    """
    rmses = np.sqrt(np.mean((oof_preds - y_true[:, None]) ** 2, axis=0))
    inv = 1.0 / (rmses + 1e-8)
    w = inv / inv.sum()
    w = np.clip(w, min_w, max_w)
    w = w / w.sum()
    return w


def nelder_mead_weights(oof_preds: np.ndarray, y_true: np.ndarray,
                        min_w: float = 0.0, max_w: float = 1.0) -> np.ndarray:
    """Find non-negative weights minimizing RMSE of the weighted blend."""
    n_models = oof_preds.shape[1]

    def loss(w):
        w = np.clip(w, 0, None)
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        return np.sqrt(np.mean((oof_preds @ w - y_true) ** 2))

    x0 = np.ones(n_models) / n_models
    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 5000})
    w = np.clip(res.x, min_w, max_w)
    w = w / w.sum()
    return w


def stacking_ridge(oof_preds: np.ndarray, y_true: np.ndarray,
                   alpha: float = 1.0) -> np.ndarray:
    """Fit a Ridge meta-learner on OOF predictions; return its coefficients."""
    ridge = Ridge(alpha=alpha, positive=True)
    ridge.fit(oof_preds, y_true)
    w = ridge.coef_
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    return w


def get_weights(strategy: str, oof_preds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    if strategy == "inverse_rmse":
        return inverse_rmse_weights(oof_preds, y_true)
    if strategy == "nelder_mead":
        return nelder_mead_weights(oof_preds, y_true)
    if strategy == "stacking":
        return stacking_ridge(oof_preds, y_true)
    raise ValueError(f"Unknown strategy: {strategy}")
