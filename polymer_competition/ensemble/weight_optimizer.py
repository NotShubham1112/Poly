"""Ensemble weight optimization for the polymer property prediction pipeline.

Provides three public strategies for combining OOF predictions from base models:
  * ``uniform``  — equal weights for every model.
  * ``optimize`` — SLSQP over R² (Ridge-style blending).
  * ``uncertainty`` — SLSQP starting from inverse-variance weights (Kendall-style
    uncertainty weighting on top of model disagreement across folds).

Also exposes :func:`stacking_meta_learner`, which fits a meta-model (Ridge by
default, with optional LightGBM/CatBoost) on stacked OOF predictions and
returns the resulting blend weights.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# OOF I/O
# ---------------------------------------------------------------------------
def load_oof_predictions(pred_dir, target, exp_ver="v1"):
    """Load out-of-fold predictions matching ``{exp_ver}_{target}_*_fold{k}.pkl``.

    Files ending in ``_test`` (test-set predictions) are ignored.
    """
    import pickle
    pred_dir = Path(pred_dir)
    models = []
    oof_dict = {}
    for pkl_path in sorted(pred_dir.glob(f"{exp_ver}_{target}_*_fold*.pkl")):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if pkl_path.stem.endswith("_test"):
            continue
        stem = pkl_path.stem  # e.g. v29_tg_mlp_5seed_fold0
        # Remove the exp_ver + target prefix to get model+seed+fold
        suffix = stem[len(f"{exp_ver}_{target}_"):]  # e.g. mlp_5seed_fold0
        # Find the fold number: last segment starting with "fold"
        segs = suffix.split("_")
        fold_part = next(s for s in reversed(segs) if s.startswith("fold"))
        fold = int(fold_part.replace("fold", ""))
        # Model name is everything before the seed/fold segments
        fold_idx = segs.index(fold_part)
        model = "_".join(segs[:fold_idx])
        if "pred" not in data or "y" not in data:
            continue
        if model not in oof_dict:
            oof_dict[model] = {"preds": {}, "targets": {}}
        oof_dict[model]["preds"][fold] = np.array(data["pred"])
        oof_dict[model]["targets"][fold] = np.array(data["y"])
    return oof_dict


def get_weights(strategy: str, oof: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Get ensemble weights per strategy.

    Strategies:
      * ``uniform`` / ``equal`` — equal weights summing to 1.
      * ``optimize`` — SLSQP over R² (default).
      * ``uncertainty`` — SLSQP warm-started from inverse-variance weights.

    Falls back to uniform on NaN/Inf or on optimization failure.
    """
    n = oof.shape[1]
    if strategy in ("uniform", "equal"):
        return np.ones(n) / n
    if np.any(np.isnan(oof)) or np.any(np.isinf(oof)):
        print("WARNING: NaN/Inf in OOF matrix, falling back to uniform weights")
        return np.ones(n) / n

    if strategy == "uncertainty":
        x0 = _inverse_variance_prior(oof)
    else:
        x0 = np.ones(n) / n

    bounds = [(0, 1)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    result = minimize(objective, x0, args=(oof, y),
                      method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-8})
    if not result.success:
        print(f"WARNING: Weight optimization failed ({result.message}), using uniform weights")
        return np.ones(n) / n
    return result.x


def _inverse_variance_prior(oof: np.ndarray) -> np.ndarray:
    """Return a normalized prior weight vector inversely proportional to each
    model's OOF prediction variance (Kendall-style uncertainty weighting).

    Lower-disagreement models get higher prior weight. A small floor is added
    to prevent any single weight from collapsing to zero before SLSQP refines.
    """
    # Per-model dispersion across OOF predictions (vector spread, not error).
    var = np.var(oof, axis=0) + 1e-8
    inv = 1.0 / var
    w = inv / inv.sum()
    # Mix with uniform so no model is fully excluded before optimization.
    n = oof.shape[1]
    return 0.5 * w + 0.5 * (np.ones(n) / n)


def model_disagreement(oof: np.ndarray) -> np.ndarray:
    """Diagnostic: per-model OOF prediction variance (model disagreement).

    Returns a 1-D array of length ``oof.shape[1]``. Higher values indicate
    the model's predictions fluctuate more across samples, which the
    ``uncertainty`` strategy interprets as lower confidence.
    """
    return np.var(oof, axis=0)


# ---------------------------------------------------------------------------
# Metrics / objectives
# ---------------------------------------------------------------------------
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def objective(weights, preds_matrix, y_true):
    blended = preds_matrix @ weights
    return -r2_score(y_true, blended)


def optimize_weights(oof_dict, n_folds=5):
    """Optimize blending weights by SLSQP over stacked OOF predictions.

    Uses :func:`_stack_oof` to build a strictly consistent stacked matrix;
    models that disagree on per-fold lengths are dropped. Returns
    ``(weights_dict, val_r2)`` or ``(None, None)`` when the OOF data is
    insufficient.
    """
    models = list(oof_dict.keys())
    if not models:
        return None, None
    all_preds, all_y, active = _stack_oof(oof_dict, models, n_folds)
    if all_preds is None:
        return None, None
    n = all_preds.shape[1]
    x0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    result = minimize(objective, x0, args=(all_preds, all_y),
                      method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-8})
    best_r2 = -result.fun
    weights = result.x
    return dict(zip(active, [float(w) for w in weights])), float(f"{best_r2:.4f}")


# ---------------------------------------------------------------------------
# Stacking meta-learner
# ---------------------------------------------------------------------------
def stacking_meta_learner(oof_matrix: np.ndarray, y: np.ndarray,
                          learner: str = "ridge", **kwargs) -> np.ndarray:
    """Train a meta-learner on stacked OOF predictions and return blending weights.

    The default ``ridge`` learner yields a closed-form weight vector; ``lgbm``
    and ``catboost`` fall back to ridge if the optional libraries are absent.
    The returned weights are normalized to sum to 1 and clipped at zero.
    """
    n_models = oof_matrix.shape[1]
    if learner == "ridge":
        return _ridge_weights(oof_matrix, y, alpha=kwargs.get("alpha", 1.0))
    if learner in ("lgbm", "lightgbm"):
        try:
            from lightgbm import LGBMRegressor
            m = LGBMRegressor(n_estimators=kwargs.get("n_estimators", 200),
                              learning_rate=kwargs.get("learning_rate", 0.05),
                              num_leaves=kwargs.get("num_leaves", 15),
                              min_child_samples=kwargs.get("min_child_samples", 20),
                              random_state=42, verbose=-1)
            m.fit(oof_matrix, y)
            return _normalize(m.feature_importances_)
        except ImportError:
            print("WARNING: lightgbm not available, falling back to ridge meta-learner")
            return _ridge_weights(oof_matrix, y)
    if learner == "catboost":
        try:
            from catboost import CatBoostRegressor
            m = CatBoostRegressor(iterations=kwargs.get("iterations", 200),
                                   depth=kwargs.get("depth", 4),
                                   learning_rate=kwargs.get("learning_rate", 0.05),
                                   random_seed=42, verbose=0)
            m.fit(oof_matrix, y)
            return _normalize(m.get_feature_importance())
        except ImportError:
            print("WARNING: catboost not available, falling back to ridge meta-learner")
            return _ridge_weights(oof_matrix, y)
    raise ValueError(f"Unknown stacking meta-learner: {learner}")


def _ridge_weights(oof_matrix: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Closed-form Ridge regression weights, normalized to sum to 1."""
    from sklearn.linear_model import Ridge
    m = Ridge(alpha=alpha, fit_intercept=False)
    m.fit(oof_matrix, y)
    w = np.clip(m.coef_, 0, None)
    s = w.sum()
    if s <= 0:
        return np.ones(oof_matrix.shape[1]) / oof_matrix.shape[1]
    return w / s


def _normalize(w: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(w, dtype=float), 0, None)
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / len(w)
    return w / s


# ---------------------------------------------------------------------------
# Persistence / CLI
# ---------------------------------------------------------------------------
def save_weights(target, weights, score, out_dir="ensembles", strategy: str = None):
    """Persist ensemble weights + score to JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"weights_{target}.json"
    payload = {"target": target, "weights": weights, "val_r2": score}
    if strategy is not None:
        payload["strategy"] = strategy
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved weights -> {out_path}")


def _stack_oof(oof_dict, models, n_folds):
    """Stack OOF preds and targets across folds, requiring consistent per-fold
    lengths across the surviving models.

    Uses a two-pass approach:
      1. For each fold, identify the modal prediction length; discard models
         whose length disagrees with the mode. Models with no entry for that
         fold are kept but ignored for that fold.
      2. Stack the surviving models' predictions across folds that have at
         least one surviving model with data.

    Returns ``(all_preds, all_y, active_models)`` or ``(None, None, [])``
    when no fold can be stacked.
    """
    models = list(models)
    if not models:
        return None, None, []

    # Pass 1: for each fold, find the modal length and drop models that disagree.
    surviving = set(models)
    for fold in range(n_folds):
        # Find the modal length across all models that have this fold.
        lengths = []
        for m in models:
            if m in surviving and fold in oof_dict[m]["preds"]:
                lengths.append(len(np.asarray(oof_dict[m]["preds"][fold])))
        if not lengths:
            continue
        # Modal length = most common value; ties broken by first occurrence.
        from collections import Counter
        mode_len, _ = Counter(lengths).most_common(1)[0]
        # Discard models that disagree with the modal length for this fold.
        for m in list(surviving):
            if fold not in oof_dict[m]["preds"]:
                continue
            if len(np.asarray(oof_dict[m]["preds"][fold])) != mode_len:
                surviving.discard(m)

    if not surviving:
        return None, None, []

    # Restrict to models with data for *every* fold so the stacked matrix
    # has uniform column count across folds.
    n_folds_actual = n_folds
    surviving = {m for m in surviving
                 if all(f in oof_dict[m]["preds"] and f in oof_dict[m]["targets"]
                        for f in range(n_folds_actual))}

    if not surviving:
        return None, None, []

    # Pass 2: identify folds where every surviving model has data of equal length.
    consistent_folds = []
    for fold in range(n_folds):
        first = next((m for m in models if m in surviving
                      and fold in oof_dict[m]["preds"]), None)
        if first is None:
            continue
        ref_len = len(np.asarray(oof_dict[first]["preds"][fold]))
        # All surviving models with data for this fold must agree.
        all_agree = all(
            fold not in oof_dict[m]["preds"]
            or len(np.asarray(oof_dict[m]["preds"][fold])) == ref_len
            for m in models if m in surviving
        )
        if all_agree:
            consistent_folds.append(fold)

    if not consistent_folds:
        return None, None, []

    # Pass 3: stack surviving models on consistent folds only
    all_preds = []
    all_y = []
    for fold in consistent_folds:
        # Pick a surviving model that actually has data for this fold,
        # so we can use its targets and predictions as the column references.
        first = next(m for m in models if m in surviving
                     and fold in oof_dict[m]["preds"]
                     and fold in oof_dict[m]["targets"])
        ref_y = np.asarray(oof_dict[first]["targets"][fold])
        cols = [np.asarray(oof_dict[m]["preds"][fold]) for m in models
                if m in surviving and fold in oof_dict[m]["preds"]]
        all_preds.append(np.column_stack(cols))
        all_y.append(ref_y)
    active = [m for m in models if m in surviving]
    return np.vstack(all_preds), np.concatenate(all_y), active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--predictions_dir", default="predictions")
    parser.add_argument("--exp_ver", default="v1")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--strategy", default="optimize",
                        choices=["uniform", "optimize", "uncertainty"])
    args = parser.parse_args()

    targets = [args.target] if args.target else (["tg", "egc"] if args.all else ["tg", "egc"])
    for target in targets:
        print(f"\n=== Optimizing weights for {target} (strategy={args.strategy}) ===")
        oof = load_oof_predictions(args.predictions_dir, target, args.exp_ver)
        if not oof:
            print(f"  No OOF predictions found for {target}")
            continue
        print(f"  Models found: {list(oof.keys())}")
        models = list(oof.keys())
        all_preds, all_y, active = _stack_oof(oof, models, args.n_folds)
        if all_preds is None:
            print(f"  Could not optimize (insufficient data)")
            continue
        if len(active) < len(models):
            print(f"  Dropping inconsistent models: "
                  f"{set(models) - set(active)}")
        if args.strategy == "uniform":
            w = get_weights("uniform", all_preds, all_y)
        else:
            w = get_weights(args.strategy, all_preds, all_y)
        weights = dict(zip(active, [float(v) for v in w]))
        score = float(f"{r2_score(all_y, all_preds @ w):.4f}")
        print(f"  Best R²: {score}")
        for model, w in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"    {model}: {w:.3f}")
        save_weights(target, weights, score, strategy=args.strategy)


if __name__ == "__main__":
    main()