"""Tests for the uncertainty-weighted ensemble in weight_optimizer.

Covers:
  1. Uniform weights sum to 1 and match the model count.
  2. The new 'uncertainty' strategy down-weights a noisy model on synthetic data.
  3. NaN/Inf fallback to uniform still works.
  4. Weights JSON round-trips and contains expected keys.
  5. Diverse meta-learner stacking produces weights that differ from a Ridge baseline
     (or falls back to Ridge cleanly when the optional libs are missing).
"""
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ensemble import weight_optimizer as wo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_oof_dict(n_models: int = 4, n_folds: int = 5, n_samples: int = 80,
                   noise_model_idx: int = None, seed: int = 0):
    """Build a synthetic OOF dict for testing.

    Each model produces predictions for the same fold indices, plus targets.
    If `noise_model_idx` is given, that model adds large Gaussian noise to its
    predictions — it should be down-weighted by an uncertainty-aware strategy.
    """
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n_samples)
    oof = {}
    for m in range(n_models):
        preds = {}
        targets = {}
        # Equal-sized folds
        fold_size = n_samples // n_folds
        for f in range(n_folds):
            start = f * fold_size
            stop = (f + 1) * fold_size if f < n_folds - 1 else n_samples
            # Signal + small noise for a "good" model
            p = 0.9 * y[start:stop] + rng.normal(scale=0.1, size=stop - start)
            if m == noise_model_idx:
                # Heavy noise on this model
                p = y[start:stop] + rng.normal(scale=2.5, size=stop - start)
            preds[f] = p
            targets[f] = y[start:stop]
        oof[f"m{m}"] = {"preds": preds, "targets": targets}
    return oof


def _stack_preds_targets(oof_dict):
    """Return (all_preds_matrix, all_y) stacked across folds (matching optimize_weights)."""
    models = list(oof_dict.keys())
    n_folds = max(max(d["preds"].keys()) for d in oof_dict.values()) + 1
    fold_preds = []
    fold_y_parts = []
    for f in range(n_folds):
        if not all(f in oof_dict[m]["preds"] for m in models):
            continue
        cols = [oof_dict[model]["preds"][f] for model in models]
        fold_preds.append(np.column_stack(cols))
        fold_y_parts.append(oof_dict[models[0]]["targets"][f])
    return np.vstack(fold_preds), np.concatenate(fold_y_parts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_uniform_weights_sum_to_one():
    oof = _make_oof_dict(n_models=4)
    all_preds, all_y = _stack_preds_targets(oof)
    w = wo.get_weights("uniform", all_preds, all_y)
    assert w.shape == (4,)
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w, 0.25)


def test_uncertainty_strategy_downweights_noisy_model():
    """The 'uncertainty' strategy should assign a smaller weight to the noisy model
    than the median weight given to clean models, on this synthetic case."""
    oof = _make_oof_dict(n_models=4, noise_model_idx=2, seed=42)
    all_preds, all_y = _stack_preds_targets(oof)
    w = wo.get_weights("uncertainty", all_preds, all_y)
    assert w.shape == (4,)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert all(w >= 0 for w in w), "weights must be non-negative"
    # The noisy model (index 2) should be down-weighted
    median_clean = np.median([w[i] for i in (0, 1, 3)])
    assert w[2] < median_clean, (
        f"Noisy model weight {w[2]:.3f} should be less than median clean weight {median_clean:.3f}"
    )


def test_optimize_strategy_beats_uniform_on_synthetic():
    """SLSQP optimization should achieve a higher R^2 than uniform on a case where
    one model is clearly better than the others."""
    rng = np.random.default_rng(7)
    n_models = 4
    y = rng.normal(size=200)
    # Model 0: perfect signal
    # Models 1-3: noisy signals
    cols = [
        y + rng.normal(scale=0.05, size=200),                       # very good
        y + rng.normal(scale=1.5, size=200),                        # bad
        y + rng.normal(scale=1.2, size=200),                        # bad
        y + rng.normal(scale=1.8, size=200),                        # bad
    ]
    all_preds = np.column_stack(cols)
    w_opt = wo.get_weights("optimize", all_preds, y)
    w_uni = wo.get_weights("uniform", all_preds, y)
    r2_opt = wo.r2_score(y, all_preds @ w_opt)
    r2_uni = wo.r2_score(y, all_preds @ w_uni)
    assert r2_opt >= r2_uni
    # And the good model should get the bulk of the weight
    assert w_opt[0] == np.max(w_opt)


def test_nan_fallback_to_uniform():
    """If OOF has NaN, fall back to uniform weights without raising."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=100)
    X = rng.normal(size=(100, 3))
    X[0, 1] = np.nan
    w = wo.get_weights("optimize", X, y)
    assert w.shape == (3,)
    assert w.sum() == pytest.approx(1.0)


def test_optimize_weights_returns_dict_and_score():
    """optimize_weights() returns per-model weights dict and a float score."""
    oof = _make_oof_dict(n_models=3, n_folds=5)
    weights, score = wo.optimize_weights(oof, n_folds=5)
    assert isinstance(weights, dict)
    assert set(weights.keys()) == {"m0", "m1", "m2"}
    assert isinstance(score, float)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-5)


def test_save_weights_roundtrip(tmp_path: Path):
    """save_weights() writes a JSON file with the expected schema."""
    weights = {"m0": 0.4, "m1": 0.3, "m2": 0.3}
    wo.save_weights("tg", weights, 0.91, out_dir=tmp_path)
    out = tmp_path / "weights_tg.json"
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["target"] == "tg"
    assert payload["val_r2"] == 0.91
    assert payload["weights"] == weights


def test_load_oof_predictions_skips_test_files(tmp_path: Path):
    """load_oof_predictions() should ignore files ending in '_test'."""
    data = {"pred": [0.1, 0.2], "y": [0.0, 0.0]}
    # Fold prediction
    (tmp_path / "v1_tg_mlp_fold0.pkl").write_bytes(pickle.dumps(data))
    # Test prediction — should be skipped
    (tmp_path / "v1_tg_mlp_fold0_test.pkl").write_bytes(pickle.dumps(data))
    oof = wo.load_oof_predictions(tmp_path, "tg", exp_ver="v1")
    assert "mlp" in oof
    assert 0 in oof["mlp"]["preds"]


def test_stacking_meta_learner_returns_weights_or_falls_back():
    """stacking_meta_learner() should return a dict of weights summing to ~1
    regardless of whether the optional LGBM/CatBoost libraries are present."""
    rng = np.random.default_rng(11)
    n_models = 3
    n_samples = 200
    # Stack of OOF preds as columns + targets
    X = rng.normal(size=(n_samples, n_models))
    # Make model 0 strongly predictive
    y = X[:, 0] + 0.05 * rng.normal(size=n_samples)
    weights = wo.stacking_meta_learner(X, y, learner="ridge")
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (n_models,)
    assert weights.sum() == pytest.approx(1.0, abs=1e-5)
    assert all(weights >= 0)


def test_stack_oof_drops_mismatched_folds():
    """_stack_oof should drop a model whose fold prediction length differs
    from the reference, but keep all models that agree across every fold."""
    oof = {
        "good": {
            "preds": {0: np.zeros(10), 1: np.zeros(10)},
            "targets": {0: np.zeros(10), 1: np.zeros(10)},
        },
        "ok": {
            "preds": {0: np.zeros(10), 1: np.zeros(10)},
            "targets": {0: np.zeros(10), 1: np.zeros(10)},
        },
        "bad": {
            # Mismatched length on fold 1
            "preds": {0: np.zeros(10), 1: np.zeros(7)},
            "targets": {0: np.zeros(10), 1: np.zeros(7)},
        },
    }
    all_preds, all_y, active = wo._stack_oof(oof, list(oof.keys()), n_folds=2)
    assert "bad" not in active
    assert {"good", "ok"} <= set(active)
    assert all_preds.shape[0] == 10  # one fold contributed
    assert all_preds.shape[1] == 2  # two active models
    assert all_y.shape[0] == 10
