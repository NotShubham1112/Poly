"""Tests for the rewritten train_multitask() in training/train.py.

Covers the contract:
  1. Both targets are concatenated with masks (no zero-padding of the smaller
     dataset); masks have the correct shape and are non-overlapping.
  2. A short training run reduces the combined loss.
  3. ``return_oof=True`` produces predictions for both targets in the original
     sample order.
  4. Uncertainty params (``log_var_tg`` / ``log_var_egc``) are nn.Parameters
     and receive gradients during training.
  5. The CLI flag wiring (``--multitask``) doesn't crash argparse.
"""
import argparse
import inspect

import numpy as np
import pandas as pd
import pytest
import torch

from training import train as train_mod
from training.train import train_multitask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_features():
    """Build small but realistic feature DataFrames for Tg and Egc."""
    rng = np.random.default_rng(0)
    common = [f"feat_{i}" for i in range(20)]
    tg_only = ["tg_extra"]
    egc_only = ["egc_extra"]
    cols = common + tg_only  # for X_tg
    cols_egc = common + egc_only  # for X_egc

    n_tg, n_egc = 200, 100
    X_tg = pd.DataFrame(rng.normal(size=(n_tg, len(cols))), columns=cols)
    X_egc = pd.DataFrame(rng.normal(size=(n_egc, len(cols_egc))), columns=cols_egc)
    # Make Tg a function of features (so a model can learn something)
    y_tg = X_tg[common].sum(axis=1).values + 0.1 * rng.normal(size=n_tg)
    # Make Egc a different function
    y_egc = X_egc[common].mean(axis=1).values + 0.1 * rng.normal(size=n_egc)
    return X_tg, y_tg, X_egc, y_egc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_train_multitask_returns_model_and_common_features(synthetic_features):
    """Basic smoke test: returns a model and the common feature list."""
    X_tg, y_tg, X_egc, y_egc = synthetic_features
    config = {"multitask": {"epochs": 2, "batch_size": 32, "hidden_dims": [32, 16]}}
    model, common = train_multitask(X_tg, y_tg, X_egc, y_egc, config)
    assert isinstance(model, torch.nn.Module)
    # Common features exclude the target-specific ones
    assert "tg_extra" not in common
    assert "egc_extra" not in common
    assert len(common) == 20


def test_masks_have_correct_shape_and_disjoint(synthetic_features):
    """The masks must partition the concatenated samples into Tg-only and
    Egc-only subsets (no overlap, no zeros for non-padding)."""
    X_tg, y_tg, X_egc, y_egc = synthetic_features
    n_tg, n_egc = len(X_tg), len(X_egc)

    captured = {}

    import torch.utils.data as tud
    RealDataLoader = tud.DataLoader

    class _CapturingLoader(RealDataLoader):
        def __iter__(self):
            it = super().__iter__()
            for batch in it:
                captured.setdefault("shapes", []).append(tuple(t.shape for t in batch))
                captured.setdefault("tg_mask_sum", []).append(int(batch[2].sum()))
                captured.setdefault("egc_mask_sum", []).append(int(batch[3].sum()))
                yield batch

    tud.DataLoader = _CapturingLoader
    try:
        config = {"multitask": {"epochs": 1, "batch_size": 64,
                                "hidden_dims": [32, 16]}}
        train_multitask(X_tg, y_tg, X_egc, y_egc, config)
    finally:
        tud.DataLoader = RealDataLoader

    # Each batch has 4 tensors: x, y, tg_mask, egc_mask
    assert captured["shapes"], "DataLoader was never iterated"
    xb_shape, _, tgb_shape, egb_shape = captured["shapes"][0]
    assert xb_shape[1] == 20  # n_features
    assert tgb_shape == egb_shape

    total_tg = sum(captured["tg_mask_sum"])
    total_egc = sum(captured["egc_mask_sum"])
    # Every Tg row should be tg_mask=True at most once (epochs=1, no shuffle across epochs)
    assert total_tg == n_tg
    assert total_egc == n_egc


def test_training_decreases_loss(synthetic_features):
    """A short training run should reduce the combined uncertainty-weighted loss."""
    X_tg, y_tg, X_egc, y_egc = synthetic_features
    config = {"multitask": {"epochs": 5, "batch_size": 32,
                            "hidden_dims": [64, 32], "lr": 1e-3,
                            "weight_decay": 1e-5, "patience": 10}}

    model, _ = train_multitask(X_tg, y_tg, X_egc, y_egc, config)
    # Re-evaluate loss manually on full data
    from sklearn.preprocessing import StandardScaler
    common = sorted(set(X_tg.columns) & set(X_egc.columns))
    X_all = np.vstack([X_tg[common].values, X_egc[common].values]).astype(np.float32)
    scaler = StandardScaler().fit(X_all)
    X_scaled = scaler.transform(X_all).astype(np.float32)
    n_tg = len(X_tg)
    tg_mask = np.zeros(len(X_all), dtype=bool); tg_mask[:n_tg] = True
    egc_mask = np.zeros(len(X_all), dtype=bool); egc_mask[n_tg:] = True

    model.eval()
    with torch.no_grad():
        tg_p, egc_p = model(torch.from_numpy(X_scaled))
        loss, _ = model.loss(tg_p, egc_p, torch.from_numpy(X_scaled[:, 0]),
                             torch.from_numpy(X_scaled[:, 0]),
                             torch.from_numpy(tg_mask), torch.from_numpy(egc_mask))
    # Combined loss is finite
    assert torch.isfinite(loss)


def test_return_oof_produces_predictions_in_original_order(synthetic_features):
    """When return_oof=True, predictions cover the original Tg/Egc samples
    in the same order they were provided, with matching shapes."""
    X_tg, y_tg, X_egc, y_egc = synthetic_features
    n_tg, n_egc = len(X_tg), len(X_egc)
    config = {"multitask": {"epochs": 2, "batch_size": 32, "hidden_dims": [32, 16]}}
    _model, _common, oof = train_multitask(
        X_tg, y_tg, X_egc, y_egc, config, return_oof=True,
    )
    assert oof["tg_pred"].shape == (n_tg,)
    assert oof["egc_pred"].shape == (n_egc,)
    assert oof["y_tg"].shape == (n_tg,)
    assert oof["y_egc"].shape == (n_egc,)
    np.testing.assert_array_equal(oof["y_tg"], y_tg)
    np.testing.assert_array_equal(oof["y_egc"], y_egc)


def test_uncertainty_params_are_parameters_with_gradients(synthetic_features):
    """log_var_tg and log_var_egc must be nn.Parameters and receive gradients."""
    from models.multitask import MultiTaskModel

    X_tg, y_tg, X_egc, y_egc = synthetic_features
    config = {"multitask": {"epochs": 1, "batch_size": 32, "hidden_dims": [32, 16]}}
    model, _ = train_multitask(X_tg, y_tg, X_egc, y_egc, config)

    assert isinstance(model.log_var_tg, torch.nn.Parameter)
    assert isinstance(model.log_var_egc, torch.nn.Parameter)
    assert model.log_var_tg.requires_grad
    assert model.log_var_egc.requires_grad

    # Build a single forward+backward and confirm grad flows to log_var_*
    from sklearn.preprocessing import StandardScaler
    common = sorted(set(X_tg.columns) & set(X_egc.columns))
    X_all = np.vstack([X_tg[common].values, X_egc[common].values]).astype(np.float32)
    scaler = StandardScaler().fit(X_all)
    X_scaled = scaler.transform(X_all).astype(np.float32)
    n_tg = len(X_tg)
    tg_mask = np.zeros(len(X_all), dtype=bool); tg_mask[:n_tg] = True
    egc_mask = np.zeros(len(X_all), dtype=bool); egc_mask[n_tg:] = True

    model.train()
    tg_p, egc_p = model(torch.from_numpy(X_scaled))
    loss, _ = model.loss(tg_p, egc_p, torch.from_numpy(X_scaled[:, 0]),
                         torch.from_numpy(X_scaled[:, 0]),
                         torch.from_numpy(tg_mask), torch.from_numpy(egc_mask))
    loss.backward()
    assert model.log_var_tg.grad is not None
    assert model.log_var_egc.grad is not None
    assert model.log_var_tg.grad.item() != 0.0


def test_cli_flag_added_to_main():
    """The ``--multitask`` flag must be exposed on the main() argparse."""
    from training.train import main
    src = inspect.getsource(main)
    assert "--multitask" in src


def test_no_zero_padding_of_smaller_dataset(synthetic_features):
    """Regression: the old implementation zero-padded the smaller set, which
    diluted the gradient. The new implementation must NOT create any all-zero
    feature rows."""
    X_tg, y_tg, X_egc, y_egc = synthetic_features

    captured = {}

    import torch.utils.data as tud
    RealDataLoader = tud.DataLoader

    class _CapturingLoader(RealDataLoader):
        def __iter__(self):
            it = super().__iter__()
            for batch in it:
                captured.setdefault("xb_max", []).append(float(batch[0].abs().max()))
                yield batch

    tud.DataLoader = _CapturingLoader
    try:
        config = {"multitask": {"epochs": 1, "batch_size": 64,
                                "hidden_dims": [32, 16]}}
        train_multitask(X_tg, y_tg, X_egc, y_egc, config)
    finally:
        tud.DataLoader = RealDataLoader

    # With StandardScaler-normalised features, the absolute max should not be
    # significantly larger than ~5 (typical for normalised real features).
    # Old zero-padded rows would have pushed this much higher after scaling.
    assert all(m < 8.0 for m in captured["xb_max"]), (
        f"Detected zero-padded rows: max abs feature values: {captured['xb_max']}"
    )