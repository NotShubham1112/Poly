import pytest
import numpy as np
from features.target_transforms import (
    boxcox_transform,
    quantile_transform,
    log_transform,
    yeo_johnson_transform,
    rank_gauss_transform,
    select_best_transform,
)
from training.train import apply_target_transform


def test_boxcox_transform():
    y = np.array([100, 150, 200, 250, 300, 350, 400], dtype=float)
    y_transformed, inv_func = boxcox_transform(y)
    
    assert y_transformed.shape == y.shape
    assert not np.any(np.isnan(y_transformed))
    
    # Inverse transform should recover original
    y_recovered = inv_func(y_transformed)
    np.testing.assert_array_almost_equal(y, y_recovered, decimal=5)

def test_quantile_transform():
    y = np.array([100, 150, 200, 250, 300, 350, 400], dtype=float)
    y_transformed, inv_func = quantile_transform(y)
    
    assert y_transformed.shape == y.shape
    assert not np.any(np.isnan(y_transformed))

def test_log_transform():
    y = np.array([0.5, 1.0, 2.0, 3.0, 5.0], dtype=float)
    y_transformed, inv_func = log_transform(y)
    
    assert y_transformed.shape == y.shape
    assert not np.any(np.isnan(y_transformed))
    
    # Inverse transform should recover original
    y_recovered = inv_func(y_transformed)
    np.testing.assert_array_almost_equal(y, y_recovered, decimal=5)


def test_yeo_johnson_roundtrip():
    y = np.random.randn(100) * 10 + 50
    y_trans, inverse = yeo_johnson_transform(y)
    y_back = inverse(y_trans)
    assert np.allclose(y, y_back, atol=1e-6)


def test_rank_gauss_roundtrip():
    y = np.random.randn(100) * 10 + 50
    y_trans, inverse = rank_gauss_transform(y)
    y_back = inverse(y_trans)
    assert np.allclose(y, y_back, atol=1e-6)


def test_select_best_includes_yeojohnson():
    y = np.random.exponential(scale=10, size=200)
    y_trans, inv_func, name = select_best_transform(y)
    y_back = inv_func(y_trans)
    assert np.allclose(y, y_back, atol=1e-6)


# --- Tests for apply_target_transform ---

def test_apply_target_transform_disabled():
    y = np.array([1.0, 2.0, 3.0])
    config = {'use_target_transform': False}
    y_t, inv, name = apply_target_transform(y, config)
    assert name is None
    assert inv is None
    assert np.allclose(y, y_t)


def test_apply_target_transform_enabled_legacy_key():
    y = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    config = {'use_target_transform': True}
    y_t, inv, name = apply_target_transform(y, config)
    assert name is not None
    assert inv is not None
    assert np.allclose(y, inv(y_t), atol=1e-6)


def test_apply_target_transform_enabled_config_path():
    y = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    config = {'training': {'target_transforms': {'enabled': True}}}
    y_t, inv, name = apply_target_transform(y, config)
    assert name is not None
    assert inv is not None
    assert np.allclose(y, inv(y_t), atol=1e-6)


def test_apply_target_transform_preserves_shape():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    config = {'training': {'target_transforms': {'enabled': True}}}
    y_t, inv, name = apply_target_transform(y, config)
    assert y_t.shape == y.shape


def test_apply_target_transform_empty_dict():
    y = np.array([1.0, 2.0, 3.0])
    config = {}
    y_t, inv, name = apply_target_transform(y, config)
    assert name is None
    assert inv is None
    assert np.allclose(y, y_t)


def test_select_best_transform_returns_inverse():
    y = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0])
    y_t, inv, name = select_best_transform(y)
    assert name in ('boxcox', 'quantile', 'log', 'yeo_johnson')
    assert callable(inv)
    assert y_t.shape == y.shape
