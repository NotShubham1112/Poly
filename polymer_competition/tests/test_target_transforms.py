import pytest
import numpy as np
from features.target_transforms import (
    boxcox_transform,
    quantile_transform,
    log_transform
)

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
