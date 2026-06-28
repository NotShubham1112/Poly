import pytest
import pandas as pd
import numpy as np
from features.preprocessing import FeaturePreprocessor


def test_preprocessing_removes_zero_variance():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3], "c": [0, 0, 0]})
    fp = FeaturePreprocessor()
    fp.fit(df)
    result = fp.transform(df)
    assert "a" not in result.columns
    assert "c" not in result.columns
    assert "b" in result.columns


def test_preprocessing_handles_nan():
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
    fp = FeaturePreprocessor()
    fp.fit(df)
    result = fp.transform(df)
    assert not result.isna().any().any()


def test_preprocessing_handles_inf():
    df = pd.DataFrame({"a": [1, np.inf, 3], "b": [4, 5, 6]})
    fp = FeaturePreprocessor()
    fp.fit(df)
    result = fp.transform(df)
    assert not result.isna().any().any()
    assert np.isinf(result.values).sum() == 0


def test_preprocessing_correlation_filter():
    np.random.seed(42)
    n = 100
    a = np.arange(n, dtype=float)
    b = a + np.random.randn(n) * 0.001  # b ≈ a, corr > 0.95
    c = np.random.randn(n) * 10  # independent
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    fp = FeaturePreprocessor()
    fp.fit(df)
    result = fp.transform(df)
    assert "c" in result.columns
    assert len(result.columns) < 3


def test_preprocessing_mi_selection():
    fp = FeaturePreprocessor()
    X = pd.DataFrame(
        {"good": [1, 2, 3, 4, 5], "noise": np.random.randn(5), "constant": [1, 1, 1, 1, 1]}
    )
    y = np.array([1, 2, 3, 4, 5])
    fp.fit(X, y=y)
    result = fp.transform(X)
    assert "good" in result.columns
    assert "constant" not in result.columns


def test_preprocessing_fit_transform_roundtrip():
    df = pd.DataFrame(
        {"a": [1.0, 2.0, np.nan, 4.0], "b": [10.0, 20.0, 30.0, 40.0]}
    )
    fp = FeaturePreprocessor()
    fp.fit(df)
    result = fp.transform(df)
    assert result.shape[0] == 4
    assert result.shape[1] == 2


def test_preprocessing_not_fitted_raises():
    fp = FeaturePreprocessor()
    with pytest.raises(AssertionError):
        fp.transform(pd.DataFrame({"a": [1]}))


def test_preprocessing_get_feature_names():
    np.random.seed(0)
    df = pd.DataFrame({"a": np.random.randn(50), "b": np.random.randn(50)})
    fp = FeaturePreprocessor()
    fp.fit(df)
    names = fp.get_feature_names()
    assert "a" in names
    assert "b" in names
