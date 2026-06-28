"""Target transformations for improved model performance."""

import numpy as np
from typing import Tuple, Callable
from scipy import stats


def boxcox_transform(y: np.ndarray) -> Tuple[np.ndarray, Callable]:
    """
    Apply Box-Cox transformation to target.

    Useful for:
    - Making distribution more normal
    - Stabilizing variance
    - Improving linear model performance

    Args:
        y: Raw target values (must be positive)

    Returns:
        Tuple of (transformed_values, inverse_function)
    """
    min_val = y.min()
    if min_val <= 0:
        y_shifted = y - min_val + 1
    else:
        y_shifted = y
        min_val = 0

    y_transformed, lambda_param = stats.boxcox(y_shifted)

    def inverse_transform(y_trans):
        if lambda_param == 0:
            y_inv = np.exp(y_trans)
        else:
            y_inv = (y_trans * lambda_param + 1) ** (1 / lambda_param)
        return y_inv + min_val

    return y_transformed, inverse_transform


def quantile_transform(y: np.ndarray) -> Tuple[np.ndarray, Callable]:
    """
    Apply quantile transformation (rank-based) to target.

    Useful for:
    - Making distribution exactly normal
    - Handling outliers
    - Tree models sometimes benefit

    Args:
        y: Raw target values

    Returns:
        Tuple of (transformed_values, inverse_function)
    """
    from sklearn.preprocessing import QuantileTransformer

    qt = QuantileTransformer(
        output_distribution='normal',
        n_quantiles=min(100, len(y)),
        random_state=42
    )

    y_reshaped = y.reshape(-1, 1)
    y_transformed = qt.fit_transform(y_reshaped).flatten()

    def inverse_transform(y_trans):
        return qt.inverse_transform(y_trans.reshape(-1, 1)).flatten()

    return y_transformed, inverse_transform


def log_transform(y: np.ndarray) -> Tuple[np.ndarray, Callable]:
    """
    Apply log transformation to target.

    Useful for:
    - Skewed distributions
    - Egc values (typically 0.1-10)
    - Ratios and percentages

    Args:
        y: Raw target values (must be positive)

    Returns:
        Tuple of (transformed_values, inverse_function)
    """
    shift = 0.0
    min_val = y.min()
    if min_val <= 0:
        shift = -min_val + 0.001
    y_shifted = y + shift

    y_transformed = np.log(y_shifted)

    def inverse_transform(y_trans):
        return np.exp(y_trans) - shift

    return y_transformed, inverse_transform


def select_best_transform(y: np.ndarray) -> Tuple[np.ndarray, Callable, str]:
    """
    Select best transformation based on distribution.

    Args:
        y: Raw target values

    Returns:
        Tuple of (transformed_values, inverse_function, transform_name)
    """
    skewness = abs(stats.skew(y))

    if skewness > 2:
        if y.min() > 0:
            return log_transform(y), "log"
        else:
            return boxcox_transform(y), "boxcox"
    elif skewness > 1:
        return boxcox_transform(y), "boxcox"
    else:
        return quantile_transform(y), "quantile"
