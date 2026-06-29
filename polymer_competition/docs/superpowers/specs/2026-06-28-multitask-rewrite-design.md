# Multi-Task Model Rewrite Design

## Overview

Rewrite `MultiTaskModel` in `models/multitask.py` to use uncertainty-weighted loss (Kendall et al. 2018) and proper masking instead of fixed gamma weighting and zero-padding.

## Current Issues

1. **Fixed gamma**: `gamma_egc=100.0` is a scalar multiplier that doesn't adapt during training
2. **Zero-padding**: Smaller dataset is padded with zeros, treating padded samples as real data
3. **No masking**: Loss includes padded samples, corrupting gradients

## Proposed Solution

### Architecture

- **Shared encoder**: Linear → BatchNorm → ReLU → Dropout (repeated)
- **Task-specific heads**: Separate heads for Tg and Egc predictions
- **Learnable parameters**: `log_var_tg` and `log_var_egc` for uncertainty weighting

### Loss Function

```
L = (1/2σ²_tg) * L_tg + log(σ_tg) + (1/2σ²_egc) * L_egc + log(σ_egc)
```

Where:
- `L_tg`, `L_egc` are MSE losses on masked samples only
- `σ²_tg`, `σ²_egc` are learned task-specific variances
- `log_var_tg`, `log_var_egc` are learnable parameters (log of variance)

### Masking

- Boolean masks (`tg_mask`, `egc_mask`) indicate which samples have valid targets
- Loss computed only on masked samples
- No zero-padding of datasets

## Interface Changes

### Old Interface
```python
model = MultiTaskModel(n_features=100, gamma_egc=100.0)
model.fit(X, y_tg, y_egc, epochs=100, batch_size=32)
tg_pred, egc_pred = model.predict(X)
```

### New Interface
```python
model = MultiTaskModel(n_features=100)
tg_pred, egc_pred = model(x)  # Forward pass
loss, logs = model.loss(tg_pred, egc_pred, tg_true, egc_true, tg_mask, egc_mask)
```

## Compatibility Impact

The new implementation is a pure PyTorch `nn.Module` without `fit()` or `predict()` methods. This will break:

- `training/train.py`: Uses `model.fit()` and `model.predict()`
- `tests/test_multitask.py`: Uses `model.predict()`

## Questions for User

1. Should we update `training/train.py` to work with the new interface, or keep a backward-compatible wrapper?
2. Should we remove the sklearn fallback entirely?
3. Should we update the existing tests in `tests/test_multitask.py`?

## Recommendation

1. Update `training/train.py` to use the new interface directly
2. Remove sklearn fallback - this is a PyTorch-specific model
3. Replace existing tests with new ones that test the PyTorch interface

## Implementation Plan

1. Rewrite `models/multitask.py` with the new implementation
2. Update `tests/test_multitask.py` with new tests
3. Update `training/train.py` to use the new interface
4. Run tests to verify everything works
5. Commit changes