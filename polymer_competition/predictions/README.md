# Predictions Directory

All team members' OOF prediction files go here.

## Naming Convention

`{person}_{model_type}_fold{n}.pkl`

Examples:
- `person1_xgb_fold0.pkl`
- `person2_polychain_fold2.pkl`
- `person3_fusionnet_fold4.pkl`

For test predictions:
- `person1_xgb_test.pkl`
- `person2_polychain_test.pkl`

Each `.pkl` file is a Python dict with the following keys:

```python
{
    "val_idx":   np.ndarray of int,   # index into train.csv
    "pred":      np.ndarray of float, # OOF predictions
    "y":         np.ndarray of float, # ground truth
    "metrics":   {"rmse": ..., "mae": ..., "r2": ..., "spearman": ...},
    "model_type": str,
    "fold":       int,
    "person":     str,
}
```

For test files, replace `val_idx` with `id` (test set IDs).
