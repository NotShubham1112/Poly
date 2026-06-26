# OOM Fix — Build Features Memory Efficiency

**Goal:** Fix Kaggle notebook OOM crash during Cell 5 (Build Features) by reducing peak memory usage.

**Architecture:** Three changes: (1) float64→float32 halves feature matrix memory, (2) vectorized feature lookup replaces dict-per-row pattern, (3) explicit garbage collection between major steps.

**Files modified:** `features/build_features.py` only.

---

### Changes

1. **`features/build_features.py` line 113**: `astype(np.float32)` after imputation
2. **`features/build_features.py` line 115**: Store `cache_df` as float32
3. **`features/build_features.py` lines 119-129**: Replace `lookup_features` — use `iloc[indices]` vectorized instead of per-row `.to_dict()`
4. **`features/build_features.py` lines 105-111**: Add `del fp_dfs, desc_df, cust_df` + `gc.collect()` after concat
5. **`features/build_features.py` line 140**: Save parquet as float32
