"""features/preprocessing.py

Feature preprocessing pipeline: imputation, variance filtering, correlation removal, scaling.
Fit on train, transform on test — no data leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression


FORCE_KEEP_COLS = [
    "balaban_j", "bertz_ct",
    "chi0n", "chi1n", "chi2n", "chi3n", "chi4n",
    "chi0v", "chi1v", "chi2v", "chi3v", "chi4v",
    "kappa1", "kappa2", "kappa3", "hall_kier_alpha",
]


class FeaturePreprocessor:
    """Feature preprocessing pipeline. Fit on train, transform on test."""

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.cols_to_drop: list[str] = []
        self.high_corr_mask: list[str] = []
        self.keep_cols: list[str] = []
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray = None) -> "FeaturePreprocessor":
        """Fit preprocessor on training data."""
        X_clean = self._clean(X.copy())

        # Impute
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X_clean),
            columns=X_clean.columns, index=X_clean.index,
        )

        # Variance threshold: remove zero-variance
        variances = X_imputed.var()
        self.cols_to_drop = list(variances[variances == 0].index)

        # Correlation filter: remove features with corr > 0.95
        remaining = [c for c in X_imputed.columns if c not in self.cols_to_drop]
        if len(remaining) > 1:
            corr_matrix = X_imputed[remaining].corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            self.high_corr_mask = [
                col for col in upper.columns if any(upper[col] > 0.95)
            ]
        else:
            self.high_corr_mask = []

        self.keep_cols = [
            c for c in X_imputed.columns
            if c not in self.cols_to_drop and c not in self.high_corr_mask
        ]

        # MI-based feature selection: keep top 520 if too many features
        # 520 = 500 base + 20 buffer for force-kept topological invariants
        must_keep = [c for c in FORCE_KEEP_COLS if c in X_imputed.columns]
        n_select = 520
        if y is not None and len(self.keep_cols) > n_select:
            mi_scores = mutual_info_regression(
                X_imputed[self.keep_cols].fillna(0), y, random_state=42
            )
            top_idx = np.argsort(mi_scores)[-n_select:]
            mi_kept = [self.keep_cols[i] for i in top_idx]
            self.keep_cols = list(dict.fromkeys(mi_kept + must_keep))

        # Fit scaler on remaining features
        if self.keep_cols:
            self.scaler.fit(X_imputed[self.keep_cols])

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame, scale: bool = False) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        assert self.fitted, "Must call fit() first"
        X_clean = self._clean(X.copy())

        X_imputed = pd.DataFrame(
            self.imputer.transform(X_clean),
            columns=X_clean.columns, index=X_clean.index,
        )

        X_out = X_imputed[self.keep_cols].copy()

        if scale:
            X_out.iloc[:] = self.scaler.transform(X_out)

        return X_out

    def _clean(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace inf with nan."""
        return X.replace([np.inf, -np.inf], np.nan)

    def get_feature_names(self) -> list[str]:
        """Return list of features after preprocessing."""
        assert self.fitted, "Must call fit() first"
        return list(self.keep_cols)
