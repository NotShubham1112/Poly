"""End-to-end integration test: load data -> build features -> train Ridge -> predict."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from features.fingerprints import morgan_fingerprints, maccs_fingerprints
from features.descriptors import compute_descriptors, select_descriptors_by_variance
from features.custom_polymer import compute_all_custom_features
from models.baselines import get_linear_model


def test_end_to_end_ridge():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    df = pd.read_csv(data_dir / "train.csv").iloc[:20]
    smiles = df["SMILES"].tolist()
    y = df["property"].values

    fps_morgan = morgan_fingerprints(smiles, radius=2, n_bits=2048)
    fps_maccs = maccs_fingerprints(smiles)

    desc_df = compute_descriptors(smiles)
    desc_df = select_descriptors_by_variance(desc_df)
    desc_cols = [c for c in desc_df.columns if c != "SMILES"]

    cust_df = compute_all_custom_features(smiles)
    cust_cols = [c for c in cust_df.columns if c != "SMILES"]

    X = np.hstack([
        fps_morgan,
        fps_maccs,
        desc_df[desc_cols].values,
        cust_df[cust_cols].astype(float).values,
    ])

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    assert X.shape[0] == len(smiles), "Feature matrix row count mismatch"
    assert not np.isnan(X).any(), "Feature matrix contains NaN after imputation"
    assert X.shape[1] > 2000, "Feature matrix has fewer columns than expected"

    model = get_linear_model("ridge", alpha=1.0)
    model.fit(X, y)
    preds = model.predict(X)

    assert np.isfinite(preds).all(), "Predictions contain non-finite values"
    assert len(preds) == len(y), "Prediction count does not match input"
    assert preds.dtype == np.float64, "Predictions are not float64"
