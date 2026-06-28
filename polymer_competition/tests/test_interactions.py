import pytest
import pandas as pd
import numpy as np

from features.interactions import (
    compute_descriptor_ratios,
    compute_fingerprint_descriptor_interactions,
)


def test_descriptor_ratios():
    df = pd.DataFrame({
        "MolWt": [100, 200],
        "HeavyAtomCount": [10, 20],
        "LogP": [1.0, 2.0],
        "TPSA": [50, 100],
    })
    ratios = compute_descriptor_ratios(df)
    assert "mw_per_atom" in ratios.columns
    assert ratios["mw_per_atom"].iloc[0] == pytest.approx(100 / 11)
    assert ratios["mw_per_atom"].iloc[1] == pytest.approx(200 / 21)
    assert "logp_tpsa_ratio" in ratios.columns


def test_descriptor_ratios_optional_columns():
    df = pd.DataFrame({"MolWt": [100], "HeavyAtomCount": [10]})
    ratios = compute_descriptor_ratios(df)
    assert "mw_per_atom" in ratios.columns
    assert ratios.shape[1] == 1


def test_descriptor_ratios_empty():
    df = pd.DataFrame({"foo": [1, 2]})
    ratios = compute_descriptor_ratios(df)
    assert ratios.empty


def test_fp_descriptor_interactions():
    fp = pd.DataFrame({"fp1": [1, 0, 1], "fp2": [0, 1, 0]})
    desc = pd.DataFrame({"d1": [10, 20, 30], "d2": [1, 2, 3]})
    result = compute_fingerprint_descriptor_interactions(fp, desc, top_k=2)
    assert result.shape[0] == 3
    assert result.shape[1] > 0


def test_fp_descriptor_interactions_zero_std():
    fp = pd.DataFrame({"fp1": [1, 1, 1], "fp2": [0, 1, 0]})
    desc = pd.DataFrame({"d1": [10, 20, 30], "d2": [5, 5, 5]})
    result = compute_fingerprint_descriptor_interactions(fp, desc, top_k=2)
    assert result.shape[0] == 3
    # fp1 has zero std -> excluded; d2 has zero std -> excluded
    # only fp2 * d1 should appear
    assert result.shape[1] == 1
