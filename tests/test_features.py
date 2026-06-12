"""
tests/test_features.py
Smoke tests for fingerprints, descriptors, custom_polymer features.
"""
import numpy as np
import pytest

from features.fingerprints import morgan_fingerprints, maccs_fingerprints
from features.descriptors import compute_descriptors
from features.custom_polymer import compute_all_custom_features, compute_cst
from models.polychain.cst import CST_DIM, compute_cst_batch


SAMPLE_SMILES = ["*CCO*", "*C(C)C*", "*c1ccc(*)cc1*"]


def test_morgan_shape():
    X = morgan_fingerprints(SAMPLE_SMILES, radius=2, n_bits=1024)
    assert X.shape == (3, 1024)
    assert X.dtype == np.uint8


def test_maccs_shape():
    X = maccs_fingerprints(SAMPLE_SMILES)
    assert X.shape == (3, 167)


def test_descriptors_columns_present():
    df = compute_descriptors(SAMPLE_SMILES)
    assert "MolWt" in df.columns
    assert "TPSA" in df.columns
    assert len(df) == 3


def test_custom_features_keys():
    df = compute_all_custom_features(SAMPLE_SMILES)
    assert "n_asterisks" in df.columns
    assert "is_branched" in df.columns
    assert "mol_weight_monomer" in df.columns
    assert "ring_count" in df.columns
    assert "aromatic_c_frac" in df.columns


def test_cst_dim():
    for smi in SAMPLE_SMILES:
        v = compute_cst(smi)
        assert v.shape == (CST_DIM,)


def test_cst_batch():
    M = compute_cst_batch(SAMPLE_SMILES)
    assert M.shape == (3, CST_DIM)
    assert not np.isnan(M).any()
