"""
tests/test_features.py
Smoke tests for fingerprints, descriptors, custom_polymer features.
"""
import numpy as np
import pytest

from features.fingerprints import morgan_fingerprints, maccs_fingerprints
from features.descriptors import compute_descriptors
from features.custom_polymer import (
    compute_all_custom_features, rigidity_index, hbond_density,
    asterisks_count, repeat_unit_length, branching_indicator,
)
from models.polychain.cst import CST_DIM, compute_cst, compute_cst_batch


SAMPLE_SMILES = ["*CCO*", "*C(C)C*", "*c1ccc(*)cc1*"]


def test_morgan_shape():
    X = morgan_fingerprints(SAMPLE_SMILES, radius=2, n_bits=2048)
    assert X.shape == (3, 2048)
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
    assert "rigidity_index" in df.columns
    assert "hbond_density" in df.columns


def test_rigidity_index():
    # Pure aliphatic *CCO* should have low rigidity
    assert rigidity_index("*CCO*") < 0.5
    # Aromatic ring should have high rigidity
    assert rigidity_index("*c1ccccc1*") > 0.5


def test_hbond_density():
    assert hbond_density("*CCO*") >= 0.0
    assert hbond_density("*CCCC*") < hbond_density("*CCO*")


def test_asterisks_count():
    assert asterisks_count("*CCO*") == 2
    assert asterisks_count("*C(*)CO*") == 3


def test_repeat_unit_length():
    assert repeat_unit_length("*CCO*") == 3  # C, C, O


def test_cst_dim():
    for smi in SAMPLE_SMILES:
        v = compute_cst(smi)
        assert v.shape == (CST_DIM,)


def test_cst_batch():
    M = compute_cst_batch(SAMPLE_SMILES)
    assert M.shape == (3, CST_DIM)
    assert not np.isnan(M).any()


def test_feature_cache_metadata():
    """After building features, metadata file exists with version info."""
    from pathlib import Path
    import yaml
    ROOT = Path(__file__).resolve().parent.parent
    meta_path = ROOT / "data" / "processed" / "metadata.yaml"
    assert meta_path.exists()
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    assert "feature_version" in meta
    assert "git_commit" in meta
    assert "rdkit_version" in meta
