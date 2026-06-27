"""
tests/test_features.py
Smoke tests for fingerprints, descriptors, custom_polymer features.
"""
import numpy as np
import pytest

from features.fingerprints import morgan_fingerprints, maccs_fingerprints
from features.descriptors import compute_descriptors
from features.polymer_descriptors import compute_polymer_descriptors
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


def test_polymer_descriptors_smoke():
    result = compute_polymer_descriptors("C=CC(C)C(=O)OC")
    assert len(result) == 32
    assert "n_heavy_atoms" in result
    assert "num_rings" in result
    assert "logp" in result
    assert "tpsa" in result
    assert not any(np.isnan(v) for v in result.values() if v != 0.0)


def test_polymer_descriptors_star():
    result = compute_polymer_descriptors("*C(=O)c1ccc(NC(=O)c2ccc(*)cc2)cc1")
    assert result["star_count"] == 2.0
    assert result["is_branched"] == 0.0
    assert result["num_rings"] >= 2
    assert result["aromatic_fraction"] > 0.0


def test_polymer_descriptors_invalid():
    result = compute_polymer_descriptors("")
    assert result["n_heavy_atoms"] == 0.0
    assert result["star_count"] == 0.0
    assert result["num_chiral_centers"] == 0.0


def test_polymer_descriptors_all_keys():
    result = compute_polymer_descriptors("CCO")
    expected_keys = [
        "n_heavy_atoms", "star_count", "is_branched",
        "num_rings", "num_aromatic_rings", "aromatic_fraction",
        "atom_F", "atom_Cl", "atom_Br", "atom_I", "atom_Si",
        "atom_O", "atom_N", "atom_S", "atom_P",
        "atom_F_frac", "atom_Cl_frac", "atom_Br_frac", "atom_I_frac",
        "atom_Si_frac", "atom_O_frac", "atom_N_frac", "atom_S_frac", "atom_P_frac",
        "rotatable_bonds", "rotatable_fraction", "flexibility_index",
        "tpsa", "logp", "mw",
        "num_chiral_centers", "has_stereo",
    ]
    for k in expected_keys:
        assert k in result, f"Missing key: {k}"
    assert len(result) == len(expected_keys)


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
