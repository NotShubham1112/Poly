import pytest
from rdkit import Chem
from features.advanced_descriptors import (
    hansen_solubility_parameters,
    free_volume_fraction,
    chain_flexibility,
    conjugation_length,
    compute_topological_invariants,
    TOPOLOGICAL_KEYS,
)

def test_hansen_solubility_parameters():
    mol = Chem.MolFromSmiles("*CCO*")
    dp, dP, dH = hansen_solubility_parameters(mol)
    assert isinstance(dp, float)
    assert isinstance(dP, float)
    assert isinstance(dH, float)
    assert dp > 0  # Dispersion always positive

def test_free_volume_fraction():
    mol = Chem.MolFromSmiles("*CCO*")
    fv = free_volume_fraction(mol)
    assert isinstance(fv, float)
    assert 0 < fv < 1  # Free volume fraction between 0 and 1

def test_chain_flexibility():
    mol = Chem.MolFromSmiles("*CCO*")
    flexibility = chain_flexibility(mol)
    assert isinstance(flexibility, float)
    assert flexibility >= 0

def test_conjugation_length():
    mol = Chem.MolFromSmiles("*c1ccc(O)cc1*")
    conj_len = conjugation_length(mol)
    assert isinstance(conj_len, int)
    assert conj_len >= 0


def test_topological_invariants_produce_values():
    mol = Chem.MolFromSmiles("CCO")
    result = compute_topological_invariants(mol)
    assert result["balaban_j"] > 0
    assert result["kappa1"] > 0
    assert len(result) == len(TOPOLOGICAL_KEYS)


def test_topological_invariants_none_mol():
    result = compute_topological_invariants(None)
    assert all(v == 0.0 for v in result.values())
