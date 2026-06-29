import pytest
from features.periodic_polymer import generate_oligomer_smiles, build_periodic_graph

def test_generate_oligomer_smiles():
    # Simple case: ethylene glycol
    smiles = "*CCO*"
    result = generate_oligomer_smiles(smiles, n_repeats=3)
    assert result == "*CCOCCOCCO*"
    
    # Aromatic case
    smiles = "*c1ccc(O)cc1*"
    result = generate_oligomer_smiles(smiles, n_repeats=2)
    assert "c1ccc" in result
    assert result.count("c1ccc") == 2

def test_build_periodic_graph():
    from rdkit import Chem
    smiles = "*CCO*"
    mol = Chem.MolFromSmiles(smiles)
    graph = build_periodic_graph(mol, n_repeats=3)
    assert graph is not None
    assert len(graph.nodes) > 0
