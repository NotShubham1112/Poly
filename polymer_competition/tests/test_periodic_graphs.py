import pytest
from features.graphs import periodic_graph, multi_scale_periodic_graphs


def test_periodic_graph_has_boundary_edge():
    g = periodic_graph("*CC(=O)OCCO*", k=3)
    assert g is not None
    # Should have periodic boundary edge (is_boundary=True in last dim)
    assert g.edge_attr[:, -1].sum() > 0


def test_periodic_graph_k3_larger_than_k1():
    g1 = periodic_graph("*CC(=O)OCCO*", k=1)
    g3 = periodic_graph("*CC(=O)OCCO*", k=3)
    assert g1 is not None
    assert g3 is not None
    assert g3.num_nodes > g1.num_nodes


def test_multi_scale_returns_three():
    graphs = multi_scale_periodic_graphs("*CC(=O)OCCO*")
    assert 'monomer' in graphs
    assert 'dimer' in graphs
    assert 'trimer' in graphs
    assert graphs['trimer'].num_nodes > graphs['monomer'].num_nodes


def test_periodic_graph_invalid_smiles():
    g = periodic_graph("invalid_smiles")
    assert g is None
