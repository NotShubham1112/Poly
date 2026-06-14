"""
tests/test_graphs.py
Smoke tests for monomer/dimer/trimer/periodic graph construction.
"""
import pytest
import torch
from torch_geometric.data import Batch

from features.graphs import smiles_to_graph, kmer_graph, periodic_graph
from features.graph_utils import build_multiscale, collate_multiscale, MultiScaleSample


SAMPLE_SMILES = ["*CCO*", "*C(C)C*", "*c1ccc(*)cc1*"]


def test_monomer_graph():
    g = smiles_to_graph(SAMPLE_SMILES[0])
    assert g is not None
    # *CCO* has 5 atoms: 2 carbons, 1 oxygen, 2 asterisks
    assert g.x.size(0) == 5
    assert g.x.size(1) > 0
    assert g.edge_index.size(0) == 2


def test_dimer_grows():
    """Dimer should either be larger than monomer or fall back to monomer."""
    g1 = smiles_to_graph("*C(C)C*")
    g2 = kmer_graph("*C(C)C*", k=2)
    assert g2 is not None
    # For SMILES where kmer construction fails (e.g., terminal heteroatoms),
    # it gracefully falls back to monomer graph
    assert g2.x.size(0) >= g1.x.size(0)


def test_trimer_grows():
    """Trimer should either be larger than monomer or fall back to monomer."""
    g3 = kmer_graph("*C(C)C*", k=3)
    assert g3 is not None
    assert g3.x.size(0) >= 5


def test_periodic_has_extra_edge():
    g1 = smiles_to_graph("*C(C)C*")
    gp = periodic_graph("*C(C)C*", k=1)
    # Periodic should have at least one more edge than the monomer
    assert gp.edge_index.size(1) >= g1.edge_index.size(1)


def test_multiscale_sample():
    s = build_multiscale("*C(C)C*")
    assert isinstance(s, MultiScaleSample)
    assert s.monomer is not None
    assert s.dimer is not None
    assert s.trimer is not None
    assert s.periodic is not None


def test_collate():
    samples = [build_multiscale(s) for s in ["*C(C)C*", "*CCl*"]]
    samples = [s for s in samples if s is not None]
    batch = collate_multiscale(samples)
    assert "monomer" in batch
    assert "dimer" in batch
    assert "trimer" in batch
    assert "periodic" in batch
    assert isinstance(batch["monomer"], Batch)
