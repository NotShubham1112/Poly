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
    assert g.x.size(0) == 4  # C, C, O, *  →  actually 3 atoms (the * gets added too)
    assert g.x.size(1) > 0
    assert g.edge_index.size(0) == 2


def test_dimer_grows():
    g1 = smiles_to_graph(SAMPLE_SMILES[0])
    g2 = kmer_graph(SAMPLE_SMILES[0], k=2)
    assert g2 is not None
    # Dimer should have at least 2× the atoms of the monomer
    assert g2.x.size(0) > g1.x.size(0)


def test_trimer_grows():
    g3 = kmer_graph(SAMPLE_SMILES[0], k=3)
    assert g3 is not None
    assert g3.x.size(0) >= 6


def test_periodic_has_extra_edge():
    g1 = smiles_to_graph(SAMPLE_SMILES[0])
    gp = periodic_graph(SAMPLE_SMILES[0], k=1)
    # Periodic should have at least one more edge than the monomer
    assert gp.edge_index.size(1) >= g1.edge_index.size(1)


def test_multiscale_sample():
    s = build_multiscale(SAMPLE_SMILES[0])
    assert isinstance(s, MultiScaleSample)
    assert s.monomer is not None
    assert s.dimer is not None
    assert s.trimer is not None
    assert s.periodic is not None


def test_collate():
    samples = [build_multiscale(s) for s in SAMPLE_SMILES]
    samples = [s for s in samples if s is not None]
    batch = collate_multiscale(samples)
    assert "monomer" in batch
    assert "dimer" in batch
    assert "trimer" in batch
    assert "periodic" in batch
    assert isinstance(batch["monomer"], Batch)
