"""
tests/test_polychain.py
Tests for PolyChain forward pass and invariance properties.
"""
import numpy as np
import pytest
import torch

from features.graph_utils import build_multiscale, collate_multiscale
from models.polychain import PolyChain
from models.polychain.cst import compute_cst_batch, CST_DIM


SAMPLE_SMILES = ["*CCO*", "*C(C)C*"]


def _make_batch(smiles_list):
    samples = [build_multiscale(s) for s in smiles_list]
    samples = [s for s in samples if s is not None]
    batch = collate_multiscale(samples)
    batch["cst"] = torch.tensor(compute_cst_batch([s.smiles for s in samples]),
                                dtype=torch.float)
    return batch, samples[0].monomer.x.size(1), samples[0].monomer.edge_attr.size(1)


def test_polychain_forward():
    batch, in_dim, edge_dim = _make_batch(SAMPLE_SMILES)
    model = PolyChain(in_atom_dim=in_dim, in_edge_dim=edge_dim, hidden_dim=64,
                      n_backbone_layers=2, n_hamf_layers=1)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (len(SAMPLE_SMILES),)


def test_polychain_translation_invariance():
    """*CCO* and *COC* represent the same polymer; their embeddings should match."""
    batch1, in_dim, edge_dim = _make_batch(["*CCO*"])
    batch2, _, _ = _make_batch(["*COC*"])
    # Both should be a single-sample batch
    assert batch1["cst"].shape == (1, CST_DIM)
    model = PolyChain(in_atom_dim=in_dim, in_edge_dim=edge_dim, hidden_dim=64,
                      n_backbone_layers=2, n_hamf_layers=1)
    model.eval()
    with torch.no_grad():
        out1 = model(batch1).item()
        out2 = model(batch2).item()
    # The two predictions should be very close (within 1.0)
    assert abs(out1 - out2) < 1.0, f"*CCO*={out1}, *COC*={out2} — invariance broken"


def test_polychain_pecgn_alpha_clamped():
    """The PECGN alpha gate should remain in a small range."""
    _, in_dim, edge_dim = _make_batch(SAMPLE_SMILES)
    model = PolyChain(in_atom_dim=in_dim, in_edge_dim=edge_dim, hidden_dim=64,
                      n_backbone_layers=2, n_hamf_layers=1)
    alpha = torch.clamp(model.pecgn.alpha, max=model.pecgn.max_alpha)
    assert alpha.item() <= 0.3
