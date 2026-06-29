"""
tests/test_gnn.py
Tests for get_embedding() on GCN, GAT, and DMPNN models.
"""
import pytest
import torch
from torch_geometric.data import Data, Batch

from models.gnn import GCNRegressor, GATRegressor, DMPNNRegressor


def _make_synthetic_batch(in_dim=9, edge_dim=3, num_graphs=2, num_nodes=3):
    g1 = Data(
        x=torch.randn(num_nodes, in_dim),
        edge_index=torch.tensor([[0, 1], [1, 2]]).T,
        edge_attr=torch.randn(2, edge_dim),
    )
    g2 = Data(
        x=torch.randn(num_nodes, in_dim),
        edge_index=torch.tensor([[0, 1], [1, 2]]).T,
        edge_attr=torch.randn(2, edge_dim),
    )
    return Batch.from_data_list([g1, g2])


def test_gcn_get_embedding_shape():
    batch = _make_synthetic_batch(in_dim=9, edge_dim=3, num_graphs=2)
    model = GCNRegressor(in_dim=9, edge_dim=3, hidden_dim=64, n_layers=2, dropout=0.1)
    model.eval()
    emb = model.get_embedding(batch)
    assert emb.shape == (2, 64), f"Expected (2, 64), got {emb.shape}"


def test_gat_get_embedding_shape():
    batch = _make_synthetic_batch(in_dim=9, edge_dim=3, num_graphs=2)
    model = GATRegressor(
        in_dim=9, edge_dim=3, hidden_dim=64, heads=4, n_layers=2, dropout=0.1
    )
    model.eval()
    emb = model.get_embedding(batch)
    assert emb.shape == (2, 64), f"Expected (2, 64), got {emb.shape}"


def test_dmpnn_get_embedding_shape():
    batch = _make_synthetic_batch(in_dim=9, edge_dim=3, num_graphs=2)
    model = DMPNNRegressor(
        in_dim=9, edge_dim=3, hidden_dim=64, n_layers=2, dropout=0.1
    )
    model.eval()
    emb = model.get_embedding(batch)
    assert emb.shape == (2, 64), f"Expected (2, 64), got {emb.shape}"



def test_gcn_single_graph():
    g = Data(
        x=torch.randn(4, 9),
        edge_index=torch.tensor([[0, 1], [1, 2], [2, 3]]).T,
    )
    batch = Batch.from_data_list([g])
    model = GCNRegressor(in_dim=9, edge_dim=3, hidden_dim=64, n_layers=2)
    model.eval()
    emb = model.get_embedding(batch)
    assert emb.shape == (1, 64), f"Expected (1, 64), got {emb.shape}"


def test_forward_still_works():
    batch = _make_synthetic_batch(in_dim=9, edge_dim=3, num_graphs=2)
    model = GCNRegressor(in_dim=9, edge_dim=3, hidden_dim=64, n_layers=2, dropout=0.1)
    model.eval()
    out = model(batch)
    assert out.shape == (2,), f"Expected (2,), got {out.shape}"


def test_gat_forward_still_works():
    batch = _make_synthetic_batch(in_dim=9, edge_dim=3, num_graphs=2)
    model = GATRegressor(
        in_dim=9, edge_dim=3, hidden_dim=64, heads=4, n_layers=2, dropout=0.1
    )
    model.eval()
    out = model(batch)
    assert out.shape == (2,), f"Expected (2,), got {out.shape}"


def test_dmpnn_forward_still_works():
    batch = _make_synthetic_batch(in_dim=9, edge_dim=3, num_graphs=2)
    model = DMPNNRegressor(
        in_dim=9, edge_dim=3, hidden_dim=64, n_layers=2, dropout=0.1
    )
    model.eval()
    out = model(batch)
    assert out.shape == (2,), f"Expected (2,), got {out.shape}"
