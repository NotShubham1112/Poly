"""
models/gnn.py
Baseline graph neural networks: GCN, GAT, MPNN (D-MPNN-style).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, global_add_pool, global_mean_pool,
    NNConv, MessagePassing,
)
from torch_geometric.utils import add_self_loops


# ----------------------------------------------------------------------------
# GCN
# ----------------------------------------------------------------------------
class GCNRegressor(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int = 128,
                 n_layers: int = 3, out_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.atom_encoder = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.atom_encoder(x))
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.activations[i](x)
            x = self.dropouts[i](x)
        g = global_add_pool(x, batch)
        return self.head(g).squeeze(-1)

    def get_embedding(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.atom_encoder(x))
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.activations[i](x)
            x = self.dropouts[i](x)
        g = global_add_pool(x, batch)
        return g


# ----------------------------------------------------------------------------
# GAT
# ----------------------------------------------------------------------------
class GATRegressor(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int = 128,
                 heads: int = 4, n_layers: int = 3, out_dim: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.atom_encoder = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads,
                                      heads=heads, dropout=dropout))
        self.activations = nn.ModuleList([nn.ELU() for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.atom_encoder(x))
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.activations[i](x)
            x = self.dropouts[i](x)
        g = global_add_pool(x, batch)
        return self.head(g).squeeze(-1)

    def get_embedding(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.atom_encoder(x))
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.activations[i](x)
            x = self.dropouts[i](x)
        g = global_add_pool(x, batch)
        return g


# ----------------------------------------------------------------------------
# D-MPNN-style edge-aware message passing
# ----------------------------------------------------------------------------
class EdgeMPN(MessagePassing):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr="add")
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_net = nn.GRUCell(hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr):
        # x: (n_atoms, node_dim)
        # edge_index: (2, n_edges)
        # edge_attr: (n_edges, edge_dim)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_i, x_j, edge_attr):
        m = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_net(m)

    def update(self, aggr_out, x):
        return self.update_net(aggr_out, x)


class DMPNNRegressor(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int = 128,
                 n_layers: int = 3, out_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.atom_encoder = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([
            EdgeMPN(hidden_dim, edge_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.activations = nn.ModuleList([nn.Identity() for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        x = F.relu(self.atom_encoder(x))
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = self.activations[i](x)
            x = self.dropouts[i](x)
        g = global_add_pool(x, batch)
        return self.head(g).squeeze(-1)

    def get_embedding(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        x = F.relu(self.atom_encoder(x))
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = self.activations[i](x)
            x = self.dropouts[i](x)
        g = global_add_pool(x, batch)
        return g


def get_gnn(model_type: str, in_dim: int, edge_dim: int, **kwargs):
    """Factory for GNN baselines."""
    if model_type == "gcn":
        return GCNRegressor(in_dim, edge_dim, **kwargs)
    if model_type == "gat":
        return GATRegressor(in_dim, edge_dim, **kwargs)
    if model_type in ("mpnn", "dmpnn"):
        return DMPNNRegressor(in_dim, edge_dim, **kwargs)
    raise ValueError(f"Unknown GNN: {model_type}")
