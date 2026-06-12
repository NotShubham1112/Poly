"""
models/graph_transformer.py
Graph Transformer using TransformerConv.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_add_pool


class GraphTransformerRegressor(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int = 128,
                 heads: int = 4, n_layers: int = 4, out_dim: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.atom_encoder = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                TransformerConv(hidden_dim, hidden_dim // heads,
                                heads=heads, dropout=dropout,
                                edge_dim=edge_dim, beta=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = dropout
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
        x = self.atom_encoder(x)
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index, edge_attr)
            x = norm(x + F.dropout(F.relu(x_new), p=self.dropout,
                                   training=self.training))
        g = global_add_pool(x, batch)
        return self.head(g).squeeze(-1)
