"""
models.polychain.backbone

Edge-aware Graph Isomorphism Network with a virtual supernode
(reads/writes the global graph state). Shared encoder for all scales
(monomer, dimer, trimer, periodic).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
try:
    from torch_geometric.nn import GINConv, global_add_pool
    from torch_geometric.utils import scatter
except ImportError:
    raise ImportError(
        "PyTorch Geometric (torch_geometric) is required for PolyChain. "
        "Install it with: conda install pyg -c pyg  or  pip install torch_geometric"
    )


# ----------------------------------------------------------------------------
# GIN layer with edge features
# ----------------------------------------------------------------------------
class GINEConv(nn.Module):
    """GIN convolution with edge features."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.lin = nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        # x: (n_atoms, node_dim); edge_index: (2, n_edges); edge_attr: (n_edges, edge_dim)
        row, col = edge_index
        # Concatenate node features with edge features for the message
        msg_in = torch.cat([x[col], edge_attr], dim=-1) if edge_attr.numel() > 0 else x[col]
        msg = self.mlp(msg_in)
        # Aggregate
        agg = scatter(msg, row, dim=0, dim_size=x.size(0), reduce='add')
        out = (1 + self.eps) * x + agg
        out = F.relu(self.lin(out))
        return out


class GINBackbone(nn.Module):
    """GIN-S (with virtual supernode) backbone.

    Parameters
    ----------
    in_dim       : input atom feature dim
    edge_dim     : input bond feature dim
    hidden_dim   : width of internal layers
    n_layers     : number of message-passing layers
    dropout      : dropout after each layer
    """

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int = 128,
                 n_layers: int = 4, dropout: float = 0.2, grad_checkpoint: bool = False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.atom_encoder = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GINEConv(hidden_dim, edge_dim, hidden_dim))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        # Virtual supernode that mediates global information
        self.virtual_node = nn.Parameter(torch.zeros(1, hidden_dim))
        self.virtual_mlp = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                           nn.ReLU(),
                           nn.Linear(hidden_dim, hidden_dim))
             for _ in range(n_layers)]
        )

        self.dropout = dropout
        self.out_dim = hidden_dim

    def reset_virtual(self, batch_size: int, device: str) -> torch.Tensor:
        return self.virtual_node.expand(batch_size, -1).to(device)

    def forward(self, data, virtual_state: torch.Tensor | None = None):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        x = self.atom_encoder(x)

        if virtual_state is None:
            virtual_state = self.reset_virtual(int(batch.max().item()) + 1, x.device)

        for i, (conv, norm, v_mlp) in enumerate(zip(self.convs, self.norms, self.virtual_mlp)):
            # Distribute virtual state to atoms
            x = x + virtual_state[batch]
            if self.training and self.grad_checkpoint:
                x = checkpoint.checkpoint(
                    lambda *args: conv(*args), x, edge_index, edge_attr, use_reentrant=False
                )
            else:
                x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.dropout(F.relu(x), p=self.dropout, training=self.training)

            # Update virtual state: aggregate, transform
            agg = scatter(x, batch, dim=0, reduce='add')
            virtual_state = v_mlp(agg) + virtual_state

        g = global_add_pool(x, batch)
        return g, virtual_state
