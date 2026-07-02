"""
models/gnn.py
Baseline graph neural networks: GCN, GAT, MPNN (D-MPNN-style).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, GINEConv, SAGEConv,
    global_add_pool, global_mean_pool,
    AttentionalAggregation,
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


# ----------------------------------------------------------------------------
# GIN — Graph Isomorphism Network with Edge Features + Global Attention
# ----------------------------------------------------------------------------
class GINEncoder(nn.Module):
    """GIN graph encoder producing a fixed-dim embedding.

    Architecture:
        atom_encoder → GINEConv × n_layers → GlobalAttention → embedding

    The embedding can be used for downstream tasks (regression head,
    hybrid model, or input to tree models).
    """
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int = 256,
                 embed_dim: int = 128, n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.atom_encoder = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            nn_seq = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn_seq, edge_dim=edge_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        gate_nn = nn.Sequential(nn.Linear(hidden_dim, 1))
        self.pool = AttentionalAggregation(gate_nn)
        self.output_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        x = F.relu(self.atom_encoder(x))
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        g = self.pool(x, batch)
        return self.output_proj(g)

    def forward_embedding(self, data):
        """Alias for forward(), returns (B, embed_dim)."""
        return self.forward(data)


class GINRegressor(nn.Module):
    """GIN with regression head for end-to-end training.

    For standalone GIN baseline — evaluates whether graph features
    alone beat the tabular baseline.
    """
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int = 256,
                 embed_dim: int = 128, n_layers: int = 3,
                 out_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.encoder = GINEncoder(in_dim, edge_dim, hidden_dim=hidden_dim,
                                  embed_dim=embed_dim, n_layers=n_layers,
                                  dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, out_dim),
        )

    def forward(self, data):
        emb = self.encoder(data)
        return self.head(emb).squeeze(-1)

    def get_embedding(self, data):
        return self.encoder(data)


class ConditionedGINRegressor(nn.Module):
    """GIN + XGB prediction as an additional input to the regression head.

    Variant B: embed_dim + scalar_pred → concat → MLP
    Variant D: embed_dim + scalar_pred + uncertainty → concat → MLP
    """
    def __init__(self, in_dim: int, edge_dim: int, aux_dim: int = 1,
                 hidden_dim: int = 256, embed_dim: int = 128,
                 n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.encoder = GINEncoder(in_dim, edge_dim, hidden_dim=hidden_dim,
                                  embed_dim=embed_dim, n_layers=n_layers,
                                  dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(embed_dim + aux_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, data, aux):
        emb = self.encoder(data)
        return self.head(torch.cat([emb, aux], dim=1)).squeeze(-1)


class FiLMGINRegressor(nn.Module):
    """GIN with FiLM conditioning from XGB prediction.

    Variant C: γ(x_xgb) * h + β(x_xgb)  where h is GIN embedding.
    The XGB prediction modulates the graph embedding rather than
    being concatenated as an additional feature.
    """
    def __init__(self, in_dim: int, edge_dim: int, aux_dim: int = 1,
                 hidden_dim: int = 256, embed_dim: int = 128,
                 film_hidden: int = 32,
                 n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.encoder = GINEncoder(in_dim, edge_dim, hidden_dim=hidden_dim,
                                  embed_dim=embed_dim, n_layers=n_layers,
                                  dropout=dropout)
        self.film_net = nn.Sequential(
            nn.Linear(aux_dim, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, embed_dim * 2),  # γ, β
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, data, aux):
        emb = self.encoder(data)
        gamma_beta = self.film_net(aux)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        emb_mod = gamma * emb + beta
        return self.head(emb_mod).squeeze(-1)


# ----------------------------------------------------------------------------
# Self-supervised GIN Pretraining
# ----------------------------------------------------------------------------

def mask_atom_types(batch, mask_ratio=0.15, mask_token=None):
    """Randomly mask node features and return atom-type labels for masked nodes.
    
    For each graph in the batch, randomly selects `mask_ratio` nodes,
    replaces their x with `mask_token` (or zeros if None), and records
    the original atom type index for supervised reconstruction.

    Returns
    -------
    x_masked : torch.Tensor — node features with masked positions replaced
    labels   : torch.Tensor — (n_nodes,) long tensor of atom-type labels,
                               -1 for non-masked positions
    """
    x = batch.x
    # Infer atom type: first 12 dims are one-hot atom type, dim 12 is UNK
    atom_type = x[:, :13].argmax(dim=1)  # (n_nodes,) — 0-12
    n = x.size(0)
    n_mask = max(1, int(n * mask_ratio))
    perm = torch.randperm(n, device=x.device)
    mask_idx = perm[:n_mask]

    if mask_token is None:
        mask_feat = torch.zeros_like(x[0:1])
    elif isinstance(mask_token, torch.Tensor):
        mask_feat = mask_token.unsqueeze(0)
    else:
        mask_feat = torch.zeros_like(x[0:1])

    labels = torch.full((n,), -1, dtype=torch.long, device=x.device)
    labels[mask_idx] = atom_type[mask_idx]

    x_masked = x.clone()
    x_masked[mask_idx] = mask_feat
    return x_masked, labels


class GINPretrainEncoder(nn.Module):
    """GINEncoder with a masked-atom prediction head for self-supervised pretraining.

    Usage::
        model = GINPretrainEncoder(in_dim, edge_dim, num_atom_types=13)
        model.train()
        for batch in dataloader:
            x_masked, labels = mask_atom_types(batch)
            batch.x = x_masked
            loss = model(batch, labels)
            loss.backward()
    
    After pretraining, extract the encoder backbone for fine-tuning:
        regressor = GINRegressor(...)
        regressor.encoder.load_state_dict(model.encoder.state_dict())
    """
    def __init__(self, in_dim: int, edge_dim: int,
                 num_atom_types: int = 13,
                 hidden_dim: int = 256, embed_dim: int = 128,
                 n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.encoder = GINEncoder(in_dim, edge_dim, hidden_dim=hidden_dim,
                                  embed_dim=embed_dim, n_layers=n_layers,
                                  dropout=dropout)
        # Node-level prediction head (operates on hidden_dim, not embed_dim)
        self.pretrain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_atom_types),
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.hidden_dim = hidden_dim

    def forward(self, data, labels=None):
        """Forward pass.

        If labels is provided, computes pretrain loss.
        Otherwise returns node embeddings (for inference / embedding extraction).
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch,
        )
        # Get node embeddings from encoder internals
        x = F.relu(self.encoder.atom_encoder(x))
        for conv, norm in zip(self.encoder.convs, self.encoder.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = self.encoder.dropout(x)

        if labels is None:
            return x  # node embeddings

        logits = self.pretrain_head(x)  # (n_nodes, num_atom_types)
        return self.loss_fn(logits, labels), logits


# ----------------------------------------------------------------------------
# Multi-Task GIN (shared encoder + separate TG / EGC heads)
# ----------------------------------------------------------------------------

class MultiTaskGIN(nn.Module):
    """GIN with shared encoder and separate regression heads for TG and EGC.

    Train with alternating TG / EGC batches. The shared encoder learns
    polymer representations useful for both tasks, which acts as a
    regularizer and can improve generalization on the smaller task.
    """
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int = 256,
                 embed_dim: int = 128, n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.encoder = GINEncoder(in_dim, edge_dim, hidden_dim=hidden_dim,
                                  embed_dim=embed_dim, n_layers=n_layers,
                                  dropout=dropout)
        self.tg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self.egc_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, data, task="tg"):
        emb = self.encoder(data)
        if task == "tg":
            return self.tg_head(emb).squeeze(-1)
        return self.egc_head(emb).squeeze(-1)

    def get_embedding(self, data):
        return self.encoder(data)


def get_gnn(model_type: str, in_dim: int, edge_dim: int, **kwargs):
    """Factory for GNN baselines."""
    if model_type == "gcn":
        return GCNRegressor(in_dim, edge_dim, **kwargs)
    if model_type == "gat":
        return GATRegressor(in_dim, edge_dim, **kwargs)
    if model_type in ("mpnn", "dmpnn"):
        return DMPNNRegressor(in_dim, edge_dim, **kwargs)
    if model_type == "gin":
        return GINRegressor(in_dim, edge_dim, **kwargs)
    raise ValueError(f"Unknown GNN: {model_type}")
