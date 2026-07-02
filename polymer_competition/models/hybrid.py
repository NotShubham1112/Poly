"""
models/hybrid.py
Hybrid model combining GIN graph encoder + Tabular MLP + fusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.gnn import GINEncoder
from models.tabular import TabularEncoder
from models.fusion import LayerNormProjectedFusion


class HybridNet(nn.Module):
    """Graph + Tabular → Projected LayerNorm Fusion → Prediction.

    Architecture:
        SMILES ─→ GINEncoder ─→ 128
        Feats  ─→ TabularEncoder ─→ 512
                    ↓
        Graph: Linear(128→256) → LayerNorm → ReLU
        Tab:   Linear(512→256) → LayerNorm → ReLU
                    ↓
                Concat(512) → 256 → 128 → 1
    """
    def __init__(self, in_dim: int = 51, edge_dim: int = 14,
                 n_features: int = 6394,
                 graph_hidden: int = 256, graph_embed: int = 128,
                 tab_hidden: int = 1024, tab_embed: int = 512,
                 fusion_proj: int = 256,
                 n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.graph_encoder = GINEncoder(
            in_dim=in_dim, edge_dim=edge_dim,
            hidden_dim=graph_hidden, embed_dim=graph_embed,
            n_layers=n_layers, dropout=dropout,
        )
        self.tab_encoder = TabularEncoder(
            in_dim=n_features, hidden_dim=tab_hidden,
            embed_dim=tab_embed, dropout=dropout,
        )
        self.fusion = LayerNormProjectedFusion(
            graph_dim=graph_embed, tab_dim=tab_embed,
            proj_dim=fusion_proj, dropout=dropout,
        )

    def forward(self, graph_batch, tab_features: torch.Tensor):
        graph_emb = self.graph_encoder(graph_batch)
        tab_emb = self.tab_encoder(tab_features)
        return self.fusion(graph_emb, tab_emb)

    def get_fusion_embedding(self, graph_batch, tab_features: torch.Tensor) -> torch.Tensor:
        """Return the 512-dim fused representation (before head)."""
        graph_emb = self.graph_encoder(graph_batch)
        tab_emb = self.tab_encoder(tab_features)
        return self.fusion._fuse(graph_emb, tab_emb)

    def freeze_graph_encoder(self):
        for p in self.graph_encoder.parameters():
            p.requires_grad = False

    def unfreeze_graph_encoder(self):
        for p in self.graph_encoder.parameters():
            p.requires_grad = True

    def graph_encoder_is_frozen(self) -> bool:
        return not next(self.graph_encoder.parameters()).requires_grad

    def get_graph_embedding(self, graph_batch):
        return self.graph_encoder(graph_batch)

    def get_tab_embedding(self, tab_features):
        return self.tab_encoder(tab_features)
