"""
models/fusion.py
Fusion strategies for combining graph + tabular embeddings.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Simple concatenation fusion.

    Graph_emb (128) + Tabular_emb (512) → Concat (640) → MLP → pred
    """
    def __init__(self, graph_dim: int = 128, tab_dim: int = 512,
                 hidden_dim: int = 256, out_dim: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(graph_dim + tab_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, graph_emb: torch.Tensor,
                tab_emb: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([graph_emb, tab_emb], dim=1)
        return self.head(fused).squeeze(-1)


class ProjectedConcatFusion(nn.Module):
    """Project graph/tabular to same dim, then concat + head.

    This is useful when graph and tabular have very different dims.
    """
    def __init__(self, graph_dim: int = 128, tab_dim: int = 512,
                 proj_dim: int = 256, out_dim: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.graph_proj = nn.Linear(graph_dim, proj_dim)
        self.tab_proj = nn.Linear(tab_dim, proj_dim)
        self.head = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, out_dim),
        )

    def forward(self, graph_emb: torch.Tensor,
                tab_emb: torch.Tensor) -> torch.Tensor:
        g = self.graph_proj(graph_emb)
        t = self.tab_proj(tab_emb)
        fused = torch.cat([g, t], dim=1)
        return self.head(fused).squeeze(-1)


class LayerNormProjectedFusion(nn.Module):
    """Project both modalities to proj_dim with LayerNorm, concat, MLP head.

    Architecture:
        graph_emb (128) → Linear → 256 → LayerNorm → ReLU
        tab_emb  (512)  → Linear → 256 → LayerNorm → ReLU
                            Concat (512)
                            Linear → 256 → ReLU → Dropout(0.2)
                            Linear → 128 → ReLU → Dropout(0.2)
                            Linear → 1
    """
    def __init__(self, graph_dim: int = 128, tab_dim: int = 512,
                 proj_dim: int = 256, out_dim: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, out_dim),
        )

    def forward(self, graph_emb: torch.Tensor,
                tab_emb: torch.Tensor) -> torch.Tensor:
        fused = self._fuse(graph_emb, tab_emb)
        return self.head(fused).squeeze(-1)

    def _fuse(self, graph_emb: torch.Tensor,
              tab_emb: torch.Tensor) -> torch.Tensor:
        g = self.graph_proj(graph_emb)
        t = self.tab_proj(tab_emb)
        return torch.cat([g, t], dim=1)
