"""
models/fusionnet.py
PolymerFusionNet – a multimodal cross-attention model.

Fuses embeddings from:
    - SMILES (ChemBERTa)
    - 2D graph (GIN)
    - 3D geometry (SchNet)
    - Morgan fingerprint
    - Text caption (T5)

This is the "single-modality-per-source" multimodal baseline that
PolyChain's HAMF generalizes to multiple *scales* instead of multiple
*modalities*.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """Multi-head cross-attention over a set of modality embeddings."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        # x: (B, n_modalities, dim)
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x


class PolymerFusionNet(nn.Module):
    """Late-fusion multimodal network.

    Inputs: list of n_modalities tensors, each of shape (B, dim).
    """

    def __init__(self, n_modalities: int = 5, dim: int = 256,
                 n_heads: int = 4, n_layers: int = 2,
                 out_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.modality_proj = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(n_modalities)]
        )
        self.fusion_layers = nn.ModuleList(
            [CrossModalAttention(dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(n_modalities * dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, modality_embeddings: list[torch.Tensor]) -> torch.Tensor:
        # Each tensor is (B, dim); project then stack
        x = torch.stack([proj(e) for proj, e in zip(self.modality_proj, modality_embeddings)],
                        dim=1)  # (B, n_modalities, dim)
        for layer in self.fusion_layers:
            x = layer(x)
        x = x.flatten(start_dim=1)  # (B, n_modalities * dim)
        return self.head(x).squeeze(-1)
