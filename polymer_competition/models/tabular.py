"""
models/tabular.py
MLP tabular encoder for the 6394 engineered features.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    """Encodes 6394 engineered features into a lower-dim embedding.

    Architecture:
        Linear(6394 → 1024) → ReLU → Dropout
        → Linear(1024 → 512) → ReLU → Dropout → embed_dim

    The output embedding is designed to complement the graph embedding.
    """
    def __init__(self, in_dim: int = 6394, hidden_dim: int = 1024,
                 embed_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
