"""
models/mlp.py
MLP heads for fingerprint / descriptor features.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FingerprintMLP(nn.Module):
    """MLP for fixed-length fingerprint vectors (Morgan, MACCS, etc.)."""

    def __init__(self, in_dim: int, hidden_dims: list[int] = [512, 256, 128],
                 out_dim: int = 1, dropout: float = 0.3, use_bn: bool = True):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DescriptorMLP(nn.Module):
    """MLP for RDKit 2D descriptors (continuous-valued input)."""

    def __init__(self, in_dim: int, hidden_dims: list[int] = [256, 128, 64],
                 out_dim: int = 1, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
