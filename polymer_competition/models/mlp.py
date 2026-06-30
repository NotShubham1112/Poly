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


class ResidualMLP(nn.Module):
    """MLP with residual (skip) connections for deeper architectures.

    Each block: Linear -> BN -> ReLU -> Dropout -> Linear -> BN + residual -> ReLU -> Dropout
    The residual connection requires matching dimensions (project if needed).
    """

    def __init__(self, in_dim: int, hidden_dims: list[int] = [1024, 512, 256],
                 out_dim: int = 1, dropouts: list[float] | float = 0.3):
        super().__init__()
        if isinstance(dropouts, (int, float)):
            dropouts = [dropouts] * len(hidden_dims)

        self.blocks = nn.ModuleList()
        self.projects = nn.ModuleList()
        prev = in_dim
        for i, h in enumerate(hidden_dims):
            block = nn.Sequential(
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropouts[i]),
                nn.Linear(h, h),
                nn.BatchNorm1d(h),
            )
            self.blocks.append(block)
            if prev != h:
                self.projects.append(nn.Sequential(nn.Linear(prev, h), nn.BatchNorm1d(h)))
            else:
                self.projects.append(nn.Identity())
            prev = h

        self.head = nn.Linear(prev, out_dim)
        self.act = nn.ReLU()
        self.drop_final = nn.Dropout(dropouts[-1] * 0.5 if dropouts else 0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block, proj in zip(self.blocks, self.projects):
            residual = proj(x)
            x = self.act(block(x) + residual)
        x = self.drop_final(x)
        return self.head(x)


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
