"""
models.polychain.hamf

Hierarchy-Aware Multi-Scale Fusion (HAMF).

The first of PolyChain's two core innovations. Treats the three
oligomer-scale embeddings (monomer/dimer/trimer) as a *sequence*
and applies a chain-structured transformer over them.

Key equations (see README §3.4):
    h_tilde_k = LayerNorm(h_k + MHA_k({h_j}_{j<=k}))
    H_MS      = [h_tilde_1; h_tilde_2; h_tilde_3]
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class CausalMultiHeadAttention(nn.Module):
    """Multi-head self-attention with a causal-like mask over the scale axis.

    Scale k can attend to scales j <= k. Implemented via an additive mask.
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1,
                 n_scales: int = 3):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        # Causal mask: 1 if j <= k, 0 otherwise. Shape (n_scales, n_scales).
        mask = torch.tril(torch.ones(n_scales, n_scales, dtype=torch.bool))
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_scales, dim)
        B, S, D = x.shape
        qkv = self.qkv(x)  # (B, S, 3*dim)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # Attention scores: (B, n_heads, S, S)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(~self.causal_mask[:S, :S].unsqueeze(0).unsqueeze(0),
                                float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.out(out)


class HAMFBlock(nn.Module):
    """Single HAMF block: pre-norm causal MHA + FFN."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1,
                 n_scales: int = 3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalMultiHeadAttention(dim, n_heads, dropout, n_scales)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class HAMF(nn.Module):
    """Hierarchy-Aware Multi-Scale Fusion.

    Parameters
    ----------
    in_dim   : per-scale input dim (assumed identical for all three scales)
    out_dim  : output dim per scale
    n_scales : number of scales (default 3: monomer/dimer/trimer)
    n_layers : number of HAMF blocks
    """

    def __init__(self, in_dim: int, out_dim: int = 256,
                 n_scales: int = 3, n_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_scales = n_scales
        self.proj = nn.Linear(in_dim, out_dim)
        # Learnable scale positional encoding
        self.scale_pe = nn.Parameter(torch.randn(n_scales, out_dim) * 0.02)
        self.blocks = nn.ModuleList([
            HAMFBlock(out_dim, n_heads, dropout, n_scales)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim

    def forward(self, scale_embeddings: list[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple scale embeddings.

        Parameters
        ----------
        scale_embeddings : list of n_scales tensors, each (B, in_dim).

        Returns
        -------
        Tensor of shape (B, n_scales * out_dim) holding the fused multi-scale repr.
        """
        # Project all scales to a common dim
        x = torch.stack([self.proj(e) for e in scale_embeddings], dim=1)  # (B, S, D)
        x = x + self.scale_pe.unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x.flatten(start_dim=1)  # (B, S*D)
