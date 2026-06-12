"""
models.polychain.pecgn

Periodic Equivariant Chain-Growth Network (PECGN).

The second of PolyChain's two core innovations. Replaces Antoniuk's
hard-wired periodic bond with a *learned, direction-aware* boundary
operator. Invariant to shifting the SMILES cut point by one repeat.

Key equation (see README §3.5):
    h_periodic = h_trimer + alpha * BoundaryOp(h_trimer, direction, c)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BoundaryOp(nn.Module):
    """The learnable, direction-aware boundary operator.

    Parameters
    ----------
    dim    : input/output dim
    cst_dim: dim of the Chain Statistics Token
    """

    def __init__(self, dim: int, cst_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + cst_dim + 1, dim * 2),  # +1 for direction flag
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, h_trimer: torch.Tensor, cst: torch.Tensor,
                direction: torch.Tensor) -> torch.Tensor:
        """Compute boundary contribution.

        Parameters
        ----------
        h_trimer  : (B, dim)
        cst       : (B, cst_dim)
        direction : (B, 1)   -- 0 for left extension, 1 for right
        """
        x = torch.cat([h_trimer, cst, direction], dim=-1)
        return self.net(x)


class PECGN(nn.Module):
    """Periodic Equivariant Chain-Growth Network.

    Wraps BoundaryOp with:
        - a learned scalar gate alpha (clamped to small values)
        - symmetric application at both chain ends (translation invariance)
        - optional residual: h_periodic = h_trimer + alpha * (BL(h) + BR(h)) / 2
    """

    def __init__(self, dim: int, cst_dim: int, init_alpha: float = 0.05,
                 max_alpha: float = 0.3):
        super().__init__()
        self.boundary = BoundaryOp(dim, cst_dim)
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.max_alpha = max_alpha

    def forward(self, h_trimer: torch.Tensor, cst: torch.Tensor) -> torch.Tensor:
        """Produce periodic-augmented embedding.

        Parameters
        ----------
        h_trimer : (B, dim)
        cst      : (B, cst_dim)

        Returns
        -------
        (B, dim) periodic embedding.
        """
        B = h_trimer.size(0)
        device = h_trimer.device
        # Apply at both ends (left and right), then average
        dir_left = torch.zeros(B, 1, device=device)
        dir_right = torch.ones(B, 1, device=device)
        bL = self.boundary(h_trimer, cst, dir_left)
        bR = self.boundary(h_trimer, cst, dir_right)
        boundary = 0.5 * (bL + bR)
        alpha = torch.clamp(self.alpha, max=self.max_alpha)
        return h_trimer + alpha * boundary
