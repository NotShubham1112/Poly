"""
models.polychain.polychain_model

End-to-end PolyChain model.

Combines:
    1. Shared GIN-S backbone     (models/polychain/backbone.py)
    2. Hierarchy-Aware Multi-Scale Fusion    (models/polychain/hamf.py)
    3. Periodic Equivariant Chain-Growth Net (models/polychain/pecgn.py)
    4. Chain Statistics Token                (models/polychain/cst.py)

The forward pass:
    multi_sample → backbone (per scale) → scale embeddings
                → HAMF (cross-scale attention) → fused multi-scale repr
                → PECGN (with CST) → periodic embedding
                → concat with CST → MLP head → property prediction
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .backbone import GINBackbone
from .hamf import HAMF
from .pecgn import PECGN
from .cst import CSTNormalizer


class PolyChain(nn.Module):
    """PolyChain: Hierarchical Periodic Transformer.

    Parameters
    ----------
    in_atom_dim  : input atom feature dim
    in_edge_dim  : input bond feature dim
    hidden_dim   : backbone / HAMF hidden dim (default 256)
    n_backbone_layers : number of GIN message-passing layers (default 4)
    n_hamf_layers     : number of HAMF blocks (default 2)
    cst_dim      : dim of raw CST features (default 32, see cst.py)
    cst_mean / cst_std : calibration stats for CST normalization
    out_dim      : regression output dim (default 1)
    dropout      : dropout in backbone + HAMF
    """

    def __init__(self,
                 in_atom_dim: int,
                 in_edge_dim: int,
                 hidden_dim: int = 256,
                 n_backbone_layers: int = 4,
                 n_hamf_layers: int = 2,
                 cst_dim: int = 32,
                 cst_mean: Optional[list[float]] = None,
                 cst_std: Optional[list[float]] = None,
                 out_dim: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        # 1. Shared backbone
        self.backbone = GINBackbone(
            in_dim=in_atom_dim, edge_dim=in_edge_dim,
            hidden_dim=hidden_dim, n_layers=n_backbone_layers,
            dropout=dropout,
        )

        # 2. HAMF (fuses 3 scale embeddings)
        self.hamf = HAMF(
            in_dim=hidden_dim, out_dim=hidden_dim,
            n_scales=3, n_layers=n_hamf_layers,
            n_heads=4, dropout=dropout,
        )

        # 3. CST processor
        import numpy as np
        self.cst_norm = CSTNormalizer(
            cst_dim=cst_dim,
            hidden_dim=hidden_dim,
            mean=np.array(cst_mean) if cst_mean is not None else None,
            std=np.array(cst_std) if cst_std is not None else None,
        )

        # 4. PECGN – takes HAMF output + CST
        self.pecgn = PECGN(dim=3 * hidden_dim, cst_dim=hidden_dim)

        # 5. Prediction head
        self.head = nn.Sequential(
            nn.Linear(3 * hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def encode_scale(self, data) -> torch.Tensor:
        """Encode one scale's batched graph data → (B, hidden_dim)."""
        g, _ = self.backbone(data)
        return g

    def forward(self, batch_dict: dict) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        batch_dict : dict with keys
            'monomer'  : PyG Batch
            'dimer'    : PyG Batch
            'trimer'   : PyG Batch
            'cst'      : (B, cst_dim) torch.Tensor

        Returns
        -------
        (B,) property predictions.
        """
        # 1. Per-scale encoding
        h1 = self.encode_scale(batch_dict["monomer"])
        h2 = self.encode_scale(batch_dict["dimer"])
        h3 = self.encode_scale(batch_dict["trimer"])

        # 2. HAMF fusion
        fused = self.hamf([h1, h2, h3])  # (B, 3*hidden_dim)

        # 3. CST embedding
        cst_emb = self.cst_norm(batch_dict["cst"])  # (B, hidden_dim)

        # 4. PECGN: periodic boundary injection
        periodic_emb = self.pecgn(fused, cst_emb)  # (B, 3*hidden_dim)

        # 5. Concat with CST and predict
        cat = torch.cat([periodic_emb, cst_emb], dim=-1)
        out = self.head(cat).squeeze(-1)
        return out

    @staticmethod
    def estimate_atom_edge_dim(graph_data) -> tuple[int, int]:
        """Helper: peek at the first graph in a DataLoader to estimate dims."""
        return graph_data.x.size(1), graph_data.edge_attr.size(1)
