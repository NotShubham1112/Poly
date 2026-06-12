"""
models.polychain.cst

Chain Statistics Token (CST) computation.

Derives a polymer-aware feature vector from SMILES alone:
    - effective repeat length
    - branching indicator
    - end-group statistics (OH, COOH, NH2, halides, vinyl)
    - ring statistics
    - aromaticity & sp3 fraction
    - heteroatom backbone flag
    - copolymer monomer count
    - molecular weight of the repeat unit

All values are normalized to a fixed-dim vector and embedded.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Reuse features already implemented in features.custom_polymer
from features.custom_polymer import (
    asterisks_count, repeat_unit_length, branching_indicator,
    num_copolymer_monomers, aromatic_carbon_fraction, sp3_carbon_fraction,
    ring_statistics, hbond_donor_acceptor, rotatable_bonds,
    end_group_counts, has_heteroatom_backbone, molecular_weight_monomer,
)


# Number of base features the CST computes from SMILES
CST_BASE_FEATURE_NAMES = [
    "n_asterisks",
    "repeat_length",
    "is_branched",
    "n_monomers_copolymer",
    "aromatic_c_frac",
    "sp3_c_frac",
    "rotatable_bonds",
    "has_heteroatom_backbone",
    "mol_weight_monomer",
    "ring_count",
    "aromatic_rings",
    "largest_ring_size",
    "smallest_ring_size",
    "avg_ring_size",
    "hbd",
    "hba",
    # end-group counts (added dynamically below)
]
ENDGROUP_NAMES = [
    "OH", "COOH", "NH2", "Cl", "Br", "F", "I", "vinyl", "C=C", "C#C",
    "NCO", "epoxide", "ester", "amide", "ether", "aromatic_ring",
]
CST_BASE_FEATURE_NAMES += [f"endgroup_{n}" for n in ENDGROUP_NAMES]
CST_DIM = len(CST_BASE_FEATURE_NAMES)


def compute_cst(smiles: str) -> np.ndarray:
    """Compute the raw CST feature vector for a single SMILES.

    Returns
    -------
    np.ndarray of shape (CST_DIM,).
    """
    base = {
        "n_asterisks": asterisks_count(smiles),
        "repeat_length": repeat_unit_length(smiles),
        "is_branched": branching_indicator(smiles),
        "n_monomers_copolymer": num_copolymer_monomers(smiles),
        "aromatic_c_frac": aromatic_carbon_fraction(smiles),
        "sp3_c_frac": sp3_carbon_fraction(smiles),
        "rotatable_bonds": rotatable_bonds(smiles),
        "has_heteroatom_backbone": has_heteroatom_backbone(smiles),
        "mol_weight_monomer": molecular_weight_monomer(smiles),
    }
    base.update(ring_statistics(smiles))
    base.update(hbond_donor_acceptor(smiles))
    base.update(end_group_counts(smiles, k=3))
    vec = np.array([base.get(n, 0.0) for n in CST_BASE_FEATURE_NAMES],
                   dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


def compute_cst_batch(smiles_list: list[str]) -> np.ndarray:
    """Batch compute CST vectors."""
    return np.stack([compute_cst(s) for s in smiles_list], axis=0)


class CSTNormalizer(nn.Module):
    """Z-score normalizer for CST features, with learnable scale/shift.

    The mean/std are computed from a calibration set and stored as buffers
    (not trainable). The downstream linear layer projects the normalized
    CST into the model's hidden dim.
    """

    def __init__(self, cst_dim: int, hidden_dim: int,
                 mean: np.ndarray | None = None, std: np.ndarray | None = None):
        super().__init__()
        if mean is None:
            mean = np.zeros(cst_dim, dtype=np.float32)
        if std is None:
            std = np.ones(cst_dim, dtype=np.float32)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float))
        self.proj = nn.Sequential(
            nn.Linear(cst_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_dim = hidden_dim

    def forward(self, cst: torch.Tensor) -> torch.Tensor:
        # cst: (B, CST_DIM)
        normed = (cst - self.mean) / (self.std + 1e-6)
        return self.proj(normed)
