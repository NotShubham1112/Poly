"""
models.polychain.pretraining.asterisk_mask

Asterisk-mask reconstruction pretraining (the polymer-specific SSL task).

We randomly mask 0/1/2/3 '*' connection points in a chain; the model
must predict the *type* of each masked '*' (left-end, right-end, or
internal) and the *identity* of its neighbor atom.

Loss = cross-entropy on * type + cross-entropy on neighbor element.
"""
from __future__ import annotations

import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem


# Three classes for * types
ASTERISK_TYPE_LEFT  = 0
ASTERISK_TYPE_RIGHT = 1
ASTERISK_TYPE_INT   = 2
NUM_ASTERISK_TYPES = 3

# A small atom-type vocabulary; padded with UNK
ATOM_VOCAB = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H", "OTHER"]
NUM_ATOM_TYPES = len(ATOM_VOCAB)
UNK_ATOM_IDX = NUM_ATOM_TYPES - 1


def _asterisk_neighbors(smiles: str) -> List[Tuple[int, str]]:
    """Return list of (star_index, neighbor_symbol) for each '*' in the SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    out = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            nbrs = atom.GetNeighbors()
            if nbrs:
                out.append((atom.GetIdx(), nbrs[0].GetSymbol()))
            else:
                out.append((atom.GetIdx(), "OTHER"))
    return out


def random_mask_asterisks(smiles: str, max_mask: int = 2) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Randomly mask a subset of '*' connection points.

    Returns
    -------
    (mask_type_targets, mask_atom_targets, masked_smiles_list)
        mask_type_targets : LongTensor of shape (max_mask,). Each entry is
                           one of NUM_ASTERISK_TYPES (left/right/internal)
                           or -100 for "not masked".
        mask_atom_targets : LongTensor of shape (max_mask,). Each entry is
                           an atom-vocab index or -100.
        masked_smiles_list : the input SMILES (placeholders for '*' kept as
                            atom type 0 in the graph; the *type* target is
                            computed from position in the original SMILES).
    """
    star_info = _asterisk_neighbors(smiles)
    n_stars = len(star_info)
    n_mask = random.randint(0, min(max_mask, n_stars)) if n_stars > 0 else 0

    # Decide which indices to mask
    masked_idx = set(random.sample(range(n_stars), n_mask)) if n_mask > 0 else set()

    type_targets = torch.full((max_mask,), -100, dtype=torch.long)
    atom_targets = torch.full((max_mask,), -100, dtype=torch.long)

    for k, (star_idx, nbr_sym) in enumerate(star_info[:max_mask]):
        if k in masked_idx:
            # Classify: left/right/int based on degree and ring context
            if k == 0:
                ttype = ASTERISK_TYPE_LEFT
            elif k == n_stars - 1:
                ttype = ASTERISK_TYPE_RIGHT
            else:
                ttype = ASTERISK_TYPE_INT
            type_targets[k] = ttype
            atom_targets[k] = ATOM_VOCAB.index(nbr_sym) if nbr_sym in ATOM_VOCAB else UNK_ATOM_IDX

    return type_targets, atom_targets, [smiles] * 1


class AsteriskMaskHead(nn.Module):
    """Heads to predict asterisk type and neighbor atom identity."""

    def __init__(self, in_dim: int, max_mask: int = 2,
                 n_types: int = NUM_ASTERISK_TYPES,
                 n_atoms: int = NUM_ATOM_TYPES):
        super().__init__()
        self.type_head = nn.Linear(in_dim, n_types)
        self.atom_head = nn.Linear(in_dim, n_atoms)
        self.max_mask = max_mask

    def forward(self, periodic_emb: torch.Tensor):
        # periodic_emb: (B, in_dim) – we predict one set of max_mask targets per sample
        type_logits = self.type_head(periodic_emb)   # (B, n_types)
        atom_logits = self.atom_head(periodic_emb)   # (B, n_atoms)
        return type_logits, atom_logits


def asterisk_mask_loss(type_logits: torch.Tensor, atom_logits: torch.Tensor,
                       type_targets: torch.Tensor, atom_targets: torch.Tensor
                       ) -> torch.Tensor:
    """Cross-entropy loss aggregated over all masked positions in a batch.

    type_targets / atom_targets: (B, max_mask) LongTensor with -100 for
    non-masked slots.
    """
    B, M = type_targets.shape
    # Flatten and filter -100
    type_logits_flat = type_logits.unsqueeze(1).expand(-1, M, -1).reshape(B * M, -1)
    atom_logits_flat = atom_logits.unsqueeze(1).expand(-1, M, -1).reshape(B * M, -1)
    type_targets_flat = type_targets.reshape(B * M)
    atom_targets_flat = atom_targets.reshape(B * M)

    type_loss = F.cross_entropy(type_logits_flat, type_targets_flat, ignore_index=-100)
    atom_loss = F.cross_entropy(atom_logits_flat, atom_targets_flat, ignore_index=-100)
    return type_loss + atom_loss
