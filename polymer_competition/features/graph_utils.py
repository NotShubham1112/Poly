"""
features/graph_utils.py

Helpers for multi-scale graph construction (shared with PolyChain).
Provides:
    - MultiScaleGraphBuilder : builds monomer/dimer/trimer/periodic graphs
    - collate_multiscale     : PyG-friendly batching for a list of multi-scale samples
"""
from __future__ import annotations

from typing import Optional

import torch
from torch_geometric.data import Batch, Data

from .graphs import (
    smiles_to_graph,
    kmer_graph,
    periodic_graph,
)


SCALE_NAMES = ("monomer", "dimer", "trimer")


class MultiScaleSample:
    """Container holding the four graph views of a single polymer SMILES."""

    __slots__ = ("monomer", "dimer", "trimer", "periodic", "smiles", "y")

    def __init__(self, monomer, dimer, trimer, periodic, smiles, y=None):
        self.monomer = monomer
        self.dimer = dimer
        self.trimer = trimer
        self.periodic = periodic
        self.smiles = smiles
        self.y = y


def build_multiscale(smiles: str, y: Optional[float] = None) -> Optional[MultiScaleSample]:
    """Build the four graph views of a single polymer SMILES.

    Returns
    -------
    MultiScaleSample or None if any view fails.
    """
    mono = smiles_to_graph(smiles, y=y)
    if mono is None:
        return None
    di = kmer_graph(smiles, k=2, y=y) or mono
    tri = kmer_graph(smiles, k=3, y=y) or mono
    per = periodic_graph(smiles, k=1, y=y) or mono
    return MultiScaleSample(mono, di, tri, per, smiles, y)


def collate_multiscale(samples: list[MultiScaleSample]) -> dict:
    """Collate a list of MultiScaleSample objects into a batched dict.

    Returns
    -------
    dict with keys 'monomer', 'dimer', 'trimer', 'periodic', 'y', 'smiles'.
    """
    if not samples:
        return {}
    valid = []
    for s in samples:
        if s.monomer is not None and s.dimer is not None and s.trimer is not None and s.periodic is not None:
            valid.append(s)
    if not valid:
        return {}
    return {
        "monomer":  Batch.from_data_list([s.monomer  for s in valid]),
        "dimer":    Batch.from_data_list([s.dimer    for s in valid]),
        "trimer":   Batch.from_data_list([s.trimer   for s in valid]),
        "periodic": Batch.from_data_list([s.periodic for s in valid]),
        "y":    torch.tensor([s.y for s in valid if s.y is not None],
                             dtype=torch.float).view(-1, 1),
        "smiles": [s.smiles for s in valid],
    }
