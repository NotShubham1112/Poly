"""
models.polychain.graph_builder

PolyChain-specific multi-scale graph constructor.

Wraps features.graphs to build monomer/dimer/trimer/periodic graphs
and bundle them into a MultiScaleSample for downstream training.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch_geometric.data import Batch

from features.graphs import (
    smiles_to_graph, kmer_graph, periodic_graph, atom_features, bond_features,
)
from features.graph_utils import MultiScaleSample, build_multiscale, collate_multiscale


def build_polychain_graphs(smiles_list: list[str],
                           y_list: Optional[list[float]] = None
                           ) -> list[MultiScaleSample]:
    """Build a list of MultiScaleSample from a list of polymer SMILES."""
    if y_list is None:
        y_list = [None] * len(smiles_list)
    out = []
    for smi, y in zip(smiles_list, y_list):
        sample = build_multiscale(smi, y=y)
        if sample is not None:
            out.append(sample)
    return out


__all__ = [
    "build_polychain_graphs",
    "build_multiscale",
    "collate_multiscale",
    "MultiScaleSample",
    "Batch",
]
