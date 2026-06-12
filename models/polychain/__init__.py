"""
models.polychain – PolyChain: Hierarchical Periodic Transformer with
                   Equivariant Multi-Scale Graph Reasoning.

Components:
    backbone : GIN-S encoder with virtual node
    hamf     : Hierarchy-Aware Multi-Scale Fusion
    pecgn    : Periodic Equivariant Chain-Growth Network
    cst      : Chain Statistics Token computation
    graph_builder : PolyChain-specific multi-scale graph constructor
    polychain_model : End-to-end model
    pretraining    : Self-supervised pretraining tasks
"""
from .polychain_model import PolyChain
from .cst import compute_cst
from .graph_builder import build_polychain_graphs

__all__ = ["PolyChain", "compute_cst", "build_polychain_graphs"]
