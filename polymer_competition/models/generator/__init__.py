"""
models.generator – Generative polymer chemistry models.

Submodules:
    tokenizer    : SELFIES tokenizer / vocabulary builder
    transformer  : GraphEncoder, GeneratorDecoder, GeneratorConfig
    loss         : GenerativeLoss (CE + property MSE)
    masking      : Lightweight SELFIES inference-time mask
    validator    : Post-hoc RDKit molecule validation
    curriculum   : Curriculum scheduler by chemical complexity
"""
from __future__ import annotations

from .tokenizer import SELFIESTokenizer
from .transformer import GeneratorConfig, GraphEncoder, GeneratorDecoder
from .loss import GenerativeLoss
from .masking import SELFIESMask
from .validator import MoleculeValidator
from .curriculum import CurriculumScheduler
from .metrics import GenerativeMetrics

__all__ = [
    "SELFIESTokenizer",
    "GeneratorConfig",
    "GraphEncoder",
    "GeneratorDecoder",
    "GenerativeLoss",
    "SELFIESMask",
    "MoleculeValidator",
    "CurriculumScheduler",
    "GenerativeMetrics",
]
