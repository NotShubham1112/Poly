from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

os.environ["RDKIT_SKIP_VALIDATION_WARNINGS"] = "1"
logging.getLogger("rdkit").setLevel(logging.ERROR)

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


class CurriculumScheduler:
    PHASE_DESCRIPTIONS = [
        "0: ring_count=0 AND heteroatom_count=0 AND heavy_atom_count<=6",
        "1: ring_count<=1 AND heteroatom_count=0",
        "2: heteroatom_count<=2 AND ring_count<=1",
        "3: heavy_atom_count<=15 AND ring_count<=3",
        "4: polymer detected",
        "5: full dataset",
    ]

    def __init__(self, df: pd.DataFrame, n_phases: int = 6):
        self.df = df.reset_index(drop=True)
        self.n_phases = n_phases
        self._descriptor_cache: dict[str, dict[str, Any]] = {}

    def get_descriptors(self, smiles: str) -> dict[str, Any]:
        cached = self._descriptor_cache.get(smiles)
        if cached is not None:
            return cached

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result = {
                "ring_count": 0,
                "heavy_atom_count": 0,
                "heteroatom_count": 0,
                "branching_index": 0,
                "is_polymer": False,
            }
            self._descriptor_cache[smiles] = result
            return result

        ring_info = mol.GetRingInfo()
        ring_count = ring_info.NumRings()

        heavy_atom_count = mol.GetNumHeavyAtoms()

        hetero_count = sum(
            1 for a in mol.GetAtoms()
            if a.GetAtomicNum() not in (0, 6)
        )

        max_degree = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "*":
                max_degree = max(max_degree, atom.GetDegree())
        branching_index = max_degree

        is_polymer = "*" in smiles

        result = {
            "ring_count": ring_count,
            "heavy_atom_count": heavy_atom_count,
            "heteroatom_count": hetero_count,
            "branching_index": branching_index,
            "is_polymer": is_polymer,
        }
        self._descriptor_cache[smiles] = result
        return result

    def _compute_descriptors_batch(self) -> pd.DataFrame:
        descriptors = []
        for i, row in self.df.iterrows():
            desc = self.get_descriptors(row["SMILES"])
            desc["idx"] = i
            descriptors.append(desc)
        return pd.DataFrame(descriptors)

    def get_subset(self, phase: int) -> pd.DataFrame:
        if phase >= self.n_phases - 1:
            return self.df

        desc_df = self._compute_descriptors_batch()

        if phase == 0:
            mask = (
                (desc_df["ring_count"] == 0)
                & (desc_df["heteroatom_count"] == 0)
                & (desc_df["heavy_atom_count"] <= 6)
            )
        elif phase == 1:
            mask = (
                (desc_df["ring_count"] <= 1)
                & (desc_df["heteroatom_count"] == 0)
            )
        elif phase == 2:
            mask = (
                (desc_df["heteroatom_count"] <= 2)
                & (desc_df["ring_count"] <= 1)
            )
        elif phase == 3:
            mask = (
                (desc_df["heavy_atom_count"] <= 15)
                & (desc_df["ring_count"] <= 3)
            )
        elif phase == 4:
            mask = desc_df["is_polymer"]
        else:
            mask = pd.Series([True] * len(self.df))

        indices = desc_df.loc[mask, "idx"].values
        return self.df.iloc[indices].reset_index(drop=True)
