from __future__ import annotations

import logging
import os

import numpy as np
from scipy.stats import ks_2samp

os.environ["RDKIT_SKIP_VALIDATION_WARNINGS"] = "1"
logging.getLogger("rdkit").setLevel(logging.ERROR)

from rdkit import Chem
from rdkit.Chem import AllChem, Scaffolds


class GenerativeMetrics:
    def __init__(self, reference_smiles: list[str] | None = None):
        self.reference_set: set[str] | None = set(reference_smiles) if reference_smiles else None
        self.reference_properties: list[float] | None = None
        self.seen_molecules: set[str] = set()

    def set_reference_properties(self, props: list[float]) -> None:
        self.reference_properties = props

    def compute(
        self, generated_smiles: list[str],
        generated_properties: list[float] | None = None,
    ) -> dict:
        if not generated_smiles:
            return {
                "validity": 0.0,
                "uniqueness": 0.0,
                "novelty": 0.0,
                "scaffold_diversity": 0.0,
                "n_valid": 0,
                "n_unique": 0,
                "n_novel": 0,
                "n_total": 0,
            }

        valid = []
        for smi in generated_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                    valid.append(smi)
                except Exception:
                    pass

        validity = len(valid) / len(generated_smiles)

        canonical_valid = []
        for smi in valid:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canonical_valid.append(Chem.MolToSmiles(mol))

        unique = list(set(canonical_valid))
        uniqueness = len(unique) / max(len(canonical_valid), 1)

        novelty = 0.0
        if self.reference_set is not None and unique:
            novel = [s for s in unique if s not in self.reference_set]
            novelty = len(novel) / len(unique)

        scaffolds = set()
        for smi in unique:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                try:
                    scaff = Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                    scaffolds.add(scaff)
                except Exception:
                    pass
        scaffold_diversity = len(scaffolds) / max(len(unique), 1)

        property_ks = None
        property_mean_shift = None
        if generated_properties is not None and self.reference_properties is not None:
            gen_arr = np.asarray(generated_properties, dtype=np.float64)
            ref_arr = np.asarray(self.reference_properties, dtype=np.float64)
            if len(gen_arr) > 0 and len(ref_arr) > 0:
                ks_stat, _ = ks_2samp(gen_arr, ref_arr)
                property_ks = round(float(ks_stat), 4)
                property_mean_shift = round(float(np.mean(gen_arr) - np.mean(ref_arr)), 4)

        self.seen_molecules.update(unique)

        return {
            "validity": round(validity, 4),
            "uniqueness": round(uniqueness, 4),
            "novelty": round(novelty, 4),
            "scaffold_diversity": round(scaffold_diversity, 4),
            "property_ks": property_ks,
            "property_mean_shift": property_mean_shift,
            "n_valid": len(valid),
            "n_unique": len(unique),
            "n_novel": len(novel) if self.reference_set is not None else 0,
            "n_total": len(generated_smiles),
        }
