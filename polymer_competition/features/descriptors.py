"""
features/descriptors.py

RDKit 2D molecular descriptors (~200 physicochemical, topological, electronic).
Returns a pandas DataFrame with descriptor columns + SMILES column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


# Canonical list of descriptor names (matches RDKit 2023.09 output)
DESCRIPTOR_NAMES: list[str] = [d[0] for d in Descriptors.descList]


def _safe_mol(smiles: str):
    if smiles is None or len(smiles) == 0:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception as e:
        import logging
        log = logging.getLogger(__name__)
        log.warning("Failed to parse SMILES: %s", e)
        return None


def compute_descriptors(smiles_list, names: list[str] | None = None) -> pd.DataFrame:
    """Compute a fixed set of RDKit 2D descriptors for each SMILES.

    Parameters
    ----------
    smiles_list : list[str]
    names       : list[str] of descriptor names (defaults to all available).

    Returns
    -------
    pd.DataFrame with columns 'SMILES' + descriptor names.
    """
    names = names or DESCRIPTOR_NAMES
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)

    rows = []
    for smi in smiles_list:
        mol = _safe_mol(smi)
        if mol is None:
            rows.append([np.nan] * len(names))
        else:
            try:
                desc = calc.CalcDescriptors(mol)
                rows.append(list(desc))
            except Exception as e:
                import logging
                log = logging.getLogger(__name__)
                log.warning("Failed to compute descriptors for '%s': %s", smi, e)
                rows.append([np.nan] * len(names))

    df = pd.DataFrame(rows, columns=names)
    df.insert(0, "SMILES", list(smiles_list))

    # Replace inf with NaN; downstream imputer will handle
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def select_descriptors_by_variance(df: pd.DataFrame, threshold: float = 1e-8) -> pd.DataFrame:
    """Drop near-constant descriptor columns."""
    numeric = df.select_dtypes(include=[np.number])
    keep = numeric.columns[numeric.var() > threshold]
    return df[["SMILES"] + list(keep)]
