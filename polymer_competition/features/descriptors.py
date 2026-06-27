"""
features/descriptors.py

RDKit 2D molecular descriptors (~200 physicochemical, topological, electronic).
Returns a pandas DataFrame with descriptor columns + SMILES column.
"""
from __future__ import annotations

import signal
from contextlib import contextmanager

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


# Some EState descriptors can hang on certain molecules
_ESTATE_HANG_NAMES = {
    "MinAbsEStateIndex", "MaxAbsEStateIndex",
    "MaxEStateIndex", "MinEStateIndex",
    "EState_VSA1", "EState_VSA2", "EState_VSA3", "EState_VSA4",
    "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8",
    "EState_VSA9", "EState_VSA10", "EState_VSA11",
    "VSA_EState1", "VSA_EState2", "VSA_EState3", "VSA_EState4",
    "VSA_EState5", "VSA_EState6", "VSA_EState7", "VSA_EState8",
    "VSA_EState9", "VSA_EState10",
}

# Canonical list of descriptor names (matches RDKit 2023.09 output) — exclude hang-prone
DESCRIPTOR_NAMES: list[str] = [
    d[0] for d in Descriptors.descList if d[0] not in _ESTATE_HANG_NAMES
]


@contextmanager
def _timeout(seconds: int = 5):
    """Raises TimeoutError if block takes longer than `seconds`."""
    def handler(signum, frame):
        raise TimeoutError("Descriptor computation timed out")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


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
                with _timeout(10):
                    desc = calc.CalcDescriptors(mol)
                rows.append(list(desc))
            except TimeoutError:
                rows.append([np.nan] * len(names))
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
