"""
features/fingerprints.py

Classical molecular fingerprints:
    - Morgan (ECFP) circular fingerprints
    - MACCS keys
    - Atom-pair fingerprints

All return numpy arrays of shape (n_samples, n_bits).
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.AtomPairs import Pairs, Torsions


def _safe_mol(smiles: str):
    """Convert SMILES to RDKit mol, stripping asterisks for fingerprinting."""
    if smiles is None or len(smiles) == 0:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None


def morgan_fingerprints(smiles_list, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Morgan (ECFP) circular fingerprints.

    Parameters
    ----------
    smiles_list : list[str]
    radius      : int
    n_bits      : int

    Returns
    -------
    np.ndarray of shape (len(smiles_list), n_bits), dtype uint8.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = _safe_mol(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        out[i] = np.array(fp, dtype=np.uint8)
    return out


def maccs_fingerprints(smiles_list) -> np.ndarray:
    """MACCS structural keys (167 bits)."""
    out = np.zeros((len(smiles_list), 167), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = _safe_mol(smi)
        if mol is None:
            continue
        keys = MACCSkeys.GenMACCSKeys(mol)
        out[i] = np.array(keys, dtype=np.uint8)
    return out


def atom_pair_fingerprints(smiles_list, n_bits: int = 1024) -> np.ndarray:
    """Atom-pair fingerprints (topological distance between atom pairs)."""
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = _safe_mol(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        out[i] = np.array(fp, dtype=np.uint8)
    return out


def topological_torsion_fingerprints(smiles_list, n_bits: int = 1024) -> np.ndarray:
    """Topological torsion fingerprints."""
    gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = _safe_mol(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        out[i] = np.array(fp, dtype=np.uint8)
    return out


def all_fingerprints(smiles_list, cfg: dict | None = None) -> dict[str, np.ndarray]:
    """Compute all fingerprint types in one pass.

    Parameters
    ----------
    cfg : dict with keys 'morgan_radius', 'morgan_bits', 'atom_pair_bits', 'torsion_bits'.
    """
    cfg = cfg or {}
    return {
        "morgan": morgan_fingerprints(
            smiles_list,
            radius=cfg.get("morgan_radius", 2),
            n_bits=cfg.get("morgan_bits", 1024),
        ),
        "maccs": maccs_fingerprints(smiles_list),
        "atom_pair": atom_pair_fingerprints(
            smiles_list, n_bits=cfg.get("atom_pair_bits", 1024)
        ),
        "torsion": topological_torsion_fingerprints(
            smiles_list, n_bits=cfg.get("torsion_bits", 1024)
        ),
    }
