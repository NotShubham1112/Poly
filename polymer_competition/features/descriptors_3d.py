"""
features/descriptors_3d.py
Compute 3D conformer descriptors for polymer SMILES.
Strips * atoms and caps with H before generating 3D conformers.

Descriptor groups:
  - Rg: Radius of Gyration
  - WHIM: Weighted Holistic Invariant Molecular descriptors (~114)
  - RDF: Radial Distribution Function descriptors (~210)
  - GETAWAY: Geometry, Topology, and Atom-Weights Assembly (~273)
  - PMI: Principal Moments of Inertia (NPR1, NPR2)

Usage:
    from features.descriptors_3d import compute_3d_descriptors_bulk
    df = compute_3d_descriptors_bulk(smiles_list)
"""
from __future__ import annotations

import os
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Descriptors3D

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')
os.environ["RDKIT_SKIP_VALIDATION_WARNINGS"] = "1"


DESC_KEYS = [
    "Rg", "Asphericity", "Eccentricity", "InertialShapeFactor",
    "NPR1", "NPR2", "PMI1", "PMI2", "PMI3",
    "SpherocityIndex",
]


def _sanitize_for_3d(smiles: str) -> Optional[Chem.Mol]:
    """Remove * atoms and cap with H for 3D conformer generation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if not any(a.GetSymbol() == '*' for a in mol.GetAtoms()):
        mol = Chem.AddHs(mol)
        return mol

    rw = Chem.RWMol(mol)
    star_info = []
    for a in mol.GetAtoms():
        if a.GetSymbol() == '*':
            nbrs = [n.GetIdx() for n in a.GetNeighbors()]
            star_info.append((a.GetIdx(), nbrs))

    for idx, nbrs in sorted(star_info, reverse=True):
        for nbr in nbrs:
            if rw.GetBondBetweenAtoms(idx, nbr):
                rw.RemoveBond(idx, nbr)
        rw.RemoveAtom(idx)

    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None

    mol = rw.GetMol()
    mol = Chem.AddHs(mol)
    return mol


def _compute_single(smiles: str, timeout: int = 30) -> Optional[dict]:
    """Compute 3D descriptors for a single SMILES. Returns dict or None."""
    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # signal.signal(signal.SIGALRM, handler)
    # signal.alarm(timeout)

    try:
        mol = _sanitize_for_3d(smiles)
        if mol is None:
            return None

        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        res = AllChem.EmbedMolecule(mol, params)
        if res != 0:
            return None

        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=100)
        except Exception:
            pass

        # Compute Rg
        conf = mol.GetConformer()
        positions = conf.GetPositions()
        centroid = positions.mean(axis=0)
        rg = float(np.sqrt(np.mean(np.sum((positions - centroid) ** 2, axis=1))))

        # Compute 3D descriptors
        descs = Descriptors3D.CalcMolDescriptors3D(mol)
        descs['Rg'] = rg
        return descs
    except Exception as e:
        return None
    # finally:
    #     signal.alarm(0)


def compute_3d_descriptors_bulk(
    smiles_list: list[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute 3D descriptors for all SMILES in a list.

    Returns a DataFrame with the same index as `smiles_list` and
    columns for each 3D descriptor (Rg, NPR1, Asphericity, etc.).
    Missing values (failed conformers) are filled with NaN.
    """
    n = len(smiles_list)
    all_descs = []
    ok = 0
    start = time.time()

    for i, smi in enumerate(smiles_list):
        descs = _compute_single(smi)
        if descs is not None:
            all_descs.append(descs)
            ok += 1
        else:
            all_descs.append(None)

        if verbose and (i + 1) % 500 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"  [{i+1}/{n}] ok={ok} {rate:.1f} smi/s eta={eta:.0f}s")

    # Convert to DataFrame
    rows = []
    for descs in all_descs:
        if descs is not None:
            rows.append({k: descs.get(k, np.nan) for k in DESC_KEYS})
        else:
            rows.append({k: np.nan for k in DESC_KEYS})

    df = pd.DataFrame(rows)
    if verbose:
        elapsed = time.time() - start
        print(f"Done: {ok}/{n} OK ({ok/n*100:.1f}%), {elapsed:.0f}s total")
    return df
