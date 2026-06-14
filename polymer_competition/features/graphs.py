"""
features/graphs.py

Base graph builders used by ALL GNNs (baselines and PolyChain).

Each builder returns a torch_geometric.data.Data object with:
    - x           : (n_atoms, n_atom_features)
    - edge_index  : (2, n_bonds)
    - edge_attr   : (n_bonds, n_bond_features)
    - y           : (1,)  [optional, for supervised training]

The SMILES is assumed to use '*' for connection points.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch
from rdkit import Chem

os.environ["RDKIT_SKIP_VALIDATION_WARNINGS"] = "1"
logging.getLogger("rdkit").setLevel(logging.ERROR)
from torch_geometric.data import Data


# ----------------------------------------------------------------------------
# Atom & bond featurization
# ----------------------------------------------------------------------------
ATOM_TYPES = [
    "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H", "*", "OTHER"
]
ATOM_DEGREES = list(range(0, 7))
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
CHIRAL_TAGS = [0, 1, 2, 3]
NUM_HS = list(range(0, 5))
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.UNSPECIFIED,
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.OTHER,
]


def _onek(idx: int, choices: list) -> list[int]:
    """One-hot encoding with UNK bucket at the end."""
    vec = [0] * (len(choices) + 1)
    if idx < len(choices):
        vec[idx] = 1
    else:
        vec[-1] = 1
    return vec


def atom_features(atom: Chem.rdchem.Atom) -> list[int | float]:
    """Featurize a single RDKit atom (asterisks handled via OTHER bucket)."""
    symbol = atom.GetSymbol()
    sym_idx = ATOM_TYPES.index(symbol) if symbol in ATOM_TYPES else ATOM_TYPES.index("OTHER")
    return (
        _onek(sym_idx, ATOM_TYPES)
        + _onek(atom.GetDegree(), ATOM_DEGREES)
        + _onek(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _onek(int(atom.GetChiralTag()), CHIRAL_TAGS)
        + _onek(atom.GetTotalNumHs(), NUM_HS)
        + _onek(int(atom.GetHybridization()), HYBRIDIZATIONS)
        + [int(atom.GetIsAromatic()), int(atom.IsInRing())]
        + [atom.GetMass() * 0.01]
    )


BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

# Bond feature dimension: one-hot(5) + [conjugated, in_ring, is_boundary](3) = 8
BOND_FEAT_DIM = len(BOND_TYPES) + 1 + 3  # 8


def bond_features(bond: Chem.rdchem.Bond, is_boundary: bool = False) -> list[int | float]:
    """Featurize a single bond, with an extra flag for *-boundary bonds."""
    bt_idx = BOND_TYPES.index(bond.GetBondType()) if bond.GetBondType() in BOND_TYPES else 3
    feat = (
        _onek(bt_idx, BOND_TYPES)
        + [int(bond.GetIsConjugated()), int(bond.IsInRing()), int(is_boundary)]
    )
    assert len(feat) == BOND_FEAT_DIM, (
        f"bond_features returned {len(feat)} dims, expected {BOND_FEAT_DIM}"
    )
    return feat


# ----------------------------------------------------------------------------
# Public builders
# ----------------------------------------------------------------------------
def _safe_mol(smiles: str) -> Optional[Chem.Mol]:
    if smiles is None or len(smiles) == 0:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def smiles_to_graph(smiles: str, y: Optional[float] = None) -> Optional[Data]:
    """Single monomer graph from SMILES (with * marked as OTHER atom type)."""
    mol = _safe_mol(smiles)
    if mol is None:
        return None

    atoms = list(mol.GetAtoms())
    n_atoms = len(atoms)
    if n_atoms == 0:
        return None

    x = torch.tensor([atom_features(a) for a in atoms], dtype=torch.float)

    edges, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])
        bf = bond_features(bond, is_boundary=False)
        edge_attrs.append(bf)
        edge_attrs.append(bf)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs else torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float)

    assert edge_attr.size(1) == BOND_FEAT_DIM, (
        f"smiles_to_graph edge_attr dim {edge_attr.size(1)} != {BOND_FEAT_DIM}"
    )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
    data.smiles = smiles
    return data


def kmer_graph(smiles: str, k: int = 2, y: Optional[float] = None) -> Optional[Data]:
    """Build a k-mer graph by concatenating k copies of the repeat unit.

    Connection points ('*') on the right of copy i are bonded to the * on the
    left of copy i+1. The final molecule is a *-terminated chain of k repeats.

    Falls back to monomer graph on construction errors (e.g., terminal heteroatoms).
    """
    mol = _safe_mol(smiles)
    if mol is None:
        return None

    # Identify * atoms
    star_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "*"]
    if len(star_atoms) < 2:
        # Fall back to monomer graph if SMILES is not a polymer (no asterisks)
        return smiles_to_graph(smiles, y=y)

    try:
        return _build_kmer(mol, smiles, k, y, star_atoms)
    except Exception:
        # Fall back to monomer for problematic SMILES
        return smiles_to_graph(smiles, y=y)


def _build_kmer(mol, smiles, k, y, star_atoms):
    """Internal helper that may raise on invalid chemistry."""
    base_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() != "*"]
    base_bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType())
                  for b in mol.GetBonds()
                  if b.GetBeginAtom().GetSymbol() != "*" and b.GetEndAtom().GetSymbol() != "*"]

    n_base = len(base_atoms)
    offset = lambda i, rep: i + rep * n_base

    rw = Chem.RWMol()
    rw.AddAtom(Chem.Atom(0))
    star_left = 0
    star_right = n_base * k

    base_idx_map = []
    for rep in range(k):
        local_map = {}
        for orig in base_atoms:
            new_atom = Chem.Atom(orig.GetAtomicNum())
            new_atom.SetFormalCharge(orig.GetFormalCharge())
            new_atom.SetIsAromatic(orig.GetIsAromatic())
            new_atom.SetNoImplicit(True)
            new_idx = rw.AddAtom(new_atom)
            local_map[orig.GetIdx()] = new_idx
            base_idx_map.append((rep, orig.GetIdx(), new_idx))
        for a1, a2, bt in base_bonds:
            rw.AddBond(local_map[a1], local_map[a2], bt)

    star_right = rw.AddAtom(Chem.Atom(0))

    for rep in range(k - 1):
        right_atom = max(idx for r, o, idx in base_idx_map if r == rep)
        left_atom = min(idx for r, o, idx in base_idx_map if r == rep + 1)
        if not rw.GetBondBetweenAtoms(int(right_atom), int(left_atom)):
            rw.AddBond(int(right_atom), int(left_atom), Chem.BondType.SINGLE)

    if base_idx_map:
        first_base = min(idx for _, _, idx in base_idx_map[:n_base])
        rw.AddBond(star_left, first_base, Chem.BondType.SINGLE)
        last_base = max(idx for _, _, idx in base_idx_map[-n_base:])
        rw.AddBond(last_base, star_right, Chem.BondType.SINGLE)

    new_mol = rw.GetMol()
    Chem.SanitizeMol(new_mol)

    x = torch.tensor([atom_features(a) for a in new_mol.GetAtoms()], dtype=torch.float)
    edges, edge_attrs = [], []
    for bond in new_mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        is_boundary = (new_mol.GetAtomWithIdx(i).GetSymbol() == "*" or
                       new_mol.GetAtomWithIdx(j).GetSymbol() == "*")
        bf = bond_features(bond, is_boundary=is_boundary)
        edges.append([i, j])
        edges.append([j, i])
        edge_attrs.append(bf)
        edge_attrs.append(bf)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float) if edge_attrs else torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float)

    assert edge_attr.size(1) == BOND_FEAT_DIM, (
        f"kmer_graph edge_attr dim {edge_attr.size(1)} != {BOND_FEAT_DIM}"
    )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
    data.smiles = f"({smiles})_{k}"
    return data


def periodic_graph(smiles: str, k: int = 1, y: Optional[float] = None) -> Optional[Data]:
    """Build a periodic polymer graph by closing the k-mer chain (Antoniuk-style).

    The right * of the last repeat is bonded to the left * of the first repeat,
    forming a closed ring. This is the baseline periodic graph used by both the
    Antoniuk baseline and PolyChain's PECGN component.
    """
    km = kmer_graph(smiles, k=k, y=y)
    if km is None:
        return None

    # Find atoms with symbol '*' (we know there are exactly two: left, right)
    num_atoms = km.x.size(0)
    # In kmer_graph, the * atoms are at index 0 and num_atoms-1
    star_left, star_right = 0, num_atoms - 1

    # Add an edge between the two * atoms
    new_edge = torch.tensor([[star_left, star_right], [star_right, star_left]],
                            dtype=torch.long)
    km.edge_index = torch.cat([km.edge_index, new_edge], dim=1)

    # UNK bond type (one-hot [0,0,0,0,1]), not conjugated, not in ring, is_boundary=True
    new_attr = torch.tensor([[0, 0, 0, 0, 1, 0, 0, 1]] * 2, dtype=torch.float)
    km.edge_attr = torch.cat([km.edge_attr, new_attr], dim=0)
    assert km.edge_attr.size(1) == BOND_FEAT_DIM, (
        f"periodic_graph edge_attr dim {km.edge_attr.size(1)} != {BOND_FEAT_DIM}"
    )

    return km
