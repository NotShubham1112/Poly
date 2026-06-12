"""
features/custom_polymer.py

Polymer-specific hand-crafted features:
    - Number of asterisks (connection points)
    - Number of end-groups (OH, COOH, NH2, halides, vinyl)
    - Aromatic / aliphatic carbon fraction
    - Branching degree
    - H-bond donor / acceptor counts
    - Ring statistics
    - Backbone flexibility proxy (rotatable bonds)
    - Repeat unit length

All features are computable from SMILES alone — no external data needed.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


# SMARTS patterns for common end-groups (computed on a synthesized k=3 chain)
END_GROUP_PATTERNS = {
    "OH":   Chem.MolFromSmarts("[OX2H]"),
    "COOH": Chem.MolFromSmarts("[CX3](=O)[OX2H]"),
    "NH2":  Chem.MolFromSmarts("[NX3H2]"),
    "Cl":   Chem.MolFromSmarts("[Cl]"),
    "Br":   Chem.MolFromSmarts("[Br]"),
    "F":    Chem.MolFromSmarts("[F]"),
    "I":    Chem.MolFromSmarts("[I]"),
    "vinyl":Chem.MolFromSmarts("C=C"),
    "C=C":  Chem.MolFromSmarts("[CX3]=[CX3]"),
    "C#C":  Chem.MolFromSmarts("[CX2]#[CX2]"),
    "NCO":  Chem.MolFromSmarts("N=C=O"),
    "epoxide": Chem.MolFromSmarts("C1OC1"),
    "ester": Chem.MolFromSmarts("[CX3](=O)[OX2]"),
    "amide": Chem.MolFromSmarts("[CX3](=O)[NX3]"),
    "ether": Chem.MolFromSmarts("[OD2]([#6])[#6]"),
    "aromatic_ring": Chem.MolFromSmarts("c1ccccc1"),
}


def _safe_mol(smiles: str) -> Chem.Mol | None:
    if smiles is None or len(smiles) == 0:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _expand_to_chain(smiles: str, k: int = 3) -> str:
    """Build a k-repeat chain string from a polymer SMILES for end-group analysis.

    We remove '*' and repeat the rest k times, joining with 'C' (aliphatic
    carbon) to mimic a polymer chain.
    """
    # Strip asterisks
    cleaned = smiles.replace("*", "")
    if not cleaned:
        return ""
    # Repeat k times joined by C-C bond
    return ".".join([cleaned] * k)


def asterisks_count(smiles: str) -> int:
    """Number of '*' connection points in the SMILES."""
    return smiles.count("*")


def repeat_unit_length(smiles: str) -> int:
    """Number of non-* heavy atoms in the repeat unit."""
    mol = _safe_mol(smiles)
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetSymbol() != "*")


def branching_indicator(smiles: str) -> int:
    """1 if any atom in the repeat unit has degree > 2, else 0."""
    mol = _safe_mol(smiles)
    if mol is None:
        return 0
    return int(any(a.GetDegree() > 2 and a.GetSymbol() != "*"
                   for a in mol.GetAtoms()))


def num_copolymer_monomers(smiles: str) -> int:
    """Count of monomer components in a copolymer SMILES (separated by '.')."""
    return max(1, len(smiles.split(".")))


def aromatic_carbon_fraction(smiles: str) -> float:
    """Fraction of carbons that are aromatic."""
    mol = _safe_mol(smiles)
    if mol is None:
        return 0.0
    carbons = [a for a in mol.GetAtoms() if a.GetSymbol() == "C"]
    if not carbons:
        return 0.0
    n_aromatic = sum(1 for c in carbons if c.GetIsAromatic())
    return n_aromatic / len(carbons)


def sp3_carbon_fraction(smiles: str) -> float:
    """Fraction of carbons that are SP3 (saturated backbone proxy)."""
    mol = _safe_mol(smiles)
    if mol is None:
        return 0.0
    carbons = [a for a in mol.GetAtoms() if a.GetSymbol() == "C"]
    if not carbons:
        return 0.0
    n_sp3 = sum(1 for c in carbons
                if c.GetHybridization() == Chem.rdchem.HybridizationType.SP3)
    return n_sp3 / len(carbons)


def ring_statistics(smiles: str) -> dict:
    """Ring counts and sizes."""
    mol = _safe_mol(smiles)
    if mol is None:
        return {"ring_count": 0, "aromatic_rings": 0, "largest_ring_size": 0,
                "smallest_ring_size": 0, "avg_ring_size": 0.0}
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    if not atom_rings:
        return {"ring_count": 0, "aromatic_rings": 0, "largest_ring_size": 0,
                "smallest_ring_size": 0, "avg_ring_size": 0.0}
    sizes = [len(r) for r in atom_rings]
    aromatic = 0
    for r in atom_rings:
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r):
            aromatic += 1
    return {
        "ring_count": len(atom_rings),
        "aromatic_rings": aromatic,
        "largest_ring_size": max(sizes),
        "smallest_ring_size": min(sizes),
        "avg_ring_size": float(np.mean(sizes)),
    }


def hbond_donor_acceptor(smiles: str) -> dict:
    """Hydrogen bond donor and acceptor counts (from RDKit)."""
    mol = _safe_mol(smiles)
    if mol is None:
        return {"hbd": 0, "hba": 0}
    return {
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
    }


def rotatable_bonds(smiles: str) -> int:
    """Number of rotatable bonds (backbone flexibility proxy)."""
    mol = _safe_mol(smiles)
    if mol is None:
        return 0
    return rdMolDescriptors.CalcNumRotatableBonds(mol)


def end_group_counts(smiles: str, k: int = 3) -> dict:
    """Count occurrences of common end-group / functional-group SMARTS
    on a synthesized k-repeat chain."""
    chain_smi = _expand_to_chain(smiles, k=k)
    chain_mol = _safe_mol(chain_smi)
    if chain_mol is None:
        return {f"endgroup_{name}": 0 for name in END_GROUP_PATTERNS}
    out = {}
    for name, patt in END_GROUP_PATTERNS.items():
        if patt is None:
            out[f"endgroup_{name}"] = 0
            continue
        matches = chain_mol.GetSubstructMatches(patt)
        out[f"endgroup_{name}"] = len(matches)
    return out


def has_heteroatom_backbone(smiles: str) -> int:
    """1 if the backbone contains N, O, S, or P (vs pure hydrocarbon)."""
    mol = _safe_mol(smiles)
    if mol is None:
        return 0
    hetero = {"N", "O", "S", "P"}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in hetero and atom.GetDegree() >= 2:
            return 1
    return 0


def molecular_weight_monomer(smiles: str) -> float:
    """Molecular weight of the repeat unit (without *)."""
    mol = _safe_mol(smiles)
    if mol is None:
        return 0.0
    return rdMolDescriptors.CalcExactMolWt(mol)


def rigidity_index(smiles: str) -> float:
    """Rigidity proxy: (aromatic_atoms + ring_atoms) / total_heavy_atoms.

    Higher values indicate a more rigid backbone.
    """
    mol = _safe_mol(smiles)
    if mol is None:
        return 0.0
    heavy = [a for a in mol.GetAtoms() if a.GetSymbol() != "*"]
    if not heavy:
        return 0.0
    rigid = sum(1 for a in heavy if a.GetIsAromatic() or a.IsInRing())
    return rigid / len(heavy)


def hbond_density(smiles: str) -> float:
    """H-bond density: (HBD + HBA) / num_heavy_atoms.

    Captures the proportion of H-bond capable sites per unit.
    """
    mol = _safe_mol(smiles)
    if mol is None:
        return 0.0
    heavy = sum(1 for a in mol.GetAtoms() if a.GetSymbol() != "*")
    if heavy == 0:
        return 0.0
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    return (hbd + hba) / heavy


def compute_all_custom_features(smiles_list) -> pd.DataFrame:
    """Compute the full suite of polymer-specific features for a list of SMILES."""
    rows = []
    for smi in smiles_list:
        if smi is None or len(smi) == 0:
            rows.append({})
            continue
        row = {
            "n_asterisks": asterisks_count(smi),
            "repeat_length": repeat_unit_length(smi),
            "is_branched": branching_indicator(smi),
            "n_monomers_copolymer": num_copolymer_monomers(smi),
            "aromatic_c_frac": aromatic_carbon_fraction(smi),
            "sp3_c_frac": sp3_carbon_fraction(smi),
            "rotatable_bonds": rotatable_bonds(smi),
            "has_heteroatom_backbone": has_heteroatom_backbone(smi),
            "mol_weight_monomer": molecular_weight_monomer(smi),
            "rigidity_index": rigidity_index(smi),
            "hbond_density": hbond_density(smi),
        }
        row.update(ring_statistics(smi))
        row.update(hbond_donor_acceptor(smi))
        row.update(end_group_counts(smi, k=3))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.insert(0, "SMILES", list(smiles_list))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df
