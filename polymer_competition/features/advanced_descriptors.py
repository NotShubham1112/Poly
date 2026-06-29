"""Advanced polymer descriptors for property prediction."""

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np
from typing import Tuple


def hansen_solubility_parameters(mol: Chem.Mol) -> Tuple[float, float, float]:
    """
    Estimate Hansen solubility parameters from group contributions.
    
    δD: Dispersion parameter (MPa^0.5)
    δP: Polar parameter (MPa^0.5)
    δH: Hydrogen bonding parameter (MPa^0.5)
    
    Reference: Hansen, C. M. (2007). Hansen Solubility Parameters: A User's Handbook.
    """
    if mol is None:
        return 0.0, 0.0, 0.0
    
    # Simplified group contribution method
    # Based on van Krevelen and Hoftyzer (1976)
    
    # Count functional groups
    n_atoms = mol.GetNumHeavyAtoms()
    
    # Dispersion (δD) - related to molecular size and polarizability
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    
    # Approximate δD from molecular properties
    # Higher MW and logp → higher δD
    dp = 15.0 + 0.5 * np.log(mw + 1) + 2.0 * logp
    
    # Polar (δP) - related to dipole moments and polarity
    # Higher TPSA → higher δP
    n_polar = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [7, 8, 9, 16])
    dP = 5.0 + 0.1 * tpsa + 1.0 * n_polar
    
    # Hydrogen bonding (δH) - related to H-bond donors/acceptors
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    dH = 10.0 + 2.0 * hbd + 1.5 * hba
    
    return dp, dP, dH


def free_volume_fraction(mol: Chem.Mol) -> float:
    """
    Estimate free volume fraction using Bondi group contributions.
    
    Free volume is critical for Tg prediction (Fox-Flory equation).
    Higher free volume → lower Tg.
    
    Reference: Bondi, A. (1964). Van der Waals volumes and radii.
    """
    if mol is None:
        return 0.0
    
    # Count atoms by type
    n_C = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    n_O = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
    n_N = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
    n_S = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16)
    n_F = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
    n_Cl = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 17)
    
    # Van der Waals volumes (cm³/mol) from Bondi
    vdw_C = 16.6  # Methylene
    vdw_O = 8.5   # Ether oxygen
    vdw_N = 10.5  # Amine
    vdw_S = 18.5  # Thioether
    vdw_F = 5.5   # Fluorine
    vdw_Cl = 19.5 # Chlorine
    
    # Calculate occupied volume
    v_occupied = (n_C * vdw_C + n_O * vdw_O + n_N * vdw_N + 
                  n_S * vdw_S + n_F * vdw_F + n_Cl * vdw_Cl)
    
    # Total molecular volume (approximate from MW)
    mw = Descriptors.MolWt(mol)
    density = 1.0  # g/cm³ approximate
    v_total = mw / density
    
    # Free volume fraction
    if v_total > 0:
        fv = 1.0 - (v_occupied / v_total)
        return max(0.0, min(1.0, fv))  # Clamp to [0, 1]
    
    return 0.5  # Default


def chain_flexibility(mol: Chem.Mol) -> float:
    """
    Estimate chain flexibility metric.
    
    Combines:
    - Rotatable bonds
    - Fraction of sp3 carbons
    - Ring strain indicators
    
    Higher flexibility → lower Tg (more conformational freedom)
    """
    if mol is None:
        return 0.0
    
    # Rotatable bonds
    n_rotatable = Descriptors.NumRotatableBonds(mol)
    
    # Fraction sp3
    frac_sp3 = Descriptors.FractionCSP3(mol)
    
    # Ring info
    n_rings = mol.GetRingInfo().NumRings()
    ring_density = n_rings / max(1, mol.GetNumHeavyAtoms())
    
    # Flexibility = normalized rotatable bonds + sp3 contribution - ring constraint
    flexibility = (n_rotatable / max(1, mol.GetNumHeavyAtoms()) + 
                   frac_sp3 * 0.3 - 
                   ring_density * 0.2)
    
    return max(0.0, flexibility)


def conjugation_length(mol: Chem.Mol) -> int:
    """
    Measure effective conjugation length in polymer backbone.
    
    Critical for Egc prediction:
    - Longer conjugation → smaller band gap
    - More aromatic rings in backbone → better π-overlap
    
    Returns:
        Number of conjugated atoms in longest path
    """
    if mol is None:
        return 0
    
    # Find aromatic atoms
    aromatic_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsAromatic()]
    
    if not aromatic_atoms:
        return 0
    
    # Find longest path through aromatic atoms
    from rdkit.Chem import rdmolops
    
    max_path_len = 0
    
    for start in aromatic_atoms:
        for end in aromatic_atoms:
            if start != end:
                path = rdmolops.GetShortestPath(mol, start, end)
                # Check if path is through aromatic atoms
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in path):
                    max_path_len = max(max_path_len, len(path))
    
    return max_path_len


TOPOLOGICAL_KEYS = [
    "balaban_j", "bertz_ct",
    "chi0n", "chi1n", "chi2n", "chi3n", "chi4n",
    "chi0v", "chi1v", "chi2v", "chi3v", "chi4v",
    "kappa1", "kappa2", "kappa3", "hall_kier_alpha",
]


def compute_topological_invariants(mol: Chem.Mol) -> dict:
    if mol is None:
        return {k: 0.0 for k in TOPOLOGICAL_KEYS}
    try:
        return {
            "balaban_j": Descriptors.BalabanJ(mol),
            "bertz_ct": Descriptors.BertzCT(mol),
            "chi0n": Descriptors.Chi0n(mol),
            "chi1n": Descriptors.Chi1n(mol),
            "chi2n": Descriptors.Chi2n(mol),
            "chi3n": Descriptors.Chi3n(mol),
            "chi4n": Descriptors.Chi4n(mol),
            "chi0v": Descriptors.Chi0v(mol),
            "chi1v": Descriptors.Chi1v(mol),
            "chi2v": Descriptors.Chi2v(mol),
            "chi3v": Descriptors.Chi3v(mol),
            "chi4v": Descriptors.Chi4v(mol),
            "kappa1": Descriptors.Kappa1(mol),
            "kappa2": Descriptors.Kappa2(mol),
            "kappa3": Descriptors.Kappa3(mol),
            "hall_kier_alpha": Descriptors.HallKierAlpha(mol),
        }
    except Exception:
        return {k: 0.0 for k in TOPOLOGICAL_KEYS}


def compute_all_advanced_features(mol: Chem.Mol) -> dict:
    """
    Compute all advanced polymer descriptors.
    
    Args:
        mol: RDKit molecule
    
    Returns:
        Dictionary of feature names and values
    """
    if mol is None:
        return {}
    
    dp, dP, dH = hansen_solubility_parameters(mol)
    
    result = {
        'hansen_dp': dp,
        'hansen_dP': dP,
        'hansen_dH': dH,
        'free_volume': free_volume_fraction(mol),
        'chain_flexibility': chain_flexibility(mol),
        'conjugation_length': conjugation_length(mol),
        'total_hansen': dp + dP + dH,
    }
    topo_features = compute_topological_invariants(mol)
    result.update(topo_features)
    return result
