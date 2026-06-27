from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def compute_polymer_descriptors(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _empty_descriptors()

    result = {}

    n_heavy = mol.GetNumHeavyAtoms()
    result["n_heavy_atoms"] = float(n_heavy)

    star_count = smiles.count("*")
    result["star_count"] = float(star_count)
    result["is_branched"] = 1.0 if star_count > 2 else 0.0

    ri = mol.GetRingInfo()
    n_rings = ri.NumRings()
    n_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    result["num_rings"] = float(n_rings)
    result["num_aromatic_rings"] = float(n_aromatic_rings)
    result["aromatic_fraction"] = float(n_aromatic_rings / max(n_rings, 1))

    atom_counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        atom_counts[sym] = atom_counts.get(sym, 0) + 1

    for elem in ["F", "Cl", "Br", "I", "Si", "O", "N", "S", "P"]:
        result[f"atom_{elem}"] = float(atom_counts.get(elem, 0))
        result[f"atom_{elem}_frac"] = float(atom_counts.get(elem, 0) / max(n_heavy, 1))

    n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_bonds = mol.GetNumBonds()
    result["rotatable_bonds"] = float(n_rotatable)
    result["rotatable_fraction"] = float(n_rotatable / max(n_bonds, 1))
    result["flexibility_index"] = float(n_rotatable / max(n_heavy, 1))

    result["tpsa"] = float(Descriptors.TPSA(mol))
    result["logp"] = float(Descriptors.MolLogP(mol))
    result["mw"] = float(Descriptors.MolWt(mol))

    n_chiral = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    result["num_chiral_centers"] = float(n_chiral)
    result["has_stereo"] = 1.0 if "@" in smiles else 0.0

    return result


def _empty_descriptors() -> dict[str, float]:
    keys = [
        "n_heavy_atoms", "star_count", "is_branched",
        "num_rings", "num_aromatic_rings", "aromatic_fraction",
        "atom_F", "atom_Cl", "atom_Br", "atom_I", "atom_Si",
        "atom_O", "atom_N", "atom_S", "atom_P",
        "atom_F_frac", "atom_Cl_frac", "atom_Br_frac", "atom_I_frac",
        "atom_Si_frac", "atom_O_frac", "atom_N_frac", "atom_S_frac", "atom_P_frac",
        "rotatable_bonds", "rotatable_fraction", "flexibility_index",
        "tpsa", "logp", "mw",
        "num_chiral_centers", "has_stereo",
    ]
    return {k: 0.0 for k in keys}
