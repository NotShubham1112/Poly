"""Periodic polymer graph generation for improved property prediction."""

from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
from typing import List, Tuple


def parse_smiles_with_stars(smiles: str) -> Tuple[str, List[int]]:
    """Parse SMILES and identify connection points (*)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    stars = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            stars.append(atom.GetIdx())
    
    return smiles, stars


def generate_oligomer_smiles(smiles: str, n_repeats: int = 3) -> str:
    """
    Generate oligomer SMILES with N repeats for periodic polymer graphs.
    
    Example:
        *CCO* → *CCOCCOCCO* (3 repeats)
        *c1ccc(O)cc1* → *c1ccc(O)cc1c1ccc(O)cc1c1ccc(O)cc1* (3 repeats)
    
    Args:
        smiles: SMILES string with * connection points
        n_repeats: Number of repeat units (default: 3)
    
    Returns:
        Expanded oligomer SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Find connection points
    stars = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "*"]
    if len(stars) != 2:
        raise ValueError(f"Expected 2 connection points, got {len(stars)}")
    
    # Get the repeat unit (remove * atoms)
    atoms_to_remove = stars
    rw_mol = Chem.RWMol(mol)
    
    # Remove * atoms and their bonds
    for idx in sorted(atoms_to_remove, reverse=True):
        rw_mol.RemoveAtom(idx)
    
    repeat_smiles = Chem.MolToSmiles(rw_mol)
    
    # Build oligomer: * + repeat * (repeat) * 
    oligomer = "*" + repeat_smiles * n_repeats + "*"
    
    return oligomer


def build_periodic_graph(mol: Chem.Mol, n_repeats: int = 3) -> nx.Graph:
    """
    Build a periodic polymer graph from a monomer.
    
    Args:
        mol: RDKit molecule with * connection points
        n_repeats: Number of repeat units
    
    Returns:
        NetworkX graph representing periodic polymer
    """
    smiles = Chem.MolToSmiles(mol)
    oligomer_smiles = generate_oligomer_smiles(smiles, n_repeats)
    
    # Parse oligomer
    oligomer_mol = Chem.MolFromSmiles(oligomer_smiles)
    if oligomer_mol is None:
        raise ValueError(f"Failed to parse oligomer: {oligomer_smiles}")
    
    # Convert to graph
    graph = nx.Graph()
    
    for atom in oligomer_mol.GetAtoms():
        if atom.GetSymbol() != "*":  # Skip connection points
            graph.add_node(
                atom.GetIdx(),
                symbol=atom.GetSymbol(),
                degree=atom.GetDegree(),
                formal_charge=atom.GetFormalCharge(),
                hybridization=str(atom.GetHybridization()),
                is_aromatic=atom.GetIsAromatic()
            )
    
    for bond in oligomer_mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        # Skip if either atom is a *
        begin_atom = oligomer_mol.GetAtomWithIdx(begin_idx)
        end_atom = oligomer_mol.GetAtomWithIdx(end_idx)
        
        if begin_atom.GetSymbol() != "*" and end_atom.GetSymbol() != "*":
            graph.add_edge(
                begin_idx,
                end_idx,
                bond_type=str(bond.GetBondType()),
                is_conjugated=bond.GetIsConjugated(),
                is_in_ring=bond.IsInRing()
            )
    
    return graph


def get_periodic_smiles_list(smiles_list: List[str], n_repeats: int = 3) -> List[str]:
    """
    Convert list of monomer SMILES to oligomer SMILES.
    
    Args:
        smiles_list: List of SMILES with * connection points
        n_repeats: Number of repeat units
    
    Returns:
        List of oligomer SMILES
    """
    result = []
    for smiles in smiles_list:
        try:
            oligomer = generate_oligomer_smiles(smiles, n_repeats)
            result.append(oligomer)
        except Exception as e:
            print(f"Warning: Failed to process {smiles}: {e}")
            result.append(smiles)  # Fallback to original
    return result
