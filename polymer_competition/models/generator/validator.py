from __future__ import annotations

import logging
import os

os.environ["RDKIT_SKIP_VALIDATION_WARNINGS"] = "1"
logging.getLogger("rdkit").setLevel(logging.ERROR)

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


class MoleculeValidator:
    def validate(self, smiles: str) -> tuple[bool, str]:
        if not smiles or len(smiles.strip()) == 0:
            return False, "empty SMILES string"

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "stage 1 failed: MolFromSmiles returned None"

        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            return False, f"stage 2 failed: sanitize ({e})"

        for atom in mol.GetAtoms():
            try:
                ev = atom.GetExplicitValence()
                tv = atom.GetTotalValence()
                if ev > tv and tv > 0:
                    return (
                        False,
                        f"stage 3 failed: valence violation atom {atom.GetIdx()} "
                        f"({atom.GetSymbol()}) explicit={ev} > total={tv}",
                    )
            except Exception as e:
                log = logging.getLogger(__name__)
                log.warning("Failed to validate atom valence: %s", e)
                pass

        try:
            Chem.Kekulize(mol)
        except Exception as e:
            return False, f"stage 4 failed: kekulize ({e})"

        edge_set = set()
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge = (min(i, j), max(i, j))
            if edge in edge_set:
                return (
                    False,
                    f"stage 5a failed: duplicate edge between atoms {i} and {j}",
                )
            edge_set.add(edge)

        frags = Chem.GetMolFrags(mol)
        if len(frags) > 1:
            return (
                False,
                f"stage 5b failed: {len(frags)} disconnected fragments",
            )

        return True, "valid"
