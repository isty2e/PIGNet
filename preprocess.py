import io
from collections import namedtuple
from pathlib import Path
from typing import Union

import numpy as np
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.PDBIO import Select
from rdkit import Chem, RDLogger
from rdkit.Chem.SaltRemover import SaltRemover
from scipy.spatial import distance_matrix

RDLogger.DisableLog("rdApp.*")


def remove_water(mol: Chem.Mol) -> Chem.Mol:
    remover = SaltRemover(defnData="[O]")

    return remover.StripMol(mol)


def extract_pocket(
    pdb: Union[Path, str], mol_ligand: Chem.Mol, cutoff: float = 5.0
) -> Chem.Mol:
    parser = PDBParser()
    if not Path(pdb).is_file():
        return None

    structure = parser.get_structure("protein", pdb)
    ligand_coords = mol_ligand.GetConformer().GetPositions()

    class PocketResidueSelector(Select):
        def accept_residue(self, residue):
            residue_coords = np.array(
                [
                    np.array(list(atom.get_vector()))
                    for atom in residue.get_atoms()
                    if "H" not in atom.get_id()
                ]
            )

            if len(residue_coords.shape) < 2:
                print(residue)
                return 0

            min_dis = np.min(distance_matrix(residue_coords, ligand_coords))
            if min_dis < cutoff:
                return 1
            else:
                return 0

    iostream = io.StringIO()
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    pdbio.save(iostream, PocketResidueSelector())
    mol_pocket = Chem.MolFromPDBBlock(iostream.getvalue())

    mol_pocket = remove_water(mol_pocket)

    return mol_pocket


def get_mols(pdb: Union[Path, str], sdf: Union[Path, str]):
    mol_ligand = Chem.MolFromMolFile(sdf)
    mol_pocket = extract_pocket(pdb, mol_ligand)

    return namedtuple(typename="complex_mols", field_names=["pocket", "ligand"])(
        mol_pocket, mol_ligand
    )
