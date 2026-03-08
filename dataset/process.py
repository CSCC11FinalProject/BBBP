# we want to augment each molecule by adding the following information:
# LogP (lipophilicity)
# TPSA (topological polar surface area)
# Molecular Weight
# Number of Hydrogen Bond Acceptors
# Number of Hydrogen Bond Donors
# Number of Rotatable Bonds
# Formal Charge

import logging
logging.getLogger("rdkit").setLevel(logging.ERROR)

from rdkit.Chem import Descriptors # type: ignore
from rdkit import Chem # type: ignore

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd # type: ignore

def get_cns_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)
    return {
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'MW': Descriptors.MolWt(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'RotatableBonds': Descriptors.NumRotatableBonds(mol),
        'Charge': Descriptors.MaxAbsPartialCharge(mol)
    }

def process_dataset(path: str):
    df = pd.read_csv(path)
    descriptors = [get_cns_descriptors(smiles) or {} for smiles in df['smiles']]
    descriptor_df = pd.DataFrame(descriptors)
    df = pd.concat([df, descriptor_df], axis=1)
    df.to_csv(path, index=False)

if __name__ == "__main__":
    process_dataset("dataset/bbbp.csv")