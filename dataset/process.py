

from rdkit import rdBase  # type: ignore
rdBase.DisableLog("rdApp.*")

from rdkit.Chem import Descriptors  # type: ignore
from rdkit import Chem  # type: ignore

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # type: ignore

from dataset.utils import get_morgan_fingerprint  # type: ignore


def get_all_rdkit_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    features = {}
    for name, func in Descriptors.descList:
        try:
            value = func(mol)
        except Exception:
            value = None
        features[name] = value
    fp = get_morgan_fingerprint(smiles)
    if fp is not None:
        for idx in range(len(fp)):
            features[f"morgan_{idx}"] = float(fp[idx])
    return features


def process_dataset(path: str):
    df = pd.read_csv(path)
    descriptors = []
    for smiles in df["smiles"]:
        row_features = get_all_rdkit_descriptors(smiles)
        descriptors.append(row_features)
    descriptor_df = pd.DataFrame(descriptors)
    df = pd.concat([df, descriptor_df], axis=1)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    process_dataset("BBBP.csv")