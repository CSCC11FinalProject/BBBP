# I decided to generate all descriptors at first, while use LASSO to select the most relevant ones for building model.

from rdkit import rdBase  # type: ignore
rdBase.DisableLog("rdApp.*")

from rdkit.Chem import Descriptors  # type: ignore
from rdkit import Chem  # type: ignore

import pandas as pd # type: ignore

def get_all_rdkit_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        mol = Chem.AddHs(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    desc = {}
    for name, func in Descriptors.descList:
        try:
            desc[name] = func(mol)
        except Exception:
            desc[name] = None

    return desc

def process_dataset(path: str):
    df = pd.read_csv(path)
    descriptors = [get_all_rdkit_descriptors(smiles) or {} for smiles in df['smiles']]
    descriptor_df = pd.DataFrame(descriptors)
    df = pd.concat([df, descriptor_df], axis=1)
    import os
    base, ext = os.path.splitext(path)
    output_path = f"{base}_all_rdkit_descriptors{ext}"
    df.to_csv(output_path, index=False)
    print(f"Saved augmented dataset to: {output_path}")

if __name__ == "__main__":
    process_dataset(".../BBBP.csv")