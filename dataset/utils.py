# this function gets the 2048 bit morgan fingerprint for a given smiles string
# we do not add this to the CSV explicitly,
# but you will need to use it instead of the smiles string
# for KNN, NaiveBayes, Random Forest
# it is your job to figure out how to use it in the models!

from rdkit import Chem  # type: ignore
from rdkit.Chem import AllChem  # type: ignore
import numpy as np  # type: ignore

def get_morgan_fingerprint(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    return np.array([int(b) for b in fp.ToBitString()], dtype=np.float32)