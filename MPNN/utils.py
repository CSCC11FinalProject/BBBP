from rdkit import rdBase  # type: ignore
rdBase.DisableLog("rdApp.*")
from rdkit import Chem  # type: ignore
import torch # type: ignore
import numpy as np # type: ignore
from torch_geometric.data import Data # type: ignore

class AtomFeaturizer:
    def __init__(self):
        self.symbols = ["B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"]
        self.valences = [0, 1, 2, 3, 4, 5, 6]
        self.hydrogens = [0, 1, 2, 3, 4]
        self.hybridizations = ["s", "sp", "sp2", "sp3"]
        self.dim = len(self.symbols) + len(self.valences) + len(self.hydrogens) + len(self.hybridizations)

    def encode(self, atom) -> np.ndarray:
        sym = atom.GetSymbol()
        val = atom.GetTotalValence()
        hyd = atom.GetTotalNumHs()
        hyb = atom.GetHybridization().name.lower()

        out = np.zeros(self.dim, dtype=np.float32)
        idx = 0
        if sym in self.symbols: out[idx + self.symbols.index(sym)] = 1.0
        idx += len(self.symbols)
        if val in self.valences: out[idx + self.valences.index(val)] = 1.0
        idx += len(self.valences)
        if hyd in self.hydrogens: out[idx + self.hydrogens.index(hyd)] = 1.0
        idx += len(self.hydrogens)
        if hyb in self.hybridizations: out[idx + self.hybridizations.index(hyb)] = 1.0
        return out

class BondFeaturizer:
    def __init__(self):
        self.bond_types = ["single", "double", "triple", "aromatic"]
        self.conjugated = [True, False]
        self.dim = len(self.bond_types) + len(self.conjugated) + 1  # +1 for None (self-loops)

    def encode(self, bond) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        if bond is None:
            out[-1] = 1.0
            return out
        
        btype = bond.GetBondType().name.lower()
        conj = bond.GetIsConjugated()

        idx = 0
        if btype in self.bond_types: out[idx + self.bond_types.index(btype)] = 1.0
        idx += len(self.bond_types)
        if conj in self.conjugated: out[idx + self.conjugated.index(conj)] = 1.0
        return out

atom_featurizer = AtomFeaturizer()
bond_featurizer = BondFeaturizer()

def smiles_to_graph(smiles: str, target: float, extra_features: list[float]) -> Data | None:
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None: return None
    flag = Chem.SanitizeMol(mol, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    node_features = []
    edge_indices = []
    edge_attrs = []

    for atom in mol.GetAtoms():
        node_features.append(atom_featurizer.encode(atom))
        edge_indices.append([atom.GetIdx(), atom.GetIdx()])
        edge_attrs.append(bond_featurizer.encode(None))

        for neighbor in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            edge_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            edge_attrs.append(bond_featurizer.encode(bond))
            
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float)
    
    y = torch.tensor([target], dtype=torch.float)
    u = torch.tensor([extra_features], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, u=u)