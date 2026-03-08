# A collection of utility functions for the MPNN

import logging
logging.getLogger("rdkit").setLevel(logging.ERROR)

from rdkit import Chem # type: ignore
import torch # type: ignore
from torch_geometric.data import Data # type: ignore

# this function gets the numeric features for a single atom
def get_node_features(atom):
    types = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Na', 'Ca', 'H']
    atom_type = [int(atom.GetSymbol() == t) for t in types]
    atom_type.append(1 if atom.GetSymbol() not in types else 0)
    is_aromatic = [1 if atom.GetIsAromatic() else 0]
    charge = [atom.GetFormalCharge()]
    hybrid = atom.GetHybridization()
    hyb_types = [Chem.rdchem.HybridizationType.SP,  Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]
    hybridization = [int(hybrid == h) for h in hyb_types]
    hybridization.append(1 if hybrid not in hyb_types else 0)
    return atom_type + is_aromatic + charge + hybridization

# this function converts a SMILES string to a graph
def smiles_to_graph(smiles: str, target: float, extra_features: list[float]) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    node_features = [get_node_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_features, dtype=torch.float)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    # Target and Extra Tabular Features
    y = torch.tensor([target], dtype=torch.float) # classification target (0 or 1)
    u = torch.tensor([extra_features], dtype=torch.float) # LogP, TPSA, etc.
    return Data(x=x, edge_index=edge_index, y=y, u=u)

