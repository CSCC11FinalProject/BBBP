# A collection of utility functions for the MPNN

from rdkit import Chem # type: ignore
import torch # type: ignore
from torch_geometric.data import Data # type: ignore

# this function converts a SMILES string to a graph
def smiles_to_graph(smiles: str, target: float, extra_features: list[float]) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    # Node features (e.g., Atomic Number)
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    x = torch.tensor(atoms, dtype=torch.float).view(-1, 1)
    # Edge Indices (Bonds)
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]] # Undirected graph
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    # Target and Extra Tabular Features
    y = torch.tensor([target], dtype=torch.float) # classification target (0 or 1)
    u = torch.tensor([extra_features], dtype=torch.float) # LogP, TPSA, etc.
    return Data(x=x, edge_index=edge_index, y=y, u=u)

