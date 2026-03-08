from torch.utils.data import Dataset # type: ignore
from torch_geometric.loader import DataLoader # type: ignore
from torch_geometric.data import Data # type: ignore
import pandas as pd # type: ignore
from utils import smiles_to_graph

class BBBPDataset(Dataset):
    def __init__(self, csv_file: str):
        super().__init__()
        self.df = pd.read_csv(csv_file).dropna() # Ensure no NaNs in LogP/TPSA
        
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Data:
        row = self.df.iloc[idx]
        # Pull extra features we computed earlier
        extra = [row['LogP'], row['TPSA'], row['MW'], row['HBA'], row['HBD']]
        data = smiles_to_graph(row['smiles'], row['p_np'], extra)
        return data

# Initialize loader
if __name__ == "__main__":
    dataset = BBBPDataset('dataset/bbbp.csv')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Dataset loaded successfully with {len(dataset)} samples.")