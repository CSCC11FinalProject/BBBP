from torch.utils.data import Dataset  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from torch_geometric.data import Data  # type: ignore
import pandas as pd  # type: ignore
from utils import smiles_to_graph

import warnings

warnings.filterwarnings("ignore")


class BBBPDataset(Dataset):
    def __init__(self, csv_file: str, usecols: list[str] | None = None):
        super().__init__()
        if usecols is None:
            usecols = [
                "smiles",
                "p_np",
                "LogP",
                "TPSA",
                "MW",
                "HBA",
                "HBD",
                "RotatableBonds",
                "Charge",
            ]
        self.df = pd.read_csv(csv_file, usecols=usecols).dropna()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Data:
        row = self.df.iloc[idx]
        extra = [
            row["LogP"],
            row["TPSA"],
            row["MW"],
            row["HBA"],
            row["HBD"],
            row["RotatableBonds"],
            row["Charge"],
        ]
        data = smiles_to_graph(row["smiles"], row["p_np"], extra)
        return data


if __name__ == "__main__":
    # initialize loader
    dataset = BBBPDataset("../dataset/BBBP.csv")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Dataset loaded successfully with {len(dataset)} samples.")

    # save distributions of the data and correlation matrix etc to the plots directory
    import os
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore

    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = dataset.df
    sns.set_theme(style="darkgrid")

    # correlation matrix on core numeric descriptors
    plt.figure(figsize=(8, 8))
    corr_cols = ["LogP", "TPSA", "MW", "HBA", "HBD", "RotatableBonds", "Charge", "p_np"]
    numerical_df = df[corr_cols]
    corr = numerical_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "correlation_matrix.png"))
    plt.close()