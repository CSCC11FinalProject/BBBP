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

    # class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x="p_np", data=df, palette="viridis")
    plt.title("Distribution of BBB Permeability (p_np)")
    plt.xlabel("Permeability (0: No, 1: Yes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "class_distribution.png"))
    plt.close()

    # key feature distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(ax=axes[0], x="p_np", y="LogP", data=df, palette="Set2")
    axes[0].set_title("LogP vs Permeability")
    sns.boxplot(ax=axes[1], x="p_np", y="TPSA", data=df, palette="Set2")
    axes[1].set_title("TPSA vs Permeability")
    sns.boxplot(ax=axes[2], x="p_np", y="MW", data=df, palette="Set2")
    axes[2].set_title("Molecular Weight vs Permeability")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_distributions.png"))
    plt.close()

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