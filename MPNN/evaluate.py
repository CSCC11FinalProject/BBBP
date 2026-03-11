from dataloader import BBBPDataset
from mpnn import MPNN

import os
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from torchmetrics import AUROC, F1Score  # type: ignore
from sklearn.metrics import confusion_matrix, roc_curve  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from rdkit import Chem  # type: ignore
from rdkit.Chem import Draw  # type: ignore


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE_DIR, "..", "dataset", "bbbp.csv")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_tuned.pt")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
SEED = 67


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_test_loader() -> tuple[DataLoader, pd.DataFrame]:
    dataset = BBBPDataset(CSV)
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    _, _, test_ds = torch.utils.data.random_split(  # type: ignore[attr-defined]
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )
    # map back to the underlying CSV rows for analysis
    test_indices = test_ds.indices  # type: ignore[attr-defined]
    test_df = dataset.df.iloc[test_indices].reset_index(drop=True)  # type: ignore[attr-defined]
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    return loader, test_df


def load_model(device: torch.device) -> MPNN:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    params = checkpoint.get("params", {})
    model = MPNN(
        hidden_channels=params.get("hidden_channels", 128),
        num_layers=params.get("num_layers", 4),
        gin_dropout=params.get("gin_dropout", 0.1),
        fusion_dropout=params.get("fusion_dropout", 0.3),
    ).to(device)
    state = checkpoint.get("state_dict")
    if state is not None:
        model.load_state_dict(state)
    return model

def investigate_false_positives(test_df: pd.DataFrame, probs: np.ndarray) -> None:
    """Inspect false positives: actual 0 (BBB-) but predicted 1 (BBB+)."""
    df = test_df.copy()
    df["probs"] = probs
    df["preds"] = (df["probs"] >= 0.5).astype(int)

    fps = df[(df["p_np"] == 0) & (df["preds"] == 1)]
    tns = df[(df["p_np"] == 0) & (df["preds"] == 0)]

    print(f"Analyzing {len(fps)} false positives...")
    if fps.empty:
        return

    print(fps[["name", "smiles", "LogP", "TPSA", "MW", "probs"]])

    comparison = pd.DataFrame(
        {
            "Feature": ["LogP", "TPSA", "MW"],
            "False Positives (Mistakes)": [
                fps["LogP"].mean(),
                fps["TPSA"].mean(),
                fps["MW"].mean(),
            ],
            "True Negatives (Correct)": [
                tns["LogP"].mean(),
                tns["TPSA"].mean(),
                tns["MW"].mean(),
            ],
        }
    )
    print(comparison)

    # save full false positive rows
    fps_path = os.path.join(PLOTS_DIR, "false_positives.csv")
    fps.to_csv(fps_path, index=False)

    # save aggregated summary as well
    summary_path = os.path.join(PLOTS_DIR, "false_positive_summary.csv")
    comparison.to_csv(summary_path, index=False)

    # save example structures
    mols = [Chem.MolFromSmiles(s) for s in fps["smiles"]]
    legends = [f"Prob: {p:.2f}" for p in fps["probs"]]
    img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(300, 300), legends=legends)
    img.save(os.path.join(PLOTS_DIR, "false_positives_structures.png"))


def evaluate_on_test() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    device = get_device()
    test_loader, test_df = build_test_loader()
    model = load_model(device)
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    auroc_metric = AUROC(task="binary").to(device)
    f1_metric = F1Score(task="binary").to(device)

    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch).squeeze(1)
            loss = criterion(logits, batch.y)
            total_loss += loss.item() * len(batch.y)

            probs = logits.sigmoid()
            auroc_metric.update(probs, batch.y.long())
            f1_metric.update(probs, batch.y.long())

            all_probs.append(probs.cpu())
            all_targets.append(batch.y.cpu())

    n_samples = len(test_loader.dataset)
    test_loss = total_loss / n_samples
    test_auc = float(auroc_metric.compute())
    test_f1 = float(f1_metric.compute())

    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_targets).numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(
        f"Test loss: {test_loss:.4f} | "
        f"Test AUC-ROC: {test_auc:.4f} | "
        f"Test F1: {test_f1:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"Specificity: {specificity:.4f}"
    )

    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["BBB-", "BBB+"],
        yticklabels=["BBB-", "BBB+"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC = {test_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"))
    plt.close()

    # qualitative analysis of false positives on the held-out test set
    investigate_false_positives(test_df, y_prob)


if __name__ == "__main__":
    evaluate_on_test()