# Hyperparameter tuning for the MPNN using Optuna

import os
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from torch.utils.data import random_split  # type: ignore
from torchmetrics import AUROC, F1Score  # type: ignore
import optuna  # type: ignore

from dataloader import BBBPDataset
from mpnn import MPNN
from tqdm import tqdm  # type: ignore

CSV_DIR = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(CSV_DIR, "..", "dataset", "BBBP.csv")
CHECKPOINT_DIR = os.path.join(CSV_DIR, "checkpoints")
SEED = 67
PATIENCE = 10
TRIAL_MAX_EPOCHS = 20
FINAL_MAX_EPOCHS = 201


def train_epoch(
    model: MPNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch).squeeze(1)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch.y)
    return total_loss / len(loader.dataset)


def evaluate(model: MPNN, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    auroc_metric = AUROC(task="binary").to(device)
    f1_metric = F1Score(task="binary").to(device)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch).squeeze(1)
            total_loss += criterion(logits, batch.y).item() * len(batch.y)
            probs = logits.sigmoid()
            auroc_metric.update(probs, batch.y.long())
            f1_metric.update(probs, batch.y.long())
    n = len(loader.dataset)
    auc = float(auroc_metric.compute())
    f1 = float(f1_metric.compute())
    return total_loss / n, auc, f1


def run_trial(
    trial: optuna.Trial,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weight: torch.Tensor,
) -> float:
    num_layers = trial.suggest_categorical("num_layers", [3, 4])
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    gin_dropout = trial.suggest_categorical("gin_dropout", [0.1, 0.2, 0.3])
    fusion_dropout = trial.suggest_categorical("fusion_dropout", [0.2, 0.3, 0.4, 0.5])

    model = MPNN(
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        gin_dropout=gin_dropout,
        fusion_dropout=fusion_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, TRIAL_MAX_EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_auc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_auc)

        # report intermediate AUC to Optuna and allow pruning of bad trials
        trial.report(val_auc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return best_val_auc


def objective(trial: optuna.Trial) -> float:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset = BBBPDataset(CSV)
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    train_targets = [dataset.df.iloc[i]["p_np"] for i in train_ds.indices]
    n_pos = sum(1 for t in train_targets if t == 1)
    n_neg = len(train_targets) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float, device=device)

    return run_trial(trial, device, train_loader, val_loader, pos_weight)


if __name__ == "__main__":
    n_trials = 16  # 16 trials across discrete grid
    study = optuna.create_study(
        direction="maximize",
        study_name="mpnn_bbbp",
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\nBest trial:")
    print(f"  Val AUC-ROC: {study.best_value:.4f}")
    print("  Params:", study.best_params)

    # Train final model with best params and save
    best = study.best_params
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset = BBBPDataset(CSV)
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    train_targets = [dataset.df.iloc[i]["p_np"] for i in train_ds.indices]
    n_pos = sum(1 for t in train_targets if t == 1)
    n_neg = len(train_targets) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float, device=device)

    model = MPNN(
        hidden_channels=best["hidden_channels"],
        num_layers=best["num_layers"],
        gin_dropout=best["gin_dropout"],
        fusion_dropout=best["fusion_dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0.0
    best_state = None
    epochs_no_improve = 0
    for epoch in range(1, FINAL_MAX_EPOCHS + 1):
        train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_auc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_auc)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, "best_tuned.pt")
    torch.save({
        "state_dict": best_state,
        "best_val_auc": best_val_auc,
        "params": best,
    }, path)
    test_loss, test_auc, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"\nBest model saved to {path}")
    print(f"Test loss: {test_loss:.4f} | Test AUC-ROC: {test_auc:.4f} | Test F1: {test_f1:.4f}")
