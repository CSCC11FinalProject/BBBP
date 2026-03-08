from dataloader import BBBPDataset
from mpnn import MPNN

import os
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from torch.utils.data import random_split  # type: ignore
from torchmetrics import AUROC, F1Score  # type: ignore
from tqdm import tqdm  # type: ignore

CSV_DIR = os.path.dirname(os.path.abspath(__file__))
CSV     = os.path.join(CSV_DIR, '..', 'dataset', 'bbbp.csv')
CHECKPOINT_DIR = os.path.join(CSV_DIR, 'checkpoints')
SEED    = 67
PATIENCE = 10

def train_epoch(model: MPNN, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device, epoch: int) -> float:
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
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader.dataset)

def evaluate(
    model: MPNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    auroc_metric = AUROC(task="binary").to(device)
    f1_metric = F1Score(task="binary").to(device)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            batch = batch.to(device)
            logits = model(batch).squeeze(1)
            total_loss += criterion(logits, batch.y).item() * len(batch.y)
            probs = logits.sigmoid()
            auroc_metric.update(probs, batch.y.long())
            f1_metric.update(probs, batch.y.long())
    n = len(loader.dataset)
    auc = float(auroc_metric.compute())
    f1 = float(f1_metric.compute())
    auroc_metric.reset()
    f1_metric.reset()
    return total_loss / n, auc, f1

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    dataset = BBBPDataset(CSV)
    n       = len(dataset)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)
    print(f"Dataset split — train: {n_train}, val: {n_val}, test: {n_test}")

    # Class balance for BCEWithLogitsLoss: pos_weight = n_neg/n_pos on training set
    train_targets = [dataset.df.iloc[i]['p_np'] for i in train_ds.indices]
    n_pos = sum(1 for t in train_targets if t == 1)
    n_neg = len(train_targets) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float, device=device)

    model = MPNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_auc = 0.0
    best_state = None
    epochs_no_improve = 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best.pt')

    for epoch in tqdm(range(1, 201), desc="Training"):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_auc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_auc)
        tqdm.write(f"Epoch {epoch:03d} | train loss: {train_loss:.4f} | "
                   f"val loss: {val_loss:.4f} | val AUC: {val_auc:.4f} | val F1: {val_f1:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            torch.save({
                'state_dict': best_state,
                'best_val_auc': best_val_auc,
                'epoch': epoch,
            }, checkpoint_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val AUC improvement for {PATIENCE} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    print(f"Best model loaded from {checkpoint_path}")
    test_loss, test_auc, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"\nTest loss: {test_loss:.4f} | Test AUC-ROC: {test_auc:.4f} | Test F1: {test_f1:.4f}")
