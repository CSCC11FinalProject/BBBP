from dataloader import BBBPDataset
from mpnn import MPNN

import torch # type: ignore
import torch.nn as nn # type: ignore
from torch_geometric.loader import DataLoader # type: ignore
from torch.utils.data import random_split # type: ignore

CSV  = '../dataset/bbbp.csv'
SEED = 42

def smoke_test(model: MPNN, loader: DataLoader, device: torch.device):
    model.eval()
    batch = next(iter(loader))
    batch = batch.to(device)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (len(batch.y), 1), f"Unexpected output shape: {out.shape}"
    print(f"Smoke test passed — batch size {len(batch.y)}, output shape {tuple(out.shape)}, "
          f"range [{out.min():.3f}, {out.max():.3f}]")

def train_epoch(model: MPNN, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch).squeeze(1)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch.y)
    return total_loss / len(loader.dataset)

def evaluate(model: MPNN, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).squeeze(1)
            total_loss += criterion(out, batch.y).item() * len(batch.y)
            correct += ((out >= 0.5).float() == batch.y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n

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

    model     = MPNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    smoke_test(model, train_loader, device)

    for epoch in range(1, 51):
        train_loss             = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc      = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d} | train loss: {train_loss:.4f} | "
              f"val loss: {val_loss:.4f} | val acc: {val_acc:.4f}")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
