import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.utils import PROJECT_ROOT

def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3, patience=5, *, checkpoint_name):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device : {device}")
    checkpoint_path = PROJECT_ROOT / "models" / checkpoint_name
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1:02d}/{epochs}", leave=False)
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (out.argmax(1) == y).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                val_loss += criterion(out, y).item()
                val_correct += (out.argmax(1) == y).sum().item()

        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        tl = train_loss / len(train_loader)
        vl = val_loss / len(val_loader)
        ta = train_correct / n_train
        va = val_correct / n_val

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} — loss: {tl:.4f} — val_loss: {vl:.4f} — val_acc: {va:.4f}"
        )

        # Early stopping + checkpoint
        if vl < best_val_loss:
            best_val_loss = vl
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping à l'epoch {epoch + 1}")
                break

    return history
