import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from src.utils import PROJECT_ROOT


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=30,
    lr=1e-3,
    patience=5,
    weight_decay=0.0,
    *,
    checkpoint_name=None,
    experiment_name=None,
):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device : {device}")
    if experiment_name is not None:
        exp_dir = PROJECT_ROOT / "models" / "experiments" / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = exp_dir / "weights.pth"
        with open(exp_dir / "config.json", "w") as f:
            json.dump(
                {
                    "experiment": experiment_name,
                    "epochs": epochs,
                    "lr": lr,
                    "patience": patience,
                    "weight_decay": weight_decay,
                    "scheduler": "ReduceLROnPlateau",
                },
                f,
                indent=2,
            )
    else:
        checkpoint_path = PROJECT_ROOT / "models" / checkpoint_name
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

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

        scheduler.step(vl)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:02d}/{epochs} — "
            f"loss: {tl:.4f} — val_loss: {vl:.4f} — "
            f"val_acc: {va:.4f} — lr: {current_lr:.2e}"
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
