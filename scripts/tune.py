import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_animal_data, prepare_data, make_test_loader, ANIMAL_CLASSES, ANIMAL_INDICES, PROJECT_ROOT
from src.models.cnn import CNN
from src.train import train_model
from src.evaluate import evaluate_model, export_metrics


class _NumpyDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.tensor(X.astype(np.float32).reshape(-1, 3, 32, 32) / 255.0)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]


def _remap(y, class_indices):
    label_map = {idx: i for i, idx in enumerate(sorted(class_indices))}
    return np.array([label_map[label] for label in y])


def _build_augmented_loaders(X_train, y_train, batch_size=64, val_split=0.1):
    y_mapped = _remap(y_train, ANIMAL_INDICES)
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = _NumpyDataset(X_train, y_mapped, transform=train_transform)
    val_size = int(len(dataset) * val_split)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
    )


def _build_augmented_test_loader(X_test, y_test, batch_size=64):
    y_mapped = _remap(y_test, ANIMAL_INDICES)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = _NumpyDataset(X_test, y_mapped, transform=normalize)
    return DataLoader(dataset, batch_size=batch_size)


def main():
    parser = argparse.ArgumentParser(description="NeuralZOO — tune CNN")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.0, dest="weight_decay")
    parser.add_argument("--augment", action="store_true", default=False)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_animal_data()

    if args.augment:
        train_loader, val_loader = _build_augmented_loaders(X_train, y_train)
        test_loader = _build_augmented_test_loader(X_test, y_test)
    else:
        train_loader, val_loader = prepare_data(X_train, y_train, ANIMAL_INDICES)
        test_loader = make_test_loader(X_test, y_test, ANIMAL_INDICES)

    model = CNN()
    history = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        weight_decay=args.weight_decay,
        experiment_name=args.experiment,
    )

    exp_dir = PROJECT_ROOT / "models" / "experiments" / args.experiment
    model.load_state_dict(torch.load(exp_dir / "weights.pth", weights_only=True))

    class_names = list(ANIMAL_CLASSES.keys())
    preds, labels = evaluate_model(model, test_loader, class_names)
    export_metrics(history, preds, labels, class_names, exp_dir=exp_dir)

    accuracy = (preds == labels).mean()
    print(f"\nAccuracy finale ({args.experiment}) : {accuracy:.4f}")


if __name__ == "__main__":
    main()
