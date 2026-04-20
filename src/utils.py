import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import pickle
from pathlib import Path

ANIMAL_CLASSES = {'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7}
ANIMAL_INDICES = list(ANIMAL_CLASSES.values())
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "cifar-10-batches-py"

def unpickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f, encoding='bytes')

def load_animal_data(data_dir=DATA_DIR):
    # Train
    all_data, all_labels = [], []
    for i in range(1, 6):
        d = unpickle(f'{data_dir}/data_batch_{i}')
        all_data.append(d[b'data'])
        all_labels.extend(d[b'labels'])
    X_train = np.vstack(all_data)
    y_train = np.array(all_labels)

    # Test
    test = unpickle(f'{data_dir}/test_batch')
    X_test = test[b'data']
    y_test = np.array(test[b'labels'])

    # Filtrage animaux
    mask_train = np.isin(y_train, ANIMAL_INDICES)
    mask_test  = np.isin(y_test,  ANIMAL_INDICES)

    return (
        X_train[mask_train], y_train[mask_train],
        X_test[mask_test],   y_test[mask_test]
    )

def prepare_data(X, y, class_indices, batch_size=64, val_split=0.1):
    # Remap labels vers 0..N-1
    label_map = {idx: i for i, idx in enumerate(sorted(class_indices))}
    y_mapped = np.array([label_map[label] for label in y])

    # Normalisation [0,255] → [0,1] puis reshape (N,3,32,32)
    X_norm = X.astype(np.float32) / 255.0
    X_reshaped = X_norm.reshape(-1, 3, 32, 32)

    X_tensor = torch.tensor(X_reshaped)
    y_tensor = torch.tensor(y_mapped, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    val_size   = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size),
    )


def make_test_loader(X, y, class_indices, batch_size=64):
    label_map = {idx: i for i, idx in enumerate(sorted(class_indices))}
    y_mapped = np.array([label_map[label] for label in y])
    X_norm = X.astype(np.float32) / 255.0
    X_tensor = torch.tensor(X_norm.reshape(-1, 3, 32, 32))
    y_tensor = torch.tensor(y_mapped, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def export_model(model, path=None):
    if path is None:
        path = PROJECT_ROOT / "models" / "cnn_cifar10.pth"
    torch.save(model.state_dict(), path)
    print(f"Modèle exporté → {path}")