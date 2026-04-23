import os
import pickle
from pathlib import Path

import numpy as np

ANIMAL_CIFAR_LABELS = {2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse"}


def _find_test_batch() -> Path:
    candidates = [
        Path(os.environ.get("DATA_PATH", "")) / "test_batch",
        Path("data/test_batch"),
        Path("/app/data/test_batch"),
        Path("test_batch"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("CIFAR-10 test_batch not found")


def load_dataset() -> dict[str, dict]:
    """Load CIFAR-10 test set, keep only animal classes. Returns {id: {image, true_label}}."""
    path = _find_test_batch()
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    images = batch[b"data"]
    labels = batch[b"labels"]

    dataset = {}
    for idx, (flat, label) in enumerate(zip(images, labels)):
        if label not in ANIMAL_CIFAR_LABELS:
            continue
        image = flat.reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)
        dataset[str(idx)] = {"image": image, "true_label": ANIMAL_CIFAR_LABELS[label]}

    return dataset
