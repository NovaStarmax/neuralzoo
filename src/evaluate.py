import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from src.utils import PROJECT_ROOT

def evaluate_model(model, test_loader, class_names):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            preds = model(X).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Rapport texte
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
    )
    plt.title("Matrice de confusion")
    plt.ylabel("Réel")
    plt.xlabel("Prédit")
    plt.tight_layout()
    plt.show()

    return all_preds, all_labels


def plot_history(history, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{title} — Loss")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def export_metrics(
    history, all_preds, all_labels, class_names, path=None
):
    if path is None:
        path = PROJECT_ROOT / "models" / "metrics.json"
        
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()
    payload = {
        "accuracy": report["accuracy"],
        "confusion_matrix": cm,
        "class_labels": class_names,
        "classification_report": report,
        "history": history,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Métriques exportées → {path}")
