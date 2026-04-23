import numpy as np
import torch
import torch.nn.functional as F

CLASSES = ["bird", "cat", "deer", "dog", "frog", "horse"]

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def run_inference(model, image_array: np.ndarray) -> dict:
    """Run CNN inference on a uint8 HWC numpy array (32x32x3)."""
    tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - _MEAN) / _STD
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).tolist()

    predicted_idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASSES[predicted_idx],
        "confidence": float(probs[predicted_idx]),
        "probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs)},
    }
