import base64
import io
import random

import numpy as np
from fastapi import APIRouter, Query
from PIL import Image

from api import state

router = APIRouter()

VALID_CLASSES = {"bird", "cat", "deer", "dog", "frog", "horse"}


def _image_to_base64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@router.get("/sample")
def sample(
    classes: list[str] = Query(default=[]),
    n: int = Query(default=9, ge=1, le=20),
):
    pool = state.dataset
    if classes:
        requested = {c.lower() for c in classes} & VALID_CLASSES
        pool = {k: v for k, v in pool.items() if v["true_label"] in requested}

    keys = random.sample(list(pool.keys()), min(n, len(pool)))
    return [
        {
            "id": key,
            "base64": _image_to_base64(pool[key]["image"]),
            "true_label": pool[key]["true_label"],
        }
        for key in keys
    ]
