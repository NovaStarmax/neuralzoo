from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api import state
from api.ml.inference import run_inference

router = APIRouter()


class PredictRequest(BaseModel):
    image_id: str


@router.post("/predict")
def predict(body: PredictRequest):
    entry = state.dataset.get(body.image_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"image_id '{body.image_id}' not found")

    try:
        result = run_inference(state.model, entry["image"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    true_label = entry["true_label"]
    return {
        "image_id": body.image_id,
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "true_label": true_label,
        "is_correct": result["predicted_class"] == true_label,
    }
