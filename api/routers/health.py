from fastapi import APIRouter

from api import state

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": state.model is not None}
