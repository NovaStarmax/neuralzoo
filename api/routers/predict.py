from fastapi import APIRouter

router = APIRouter()


@router.post("/predict")
def predict():
    return {"message": "not implemented yet"}
