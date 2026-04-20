from fastapi import APIRouter

router = APIRouter()


@router.get("/sample")
def sample():
    return {"message": "not implemented yet"}
