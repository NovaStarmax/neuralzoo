import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import state
from api.data.loader import load_dataset
from api.ml.model import load_model
from api.routers import health, metrics, predict, sample


def _find_model_path() -> str:
    candidates = [
        os.environ.get("MODEL_PATH", ""),
        "models/cnn_cifar10.pth",
        "/app/models/cnn_cifar10.pth",
        "cnn_cifar10.pth",
    ]
    for p in candidates:
        if p and Path(p).exists():
            return p
    raise FileNotFoundError("cnn_cifar10.pth not found")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.model = load_model(_find_model_path())
    state.dataset = load_dataset()
    yield
    state.model = None
    state.dataset = {}


app = FastAPI(title="NeuralZOO API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(sample.router)
app.include_router(predict.router)
app.include_router(metrics.router)
