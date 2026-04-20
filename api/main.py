from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import health, predict, sample, metrics

app = FastAPI(title="NeuralZOO API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(sample.router)
app.include_router(metrics.router)
