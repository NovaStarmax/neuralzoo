import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

_cached: dict | None = None


def _find_metrics_path() -> Path:
    candidates = [
        os.environ.get("METRICS_PATH", ""),
        "models/metrics.json",
        "/app/models/metrics.json",
        "metrics.json",
    ]
    for raw in candidates:
        if raw:
            p = Path(raw)
            if p.is_file():
                return p
    raise FileNotFoundError("metrics.json not found")


def _load_metrics() -> dict:
    global _cached
    if _cached is None:
        with open(_find_metrics_path()) as f:
            _cached = json.load(f)
    return _cached


@router.get("/metrics")
def metrics():
    try:
        return _load_metrics()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
