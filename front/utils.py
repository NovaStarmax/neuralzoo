import requests

from config import API_BASE_URL


def get_sample(classes: list[str], n: int) -> list[dict]:
    params = {"n": n}
    for c in classes:
        params.setdefault("classes", [])
        if isinstance(params["classes"], list):
            params["classes"].append(c)
    resp = requests.get(f"{API_BASE_URL}/sample", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def predict(image_id: str) -> dict:
    resp = requests.post(
        f"{API_BASE_URL}/predict",
        json={"image_id": image_id},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def get_metrics() -> dict:
    resp = requests.get(f"{API_BASE_URL}/metrics", timeout=10)
    resp.raise_for_status()
    return resp.json()
