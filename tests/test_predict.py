
VALID_CLASSES = {"bird", "cat", "deer", "dog", "frog", "horse"}


def test_predict_valid_id(client, valid_image_id):
    resp = client.post("/predict", json={"image_id": valid_image_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["image_id"] == valid_image_id
    assert data["predicted_class"] in VALID_CLASSES
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["true_label"] in VALID_CLASSES
    assert isinstance(data["is_correct"], bool)


def test_predict_probabilities_sum(client, valid_image_id):
    resp = client.post("/predict", json={"image_id": valid_image_id})
    probs = resp.json()["probabilities"]
    assert set(probs.keys()) == VALID_CLASSES
    assert abs(sum(probs.values()) - 1.0) < 1e-5


def test_predict_unknown_id_returns_404(client):
    resp = client.post("/predict", json={"image_id": "999999"})
    assert resp.status_code == 404


def test_predict_empty_body_returns_422(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_predict_missing_body_returns_422(client):
    resp = client.post("/predict")
    assert resp.status_code == 422
