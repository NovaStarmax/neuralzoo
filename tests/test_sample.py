VALID_CLASSES = {"bird", "cat", "deer", "dog", "frog", "horse"}


def test_sample_default_count(client):
    resp = client.get("/sample")
    assert resp.status_code == 200
    assert len(resp.json()) == 9


def test_sample_custom_count(client):
    resp = client.get("/sample", params={"n": 5})
    assert resp.status_code == 200
    assert len(resp.json()) == 5


def test_sample_image_fields(client):
    resp = client.get("/sample", params={"n": 1})
    img = resp.json()[0]
    assert "id" in img
    assert "base64" in img
    assert "true_label" in img
    assert img["true_label"] in VALID_CLASSES


def test_sample_class_filter(client):
    resp = client.get("/sample", params={"classes": ["cat", "dog"], "n": 10})
    assert resp.status_code == 200
    for img in resp.json():
        assert img["true_label"] in {"cat", "dog"}


def test_sample_n_too_large_clamped(client):
    resp = client.get("/sample", params={"n": 20})
    assert resp.status_code == 200
    assert len(resp.json()) <= 20


def test_sample_n_zero_returns_422(client):
    resp = client.get("/sample", params={"n": 0})
    assert resp.status_code == 422
