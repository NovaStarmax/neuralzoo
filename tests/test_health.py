def test_health_status_code(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_body(client):
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
