EXPECTED_CLASSES = ["bird", "cat", "deer", "dog", "frog", "horse"]


def test_metrics_status(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_metrics_accuracy(client):
    data = client.get("/metrics").json()
    assert 0.0 <= data["accuracy"] <= 1.0


def test_metrics_confusion_matrix(client):
    data = client.get("/metrics").json()
    cm = data["confusion_matrix"]
    assert len(cm) == 6
    assert all(len(row) == 6 for row in cm)


def test_metrics_class_labels(client):
    data = client.get("/metrics").json()
    assert data["class_labels"] == EXPECTED_CLASSES


def test_metrics_classification_report(client):
    data = client.get("/metrics").json()
    report = data["classification_report"]
    for cls in EXPECTED_CLASSES:
        assert cls in report
        assert "precision" in report[cls]
        assert "recall" in report[cls]
        assert "f1-score" in report[cls]


def test_metrics_history(client):
    data = client.get("/metrics").json()
    history = data["history"]
    assert "train_loss" in history
    assert "val_loss" in history
    assert len(history["train_loss"]) == len(history["val_loss"])
