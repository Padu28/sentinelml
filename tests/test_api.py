"""API tests. Skipped automatically when FastAPI / httpx are not installed."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    os.environ["SENTINELML_DB"] = str(tmp_path_factory.mktemp("db") / "api.db")
    from sentinelml.common.config import get_settings

    get_settings.cache_clear()
    from fastapi.testclient import TestClient

    from sentinelml.ingestion.loaders import load_stream
    from sentinelml.models.train import train_model
    from sentinelml.retraining.registry import ModelRegistry
    from sentinelml.storage.database import Database

    s = get_settings()
    reg = ModelRegistry(Database(s.db_path), tmp_path_factory.mktemp("models"))
    df = load_stream("sudden", 0)
    reg.register(train_model(df[df.batch_id < 12]), "logistic_regression", {"roc_auc": 0.9}, 3000, status="champion")

    import api.main as main

    main.registry = reg
    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_with_explanation(client):
    payload = {"record": {"sensor_a": 1.2, "sensor_b": -0.3, "sensor_c": 0.5, "load_kw": 80, "temp_c": 25,
                          "region": "north"}, "explain": True}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["top_factors"]


def test_validate_flags_bad_batch(client):
    recs = [{"timestamp": "2026-01-01", "sensor_a": 0, "sensor_b": None, "sensor_c": 0, "load_kw": -5,
             "temp_c": 999, "region": "mars"}] * 3
    r = client.post("/data/validate", json={"records": recs})
    assert r.status_code == 200
    assert r.json()["passed"] is False


def test_validate_rejects_empty(client):
    assert client.post("/data/validate", json={"records": []}).status_code == 422
