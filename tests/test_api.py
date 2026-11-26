"""API tests. They train a tiny model on made-up rows, so no CSV is needed."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from coldstore.train import fit

PARAMS = {
    "target": "spoilage_pct",
    "features": {
        "numeric": ["storage_days", "temp_set_c", "humidity_pct", "power_cuts_hours",
                    "bag_weight_kg", "door_openings_per_day"],
        "categorical": ["crop", "chamber"],
    },
    "model": {"random_state": 0},
}

GOOD_LOT = {
    "storage_days": 120, "temp_set_c": 5.0, "humidity_pct": 85, "power_cuts_hours": 30.0,
    "bag_weight_kg": 50.0, "door_openings_per_day": 20, "crop": "potato", "chamber": "Chamber 1",
}


def tiny_model():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "storage_days": rng.integers(5, 180, n),
        "temp_set_c": rng.uniform(1, 10, n),
        "humidity_pct": rng.integers(70, 98, n),
        "power_cuts_hours": rng.uniform(0, 60, n),
        "bag_weight_kg": rng.uniform(40, 60, n),
        "door_openings_per_day": rng.integers(0, 40, n),
        "crop": rng.choice(["potato", "onion", "apple"], n),
        "chamber": rng.choice(["Chamber 1", "Chamber 2", "Chamber 3"], n),
    })
    df["spoilage_pct"] = 0.05 * df.storage_days + 0.5 * df.temp_set_c
    return fit(df, {"type": "ridge", "alpha": 1.0}, PARAMS)


@pytest.fixture
def client():
    app.state.model = tiny_model()
    with TestClient(app) as c:
        yield c
    app.state.model = None


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_a_number(client):
    r = client.post("/predict", json=GOOD_LOT)
    assert r.status_code == 200
    assert isinstance(r.json()["spoilage_pct"], float)


def test_storage_days_above_350_is_rejected(client):
    assert client.post("/predict", json={**GOOD_LOT, "storage_days": 351}).status_code == 422
    assert client.post("/predict", json={**GOOD_LOT, "storage_days": 350}).status_code == 200


def test_bag_weight_below_20_is_rejected(client):
    assert client.post("/predict", json={**GOOD_LOT, "bag_weight_kg": 19.9}).status_code == 422
    assert client.post("/predict", json={**GOOD_LOT, "bag_weight_kg": 20}).status_code == 200


def test_missing_or_extra_field_is_rejected(client):
    missing = {k: v for k, v in GOOD_LOT.items() if k != "crop"}
    assert client.post("/predict", json=missing).status_code == 422
    leak = {**GOOD_LOT, "spoilage_found_at_unloading": 10.0}
    assert client.post("/predict", json=leak).status_code == 422
