"""
test_api.py
-----------
Basic tests for the FastAPI service using FastAPI's TestClient.

Run with:
    pytest -q

These tests make sure the API contracts work end-to-end. The /predict tests
are skipped automatically if no trained model is present, so the suite still
passes on a fresh checkout before training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure the project root is importable when pytest is run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app  # noqa: E402
from src.utils import MODEL_PATH  # noqa: E402

client = TestClient(app)

VALID_CUSTOMER = {
    "age": 35,
    "job": 2,
    "credit_amount": 4500,
    "duration": 24,
    "housing": "own",
    "saving_accounts": "moderate",
    "checking_account": "moderate",
    "purpose": "car",
}

model_required = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Trained model not found. Run `python -m src.train_model` first.",
)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


@model_required
def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "credit_risk_model"
    assert "model_type" in body


@model_required
def test_predict_valid():
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"good", "bad"}
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_level"] in {"Low", "Medium", "High"}


def test_predict_invalid_input():
    """Bad category / out-of-range values should fail validation (422)."""
    bad = dict(VALID_CUSTOMER, housing="mansion", age=5)
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
