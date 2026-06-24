"""
main.py
-------
FastAPI service that serves credit-risk predictions.

Endpoints:
    GET  /health      -> simple liveness + model-loaded status
    GET  /model-info  -> metadata about the deployed model & project
    POST /predict     -> credit-risk prediction for one applicant

Run locally with:
    uvicorn api.main:app --reload --port 8000
Then open http://localhost:8000/docs for interactive docs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

# Make sure the project root is importable when launched from anywhere
# (e.g. inside Docker, or via `uvicorn api.main:app`).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import (  # noqa: E402
    CustomerData,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.data_preprocessing import TARGET_MAP_INVERSE, encode_record  # noqa: E402
from src.utils import MODEL_PATH  # noqa: E402

PROJECT_DESCRIPTION = (
    "AI Banking Risk Intelligence Platform - predicts the credit risk of loan "
    "applicants to support faster, more consistent, and auditable lending decisions."
)

app = FastAPI(
    title="AI Banking Risk Intelligence Platform",
    description=PROJECT_DESCRIPTION,
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Load the model once at import time. If it is missing, the app still starts
# (health check works) but /predict will return a clear 503 error.
# ---------------------------------------------------------------------------
MODEL_BUNDLE: dict | None = None


def load_model() -> None:
    global MODEL_BUNDLE
    model_path = Path(os.getenv("MODEL_PATH", MODEL_PATH))
    if model_path.exists():
        MODEL_BUNDLE = joblib.load(model_path)
        print(f"[api] Loaded model from {model_path}")
    else:
        MODEL_BUNDLE = None
        print(f"[api] WARNING: no model found at {model_path}. Train it first.")


load_model()


def score_to_level(score: float) -> str:
    """Bucket a probability into a human-friendly risk band."""
    if score < 0.33:
        return "Low"
    if score < 0.66:
        return "Medium"
    return "High"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe: is the service up and is a model loaded?"""
    return HealthResponse(status="ok", model_loaded=MODEL_BUNDLE is not None)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return metadata about the deployed model and project."""
    if MODEL_BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train it first.")
    return ModelInfoResponse(
        model_name="credit_risk_model",
        model_type=MODEL_BUNDLE.get("model_type", "unknown"),
        version=MODEL_BUNDLE.get("version", "0.1.0"),
        purpose="Predict whether a loan applicant is a good or bad credit risk.",
        project=PROJECT_DESCRIPTION,
        trained_at=MODEL_BUNDLE.get("trained_at"),
        metrics=MODEL_BUNDLE.get("metrics"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData) -> PredictionResponse:
    """Predict credit risk for a single applicant."""
    if MODEL_BUNDLE is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m src.train_model` first.",
        )

    model = MODEL_BUNDLE["model"]

    # Encode the incoming record the exact same way as the training data.
    features = encode_record(customer.model_dump())

    risk_score = float(model.predict_proba(features)[0, 1])  # P(bad)
    pred_label = int(risk_score >= 0.5)

    return PredictionResponse(
        prediction=TARGET_MAP_INVERSE[pred_label],  # 'good' or 'bad'
        risk_score=round(risk_score, 4),
        risk_level=score_to_level(risk_score),
    )
