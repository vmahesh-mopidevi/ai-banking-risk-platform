"""
train_model.py
--------------
Train the credit-risk classifier, track everything with MLflow, and save the
final model to models/credit_risk_model.pkl.

Run it with:
    python -m src.train_model

What gets logged to MLflow:
    - Parameters (model type + hyper-parameters)
    - Metrics (accuracy, precision, recall, f1, roc_auc)
    - The trained model as an artifact
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

# MLflow 3.x blocks the legacy ./mlruns file backend unless explicitly allowed.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_preprocessing import preprocess_data
from src.model_registry import REGISTERED_MODEL_NAME, register_and_promote
from src.utils import MLRUNS_DIR, MODEL_PATH, ensure_dirs

# XGBoost is preferred, but it is an optional dependency. If it cannot be
# imported OR its native library fails to load (e.g. missing libomp on macOS),
# we transparently fall back to scikit-learn's RandomForest.
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on the environment
    print(f"[train] XGBoost unavailable ({type(exc).__name__}); using RandomForest. "
          f"Tip: `brew install libomp` to enable XGBoost on macOS.")
    XGBOOST_AVAILABLE = False


EXPERIMENT_NAME = "credit_risk_classification"


def build_model():
    """Return (model, model_type, params): XGBoost if available, else RandomForest."""
    if XGBOOST_AVAILABLE:
        params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "eval_metric": "logloss",
        }
        model = XGBClassifier(**params)
        return model, "XGBoostClassifier", params

    params = {
        "n_estimators": 300,
        "max_depth": 8,
        "random_state": 42,
        "class_weight": "balanced",
    }
    model = RandomForestClassifier(**params)
    return model, "RandomForestClassifier", params


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Standard binary-classification metrics (positive class = 'bad')."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def train() -> dict:
    ensure_dirs()

    # Point MLflow at a local folder so runs are tracked without any server.
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test, feature_names = preprocess_data()
    model, model_type, params = build_model()
    print(f"[train] Using model: {model_type}")

    with mlflow.start_run(run_name=f"{model_type}_run"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # P(risk == 'bad')
        metrics = compute_metrics(y_test, y_pred, y_proba)

        # --- Log to MLflow -------------------------------------------------
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params)
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_metrics(metrics)
        if model_type == "XGBoostClassifier":
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME,
            )
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME,
            )

        trained_at = datetime.now(timezone.utc).isoformat()
        bundle = {
            "model": model,
            "model_type": model_type,
            "feature_names": feature_names,
            "metrics": metrics,
            "trained_at": trained_at,
            "version": "0.1.0",
        }
        joblib.dump(bundle, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="api_bundle")

        register_and_promote(
            model_type=model_type,
            metrics=metrics,
            trained_at=trained_at,
            version_label=bundle["version"],
        )

        print("[train] Metrics:")
        for name, value in metrics.items():
            print(f"         {name:10s}: {value:.4f}")

    print(f"[train] Saved model bundle to {MODEL_PATH}")

    return metrics


if __name__ == "__main__":
    train()
