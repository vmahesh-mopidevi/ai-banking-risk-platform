"""
model_registry.py
-----------------
Helpers for MLflow Model Registry: register trained models, set aliases
(e.g. Production), and load the live model bundle for the API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import joblib
import mlflow
from mlflow import MlflowClient

from src.utils import MLRUNS_DIR, MODEL_PATH

REGISTERED_MODEL_NAME = "credit_risk_model"
DEFAULT_MODEL_ALIAS = "Production"


def configure_mlflow() -> None:
    """Point MLflow at the local tracking store (override via MLFLOW_TRACKING_URI)."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", MLRUNS_DIR.as_uri())
    mlflow.set_tracking_uri(tracking_uri)


def register_and_promote(
    *,
    model_type: str,
    metrics: dict,
    trained_at: str,
    version_label: str = "0.1.0",
    alias: str = DEFAULT_MODEL_ALIAS,
) -> str:
    """
    Tag the latest registered version and point *alias* at it
    (typically Production). Returns the registry version number as a string.
    """
    configure_mlflow()
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            f"No registered versions found for '{REGISTERED_MODEL_NAME}'. "
            "Did log_model(..., registered_model_name=...) run?"
        )

    latest = max(versions, key=lambda mv: int(mv.version))
    version = latest.version

    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "model_type", model_type)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "trained_at", trained_at)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "version_label", version_label)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "metrics", json.dumps(metrics))

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, alias, version)
    print(f"[registry] Alias '{alias}' -> {REGISTERED_MODEL_NAME} v{version}")
    return version


def load_production_bundle(
    *,
    model_name: str = REGISTERED_MODEL_NAME,
    alias: str = DEFAULT_MODEL_ALIAS,
) -> dict | None:
    """Download and load the API bundle artifact for the aliased registry version."""
    configure_mlflow()
    client = MlflowClient()

    try:
        model_version = client.get_model_version_by_alias(model_name, alias)
    except Exception:
        print(f"[registry] No alias '{alias}' on model '{model_name}'")
        return None

    artifact_rel_path = f"api_bundle/{MODEL_PATH.name}"
    bundle_path = client.download_artifacts(model_version.run_id, artifact_rel_path)
    bundle = joblib.load(bundle_path)
    bundle["registry_version"] = str(model_version.version)
    bundle["registry_alias"] = alias
    bundle["model_source"] = "registry"
    print(
        f"[registry] Loaded {model_name} v{model_version.version} "
        f"(alias={alias}) from run {model_version.run_id}"
    )
    return bundle


def load_file_bundle(model_path: Path | None = None) -> dict | None:
    """Load the local joblib bundle written by train_model.py."""
    path = model_path or MODEL_PATH
    if not path.exists():
        return None
    bundle = joblib.load(path)
    bundle["model_source"] = "file"
    print(f"[api] Loaded model bundle from {path}")
    return bundle


def load_model_bundle(
    *,
    source: Literal["auto", "registry", "file"] = "auto",
    model_path: Path | None = None,
    model_name: str = REGISTERED_MODEL_NAME,
    alias: str = DEFAULT_MODEL_ALIAS,
) -> dict | None:
    """
    Load the model bundle for serving.

    source:
        auto     -> try registry first, then local file
        registry -> registry only
        file     -> local file only
    """
    if source in {"auto", "registry"}:
        bundle = load_production_bundle(model_name=model_name, alias=alias)
        if bundle is not None:
            return bundle
        if source == "registry":
            return None

    return load_file_bundle(model_path)
