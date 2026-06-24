"""
promote_model.py
----------------
Point a registry alias (e.g. Production) at a specific model version.

Examples:
    python -m src.promote_model --version 2 --alias Production
    python -m src.promote_model --alias Staging   # uses latest version
"""

from __future__ import annotations

import argparse

from mlflow import MlflowClient

from src.model_registry import REGISTERED_MODEL_NAME, configure_mlflow


def promote(version: str | None, alias: str) -> None:
    configure_mlflow()
    client = MlflowClient()

    if version is None:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        if not versions:
            raise SystemExit(f"No versions found for '{REGISTERED_MODEL_NAME}'. Train first.")
        version = str(max(int(v.version) for v in versions))

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, alias, version)
    print(f"[registry] Alias '{alias}' -> {REGISTERED_MODEL_NAME} v{version}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set an MLflow model alias on a version.")
    parser.add_argument(
        "--version",
        help="Registry version number (default: latest registered version)",
    )
    parser.add_argument(
        "--alias",
        default="Production",
        help="Alias name to set, e.g. Production or Staging (default: Production)",
    )
    args = parser.parse_args()
    promote(args.version, args.alias)


if __name__ == "__main__":
    main()
