"""
evaluate_model.py
-----------------
Load the trained model bundle and evaluate it on a fresh test split.

Prints a full classification report and saves a JSON + text summary into the
reports/ folder.

Run it with:
    python -m src.evaluate_model
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.data_preprocessing import TARGET_MAP_INVERSE, preprocess_data
from src.utils import MODEL_PATH, REPORTS_DIR, ensure_dirs


def load_model_bundle(path=MODEL_PATH) -> dict:
    """Load the joblib bundle saved by train_model.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train it first with: python -m src.train_model"
        )
    return joblib.load(path)


def evaluate() -> dict:
    ensure_dirs()
    bundle = load_model_bundle()
    model = bundle["model"]

    # Re-create the same split (same random_state) used during training.
    _, X_test, _, y_test, _ = preprocess_data()

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    target_names = [TARGET_MAP_INVERSE[0], TARGET_MAP_INVERSE[1]]  # ['good', 'bad']

    report_text = classification_report(y_test, y_pred, target_names=target_names)
    report_dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred).tolist()
    roc_auc = roc_auc_score(y_test, y_proba)

    print("=" * 60)
    print(f"Model: {bundle.get('model_type', 'unknown')}")
    print("=" * 60)
    print(report_text)
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion matrix [rows=true, cols=pred] (order: good, bad):")
    print(cm)

    summary = {
        "model_type": bundle.get("model_type"),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "roc_auc": roc_auc,
        "classification_report": report_dict,
        "confusion_matrix": cm,
    }

    # Save both a machine-readable JSON and a human-readable text file.
    json_path = REPORTS_DIR / "evaluation_summary.json"
    txt_path = REPORTS_DIR / "evaluation_report.txt"
    json_path.write_text(json.dumps(summary, indent=2))
    txt_path.write_text(
        f"Model: {bundle.get('model_type')}\n\n{report_text}\n\nROC AUC: {roc_auc:.4f}\n"
    )
    print(f"\n[eval] Saved summary to {json_path}")
    print(f"[eval] Saved report to {txt_path}")

    return summary


if __name__ == "__main__":
    evaluate()
