"""
utils.py
--------
Shared helpers and configuration used across the project.

Keeping paths, column names, and the synthetic-data generator in one place
means the preprocessing, training, evaluation, and API code all agree on the
same "contract" (same columns, same file locations).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# This file lives in <project>/src/utils.py, so the project root is two
# levels up. Building paths from here keeps the code working no matter where
# the scripts are launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

DATA_PATH = DATA_DIR / "german_credit_data.csv"
MODEL_PATH = MODELS_DIR / "credit_risk_model.pkl"

# ---------------------------------------------------------------------------
# Dataset "schema": which columns are numeric, categorical, and the target.
# The API and the preprocessing both rely on these lists.
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["age", "job", "credit_amount", "duration"]
CATEGORICAL_FEATURES = ["housing", "saving_accounts", "checking_account", "purpose"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "risk"

# Allowed values for each categorical column (used to build the dataset and
# to validate / encode API requests).
CATEGORY_OPTIONS = {
    "housing": ["own", "rent", "free"],
    "saving_accounts": ["little", "moderate", "quite rich", "rich"],
    "checking_account": ["little", "moderate", "rich"],
    "purpose": [
        "car",
        "furniture/equipment",
        "radio/TV",
        "domestic appliances",
        "repairs",
        "education",
        "business",
        "vacation/others",
    ],
}


def ensure_dirs() -> None:
    """Create the output folders if they do not exist yet."""
    for directory in (DATA_DIR, MODELS_DIR, REPORTS_DIR, MLRUNS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def generate_synthetic_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Build a realistic-looking synthetic German-credit-style dataset.

    We do NOT assign the risk label randomly. Instead we compute a hidden
    "risk score" from the features (older, wealthier, smaller/shorter loans =
    safer) and label the customer 'bad' when that score is high. This gives the
    model a genuine signal to learn, so the demo metrics are meaningful.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(19, 75, size=n_samples)
    # job: 0=unskilled non-resident ... 3=highly skilled
    job = rng.integers(0, 4, size=n_samples)
    credit_amount = rng.integers(250, 20000, size=n_samples)
    duration = rng.integers(4, 72, size=n_samples)  # loan length in months

    housing = rng.choice(CATEGORY_OPTIONS["housing"], size=n_samples, p=[0.55, 0.35, 0.10])
    saving_accounts = rng.choice(
        CATEGORY_OPTIONS["saving_accounts"], size=n_samples, p=[0.55, 0.25, 0.12, 0.08]
    )
    checking_account = rng.choice(
        CATEGORY_OPTIONS["checking_account"], size=n_samples, p=[0.5, 0.35, 0.15]
    )
    purpose = rng.choice(CATEGORY_OPTIONS["purpose"], size=n_samples)

    # ----- Build a latent risk score (higher => more likely 'bad') ----------
    score = np.zeros(n_samples, dtype=float)
    score += (credit_amount / 20000.0) * 1.6        # bigger loans are riskier
    score += (duration / 72.0) * 1.4                # longer loans are riskier
    score += (40 - np.clip(age, 19, 60)) / 40.0     # younger applicants riskier
    score += (3 - job) * 0.15                        # lower skill => riskier

    savings_risk = {"little": 0.6, "moderate": 0.3, "quite rich": 0.1, "rich": 0.0}
    checking_risk = {"little": 0.5, "moderate": 0.2, "rich": 0.0}
    housing_risk = {"own": 0.0, "rent": 0.3, "free": 0.4}

    score += np.array([savings_risk[s] for s in saving_accounts])
    score += np.array([checking_risk[c] for c in checking_account])
    score += np.array([housing_risk[h] for h in housing])

    # Add a little noise so the relationship is not perfectly separable.
    score += rng.normal(0, 0.35, size=n_samples)

    # Threshold at the ~70th percentile so roughly 30% of customers are 'bad',
    # which mirrors the real German Credit dataset class balance.
    threshold = np.quantile(score, 0.70)
    risk = np.where(score >= threshold, "bad", "good")

    df = pd.DataFrame(
        {
            "age": age,
            "job": job,
            "credit_amount": credit_amount,
            "duration": duration,
            "housing": housing,
            "saving_accounts": saving_accounts,
            "checking_account": checking_account,
            "purpose": purpose,
            "risk": risk,
        }
    )
    return df
