"""
data_preprocessing.py
---------------------
Turns the raw credit CSV into clean, numeric, model-ready arrays.

Pipeline:
    1. Load the CSV (auto-generate a synthetic one if it is missing).
    2. Handle missing values.
    3. Encode categorical columns into integers.
    4. Encode the target ('good' -> 0, 'bad' -> 1).
    5. Split into train/test sets.

We use a simple, deterministic ordinal encoding based on the fixed category
lists in utils.py. The big advantage: the FastAPI service can encode a single
incoming request the exact same way, with no extra encoder file to load.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import (
    CATEGORICAL_FEATURES,
    CATEGORY_OPTIONS,
    DATA_PATH,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    ensure_dirs,
    generate_synthetic_data,
)

# 'bad' is the positive class (1): the thing we care about detecting.
TARGET_MAP = {"good": 0, "bad": 1}
TARGET_MAP_INVERSE = {0: "good", 1: "bad"}


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """
    Load the credit dataset from CSV.

    If the file does not exist yet, generate a synthetic dataset and save it,
    so the project runs end-to-end out of the box.
    """
    ensure_dirs()
    path = pd.io.common.stringify_path(path)
    try:
        df = pd.read_csv(path)
        print(f"[data] Loaded {len(df)} rows from {path}")
    except FileNotFoundError:
        print(f"[data] {path} not found - generating a synthetic dataset instead.")
        df = generate_synthetic_data()
        df.to_csv(path, index=False)
        print(f"[data] Saved synthetic dataset to {path}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps: numeric columns with the median, categoricals with the mode."""
    df = df.copy()
    for col in NUMERIC_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical text columns into integer codes.

    Each category maps to its index in CATEGORY_OPTIONS (a stable, known order).
    Unknown values fall back to 0 so the pipeline never crashes on bad input.
    """
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        mapping = {value: idx for idx, value in enumerate(CATEGORY_OPTIONS[col])}
        df[col] = df[col].map(mapping).fillna(0).astype(int)
    return df


def encode_record(record: dict) -> pd.DataFrame:
    """
    Encode a SINGLE customer record (a dict) into a one-row DataFrame with the
    same columns and order the model was trained on. Used by the FastAPI app.
    """
    df = pd.DataFrame([record])
    df = handle_missing_values(df)
    df = encode_features(df)
    return df[FEATURE_COLUMNS]


def preprocess_data(test_size: float = 0.2, random_state: int = 42):
    """
    Run the full preprocessing pipeline and return a train/test split.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    df = load_data()
    df = handle_missing_values(df)

    y = df[TARGET_COLUMN].map(TARGET_MAP).astype(int)
    X = encode_features(df)[FEATURE_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # keep the good/bad ratio the same in both splits
    )
    print(f"[data] Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    return X_train, X_test, y_train, y_test, FEATURE_COLUMNS


if __name__ == "__main__":
    # Quick manual check: python -m src.data_preprocessing
    X_train, X_test, y_train, y_test, features = preprocess_data()
    print("Features:", features)
    print("Sample encoded row:\n", X_train.head(1).to_dict(orient="records")[0])
