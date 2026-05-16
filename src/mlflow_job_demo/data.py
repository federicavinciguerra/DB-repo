from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["size_sqm", "num_rooms", "age_years", "distance_km"]
TARGET_COLUMN = "price"


def generate_dataset(
    n: int = 5000,
    noise_std: float = 20000.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic housing-price regression dataset."""
    if n <= 0:
        raise ValueError("n must be greater than 0")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    rng = np.random.default_rng(random_state)

    size_sqm = rng.uniform(40, 250, n)
    num_rooms = rng.integers(1, 8, n)
    age_years = rng.uniform(0, 50, n)
    distance_km = rng.uniform(0, 30, n)
    noise = rng.normal(0, noise_std, n)

    price = (
        3000 * size_sqm
        + 20000 * num_rooms
        - 1000 * age_years
        - 5000 * distance_km
        + noise
    )

    return pd.DataFrame(
        {
            "size_sqm": size_sqm,
            "num_rooms": num_rooms,
            "age_years": age_years,
            "distance_km": distance_km,
            "price": price,
        }
    )


def create_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create train, validation, and test splits.

    val_size and test_size are interpreted as fractions of the full dataset.
    For example, test_size=0.15 and val_size=0.15 produces approximately:
    train=70%, validation=15%, test=15%.
    """
    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    relative_val_size = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=relative_val_size,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
