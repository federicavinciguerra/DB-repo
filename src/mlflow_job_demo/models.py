from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor


def build_model(
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    random_state: int = 42,
) -> GradientBoostingRegressor:
    """Build the scikit-learn regression model."""
    if n_estimators <= 0:
        raise ValueError("n_estimators must be greater than 0")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if max_depth <= 0:
        raise ValueError("max_depth must be greater than 0")

    return GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
