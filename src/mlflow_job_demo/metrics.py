from __future__ import annotations

from typing import Dict

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute final regression metrics."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
