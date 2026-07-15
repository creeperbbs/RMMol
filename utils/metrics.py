"""Evaluation metrics used by RMMol experiments."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """Return common regression metrics."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if not np.any(mask):
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan}

    y_true_arr = y_true_arr[mask]
    y_pred_arr = y_pred_arr[mask]
    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(mean_squared_error(y_true_arr, y_pred_arr, squared=False)),
        "r2": float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 1 else np.nan,
    }


def classification_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Return binary classification metrics with safe single-class handling."""
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score_arr) & ~pd_isna(y_true_arr)
    if not np.any(mask):
        return {"roc_auc": np.nan, "pr_auc": np.nan, "accuracy": np.nan}

    y_true_arr = y_true_arr[mask].astype(int)
    y_score_arr = y_score_arr[mask]
    y_pred = (y_score_arr >= threshold).astype(int)
    has_two_classes = len(np.unique(y_true_arr)) >= 2
    return {
        "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)) if has_two_classes else np.nan,
        "pr_auc": float(average_precision_score(y_true_arr, y_score_arr)) if has_two_classes else np.nan,
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
    }


def pd_isna(values: np.ndarray) -> np.ndarray:
    """Small pandas-free missing-value helper for numeric or object arrays."""
    return np.asarray([value is None or (isinstance(value, float) and np.isnan(value)) for value in values])


def cliff_delta_ratio(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    pair_indices: Iterable[tuple[int, int]],
    min_delta: float = 1.0,
) -> Optional[float]:
    """Compute the fraction of true activity cliffs recovered by predicted deltas."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    pairs = list(pair_indices)
    if not pairs:
        return None

    true_cliffs = 0
    recovered = 0
    for i, j in pairs:
        if abs(y_true_arr[i] - y_true_arr[j]) >= min_delta:
            true_cliffs += 1
            if abs(y_pred_arr[i] - y_pred_arr[j]) >= min_delta:
                recovered += 1
    if true_cliffs == 0:
        return None
    return recovered / true_cliffs
