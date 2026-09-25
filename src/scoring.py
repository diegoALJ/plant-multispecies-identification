"""Score aggregation helpers."""

from __future__ import annotations

import numpy as np


def max_mean_blend(max_scores: np.ndarray, mean_scores: np.ndarray, max_weight: float = 0.7) -> np.ndarray:
    """Blend max and mean tile scores into image-level scores."""
    mean_weight = 1.0 - max_weight
    return (max_weight * max_scores + mean_weight * mean_scores).astype(np.float32)


def summarize_scores(scores: np.ndarray) -> dict:
    """Return a compact score summary."""
    return {
        "shape": scores.shape,
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
    }
