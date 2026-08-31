"""Coverage of real images by synthetic neighbors in embedding space."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .core import normalize_rows, validate_matching_dimensions


DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25)


def nearest_synthetic_distances(
    real_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Return each real image's minimum cosine distance to a synthetic image."""
    real = normalize_rows(real_embeddings, "real_embeddings")
    synthetic = normalize_rows(synthetic_embeddings, "synthetic_embeddings")
    validate_matching_dimensions(real, synthetic)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    minimum_distances: list[np.ndarray] = []
    for start in range(0, real.shape[0], chunk_size):
        similarities = real[start : start + chunk_size] @ synthetic.T
        distances = 1.0 - np.clip(similarities, -1.0, 1.0)
        minimum_distances.append(distances.min(axis=1))
    return np.concatenate(minimum_distances)


def normalized_coverage_auc(thresholds: Sequence[float], fractions: Sequence[float]) -> float:
    """Area under a coverage curve, normalized to the threshold interval."""
    x = np.asarray(thresholds, dtype=float)
    y = np.asarray(fractions, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("thresholds and fractions must be equal-length one-dimensional arrays")
    order = np.argsort(x)
    x, y = x[order], y[order]
    span = x[-1] - x[0]
    if span <= 0:
        raise ValueError("thresholds must span a positive interval")
    return float(np.trapz(y, x) / span)


def coverage_curve(
    real_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    *,
    generator: str = "synthetic",
    severity: str = "unspecified",
    strict: bool = True,
    ddof: int = 1,
) -> pd.DataFrame:
    """Compute Algorithm S1 coverage summaries for one synthetic set.

    The published algorithm uses ``distance < threshold``. Set ``strict=False``
    to use ``<=`` for sensitivity analysis.
    """
    thresholds_array = np.asarray(thresholds, dtype=float)
    if thresholds_array.ndim != 1 or thresholds_array.size == 0:
        raise ValueError("At least one threshold is required")
    if not np.isfinite(thresholds_array).all() or np.any(thresholds_array < 0):
        raise ValueError("Thresholds must be finite and nonnegative")

    distances = nearest_synthetic_distances(real_embeddings, synthetic_embeddings)
    n_real = len(distances)
    n_synthetic = np.asarray(synthetic_embeddings).shape[0]
    mean_distance = float(distances.mean())
    std_distance = float(distances.std(ddof=ddof)) if n_real > ddof else 0.0
    rows = []
    for threshold in thresholds_array:
        covered = distances < threshold if strict else distances <= threshold
        count = int(covered.sum())
        rows.append(
            {
                "generator": generator,
                "severity": severity,
                "num_real": n_real,
                "num_synthetic": n_synthetic,
                "threshold": float(threshold),
                "num_real_covered": count,
                "coverage_fraction": count / n_real,
                "mean_min_distance": mean_distance,
                "std_min_distance": std_distance,
            }
        )
    frame = pd.DataFrame(rows).sort_values("threshold", ignore_index=True)
    if len(frame) > 1:
        frame["coverage_auc"] = normalized_coverage_auc(
            frame["threshold"], frame["coverage_fraction"]
        )
    else:
        frame["coverage_auc"] = np.nan
    return frame
