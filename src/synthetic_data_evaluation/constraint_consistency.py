"""Constraint and consistency relative to a real-class centroid."""

from __future__ import annotations

import numpy as np

from .core import normalize_rows, validate_matching_dimensions


def normalized_centroid(embeddings: np.ndarray) -> np.ndarray:
    normalized = normalize_rows(embeddings)
    centroid = normalized.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        raise ValueError("The class centroid has zero norm")
    return centroid / norm


def leave_one_out_real_similarities(real_embeddings: np.ndarray) -> np.ndarray:
    """Similarity of each real image to a centroid that excludes that image."""
    real = normalize_rows(real_embeddings, "real_embeddings")
    if len(real) < 2:
        raise ValueError("Leave-one-out similarities require at least two real images")
    total = real.sum(axis=0, keepdims=True)
    centroids = total - real
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("A leave-one-out centroid has zero norm")
    centroids /= norms
    return np.einsum("ij,ij->i", real, centroids)


def _summary(values: np.ndarray, ddof: int) -> dict[str, float | int]:
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=ddof)) if len(values) > ddof else 0.0,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "range": float(values.max() - values.min()),
    }


def constraint_consistency(
    real_embeddings: np.ndarray,
    synthetic_embeddings: np.ndarray,
    *,
    generator: str = "synthetic",
    severity: str = "unspecified",
    ddof: int = 1,
    leave_one_out_real: bool = False,
) -> dict[str, float | int | str]:
    """Compute the manuscript and Algorithm S2 summaries for one synthetic set.

    The final manuscript defines constraint as similarity to the normalized
    real-class centroid. Consistency is summarized by the signed mean gap
    ``synthetic_mean - real_mean`` and dispersion ratio
    ``synthetic_std / real_std``. For traceability to the supplied Algorithm S2
    PDF, the synthetic-centroid misalignment and synthetic dispersion are also
    returned.
    """
    real = normalize_rows(real_embeddings, "real_embeddings")
    synthetic = normalize_rows(synthetic_embeddings, "synthetic_embeddings")
    validate_matching_dimensions(real, synthetic)
    centroid = normalized_centroid(real)

    real_similarities = (
        leave_one_out_real_similarities(real) if leave_one_out_real else real @ centroid
    )
    synthetic_similarities = synthetic @ centroid
    real_summary = _summary(real_similarities, ddof)
    synthetic_summary = _summary(synthetic_similarities, ddof)
    real_std = float(real_summary["std"])
    synthetic_std = float(synthetic_summary["std"])

    synthetic_centroid = normalized_centroid(synthetic)
    signed_gap = float(synthetic_summary["mean"] - real_summary["mean"])
    return {
        "generator": generator,
        "severity": severity,
        "num_real": int(real_summary["n"]),
        "num_synthetic": int(synthetic_summary["n"]),
        "real_mean_cosine_similarity": float(real_summary["mean"]),
        "real_std_cosine_similarity": real_std,
        "real_minimum_cosine_similarity": float(real_summary["minimum"]),
        "real_maximum_cosine_similarity": float(real_summary["maximum"]),
        "real_range_cosine_similarity": float(real_summary["range"]),
        "synthetic_mean_cosine_similarity": float(synthetic_summary["mean"]),
        "synthetic_std_cosine_similarity": synthetic_std,
        "synthetic_minimum_cosine_similarity": float(synthetic_summary["minimum"]),
        "synthetic_maximum_cosine_similarity": float(synthetic_summary["maximum"]),
        "synthetic_range_cosine_similarity": float(synthetic_summary["range"]),
        "signed_mean_gap": signed_gap,
        "absolute_mean_gap": abs(signed_gap),
        "dispersion_ratio": synthetic_std / real_std if real_std > 0 else np.nan,
        "synthetic_centroid_misalignment": float(1.0 - centroid @ synthetic_centroid),
        "synthetic_dispersion": synthetic_std,
        "real_similarity_mode": "leave_one_out" if leave_one_out_real else "shared_centroid",
    }
