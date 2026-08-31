"""Association analysis between image-quality measures and downstream utility."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr


def _rank_residuals(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    ranked = rankdata(values, method="average")
    design = np.column_stack(
        [np.ones(len(groups))]
        + [(groups == level).astype(float) for level in np.unique(groups)[1:]]
    )
    fitted = design @ np.linalg.lstsq(design, ranked, rcond=None)[0]
    return ranked - fitted


def partial_spearman_by_group(
    quality: Sequence[float],
    utility: Sequence[float],
    groups: Sequence[str],
) -> float:
    """Spearman correlation after removing group-level rank means."""
    x = np.asarray(quality, dtype=float)
    y = np.asarray(utility, dtype=float)
    group_values = np.asarray(groups)
    if x.shape != y.shape or x.shape != group_values.shape or x.ndim != 1:
        raise ValueError("quality, utility, and groups must be matching one-dimensional arrays")
    if len(np.unique(group_values)) < 2:
        raise ValueError("At least two groups are required for adjustment")
    return float(spearmanr(_rank_residuals(x, group_values), _rank_residuals(y, group_values)).statistic)


def spearman_permutation_test(
    quality: Sequence[float],
    utility: Sequence[float],
    *,
    groups: Sequence[str] | None = None,
    n_permutations: int = 10000,
    seed: int = 2026,
) -> dict[str, float]:
    """Test pooled or group-adjusted Spearman association by permutation."""
    x = np.asarray(quality, dtype=float)
    y = np.asarray(utility, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 3:
        raise ValueError("quality and utility must be matching one-dimensional arrays of length >= 3")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")
    group_values = None if groups is None else np.asarray(groups)
    if group_values is not None and group_values.shape != x.shape:
        raise ValueError("groups must have the same shape as quality and utility")
    statistic = (
        float(spearmanr(x, y).statistic)
        if group_values is None
        else partial_spearman_by_group(x, y, group_values)
    )
    rng = np.random.default_rng(seed)
    null_statistics = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        if group_values is None:
            permuted = rng.permutation(y)
        else:
            permuted = y.copy()
            for group in np.unique(group_values):
                mask = group_values == group
                permuted[mask] = rng.permutation(y[mask])
        null_statistics[index] = (
            float(spearmanr(x, permuted).statistic)
            if group_values is None
            else partial_spearman_by_group(x, permuted, group_values)
        )
    p_value = (np.count_nonzero(np.abs(null_statistics) >= abs(statistic)) + 1) / (
        n_permutations + 1
    )
    return {"rho": statistic, "permutation_p_value": float(p_value)}


def generator_cluster_bootstrap(
    quality: Sequence[float],
    utility: Sequence[float],
    generators: Sequence[str],
    *,
    groups: Sequence[str] | None = None,
    n_resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 2026,
) -> tuple[float, float]:
    """Bootstrap an association by resampling generator clusters."""
    x = np.asarray(quality, dtype=float)
    y = np.asarray(utility, dtype=float)
    cluster_values = np.asarray(generators)
    group_values = None if groups is None else np.asarray(groups)
    if x.shape != y.shape or x.shape != cluster_values.shape:
        raise ValueError("quality, utility, and generators must have matching shapes")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if group_values is not None and group_values.shape != x.shape:
        raise ValueError("groups must have the same shape as quality and utility")
    unique_clusters = np.unique(cluster_values)
    if len(unique_clusters) < 2:
        raise ValueError("At least two generator clusters are required")

    rng = np.random.default_rng(seed)
    statistics: list[float] = []
    for _ in range(n_resamples):
        selected = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        indices = np.concatenate([np.flatnonzero(cluster_values == cluster) for cluster in selected])
        if np.unique(x[indices]).size < 2 or np.unique(y[indices]).size < 2:
            continue
        statistic = (
            spearmanr(x[indices], y[indices]).statistic
            if group_values is None
            else partial_spearman_by_group(x[indices], y[indices], group_values[indices])
        )
        if np.isfinite(statistic):
            statistics.append(float(statistic))
    if not statistics:
        raise RuntimeError("No valid cluster-bootstrap samples were produced")
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(statistics, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)
