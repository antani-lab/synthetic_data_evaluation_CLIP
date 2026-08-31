"""Downstream classification metrics and paired bootstrap comparisons."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm-adjusted P values in the input order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be a one-dimensional array in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.maximum.accumulate((len(values) - np.arange(len(values))) * ranked)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    classes: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the manuscript's accuracy, recall, F1, and macro OVR ROC-AUC."""
    y_true = np.asarray(labels)
    y_probability = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or y_probability.ndim != 2 or len(y_true) != len(y_probability):
        raise ValueError("labels must be 1-D and probabilities must be a matching 2-D array")
    if not np.isfinite(y_probability).all() or np.any(y_probability < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    row_sums = y_probability.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("Each probability row must sum to one")
    class_values = np.asarray(classes if classes is not None else np.unique(y_true))
    if y_probability.shape[1] != len(class_values):
        raise ValueError("The number of probability columns must equal the number of classes")

    predictions = class_values[np.argmax(y_probability, axis=1)]
    recalls = recall_score(y_true, predictions, labels=class_values, average=None, zero_division=0)
    f1_values = f1_score(y_true, predictions, labels=class_values, average=None, zero_division=0)
    result: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=class_values, average="macro", zero_division=0)),
        "macro_roc_auc_ovr": float(
            roc_auc_score(y_true, y_probability, labels=class_values, average="macro", multi_class="ovr")
        ),
    }
    for index, class_value in enumerate(class_values):
        result[f"recall_{class_value}"] = float(recalls[index])
        result[f"f1_{class_value}"] = float(f1_values[index])
    return result


def _stratified_resample_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for class_value in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_value)
        parts.append(rng.choice(class_indices, size=len(class_indices), replace=True))
    return np.concatenate(parts)


def paired_stratified_bootstrap(
    labels: np.ndarray,
    baseline_probabilities: np.ndarray,
    condition_probabilities: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_resamples: int = 5000,
    confidence_level: float = 0.95,
    seed: int = 2026,
) -> dict[str, float]:
    """Estimate a paired condition-minus-baseline change on the same images."""
    y_true = np.asarray(labels)
    baseline = np.asarray(baseline_probabilities)
    condition = np.asarray(condition_probabilities)
    if len(y_true) != len(baseline) or len(y_true) != len(condition):
        raise ValueError("Baseline and condition predictions must refer to the same images")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    estimate_baseline = float(metric(y_true, baseline))
    estimate_condition = float(metric(y_true, condition))
    rng = np.random.default_rng(seed)
    changes = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample = _stratified_resample_indices(y_true, rng)
        changes[index] = metric(y_true[sample], condition[sample]) - metric(
            y_true[sample], baseline[sample]
        )
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(changes, [alpha / 2.0, 1.0 - alpha / 2.0])
    lower_tail = (np.count_nonzero(changes <= 0) + 1) / (n_resamples + 1)
    upper_tail = (np.count_nonzero(changes >= 0) + 1) / (n_resamples + 1)
    return {
        "baseline": estimate_baseline,
        "condition": estimate_condition,
        "change": estimate_condition - estimate_baseline,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
        "n_images": int(len(y_true)),
        "n_resamples": int(n_resamples),
    }
