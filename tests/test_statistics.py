import numpy as np

from synthetic_data_evaluation.downstream import (
    classification_metrics,
    holm_adjust,
    paired_stratified_bootstrap,
)
from synthetic_data_evaluation.quality_utility import partial_spearman_by_group


def test_holm_adjustment_preserves_input_order():
    adjusted = holm_adjust(np.array([0.04, 0.01, 0.03]))
    np.testing.assert_allclose(adjusted, [0.06, 0.03, 0.06])


def test_classification_metrics_and_paired_bootstrap():
    labels = np.array([0, 0, 1, 1, 2, 2])
    baseline = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.4, 0.5, 0.1],
            [0.2, 0.7, 0.1],
            [0.4, 0.3, 0.3],
            [0.2, 0.2, 0.6],
            [0.4, 0.3, 0.3],
        ]
    )
    condition = np.eye(3)[labels] * 0.8 + 0.2 / 3.0
    condition /= condition.sum(axis=1, keepdims=True)
    metrics = classification_metrics(labels, condition)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0

    def accuracy(y_true, probabilities):
        return classification_metrics(y_true, probabilities)["accuracy"]

    result = paired_stratified_bootstrap(
        labels,
        baseline,
        condition,
        accuracy,
        n_resamples=100,
        seed=7,
    )
    assert result["change"] > 0
    assert result["n_images"] == 6


def test_severity_adjusted_partial_rank_removes_between_group_pattern():
    quality = np.array([1, 2, 3, 11, 12, 13], dtype=float)
    utility = np.array([4, 5, 6, 14, 15, 16], dtype=float)
    severity = np.array(["mild"] * 3 + ["severe"] * 3)
    statistic = partial_spearman_by_group(quality, utility, severity)
    assert statistic > 0.99
