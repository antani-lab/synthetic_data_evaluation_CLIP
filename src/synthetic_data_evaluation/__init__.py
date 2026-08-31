"""Task-grounded evaluation of synthetic medical images."""

from .constraint_consistency import constraint_consistency
from .coverage import DEFAULT_THRESHOLDS, coverage_curve, normalized_coverage_auc
from .downstream import classification_metrics, holm_adjust, paired_stratified_bootstrap
from .quality_utility import generator_cluster_bootstrap, partial_spearman_by_group

__all__ = [
    "DEFAULT_THRESHOLDS",
    "constraint_consistency",
    "coverage_curve",
    "classification_metrics",
    "generator_cluster_bootstrap",
    "holm_adjust",
    "normalized_coverage_auc",
    "paired_stratified_bootstrap",
    "partial_spearman_by_group",
]

__version__ = "0.1.0"
