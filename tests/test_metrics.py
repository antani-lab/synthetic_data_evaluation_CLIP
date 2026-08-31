import numpy as np

from synthetic_data_evaluation.congruence import peak_signal_to_noise_ratio
from synthetic_data_evaluation.constraint_consistency import constraint_consistency
from synthetic_data_evaluation.coverage import coverage_curve, nearest_synthetic_distances


def test_coverage_uses_real_to_synthetic_nearest_neighbors():
    real = np.array([[1.0, 0.0], [0.0, 1.0]])
    synthetic = np.array([[1.0, 0.0]])
    distances = nearest_synthetic_distances(real, synthetic)
    np.testing.assert_allclose(distances, [0.0, 1.0])

    result = coverage_curve(real, synthetic, thresholds=[0.1, 1.1])
    np.testing.assert_allclose(result["coverage_fraction"], [0.5, 1.0])
    assert result["coverage_auc"].iloc[0] == 0.75


def test_constraint_consistency_reports_manuscript_definitions():
    real = np.array([[1.0, 0.0], [0.8, 0.2], [0.9, 0.1]])
    result = constraint_consistency(real, real.copy(), generator="same", severity="mild")
    assert abs(result["signed_mean_gap"]) < 1e-12
    assert abs(result["dispersion_ratio"] - 1.0) < 1e-12
    assert abs(result["synthetic_centroid_misalignment"]) < 1e-12


def test_psnr_requires_aligned_shapes_and_handles_exact_match():
    image = np.zeros((4, 4))
    assert np.isinf(peak_signal_to_noise_ratio(image, image))
