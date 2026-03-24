"""
Unit tests for mlwatch.utils.stats
"""

import numpy as np
import pytest

from mlwatch.utils.stats import (
    compute_psi,
    compute_ks_test,
    compute_chi_square,
    compute_z_scores,
    zscore_outlier_flags,
    per_feature_drift_report,
    psi_label,
)


class TestPSI:
    def test_identical_distributions_zero_psi(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=500)
        assert compute_psi(data, data) == pytest.approx(0.0, abs=0.05)

    def test_different_distributions_high_psi(self):
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, 500)
        prod = rng.normal(5, 1, 500)   # large shift
        psi = compute_psi(base, prod)
        assert psi > 0.2, f"Expected PSI > 0.2 for shifted distributions, got {psi}"

    def test_psi_label(self):
        assert psi_label(0.05) == "stable"
        assert psi_label(0.15) == "moderate"
        assert psi_label(0.25) == "significant"

    def test_degenerate_all_same(self):
        data = np.ones(100)
        psi = compute_psi(data, data)
        assert psi == 0.0

    def test_moderate_drift(self):
        rng = np.random.default_rng(42)
        base = rng.normal(0, 1, 1000)
        prod = rng.normal(0.5, 1, 1000)
        psi = compute_psi(base, prod)
        assert 0.0 < psi < 0.5


class TestKSTest:
    def test_identical_distributions_high_pvalue(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=500)
        _, pvalue = compute_ks_test(data, data)
        assert pvalue == pytest.approx(1.0)

    def test_different_distributions_low_pvalue(self):
        rng = np.random.default_rng(1)
        base = rng.normal(0, 1, 500)
        prod = rng.normal(5, 1, 500)
        stat, pvalue = compute_ks_test(base, prod)
        assert stat > 0.5
        assert pvalue < 0.05

    def test_returns_tuple(self):
        rng = np.random.default_rng(2)
        result = compute_ks_test(rng.normal(size=100), rng.normal(size=100))
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestChiSquare:
    def test_identical_categorical(self):
        cats = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2] * 50)
        stat, pvalue = compute_chi_square(cats, cats.copy())
        assert pvalue > 0.05

    def test_shifted_categorical(self):
        base = np.array([0] * 100 + [1] * 100 + [2] * 100)
        prod = np.array([0] * 10 + [1] * 10 + [2] * 280)  # heavy shift
        stat, pvalue = compute_chi_square(base, prod)
        assert pvalue < 0.05


class TestZScore:
    def test_z_scores_shape(self):
        data = np.random.randn(50, 5)
        z = compute_z_scores(data)
        assert z.shape == (50, 5)

    def test_outlier_flags_basic(self):
        rng = np.random.default_rng(10)
        baseline = rng.normal(size=(200, 3))
        inliers = rng.normal(size=(10, 3))
        extreme_outlier = np.array([[100, 100, 100]])

        mean = baseline.mean(axis=0)
        std = baseline.std(axis=0)

        inlier_flags = zscore_outlier_flags(inliers, mean, std, sigma_threshold=4.0)
        outlier_flags = zscore_outlier_flags(extreme_outlier, mean, std, sigma_threshold=4.0)

        assert not inlier_flags.all(), "Inliers should NOT be flagged"
        assert outlier_flags.all(), "Extreme outlier SHOULD be flagged"


class TestPerFeatureDriftReport:
    def test_report_structure(self):
        rng = np.random.default_rng(99)
        baseline = rng.normal(size=(200, 3))
        production = rng.normal(size=(100, 3))
        report = per_feature_drift_report(baseline, production, feature_names=["a", "b", "c"])

        assert set(report.keys()) == {"a", "b", "c"}
        for feat, info in report.items():
            assert "psi" in info
            assert "psi_severity" in info
            assert "is_drifted" in info

    def test_report_drifted_feature(self):
        rng = np.random.default_rng(77)
        baseline = np.column_stack([rng.normal(0, 1, 500), rng.normal(0, 1, 500)])
        production = np.column_stack([rng.normal(10, 1, 200), rng.normal(0, 1, 200)])
        report = per_feature_drift_report(baseline, production, feature_names=["drifted", "stable"])

        assert report["drifted"]["is_drifted"] is True
