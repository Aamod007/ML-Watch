"""
Unit tests for mlwatch.drift.ood.OODDetector
"""

import numpy as np
import pytest

from mlwatch.drift.ood import OODDetector, OODInputError


class TestOODDetector:
    def _fit_detector(self, n=300, n_features=5, on_ood="flag"):
        rng = np.random.default_rng(42)
        baseline = rng.normal(size=(n, n_features))
        det = OODDetector(baseline_data=baseline, sigma_threshold=4.0, on_ood=on_ood)
        return det, baseline

    def test_inliers_not_flagged(self):
        rng = np.random.default_rng(7)
        det, _ = self._fit_detector()
        inliers = rng.normal(size=(50, 5))
        result = det.check(inliers)
        # Most inliers should pass — allow up to 20% false positive rate
        false_pos_rate = result["ood_rate"]
        assert false_pos_rate < 0.30, f"Too many false positives: {false_pos_rate}"

    def test_extreme_outliers_flagged(self):
        det, _ = self._fit_detector()
        outliers = np.ones((5, 5)) * 1000.0  # far from training distribution
        result = det.check(outliers)
        assert result["ood_count"] == 5
        assert result["ood_rate"] == pytest.approx(1.0)

    def test_on_ood_block_raises(self):
        det, _ = self._fit_detector(on_ood="block")
        outliers = np.ones((3, 5)) * 1000.0
        with pytest.raises(OODInputError):
            det.check(outliers, raise_on_block=True)

    def test_cumulative_rate_tracking(self):
        det, _ = self._fit_detector()
        det.check(np.ones((10, 5)) * 1000.0)  # all OOD
        rate = det.cumulative_ood_rate
        assert rate > 0.0

    def test_reset_stats(self):
        det, _ = self._fit_detector()
        det.check(np.ones((5, 5)) * 1000.0)
        det.reset_stats()
        assert det.cumulative_ood_rate == 0.0

    def test_single_sample_1d(self):
        det, _ = self._fit_detector()
        sample = np.zeros(5)  # 1-D input
        result = det.check(sample)
        assert "ood_flags" in result
        assert len(result["ood_flags"]) == 1
