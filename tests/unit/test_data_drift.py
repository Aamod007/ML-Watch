"""
Unit tests for mlwatch.drift.data_drift.DataDriftDetector
"""

import numpy as np
import pytest

from mlwatch.drift.data_drift import DataDriftDetector


class TestDataDriftDetector:
    def _make_detector(self, n=500):
        rng = np.random.default_rng(0)
        baseline = rng.normal(size=(n, 4))
        det = DataDriftDetector(baseline_data=baseline)
        return det, baseline

    def test_no_drift_on_same_distribution(self):
        rng = np.random.default_rng(1)
        det, _ = self._make_detector()
        production = rng.normal(size=(200, 4))
        result = det.detect(production)
        assert result["overall_drifted"] is False

    def test_drift_detected_on_shifted_data(self):
        det, _ = self._make_detector(n=1000)
        rng = np.random.default_rng(2)
        # Severely shifted production data
        production = rng.normal(loc=10.0, size=(200, 4))
        result = det.detect(production)
        assert result["overall_drifted"] is True
        assert len(result["drifted_features"]) > 0

    def test_no_baseline_returns_error(self):
        det = DataDriftDetector()
        result = det.detect(np.random.randn(10, 3))
        assert "error" in result

    def test_feature_dimension_mismatch_raises(self):
        det, _ = self._make_detector()
        with pytest.raises(ValueError, match="dimension mismatch"):
            det.detect(np.random.randn(10, 7))  # wrong number of features

    def test_psi_per_feature_returns_dict(self):
        rng = np.random.default_rng(3)
        det, _ = self._make_detector()
        production = rng.normal(size=(100, 4))
        psi_dict = det.psi_per_feature(production)
        assert isinstance(psi_dict, dict)
        assert len(psi_dict) == 4

    def test_update_baseline(self):
        rng = np.random.default_rng(5)
        det = DataDriftDetector()
        assert det._baseline is None
        det.set_baseline(rng.normal(size=(200, 3)))
        assert det._baseline is not None
        assert det._baseline.shape[1] == 3
