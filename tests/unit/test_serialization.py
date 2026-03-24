"""Unit tests for serialization utilities."""

import tempfile
import os
import numpy as np
import pytest

from mlwatch.utils.serialization import save_baseline, load_baseline, baseline_exists


class TestSerialization:
    def test_save_and_load_roundtrip(self):
        rng = np.random.default_rng(100)
        data = rng.normal(size=(300, 6))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "baseline.json")
            saved_path = save_baseline(data, path, compress=True)
            loaded = load_baseline(saved_path)

        assert loaded["n_samples"] == 300
        assert loaded["n_features"] == 6
        np.testing.assert_allclose(loaded["means"], data.mean(axis=0), rtol=1e-5)
        np.testing.assert_allclose(loaded["stds"], data.std(axis=0), rtol=1e-5)

    def test_feature_names_preserved(self):
        data = np.random.randn(50, 3)
        names = ["alpha", "beta", "gamma"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bl.json")
            saved = save_baseline(data, path, feature_names=names, compress=False)
            loaded = load_baseline(saved)
        assert loaded["feature_names"] == names

    def test_baseline_exists(self):
        data = np.random.randn(50, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bl.json")
            assert not baseline_exists(path)
            save_baseline(data, path, compress=True)
            assert baseline_exists(path + ".gz") or baseline_exists(path)

    def test_raw_sample_stored(self):
        data = np.random.randn(100, 4)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bl.json")
            saved = save_baseline(data, path, max_sample_rows=50, compress=False)
            loaded = load_baseline(saved)
        assert loaded["raw_sample"] is not None
        assert len(loaded["raw_sample"]) == 50

    def test_mismatched_feature_names_raises(self):
        data = np.random.randn(50, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bl.json")
            with pytest.raises(ValueError, match="feature_names"):
                save_baseline(data, path, feature_names=["only_one"])

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_baseline("/nonexistent/path/baseline.json")
