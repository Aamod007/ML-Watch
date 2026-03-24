"""
Unit tests for mlwatch.core.monitor.ModelMonitor and mlwatch.alerts.alerter.Alerter
"""

import time
import threading
import numpy as np
import pytest

from mlwatch.core.monitor import ModelMonitor
from mlwatch.alerts.alerter import Alerter, ThresholdRule, AlertSeverity


# ---------------------------------------------------------------------------
# Fake sklearn-compatible model for testing
# ---------------------------------------------------------------------------

class FakeModel:
    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.column_stack([np.ones(len(X)) * 0.5, np.ones(len(X)) * 0.5])


# ---------------------------------------------------------------------------
# ModelMonitor tests
# ---------------------------------------------------------------------------

class TestModelMonitor:
    def _make_monitor(self, baseline=True):
        rng = np.random.default_rng(0)
        model = FakeModel()
        bl = rng.normal(size=(200, 4)) if baseline else None
        return ModelMonitor(
            model=model,
            framework="sklearn",
            cloud="local",
            baseline_data=bl,
            thresholds={"error_rate": 0.5},
        )

    def test_predict_returns_array(self):
        monitor = self._make_monitor()
        X = np.random.randn(10, 4)
        result = monitor.predict(X)
        assert result is not None

    def test_call_count_increments(self):
        monitor = self._make_monitor()
        for _ in range(5):
            monitor.predict(np.random.randn(3, 4))
        assert monitor._call_count == 5

    def test_summary_structure(self):
        monitor = self._make_monitor()
        monitor.predict(np.random.randn(5, 4))
        time.sleep(0.05)  # allow async post-inference thread to complete
        summary = monitor.summary()
        assert "call_count" in summary
        assert "performance" in summary
        assert "exporter_stats" in summary

    def test_sample_rate_reduces_monitoring(self):
        monitor = self._make_monitor()
        monitor.sample_rate = 0.0  # effectively never monitor
        for _ in range(10):
            monitor.predict(np.random.randn(5, 4))
        # With sample_rate=0, no latency records should exist
        assert monitor._metrics._total_requests == 0

    def test_close_does_not_raise(self):
        monitor = self._make_monitor()
        monitor.close()  # should flush and close cleanly

    def test_repr(self):
        monitor = self._make_monitor()
        r = repr(monitor)
        assert "ModelMonitor" in r
        assert "sklearn" in r


# ---------------------------------------------------------------------------
# Alerter tests
# ---------------------------------------------------------------------------

class TestAlerter:
    def test_threshold_breach_fires_alert(self):
        alerter = Alerter(thresholds={"psi": 0.2})
        fired = alerter.check({"psi": 0.5})
        assert len(fired) == 1
        assert fired[0].metric == "psi"

    def test_threshold_under_limit_no_alert(self):
        alerter = Alerter(thresholds={"psi": 0.2})
        fired = alerter.check({"psi": 0.1})
        assert len(fired) == 0

    def test_cooldown_suppresses_repeated_alerts(self):
        alerter = Alerter(thresholds={"psi": 0.2}, cooldown_s=9999)
        first = alerter.check({"psi": 0.5})
        second = alerter.check({"psi": 0.5})  # should be suppressed
        assert len(first) == 1
        assert len(second) == 0

    def test_fired_alerts_history(self):
        alerter = Alerter(thresholds={"error_rate": 0.01}, cooldown_s=0)
        alerter.check({"error_rate": 0.5})
        history = alerter.fired_alerts()
        assert len(history) == 1
        assert "metric" in history[0]

    def test_custom_rule(self):
        alerter = Alerter()
        rule = ThresholdRule(
            metric="custom",
            threshold=100.0,
            operator="gt",
            severity=AlertSeverity.CRITICAL,
            cooldown_s=0,
        )
        alerter.add_rule(rule)
        fired = alerter.check({"custom": 200.0})
        assert len(fired) == 1
        assert fired[0].severity == AlertSeverity.CRITICAL

    def test_alert_to_dict_has_required_keys(self):
        alerter = Alerter(thresholds={"psi": 0.2}, cooldown_s=0)
        fired = alerter.check({"psi": 0.9})
        d = fired[0].to_dict()
        for key in ("source", "severity", "metric", "value", "threshold", "timestamp"):
            assert key in d
