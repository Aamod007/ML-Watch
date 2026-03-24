"""
mlwatch.core.decorators
========================
The ``@watch`` decorator API — zero-code-change monitoring.

Usage::

    from mlwatch import watch

    @watch(framework="sklearn", cloud="gcp", baseline=X_train)
    def predict(model, X):
        return model.predict(X)
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def watch(
    framework: str = "auto",
    cloud: str = "local",
    baseline: Optional[np.ndarray] = None,
    alert_webhook: Optional[str] = None,
    thresholds: Optional[Dict[str, float]] = None,
    sample_rate: float = 1.0,
    namespace: str = "mlwatch",
    **cloud_kwargs,
) -> Callable:
    """
    Function decorator that wraps any predict function with monitoring.

    The decorated function signature is preserved — ``@watch`` is transparent
    to callers.

    Parameters
    ----------
    framework:     ML framework identifier.
    cloud:         Cloud backend for metric export.
    baseline:      Training data array for drift detection.
    alert_webhook: Webhook URL for alert delivery.
    thresholds:    Threshold rules dict.
    sample_rate:   Fraction of calls to monitor.
    namespace:     Metric namespace prefix.
    **cloud_kwargs: Passed to the cloud exporter constructor.

    Returns
    -------
    Decorator callable.
    """
    def decorator(fn: Callable) -> Callable:
        # Build monitoring components once at decoration time (not per-call)
        from mlwatch.core.metrics import MetricsCollector
        from mlwatch.alerts.alerter import Alerter
        from mlwatch.drift.data_drift import DataDriftDetector
        from mlwatch.drift.ood import OODDetector

        _metrics = MetricsCollector()
        _alerter = Alerter(webhook_url=alert_webhook, thresholds=thresholds or {})
        _drift = DataDriftDetector()
        _ood = OODDetector()

        if baseline is not None:
            arr = np.atleast_2d(np.asarray(baseline, dtype=float))
            _drift.set_baseline(arr)
            _ood.fit(arr)

        # Build exporter lazily on first call
        _exporter_holder: Dict[str, Any] = {}

        def _get_exporter():
            if "exporter" not in _exporter_holder:
                from mlwatch.core.monitor import _build_exporter
                _exporter_holder["exporter"] = _build_exporter(
                    cloud, namespace=namespace, **cloud_kwargs
                )
            return _exporter_holder["exporter"]

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if sample_rate < 1.0 and random.random() > sample_rate:
                return fn(*args, **kwargs)

            t0 = time.perf_counter()
            result = None
            errored = False
            try:
                result = fn(*args, **kwargs)
                _metrics.record_success()
            except Exception as exc:
                errored = True
                _metrics.record_error()
                raise
            finally:
                latency_ms = (time.perf_counter() - t0) * 1000.0
                _metrics.record_latency(latency_ms)

            # Export
            try:
                perf = _metrics.summary()
                export_snapshot = {
                    "latency_ms": round(latency_ms, 3),
                    "error_rate": perf.get("error_rate", 0.0),
                    **{k: v for k, v in perf.items() if k.startswith("latency_p")},
                }
                _get_exporter().emit_batch(export_snapshot)
                _alerter.check(export_snapshot)
            except Exception as exc:  # noqa: BLE001
                logger.warning("@watch: export error (fail-open) — %s", exc)

            return result

        # Attach monitoring components for introspection
        wrapper._monitor_metrics = _metrics
        wrapper._monitor_alerter = _alerter
        wrapper._monitor_drift = _drift

        return wrapper

    return decorator
