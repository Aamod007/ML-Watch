"""
mlwatch.core.session
=====================
MonitorSession — per-request context tracker and context manager API.

Usage::

    with monitor_session(model, framework="tensorflow", cloud="azure") as session:
        preds = model.predict(X_batch)
        session.log_metric("custom_score", 0.94)
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, Callable, Dict, Generator, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MonitorSession:
    """
    Scoped monitoring context for a single inference batch.

    Automatically measures wall-clock latency, records errors/successes to
    the shared :class:`~mlwatch.core.metrics.MetricsCollector`, and exports
    a metrics snapshot at the end of the context.

    Parameters
    ----------
    metrics_collector: Shared MetricsCollector from the parent ModelMonitor.
    cloud_exporter:    Optional CloudExporter for metric upload.
    alerter:           Optional Alerter for threshold checks.
    dimensions:        Extra key/value labels attached to all exported metrics.
    """

    def __init__(
        self,
        metrics_collector=None,
        cloud_exporter=None,
        alerter=None,
        dimensions: Optional[Dict[str, str]] = None,
    ) -> None:
        self._metrics = metrics_collector
        self._exporter = cloud_exporter
        self._alerter = alerter
        self._dimensions = dimensions or {}
        self._start_time: Optional[float] = None
        self._custom: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "MonitorSession":
        self._start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> bool:
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0

        if self._metrics is not None:
            self._metrics.record_latency(elapsed_ms)
            if exc_type is not None:
                self._metrics.record_error()
            else:
                self._metrics.record_success()

        # Log custom metrics to collector
        for name, value in self._custom.items():
            if self._metrics:
                self._metrics.record_custom(name, value)

        # Export snapshot
        self._export_snapshot(elapsed_ms, errored=exc_type is not None)

        # Fail-open: don't suppress exceptions
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_metric(self, name: str, value: float) -> None:
        """
        Log a custom metric within the current session.

        Parameters
        ----------
        name:  Metric name.
        value: Numeric value.
        """
        self._custom[name] = float(value)
        logger.debug("MonitorSession.log_metric: %s = %s", name, value)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _export_snapshot(self, latency_ms: float, errored: bool) -> None:
        if self._exporter is None and self._alerter is None:
            return

        snapshot: Dict[str, float] = {
            "latency_ms": round(latency_ms, 3),
            "error": float(errored),
            **self._custom,
        }

        if self._exporter:
            try:
                self._exporter.emit_batch(snapshot, dimensions=self._dimensions)
            except Exception as exc:  # noqa: BLE001
                logger.warning("MonitorSession: export error — %s", exc)

        if self._alerter:
            try:
                self._alerter.check(snapshot, extra={"session": True})
            except Exception as exc:  # noqa: BLE001
                logger.warning("MonitorSession: alert check error — %s", exc)


# ---------------------------------------------------------------------------
# Convenience context manager factory
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def monitor_session(
    model,
    framework: str = "sklearn",
    cloud: str = "local",
    **kwargs,
) -> Generator[MonitorSession, None, None]:
    """
    Convenience factory that creates a lightweight :class:`MonitorSession`
    without a full :class:`~mlwatch.core.monitor.ModelMonitor`.

    Usage::

        with monitor_session(model, framework="tensorflow", cloud="azure") as s:
            preds = model.predict(X)
            s.log_metric("custom_score", 0.94)
    """
    from mlwatch.cloud.local import LocalExporter
    from mlwatch.core.metrics import MetricsCollector

    exporter = LocalExporter(namespace="mlwatch")
    collector = MetricsCollector()

    session = MonitorSession(
        metrics_collector=collector,
        cloud_exporter=exporter,
    )

    with session:
        yield session
