"""
mlwatch.frameworks.tensorflow
==============================
TensorFlow / Keras adapter using a native ``tf.keras.callbacks.Callback``.

Supports:
- TF SavedModel
- Keras Sequential / Functional / Subclassed APIs
- Compatible with ``model.fit()`` and ``model.predict()``

Usage (callback API)::

    from mlwatch.frameworks.tensorflow import MonitorCallback

    model.predict(X, callbacks=[MonitorCallback(cloud="gcp", baseline=X_train)])
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

from mlwatch.frameworks.base import FrameworkAdapter

logger = logging.getLogger(__name__)


class TensorFlowAdapter(FrameworkAdapter):
    """Adapter for TensorFlow / Keras models."""

    def predict(self, inputs: Any) -> Any:
        """Run model.predict(); converts numpy ↔ tensor as needed."""
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for framework='tensorflow'. "
                "Install with: pip install tensorflow"
            ) from exc

        if isinstance(inputs, np.ndarray):
            return self.model.predict(inputs, verbose=0)
        return self.model.predict(inputs, verbose=0)


class MonitorCallback:
    """
    Keras-compatible callback that plugs into ``model.predict()`` and
    ``model.fit()`` natively.

    Usage::

        model.predict(X, callbacks=[MonitorCallback(cloud="gcp", baseline=X_train)])

    Parameters
    ----------
    cloud:         Cloud backend for metric export.
    baseline:      Training feature matrix for drift baseline.
    alert_webhook: Alert webhook URL.
    thresholds:    Alert threshold dict.
    namespace:     Metric namespace prefix.
    """

    def __init__(
        self,
        model=None,
        cloud: str = "local",
        baseline: Optional[np.ndarray] = None,
        alert_webhook: Optional[str] = None,
        thresholds: Optional[dict] = None,
        namespace: str = "mlwatch",
        **cloud_kwargs,
    ) -> None:
        try:
            import tensorflow as tf

            class _InnerCallback(tf.keras.callbacks.Callback):
                """Inner Keras callback with monitoring wired in."""

                def __init__(cb_self, outer) -> None:
                    super().__init__()
                    cb_self._outer = outer
                    cb_self._batch_start: float = 0.0

                def on_predict_batch_begin(cb_self, batch, logs=None):
                    cb_self._batch_start = time.perf_counter()

                def on_predict_batch_end(cb_self, batch, logs=None):
                    latency_ms = (time.perf_counter() - cb_self._batch_start) * 1000.0
                    cb_self._outer._metrics.record_latency(latency_ms)
                    cb_self._outer._metrics.record_success()

                    snap = {"latency_ms": round(latency_ms, 3)}
                    try:
                        cb_self._outer._exporter.emit_batch(snap)
                        cb_self._outer._alerter.check(snap)
                    except Exception as exc:
                        logger.warning("MonitorCallback: export error — %s", exc)

            self._keras_callback = _InnerCallback(self)

        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for MonitorCallback. "
                "Install with: pip install tensorflow"
            ) from exc

        from mlwatch.core.metrics import MetricsCollector
        from mlwatch.alerts.alerter import Alerter
        from mlwatch.core.monitor import _build_exporter

        self._metrics = MetricsCollector()
        self._alerter = Alerter(webhook_url=alert_webhook, thresholds=thresholds or {})
        self._exporter = _build_exporter(cloud, namespace=namespace, **cloud_kwargs)

    def __iter__(self):
        """Make MonitorCallback iterable so ``callbacks=[mc]`` works."""
        yield self._keras_callback

    def __len__(self):
        return 1
