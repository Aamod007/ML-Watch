"""
mlwatch.core.metrics
=====================
MetricsCollector — real-time aggregation engine for inference performance metrics.

Tracks:
  - Inference latency (ring-buffer for rolling p50/p95/p99)
  - Error rate
  - Throughput (RPS)
  - Confidence scores
  - Custom user-defined metrics
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_WINDOW = 1000  # default rolling window size


class MetricsCollector:
    """
    Thread-safe ring-buffer based metrics aggregation engine.

    All collections use fixed-size double-ended queues (deques) so memory
    usage is bounded regardless of request volume.

    Parameters
    ----------
    window_size: Maximum number of observations to keep per metric.
    """

    def __init__(self, window_size: int = _WINDOW) -> None:
        self._window = window_size
        self._lock = threading.Lock()

        # Core metric buffers
        self._latencies: Deque[float] = collections.deque(maxlen=window_size)
        self._errors: Deque[int] = collections.deque(maxlen=window_size)   # 0/1 per request
        self._confidences: Deque[float] = collections.deque(maxlen=window_size)
        self._timestamps: Deque[float] = collections.deque(maxlen=window_size)

        # Cumulative totals (never trimmed)
        self._total_requests: int = 0
        self._total_errors: int = 0

        # Custom metric buffers: name → deque
        self._custom: Dict[str, Deque[float]] = {}

        # Custom metric hooks registered by user
        self._custom_hooks: List[Callable[[], Dict[str, float]]] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_latency(self, latency_ms: float) -> None:
        """Record inference latency in milliseconds."""
        with self._lock:
            self._latencies.append(latency_ms)
            self._timestamps.append(time.time())
            self._total_requests += 1

    def record_error(self) -> None:
        """Record a prediction error / exception."""
        with self._lock:
            self._errors.append(1)
            self._total_errors += 1

    def record_success(self) -> None:
        """Record a successful prediction (for error-rate denominator)."""
        with self._lock:
            self._errors.append(0)

    def record_confidence(self, confidence: float) -> None:
        """Record a per-prediction confidence / probability score."""
        with self._lock:
            self._confidences.append(float(confidence))

    def record_custom(self, name: str, value: float) -> None:
        """Record an arbitrary named metric value."""
        with self._lock:
            if name not in self._custom:
                self._custom[name] = collections.deque(maxlen=self._window)
            self._custom[name].append(float(value))

    def register_hook(self, callback: Callable[[], Dict[str, float]]) -> None:
        """
        Register a user-defined metric callback.

        The callback will be called on each :meth:`summary` invocation and
        its returned dict values will be merged into the summary output.

        Parameters
        ----------
        callback: Zero-argument callable returning ``{metric_name: value}``.
        """
        self._custom_hooks.append(callback)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def latency_percentiles(self) -> Dict[str, float]:
        """Return rolling latency percentiles (p50, p95, p99) in milliseconds."""
        with self._lock:
            if not self._latencies:
                return {}
            arr = np.array(self._latencies)
        return {
            "latency_p50_ms": float(np.percentile(arr, 50)),
            "latency_p95_ms": float(np.percentile(arr, 95)),
            "latency_p99_ms": float(np.percentile(arr, 99)),
            "latency_mean_ms": float(arr.mean()),
        }

    def error_rate(self) -> float:
        """Rolling error rate (fraction of requests that errored)."""
        with self._lock:
            if not self._errors:
                return 0.0
            return float(np.mean(list(self._errors)))

    def throughput_rps(self, window_s: float = 60.0) -> float:
        """
        Approximate requests-per-second over the last `window_s` seconds.
        """
        with self._lock:
            if not self._timestamps:
                return 0.0
            now = time.time()
            cutoff = now - window_s
            recent = sum(1 for ts in self._timestamps if ts >= cutoff)
            elapsed = min(window_s, now - self._timestamps[0])
        return recent / max(elapsed, 1.0)

    def confidence_stats(self) -> Dict[str, float]:
        """Return rolling statistics over confidence scores."""
        with self._lock:
            if not self._confidences:
                return {}
            arr = np.array(self._confidences)
        return {
            "confidence_mean": float(arr.mean()),
            "confidence_p10": float(np.percentile(arr, 10)),
            "confidence_std": float(arr.std()),
        }

    def custom_stats(self) -> Dict[str, float]:
        """Return rolling means for all custom metrics."""
        with self._lock:
            return {
                name: float(np.mean(list(buf)))
                for name, buf in self._custom.items()
                if buf
            }

    def summary(self) -> Dict[str, Any]:
        """Return a complete performance metrics snapshot."""
        result: Dict[str, Any] = {
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "error_rate": round(self.error_rate(), 6),
            "throughput_rps": round(self.throughput_rps(), 4),
            **self.latency_percentiles(),
            **self.confidence_stats(),
            **self.custom_stats(),
        }
        # Invoke registered hooks
        for hook in self._custom_hooks:
            try:
                result.update(hook())
            except Exception as exc:  # noqa: BLE001
                logger.warning("MetricsCollector: custom hook error — %s", exc)
        return result
