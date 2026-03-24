"""
mlwatch.cloud.base
==================
Abstract base class for all cloud metric exporters.
"""

from __future__ import annotations

import abc
import logging
import threading
import queue
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetricPoint:
    """A single metric observation to be exported."""
    __slots__ = ("name", "value", "dimensions", "unit", "timestamp")

    def __init__(
        self,
        name: str,
        value: float,
        dimensions: Optional[Dict[str, str]] = None,
        unit: str = "None",
        timestamp: Optional[float] = None,
    ) -> None:
        import time
        self.name = name
        self.value = value
        self.dimensions = dimensions or {}
        self.unit = unit
        self.timestamp = timestamp or time.time()


class CloudExporter(abc.ABC):
    """
    Abstract base for cloud metric exporters.

    Subclasses implement :meth:`_flush_batch` which receives a list of
    :class:`MetricPoint` objects.  The base class handles async batching,
    background thread management, and fail-open error handling.

    Parameters
    ----------
    namespace:    Cloud metric namespace / prefix (e.g., ``'mlwatch'``).
    async_export: If True (default), metrics are buffered and exported by a
                  daemon background thread.
    batch_size:   Maximum metrics per cloud API call.
    flush_interval_s: How often (seconds) the background thread flushes.
    """

    def __init__(
        self,
        namespace: str = "mlwatch",
        async_export: bool = True,
        batch_size: int = 20,
        flush_interval_s: float = 10.0,
        max_queue: int = 10_000,
    ) -> None:
        self.namespace = namespace
        self.async_export = async_export
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s

        self._queue: queue.Queue[MetricPoint] = queue.Queue(maxsize=max_queue)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._metrics_exported: int = 0
        self._export_errors: int = 0

        if async_export:
            self._start_background_thread()

    # ------------------------------------------------------------------
    # Abstract method — subclasses implement cloud-specific flush
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _flush_batch(self, batch: List[MetricPoint]) -> None:
        """Export a batch of MetricPoints to the cloud backend."""

    # ------------------------------------------------------------------
    # Public emit API
    # ------------------------------------------------------------------

    def emit(
        self,
        name: str,
        value: float,
        dimensions: Optional[Dict[str, str]] = None,
        unit: str = "None",
    ) -> None:
        """
        Queue a single metric point for export.

        Parameters
        ----------
        name:       Metric name (will be prefixed with namespace).
        value:      Numeric metric value.
        dimensions: Key-value labels / tags sent with the metric.
        unit:       Optional unit identifier (backend-specific).
        """
        point = MetricPoint(
            name=f"{self.namespace}/{name}",
            value=float(value),
            dimensions=dimensions or {},
            unit=unit,
        )

        if self.async_export:
            try:
                self._queue.put_nowait(point)
            except queue.Full:
                logger.warning(
                    "CloudExporter: metric queue full — dropping metric '%s'. "
                    "Consider reducing export frequency.",
                    name,
                )
        else:
            self._safe_flush([point])

    def emit_batch(self, metrics: Dict[str, float], dimensions: Optional[Dict[str, str]] = None) -> None:
        """Emit multiple metrics at once from a dict."""
        for name, value in metrics.items():
            self.emit(name, value, dimensions=dimensions)

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _start_background_thread(self) -> None:
        self._thread = threading.Thread(
            target=self._export_loop,
            daemon=True,
            name=f"mlwatch-exporter-{self.__class__.__name__}",
        )
        self._thread.start()
        logger.debug("CloudExporter: background export thread started.")

    def _export_loop(self) -> None:
        """Drain the queue at regular intervals until stop is signalled."""
        import time
        while not self._stop_event.is_set():
            self._drain()
            time.sleep(self.flush_interval_s)
        # Final drain before shutdown
        self._drain()

    def _drain(self) -> None:
        """Drain all queued metrics into batches and flush each batch."""
        batch: List[MetricPoint] = []
        try:
            while True:
                batch.append(self._queue.get_nowait())
                if len(batch) >= self.batch_size:
                    self._safe_flush(batch)
                    batch = []
        except queue.Empty:
            pass
        if batch:
            self._safe_flush(batch)

    def _safe_flush(self, batch: List[MetricPoint]) -> None:
        """Flush a batch and absorb any errors (fail-open design)."""
        try:
            self._flush_batch(batch)
            self._metrics_exported += len(batch)
        except Exception as exc:  # noqa: BLE001
            self._export_errors += 1
            logger.warning(
                "CloudExporter: export error (batch=%d) — %s. "
                "Monitoring continues but metrics were dropped.",
                len(batch),
                exc,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Force an immediate synchronous flush of all queued metrics."""
        self._drain()

    def close(self) -> None:
        """Signal the background thread to stop and flush remaining metrics."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=15)
        logger.info(
            "CloudExporter: closed. Exported=%d, Errors=%d.",
            self._metrics_exported,
            self._export_errors,
        )

    def stats(self) -> Dict[str, int]:
        return {
            "metrics_exported": self._metrics_exported,
            "export_errors": self._export_errors,
            "queue_size": self._queue.qsize(),
        }
