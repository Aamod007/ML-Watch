"""
mlwatch.cloud.local
====================
Local fallback exporter — writes metrics to a JSON file or stdout.
Used in dev mode when no cloud backend is configured (``cloud='local'``).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from mlwatch.cloud.base import CloudExporter, MetricPoint

logger = logging.getLogger(__name__)


class LocalExporter(CloudExporter):
    """
    Development / testing exporter that writes metrics to a JSON lines file
    (or prints to stdout if no filepath is given).

    Parameters
    ----------
    output_path: File path for JSON lines output. If None, prints to stdout.
    namespace:   Metric name prefix.
    async_export: Match base class default (True).
    """

    def __init__(
        self,
        output_path: Optional[str] = None,
        namespace: str = "mlwatch",
        async_export: bool = True,
    ) -> None:
        super().__init__(namespace=namespace, async_export=async_export)
        self.output_path = output_path
        self._file_lock = threading.Lock()

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info("LocalExporter: writing metrics to %s", output_path)
        else:
            logger.info("LocalExporter: writing metrics to stdout.")

    def _flush_batch(self, batch: List[MetricPoint]) -> None:
        records = [
            {
                "timestamp": datetime.fromtimestamp(p.timestamp, tz=timezone.utc).isoformat(),
                "metric": p.name,
                "value": p.value,
                "dimensions": p.dimensions,
                "unit": p.unit,
            }
            for p in batch
        ]

        if self.output_path:
            with self._file_lock:
                with open(self.output_path, "a", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec) + "\n")
        else:
            for rec in records:
                print(json.dumps(rec), flush=True)  # noqa: T201
