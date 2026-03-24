"""
mlwatch.cloud.gcp
==================
GCP Cloud Monitoring (Stackdriver) exporter.

Install: pip install mlwatch[gcp]

Authentication: Service Account JSON key via GOOGLE_APPLICATION_CREDENTIALS
environment variable, or Workload Identity in GKE.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from mlwatch.cloud.base import CloudExporter, MetricPoint

logger = logging.getLogger(__name__)


class GCPExporter(CloudExporter):
    """
    Exports mlwatch metrics to GCP Cloud Monitoring (formerly Stackdriver).

    Parameters
    ----------
    project_id:   GCP project ID.  Required.
    namespace:    Metric type prefix (default ``'mlwatch'``).
    async_export: Buffer and export asynchronously.
    """

    def __init__(
        self,
        project_id: str,
        namespace: str = "mlwatch",
        async_export: bool = True,
    ) -> None:
        super().__init__(namespace=namespace, async_export=async_export)
        self.project_id = project_id
        self._client = None  # lazy init

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from google.cloud import monitoring_v3
        except ImportError as exc:
            raise ImportError(
                "google-cloud-monitoring is required for GCP export. "
                "Install with: pip install mlwatch[gcp]"
            ) from exc
        self._client = monitoring_v3.MetricServiceClient()
        logger.info("GCPExporter: Cloud Monitoring client initialised (project=%s).", self.project_id)
        return self._client

    def _flush_batch(self, batch: List[MetricPoint]) -> None:
        try:
            from google.cloud import monitoring_v3
            from google.protobuf import timestamp_pb2
        except ImportError as exc:
            raise ImportError(
                "google-cloud-monitoring is required. pip install mlwatch[gcp]"
            ) from exc

        client = self._get_client()
        project_name = f"projects/{self.project_id}"
        now = time.time()

        time_series = []
        for p in batch:
            metric_type = (
                f"custom.googleapis.com/{self.namespace}/{p.name.split('/', 1)[-1]}"
            )
            series = monitoring_v3.TimeSeries(
                metric=monitoring_v3.types.Metric(
                    type_=metric_type,
                    labels={k: str(v) for k, v in p.dimensions.items()},
                ),
                resource=monitoring_v3.types.MonitoredResource(
                    type_="global",
                ),
                points=[
                    monitoring_v3.types.Point(
                        interval=monitoring_v3.types.TimeInterval(
                            end_time=timestamp_pb2.Timestamp(seconds=int(p.timestamp))
                        ),
                        value=monitoring_v3.types.TypedValue(double_value=p.value),
                    )
                ],
            )
            time_series.append(series)

        client.create_time_series(name=project_name, time_series=time_series)
        logger.debug("GCPExporter: flushed %d metrics to Cloud Monitoring.", len(batch))
