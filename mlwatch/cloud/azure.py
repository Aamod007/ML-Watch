"""
mlwatch.cloud.azure
====================
Azure Monitor Ingestion exporter.

Install: pip install mlwatch[azure]

Authentication: Managed Identity or Service Principal via AZURE_CLIENT_ID /
AZURE_CLIENT_SECRET / AZURE_TENANT_ID environment variables.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from mlwatch.cloud.base import CloudExporter, MetricPoint

logger = logging.getLogger(__name__)


class AzureExporter(CloudExporter):
    """
    Exports mlwatch metrics to Azure Monitor via the Logs Ingestion API.

    Parameters
    ----------
    endpoint:          Data collection endpoint URL (from your DCE resource).
    rule_id:           Data Collection Rule immutable ID.
    stream_name:       Stream name defined in the DCR (e.g., ``'Custom-mlwatch'``).
    namespace:         Metric name prefix.
    async_export:      Buffer and export asynchronously.
    credential:        Optional ``azure.core.credentials.TokenCredential``.
                       Defaults to ``DefaultAzureCredential``.
    """

    def __init__(
        self,
        endpoint: str,
        rule_id: str,
        stream_name: str,
        namespace: str = "mlwatch",
        async_export: bool = True,
        credential=None,
    ) -> None:
        super().__init__(namespace=namespace, async_export=async_export)
        self.endpoint = endpoint
        self.rule_id = rule_id
        self.stream_name = stream_name
        self._credential = credential
        self._client = None  # lazy init

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from azure.monitor.ingestion import LogsIngestionClient
            from azure.identity import DefaultAzureCredential
        except ImportError as exc:
            raise ImportError(
                "azure-monitor-ingestion and azure-identity are required. "
                "Install with: pip install mlwatch[azure]"
            ) from exc

        credential = self._credential or DefaultAzureCredential()
        self._client = LogsIngestionClient(endpoint=self.endpoint, credential=credential)
        logger.info("AzureExporter: Logs Ingestion client initialised.")
        return self._client

    def _flush_batch(self, batch: List[MetricPoint]) -> None:
        client = self._get_client()

        logs = [
            {
                "TimeGenerated": datetime.fromtimestamp(p.timestamp, tz=timezone.utc).isoformat(),
                "MetricName": p.name,
                "MetricValue": p.value,
                "Unit": p.unit,
                **{f"dim_{k}": v for k, v in p.dimensions.items()},
            }
            for p in batch
        ]

        client.upload(
            rule_id=self.rule_id,
            stream_name=self.stream_name,
            logs=logs,
        )
        logger.debug("AzureExporter: flushed %d metrics to Azure Monitor.", len(batch))
