"""
mlwatch.cloud.aws
==================
AWS CloudWatch exporter using boto3.

Install: pip install mlwatch[aws]

Authentication: Uses the boto3 credential chain:
- IAM Role (recommended in production on EC2/ECS/Lambda)
- AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY environment variables
- ~/.aws/credentials file
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from mlwatch.cloud.base import CloudExporter, MetricPoint

logger = logging.getLogger(__name__)

_UNIT_MAP = {
    "ms": "Milliseconds",
    "s": "Seconds",
    "bytes": "Bytes",
    "count": "Count",
    "percent": "Percent",
    "None": "None",
}


class AWSExporter(CloudExporter):
    """
    Exports mlwatch metrics to AWS CloudWatch.

    Parameters
    ----------
    namespace:     CloudWatch metric namespace (e.g., ``'mlwatch'``).
    region:        AWS region name (e.g., ``'us-east-1'``). If None, uses
                   the default boto3 region.
    profile_name:  AWS credentials profile.  None = default credential chain.
    async_export:  Buffer and export asynchronously (recommended).
    storage_resolution: 1 (high-resolution, 1-second) or 60 (standard).
    """

    def __init__(
        self,
        namespace: str = "mlwatch",
        region: Optional[str] = None,
        profile_name: Optional[str] = None,
        async_export: bool = True,
        storage_resolution: int = 60,
    ) -> None:
        super().__init__(namespace=namespace, async_export=async_export)
        self.region = region
        self.profile_name = profile_name
        self.storage_resolution = storage_resolution
        self._client = None  # lazy init

    def _get_client(self):
        """Lazy-init the boto3 CloudWatch client."""
        if self._client is not None:
            return self._client
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for AWS CloudWatch export. "
                "Install with: pip install mlwatch[aws]"
            ) from exc

        session = boto3.Session(
            region_name=self.region,
            profile_name=self.profile_name,
        )
        self._client = session.client("cloudwatch")
        logger.info(
            "AWSExporter: CloudWatch client initialised (region=%s).", self.region or "default"
        )
        return self._client

    def _flush_batch(self, batch: List[MetricPoint]) -> None:
        client = self._get_client()

        # CloudWatch accepts max 1000 metrics per PutMetricData call
        for i in range(0, len(batch), 20):
            chunk = batch[i : i + 20]
            metric_data = []
            for p in chunk:
                metric_name = p.name.split("/", 1)[-1]  # strip namespace
                dimensions = [
                    {"Name": k, "Value": v} for k, v in p.dimensions.items()
                ]
                metric_data.append(
                    {
                        "MetricName": metric_name,
                        "Dimensions": dimensions,
                        "Timestamp": datetime.fromtimestamp(
                            p.timestamp, tz=timezone.utc
                        ),
                        "Value": p.value,
                        "Unit": _UNIT_MAP.get(p.unit, "None"),
                        "StorageResolution": self.storage_resolution,
                    }
                )

            client.put_metric_data(
                Namespace=self.namespace,
                MetricData=metric_data,
            )
            logger.debug("AWSExporter: flushed %d metrics to CloudWatch.", len(chunk))
