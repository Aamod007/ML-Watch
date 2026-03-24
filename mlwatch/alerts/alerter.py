"""
mlwatch.alerts.alerter
=======================
Threshold-based alerting engine with webhook support (Slack, PagerDuty, HTTP).

Features:
- Configurable threshold rules per metric
- Multi-severity: INFO, WARNING, CRITICAL
- Alert cooldown / deduplication to prevent notification storms
- Async webhook delivery (non-blocking)
- Fail-open: alert delivery errors never propagate to inference path
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------

class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Alert event
# ---------------------------------------------------------------------------

class Alert:
    """Represents a single triggered alert event."""

    def __init__(
        self,
        metric: str,
        value: float,
        threshold: float,
        severity: AlertSeverity,
        message: str,
        extra: Optional[Dict] = None,
    ) -> None:
        self.metric = metric
        self.value = value
        self.threshold = threshold
        self.severity = severity
        self.message = message
        self.extra = extra or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": "mlwatch",
            "severity": self.severity.value,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "message": self.message,
            "timestamp": self.timestamp,
            **self.extra,
        }

    def to_slack_payload(self) -> Dict[str, Any]:
        """Format the alert as a Slack webhook message payload."""
        emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}[self.severity.value]
        color = {"INFO": "#36a64f", "WARNING": "#ffcc00", "CRITICAL": "#ff0000"}[
            self.severity.value
        ]
        return {
            "text": f"{emoji} *mlwatch {self.severity}*: {self.message}",
            "attachments": [
                {
                    "color": color,
                    "fields": [
                        {"title": "Metric", "value": self.metric, "short": True},
                        {"title": "Value", "value": str(round(self.value, 6)), "short": True},
                        {"title": "Threshold", "value": str(self.threshold), "short": True},
                        {"title": "Timestamp", "value": self.timestamp, "short": True},
                    ],
                    "footer": "mlwatch monitoring",
                }
            ],
        }


# ---------------------------------------------------------------------------
# Rule definition
# ---------------------------------------------------------------------------

class ThresholdRule:
    """
    Defines a single alerting threshold rule.

    Parameters
    ----------
    metric:       Metric name to watch (e.g., ``'psi'``, ``'latency_p95_ms'``).
    threshold:    Numeric threshold value.
    operator:     Comparison operator: ``'gt'``, ``'gte'``, ``'lt'``, ``'lte'``.
    severity:     ``AlertSeverity`` level to use when rule triggers.
    cooldown_s:   Minimum seconds between repeated alerts for this metric.
    """

    _OPS = {
        "gt": lambda v, t: v > t,
        "gte": lambda v, t: v >= t,
        "lt": lambda v, t: v < t,
        "lte": lambda v, t: v <= t,
    }

    def __init__(
        self,
        metric: str,
        threshold: float,
        operator: str = "gt",
        severity: AlertSeverity = AlertSeverity.WARNING,
        cooldown_s: float = 300.0,
    ) -> None:
        if operator not in self._OPS:
            raise ValueError(f"operator must be one of {list(self._OPS)}, got '{operator}'")
        self.metric = metric
        self.threshold = threshold
        self.operator = operator
        self.severity = severity
        self.cooldown_s = cooldown_s
        self._last_fired: float = 0.0

    def evaluate(self, value: float) -> bool:
        """Return True if the threshold is breached AND cooldown has elapsed."""
        if not self._OPS[self.operator](value, self.threshold):
            return False
        if time.monotonic() - self._last_fired < self.cooldown_s:
            logger.debug(
                "Alert rule '%s' suppressed by cooldown (%.0fs remaining).",
                self.metric,
                self.cooldown_s - (time.monotonic() - self._last_fired),
            )
            return False
        self._last_fired = time.monotonic()
        return True


# ---------------------------------------------------------------------------
# Alerter
# ---------------------------------------------------------------------------

class Alerter:
    """
    Central alert dispatcher.  Evaluates metric values against registered
    threshold rules and dispatches webhook notifications asynchronously.

    Parameters
    ----------
    webhook_url:  URL to POST alert JSON payloads to.  Automatically detects
                  Slack format (hooks.slack.com) vs. generic JSON.
    thresholds:   Dict of metric → threshold value shortcuts.  These are
                  converted to ``ThresholdRule`` objects with sensible defaults.
    cooldown_s:   Default cooldown seconds for auto-created rules.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        thresholds: Optional[Dict[str, float]] = None,
        cooldown_s: float = 300.0,
    ) -> None:
        self.webhook_url = webhook_url
        self.cooldown_s = cooldown_s
        self._rules: Dict[str, ThresholdRule] = {}
        self._fired_alerts: List[Alert] = []
        self._lock = threading.Lock()

        if thresholds:
            self._register_default_rules(thresholds)

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def _register_default_rules(self, thresholds: Dict[str, float]) -> None:
        """Convert the user-supplied thresholds dict into ThresholdRule objects."""
        # Metrics where we alert on values ABOVE threshold
        gt_metrics = {"psi", "ks_statistic", "ood_rate", "error_rate", "latency_p50_ms",
                      "latency_p95_ms", "latency_p99_ms"}
        # Metrics where we alert on values BELOW threshold (degrade alert)
        lt_metrics = {"ks_pvalue", "confidence_mean"}

        severity_map = {
            "psi": AlertSeverity.WARNING,
            "ood_rate": AlertSeverity.WARNING,
            "error_rate": AlertSeverity.CRITICAL,
            "latency_p95_ms": AlertSeverity.WARNING,
            "latency_p99_ms": AlertSeverity.CRITICAL,
        }

        for metric, threshold in thresholds.items():
            operator = "lt" if metric in lt_metrics else "gt"
            severity = severity_map.get(metric, AlertSeverity.WARNING)
            self.add_rule(
                ThresholdRule(metric, threshold, operator, severity, self.cooldown_s)
            )

    def add_rule(self, rule: ThresholdRule) -> None:
        """Register a custom :class:`ThresholdRule`."""
        self._rules[rule.metric] = rule
        logger.debug(
            "Alerter: rule registered — metric='%s' threshold=%s operator='%s'",
            rule.metric,
            rule.threshold,
            rule.operator,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def check(self, metrics: Dict[str, float], extra: Optional[Dict] = None) -> List[Alert]:
        """
        Evaluate a metrics dict against all registered rules.

        Fires webhook notifications asynchronously for any triggered rules.

        Parameters
        ----------
        metrics: Dict of metric_name → current value.
        extra:   Additional context included in alert payloads.

        Returns
        -------
        List of :class:`Alert` objects that were triggered this call.
        """
        fired: List[Alert] = []
        for metric, value in metrics.items():
            rule = self._rules.get(metric)
            if rule is None:
                continue
            if rule.evaluate(value):
                msg = (
                    f"Metric '{metric}' breached threshold: "
                    f"{value:.4f} {rule.operator} {rule.threshold}"
                )
                alert = Alert(
                    metric=metric,
                    value=value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    message=msg,
                    extra=extra,
                )
                fired.append(alert)
                logger.log(
                    logging.WARNING if rule.severity == AlertSeverity.WARNING else logging.ERROR,
                    "[mlwatch ALERT] %s — %s",
                    rule.severity.value,
                    msg,
                )
                if self.webhook_url:
                    self._dispatch_async(alert)

        with self._lock:
            self._fired_alerts.extend(fired)

        return fired

    # ------------------------------------------------------------------
    # Webhook delivery
    # ------------------------------------------------------------------

    def _dispatch_async(self, alert: Alert) -> None:
        """Fire-and-forget webhook delivery in a daemon thread."""
        thread = threading.Thread(
            target=self._send_webhook,
            args=(alert,),
            daemon=True,
            name=f"mlwatch-alert-{alert.metric}",
        )
        thread.start()

    def _send_webhook(self, alert: Alert) -> None:
        """POST the alert to the configured webhook URL."""
        try:
            import httpx  # lazy import — optional dependency
        except ImportError:
            logger.warning(
                "mlwatch: httpx not installed — cannot send webhook alert. "
                "Install with: pip install httpx"
            )
            return

        try:
            is_slack = "hooks.slack.com" in (self.webhook_url or "")
            payload = alert.to_slack_payload() if is_slack else alert.to_dict()

            resp = httpx.post(
                self.webhook_url,
                json=payload,
                timeout=10.0,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code >= 400:
                logger.warning(
                    "Alerter: webhook POST to %s returned HTTP %d.",
                    self.webhook_url,
                    resp.status_code,
                )
            else:
                logger.debug(
                    "Alerter: webhook delivered for metric='%s' → HTTP %d.",
                    alert.metric,
                    resp.status_code,
                )
        except Exception as exc:  # noqa: BLE001
            # Fail-open: never propagate alerting errors to inference path
            logger.warning("Alerter: webhook delivery error — %s", exc)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def fired_alerts(self) -> List[Dict]:
        """Return all alerts fired since init as list of dicts."""
        with self._lock:
            return [a.to_dict() for a in self._fired_alerts]

    def clear_history(self) -> None:
        with self._lock:
            self._fired_alerts.clear()
