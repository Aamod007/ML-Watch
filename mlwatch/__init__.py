"""
mlwatch — ML Model Monitoring Library for Cloud Production
==========================================================
A unified, framework-agnostic SDK for detecting data drift, tracking prediction
quality, monitoring system performance, and surfacing anomalies in production ML.

Usage::

    from mlwatch import ModelMonitor

    monitor = ModelMonitor(
        model=my_model,
        framework="pytorch",
        cloud="aws",
        baseline_data=X_train,
        alert_webhook="https://hooks.slack.com/...",
        thresholds={"psi": 0.2, "latency_p95_ms": 300},
    )
    prediction = monitor.predict(X_input)
"""

from mlwatch.core.monitor import ModelMonitor
from mlwatch.core.session import MonitorSession, monitor_session
from mlwatch.core.decorators import watch

__all__ = [
    "ModelMonitor",
    "MonitorSession",
    "monitor_session",
    "watch",
]

__version__ = "0.1.0"
__author__ = "ML Platform Team"
__license__ = "Apache 2.0"
