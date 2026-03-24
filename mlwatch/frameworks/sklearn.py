"""
mlwatch.frameworks.sklearn
===========================
Scikit-learn pipeline / estimator adapter.

Provides:
- ``SklearnAdapter``  — used internally by ModelMonitor
- ``MonitoredPipeline`` — standalone transparent proxy wrapping any sklearn estimator
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from mlwatch.frameworks.base import FrameworkAdapter

logger = logging.getLogger(__name__)


class SklearnAdapter(FrameworkAdapter):
    """
    Adapter for scikit-learn estimators.

    Intercepts ``predict()``, ``predict_proba()``, and ``transform()`` calls
    through the uniform ``predict()`` interface.  The actual method called
    depends on what the estimator supports.
    """

    def predict(self, inputs: Any) -> Any:
        inputs_arr = self.to_numpy(inputs)
        if hasattr(self.model, "predict_proba"):
            try:
                return self.model.predict_proba(inputs_arr)
            except Exception:
                pass
        return self.model.predict(inputs_arr)


class MonitoredPipeline:
    """
    Transparent proxy that wraps a sklearn Pipeline or estimator with
    mlwatch monitoring.

    Fully compatible with GridSearchCV and joblib serialization.

    Usage::

        from mlwatch.frameworks.sklearn import MonitoredPipeline

        monitored = MonitoredPipeline(pipeline, cloud="azure", baseline=X_train)
        y_pred = monitored.predict(X_test)

    Parameters
    ----------
    estimator:    Any sklearn estimator or Pipeline.
    cloud:        Cloud exporter backend.
    baseline:     Training feature matrix for drift baseline.
    alert_webhook: Alert delivery URL.
    thresholds:   Alert threshold dict.
    sample_rate:  Fraction of calls to monitor.
    namespace:    Metric namespace prefix.
    """

    def __init__(
        self,
        estimator,
        cloud: str = "local",
        baseline: Optional[np.ndarray] = None,
        alert_webhook: Optional[str] = None,
        thresholds: Optional[dict] = None,
        sample_rate: float = 1.0,
        namespace: str = "mlwatch",
        **cloud_kwargs,
    ) -> None:
        from mlwatch.core.monitor import ModelMonitor

        self._monitor = ModelMonitor(
            model=estimator,
            framework="sklearn",
            cloud=cloud,
            baseline_data=baseline,
            alert_webhook=alert_webhook,
            thresholds=thresholds,
            sample_rate=sample_rate,
            namespace=namespace,
            **cloud_kwargs,
        )
        self._estimator = estimator

    # sklearn API passthrough

    def predict(self, X) -> Any:
        return self._monitor.predict(X)

    def predict_proba(self, X) -> Any:
        if hasattr(self._estimator, "predict_proba"):
            return self._monitor.predict(X)
        raise AttributeError(f"{type(self._estimator).__name__} has no predict_proba")

    def transform(self, X) -> Any:
        """For transformers (e.g., pipeline with preprocessors)."""
        if hasattr(self._estimator, "transform"):
            return self._estimator.transform(X)
        raise AttributeError(f"{type(self._estimator).__name__} has no transform")

    def fit(self, X, y=None, **fit_params):
        return self._estimator.fit(X, y, **fit_params)

    @property
    def monitor(self):
        """Access the underlying ModelMonitor for reporting."""
        return self._monitor

    def __repr__(self) -> str:
        return f"MonitoredPipeline({self._estimator!r})"

    def __getattr__(self, name: str):
        """Fall through to the wrapped estimator for any unrecognised attribute."""
        return getattr(self._estimator, name)
