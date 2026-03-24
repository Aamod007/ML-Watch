"""
mlwatch.core.monitor
=====================
ModelMonitor — the main entry point for mlwatch.

Drop-in replacement for model.predict() that automatically monitors:
  - Data drift (KS-test, PSI)
  - OOD input detection
  - Inference latency and error rate
  - Prediction output distribution
  - Threshold-based alerting
  - Cloud metric export
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np

from mlwatch.core.metrics import MetricsCollector
from mlwatch.drift.data_drift import DataDriftDetector
from mlwatch.drift.pred_drift import PredictionDriftMonitor
from mlwatch.drift.ood import OODDetector
from mlwatch.alerts.alerter import Alerter

logger = logging.getLogger(__name__)

Framework = Literal["pytorch", "tensorflow", "sklearn", "auto"]
Cloud = Literal["aws", "gcp", "azure", "local"]


def _build_exporter(cloud: str, **kwargs):
    """Factory for cloud exporters. Lazy imports to keep base package light."""
    cloud = cloud.lower()
    if cloud == "aws":
        from mlwatch.cloud.aws import AWSExporter
        return AWSExporter(**kwargs)
    if cloud == "gcp":
        from mlwatch.cloud.gcp import GCPExporter
        return GCPExporter(**kwargs)
    if cloud == "azure":
        from mlwatch.cloud.azure import AzureExporter
        return AzureExporter(**kwargs)
    # Default: local JSON/stdout
    from mlwatch.cloud.local import LocalExporter
    output_path = kwargs.pop("output_path", None)
    return LocalExporter(output_path=output_path)


def _build_framework_adapter(framework: str, model):
    """Factory for framework adapters."""
    framework = framework.lower()
    if framework == "pytorch":
        from mlwatch.frameworks.pytorch import PyTorchAdapter
        return PyTorchAdapter(model)
    if framework in ("tensorflow", "keras"):
        from mlwatch.frameworks.tensorflow import TensorFlowAdapter
        return TensorFlowAdapter(model)
    if framework == "sklearn":
        from mlwatch.frameworks.sklearn import SklearnAdapter
        return SklearnAdapter(model)
    # Auto-detect
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            from mlwatch.frameworks.pytorch import PyTorchAdapter
            return PyTorchAdapter(model)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        if isinstance(model, (tf.Module, tf.keras.Model)):
            from mlwatch.frameworks.tensorflow import TensorFlowAdapter
            return TensorFlowAdapter(model)
    except ImportError:
        pass
    from mlwatch.frameworks.sklearn import SklearnAdapter
    return SklearnAdapter(model)


class ModelMonitor:
    """
    Unified ML model monitoring wrapper.

    Wraps any PyTorch / TensorFlow / Scikit-learn model with a drop-in
    ``predict()`` call that automatically handles:
    - Pre-inference OOD detection
    - Inference latency and error tracking
    - Post-inference drift comparison (async)
    - Prediction output distribution tracking
    - Cloud metric export
    - Threshold-based alerting

    Parameters
    ----------
    model:              The model to monitor. Any PyTorch ``nn.Module``,
                        Keras ``Model``, or sklearn estimator.
    framework:          ``'pytorch'``, ``'tensorflow'``, ``'sklearn'``, or ``'auto'``.
    cloud:              ``'aws'``, ``'gcp'``, ``'azure'``, or ``'local'``.
    baseline_data:      Training feature matrix for drift baseline.
    baseline_labels:    Training labels (optional, for label distribution baseline).
    alert_webhook:      Slack / PagerDuty / custom URL for alert delivery.
    thresholds:         Dict of metric → threshold (see Configuration Reference in PRD).
    sample_rate:        Fraction of requests to monitor (1.0 = 100%).
    async_export:       Export metrics asynchronously (non-blocking).
    log_inputs:         If True, raw inputs are logged to the exporter.
    namespace:          Cloud metric namespace / prefix.
    cloud_kwargs:       Extra keyword arguments forwarded to the cloud exporter.
    """

    def __init__(
        self,
        model: Any,
        framework: Framework = "auto",
        cloud: Cloud = "local",
        baseline_data: Optional[np.ndarray] = None,
        baseline_labels: Optional[np.ndarray] = None,
        alert_webhook: Optional[str] = None,
        thresholds: Optional[Dict[str, float]] = None,
        sample_rate: float = 1.0,
        async_export: bool = True,
        log_inputs: bool = False,
        namespace: str = "mlwatch",
        **cloud_kwargs,
    ) -> None:
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")

        self.model = model
        self.framework = framework
        self.cloud = cloud
        self.sample_rate = sample_rate
        self.log_inputs = log_inputs
        self.namespace = namespace

        # ---- Build sub-components ----
        self._adapter = _build_framework_adapter(framework, model)
        self._exporter = _build_exporter(
            cloud, namespace=namespace, async_export=async_export, **cloud_kwargs
        )
        self._metrics = MetricsCollector()
        self._alerter = Alerter(
            webhook_url=alert_webhook,
            thresholds=thresholds or {},
        )
        self._drift_detector = DataDriftDetector(psi_threshold=0.2, ks_pvalue_threshold=0.05)
        self._pred_drift = PredictionDriftMonitor()
        self._ood_detector = OODDetector()

        # ---- Register baseline ----
        if baseline_data is not None:
            baseline_arr = np.atleast_2d(np.asarray(baseline_data, dtype=float))
            self._drift_detector.set_baseline(baseline_arr)
            self._ood_detector.fit(baseline_arr)

        self._baseline_labels = baseline_labels
        self._call_count: int = 0

        logger.info(
            "ModelMonitor initialised: framework=%s, cloud=%s, sample_rate=%.0f%%",
            framework,
            cloud,
            sample_rate * 100,
        )

    # ------------------------------------------------------------------
    # Core predict
    # ------------------------------------------------------------------

    def predict(self, inputs: Any) -> Any:
        """
        Drop-in predict wrapper with full monitoring.

        Parameters
        ----------
        inputs: Model input (numpy array, torch.Tensor, etc.)

        Returns
        -------
        Model predictions — identical type/shape as the underlying model.
        """
        self._call_count += 1

        # Sampling gate
        if self.sample_rate < 1.0 and random.random() > self.sample_rate:
            return self._adapter.predict(inputs)

        # Pre-inference: OOD check
        ood_result: Optional[Dict] = None
        inputs_arr: Optional[np.ndarray] = None
        try:
            inputs_arr = self._to_numpy(inputs)
            if self._ood_detector._baseline_mean is not None:
                ood_result = self._ood_detector.check(inputs_arr, raise_on_block=True)
        except Exception as exc:
            logger.warning("ModelMonitor: OOD check error (fail-open) — %s", exc)

        # Inference with latency tracking
        t0 = time.perf_counter()
        error_occurred = False
        predictions = None
        try:
            predictions = self._adapter.predict(inputs)
            self._metrics.record_success()
        except Exception as exc:
            error_occurred = True
            self._metrics.record_error()
            logger.error("ModelMonitor: inference error — %s", exc)
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self._metrics.record_latency(latency_ms)

        # Post-inference: async drift + metrics export
        self._post_inference_async(inputs_arr, predictions, latency_ms, ood_result)

        return predictions

    # ------------------------------------------------------------------
    # Post-inference processing (dispatched to background)
    # ------------------------------------------------------------------

    def _post_inference_async(
        self,
        inputs_arr: Optional[np.ndarray],
        predictions: Any,
        latency_ms: float,
        ood_result: Optional[Dict],
    ) -> None:
        """Schedule post-inference work. Runs synchronously for now; can be
        offloaded to a ThreadPoolExecutor for high-throughput deployments."""
        import threading
        t = threading.Thread(
            target=self._post_inference,
            args=(inputs_arr, predictions, latency_ms, ood_result),
            daemon=True,
        )
        t.start()

    def _post_inference(
        self,
        inputs_arr: Optional[np.ndarray],
        predictions: Any,
        latency_ms: float,
        ood_result: Optional[Dict],
    ) -> None:
        try:
            # Collect summary metrics
            perf = self._metrics.summary()

            # Data drift (only if baseline is registered)
            drift_result: Dict = {}
            if inputs_arr is not None and self._drift_detector._baseline is not None:
                try:
                    drift_result = self._drift_detector.detect(inputs_arr)
                except Exception as exc:
                    logger.warning("ModelMonitor: drift detection error — %s", exc)

            # Prediction drift
            if predictions is not None:
                try:
                    pred_arr = self._to_numpy(predictions)
                    self._pred_drift.record(pred_arr.ravel())
                except Exception as exc:
                    logger.warning("ModelMonitor: pred drift record error — %s", exc)

            # Build exportable metrics snapshot
            export_metrics: Dict[str, float] = {
                "latency_ms": round(latency_ms, 3),
                "error_rate": perf.get("error_rate", 0.0),
                "throughput_rps": perf.get("throughput_rps", 0.0),
                **{k: v for k, v in perf.items() if k.startswith("latency_p")},
            }

            if drift_result.get("summary_psi") is not None:
                export_metrics["psi"] = drift_result["summary_psi"]

            if ood_result:
                export_metrics["ood_rate"] = ood_result.get("ood_rate", 0.0)

            # Export to cloud
            try:
                self._exporter.emit_batch(export_metrics)
            except Exception as exc:
                logger.warning("ModelMonitor: export error (fail-open) — %s", exc)

            # Threshold alerting
            try:
                self._alerter.check(export_metrics)
            except Exception as exc:
                logger.warning("ModelMonitor: alert check error — %s", exc)

        except Exception as exc:  # noqa: BLE001
            # Ultimate fail-open safety net
            logger.warning("ModelMonitor: post-inference error — %s", exc)

    # ------------------------------------------------------------------
    # Public reporting API
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        """Return a complete monitoring snapshot."""
        return {
            "call_count": self._call_count,
            "performance": self._metrics.summary(),
            "exporter_stats": self._exporter.stats(),
            "fired_alerts": self._alerter.fired_alerts(),
            "ood_cumulative_rate": self._ood_detector.cumulative_ood_rate,
        }

    def drift_report(self, production_data: np.ndarray) -> Dict:
        """Run a manual on-demand drift report against the baseline."""
        return self._drift_detector.detect(production_data)

    def register_metric_hook(self, callback) -> None:
        """Register a custom metric callback (see MetricsCollector.register_hook)."""
        self._metrics.register_hook(callback)

    def flush(self) -> None:
        """Force immediate export of all queued metrics to the cloud."""
        self._exporter.flush()

    def close(self) -> None:
        """Shutdown the monitor, flush all pending metrics, and close connections."""
        self._exporter.flush()
        self._exporter.close()

    def __repr__(self) -> str:
        return (
            f"ModelMonitor(framework={self.framework!r}, cloud={self.cloud!r}, "
            f"calls={self._call_count}, sample_rate={self.sample_rate})"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(obj: Any) -> np.ndarray:
        """Convert a model input/output to a numpy array."""
        if isinstance(obj, np.ndarray):
            return obj
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy()
        except ImportError:
            pass
        try:
            import tensorflow as tf
            if tf.is_tensor(obj):
                return obj.numpy()
        except ImportError:
            pass
        return np.atleast_2d(np.asarray(obj, dtype=float))
