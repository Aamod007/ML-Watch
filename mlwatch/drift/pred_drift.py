"""
mlwatch.drift.pred_drift
=========================
Tracks how model output distributions change over a rolling time window.

Capabilities:
  - Output histogram tracking for classifiers and regressors
  - Confidence score degradation alerts
  - Label distribution shift monitoring
  - Rolling mean / std / percentile statistics
"""

from __future__ import annotations

import collections
import logging
import time
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PredictionDriftMonitor:
    """
    Maintains a rolling window of prediction outputs and surfaces distributional
    shift relative to an optional baseline output distribution.

    Parameters
    ----------
    window_size:          Maximum number of recent predictions to keep in the
                          rolling buffer for computing statistics.
    confidence_threshold: Alert if the rolling mean confidence drops below this
                          (for classifiers that return probabilities).
    baseline_outputs:     Optional reference outputs from validation / production
                          warm-up period for drift comparison.
    """

    def __init__(
        self,
        window_size: int = 1000,
        confidence_threshold: float = 0.5,
        baseline_outputs: Optional[np.ndarray] = None,
    ) -> None:
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold

        # Deque gives O(1) append + automatic eviction of old entries
        self._output_window: Deque[float] = collections.deque(maxlen=window_size)
        self._confidence_window: Deque[float] = collections.deque(maxlen=window_size)
        self._timestamps: Deque[float] = collections.deque(maxlen=window_size)

        self._baseline_outputs: Optional[np.ndarray] = None
        if baseline_outputs is not None:
            self.set_baseline(baseline_outputs)

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def set_baseline(self, outputs: np.ndarray) -> None:
        """Register baseline output distribution (e.g., validation predictions)."""
        self._baseline_outputs = np.asarray(outputs, dtype=float).ravel()
        logger.info(
            "PredictionDriftMonitor: baseline set (%d predictions).",
            len(self._baseline_outputs),
        )

    # ------------------------------------------------------------------
    # Record predictions
    # ------------------------------------------------------------------

    def record(
        self,
        outputs: np.ndarray,
        confidences: Optional[np.ndarray] = None,
    ) -> None:
        """
        Append a batch of predictions to the rolling window.

        Parameters
        ----------
        outputs:     Model outputs (class indices, regression values, or logits).
        confidences: Per-prediction confidence / probability scores.  When a
                     classifier returns probabilities the caller can pass
                     ``proba.max(axis=1)`` here.
        """
        ts = time.time()
        outputs = np.asarray(outputs, dtype=float).ravel()

        for val in outputs:
            self._output_window.append(float(val))
            self._timestamps.append(ts)

        if confidences is not None:
            confidences = np.asarray(confidences, dtype=float).ravel()
            for c in confidences:
                self._confidence_window.append(float(c))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def rolling_stats(self) -> Dict[str, float]:
        """Return basic rolling window statistics over recent predictions."""
        if not self._output_window:
            return {}

        arr = np.array(self._output_window)
        stats: Dict[str, float] = {
            "count": len(arr),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
        }

        if self._confidence_window:
            conf = np.array(self._confidence_window)
            stats["confidence_mean"] = float(conf.mean())
            stats["confidence_p10"] = float(np.percentile(conf, 10))
            stats["confidence_below_threshold"] = float(
                (conf < self.confidence_threshold).mean()
            )

        return stats

    def detect_drift(self) -> Dict:
        """
        Compare the current rolling window to the baseline output distribution.

        Returns a drift report dict comparable to DataDriftDetector.detect().
        """
        if self._baseline_outputs is None or not self._output_window:
            return {"error": "insufficient_data"}

        from mlwatch.utils.stats import compute_psi, compute_ks_test, psi_label

        production = np.array(self._output_window)
        psi = compute_psi(self._baseline_outputs, production)
        ks_stat, ks_pvalue = compute_ks_test(self._baseline_outputs, production)

        # Confidence degradation alert
        conf_alert = False
        conf_mean = None
        if self._confidence_window:
            conf_mean = float(np.array(self._confidence_window).mean())
            conf_alert = conf_mean < self.confidence_threshold

        result = {
            "output_psi": round(psi, 6),
            "psi_severity": psi_label(psi),
            "ks_statistic": round(ks_stat, 6),
            "ks_pvalue": round(ks_pvalue, 6),
            "output_drifted": psi > 0.2 or ks_pvalue < 0.05,
            "confidence_mean": conf_mean,
            "confidence_alert": conf_alert,
            "rolling_stats": self.rolling_stats(),
        }

        if result["output_drifted"]:
            logger.warning(
                "PredictionDriftMonitor: output drift detected! "
                "PSI=%.4f, KS-pvalue=%.4f",
                psi,
                ks_pvalue,
            )

        if conf_alert:
            logger.warning(
                "PredictionDriftMonitor: low confidence alert — mean=%.4f < threshold=%.4f",
                conf_mean,
                self.confidence_threshold,
            )

        return result

    def label_distribution(self, n_classes: Optional[int] = None) -> Dict[str, float]:
        """
        Return the label distribution as proportions (classifier use-case).

        Parameters
        ----------
        n_classes: Expected number of classes. Inferred from window if None.
        """
        if not self._output_window:
            return {}

        arr = np.array(self._output_window, dtype=int)
        classes = np.arange(n_classes) if n_classes else np.unique(arr)
        total = len(arr)
        return {
            f"class_{c}": float((arr == c).sum() / total) for c in classes
        }
