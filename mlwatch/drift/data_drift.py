"""
mlwatch.drift.data_drift
=========================
Data drift detectors that compare production feature distributions against a
stored training baseline.

Supported statistical tests:
  - PSI  (Population Stability Index) — continuous & discrete
  - KS   (Kolmogorov-Smirnov two-sample test) — continuous only
  - Chi² (Chi-square goodness-of-fit) — categorical only
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from mlwatch.utils.stats import (
    compute_psi,
    compute_ks_test,
    compute_chi_square,
    per_feature_drift_report,
    psi_label,
)

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """
    Stateless drift detector that computes per-feature drift scores for a
    batch of production observations against a registered baseline.

    Parameters
    ----------
    baseline_data:     2-D numpy array (n_baseline, n_features) — training data
                       reference. Can be set later via :meth:`set_baseline`.
    feature_names:     Optional column names for reporting.
    categorical_mask:  Boolean array indicating which features are categorical.
    psi_threshold:     PSI score above which a feature is flagged as drifted.
    ks_pvalue_threshold: KS-test p-value below which a feature is flagged.
    """

    def __init__(
        self,
        baseline_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        categorical_mask: Optional[np.ndarray] = None,
        psi_threshold: float = 0.2,
        ks_pvalue_threshold: float = 0.05,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self._feature_names = feature_names
        self._categorical_mask = categorical_mask
        self._baseline: Optional[np.ndarray] = None
        self._baseline_stats: Optional[Dict] = None

        if baseline_data is not None:
            self.set_baseline(baseline_data, feature_names, categorical_mask)

    # ------------------------------------------------------------------
    # Baseline registration
    # ------------------------------------------------------------------

    def set_baseline(
        self,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        categorical_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Register training data as the drift reference baseline."""
        data = np.atleast_2d(np.asarray(data, dtype=float))
        n_features = data.shape[1]

        if feature_names:
            self._feature_names = feature_names
        if self._feature_names is None:
            self._feature_names = [f"feature_{i}" for i in range(n_features)]

        if categorical_mask is not None:
            self._categorical_mask = np.asarray(categorical_mask, dtype=bool)
        if self._categorical_mask is None:
            self._categorical_mask = np.zeros(n_features, dtype=bool)

        self._baseline = data
        self._baseline_stats = {
            "means": data.mean(axis=0),
            "stds": data.std(axis=0),
            "mins": data.min(axis=0),
            "maxs": data.max(axis=0),
            "n_samples": len(data),
            "n_features": n_features,
        }
        logger.info(
            "DataDriftDetector: baseline registered (%d samples, %d features).",
            *data.shape,
        )

    def from_baseline_stats(self, stats: dict) -> None:
        """
        Hydrate from a deserialized baseline dict (see utils.serialization).
        """
        self._baseline_stats = stats
        self._feature_names = stats.get("feature_names")
        self._baseline = stats.get("raw_sample")  # may be None

    # ------------------------------------------------------------------
    # Core drift analysis
    # ------------------------------------------------------------------

    def detect(self, production_data: np.ndarray) -> Dict:
        """
        Run drift detection on a batch of production observations.

        Parameters
        ----------
        production_data: (n_samples, n_features) array of production inputs.

        Returns
        -------
        dict with keys:
            - ``feature_report``: per-feature drift results
            - ``overall_drifted``: True if any feature is drifted
            - ``drifted_features``: list of drifted feature names
            - ``summary_psi``: mean PSI across all features
        """
        if self._baseline is None:
            logger.warning(
                "DataDriftDetector: no baseline registered — skipping drift check."
            )
            return {"error": "no_baseline"}

        production_data = np.atleast_2d(np.asarray(production_data, dtype=float))

        if production_data.shape[1] != self._baseline.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: baseline has "
                f"{self._baseline.shape[1]} features, production has "
                f"{production_data.shape[1]}."
            )

        report = per_feature_drift_report(
            baseline=self._baseline,
            production=production_data,
            feature_names=self._feature_names,
            categorical_mask=self._categorical_mask,
        )

        drifted = [k for k, v in report.items() if v["is_drifted"]]
        mean_psi = float(np.mean([v["psi"] for v in report.values()]))

        result = {
            "feature_report": report,
            "overall_drifted": len(drifted) > 0,
            "drifted_features": drifted,
            "summary_psi": round(mean_psi, 6),
            "psi_severity": psi_label(mean_psi),
        }

        if drifted:
            logger.warning(
                "DataDriftDetector: DRIFT detected in %d/%d features: %s  "
                "(mean PSI=%.4f)",
                len(drifted),
                len(report),
                drifted,
                mean_psi,
            )
        else:
            logger.debug(
                "DataDriftDetector: no drift detected (mean PSI=%.4f).", mean_psi
            )

        return result

    def psi_per_feature(self, production_data: np.ndarray) -> Dict[str, float]:
        """Convenience method — returns just {feature_name: psi_score} dict."""
        result = self.detect(production_data)
        if "feature_report" not in result:
            return {}
        return {k: v["psi"] for k, v in result["feature_report"].items()}
