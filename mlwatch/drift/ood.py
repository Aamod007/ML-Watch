"""
mlwatch.drift.ood
==================
Out-of-Distribution (OOD) detection for pre-inference input screening.

Algorithms:
- Isolation Forest (primary, sklearn-based)
- Z-score per-feature outlier detection (lightweight fallback)

The OOD detector can be configured to:
  - ``log``   — record the OOD event and continue inference
  - ``flag``  — mark inputs as OOD in returned metadata, continue inference
  - ``block`` — raise OODInputError to halt inference (use with caution)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from mlwatch.utils.stats import zscore_outlier_flags

logger = logging.getLogger(__name__)


class OODInputError(ValueError):
    """Raised when OOD inputs are detected and ``on_ood='block'``."""


OODAction = Literal["log", "flag", "block"]


class OODDetector:
    """
    Pre-inference out-of-distribution detector.

    Parameters
    ----------
    baseline_data:   Training data used to fit the Isolation Forest and Z-score
                     statistics.  Must be set before calling :meth:`check`.
    contamination:   Expected proportion of outliers in training data
                     (passed to IsolationForest).
    sigma_threshold: Z-score threshold per feature for the lightweight detector.
    on_ood:          Action to take when OOD inputs are detected.
                     ``'log'`` (default), ``'flag'``, or ``'block'``.
    use_isolation_forest: If False, only Z-score detection is used (faster,
                          no sklearn dependency at runtime).
    """

    def __init__(
        self,
        baseline_data: Optional[np.ndarray] = None,
        contamination: float = 0.05,
        sigma_threshold: float = 4.0,
        on_ood: OODAction = "log",
        use_isolation_forest: bool = True,
    ) -> None:
        self.contamination = contamination
        self.sigma_threshold = sigma_threshold
        self.on_ood = on_ood
        self.use_isolation_forest = use_isolation_forest

        self._iso_forest = None
        self._baseline_mean: Optional[np.ndarray] = None
        self._baseline_std: Optional[np.ndarray] = None
        self._n_ood_total: int = 0
        self._n_checked_total: int = 0

        if baseline_data is not None:
            self.fit(baseline_data)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, baseline_data: np.ndarray) -> None:
        """
        Fit the OOD detector on reference (training) data.

        Parameters
        ----------
        baseline_data: (n_samples, n_features) training feature matrix.
        """
        data = np.atleast_2d(np.asarray(baseline_data, dtype=float))
        self._baseline_mean = data.mean(axis=0)
        self._baseline_std = data.std(axis=0)

        if self.use_isolation_forest:
            try:
                from sklearn.ensemble import IsolationForest
                self._iso_forest = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_jobs=-1,
                )
                self._iso_forest.fit(data)
                logger.info(
                    "OODDetector: IsolationForest fitted on %d samples, "
                    "%d features.",
                    *data.shape,
                )
            except ImportError:
                logger.warning(
                    "OODDetector: scikit-learn not found — "
                    "falling back to Z-score detection only."
                )
                self._iso_forest = None
        else:
            logger.info("OODDetector: Z-score mode (no Isolation Forest).")

    # ------------------------------------------------------------------
    # Inference-time check
    # ------------------------------------------------------------------

    def check(
        self,
        inputs: np.ndarray,
        raise_on_block: bool = True,
    ) -> Dict:
        """
        Screen inputs for out-of-distribution samples.

        Parameters
        ----------
        inputs:         (n_samples, n_features) array to screen. Can also be a
                        1-D array for single-sample inference.
        raise_on_block: If ``on_ood='block'`` and OOD is detected, raise
                        :exc:`OODInputError` when this is True (default).

        Returns
        -------
        dict with keys:
            - ``ood_flags``:  boolean array (n_samples,) — True = OOD
            - ``ood_count``:  int
            - ``ood_rate``:   float (fraction of inputs flagged)
            - ``scores``:     per-sample OOD scores (lower = more OOD)
            - ``action_taken``: what was done (``'none'``, ``'flagged'``, ``'blocked'``)
        """
        inputs = np.atleast_2d(np.asarray(inputs, dtype=float))
        n_samples = len(inputs)
        self._n_checked_total += n_samples

        # --- Z-score flags (fast, always computed) ---
        z_flags = zscore_outlier_flags(
            inputs,
            baseline_mean=self._baseline_mean,
            baseline_std=self._baseline_std,
            sigma_threshold=self.sigma_threshold,
        )

        # --- Isolation Forest scores (if fitted) ---
        iso_scores: Optional[np.ndarray] = None
        iso_flags: Optional[np.ndarray] = None
        if self._iso_forest is not None:
            # IsolationForest.predict → +1 = inlier, -1 = outlier
            iso_pred = self._iso_forest.predict(inputs)
            iso_flags = iso_pred == -1
            # decision_function: lower = more anomalous
            iso_scores = self._iso_forest.decision_function(inputs)

        # Combine: flag if EITHER method flags the sample
        if iso_flags is not None:
            ood_flags = z_flags | iso_flags
        else:
            ood_flags = z_flags
            iso_scores = np.zeros(n_samples)

        ood_count = int(ood_flags.sum())
        ood_rate = ood_count / n_samples
        self._n_ood_total += ood_count

        result: Dict = {
            "ood_flags": ood_flags,
            "ood_count": ood_count,
            "ood_rate": round(ood_rate, 6),
            "scores": iso_scores if iso_scores is not None else np.zeros(n_samples),
            "zscore_flags": z_flags,
            "action_taken": "none",
        }

        if ood_count > 0:
            if self.on_ood in ("log", "flag"):
                log_fn = logger.warning if ood_rate > 0.1 else logger.debug
                log_fn(
                    "OODDetector: %d/%d inputs flagged as OOD (rate=%.2f%%).",
                    ood_count,
                    n_samples,
                    ood_rate * 100,
                )
                result["action_taken"] = "flagged"

            elif self.on_ood == "block":
                logger.error(
                    "OODDetector: BLOCKING inference — %d OOD inputs detected.",
                    ood_count,
                )
                result["action_taken"] = "blocked"
                if raise_on_block:
                    raise OODInputError(
                        f"{ood_count}/{n_samples} inputs are out-of-distribution. "
                        "Inference blocked. Set `on_ood='flag'` to allow OOD inputs."
                    )

        return result

    # ------------------------------------------------------------------
    # Cumulative stats
    # ------------------------------------------------------------------

    @property
    def cumulative_ood_rate(self) -> float:
        """Overall OOD rate since the detector was created / last reset."""
        if self._n_checked_total == 0:
            return 0.0
        return self._n_ood_total / self._n_checked_total

    def reset_stats(self) -> None:
        """Reset cumulative OOD counters."""
        self._n_ood_total = 0
        self._n_checked_total = 0
