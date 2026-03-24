"""
mlwatch.utils.stats
===================
Statistical test utilities for drift and anomaly detection.

Provides:
- KS-test (Kolmogorov-Smirnov) for continuous features
- PSI (Population Stability Index) for distribution shift magnitude
- Chi-square test for categorical features
- Z-score based outlier detection
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PSI — Population Stability Index
# ---------------------------------------------------------------------------

def compute_psi(
    baseline: np.ndarray,
    production: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-8,
) -> float:
    """
    Compute the Population Stability Index (PSI) between a baseline and
    production distribution.

    PSI interpretation:
        < 0.1   → No significant change (stable)
        0.1-0.2 → Moderate change (some drift)
        > 0.2   → Significant change (retrain signal)

    Parameters
    ----------
    baseline:   1-D array of training / reference values.
    production: 1-D array of production values.
    n_bins:     Number of equal-width bins to use.
    eps:        Small constant to avoid log(0).

    Returns
    -------
    float: PSI score.
    """
    baseline = np.asarray(baseline, dtype=float).ravel()
    production = np.asarray(production, dtype=float).ravel()

    # Determine bin edges on the combined range
    combined_min = min(baseline.min(), production.min())
    combined_max = max(baseline.max(), production.max())

    if combined_min == combined_max:
        # Degenerate: all values identical ⇒ no drift
        return 0.0

    bins = np.linspace(combined_min, combined_max, n_bins + 1)
    bins[0] -= 1e-6  # include the minimum value in the first bin
    bins[-1] += 1e-6

    base_counts, _ = np.histogram(baseline, bins=bins)
    prod_counts, _ = np.histogram(production, bins=bins)

    base_pct = base_counts / (base_counts.sum() + eps)
    prod_pct = prod_counts / (prod_counts.sum() + eps)

    # Clip to avoid log(0)
    base_pct = np.clip(base_pct, eps, None)
    prod_pct = np.clip(prod_pct, eps, None)

    psi = np.sum((prod_pct - base_pct) * np.log(prod_pct / base_pct))
    return float(psi)


def psi_label(psi: float) -> str:
    """Return human-readable drift severity label for a PSI score."""
    if psi < 0.1:
        return "stable"
    if psi < 0.2:
        return "moderate"
    return "significant"


# ---------------------------------------------------------------------------
# KS-Test — Kolmogorov-Smirnov
# ---------------------------------------------------------------------------

def compute_ks_test(
    baseline: np.ndarray,
    production: np.ndarray,
) -> Tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test for continuous drift detection.

    Parameters
    ----------
    baseline:   1-D reference array (training data).
    production: 1-D production array.

    Returns
    -------
    (statistic, p_value): KS statistic and p-value.
        Low p-value (< threshold, e.g. 0.05) signals drift.
    """
    try:
        from scipy import stats as scipy_stats  # lazy import
    except ImportError as exc:
        raise ImportError(
            "scipy is required for KS-test drift detection. "
            "Install it with: pip install scipy"
        ) from exc

    baseline = np.asarray(baseline, dtype=float).ravel()
    production = np.asarray(production, dtype=float).ravel()
    result = scipy_stats.ks_2samp(baseline, production)
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# Chi-Square Test — Categorical Drift
# ---------------------------------------------------------------------------

def compute_chi_square(
    baseline: np.ndarray,
    production: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    """
    Chi-square goodness-of-fit test for categorical feature drift.

    Both arrays should contain integer-encoded or string category labels.

    Returns
    -------
    (statistic, p_value): Chi-square statistic and p-value.
    """
    try:
        from scipy import stats as scipy_stats
    except ImportError as exc:
        raise ImportError(
            "scipy is required for Chi-square drift detection. "
            "Install it with: pip install scipy"
        ) from exc

    baseline = np.asarray(baseline).ravel()
    production = np.asarray(production).ravel()

    # Build a shared category universe
    categories = np.union1d(np.unique(baseline), np.unique(production))

    base_counts = np.array(
        [np.sum(baseline == c) for c in categories], dtype=float
    )
    prod_counts = np.array(
        [np.sum(production == c) for c in categories], dtype=float
    )

    # Scale baseline to total of production
    base_expected = base_counts / (base_counts.sum() + eps) * prod_counts.sum()
    base_expected = np.clip(base_expected, eps, None)

    result = scipy_stats.chisquare(prod_counts, f_exp=base_expected)
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# Z-Score Outlier Detection
# ---------------------------------------------------------------------------

def compute_z_scores(
    data: np.ndarray,
    baseline_mean: Optional[np.ndarray] = None,
    baseline_std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute per-feature Z-scores for an input matrix.

    If baseline statistics are not provided, they are computed from `data`.

    Parameters
    ----------
    data:           (n_samples, n_features) array.
    baseline_mean:  (n_features,) mean vector from training baseline.
    baseline_std:   (n_features,) std-dev vector from training baseline.

    Returns
    -------
    np.ndarray: (n_samples, n_features) absolute Z-score matrix.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    mean = baseline_mean if baseline_mean is not None else data.mean(axis=0)
    std = baseline_std if baseline_std is not None else data.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # avoid division by zero
    return np.abs((data - mean) / std)


def zscore_outlier_flags(
    data: np.ndarray,
    baseline_mean: Optional[np.ndarray] = None,
    baseline_std: Optional[np.ndarray] = None,
    sigma_threshold: float = 3.0,
) -> np.ndarray:
    """
    Return a boolean array (n_samples,) indicating OOD samples by Z-score.

    A sample is flagged if ANY feature exceeds `sigma_threshold` standard
    deviations from the baseline mean.
    """
    z = compute_z_scores(data, baseline_mean, baseline_std)
    return z.max(axis=1) > sigma_threshold


# ---------------------------------------------------------------------------
# Per-Feature Drift Summary
# ---------------------------------------------------------------------------

def per_feature_drift_report(
    baseline: np.ndarray,
    production: np.ndarray,
    feature_names: Optional[list] = None,
    categorical_mask: Optional[np.ndarray] = None,
) -> Dict[str, dict]:
    """
    Compute a full per-feature drift report combining PSI and KS-test.

    Parameters
    ----------
    baseline:         (n_baseline, n_features) reference array.
    production:       (n_production, n_features) production array.
    feature_names:    Optional list of column names (length = n_features).
    categorical_mask: Boolean array of length n_features; True = categorical.

    Returns
    -------
    Dict mapping feature name → {psi, ks_stat, ks_pvalue, is_drifted, severity}
    """
    baseline = np.atleast_2d(np.asarray(baseline, dtype=float))
    production = np.atleast_2d(np.asarray(production, dtype=float))
    n_features = baseline.shape[1]

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    if categorical_mask is None:
        categorical_mask = np.zeros(n_features, dtype=bool)

    report: Dict[str, dict] = {}
    for i, name in enumerate(feature_names):
        base_col = baseline[:, i]
        prod_col = production[:, i]

        psi = compute_psi(base_col, prod_col)
        if categorical_mask[i]:
            stat, pvalue = compute_chi_square(base_col.astype(int), prod_col.astype(int))
            test_used = "chi_square"
        else:
            stat, pvalue = compute_ks_test(base_col, prod_col)
            test_used = "ks_test"

        report[name] = {
            "psi": round(psi, 6),
            "psi_severity": psi_label(psi),
            "test_used": test_used,
            "test_statistic": round(stat, 6),
            "test_pvalue": round(pvalue, 6),
            "is_drifted": psi > 0.2 or pvalue < 0.05,
        }

    return report
