"""
mlwatch.utils.serialization
============================
Helpers for saving and loading baseline statistics to/from disk or cloud
object storage, so that drift comparisons survive process restarts.

Baseline Format (compressed JSON)
-----------------------------------
{
    "version": "0.1",
    "created_at": "<ISO-8601 timestamp>",
    "n_samples": 1000,
    "n_features": 20,
    "feature_names": ["f0", "f1", ...],
    "means": [0.5, 1.2, ...],
    "stds": [0.1, 0.3, ...],
    "mins": [...],
    "maxs": [...],
    "raw_sample": null   # optional 1000-sample subsample for KS-test
}
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Core save / load
# ---------------------------------------------------------------------------

def save_baseline(
    data: np.ndarray,
    path: str | os.PathLike,
    feature_names: Optional[List[str]] = None,
    max_sample_rows: int = 2000,
    compress: bool = True,
) -> str:
    """
    Compute and persist summary statistics from `data` to a JSON(.gz) file.

    Parameters
    ----------
    data:            2-D array of shape (n_samples, n_features).
    path:            Destination file path (.json or .json.gz).
    feature_names:   Optional list of column names.
    max_sample_rows: Maximum rows to store as a raw subsample for KS-tests.
    compress:        If True, gzip-compress the output file.

    Returns
    -------
    str: Resolved absolute path of the saved file.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    n_samples, n_features = data.shape

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but data has "
            f"{n_features} columns."
        )

    # Subsample for KS-test (avoid storing huge raw arrays)
    if n_samples > max_sample_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_samples, size=max_sample_rows, replace=False)
        sample = data[idx]
    else:
        sample = data

    payload: Dict[str, Any] = {
        "version": "0.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": n_samples,
        "n_features": n_features,
        "feature_names": feature_names,
        "means": data.mean(axis=0).tolist(),
        "stds": data.std(axis=0).tolist(),
        "mins": data.min(axis=0).tolist(),
        "maxs": data.max(axis=0).tolist(),
        "raw_sample": sample.tolist(),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw_bytes = json.dumps(payload, cls=_NumpyEncoder, indent=2).encode("utf-8")

    if compress or path.suffix == ".gz":
        if not str(path).endswith(".gz"):
            path = path.with_suffix(path.suffix + ".gz")
        with gzip.open(path, "wb") as f:
            f.write(raw_bytes)
    else:
        path.write_bytes(raw_bytes)

    logger.info("Baseline saved → %s  (%d samples, %d features)", path, n_samples, n_features)
    return str(path.resolve())


def load_baseline(path: str | os.PathLike) -> Dict[str, Any]:
    """
    Load a saved baseline from a JSON or gzip-compressed JSON file.

    Returns
    -------
    dict: The baseline payload with numpy arrays for statistics.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline file not found: {path}")

    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            payload = json.loads(f.read().decode("utf-8"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))

    # Convert lists back to numpy arrays for convenience
    for key in ("means", "stds", "mins", "maxs"):
        if key in payload:
            payload[key] = np.array(payload[key])
    if payload.get("raw_sample") is not None:
        payload["raw_sample"] = np.array(payload["raw_sample"])

    logger.info(
        "Baseline loaded ← %s  (%d samples, %d features, created %s)",
        path,
        payload.get("n_samples", 0),
        payload.get("n_features", 0),
        payload.get("created_at", "unknown"),
    )
    return payload


def baseline_exists(path: str | os.PathLike) -> bool:
    """Return True if a baseline file exists at `path` (checks both .json and .json.gz)."""
    path = Path(path)
    return path.exists() or Path(str(path) + ".gz").exists()
