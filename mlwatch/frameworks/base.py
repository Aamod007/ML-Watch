"""
mlwatch.frameworks.base
========================
Abstract framework adapter interface.

Each framework adapter is responsible for:
1. Calling the underlying model's forward/predict method
2. Optionally registering framework-native hooks (PyTorch forward hooks,
   Keras callbacks, etc.)
3. Converting inputs/outputs to numpy arrays for monitoring
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np


class FrameworkAdapter(abc.ABC):
    """
    Abstract base class for ML framework adapters.

    Subclasses wrap a specific model type and expose a uniform
    ``predict(inputs) → Any`` interface.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    @abc.abstractmethod
    def predict(self, inputs: Any) -> Any:
        """Run inference and return predictions."""

    def to_numpy(self, obj: Any) -> np.ndarray:
        """Best-effort conversion of model output to numpy."""
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
        return np.asarray(obj)
