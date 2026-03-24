"""
mlwatch.frameworks.pytorch
===========================
PyTorch ``nn.Module`` adapter using forward hooks.

Registers a forward hook that fires automatically on every forward pass —
no changes to model architecture required.  Compatible with:
- ``torch.nn.Module``
- TorchScript (``torch.jit.ScriptModule``)
- ONNX-exported PyTorch models
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from mlwatch.frameworks.base import FrameworkAdapter

logger = logging.getLogger(__name__)


class PyTorchAdapter(FrameworkAdapter):
    """
    Adapter for PyTorch ``nn.Module`` models.

    Parameters
    ----------
    model: A ``torch.nn.Module`` instance.
    """

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        self._last_output: Optional[Any] = None
        self._hook_handle = None
        self._register_hook()

    def _register_hook(self) -> None:
        """Register a PyTorch forward hook to capture outputs."""
        try:
            def _hook_fn(module, input, output):
                self._last_output = output

            self._hook_handle = self.model.register_forward_hook(_hook_fn)
            logger.debug("PyTorchAdapter: forward hook registered.")
        except Exception as exc:
            logger.warning(
                "PyTorchAdapter: could not register forward hook — %s. "
                "Falling back to direct call.",
                exc,
            )

    def predict(self, inputs: Any) -> Any:
        """
        Run a forward pass.

        Inputs can be:
        - ``torch.Tensor``
        - numpy array (auto-converted via ``torch.from_numpy``)
        - Any other type passed directly to the model
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for framework='pytorch'. "
                "Install with: pip install torch"
            ) from exc

        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()

        with torch.no_grad():
            output = self.model(inputs)

        return output

    def close(self) -> None:
        """Remove the registered forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            logger.debug("PyTorchAdapter: forward hook removed.")
