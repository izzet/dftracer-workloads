from typing import Any
import logging

import torch

from utils.context import noop_context

log = logging.getLogger(__name__)

# ensure compatibility with Pytorch Lightning
def get_precision_dtype(precision: int | str | None):
    if precision is None:
        log.debug("Using default 32-bit precision.")
        return torch.float32

    precision = str(precision).lower()
    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16

    elif precision in ("16", "fp16"):
        return torch.float16

    elif precision in ("32"):
        return torch.float32

    elif precision in ("64"):
        log.warning("Using 64-bit precision, it potentially will slow down your pipeline")
        return torch.float64
    else:
        raise ValueError(f"Invalid precision={precision}")


# https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/fabric/plugins/precision/utils.py
class DtypeContextManager:
    """A context manager to change the default tensor type when tensors get created.

    See: :func:`torch.set_default_dtype`

    """

    def __init__(self, dtype: torch.dtype) -> None:
        self._previous_dtype: torch.dtype = torch.get_default_dtype()
        self._new_dtype = dtype

    def __enter__(self) -> None:
        torch.set_default_dtype(self._new_dtype)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch.set_default_dtype(self._previous_dtype)


def setup_precision_context(dtype: torch.dtype, device_type: str, mixed: bool = True, **kwargs):
    if dtype is None:
        return noop_context()
    if "cuda" in device_type:
        if mixed:
            return torch.autocast(device_type=device_type, dtype=dtype, **kwargs)
        return DtypeContextManager(dtype=dtype)

    enabled = kwargs["enabled"] if "enabled" in kwargs else True

    if enabled:
        if mixed:
            log.warning(
                "Mixed precision is requested but cuda device is not available, falling back to use `DtypeContextManager`."
            )
        return DtypeContextManager(dtype=dtype)

    log.info("Precision context is disabled.")
    return noop_context()


__all__ = ["setup_precision_context", "get_precision_dtype", "DtypeContextManager"]
