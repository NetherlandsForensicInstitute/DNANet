"""RFU signal scaling and normalization utilities.

These functions scale electropherogram data for neural network input
and invert the scaling for evaluation in original RFU space.

Scaling modes:
    - **log + clamp**: ``log1p(clamp(x, 0, max_rfu)) / log1p(max_rfu)``
    - **log only**: ``log1p(clamp(x, 0))``
    - **linear**: ``clamp(x / max_rfu, 0, 1)``
    - **none**: identity

The default ``RFU_MAX_VALUE = 32768`` (2^15) is the maximum value for
16-bit unsigned RFU measurements in HID files.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


RFU_MAX_VALUE: int = 32768


def scale_rfu_torch(
    data: Tensor,
    log_scale: bool = True,
    max_rfu: int | None = RFU_MAX_VALUE,
) -> Tensor:
    """Scale RFU data for model input.

    Args:
        data: Raw RFU tensor of any shape.
        log_scale: Apply log1p scaling.
        max_rfu: Maximum RFU value for normalization. If provided with
            ``log_scale=True``, output is in [0, 1].

    Returns:
        Scaled tensor (always float32).
    """
    x = data.float()

    if log_scale and max_rfu is not None:
        m = float(max_rfu)
        x = torch.log1p(x.clamp(0.0, m)) / torch.log1p(torch.tensor(m))
        x = x.clamp(0.0, 1.0)
    elif log_scale:
        x = torch.log1p(x.clamp(0.0))
    elif max_rfu is not None:
        x = (x / float(max_rfu)).clamp(0.0, 1.0)

    return x


def inverse_scale_rfu_torch(
    data: Tensor,
    log_scale: bool = True,
    max_rfu: int | None = RFU_MAX_VALUE,
) -> Tensor:
    """Invert :func:`scale_rfu_torch` back to original RFU space.

    Args:
        data: Scaled tensor.
        log_scale: Whether log1p was used during scaling.
        max_rfu: Max RFU value used during scaling.

    Returns:
        Tensor in original RFU space, clamped ≥ 0.
    """
    x = data.float()

    # Handle trailing singleton dim from older data formats
    had_trailing = x.dim() >= 3 and x.shape[-1] == 1
    if had_trailing:
        x = x.squeeze(-1)

    if log_scale and max_rfu is not None:
        m = torch.log1p(torch.tensor(float(max_rfu), device=x.device, dtype=x.dtype))
        x = torch.expm1(x * m)
    elif log_scale:
        x = torch.expm1(x)
    elif max_rfu is not None:
        x = x * float(max_rfu)

    x = x.clamp(min=0.0)

    if had_trailing:
        x = x.unsqueeze(-1)

    return x


def scale_rfu_numpy(
    data: np.ndarray,
    log_scale: bool = True,
    max_rfu: int | None = RFU_MAX_VALUE,
) -> np.ndarray:
    """NumPy equivalent of :func:`scale_rfu_torch`.

    Args:
        data: Raw RFU array of any shape.
        log_scale: Apply log1p scaling.
        max_rfu: Maximum RFU for normalization.

    Returns:
        Scaled float64 array.
    """
    x = data.astype(np.float64)

    if log_scale and max_rfu is not None:
        x = np.log1p(np.clip(x, 0.0, max_rfu)) / np.log1p(max_rfu)
        x = np.clip(x, 0.0, 1.0)
    elif log_scale:
        x = np.log1p(np.clip(x, 0.0, None))
    elif max_rfu is not None:
        x = np.clip(x / float(max_rfu), 0.0, 1.0)

    return x
