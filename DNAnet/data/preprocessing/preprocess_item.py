from typing import Optional

import torch

from DNAnet.data.data_models.hid_image import HIDImage

RFU_MAX_VALUE = 32768 # 2^15, the maximum value for a 16-bit unsigned integer, which is the data type used for RFU values in HIDImages

def preprocess_profile_torch(img: HIDImage,
                             log_scale: bool = True,
                             max_rfu_scale_value: Optional[int] = RFU_MAX_VALUE,
                             num_dyes_included: int = 6) -> torch.Tensor:
    """
    Preprocess the HIDImage data using PyTorch tensors by applying the following (optional) steps:
        1. Select the first `num_dyes_included` dyes.
        2. Apply log1p scaling and divide by `max_rfu_scale_value`.
    """
    data = torch.tensor(img.data, dtype=torch.float32)
    if 0 < num_dyes_included <= 6:
        if num_dyes_included == 6 and data.shape[0] == 5:
            raise ValueError("Expected 6 dyes, but the data only has 5 dyes. Please load the data with include_size_standard=True")
        # Select the first `num_dyes_included` dyes
        data = data[:num_dyes_included]

    data = _scale_data_torch(data, log_scale, max_rfu_scale_value)

    return data


def _scale_data_torch(
        data: torch.Tensor,
        log_scale: bool,
        max_rfu_value: Optional[int],
) -> torch.Tensor:
    """
    Scales the data using log1p scaling and/or max RFU value normalization.
    Depending on the parameters provided, the function applies the following transformations:

    1. If both `log_scale` is True and `max_rfu_value`
         is provided, the data is clipped to the range [0, max_rfu_value],
         log1p scaled, and then normalized by dividing by log1p(max_rfu_value).
    2. If only `log_scale` is True, the data is clipped to the range [0, ∞)
            and log1p scaled.
    3. If only `max_rfu_value` is provided, the data is normalized by dividing by `max_rfu_value`
            and clipped to the range [0, 1].

    :param data: The input data tensor to be scaled.
    :param log_scale: Whether to apply log1p scaling to the data.
    :param max_rfu_value: An optional maximum RFU value for normalization. If provided
        and `log_scale` is True, the data will be normalized by log1p(max_rfu_value) after log scaling.
    :return: The scaled data tensor.

    """
    if not torch.is_tensor(data):
        raise TypeError("data must be a torch.Tensor")

    # Ensure floating point for log/div operations; keep device
    orig_dtype = data.dtype
    x = data if torch.is_floating_point(data) else data.to(torch.float32)

    if log_scale and max_rfu_value is not None:
        m = torch.tensor(float(max_rfu_value), device=x.device, dtype=x.dtype)
        x = torch.clamp(x, 0.0, m)
        x = torch.log1p(x) / torch.log1p(m)
        x = torch.clamp(x, 0.0, 1.0)
    elif log_scale:
        x = torch.clamp(x, 0.0)
        x = torch.log1p(x)
    elif max_rfu_value is not None:
        m = torch.tensor(float(max_rfu_value), device=x.device, dtype=x.dtype)
        x = x / m
        x = torch.clamp(x, 0.0, 1.0)

    return x.to(orig_dtype) if torch.is_floating_point(data) else x
