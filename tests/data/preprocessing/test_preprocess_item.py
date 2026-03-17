import pytest
import torch
import numpy as np
from DNAnet.data.preprocessing.preprocess_item import (
    preprocess_profile_torch,
    _scale_data_torch,
    inverse_scale_data,
    RFU_MAX_VALUE
)

def test_scale_data_torch_log_and_max():
    data = torch.tensor([0.0, 100.0, 32768.0])
    max_rfu = 32768
    # Case 1: log_scale=True, max_rfu_value set
    # x = log1p(clamp(x, 0, max_rfu)) / log1p(max_rfu)
    scaled = _scale_data_torch(data, log_scale=True, max_rfu_value=max_rfu)

    expected = torch.log1p(data) / np.log1p(max_rfu)
    torch.testing.assert_close(scaled, expected)
    assert scaled.max() <= 1.0
    assert scaled.min() >= 0.0

def test_scale_data_torch_log_only():
    data = torch.tensor([0.0, 100.0, 32768.0])
    # Case 2: log_scale=True, max_rfu_value=None
    # x = log1p(clamp(x, 0, inf))
    scaled = _scale_data_torch(data, log_scale=True, max_rfu_value=None)

    expected = torch.log1p(data)
    torch.testing.assert_close(scaled, expected)

def test_scale_data_torch_max_only():
    data = torch.tensor([0.0, 16384.0, 32768.0, 40000.0])
    max_rfu = 32768
    # Case 3: log_scale=False, max_rfu_value set
    # x = clamp(x / max_rfu, 0, 1)
    scaled = _scale_data_torch(data, log_scale=False, max_rfu_value=max_rfu)

    expected = torch.clamp(data / max_rfu, 0.0, 1.0)
    torch.testing.assert_close(scaled, expected)

def test_scale_data_torch_none():
    data = torch.tensor([0.0, 100.0, 32768.0])
    # Case 4: log_scale=False, max_rfu_value=None
    # Returns original data (but ensures float32 if not already)
    scaled = _scale_data_torch(data, log_scale=False, max_rfu_value=None)
    torch.testing.assert_close(scaled, data)

def test_inverse_scale_data_log_and_max():
    max_rfu = 32768
    original_data = torch.tensor([0.0, 100.0, 1000.0, 32768.0])
    scaled = _scale_data_torch(original_data, log_scale=True, max_rfu_value=max_rfu)

    inversed = inverse_scale_data(scaled, log_scale=True, max_rfu_scale_value=max_rfu)
    torch.testing.assert_close(inversed, original_data)

def test_inverse_scale_data_log_only():
    original_data = torch.tensor([0.0, 100.0, 1000.0, 32768.0])
    scaled = _scale_data_torch(original_data, log_scale=True, max_rfu_value=None)

    inversed = inverse_scale_data(scaled, log_scale=True, max_rfu_scale_value=None)
    torch.testing.assert_close(inversed, original_data)

def test_inverse_scale_data_max_only():
    max_rfu = 32768
    original_data = torch.tensor([0.0, 100.0, 1000.0, 32768.0])
    scaled = _scale_data_torch(original_data, log_scale=False, max_rfu_value=max_rfu)

    inversed = inverse_scale_data(scaled, log_scale=False, max_rfu_scale_value=max_rfu)
    torch.testing.assert_close(inversed, original_data)


def test_preprocess_profile_torch_with_real_image(hid_image):
    # hid_image is a fixture from conftest.py
    # Test with default parameters (log_scale=True, max_rfu_scale_value=32768, num_dyes_included=6)
    preprocessed = preprocess_profile_torch(hid_image)

    assert torch.is_tensor(preprocessed)
    # Check shape: (dyes, time, 1) or (dyes, time)?
    # HIDImage.data returns (6, time, 1) or (6, time)?
    # Let's check HIDImage data shape.
    raw_data = torch.tensor(hid_image.data, dtype=torch.float32)
    assert preprocessed.shape[0] == 6 # num_dyes_included
    assert preprocessed.shape[1] == raw_data.shape[1]

    # Verify scaling
    max_rfu = RFU_MAX_VALUE
    expected = torch.log1p(torch.clamp(raw_data[:6], 0.0, float(max_rfu))) / np.log1p(max_rfu)
    torch.testing.assert_close(preprocessed, expected)


