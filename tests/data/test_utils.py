import numpy as np
import pytest
from numpy.testing import assert_array_equal

from DNAnet.data.parsing import get_peak_data
from DNAnet.data.utils import (
    basepair_interpolator,
    extract_ss_peaks,
    find_peak_boundary,
    find_peak_idx_near_or_in_range,
    find_peak_near_idx,
    find_peaks_above_threshold,
)


@pytest.mark.parametrize(
    "array, result",
    [
        (np.array([1000, 1001, 1010, 1003, 1002]), np.array([2])),
        (np.array([100, 101, 110, 103, 102]), np.array([])),
        (np.array([1000, 1003, 1003, 999]), np.array([1, 2])),
        (np.array([1000, 1001, 800, 800, 799]), np.array([1, 3])),
    ],
)
def test_find_peaks_above_threshold(array, result):
    assert np.array_equal(find_peaks_above_threshold(array, 200), result)


@pytest.mark.parametrize(
    "array, idx, result",
    [
        (np.array([110, 90, 101, 103, 110, 103, 90, 101]), 4, (2, 5)),
        (np.array([90, 90, 90, 90, 110, 90, 90, 90]), 4, (4, 4)),
        (np.array([110, 100, 101, 103, 110, 103, 100, 101]), 4, (0, 7)),
    ],
)
def test_find_peak_boundary(array, idx, result):
    assert find_peak_boundary(array, idx, threshold=100) == result


@pytest.mark.parametrize(
    "array, idx, result",
    [
        (np.array([100, 101, 110, 103, 102, 101, 120, 101]), 1, np.array([2])),
        (np.array([100, 101, 110, 103, 102, 101, 120, 101]), 7, np.array([6])),
        (np.array([100, 101, 110, 103, 102, 101, 120, 101]), 4, np.array([2])),
    ],
)
def test_find_peak_near_idx(array, idx, result):
    assert find_peak_near_idx(array, idx) == result


@pytest.mark.parametrize(
    "array, range, result",
    [
        (np.array([100, 101, 110, 103, 102, 101, 120, 101]), np.array([1, 2, 3]), np.array([2])),
        (np.array([100, 101, 110, 103, 102, 101, 120, 101]), np.array([0, 1]), np.array([2])),
        (np.array([100, 101, 110, 103, 202, 101, 120, 101]), np.array([0, 1, 2, 3, 4, 5, 6]),
         np.array([4])),
        (np.array([80, 81, 90, 83, 82, 81, 90, 81]), np.array([1, 2, 3]), np.array([])),
    ],
)
def test_find_peak_idx_near_or_in_range(array, range, result):
    out = find_peak_idx_near_or_in_range(array, range, threshold=100)
    assert np.array_equal(out, result)


def test_extract_peaks():
    profile = get_peak_data(f'{pytest.RESOURCES_DIR}/profiles/RD/1A2_A01_01.hid')[-1]
    result = np.array([2796, 2813, 2901, 2917, 2961, 2989, 3038, 3095, 3144,
                       3387, 3645, 3702, 3881, 4120, 4348, 4578, 4806, 5033,
                       5257, 5531, 5798, 6059, 6323, 6579, 6827, 7077, 7318,
                       7560, 7788, 8018, 8238])
    assert np.array_equal(extract_ss_peaks(profile), result)
    # adapt the profile so the final peak is below the 180rfu threshold, in
    # that case, this final peak should be caught in the if-statement
    profile[8199:] = 0
    profile[result[-1]] = 150
    assert np.array_equal(extract_ss_peaks(profile), result)


def test_interpolate_basepairs_integers():
    indices = [0, 3, 6]
    original_x_values = [1, 7, 13]
    length = 6
    interp = basepair_interpolator(indices, original_x_values)
    # returns with linear interpolation
    assert_array_equal(interp(np.arange(length)), np.array([1., 3., 5., 7., 9., 11.]))

    # Extrapolation
    indices = [2, 3]
    original_x_values = [12, 13]
    length = 3
    interp = basepair_interpolator(indices, original_x_values, extrapolate=True)
    assert_array_equal(interp(np.arange(length)), np.array([10., 11., 12.]))
    assert_array_equal(interp(0), np.array([10.]))
    assert_array_equal(interp(1), np.array([11.]))
    assert_array_equal(interp(4), np.array([14.]))
    assert_array_equal(interp(5), np.array([15.]))


def test_interpolate_basepairs_float():
    indices = [97.82, 102.13]
    original_x_values = [97.64, 101.95]
    interp = basepair_interpolator(indices, original_x_values)
    assert_array_equal(interp(101.22), np.array([101.04]))

    # Extrapolation
    interp = basepair_interpolator(indices, original_x_values, extrapolate=True)
    assert_array_equal(interp(93.47), np.array([93.29]))