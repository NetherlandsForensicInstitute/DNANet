import numpy as np
from scipy.ndimage import percentile_filter
from scipy.signal import savgol_filter





def _to_2d(x: np.ndarray):
    """Return (X2d, was_1d) where X2d has shape (C, N)."""
    if x.ndim == 1:
        return x[None, :], True
    if x.ndim == 2:
        return x, False
    raise ValueError("Input must be shape (N,) or (C, N).")

def _from_2d(x2d: np.ndarray, was_1d: bool):
    """Restore original shape"""
    return x2d[0] if was_1d else x2d

def percentile_window_baseline(x: np.ndarray, window_size: int, percentile: float) -> np.ndarray:
    """
    Baseline using a sliding window percentile filter. No baseline is calculated for the internal standard.
    Returns baseline with same shape as input.
    """
    if window_size % 2 == 0:
        window_size += 1  # make odd
    data, was_1d = _to_2d(x)
    out = np.zeros_like(data, dtype=data.dtype)

    # do not do baseline subtraction for the internal standard
    num_dyes = min(data.shape[0], 5)
    # num_dyes = data.shape[0]
    for i in range(num_dyes):
        out[i] = percentile_filter(data[i], percentile=percentile, size=window_size, mode="nearest")

    return _from_2d(out, was_1d)


# -----------
# Genemarker baselines
# -----------

def baseline_classic(x: np.ndarray) -> np.ndarray:
    """
    Classic baseline (local ~550-pt window; 20th percentile).
    Returns baseline with same shape as input.
    """
    return percentile_window_baseline(x, window_size=551, percentile=20.0)


def baseline_superior(x: np.ndarray) -> np.ndarray:
    """
    Superior baseline (more local ~100-pt window; 20th percentile).
    Returns baseline with same shape as input.
    Baseline is not subtracted from the internal standard
    """
    return percentile_window_baseline(x, window_size=100, percentile=20.0)


def baseline_enhanced(x: np.ndarray) -> np.ndarray:
    """
    Enhanced baseline for sloped/excess baselines.
    Piecewise (300-pt) weighted linear fits where weights are
    1 / (1 + |second derivative of |signal||), computed with a ~31-pt Savitzky–Golay.
    Returns baseline with same shape as input.
    """
    X, was_1d = _to_2d(x)
    out = np.zeros_like(X)

    window = 300
    step = window // 2  # 50% overlap

    for c in range(X.shape[0]):
        y = X[c]
        n = y.size
        base = np.zeros(n)
        counts = np.zeros(n)

        for start in range(0, n, step):
            end = min(start + window, n)
            segment = y[start:end]
            m = segment.size

            # Choose a Savitzky–Golay window near 30 pts, valid and odd.
            win = 31
            if m < win:
                win = m if m % 2 == 1 else max(3, m - 1)

            xloc = np.arange(m)

            if win >= 5:  # enough points for a 2nd-derivative SG filter
                d2 = savgol_filter(np.abs(segment), window_length=win, polyorder=2, deriv=2, delta=1.0)
                w = 1.0 / (1.0 + np.abs(d2))
                p = np.polyfit(xloc, segment, deg=1, w=w)
            else:
                # Tiny edge segments: simple unweighted linear fit (or constant if m<2)
                if m >= 2:
                    p = np.polyfit(xloc, segment, deg=1)
                else:
                    p = np.array([0.0, float(segment[0])])  # constant

            seg_base = np.polyval(p, xloc)
            base[start:end] += seg_base
            counts[start:end] += 1.0

        out[c] = base / np.maximum(counts, 1.0)

    return _from_2d(out, was_1d)
