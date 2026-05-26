"""Baseline estimation algorithms for electropherogram signals.

These algorithms estimate and subtract the fluorescence background so that
only true peaks remain. The original DNANet had three baseline methods
(classic, superior, enhanced) — all are preserved here.

Design pattern: **Strategy (via plain functions)**
    Each baseline algorithm is a pure function with the same signature:
    ``(signal: ndarray) -> baseline: ndarray``. This is a lightweight
    form of the Strategy pattern — no class hierarchy needed. The caller
    picks the function at configuration time.

    If you need to add a new baseline method, just write a new function
    with the same ``(ndarray) -> ndarray`` signature and register it in
    the ``BASELINE_METHODS`` dictionary at the bottom of this file.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.ndimage import percentile_filter
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """Ensure input is 2D ``(channels, samples)``. Returns ``(array_2d, was_1d)``."""
    if x.ndim == 1:
        return x[np.newaxis, :], True
    if x.ndim == 2:
        return x, False
    raise ValueError(f"Expected 1D or 2D array, got {x.ndim}D.")


def _from_2d(x: np.ndarray, was_1d: bool) -> np.ndarray:
    """Restore original dimensionality."""
    return x[0] if was_1d else x


# ---------------------------------------------------------------------------
# Core: percentile window baseline
# ---------------------------------------------------------------------------

def percentile_window_baseline(
    x: np.ndarray,
    window_size: int,
    percentile: float,
    skip_last_channel: bool = False,
) -> np.ndarray:
    """Sliding-window percentile baseline.

    Applies a percentile filter independently to each dye channel.

    Args:
        x: Signal array of shape ``(channels, samples)`` or ``(samples,)``.
        window_size: Filter window size (will be forced to odd).
        percentile: Percentile to use (e.g. 20.0).
        skip_last_channel: If True, leave the last channel (size standard)
            at zero. Use when the input includes the size-standard channel.

    Returns:
        Baseline estimate with the same shape as ``x``.
    """
    if window_size % 2 == 0:
        window_size += 1

    data, was_1d = _to_2d(x)
    baseline = np.zeros_like(data)

    n_channels = data.shape[0] - 1 if skip_last_channel else data.shape[0]
    for i in range(n_channels):
        baseline[i] = percentile_filter(
            data[i], percentile=percentile, size=window_size, mode="nearest"
        )

    return _from_2d(baseline, was_1d)


# ---------------------------------------------------------------------------
# Named baseline methods
# ---------------------------------------------------------------------------

def baseline_classic(x: np.ndarray) -> np.ndarray:
    """Classic GeneMarker baseline (550-pt window, 20th percentile)."""
    return percentile_window_baseline(x, window_size=551, percentile=20.0)


def baseline_superior(x: np.ndarray) -> np.ndarray:
    """Superior baseline (100-pt window, 20th percentile).

    More local than classic — better at following rapid baseline changes.
    This is the default used during HID file loading.
    """
    return percentile_window_baseline(x, window_size=100, percentile=20.0)


def baseline_enhanced(x: np.ndarray) -> np.ndarray:
    """Enhanced baseline for sloped/excess baselines.

    Uses piecewise (300-pt, 50% overlap) weighted linear fits where weights
    are ``1 / (1 + |d²/dx² signal|)`` computed with a Savitzky-Golay filter.
    This suppresses peaks while following the true baseline slope.
    """
    data, was_1d = _to_2d(x)
    out = np.zeros_like(data)
    window = 300
    step = window // 2  # 50% overlap

    for c in range(data.shape[0]):
        y = data[c]
        n = y.size
        base = np.zeros(n)
        counts = np.zeros(n)

        for start in range(0, n, step):
            end = min(start + window, n)
            segment = y[start:end]
            m = segment.size
            xloc = np.arange(m)

            # Savitzky-Golay window: ~31 pts, must be odd and <= segment length
            win = min(31, m if m % 2 == 1 else max(3, m - 1))

            if win >= 5:
                d2 = savgol_filter(
                    np.abs(segment), window_length=win, polyorder=2, deriv=2, delta=1.0
                )
                w = 1.0 / (1.0 + np.abs(d2))
                p = np.polyfit(xloc, segment, deg=1, w=w)
            elif m >= 2:
                p = np.polyfit(xloc, segment, deg=1)
            else:
                p = np.array([0.0, float(segment[0])])

            base[start:end] += np.polyval(p, xloc)
            counts[start:end] += 1.0

        out[c] = base / np.maximum(counts, 1.0)

    return _from_2d(out, was_1d)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def fft_lowpass_smooth(x: np.ndarray, keep_fraction: float) -> np.ndarray:
    """Gaussian low-pass filter in the Fourier domain.

    Args:
        x: Signal array (1D or 2D).
        keep_fraction: Fraction of frequencies to keep (0–1). Lower = smoother.
    """
    x = np.asarray(x)
    data, was_1d = _to_2d(x)
    n = data.shape[-1]

    # Reflect-pad to reduce edge artifacts
    pad = n // 2
    padded = np.concatenate(
        [data[..., pad:0:-1], data, data[..., -2 : -pad - 2 : -1]], axis=-1
    )
    m = padded.shape[-1]

    fft = np.fft.rfft(padded, axis=-1)
    freqs = np.fft.rfftfreq(m)
    cutoff = keep_fraction * 0.5
    kernel = np.exp(-(freqs / max(cutoff, 1e-12)) ** 2)
    smoothed = np.fft.irfft(fft * kernel, n=m, axis=-1)

    # Remove padding
    result = smoothed[..., pad : pad + n]
    return _from_2d(result, was_1d)


# ---------------------------------------------------------------------------
# Registry: maps config names to functions
# ---------------------------------------------------------------------------

BaselineMethod = Callable[[np.ndarray], np.ndarray]

BASELINE_METHODS: dict[str, BaselineMethod] = {
    "classic": baseline_classic,
    "superior": baseline_superior,
    "enhanced": baseline_enhanced,
}


def get_baseline_method(name: str) -> BaselineMethod:
    """Look up a baseline method by name.

    Args:
        name: One of ``"classic"``, ``"superior"``, ``"enhanced"``.

    Raises:
        ValueError: If the name is not recognized.
    """
    if name not in BASELINE_METHODS:
        raise ValueError(
            f"Unknown baseline method '{name}'. "
            f"Available: {list(BASELINE_METHODS.keys())}"
        )
    return BASELINE_METHODS[name]
