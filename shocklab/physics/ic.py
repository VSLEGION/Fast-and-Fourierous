"""Initial conditions: Gaussian random field (for the FNO dataset) and presets.

The GRF sampler ``random_ic`` is ported VERBATIM from 01_physics_and_data.ipynb.
It draws u0 ~ N(0, 625 (-Delta + 25 I)^{-2}) on periodic [-1, 1): a random
Fourier series whose mode-n amplitude has std sqrt(625 / ((pi n)^2 + 25)^2),
then rescales each sample so its peak |u0| lands in ``amp`` (default [0.5, 1.0]).
This is the FNO paper's distribution and keeps the classic -sin(pi x) case
in-distribution (CLAUDE.md §3).

The presets below are the "values we know the answer to" used as UI initial
conditions (CLAUDE.md §6.3). They are pure functions of the grid x.
"""

from __future__ import annotations

import numpy as np

from .solver import x_grid


def random_ic(
    B: int, N: int, rng: np.random.Generator, amp: tuple[float, float] = (0.5, 1.0)
) -> np.ndarray:
    """Gaussian-random-field initial conditions on periodic [-1, 1).

    Args:
        B: batch size (number of samples).
        N: grid size.
        rng: a ``numpy`` Generator (seed it for reproducibility).
        amp: each sample is rescaled so peak |u0| ~ U[amp[0], amp[1]].

    Returns:
        (B, N) array of initial conditions with zero mean (before rescaling).
    """
    # Gaussian random field initial conditions on periodic [-1, 1), rescaled to peak |u0| in amp.
    x = -1 + 2 * np.arange(N) / N
    n = np.arange(1, N // 2)
    std = np.sqrt(625.0 / ((np.pi * n) ** 2 + 25.0) ** 2)
    a = rng.standard_normal((B, len(n))) * std
    b = rng.standard_normal((B, len(n))) * std
    u0 = a @ np.cos(np.pi * np.outer(n, x)) + b @ np.sin(np.pi * np.outer(n, x))
    u0 *= rng.uniform(*amp, size=(B, 1)) / np.abs(u0).max(axis=1, keepdims=True)
    return u0


# --- Presets (CLAUDE.md §6.3) -------------------------------------------------
# Each returns u0 sampled on the given grid x (use x_grid(N) for the default grid).


def classic(x: np.ndarray) -> np.ndarray:
    """The Raissi benchmark u0 = -sin(pi x) (EXACT truth is available)."""
    return -np.sin(np.pi * x)


def shifted_classic(x: np.ndarray, s: float = 0.0) -> np.ndarray:
    """Translated classic u0 = -sin(pi (x - s)) (SYMMETRY truth: shift the exact solution)."""
    return -np.sin(np.pi * (x - s))


def boosted_classic(x: np.ndarray, c: float = 0.0) -> np.ndarray:
    """Galilean-boosted classic u0 = -sin(pi x) + c (SYMMETRY truth; OOD: nonzero mean)."""
    return -np.sin(np.pi * x) + c


def linear_limit(x: np.ndarray, eps: float = 0.05) -> np.ndarray:
    """Small-amplitude u0 = eps sin(pi x); the heat-equation limit (ANALYTIC truth; OOD amplitude)."""
    return eps * np.sin(np.pi * x)


def two_wave(
    x: np.ndarray,
    A: float = 0.6,
    B: float = 0.4,
    phi1: float = 0.0,
    phi2: float = 0.0,
) -> np.ndarray:
    """Two-mode u0 = A sin(pi x + phi1) + B sin(2 pi x + phi2) (SOLVER truth)."""
    return A * np.sin(np.pi * x + phi1) + B * np.sin(2 * np.pi * x + phi2)


def gaussian_pulse(
    x: np.ndarray,
    x0: float = 0.0,
    width: float = 0.15,
    amp: float = 1.0,
    zero_mean: bool = True,
) -> np.ndarray:
    """Zero-mean periodic Gaussian pulse centered at x0 (SOLVER truth).

    Summed over periodic images so it is exactly periodic on [-1, 1).
    """
    u = np.zeros_like(x, dtype=float)
    for m in (-1, 0, 1):  # periodic images; width is small so three suffice
        u += np.exp(-(((x - x0 - 2 * m) / width) ** 2))
    u *= amp
    if zero_mean:
        u = u - u.mean()
    return u


def n_wave(x: np.ndarray, amp: float = 1.0, n_modes: int = 16) -> np.ndarray:
    """Smoothed (band-limited) sawtooth / N-wave, zero mean (SOLVER truth).

    A sawtooth has Fourier series sum_n (2/(pi n)) (-1)^{n+1} sin(n pi x); we
    truncate it to ``n_modes`` so it stays resolved (modes <= 32, CLAUDE.md §6.3).
    """
    u = np.zeros_like(x, dtype=float)
    for k in range(1, n_modes + 1):
        u += ((-1) ** (k + 1)) * np.sin(k * np.pi * x) / k
    u *= amp / np.abs(u).max()
    return u


def project_periodic_lowpass(
    y: np.ndarray,
    max_mode: int = 32,
    zero_mean: bool = False,
    clamp: float = 1.2,
) -> np.ndarray:
    """Project a sampled curve onto a periodic, low-pass IC (for hand-drawn input).

    ``y`` must be sampled on the periodic grid x_grid(len(y)). Fourier modes above
    ``max_mode`` are zeroed, the mean is optionally removed, and the amplitude is
    clamped to [-clamp, clamp] (CLAUDE.md §6.3 item 9).
    """
    from scipy import fft as sfft

    y = np.asarray(y, dtype=float)
    N = y.shape[-1]
    yh = sfft.rfft(y)
    yh[max_mode + 1 :] = 0.0
    if zero_mean:
        yh[0] = 0.0
    out = sfft.irfft(yh, n=N)
    return np.clip(out, -clamp, clamp)


def preset_grid(n: int) -> np.ndarray:
    """Convenience: the default periodic grid x_j = -1 + 2j/N."""
    return x_grid(n)
