"""Physics invariants, the PDE residual, the shock locator, and symmetry transforms.

Burgers' equation u_t + u u_x = nu u_xx on periodic [-1, 1) has the exact
properties used throughout Shock Lab's physics checks and UI (CLAUDE.md §1, §6):

  - the mean int u dx is conserved,
  - the energy E(t) = 1/2 int u^2 dx is non-increasing (dE/dt = -nu int u_x^2 dx),
  - translation equivariance: u0(x - s)  ->  u(x - s, t),
  - Galilean boost: u0 + c  ->  u(x - c t, t) + c.

All integrals use the periodic grid x_j = -1 + 2j/N (spacing dx = 2/N, domain
length 2). Spatial derivatives are spectral; temporal derivatives (for the PDE
residual) are 2nd-order finite differences in t (one-sided at the ends).
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sfft

from .solver import NU

# --- Spectral differentiation on the periodic grid ---------------------------


def _wavenumbers(n: int) -> np.ndarray:
    """Real-FFT wavenumbers k = pi * [0, 1, ..., N/2] for the grid x_j = -1 + 2j/N."""
    return np.pi * sfft.rfftfreq(n, d=1.0 / n)


def spectral_derivative(u: np.ndarray, order: int = 1, axis: int = -1) -> np.ndarray:
    """d^order u / dx^order of a periodic field, computed spectrally.

    Args:
        u: field sampled on x_j = -1 + 2j/N along ``axis``.
        order: 1 or 2.
        axis: spatial axis.
    """
    u = np.asarray(u, dtype=float)
    N = u.shape[axis]
    k = _wavenumbers(N)
    shape = [1] * u.ndim
    shape[axis] = k.shape[0]
    kk = k.reshape(shape)
    uh = sfft.rfft(u, axis=axis)
    if order == 1:
        uh = 1j * kk * uh
    elif order == 2:
        uh = -(kk**2) * uh
    else:
        raise ValueError(f"order must be 1 or 2, got {order}")
    return sfft.irfft(uh, n=N, axis=axis)


# --- Conserved / monotone quantities -----------------------------------------


def mass(u: np.ndarray, axis: int = -1) -> np.ndarray:
    """int u dx over the periodic domain (conserved in time)."""
    u = np.asarray(u, dtype=float)
    N = u.shape[axis]
    return u.sum(axis=axis) * (2.0 / N)


def energy(u: np.ndarray, axis: int = -1) -> np.ndarray:
    """E = 1/2 int u^2 dx over the periodic domain (non-increasing in time)."""
    u = np.asarray(u, dtype=float)
    N = u.shape[axis]
    return 0.5 * (u**2).sum(axis=axis) * (2.0 / N)


# --- PDE residual -------------------------------------------------------------


def pde_residual(U: np.ndarray, t: np.ndarray, nu: float = NU) -> np.ndarray:
    """Residual r = u_t + u u_x - nu u_xx of a space-time field.

    Args:
        U: field of shape (..., nt, nx) on the periodic x-grid, snapshots at ``t``.
        t: (nt,) snapshot times (may be non-uniform).
        nu: viscosity.

    Returns:
        The residual field, same shape as ``U``. Spatial derivatives are spectral;
        u_t is a 2nd-order finite difference in t (one-sided at the ends).
    """
    U = np.asarray(U, dtype=float)
    u_x = spectral_derivative(U, order=1, axis=-1)
    u_xx = spectral_derivative(U, order=2, axis=-1)
    u_t = np.gradient(U, np.asarray(t, dtype=float), axis=-2, edge_order=2)
    return u_t + U * u_x - nu * u_xx


# --- Shock locator ------------------------------------------------------------


def shock_location(u: np.ndarray, x: np.ndarray) -> np.ndarray:
    """x-position of the steepest gradient (argmax |u_x|).

    Accepts a single profile (nx,) -> scalar, or a stack (..., nx) -> (...,).
    """
    u = np.asarray(u, dtype=float)
    x = np.asarray(x, dtype=float)
    u_x = spectral_derivative(u, order=1, axis=-1)
    idx = np.argmax(np.abs(u_x), axis=-1)
    return x[idx]


# --- Symmetry transforms (exact for band-limited periodic fields) -------------


def translate(u: np.ndarray, s: float, axis: int = -1) -> np.ndarray:
    """Shift a periodic field: returns v with v(x) = u(x - s), via a spectral phase.

    Exact for band-limited signals (any ``s``, not only grid multiples).
    """
    u = np.asarray(u, dtype=float)
    N = u.shape[axis]
    k = _wavenumbers(N)
    shape = [1] * u.ndim
    shape[axis] = k.shape[0]
    phase = np.exp(-1j * k * s).reshape(shape)
    uh = sfft.rfft(u, axis=axis) * phase
    return sfft.irfft(uh, n=N, axis=axis)


def galilean_boost(U: np.ndarray, t: np.ndarray, c: float) -> np.ndarray:
    """Galilean boost of a solution: returns V(x,t) = U(x - c t, t) + c.

    Maps the solution for IC u0 to the solution for IC u0 + c (CLAUDE.md §1).

    Args:
        U: (nt, nx) space-time field, snapshots at times ``t``.
        t: (nt,) snapshot times.
        c: boost velocity.
    """
    U = np.asarray(U, dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.empty_like(U)
    for i in range(U.shape[-2]):
        out[..., i, :] = translate(U[..., i, :], c * t[i], axis=-1) + c
    return out
