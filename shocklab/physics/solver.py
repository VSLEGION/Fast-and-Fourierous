"""ETDRK4 pseudo-spectral solver for viscous Burgers' equation.

Governing equation on a periodic domain x in [-1, 1), t in [0, 1]:

    u_t + u u_x = nu u_xx,   nu = 0.01 / pi

This is the trusted reference ("truth") for any initial condition. It is ported
VERBATIM from 01_physics_and_data.ipynb (function ``burgers_solve``). The
numerics are verified against the exact Cole-Hopf solution in
tests/test_physics.py and MUST NOT be changed without re-running those tests and
telling the user (CLAUDE.md §0, §1):

  - ETDRK4 time stepping (Cox-Matthews 2002) with Kassam-Trefethen (2005)
    contour-integral coefficients, M = 32 points, to avoid 0/0 at small k.
  - Grid convention x_j = -1 + 2j/N, j = 0..N-1 (left-closed, no duplicate
    endpoint); wavenumbers k = pi * n.
  - The viscous term is treated exactly in Fourier space; the nonlinear term
    u u_x = (u^2/2)_x is computed on the grid with the 2/3 dealiasing rule.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sfft

# viscosity (Raissi, Perdikaris & Karniadakis 2019 benchmark value) and time horizon
NU: float = 0.01 / np.pi
T_END: float = 1.0


def x_grid(n: int) -> np.ndarray:
    """Periodic grid x_j = -1 + 2j/N, j = 0..N-1 (left-closed, no duplicate endpoint)."""
    return -1 + 2 * np.arange(n) / n


def burgers_solve(
    u0: np.ndarray,
    T: float = T_END,
    n_out: int = 51,
    dt: float = 4e-4,
    nu: float = NU,
) -> np.ndarray:
    """ETDRK4 pseudo-spectral solver for u_t + u u_x = nu u_xx on periodic [-1, 1).

    Args:
        u0: (B, N) batch of initial conditions on x_j = -1 + 2j/N (a 1-D array
            is promoted to a single-sample batch).
        T: final time.
        n_out: number of stored snapshots, including t = 0.
        dt: time step. ``T / dt`` must be an integer multiple of ``n_out - 1``.
        nu: viscosity.

    Returns:
        U: (B, n_out, N) snapshots at t = linspace(0, T, n_out).
    """
    # ETDRK4 pseudo-spectral solver for u_t + u u_x = nu u_xx on periodic [-1, 1).
    # u0: (B, N) batch of initial conditions on x_j = -1 + 2j/N.
    # Returns U: (B, n_out, N), snapshots at t = linspace(0, T, n_out).
    u0 = np.atleast_2d(u0).astype(float)
    B, N = u0.shape
    k = np.pi * sfft.rfftfreq(N, d=1.0 / N)  # wavenumbers pi*n
    Lk = -nu * k**2  # linear (viscous) operator, diagonal in Fourier space
    E, E2 = np.exp(dt * Lk), np.exp(dt * Lk / 2)

    # ETDRK4 coefficients via contour integrals (Kassam & Trefethen 2005), avoids 0/0 at small k
    M = 32
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * Lk[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))

    dealias = (np.arange(len(k)) < (2 * (N // 2)) // 3).astype(float)  # 2/3 rule
    g = -0.5j * k * dealias

    def nonlinear(vh):  # Fourier transform of -(u^2/2)_x
        u = sfft.irfft(vh, n=N, workers=-1)
        return g * sfft.rfft(u * u, workers=-1)

    steps = int(round(T / dt))
    every = steps // (n_out - 1)
    assert every * (n_out - 1) == steps, "choose dt so that T/dt is a multiple of n_out-1"
    vh = sfft.rfft(u0, workers=-1)
    out = [u0.copy()]
    for s in range(1, steps + 1):
        Nv = nonlinear(vh)
        a = E2 * vh + Q * Nv
        Na = nonlinear(a)
        b = E2 * vh + Q * Na
        Nb = nonlinear(b)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = nonlinear(c)
        vh = E * vh + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if s % every == 0:
            out.append(sfft.irfft(vh, n=N, workers=-1))
    return np.stack(out, axis=1)
