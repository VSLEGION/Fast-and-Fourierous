"""Exact Cole-Hopf solution of Burgers' equation for u0 = -sin(pi x).

The Cole-Hopf transform u = -2 nu phi_x / phi turns Burgers' equation
u_t + u u_x = nu u_xx into the linear heat equation phi_t = nu phi_xx, whose
solution is a Gaussian convolution. For u0 = -sin(pi x) this gives

    u(x,t) = - [ int sin(pi(x-eta)) f(x-eta) exp(-eta^2 / 4 nu t) d eta ]
               / [ int          f(x-eta) exp(-eta^2 / 4 nu t) d eta ],
    f(y) = exp(-cos(pi y) / (2 pi nu)).

With nu = 0.01/pi, f swings between exp(-50) and exp(+50), so we work with
log-weights and subtract the row max before exponentiating (log-sum-exp).

Ported VERBATIM from 01_physics_and_data.ipynb (function ``burgers_exact_sine``).
This is the ground truth the solver is checked against; do not change its
numerics without re-running tests/test_physics.py (CLAUDE.md §0, §1).
"""

from __future__ import annotations

import numpy as np

from .solver import NU


def burgers_exact_sine(
    x: np.ndarray,
    t: float,
    nu: float = NU,
    n_eta: int = 40001,
    L: float = 3.0,
    chunk: int = 128,
) -> np.ndarray:
    """Exact Cole-Hopf solution for u0 = -sin(pi x) at a single time ``t``.

    Args:
        x: array of spatial points (any shape 1-D array).
        t: time (``t == 0`` returns the initial condition exactly).
        nu: viscosity.
        n_eta: number of quadrature points for the convolution integral.
        L: half-width of the (truncated) convolution integration range.
        chunk: number of x-points processed at once (keeps memory small).

    Returns:
        u(x, t) as an array with the same shape as ``x``.
    """
    # Exact Cole-Hopf solution for u0 = -sin(pi x). Works on any array of x at a single time t.
    x = np.asarray(x, dtype=float)
    if t == 0:
        return -np.sin(np.pi * x)
    eta = np.linspace(-L, L, n_eta)
    out = np.empty_like(x)
    for i in range(0, len(x), chunk):  # chunk over x to keep memory small
        X = x[i : i + chunk, None] - eta[None, :]
        logw = -np.cos(np.pi * X) / (2 * np.pi * nu) - eta[None, :] ** 2 / (4 * nu * t)
        logw -= logw.max(axis=1, keepdims=True)  # log-sum-exp trick
        w = np.exp(logw)
        out[i : i + chunk] = -(np.sin(np.pi * X) * w).sum(1) / w.sum(1)
    return out
