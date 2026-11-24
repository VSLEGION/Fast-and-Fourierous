"""Differentiable Burgers PDE residual for the PINO loss and test-time refinement.

The residual r = u_t + u u_x - nu u_xx is evaluated on a field u of shape
(..., nx, nt) sampled on the periodic grid x_j = -1 + 2j/N:

  * u_x, u_xx are computed SPECTRALLY in x (exact for the band-limited field),
  * u_t is a 2nd-order finite difference in t (one-sided at the ends),

matching CLAUDE.md §5.3. Everything is torch and autograd-friendly, so the loss
can be backpropagated into the FNO weights (training) or into a single instance
(test-time "Refine with physics"). FFTs stay in float32/complex64.
"""

from __future__ import annotations

import torch


def spectral_dx(u: torch.Tensor, order: int, axis: int = -2) -> torch.Tensor:
    """d^order u / dx^order along ``axis`` (the periodic x-axis), computed spectrally."""
    nx = u.shape[axis]
    k = torch.pi * torch.fft.rfftfreq(nx, d=1.0 / nx, device=u.device, dtype=u.dtype)
    shape = [1] * u.ndim
    shape[axis] = k.shape[0]
    kk = k.reshape(shape)
    uh = torch.fft.rfft(u, dim=axis)
    if order == 1:
        uh = (1j * kk) * uh
    elif order == 2:
        uh = (-(kk**2)) * uh
    else:
        raise ValueError(f"order must be 1 or 2, got {order}")
    return torch.fft.irfft(uh, n=nx, dim=axis)


def fd_dt(u: torch.Tensor, dt: float) -> torch.Tensor:
    """du/dt along the last axis: 2nd-order central interior, 2nd-order one-sided ends."""
    left = (-3 * u[..., :1] + 4 * u[..., 1:2] - u[..., 2:3]) / (2 * dt)
    center = (u[..., 2:] - u[..., :-2]) / (2 * dt)
    right = (3 * u[..., -1:] - 4 * u[..., -2:-1] + u[..., -3:-2]) / (2 * dt)
    return torch.cat([left, center, right], dim=-1)


def pde_residual(u: torch.Tensor, dt: float, nu: float) -> torch.Tensor:
    """Burgers residual r = u_t + u u_x - nu u_xx for u of shape (..., nx, nt)."""
    u_x = spectral_dx(u, order=1, axis=-2)
    u_xx = spectral_dx(u, order=2, axis=-2)
    u_t = fd_dt(u, dt)
    return u_t + u * u_x - nu * u_xx
