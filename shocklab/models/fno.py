"""Space-time Fourier Neural Operator (FNO-2D) for Burgers' equation (CLAUDE.md §5.2).

The FNO learns the SOLUTION OPERATOR from data: given an initial condition u0(x),
it maps to the full space-time field u(x, t) in one forward pass.

Input channels on the (x, t) grid:
  * u0(x), broadcast over t,
  * t, broadcast over x.
There is DELIBERATELY no x-coordinate channel. Every operation is then either a
periodic convolution (the spectral layer) or a pointwise map, so the model is
EXACTLY translation-equivariant in x -- shifting u0 shifts the output. The UI
demonstrates this; the equivariance is pinned by tests/test_models.py.

Architecture: lift (1x1 conv, width 64) -> 4 Fourier layers
v <- GELU(W v + F^-1(R . F v)) with modes (k_x=32, k_t=16) -> project (64->128->1).

x is periodic (full FFT, +/- modes kept); t is NOT -- the t-axis is zero-padded by
~25% before the FFT and cropped after, to avoid wraparound. The spectral weights
are indexed by mode number, independent of grid size, which is what makes the
operator resolution-invariant (zero-shot super-resolution). FFTs stay in
float32/complex64 and are never autocast to half precision.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv2d(nn.Module):
    """2-D spectral convolution: FFT, keep low modes, multiply by learned complex weights, iFFT.

    Layout is (B, C, nx, nt): the nx axis (periodic x) uses a full FFT with the
    lowest +/- ``modes_x`` modes kept; the nt axis uses the real FFT with the
    lowest ``modes_t`` modes. Weights are fixed-size in mode space, so the same
    layer applies at any grid resolution.
    """

    def __init__(self, in_ch: int, out_ch: int, modes_x: int, modes_t: int) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes_x = modes_x
        self.modes_t = modes_t
        scale = 1.0 / (in_ch * out_ch)
        # two blocks: positive and negative x-frequencies (t uses rfft -> non-negative only)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes_x, modes_t, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_ch, out_ch, modes_x, modes_t, dtype=torch.cfloat)
        )

    @staticmethod
    def _compl_mul(inp: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # (B, in, mx, mt) x (in, out, mx, mt) -> (B, out, mx, mt)
        return torch.einsum("bixt,ioxt->boxt", inp, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, nx, nt = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))  # (B, C, nx, nt//2+1)
        mx = min(self.modes_x, nx // 2)
        mt = min(self.modes_t, nt // 2 + 1)
        out_ft = torch.zeros(b, self.out_ch, nx, nt // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :mx, :mt] = self._compl_mul(
            x_ft[:, :, :mx, :mt], self.weights1[:, :, :mx, :mt]
        )
        out_ft[:, :, -mx:, :mt] = self._compl_mul(
            x_ft[:, :, -mx:, :mt], self.weights2[:, :, :mx, :mt]
        )
        return torch.fft.irfft2(out_ft, s=(nx, nt), dim=(-2, -1))


class FNO2d(nn.Module):
    """Space-time FNO mapping u0(x) -> u(x, t)."""

    def __init__(
        self,
        modes_x: int = 32,
        modes_t: int = 16,
        width: int = 64,
        n_layers: int = 4,
        t_pad_frac: float = 0.25,
    ) -> None:
        super().__init__()
        self.width = width
        self.t_pad_frac = t_pad_frac
        self.lift = nn.Conv2d(2, width, 1)  # (u0, t) -> width
        self.spectral = nn.ModuleList(
            SpectralConv2d(width, width, modes_x, modes_t) for _ in range(n_layers)
        )
        self.w = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(n_layers))
        self.proj1 = nn.Conv2d(width, 128, 1)
        self.proj2 = nn.Conv2d(128, 1, 1)

    def forward(self, u0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """u0: (B, nx); t: (nt,). Returns u: (B, nx, nt)."""
        b, nx = u0.shape
        nt = t.shape[0]
        u0_bc = u0[:, :, None].expand(b, nx, nt)
        t_bc = t.to(u0)[None, None, :].expand(b, nx, nt)
        x = torch.stack([u0_bc, t_bc], dim=1)  # (B, 2, nx, nt)

        x = self.lift(x)
        pad = max(1, round(self.t_pad_frac * nt))  # zero-pad non-periodic t, crop later
        x = F.pad(x, (0, pad))
        for spec, w in zip(self.spectral, self.w, strict=True):
            x = F.gelu(spec(x) + w(x))
        x = x[..., :nt]  # crop t back

        x = F.gelu(self.proj1(x))
        x = self.proj2(x)  # (B, 1, nx, nt)
        return x.squeeze(1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
