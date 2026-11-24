"""The 'fundamental' (Raissi-style) PINN for the classic Burgers case.

u_theta(x, t) is a plain MLP: inputs (x, t), 8 hidden layers x 20 units, tanh,
Xavier init (CLAUDE.md §5.1). It learns ONE solution from the physics alone --
the PDE residual u_t + u u_x - nu u_xx, the initial condition, and a soft
periodic boundary condition -- with no data.

Inputs are normalized to [-1, 1] inside ``forward`` for conditioning; because
the normalization is affine, autograd through it yields the correct u_x, u_xx,
u_t (the trainer differentiates w.r.t. the raw x, t it passes in).
"""

from __future__ import annotations

import torch
from torch import nn


class PINN(nn.Module):
    """MLP u_theta(x, t) for Burgers' equation on [x_lo, x_hi] x [t_lo, t_hi]."""

    def __init__(
        self,
        n_hidden_layers: int = 8,
        width: int = 20,
        x_lo: float = -1.0,
        x_hi: float = 1.0,
        t_lo: float = 0.0,
        t_hi: float = 1.0,
    ) -> None:
        super().__init__()
        dims = [2] + [width] * n_hidden_layers + [1]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        for lin in self.layers:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.register_buffer("_lo", torch.tensor([x_lo, t_lo]))
        self.register_buffer("_hi", torch.tensor([x_hi, t_hi]))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x, t: (N, 1) tensors; returns u: (N, 1)."""
        h = torch.cat([x, t], dim=-1)
        h = 2.0 * (h - self._lo) / (self._hi - self._lo) - 1.0  # -> [-1, 1]^2
        for lin in self.layers[:-1]:
            h = torch.tanh(lin(h))
        return self.layers[-1](h)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
