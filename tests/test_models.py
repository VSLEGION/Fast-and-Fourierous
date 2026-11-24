"""Model-level tests (M2: PINN; M3 adds FNO equivariance)."""

from __future__ import annotations

import numpy as np
import torch

from shocklab.eval import metrics
from shocklab.models.fno import FNO2d
from shocklab.models.pinn import PINN


def test_pinn_forward_shape():
    model = PINN()
    x = torch.rand(32, 1)
    t = torch.rand(32, 1)
    u = model(x, t)
    assert u.shape == (32, 1)


def test_pinn_architecture():
    """8 hidden layers x 20 units (CLAUDE.md §5.1)."""
    model = PINN(n_hidden_layers=8, width=20)
    assert len(model.layers) == 9  # 8 hidden + 1 output
    hidden = [lin.out_features for lin in model.layers[:-1]]
    assert hidden == [20] * 8


def test_pinn_second_derivative_is_finite():
    """The PDE residual needs u_t, u_x, u_xx via autograd; they must be finite."""
    model = PINN().double()
    x = torch.rand(16, 1, dtype=torch.float64, requires_grad=True)
    t = torch.rand(16, 1, dtype=torch.float64, requires_grad=True)
    u = model(x, t)
    ones = torch.ones_like(u)
    u_x = torch.autograd.grad(u, x, ones, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, ones, create_graph=True)[0]
    assert torch.isfinite(u_xx).all() and torch.isfinite(u_t).all()


def test_metrics_basic():
    truth = np.array([[0.0, 1.0, 0.0, -1.0]])
    assert metrics.rel_l2(truth, truth) == 0.0
    assert (
        metrics.linf(truth + 0.1, truth) == 0.1
        or abs(metrics.linf(truth + 0.1, truth) - 0.1) < 1e-9
    )


def test_shock_position_error_zero_for_identical():
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 256, endpoint=False)
    t = np.linspace(0, 1, 51)
    field = np.tanh(10 * (x[None, :] - 0.1)) * np.ones((51, 1)) + 0.01 * rng.standard_normal(
        (51, 256)
    )
    err = metrics.shock_position_error(field, field, x, t)
    assert err == 0.0


# --- FNO (M3) ----------------------------------------------------------------


def test_fno_output_shapes_any_resolution():
    """The FNO runs at arbitrary nx, nt (structural zero-shot super-resolution)."""
    model = FNO2d().eval()
    for nx in (64, 256, 512):
        for nt in (51, 101):
            with torch.no_grad():
                y = model(torch.randn(3, nx), torch.linspace(0, 1, nt))
            assert y.shape == (3, nx, nt)


def test_fno_translation_equivariance():
    """M3 done-criterion: shifting u0 in x shifts the output (no x-channel, periodic convs)."""
    torch.manual_seed(0)
    model = FNO2d().eval()
    nx, nt, shift = 128, 51, 17
    u0 = torch.randn(2, nx)
    t = torch.linspace(0, 1, nt)
    with torch.no_grad():
        y = model(u0, t)
        y_rolled = torch.roll(y, shifts=shift, dims=1)
        y_from_shifted = model(torch.roll(u0, shifts=shift, dims=1), t)
    err = (y_from_shifted - y_rolled).abs().max().item() / y.abs().max().item()
    assert err < 1e-4, f"equivariance error {err:.2e}"


# --- PINO physics residual (M4) ----------------------------------------------


def test_physics_loss_residual_matches_numpy():
    """The torch PDE residual (for the PINO loss) matches the numpy invariants residual."""
    from shocklab.models.physics_loss import pde_residual
    from shocklab.physics import invariants as inv
    from shocklab.physics.solver import NU

    nx, nt = 128, 51
    x = -1 + 2 * np.arange(nx) / nx
    t = np.linspace(0, 1, nt)
    field = np.zeros((nt, nx))
    for k, amp in [(1, 0.6), (2, -0.3), (3, 0.2)]:
        field += amp * np.sin(k * np.pi * x)[None, :] * np.cos(0.5 * t)[:, None]

    res_np = inv.pde_residual(field, t, NU)  # (nt, nx)
    u_torch = torch.tensor(field.T[None], dtype=torch.float64)  # (1, nx, nt)
    res_t = pde_residual(u_torch, float(t[1] - t[0]), NU).numpy()[0].T  # (nt, nx)
    assert np.abs(res_np - res_t).max() < 1e-8
