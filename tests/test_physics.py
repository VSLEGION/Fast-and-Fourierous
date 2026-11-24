"""M1 physics-core tests (CLAUDE.md §9).

These pin the verified numerics of the ported solver, exact solution and GRF ICs.
The physics core must not change without these staying green. Thresholds are set
with generous margins above the measured errors so the suite is meaningful but
not flaky.
"""

from __future__ import annotations

import numpy as np

from shocklab.physics import invariants as inv
from shocklab.physics.exact import burgers_exact_sine
from shocklab.physics.ic import random_ic
from shocklab.physics.solver import NU, burgers_solve, x_grid


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# --- Solver vs exact Cole-Hopf (classic case) --------------------------------


def test_solver_matches_exact_classic():
    """ETDRK4 vs exact at N=2048, rel L2 < 1e-8 for t in {0.25, 0.5, 0.75, 1}."""
    N = 2048
    x = x_grid(N)
    U = burgers_solve(-np.sin(np.pi * x), n_out=5, dt=4e-4)[0]  # t = 0, .25, .5, .75, 1
    for i, t in enumerate([0.25, 0.5, 0.75, 1.0]):
        err = rel_l2(U[i + 1], burgers_exact_sine(x, t))
        assert err < 1e-8, f"t={t}: rel L2 {err:.2e}"


def test_spectral_convergence_classic():
    """Error falls by orders of magnitude as N doubles (spectral convergence)."""
    errs = {}
    for N, dt in [(512, 5e-4), (1024, 5e-4), (2048, 4e-4)]:
        x = x_grid(N)
        U = burgers_solve(-np.sin(np.pi * x), n_out=5, dt=dt)[0][-1]
        errs[N] = rel_l2(U, burgers_exact_sine(x, 1.0))
    assert errs[2048] < errs[1024] < errs[512], errs
    assert errs[2048] < 1e-8, errs


def test_self_convergence_shock_ic():
    """A non-classic shock-forming IC agrees between N=2048 and N=4096 (ref resolution is converged)."""

    def ic(xx):
        return (
            0.8 * np.sin(np.pi * xx)
            - 0.5 * np.sin(2 * np.pi * xx + 0.7)
            + 0.3 * np.cos(3 * np.pi * xx)
        )

    u2048 = burgers_solve(ic(x_grid(2048)), n_out=2, dt=4e-4)[0][-1]
    u4096 = burgers_solve(ic(x_grid(4096)), n_out=2, dt=2e-4)[0][-1]
    err = rel_l2(u2048, u4096[::2])
    assert err < 1e-6, f"self-convergence rel L2 {err:.2e}"


# --- Invariants ---------------------------------------------------------------


def test_mass_conserved():
    """int u dx is conserved to machine precision (< 1e-10)."""
    rng = np.random.default_rng(0)
    U = burgers_solve(random_ic(4, 2048, rng), n_out=51, dt=4e-4)
    m = inv.mass(U, axis=-1)  # (4, 51)
    drift = float(np.abs(m - m[:, :1]).max())
    assert drift < 1e-10, f"mass drift {drift:.2e}"


def test_energy_non_increasing():
    """E(t) = 1/2 int u^2 dx is non-increasing (dE/dt = -nu int u_x^2 dx <= 0)."""
    rng = np.random.default_rng(1)
    U = burgers_solve(random_ic(4, 2048, rng), n_out=51, dt=4e-4)
    dE = np.diff(inv.energy(U, axis=-1), axis=-1)
    assert dE.max() < 1e-9, f"max energy increase {dE.max():.2e}"


# --- Symmetry transforms ------------------------------------------------------


def test_translation_equivariance():
    """Solving a shifted IC equals shifting the solution (exact: periodic spectral)."""
    N = 1024
    x = x_grid(N)
    s = 0.3
    u0 = -np.sin(np.pi * x)
    U = burgers_solve(u0, n_out=11, dt=5e-4)[0]
    U_shift = burgers_solve(inv.translate(u0, s), n_out=11, dt=5e-4)[0]
    err = rel_l2(U_shift, inv.translate(U, s, axis=-1))
    assert err < 1e-10, f"translation rel L2 {err:.2e}"


def test_galilean_boost_equivariance():
    """Solving u0 + c equals boosting the solution: u(x - c t, t) + c."""
    N = 1024
    x = x_grid(N)
    c = 0.2
    t = np.linspace(0, 1, 11)
    u0 = -np.sin(np.pi * x)
    U = burgers_solve(u0, n_out=11, dt=5e-4)[0]
    U_boost = burgers_solve(u0 + c, n_out=11, dt=5e-4)[0]
    err = rel_l2(U_boost, inv.galilean_boost(U, t, c))
    assert err < 1e-6, f"galilean boost rel L2 {err:.2e}"


def test_linear_limit_heat_equation():
    """Small-amplitude u0 = eps sin(pi x) -> eps e^{-nu pi^2 t} sin(pi x); error is O(eps)."""
    N = 1024
    x = x_grid(N)

    def err_at(eps):
        u = burgers_solve(eps * np.sin(np.pi * x), n_out=2, dt=5e-4)[0][-1]
        analytic = eps * np.exp(-NU * np.pi**2 * 1.0) * np.sin(np.pi * x)
        return rel_l2(u, analytic)

    e_big, e_small = err_at(0.05), err_at(0.01)
    assert e_big < 0.1, f"eps=0.05 rel L2 {e_big:.2e}"
    assert e_small < e_big, (
        f"linear limit should tighten as eps shrinks: {e_small:.2e} vs {e_big:.2e}"
    )


def test_shock_location_classic():
    """The classic shock sits at x = 0 after it forms."""
    N = 2048
    x = x_grid(N)
    u1 = burgers_solve(-np.sin(np.pi * x), n_out=2, dt=4e-4)[0][-1]
    loc = float(inv.shock_location(u1, x))
    assert abs(loc) < 0.01, f"shock at x={loc:.4f}"


# --- Data contract (CLAUDE.md §3) --------------------------------------------


def test_data_split_shapes():
    """make_split / make_classic produce the §3 shapes, dtype and subsampled grids."""
    from shocklab.data.generate import make_classic, make_split

    u_tr = make_split(2, store_every=8, seed=0, batch=2)
    assert u_tr.shape == (2, 51, 256) and u_tr.dtype == np.float32

    u_te = make_split(2, store_every=2, seed=1, batch=2)
    assert u_te.shape == (2, 51, 1024) and u_te.dtype == np.float32

    u_c, t_c, x_c = make_classic()
    assert u_c.shape == (101, 1024) and u_c.dtype == np.float32
    assert t_c.shape == (101,) and x_c.shape == (1024,)


def test_generate_writes_contract(tmp_path):
    """The CLI writer produces the three files with the right keys, shapes and nu."""
    from shocklab.data.generate import generate

    generate(tmp_path, n_train=2, n_test=2, batch=2)
    expect = {
        "burgers_train.npz": (2, 51, 256),
        "burgers_test.npz": (2, 51, 1024),
        "burgers_classic.npz": (101, 1024),
    }
    for fn, ushape in expect.items():
        d = np.load(tmp_path / fn)
        assert set(d.files) >= {"u", "t", "x", "nu"}, d.files
        assert d["u"].shape == ushape and d["u"].dtype == np.float32
        assert np.isclose(float(d["nu"]), NU)
