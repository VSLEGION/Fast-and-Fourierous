"""Compute the ground-truth field for an initial condition, with an LRU cache (CLAUDE.md §6.2).

Truth source is chosen from the preset (and falls back to the solver for anything
custom):

  EXACT     Cole-Hopf, classic -sin(pi x)
  SYMMETRY  shift u(x-s, t) or Galilean boost u(x-c t, t) + c of the exact classic
  ANALYTIC  linear limit eps e^{-nu pi^2 t} sin(pi x)
  HELD-OUT  a stored test-set sample
  SOLVER    ETDRK4 at verified resolution, for everything else (drawn ICs, two-wave, ...)

Results are cached by (preset, params, nu, grid, u0 hash). The solver runs are the
expensive ones; they are meant to be called off the event loop (asyncio.to_thread).
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict

import numpy as np

from shocklab.data.datasets import load_test
from shocklab.physics import invariants as inv
from shocklab.physics.exact import burgers_exact_sine
from shocklab.physics.solver import NU, burgers_solve, x_grid


def _resample_periodic_1d(u0: np.ndarray, N: int) -> np.ndarray:
    """Spectral resample a periodic band-limited signal from len(u0) to N points."""
    n = len(u0)
    if N == n:
        return u0.astype(float)
    uh = np.fft.rfft(u0)
    out = np.zeros(N // 2 + 1, dtype=complex)
    m = min(len(uh), len(out))
    out[:m] = uh[:m] * (N / n)
    return np.fft.irfft(out, n=N)


def _resample_field(field: np.ndarray, nx: int, nt: int) -> np.ndarray:
    """Linearly resample a (nt_src, nx_src) field to (nt, nx) for display/metrics."""
    nt_src, nx_src = field.shape
    xs, xd = np.linspace(0, 1, nx_src), np.linspace(0, 1, nx)
    ts, td = np.linspace(0, 1, nt_src), np.linspace(0, 1, nt)
    fx = np.stack([np.interp(xd, xs, field[i]) for i in range(nt_src)])  # (nt_src, nx)
    return np.stack([np.interp(td, ts, fx[:, j]) for j in range(nx)], axis=1)  # (nt, nx)


def _solve_resolution(nx: int, nu: float) -> int:
    """N >= next_pow2(4/nu), also >= nx and >= 256, capped at 8192 (CLAUDE.md §1)."""
    target = max(4.0 / nu, nx, 256)
    N = 1 << int(math.ceil(math.log2(target)))
    return min(N, 8192)


def _pick_dt(N: int, nt: int) -> float:
    """dt keeping the dealiased CFL < ~0.9 and T/dt a multiple of (nt-1)."""
    kmax = np.pi * ((2 * (N // 2)) // 3)
    dt_cfl = 0.9 / (1.2 * kmax)
    m = nt - 1
    steps = int(math.ceil((1.0 / dt_cfl) / m)) * m
    return 1.0 / steps


def _exact_classic(nx: int, nt: int, nu: float) -> np.ndarray:
    x = x_grid(nx)
    t = np.linspace(0, 1, nt)
    return np.stack([burgers_exact_sine(x, float(ti), nu=nu) for ti in t])  # (nt, nx)


def _solver_truth(u0: np.ndarray, nu: float, nx: int, nt: int) -> np.ndarray:
    N = _solve_resolution(nx, nu)
    u0_N = _resample_periodic_1d(u0, N)
    dt = _pick_dt(N, nt)
    U = burgers_solve(u0_N, n_out=nt, dt=dt, nu=nu)[0]  # (nt, N)
    if N % nx == 0:
        return U[:, :: N // nx]
    return _resample_field(U, nx, nt)


def compute_truth(
    preset: str | None,
    params: dict[str, float],
    u0: np.ndarray,
    nu: float,
    nx: int,
    nt: int,
) -> tuple[np.ndarray, str]:
    """Return (field (nt, nx), source chip). The models were trained at the default nu;
    a changed nu is still honored for the truth (EXACT/SYMMETRY/ANALYTIC/SOLVER all
    generalize), but HELD-OUT is only valid at the default nu and otherwise re-solves."""
    x = x_grid(nx)
    t = np.linspace(0, 1, nt)
    p = params or {}

    if preset == "classic":
        return _exact_classic(nx, nt, nu), "EXACT"
    if preset == "shifted":
        base = _exact_classic(nx, nt, nu)
        return inv.translate(base, float(p.get("s", 0.3)), axis=-1), "SYMMETRY"
    if preset == "boosted":
        base = _exact_classic(nx, nt, nu)
        return inv.galilean_boost(base, t, float(p.get("c", 0.4))), "SYMMETRY"
    if preset == "linear":
        eps = float(p.get("eps", 0.05))
        field = eps * np.exp(-nu * np.pi**2 * t)[:, None] * np.sin(np.pi * x)[None, :]
        return field, "ANALYTIC"
    if preset == "heldout" and abs(nu - NU) < 1e-12:
        test = load_test()  # (200, 51, 1024)
        idx = int(p.get("index", 0)) % test["u"].shape[0]
        return _resample_field(test["u"][idx], nx, nt), "HELD-OUT"

    # drawn / two_wave / n_wave / gaussian / null / non-default-nu heldout
    return _solver_truth(u0, nu, nx, nt), "SOLVER"


class TruthCache:
    """LRU cache of computed truth fields, keyed by the request."""

    def __init__(self, maxsize: int = 128) -> None:
        self.maxsize = maxsize
        self._store: OrderedDict[str, tuple[np.ndarray, str]] = OrderedDict()

    @staticmethod
    def _key(preset, params, u0, nu, nx, nt) -> str:
        h = hashlib.blake2b(
            np.ascontiguousarray(u0, dtype="<f4").tobytes(), digest_size=16
        ).hexdigest()
        return f"{preset}|{sorted((params or {}).items())}|{nu:.9g}|{nx}|{nt}|{h}"

    def get(self, preset, params, u0, nu, nx, nt) -> tuple[np.ndarray, str, float]:
        """Return (field, source, ms). Cached hits report ms ~ 0."""
        key = self._key(preset, params, u0, nu, nx, nt)
        if key in self._store:
            self._store.move_to_end(key)
            field, source = self._store[key]
            return field, source, 0.0
        t0 = time.perf_counter()
        field, source = compute_truth(preset, params, u0, nu, nx, nt)
        ms = (time.perf_counter() - t0) * 1e3
        self._store[key] = (field, source)
        if len(self._store) > self.maxsize:
            self._store.popitem(last=False)
        return field, source, ms
