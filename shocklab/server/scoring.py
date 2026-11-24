"""Per-model metric scorecard vs truth on a common grid (CLAUDE.md §6.5)."""

from __future__ import annotations

import numpy as np

from shocklab.eval import metrics
from shocklab.physics import invariants as inv


def scorecard(pred: np.ndarray, truth: np.ndarray, x: np.ndarray, t: np.ndarray, nu: float) -> dict:
    """Accuracy + physics + conservation metrics for one model's field vs truth.

    pred, truth: (nt, nx). Returns rel_l2, linf, shock_dx, a normalized PDE
    residual (model residual / truth residual), mean-conservation drift, and an
    energy-monotonicity flag.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    rp = float(np.sqrt(np.mean(inv.pde_residual(pred, t, nu) ** 2)))
    rt = float(np.sqrt(np.mean(inv.pde_residual(truth, t, nu) ** 2)))
    mass = inv.mass(pred, axis=-1)  # (nt,)
    energy = inv.energy(pred, axis=-1)
    e_increase = float(np.diff(energy).max()) if len(energy) > 1 else 0.0
    return {
        "rel_l2": metrics.rel_l2(pred, truth),
        "linf": metrics.linf(pred, truth),
        "shock_dx": metrics.shock_position_error(pred, truth, x, t),
        "residual": rp / rt if rt > 0 else rp,  # normalized by the truth's own residual
        "residual_rms": rp,
        "mass_drift": float(np.abs(mass - mass[0]).max()),
        "energy_increase": e_increase,
        "energy_ok": bool(e_increase < 1e-3),
    }
