"""Accuracy metrics shared by every model, computed vs truth on a common grid.

All metrics take fields of shape (nt, nx) on the periodic grid x_j = -1 + 2j/N
and are computed in float64 (CLAUDE.md §6.5, §10):

  - rel_l2:  relative L2 over the whole space-time field,
  - linf:    max absolute error,
  - shock_position_error: mean |x_shock(pred) - x_shock(truth)| for t > t_b,
    with the shock located as argmax |u_x| plus a parabolic sub-grid refinement.
"""

from __future__ import annotations

import numpy as np

T_B = 1.0 / np.pi  # breaking time for the classic case u0 = -sin(pi x)


def rel_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    """Relative L2 error ||pred - truth|| / ||truth|| over the whole field."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))


def linf(pred: np.ndarray, truth: np.ndarray) -> float:
    """Max absolute error."""
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return float(np.max(np.abs(pred - truth)))


def mean_rel_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    """Mean over samples of per-sample relative L2 (the standard FNO test metric).

    pred, truth: (N, ...); the norm is over everything but the leading sample axis.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    n = pred.shape[0]
    d = (pred - truth).reshape(n, -1)
    tgt = truth.reshape(n, -1)
    return float(np.mean(np.linalg.norm(d, axis=1) / np.linalg.norm(tgt, axis=1)))


def _shock_x(field: np.ndarray, x: np.ndarray) -> np.ndarray:
    """x-location of steepest |u_x| per row, with parabolic sub-grid refinement.

    ``field`` is (..., nx); returns (...,). Uses finite differences (robust to
    fields that are not exactly periodic, e.g. a PINN prediction).
    """
    field = np.asarray(field, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    nx = field.shape[-1]
    dx = x[1] - x[0]
    g = np.abs(np.gradient(field, x, axis=-1))
    flat = g.reshape(-1, nx)
    out = np.empty(flat.shape[0])
    for r in range(flat.shape[0]):
        i = int(np.argmax(flat[r]))
        if 0 < i < nx - 1:
            ym1, y0, yp1 = flat[r, i - 1], flat[r, i], flat[r, i + 1]
            denom = ym1 - 2 * y0 + yp1
            delta = 0.5 * (ym1 - yp1) / denom if denom != 0 else 0.0
            out[r] = x[i] + np.clip(delta, -1.0, 1.0) * dx
        else:
            out[r] = x[i]
    return out.reshape(field.shape[:-1])


def shock_position_error(
    pred: np.ndarray,
    truth: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    t_b: float = T_B,
) -> float:
    """Mean |x_shock(pred) - x_shock(truth)| over snapshots with t > t_b.

    pred, truth: (nt, nx); x: (nx,); t: (nt,). Returns ``nan`` if no t > t_b.
    """
    t = np.asarray(t, dtype=np.float64)
    mask = t > t_b
    if not mask.any():
        return float("nan")
    xs_pred = _shock_x(np.asarray(pred)[mask], x)
    xs_truth = _shock_x(np.asarray(truth)[mask], x)
    return float(np.mean(np.abs(xs_pred - xs_truth)))
