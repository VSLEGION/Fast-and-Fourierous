"""The UI presets -- "values we know the answer to" -- and their truth sources (CLAUDE.md §6.3).

Each preset carries its truth-source chip and the slider parameters the UI should
render. ``build_u0`` reconstructs the initial condition on a grid from a preset id
and parameters (the client normally builds u0 itself and sends it; the server uses
this for validation and for held-out samples).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shocklab.physics import ic
from shocklab.physics.solver import x_grid

# id -> metadata. ``source`` is the truth chip; ``params`` are slider specs.
PRESETS: list[dict[str, Any]] = [
    {"id": "classic", "label": "Classic  −sin(πx)", "source": "EXACT", "params": {}},
    {
        "id": "shifted",
        "label": "Shifted classic",
        "source": "SYMMETRY",
        "params": {"s": {"min": -1.0, "max": 1.0, "default": 0.3, "step": 0.01}},
    },
    {
        "id": "boosted",
        "label": "Boosted classic (OOD)",
        "source": "SYMMETRY",
        "ood": True,
        "params": {"c": {"min": -0.6, "max": 0.6, "default": 0.4, "step": 0.01}},
    },
    {
        "id": "linear",
        "label": "Linear limit (OOD)",
        "source": "ANALYTIC",
        "ood": True,
        "params": {"eps": {"min": 0.01, "max": 0.3, "default": 0.05, "step": 0.01}},
    },
    {
        "id": "two_wave",
        "label": "Two-wave",
        "source": "SOLVER",
        "params": {
            "A": {"min": -1.0, "max": 1.0, "default": 0.6, "step": 0.01},
            "B": {"min": -1.0, "max": 1.0, "default": 0.4, "step": 0.01},
            "phi1": {"min": 0.0, "max": 6.283, "default": 0.0, "step": 0.01},
            "phi2": {"min": 0.0, "max": 6.283, "default": 0.0, "step": 0.01},
        },
    },
    {
        "id": "n_wave",
        "label": "Smoothed N-wave",
        "source": "SOLVER",
        "params": {"amp": {"min": 0.3, "max": 1.2, "default": 1.0, "step": 0.01}},
    },
    {
        "id": "gaussian",
        "label": "Gaussian pulse (zero mean)",
        "source": "SOLVER",
        "params": {
            "x0": {"min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01},
            "width": {"min": 0.05, "max": 0.4, "default": 0.15, "step": 0.01},
        },
    },
    {
        "id": "heldout",
        "label": "Held-out test sample",
        "source": "HELD-OUT",
        "params": {"index": {"min": 0, "max": 199, "default": 0, "step": 1}},
    },
    {"id": "drawn", "label": "Hand-drawn", "source": "SOLVER", "params": {}},
]

PRESET_BY_ID = {p["id"]: p for p in PRESETS}


def build_u0(preset_id: str, params: dict[str, float], nx: int) -> np.ndarray:
    """Reconstruct u0 on x_grid(nx) from a preset id and parameters."""
    x = x_grid(nx)
    p = params or {}
    if preset_id == "classic":
        return ic.classic(x)
    if preset_id == "shifted":
        return ic.shifted_classic(x, float(p.get("s", 0.3)))
    if preset_id == "boosted":
        return ic.boosted_classic(x, float(p.get("c", 0.4)))
    if preset_id == "linear":
        return ic.linear_limit(x, float(p.get("eps", 0.05)))
    if preset_id == "two_wave":
        return ic.two_wave(
            x,
            float(p.get("A", 0.6)),
            float(p.get("B", 0.4)),
            float(p.get("phi1", 0.0)),
            float(p.get("phi2", 0.0)),
        )
    if preset_id == "n_wave":
        return ic.n_wave(x, float(p.get("amp", 1.0)))
    if preset_id == "gaussian":
        return ic.gaussian_pulse(x, float(p.get("x0", 0.0)), float(p.get("width", 0.15)))
    raise ValueError(f"preset {preset_id!r} has no server-side u0 builder")
