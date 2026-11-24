"""WebSocket / REST message schemas and the base64 float32 field codec (CLAUDE.md §6.6).

Arrays travel as base64 little-endian float32. Space-time fields are [nt][nx]
row-major. These pydantic models mirror web/src/api/types.ts (keep in sync).
"""

from __future__ import annotations

import base64
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from shocklab.physics.solver import NU

# --- base64 float32 codec ----------------------------------------------------


def encode_array(arr: np.ndarray) -> str:
    """Encode a numeric array as base64 little-endian float32 (row-major)."""
    a = np.ascontiguousarray(np.asarray(arr), dtype="<f4")
    return base64.b64encode(a.tobytes()).decode("ascii")


def decode_array(b64: str) -> np.ndarray:
    """Decode base64 little-endian float32 into a 1-D float64 array."""
    return np.frombuffer(base64.b64decode(b64), dtype="<f4").astype(np.float64)


def field_payload(field: np.ndarray) -> dict[str, Any]:
    """Pack a (nt, nx) field as {b64, nx, nt} (row-major [nt][nx])."""
    field = np.asarray(field)
    return {
        "b64": encode_array(field.ravel(order="C")),
        "nx": int(field.shape[1]),
        "nt": int(field.shape[0]),
    }


# --- incoming messages -------------------------------------------------------


class U0(BaseModel):
    b64: str
    n: int

    def array(self) -> np.ndarray:
        return decode_array(self.b64)[: self.n]


class Grid(BaseModel):
    nx: int = 256
    nt: int = 101


class SimulateMsg(BaseModel):
    type: Literal["simulate"]
    id: str
    u0: U0
    nu: float = NU
    grid: Grid = Grid()
    models: list[str] = ["fno", "pino", "pinn"]
    preset: str | None = None
    preset_params: dict[str, float | int] = {}


class TrainPinnMsg(BaseModel):
    type: Literal["train_pinn"]
    id: str
    u0: U0
    config: str = "fast"
    preset: str | None = None


class RefineMsg(BaseModel):
    type: Literal["refine"]
    id: str
    u0: U0
    steps: int = 300
    grid: Grid = Grid()


class CancelMsg(BaseModel):
    type: Literal["cancel"]
    id: str


def parse_message(raw: dict[str, Any]) -> BaseModel:
    """Parse an incoming WS message dict into the right pydantic model."""
    t = raw.get("type")
    table = {
        "simulate": SimulateMsg,
        "train_pinn": TrainPinnMsg,
        "refine": RefineMsg,
        "cancel": CancelMsg,
    }
    if t not in table:
        raise ValueError(f"unknown message type: {t!r}")
    return table[t].model_validate(raw)
