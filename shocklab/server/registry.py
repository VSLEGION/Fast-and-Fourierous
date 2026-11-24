"""Model registry: load the trained checkpoints and expose prediction methods (CLAUDE.md §6).

Loads the FNO, PINO and PINN checkpoints (whichever exist) onto the device with
their metrics.json (training cost, provenance). No silent fallback: missing
checkpoints are simply reported as unavailable via GET /api/models.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from shocklab.models.fno import FNO2d
from shocklab.models.pinn import PINN
from shocklab.train.common import REPO_ROOT, get_device

CKPT_DIR = REPO_ROOT / "checkpoints"


class Registry:
    """Holds the loaded models and their metadata, and runs inference."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or get_device("cuda")
        self.entries: dict[str, dict[str, Any]] = {}
        for name in ("fno", "pino"):
            self._load_operator(name)
        for name in ("pinn_classic", "pinn_fast"):
            self._load_pinn(name)

    # --- loading -------------------------------------------------------------

    def _read_meta(self, ckpt: Path) -> dict[str, Any]:
        meta = {}
        mj = ckpt / "metrics.json"
        if mj.exists():
            meta = json.loads(mj.read_text())
        return meta

    def _load_operator(self, name: str) -> None:
        ckpt = CKPT_DIR / name
        if not (ckpt / "model.pt").exists():
            return
        cfg = yaml.safe_load((ckpt / "config.yaml").read_text())
        model = FNO2d(
            cfg["modes_x"], cfg["modes_t"], cfg["width"], cfg["n_layers"], cfg["t_pad_frac"]
        )
        model.load_state_dict(torch.load(ckpt / "model.pt", map_location=self.device))
        model.to(self.device).eval()
        self.entries[name] = {"model": model, "kind": "operator", "meta": self._read_meta(ckpt)}

    def _load_pinn(self, name: str) -> None:
        ckpt = CKPT_DIR / name
        if not (ckpt / "model.pt").exists():
            return
        cfg = yaml.safe_load((ckpt / "config.yaml").read_text())
        model = PINN(cfg["n_hidden_layers"], cfg["width"])
        model.load_state_dict(torch.load(ckpt / "model.pt", map_location=self.device))
        model.to(self.device).eval()
        self.entries[name] = {"model": model, "kind": "pinn", "meta": self._read_meta(ckpt)}

    def available(self) -> list[str]:
        return list(self.entries)

    def model(self, name: str) -> torch.nn.Module:
        return self.entries[name]["model"]

    def info(self) -> dict[str, Any]:
        """For GET /api/models: availability + metrics for every expected model."""
        out = {}
        for name in ("fno", "pino", "pinn_classic", "pinn_fast"):
            e = self.entries.get(name)
            out[name] = {
                "available": e is not None,
                "kind": e["kind"] if e else None,
                "metrics": e["meta"].get("results", {}) if e else {},
                "env": e["meta"].get("env", {}) if e else {},
            }
        return out

    # --- inference -----------------------------------------------------------

    @torch.no_grad()
    def predict_operator(self, name: str, u0: np.ndarray, nt: int) -> tuple[np.ndarray, float]:
        """FNO/PINO forward: u0 (nx,) -> (field (nt, nx), ms)."""
        model = self.model(name)
        ub = torch.as_tensor(u0[None], device=self.device, dtype=torch.float32)
        tt = torch.linspace(0, 1, nt, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = model(ub, tt)  # (1, nx, nt)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        return pred.transpose(-1, -2)[0].cpu().numpy(), ms

    @torch.no_grad()
    def predict_pinn(self, name: str, x: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, float]:
        """PINN forward on a grid (classic case only): -> (field (nt, nx), ms)."""
        model = self.model(name)
        X, T = np.meshgrid(x, t)  # (nt, nx)
        xx = torch.as_tensor(X.reshape(-1, 1), device=self.device, dtype=torch.float32)
        tt = torch.as_tensor(T.reshape(-1, 1), device=self.device, dtype=torch.float32)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = model(xx, tt).reshape(T.shape)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        return pred.cpu().numpy(), ms
