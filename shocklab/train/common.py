"""Shared training utilities: seeding, device, timers, run logging and checkpoints.

Every run writes to checkpoints/<model>/ (CLAUDE.md §4, §10):
    model.pt        state dict
    config.yaml     the exact config used
    metrics.json    config + git hash + GPU + torch version + timings + results
    train_log.csv   per-iteration loss log
"""

from __future__ import annotations

import csv
import json
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    """Seed random, numpy and torch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "cuda") -> torch.device:
    """Return the compute device. No silent fallback: warn loudly if CUDA is missing."""
    if prefer == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("WARNING: config requested CUDA but torch.cuda.is_available() is False; using CPU.")
    return torch.device("cpu")


def git_hash() -> str:
    """Short git commit hash of the working tree, or 'nogit'/'dirty' markers."""
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).returncode  # 1 if there are unstaged changes
        return f"{h}+dirty" if dirty else h
    except Exception:
        return "nogit"


def env_info() -> dict[str, Any]:
    """Environment provenance recorded in every metrics.json."""
    info: dict[str, Any] = {
        "git_hash": git_hash(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


class Timer:
    """Context manager that records wall-clock seconds in ``.seconds``."""

    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        self.seconds = 0.0
        return self

    def elapsed(self) -> float:
        """Seconds since entering the context (usable while still inside it)."""
        return time.perf_counter() - self._t0

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._t0


class CSVLogger:
    """Append-only CSV logger with a fixed set of columns."""

    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=fieldnames)
        self._w.writeheader()

    def log(self, **row: Any) -> None:
        self._w.writerow(row)

    def close(self) -> None:
        self._f.close()


def save_metrics(outdir: Path, config: dict[str, Any], results: dict[str, Any]) -> Path:
    """Write metrics.json = {config, env, results} and config.yaml. Returns the json path."""
    import yaml

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    payload = {"config": config, "env": env_info(), "results": results}
    path = outdir / "metrics.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
