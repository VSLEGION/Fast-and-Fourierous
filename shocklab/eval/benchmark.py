"""Inference latency benchmark (CLAUDE.md §5.2, §6): p50/p95 for batch-1, after warmup."""

from __future__ import annotations

import time

import numpy as np
import torch


@torch.no_grad()
def latency_ms(
    model: torch.nn.Module,
    u0: torch.Tensor,
    t: torch.Tensor,
    n_warmup: int = 20,
    n_runs: int = 100,
) -> dict[str, float]:
    """Measure single-sample forward latency in milliseconds.

    u0: (1, nx); t: (nt,). Returns {p50, p95, mean, min} in ms. Each run is
    synchronized on CUDA so the timings reflect real device work.
    """
    cuda = u0.is_cuda
    for _ in range(n_warmup):
        model(u0, t)
    if cuda:
        torch.cuda.synchronize()

    times = np.empty(n_runs)
    for i in range(n_runs):
        t0 = time.perf_counter()
        model(u0, t)
        if cuda:
            torch.cuda.synchronize()
        times[i] = (time.perf_counter() - t0) * 1e3

    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "mean": float(times.mean()),
        "min": float(times.min()),
    }
