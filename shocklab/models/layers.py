"""Shared neural-network pieces used by the data-driven models (FNO, PINO)."""

from __future__ import annotations

import torch


def rel_l2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean per-sample relative L2 error ||pred - target|| / ||target|| (the FNO training loss).

    pred, target: (B, ...); the norm is taken over everything but the batch axis.
    """
    b = pred.shape[0]
    diff = (pred - target).reshape(b, -1)
    tgt = target.reshape(b, -1)
    return (diff.norm(dim=1) / (tgt.norm(dim=1) + eps)).mean()
