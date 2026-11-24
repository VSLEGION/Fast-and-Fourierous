"""WebSocket protocol handler: simulate / train_pinn / refine / cancel (CLAUDE.md §6.6).

Design:
  * One outgoing queue per connection, drained by a single ``sender`` task, so
    both async code and background threads can emit messages safely (threads use
    loop.call_soon_threadsafe).
  * ``simulate`` runs as a cancellable task; a newer simulate supersedes the
    previous (latest-IC-wins for the real-time flow). Predictions go out
    immediately; the truth (and metrics) follow from a threadpool so the event
    loop never blocks.
  * ``train_pinn`` / ``refine`` are the long tasks -- at most one at a time, run
    in a thread, streaming progress, and stopped via a threading.Event on cancel.
"""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import orjson
import torch
from fastapi import WebSocket, WebSocketDisconnect

from shocklab.eval.ood import ood_report
from shocklab.physics.solver import NU, x_grid
from shocklab.server.schemas import (
    CancelMsg,
    RefineMsg,
    SimulateMsg,
    TrainPinnMsg,
    field_payload,
    parse_message,
)
from shocklab.server.scoring import scorecard
from shocklab.server.truth_cache import _resample_periodic_1d
from shocklab.train.pinn import _predict_grid, train_pinn_stream
from shocklab.train.pino import refine


class Connection:
    """Handles one WebSocket client for the whole session."""

    def __init__(self, ws: WebSocket, registry, truth_cache) -> None:
        self.ws = ws
        self.reg = registry
        self.truth = truth_cache
        self.device = registry.device
        self.out_q: asyncio.Queue = asyncio.Queue()
        self.sim_task: asyncio.Task | None = None
        self.long_task: asyncio.Task | None = None
        self.cancel_event: threading.Event | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

    # --- lifecycle -----------------------------------------------------------

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        sender = asyncio.create_task(self._sender())
        try:
            while True:
                raw = await self.ws.receive_text()
                try:
                    msg = parse_message(orjson.loads(raw))
                except Exception as e:  # noqa: BLE001 - report malformed input
                    await self.out_q.put({"type": "error", "id": None, "error": str(e)})
                    continue
                await self._dispatch(msg)
        except WebSocketDisconnect:
            pass
        finally:
            if self.cancel_event is not None:
                self.cancel_event.set()
            for tsk in (self.sim_task, self.long_task):
                if tsk is not None:
                    tsk.cancel()
            sender.cancel()

    async def _sender(self) -> None:
        while True:
            msg = await self.out_q.get()
            await self.ws.send_text(orjson.dumps(msg).decode("ascii"))

    def _emit(self, msg: dict) -> None:
        """Thread-safe enqueue (used by background-thread callbacks)."""
        self.loop.call_soon_threadsafe(self.out_q.put_nowait, msg)

    # --- dispatch ------------------------------------------------------------

    async def _dispatch(self, msg) -> None:
        if isinstance(msg, SimulateMsg):
            if self.sim_task is not None and not self.sim_task.done():
                self.sim_task.cancel()
            self.sim_task = asyncio.create_task(self._simulate(msg))
        elif isinstance(msg, TrainPinnMsg):
            await self._start_long("train_pinn", msg)
        elif isinstance(msg, RefineMsg):
            await self._start_long("refine", msg)
        elif isinstance(msg, CancelMsg):
            if self.cancel_event is not None:
                self.cancel_event.set()
            if self.sim_task is not None and not self.sim_task.done():
                self.sim_task.cancel()

    # --- simulate ------------------------------------------------------------

    async def _simulate(self, msg: SimulateMsg) -> None:
        try:
            u0 = msg.u0.array()
            nx, nt = msg.grid.nx, msg.grid.nt
            if len(u0) != nx:
                u0 = _resample_periodic_1d(u0, nx)
            x, t = x_grid(nx), np.linspace(0, 1, nt)

            level, reasons = ood_report(u0)
            await self.out_q.put({"type": "ood", "id": msg.id, "level": level, "reasons": reasons})

            preds: dict[str, np.ndarray] = {}
            for name in msg.models:
                if name in ("fno", "pino") and name in self.reg.available():
                    field, ms = self.reg.predict_operator(name, u0, nt)
                    preds[name] = field
                    await self.out_q.put(
                        {
                            "type": "prediction",
                            "id": msg.id,
                            "model": name,
                            "field": field_payload(field),
                            "ms": round(ms, 3),
                        }
                    )
                elif name == "pinn":
                    if msg.preset == "classic" and "pinn_classic" in self.reg.available():
                        field, ms = self.reg.predict_pinn("pinn_classic", x, t)
                        preds["pinn"] = field
                        await self.out_q.put(
                            {
                                "type": "prediction",
                                "id": msg.id,
                                "model": "pinn",
                                "field": field_payload(field),
                                "ms": round(ms, 3),
                            }
                        )
                    else:
                        await self.out_q.put(
                            {"type": "pinn_status", "id": msg.id, "status": "RETRAIN_REQUIRED"}
                        )

            field, source, tms = await asyncio.to_thread(
                self.truth.get, msg.preset, msg.preset_params, u0, msg.nu, nx, nt
            )
            await self.out_q.put(
                {
                    "type": "truth",
                    "id": msg.id,
                    "source": source,
                    "field": field_payload(field),
                    "ms": round(tms, 1),
                }
            )

            per_model = await asyncio.to_thread(self._metrics, preds, field, x, t, msg.nu)
            await self.out_q.put({"type": "metrics", "id": msg.id, "per_model": per_model})
            await self.out_q.put({"type": "done", "id": msg.id})
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - surface errors to the client, don't crash the socket
            await self.out_q.put({"type": "error", "id": msg.id, "error": repr(e)})

    def _metrics(self, preds, truth, x, t, nu) -> dict:
        return {name: scorecard(pf, truth, x, t, nu) for name, pf in preds.items()}

    # --- long tasks ----------------------------------------------------------

    async def _start_long(self, task: str, msg) -> None:
        if self.long_task is not None and not self.long_task.done():
            await self.out_q.put(
                {"type": "error", "id": msg.id, "error": "a long task is already running"}
            )
            return
        cancel = threading.Event()
        self.cancel_event = cancel
        runner = self._run_train_pinn if task == "train_pinn" else self._run_refine
        self.long_task = asyncio.create_task(asyncio.to_thread(runner, msg, cancel))

    def _run_train_pinn(self, msg: TrainPinnMsg, cancel: threading.Event) -> None:
        try:
            u0 = msg.u0.array()
            nx = len(u0)

            def cb(it, losses, field):
                self._emit(
                    {
                        "type": "progress",
                        "id": msg.id,
                        "task": "train_pinn",
                        "iter": it,
                        "losses": losses,
                        "preview": field_payload(field),
                    }
                )

            iters = 4000 if msg.config == "fast" else 2000
            model = train_pinn_stream(
                u0, NU, self.device, iters=iters, callback=cb, should_cancel=cancel.is_set
            )
            field = _predict_grid(model, self.device, x_grid(nx), np.linspace(0, 1, 101))
            self._emit(
                {
                    "type": "done",
                    "id": msg.id,
                    "task": "train_pinn",
                    "field": field_payload(field),
                    "cancelled": cancel.is_set(),
                }
            )
        except Exception as e:  # noqa: BLE001
            self._emit({"type": "error", "id": msg.id, "error": repr(e)})

    def _run_refine(self, msg: RefineMsg, cancel: threading.Event) -> None:
        try:
            u0 = msg.u0.array()
            nx, nt = msg.grid.nx, msg.grid.nt
            if len(u0) != nx:
                u0 = _resample_periodic_1d(u0, nx)
            if "pino" not in self.reg.available():
                self._emit({"type": "error", "id": msg.id, "error": "PINO checkpoint not loaded"})
                return
            model = self.reg.model("pino")
            t = torch.linspace(0, 1, nt, device=self.device)
            u0_t = torch.as_tensor(u0[None], device=self.device, dtype=torch.float32)

            def cb(step, pde, ic, pred):
                if step % 20 == 0 or step == msg.steps:
                    preview = pred.transpose(-1, -2)[0].detach().cpu().numpy()
                    self._emit(
                        {
                            "type": "progress",
                            "id": msg.id,
                            "task": "refine",
                            "iter": step,
                            "losses": {"pde": pde, "ic": ic},
                            "preview": field_payload(preview),
                        }
                    )

            final, _ = refine(
                model, u0_t, t, steps=msg.steps, callback=cb, should_cancel=cancel.is_set
            )
            field = final.transpose(-1, -2)[0].cpu().numpy()
            self._emit(
                {
                    "type": "done",
                    "id": msg.id,
                    "task": "refine",
                    "field": field_payload(field),
                    "cancelled": cancel.is_set(),
                }
            )
        except Exception as e:  # noqa: BLE001
            self._emit({"type": "error", "id": msg.id, "error": repr(e)})
