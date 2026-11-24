"""M5 server tests: every WebSocket message type + the REST endpoints (CLAUDE.md §6.6, §9).

Uses the real loaded checkpoints (this machine has them + a GPU). Grids are kept
small (nt=21) so the exact/solver truths stay fast.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from shocklab.physics.ic import classic
from shocklab.physics.solver import NU, x_grid
from shocklab.server.app import create_app
from shocklab.server.schemas import encode_array

NX = 256
U0_B64 = encode_array(classic(x_grid(NX)).astype(np.float32))


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def simulate_msg(models, preset="classic", nt=21, mid="s1"):
    return {
        "type": "simulate",
        "id": mid,
        "u0": {"b64": U0_B64, "n": NX},
        "nu": NU,
        "grid": {"nx": NX, "nt": nt},
        "models": models,
        "preset": preset,
    }


def _drain_until(ws, stop_type, limit=40):
    msgs = []
    for _ in range(limit):
        m = ws.receive_json()
        msgs.append(m)
        if m["type"] == stop_type:
            break
    return msgs


# --- REST --------------------------------------------------------------------


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "fno" in body["models"]


def test_presets(client):
    r = client.get("/api/presets")
    assert r.status_code == 200
    presets = r.json()["presets"]
    assert len(presets) == 9
    assert {p["id"] for p in presets} >= {"classic", "boosted", "linear", "heldout", "drawn"}


def test_models_info(client):
    info = client.get("/api/models").json()
    assert info["fno"]["available"] is True
    assert "test_rel_l2_256" in info["fno"]["metrics"]


# --- WebSocket: simulate -----------------------------------------------------


def test_simulate_full_flow(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json(simulate_msg(["fno", "pino", "pinn"]))
        msgs = _drain_until(ws, "done")
    types = [m["type"] for m in msgs]
    assert "ood" in types
    assert any(m["type"] == "prediction" and m["model"] == "fno" for m in msgs)
    assert any(m["type"] == "prediction" and m["model"] == "pino" for m in msgs)
    assert any(
        m["type"] == "prediction" and m["model"] == "pinn" for m in msgs
    )  # classic -> PINN runs
    assert any(m["type"] == "truth" and m["source"] == "EXACT" for m in msgs)
    assert any(m["type"] == "metrics" for m in msgs)
    assert "done" in types
    # metrics scorecard has the expected keys
    met = next(m for m in msgs if m["type"] == "metrics")["per_model"]["fno"]
    assert {"rel_l2", "linf", "shock_dx", "residual", "mass_drift", "energy_ok"} <= set(met)


def test_simulate_pinn_retrain_required_offclassic(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json(simulate_msg(["fno", "pinn"], preset="gaussian"))
        msgs = _drain_until(ws, "done")
    assert any(m["type"] == "pinn_status" and m["status"] == "RETRAIN_REQUIRED" for m in msgs)
    assert any(m["type"] == "truth" and m["source"] == "SOLVER" for m in msgs)


def test_fno_end_to_end_under_50ms(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json(simulate_msg(["fno"], mid="warm"))  # warm up
        _drain_until(ws, "done")
        t0 = time.perf_counter()
        ws.send_json(simulate_msg(["fno"], mid="timed"))
        while True:
            m = ws.receive_json()
            if m["type"] == "prediction" and m["model"] == "fno":
                break
        e2e_ms = (time.perf_counter() - t0) * 1e3
    assert m["ms"] < 50.0, f"FNO inference {m['ms']:.1f} ms"
    assert e2e_ms < 50.0, f"FNO end-to-end {e2e_ms:.1f} ms"


# --- WebSocket: refine, train_pinn, cancel -----------------------------------


def test_refine(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "refine",
                "id": "r1",
                "u0": {"b64": U0_B64, "n": NX},
                "steps": 40,
                "grid": {"nx": NX, "nt": 21},
            }
        )
        msgs = _drain_until(ws, "done", limit=80)
    types = [m["type"] for m in msgs]
    assert "progress" in types and "done" in types
    prog = next(m for m in msgs if m["type"] == "progress")
    assert prog["task"] == "refine" and "preview" in prog


def test_train_pinn_and_cancel(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {"type": "train_pinn", "id": "tp", "u0": {"b64": U0_B64, "n": NX}, "config": "fast"}
        )
        saw_progress = False
        for _ in range(60):
            m = ws.receive_json()
            if m["type"] == "progress" and m["task"] == "train_pinn":
                saw_progress = True
                ws.send_json({"type": "cancel", "id": "tp"})
                break
        assert saw_progress
        done = None
        for _ in range(20):
            m = ws.receive_json()
            if m["type"] == "done":
                done = m
                break
        assert done is not None and done.get("task") == "train_pinn"


def test_malformed_message_reports_error(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "nonsense", "id": "x"})
        m = ws.receive_json()
        assert m["type"] == "error"
