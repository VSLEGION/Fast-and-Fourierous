"""FastAPI app: REST endpoints + the /ws protocol, serving the built frontend in prod.

REST (CLAUDE.md §6.6):
    GET /api/health   -> status, device, loaded models
    GET /api/presets  -> the UI preset definitions
    GET /api/models   -> loaded checkpoints with their metrics.json

Everything real-time goes over /ws (see ws.py). In production the built frontend
(web/dist) is served at / so the whole app runs offline from one origin.
"""

from __future__ import annotations

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from shocklab.data.datasets import load_test
from shocklab.server.presets import PRESETS
from shocklab.server.registry import Registry
from shocklab.server.schemas import encode_array
from shocklab.server.truth_cache import TruthCache, _resample_periodic_1d
from shocklab.server.ws import Connection
from shocklab.train.common import REPO_ROOT


def create_app(registry: Registry | None = None) -> FastAPI:
    app = FastAPI(title="Shock Lab", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # local classroom demo, single machine
        allow_methods=["*"],
        allow_headers=["*"],
    )
    reg = registry or Registry()
    cache = TruthCache()
    app.state.registry = reg
    app.state.truth = cache

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "device": str(reg.device), "models": reg.available()}

    @app.get("/api/presets")
    def presets() -> dict:
        return {"presets": PRESETS}

    @app.get("/api/models")
    def models() -> dict:
        return reg.info()

    @app.get("/api/heldout/{index}")
    def heldout(index: int, nx: int = 256) -> dict:
        """The initial condition of held-out test sample ``index`` on an nx grid."""
        test = load_test()  # (200, 51, 1024)
        i = index % test["u"].shape[0]
        u0 = test["u"][i, 0, :]
        if nx != u0.shape[0]:
            u0 = _resample_periodic_1d(u0, nx)
        return {"index": i, "u0": {"b64": encode_array(u0), "n": nx}}

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        await Connection(websocket, reg, cache).run()

    dist = REPO_ROOT / "web" / "dist"
    if dist.exists():  # built frontend served at / in production (offline)
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="static")

    return app


app = create_app()
