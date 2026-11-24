"""Run the Shock Lab server: python -m shocklab.server  (serves http://localhost:8000)."""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    p = argparse.ArgumentParser(description="Run the Shock Lab FastAPI server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    # import string so uvicorn owns the app lifecycle; models load on startup
    uvicorn.run("shocklab.server.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
