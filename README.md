# Shock Lab — Burgers' shocks, from a PINN to an FNO to a PINO

An interactive, presentation-grade web app comparing three ways to solve the same
PDE — viscous **Burgers' equation** on a periodic domain:

$$u_t + u\,u_x = \nu\,u_{xx}, \qquad x \in [-1, 1)\ \text{periodic}, \quad t \in [0, 1], \quad \nu = \tfrac{0.01}{\pi}$$

| Model | What it does | Classroom point |
|---|---|---|
| **PINN** | learns *one* solution from the physics alone (no data) | sharp at the shock, but slow and single-use |
| **FNO** | learns the *solution operator*: any IC → full evolution in ~ms | fast, resolution-invariant, needs data |
| **PINO** | an FNO also trained on the PINN's physics loss | lower physics residual; refine-able on new inputs |

You pick or draw an initial condition whose true answer we know; every model
predicts in real time, and the UI makes their differences obvious — accuracy at
the shock, physics consistency, speed, resolution invariance, and out-of-distribution
behaviour.

---

## Measured results (reproduced, not hand-typed)

Every number comes from code and is stored in `checkpoints/<model>/metrics.json`
with its config, git hash and GPU. On this machine (RTX 5070 Ti, Python 3.13,
`torch 2.11.0+cu128`):

| | PINN (classic) | FNO | PINO |
|---|---|---|---|
| test rel L2 | **4.8e-4** (classic) | **0.54 %** @256 | 1.0 % @256 |
| zero-shot @1024 | — | **0.54 %** (ratio 0.998) | 1.0 % |
| PDE residual vs FNO | — | 1× (6.7× floor) | **5× lower** (1.3× floor) |
| inference | ~ms | **p50 3 ms** / p95 3.5 ms | 3–4 ms |
| train time | 22 min (fast 64 s) | 31 min | 110 min |

- **Solver vs exact (Cole–Hopf), N=2048:** rel L2 ≤ 1.4e-10.
- **FNO translation-equivariance:** shifted input → shifted output to float tolerance.
- **PINO "Refine with physics"** on OOD presets: boosted 84 % → 16 %, linear 0.45 % → 0.07 %.

---

## Setup (Windows / PowerShell)

```powershell
.\scripts\setup.ps1        # venv + deps + PyTorch (cu128) + npm install
```

Or manually: `python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt`,
then install PyTorch for your CUDA version (see `scripts/setup.ps1`), then `cd web; npm install`.

## Generate data, train, run

```powershell
python -m shocklab.data.generate --out data          # the 3 .npz files (CLAUDE.md §3)

python -m shocklab.train.pinn --config configs/pinn_classic.yaml
python -m shocklab.train.fno  --config configs/fno.yaml
python -m shocklab.train.pino --config configs/pino.yaml

.\scripts\run_demo.ps1                                # build web + serve at http://localhost:8000
```

`run_demo.ps1` builds the frontend into `web/dist` and starts the FastAPI server,
which serves the whole app (UI + API + WebSocket) **from one origin, fully offline**
— no CDN fonts or scripts at runtime. For development with hot reload:
`python -m shocklab.server` and, in another shell, `cd web; npm run dev` (proxies to :8000).

## Demo script (presenter mode)

Open **http://localhost:8000**, press **Present ▸**, and step the story with **← / →**:

1. **Classic shock** `u₀ = −sin(πx)` — all three models vs the exact answer.
2. **PINN** — learned one solution from physics; razor-sharp at the shock.
3. **FNO** — maps any IC to the full field in ~3 ms.
4. **Draw / Gaussian pulse** — a new IC, solved instantly.
5. **Shift** — the FNO is *exactly* translation-equivariant.
6. **Boost** (nonzero mean) — out of distribution; the operator struggles.
7. **Refine with physics** — no data, just the PDE residual; the shock snaps into place.
8. **Cost** — train once, query forever vs retrain-every-time.

**Shortcuts:** `1–9` presets · `Space` play/pause · `R` refine · `T` train PINN ·
`D` theme · `E` export PNG · `←/→` story · `Esc` exit. Default theme is **Paper**
(for projectors); toggle **Studio** for a dark room.

---

## Project layout

```
shocklab/
  physics/   solver (ETDRK4), exact (Cole–Hopf), ic (GRF + presets), invariants
  data/      generate (CLI), datasets (splits)
  models/    pinn, fno (SpectralConv2d/FNO2d), physics_loss, layers
  train/     pinn, fno, pino (+ refine), common (seeds/ckpt/timers)
  eval/      metrics, benchmark (latency), ood
  server/    app (FastAPI), ws (protocol), registry, truth_cache, presets, scoring, schemas
web/         Vite + React + TypeScript + WebGL2 frontend
configs/     pinn_classic, pinn_fast, fno, pino
checkpoints/ <model>/{model.pt, config.yaml, metrics.json, train_log.csv, figures/}  (git-ignored)
tests/       test_physics, test_models, test_metrics, test_api
01..04_*.ipynb   step-by-step notebooks (import shocklab)
```

## References

- Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks*, J. Comput. Phys. 378 (2019).
- Li et al., *Fourier Neural Operator for Parametric PDEs*, ICLR 2021.
- Cox & Matthews (2002); Kassam & Trefethen (2005): the ETDRK4 reference solver.

The visual style is *inspired by* PhysicsX's published design language; it uses none
of their name, logo, fonts, or images.
