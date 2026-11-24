# CLAUDE.md: Shock Lab (Burgers' shocks: PINN → FNO → PINO)

You are building **Shock Lab**, an interactive, presentation-grade web app for a university class demo. It compares three ways of solving the same PDE:

1. **PINN.** Learns *one* solution from the physics alone.
2. **FNO.** Learns the *solution operator* from data: any initial condition in, the full evolution out, in milliseconds.
3. **PINO.** An FNO also trained on the PINN's physics loss, which can be refined on a new input in seconds.

In the app, the user picks or draws an initial condition whose true answer we know. Every model predicts in real time. The UI makes their differences obvious: accuracy at the shock, physics consistency, speed, resolution invariance, and what happens out of distribution.

The owner is an aerospace engineering student presenting this in class. Accuracy and honesty matter more than flash, and the app should still look excellent.

---

## 0. Ground rules (read first)

- **The user's chat instructions override this file.** If something here is unclear, ask, then update this file.
- **Work milestone by milestone (§9).** At the end of each milestone: run the tests, report the *measured* numbers, and stop for review. Don't start the next milestone unasked.
- **Never fake or hand-type results.** Every number shown in the UI or in reports comes from code that computed it, stored in `checkpoints/<model>/metrics.json` together with the config and git hash. If a model underperforms, say so and propose fixes. Never tweak the truth, the metrics, or the test set to make it look better.
- **The test set is held out.** Use it only for final evaluation and as UI presets. Never train, tune or select checkpoints on it; use the validation split (§3) for that.
- **The physics core is verified. Don't change its numerics** (solver, constants, grid conventions) without re-running `tests/test_physics.py` and telling the user.
- **Windows + NVIDIA GPU machine.** Use `pathlib` everywhere and PowerShell for scripts. Use `num_workers=0` in DataLoaders (or proper `if __name__ == "__main__":` guards). Avoid symlinks and bash-only tooling.
- **Must run offline** in a classroom: no CDN fonts or scripts at runtime. Bundle everything.
- **Branding.** The visual style is *inspired by* PhysicsX's published design language (§7). Never use their name, logo, proprietary fonts, or copy their images in the app.

---

## 1. The physics: fixed and verified

Viscous Burgers' equation on a **periodic** domain:

```
u_t + u u_x = ν u_xx,   x ∈ [-1, 1) periodic,   t ∈ [0, 1],   ν = 0.01/π ≈ 3.183e-3
```

- **Grid convention:** `x_j = -1 + 2j/N`, `j = 0..N-1` (left-closed, no duplicate endpoint). Wavenumbers are `k = π·n`.
- **Classic case:** `u0 = -sin(πx)`. Its shock forms at `x = 0` just after the breaking time `t_b = 1/π ≈ 0.318`. Shock thickness is `~2ν/|u| ≈ 0.006`. This is the Raissi–Perdikaris–Karniadakis (2019) PINN benchmark.
- **Exact solution (classic case):** the Cole–Hopf integral, evaluated with log-sum-exp. It's implemented in `01_physics_and_data.ipynb` as `burgers_exact_sine`.
- **Reference solver ("truth" for any IC):** pseudo-spectral ETDRK4 (Cox–Matthews; Kassam–Trefethen contour coefficients, M = 32), with the 2/3 dealiasing rule. It's implemented in `01_physics_and_data.ipynb` as `burgers_solve`.
- **Measured accuracy of the solver** (relative L2 vs exact, classic case):

  | N | dt | t=0.25 | t=0.5 | t=0.75 | t=1.0 | CPU time |
  |---|---|---|---|---|---|---|
  | 1024 | 5e-4 | 2.8e-12 | 3.3e-6 | 9.9e-7 | 1.1e-7 | ~0.2 s |
  | 2048 | 4e-4 | 1.1e-12 | 8.3e-11 | 1.4e-10 | 5.6e-11 | ~0.35 s |

  On random ICs, N = 2048 vs 4096 agree to ~2e-6. Solve at **N = 2048, dt = 4e-4** and subsample for storage.
- **Resolution rule for other ν** (truth only): `N ≥ next_pow2(4/ν)`, capped at 8192. `dt` must keep `max|u|·k_max_dealiased·dt ≲ 0.9`. Check with a self-convergence test (N vs 2N) in tests.
- **Invariants for physics checks:**
  - Mean ∫u dx is conserved exactly.
  - Energy E(t) = ½∫u² dx is non-increasing.
  - Translation: `u0(x−s) → u(x−s, t)`.
  - Galilean boost: `u0 + c → u(x − c t, t) + c`.
  - Small amplitude ε: `u ≈ ε e^{-ν π² t} sin(πx)` for `u0 = ε sin(πx)` (linear heat-equation limit).

---

## 2. Current repo state

```
Burgers_PINN_to_FNO/
  01_physics_and_data.ipynb   # verified solver, exact solution, GRF ICs, dataset generation (Step 1)
  README.md
  CLAUDE.md                   # this file
```

`data/` doesn't exist yet. It's produced by notebook 01 or by the ported generator (M1).

---

## 3. Data contract (from notebook 01; keep it identical)

| File | `u` shape | dtype | Grid | Snapshots | Notes |
|---|---|---|---|---|---|
| `data/burgers_train.npz` | (1000, 51, 256) | float32 | 256 | t = linspace(0,1,51) | seed 0 |
| `data/burgers_test.npz` | (200, 51, 1024) | float32 | 1024 | 51 | seed 1, **held out** |
| `data/burgers_classic.npz` | (101, 1024) | float32 | 1024 | t = linspace(0,1,101) | `-sin(πx)` |

- Each file also stores `t`, `x` and `nu`. `u[:, 0, :]` is the initial condition.
- **Initial conditions:** Gaussian random field with mode-n std = `sqrt(625/((πn)² + 25)²)` (FNO paper's `625(−Δ+25I)^{-2}` on our domain). Zero mean. Each sample is rescaled so its peak `|u0|` ~ U[0.5, 1.0]. The classic `-sin(πx)` is in-distribution.
- **Validation split:** the last 100 of the 1,000 training samples (indices 900–999). Train on 0–899.

---

## 4. Target architecture

```
Burgers_PINN_to_FNO/
  CLAUDE.md  README.md  pyproject.toml  requirements.txt  .gitignore
  01_physics_and_data.ipynb          # keep; later notebooks import from shocklab
  configs/        pinn_classic.yaml  pinn_fast.yaml  fno.yaml  pino.yaml
  shocklab/
    physics/      solver.py (ETDRK4, ported verbatim)  exact.py (Cole–Hopf)  ic.py (GRF + presets)
                  invariants.py (mean, energy, residual, shock locator, symmetry transforms)
    data/         generate.py (CLI port of notebook section 5)  datasets.py
    models/       pinn.py  fno.py (SpectralConv2d, FNO2d)  layers.py
    train/        pinn.py  fno.py  pino.py  common.py (logging, ckpt, seeds, timers)
    eval/         metrics.py  benchmark.py  ood.py
    server/       app.py (FastAPI)  ws.py  registry.py  schemas.py  truth_cache.py  presets.py
  checkpoints/    <model>/{model.pt, config.yaml, metrics.json, train_log.csv}   (git-ignored)
  web/            Vite + React + TypeScript frontend (§6–7)
  scripts/        setup.ps1  train_all.ps1  run_demo.ps1  bench_latency.py
  tests/          test_physics.py  test_models.py  test_metrics.py  test_api.py
```

**Stack.**
- **Python:** 3.11, PyTorch with CUDA, numpy, scipy, FastAPI, uvicorn[standard], pydantic v2, orjson, PyYAML, pytest, ruff.
- **Frontend:** Node 20+, Vite, React 18, TypeScript (strict), Zustand (state), d3-scale / d3-axis / d3-shape (axes and lines only), raw **WebGL2** for heatmaps (colormap LUT texture), KaTeX (equations), Framer Motion (subtle transitions), `@fontsource` packages for offline fonts.
- **Install PyTorch** with the index URL for the machine's CUDA driver. Check `nvidia-smi` and https://pytorch.org/get-started/locally/; don't guess the CUDA version.

---

## 5. Models: specs and acceptance targets

The targets below are goals, not promises. Report what you measure.

### 5.1 PINN, "fundamental" (Raissi-style), classic case only

- **Network:** `u_θ(x,t)` is an MLP with inputs (x, t), 8 hidden layers × 20 units, tanh, Xavier init (Raissi used 9 × 20).
- **Loss:**
  - PDE residual `r = u_t + u u_x − ν u_xx` via `torch.autograd` on N_f = 20,000 collocation points (uniform random, resampled every 1,000 iterations).
  - IC loss on 256 points.
  - Soft periodic BC: `u(−1,t)=u(1,t)` and `u_x(−1,t)=u_x(1,t)` on 100 t-points.
- **Optimizers:** Adam (lr 1e-3, ~10k iterations), then L-BFGS (`torch.optim.LBFGS`, strong-Wolfe line search) until converged or 5k iterations. Run in **float32**, then L-BFGS in **float64** if it stalls.
- **Evaluate** on the 1024 × 101 grid against `burgers_classic.npz`. Report relative L2, L∞, shock-position error for t > t_b, and wall-clock training time.
- **Targets:** rel L2 ≤ 1e-2 (stretch ≤ 3e-3). The error map should show the residual error concentrated at the shock. That's the point of the demo.
- **Fast config** (`pinn_fast.yaml`) for live retraining in the UI: ≤ ~90 s on the GPU, accuracy reported honestly.
- **Stretch variants (each is a neat comparison, toggleable):**
  - Hard periodicity via Fourier-feature input `[sin(mπx), cos(mπx)]`.
  - Hard IC via `u = u0(x) + t·N(x,t)`.
  - Residual-based adaptive sampling.

### 5.2 FNO, space–time FNO-2D (data only)

- **Input channels** on the (x, t) grid:
  - `u0(x)` broadcast over t
  - `t`
  - **Deliberately no x-coordinate channel.** All operations are then periodic convolutions plus pointwise maps, so the model is exactly **translation-equivariant**, which the UI demonstrates. Document this choice in code comments.
- **Architecture:** lift P (1×1 conv to width 64), then 4 Fourier layers `v ← GELU(W v + F⁻¹(R·F v))` with modes (k_x = 32, k_t = 16), then project Q (64 → 128 → 1).
  - **x is periodic. t is not:** pad t by ~25% with zeros before the FFT and crop after.
  - Keep the spectral convolution in **float32/complex64**; don't autocast FFTs to half precision.
- **Training:** relative-L2 loss over the full space–time field, Adam lr 1e-3 with cosine decay, ~500 epochs, batch size 20, on 256 × 51. Select the checkpoint on validation.
- **Evaluate:**
  - Test set at 256 (subsample the 1024 test data).
  - Zero-shot at 1024 in x: discretization invariance.
  - Zero-shot at 101 in t, against the classic file.
  - Inference latency (p50/p95, batch 1, after warmup).
- **Targets:** test rel L2 ≤ 3% at 256. Similar error at 1024 (report the ratio). Inference ≤ 10 ms on the GPU.

### 5.3 PINO: the FNO plus physics

- **Model:** same architecture as §5.2, initialized from the FNO checkpoint (or from scratch; report both if time allows).
- **Loss:** `L = L_data + λ_pde·L_pde + λ_ic·L_ic`.
  - **L_pde:** the residual evaluated on a *finer* time grid (256 × 101, querying the model there). u_x and u_xx are computed spectrally in x; u_t uses second-order finite differences in t (one-sided at the ends). Normalize it by the truth's own residual level so the FD error isn't penalized.
- **Test-time "Refine with physics"** (instance-wise fine-tuning): for one new IC, start from the PINO weights and minimize `L_pde + L_ic` only (no data) for 100–500 Adam steps, streaming progress. This is the bridge between the PINN (physics, slow) and the FNO (data, fast). It must not modify the saved checkpoint.
- **Targets:** a much lower PDE residual than the FNO at similar or better data error. Refinement improves OOD presets (§6.3). Report honestly if it doesn't.

### 5.4 Stretch (only if asked)

- A U-Net or MLP grid baseline that *fails* zero-shot super-resolution (to contrast with the FNO).
- A DeepONet.
- A ν-conditioned FNO.

---

## 6. App behavior

### 6.1 Real-time flow

1. The user changes the IC (preset, sliders, symmetry tools, or drawing; debounce ~40 ms).
2. The client sends `simulate` over WebSocket.
3. The server returns **model predictions immediately** (target < 50 ms end-to-end for FNO/PINO).
4. **Truth** follows when ready (threadpool, LRU-cached by IC hash and N). Exact or stored truths are instant.
5. The server sends **metrics** once truth is available. Panels fade in; nothing blocks the UI.

### 6.2 Truth sources (always shown as a chip on the Truth panel)

| Chip | Meaning |
|---|---|
| `EXACT` | Cole–Hopf, for the classic case |
| `SYMMETRY` | Derived exactly from a known solution: shift `u(x−s,t)`, or Galilean boost `u(x−ct,t)+c` |
| `ANALYTIC` | Linear limit `ε e^{-νπ²t} sin(πx)`, also cross-checked with the solver |
| `HELD-OUT` | A stored test-set sample |
| `SOLVER` | ETDRK4 at verified resolution, for anything else (drawn ICs etc.) |

### 6.3 Presets: "values we know the answer to"

1. Classic `-sin(πx)` (EXACT). Every model, including the PINN.
2. Shifted classic `-sin(π(x−s))` with an s slider (SYMMETRY). Shows the FNO's exact equivariance; the PINN would need retraining.
3. Boosted classic `-sin(πx) + c` with a c slider (SYMMETRY). **OOD:** the training ICs have zero mean. The FNO is expected to struggle; then try "Refine with physics".
4. Linear limit `ε sin(πx)`, ε = 0.05 (ANALYTIC). **OOD amplitude.**
5. Two-wave `A sin(πx+φ₁) + B sin(2πx+φ₂)`, with parametric sliders (SOLVER).
6. Smoothed N-wave / sawtooth (SOLVER).
7. Zero-mean Gaussian pulse (SOLVER).
8. Held-out test samples, browsable by index (HELD-OUT).
9. Hand-drawn (SOLVER). The drawing is projected to a periodic, low-pass curve (keep modes ≤ 32). A toggle enforces zero mean, and the amplitude is clamped to [−1.2, 1.2].

**A ν slider affects the Truth only.** The models were trained at a fixed ν and don't take ν as input. Show this as a lesson: "the operator was learned for one ν", with an OOD badge.

### 6.4 OOD meter

Shown next to the IC, with plain-language reasons. It scores the IC against the training distribution:
- peak amplitude vs [0.5, 1.0]
- mean vs 0
- the spectrum: per-mode z-scores against the GRF std, and the energy fraction above mode 16

Output is `IN-DISTRIBUTION`, `NEAR`, or `OUT-OF-DISTRIBUTION`.

### 6.5 Metrics (per model, computed server-side vs truth on a common grid)

- **Accuracy:** relative L2 (space–time), L∞, and shock-position error.
- **Physics score:** the normalized PDE residual. Also mean-conservation drift and energy-monotonicity violations.
- **Speed:** inference ms, plus training cost from `metrics.json`.
- **PINN on a non-classic IC:** show `RETRAIN REQUIRED` with a "Train PINN on this IC (~N s)" button. This streams the live training (loss curves plus the field morphing) over WebSocket.

### 6.6 WebSocket protocol (`/ws`)

```jsonc
// client → server
{"type":"simulate","id":"uuid","u0":{"b64":"...float32 LE...","n":256},"nu":0.0031831,
 "grid":{"nx":256,"nt":101},"models":["fno","pino","pinn"],"preset":"classic|null"}
{"type":"train_pinn","id":"uuid","u0":{...},"config":"fast"}
{"type":"refine","id":"uuid","u0":{...},"steps":300}
{"type":"cancel","id":"uuid"}
// server → client
{"type":"prediction","id":"...","model":"fno","field":{"b64":"...","nx":256,"nt":101},"ms":3.1}
{"type":"truth","id":"...","source":"EXACT|SYMMETRY|ANALYTIC|HELD-OUT|SOLVER","field":{...},"ms":340}
{"type":"metrics","id":"...","per_model":{"fno":{"rel_l2":0.021,"linf":0.18,"shock_dx":0.004,"residual":1.7,"mass_drift":2e-6,"energy_ok":true}}}
{"type":"ood","id":"...","level":"NEAR","reasons":["amplitude 1.2 > 1.0"]}
{"type":"progress","id":"...","task":"train_pinn|refine","iter":1200,"losses":{"pde":..,"ic":..,"bc":..},"preview":{...}}
{"type":"done"|"error","id":"...", ...}
```

- Arrays travel as base64 little-endian float32. Fields are `[nt][nx]` row-major.
- REST is used only for `GET /api/health`, `GET /api/presets` and `GET /api/models` (loaded checkpoints with their `metrics.json`).
- Only one long task (train or refine) runs at a time. Make it cancellable, and never block the event loop: use a thread or `asyncio.to_thread`.

---

## 7. Visual design: PhysicsX-inspired, our own brand

**Design language** (studied from physicsx.ai and their FNO article):
- Monochrome, with lots of white space and hairline (1 px) strokes.
- A neutral grotesk at **weight 400** with slightly negative tracking on headings.
- **Monospace uppercase micro-labels** with ~+2% letter-spacing, e.g. `01 — INITIAL CONDITION`, `REL L2`, `INFER MS`.
- Diagram panels in a **warm off-white** with thin dark linework and LaTeX math.
- Dark cinematic sections with an **orange → violet** glass/gradient accent.
- It reads like an engineering instrument, not a dashboard template.

**Themes** (toggle; default **Paper** for projectors):

| Token | Paper (light) | Studio (dark) |
|---|---|---|
| `--bg` | `#F7F7F5` | `#0B0B0C` |
| `--panel` | `#EEEEE8` (warm diagram panel) | `#151515` |
| `--surface` | `#FFFFFF` | `#1E1E1E` |
| `--ink` | `#1E1E1E` | `#F2F2F0` |
| `--muted` | `#6B6B66` | `#9A9A94` |
| `--hairline` | `rgba(30,30,30,.14)` | `rgba(255,255,255,.12)` |
| `--accent-grad` | `#FF7A1A → #7B5CFF` (used sparingly: hero, active states, progress) | same |

- **Type:** "Inter Tight" (or Geist) for UI and display, "JetBrains Mono" (or Geist Mono) for labels and numbers. Both come from `@fontsource`, offline. Numbers use tabular figures.
- **Shape:** panel radius 12 px, pills 999 px, 1 px borders, no drop shadows (at most a 1 px inner hairline), 8 px spacing grid.
- **Model colors** (colorblind-checked; also distinguished by dash style and labels):

  | Model | Color | Line |
  |---|---|---|
  | Truth | `--ink` | solid 2 px |
  | FNO | `#FF6B1A` | solid |
  | PINO | `#00A3A3` | solid |
  | PINN | `#7B5CFF` | solid |
  | Baseline | `#8A8A85` | dashed |

- **Colormaps:**
  - The field `u` uses a perceptually uniform **diverging** map centered on 0 (e.g. cmocean `balance` or `RdBu_r`, exported to a 256-entry LUT in `web/src/viz/colormaps.ts`).
  - Errors use a **sequential log** map (magma/inferno-like) with a shared range across models.
  - Colorbars are thin, with mono tick labels.
- **Motion:** 150–250 ms ease-out. Numbers tween, panels cross-fade. No bounce. Respect `prefers-reduced-motion`.

**Layout** (12-column grid; desktop 1440 × 900 and a projector at 1920 × 1080 must both look great):

```
┌ Top bar: SHOCK LAB · equation in KaTeX · theme toggle · presenter mode ─────────────────┐
├ Left rail (IC lab) ──────┬ Main stage ──────────────────────────────────────────────────┤
│ 01 INITIAL CONDITION     │ 02 SPACE–TIME FIELDS  [Truth][FNO][PINO][PINN] small multiples│
│  presets · sliders ·     │    shared colorbar · synced crosshair · characteristics overlay│
│  draw canvas · symmetry  │ 03 ERROR |model − truth| (log, shared scale)                   │
│  ν (truth only) · grid N │ 04 SNAPSHOT u(x, t*) all models + Shock Lens (auto-zoom inset) │
│  OOD meter + reasons     │    time scrubber ▶ · speed · t_b marker                        │
│                          │ 05 SPECTRUM |û_k| log-log at t*, FNO k_max shaded              │
│                          │ 06 METRICS cards per model · physics score · speed             │
└──────────────────────────┴───────────────────────────────────────────────────────────────┘
```

**Signature visualizers** (build these well; they carry the demo):
1. **Space–time heatmaps** (t horizontal, x vertical, like the classic PINN figure). Rendered in WebGL2 from a float texture plus a LUT, at 60 fps. Hovering shows a synced crosshair and values for every model.
2. **Characteristics overlay** on Truth: inviscid characteristics `x0 + u0(x0)t`, drawn until the breaking time `t_b = −1/min u0'`, which is marked. This shows *why* the shock forms.
3. **Shock Lens:** an auto-detected (max |u_x|) magnified inset on the snapshot plot. It shows the PINN smearing, FNO Gibbs wiggles, and how sharp PINO is.
4. **Spectrum panel:** shows energy beyond the FNO's mode cutoff, tying back to the "where FNO falls short" story.
5. **Resolution sweep:** evaluates the same IC at 64…2048 and plots error vs N (FNO flat; baseline rising, if built).
6. **Cost-amortization chart:** cumulative wall-clock vs number of queries for PINN (retrain per query) vs FNO/PINO (one-time training + ms per query), with a crossover marker. Uses *measured* times from `metrics.json`.
7. **Live training view:** log-scale loss curves (PDE / IC / BC) and the field morphing, for "Train PINN" and "Refine with physics".

**Presenter mode:**
- Larger type, hidden chrome, and a **Story** sequence (← / → to step): classic case → PINN → FNO instant → draw your own → shift (equivariance) → boost (OOD) → refine with physics → resolution sweep → cost chart.
- Shortcuts: `1–9` presets, `Space` play/pause, `R` refine, `T` train PINN, `D` toggle theme, `E` export PNG of the current view.

**Accessibility:** keyboard-navigable controls, aria-labels on canvases (with a text summary of the metrics), and contrast at WCAG AA or better.

---

## 8. Commands (PowerShell)

```powershell
# setup
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # then install torch for this CUDA version (see §4)
cd web; npm install; cd ..

# data (same output as notebook 01)
python -m shocklab.data.generate --out data

# train
python -m shocklab.train.pinn --config configs/pinn_classic.yaml
python -m shocklab.train.fno  --config configs/fno.yaml
python -m shocklab.train.pino --config configs/pino.yaml

# run
python -m shocklab.server                 # http://localhost:8000 (serves web/dist in prod)
cd web; npm run dev                        # dev server with proxy to :8000
.\scripts\run_demo.ps1                     # build web + start server (class demo)

# quality
pytest -q;  ruff check .;  cd web; npm run lint; npm run typecheck
python scripts/bench_latency.py            # p50/p95 per model
```

---

## 9. Milestones and definition of done

| # | Milestone | Done when |
|---|---|---|
| M0 | Repo scaffold, env, `.gitignore` (data/, checkpoints/, node_modules/, .venv/), `git init` | `python -c "import torch; print(torch.cuda.is_available())"` prints True; ruff and pytest run |
| M1 | Physics core: port solver, exact solution and GRF **verbatim** from notebook 01; `invariants.py`; `data/generate.py` | `test_physics.py`: solver vs exact at N=2048 rel L2 < 1e-8 for t ∈ {0.25, 0.5, 0.75, 1}; N vs 2N self-convergence; mean conserved < 1e-10; energy non-increasing; shift and boost symmetry checks; generated data matches the §3 shapes |
| M2 | PINN classic plus fast config | `metrics.json` with rel L2, L∞, shock Δx and train time; figures saved; target in §5.1 met or a written explanation of the gap |
| M3 | FNO-2D | Test rel L2 at 256 and at 1024 (zero-shot), latency p50/p95; translation-equivariance test passes (shifted input gives shifted output, to float tolerance) |
| M4 | PINO plus test-time refine | Residual and data error vs FNO reported; refine improves the boosted and linear-limit presets (or an honest report) |
| M5 | Server: registry, WebSocket protocol, truth cache, presets, OOD, metrics | `test_api.py` covers every message type; FNO end-to-end < 50 ms locally; long tasks are cancellable |
| M6 | Frontend: layout, themes, the 7 signature visualizers, IC lab, presenter mode | Manual QA screenshots of every preset in both themes at 1440 × 900 and 1920 × 1080; no console errors; 60 fps scrubbing |
| M7 | Polish: export PNG, story mode, offline check (unplug the network), README demo script, notebooks 02–04 as thin wrappers importing `shocklab` | Complete run-through in presenter mode with the network off |

---

## 10. Conventions

- **Python:** type hints everywhere. Docstrings include the governing equation in plain text. Use `ruff format`. Keep modules small. Configs are dataclasses or YAML. Seed everything (`torch`, `numpy`, `random`), and use deterministic eval.
- **Precision:** physics and truth in **float64**; models in **float32** (TF32 allowed for training, not for spectral layers); metrics computed in float64.
- **Logging:** CSV per run (`iter, time, losses…`), plus `metrics.json` (config, git hash, GPU name, torch version, timings).
- **Frontend:** TypeScript strict. No chart library for heatmaps. One `viz/` module per visualizer, and pure functions for scales and colormaps. Wire the WebSocket protocol types in `web/src/api/types.ts` to match `server/schemas.py` (keep them in sync).
- **No silent fallbacks.** If CUDA isn't available, the model checkpoint is missing, or truth is still computing, show it in the UI.
- **Commits:** one logical change each, with a message naming the milestone (e.g. `M3: add zero-shot 1024 eval`).
