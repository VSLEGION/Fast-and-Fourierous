"""Train the 'fundamental' PINN on the classic Burgers case u0 = -sin(pi x).

Pipeline (CLAUDE.md §5.1):
  1. Adam (lr 1e-3, ~10k iters) on L = w_pde*L_pde + w_ic*L_ic + w_bc*L_bc,
     with collocation points resampled every ``resample_every`` iterations.
  2. L-BFGS (strong Wolfe) on fixed collocation points, optionally in float64.

Then evaluate on the 1024 x 101 grid against data/burgers_classic.npz: relative
L2, L-infinity, shock-position error for t > t_b, and wall-clock training time;
save model.pt, config.yaml, metrics.json, train_log.csv and figures.

Usage:
    python -m shocklab.train.pinn --config configs/pinn_classic.yaml
    python -m shocklab.train.pinn --config configs/pinn_fast.yaml --out checkpoints/pinn_fast
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from shocklab.eval import metrics
from shocklab.models.pinn import PINN
from shocklab.physics.solver import NU, x_grid
from shocklab.train.common import REPO_ROOT, CSVLogger, Timer, get_device, save_metrics, set_seed

# The PDE residual uses second derivatives; keep full float32 precision (no TF32).
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@dataclass
class PINNConfig:
    name: str = "pinn_classic"
    seed: int = 0
    device: str = "cuda"
    n_hidden_layers: int = 8
    width: int = 20
    n_collocation: int = 20000
    n_ic: int = 256
    n_bc: int = 100
    resample_every: int = 1000
    w_pde: float = 1.0
    w_ic: float = 1.0
    w_bc: float = 1.0
    adam_iters: int = 10000
    adam_lr: float = 1.0e-3
    lbfgs_iters: int = 5000
    lbfgs_history: int = 50
    lbfgs_float64: bool = True
    log_every: int = 100

    @classmethod
    def from_yaml(cls, path: Path) -> PINNConfig:
        d = yaml.safe_load(Path(path).read_text())
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def pde_residual(model: PINN, x: torch.Tensor, t: torch.Tensor, nu: float) -> torch.Tensor:
    """r = u_t + u u_x - nu u_xx via autograd (x, t must require grad)."""
    u = model(x, t)
    u_t = _grad(u, t)
    u_x = _grad(u, x)
    u_xx = _grad(u_x, x)
    return u_t + u * u_x - nu * u_xx


def sample_collocation(n: int, device: torch.device, dtype: torch.dtype):
    x = (torch.rand(n, 1, device=device, dtype=dtype) * 2 - 1).requires_grad_(True)  # [-1, 1)
    t = torch.rand(n, 1, device=device, dtype=dtype).requires_grad_(True)  # [0, 1]
    return x, t


def compute_losses(model, coll, ic, bc, nu):
    cx, ct = coll
    r = pde_residual(model, cx, ct, nu)
    l_pde = (r**2).mean()

    ic_x, ic_t, ic_u = ic
    l_ic = ((model(ic_x, ic_t) - ic_u) ** 2).mean()

    xm1, xp1, t_bc = bc
    um, up = model(xm1, t_bc), model(xp1, t_bc)
    um_x, up_x = _grad(um, xm1), _grad(up, xp1)
    l_bc = ((um - up) ** 2).mean() + ((um_x - up_x) ** 2).mean()
    return l_pde, l_ic, l_bc


def make_fixed_points(cfg: PINNConfig, device: torch.device, dtype: torch.dtype):
    """IC and BC point sets (fixed throughout training)."""
    xi = torch.tensor(x_grid(cfg.n_ic).reshape(-1, 1), device=device, dtype=dtype)
    ic = (xi, torch.zeros_like(xi), -torch.sin(torch.pi * xi))
    t_bc = torch.linspace(0, 1, cfg.n_bc, device=device, dtype=dtype).reshape(-1, 1)
    xm1 = (-torch.ones_like(t_bc)).requires_grad_(True)
    xp1 = torch.ones_like(t_bc).requires_grad_(True)
    bc = (xm1, xp1, t_bc)
    return ic, bc


def train(cfg: PINNConfig, outdir: Path) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    model = PINN(n_hidden_layers=cfg.n_hidden_layers, width=cfg.width).to(device)
    logger = CSVLogger(
        outdir / "train_log.csv",
        ["phase", "iter", "time_s", "loss", "loss_pde", "loss_ic", "loss_bc"],
    )
    nu = NU

    # ---- Phase 1: Adam (float32) --------------------------------------------
    ic, bc = make_fixed_points(cfg, device, torch.float32)
    coll = sample_collocation(cfg.n_collocation, device, torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.adam_lr)
    with Timer() as adam_timer:
        for it in range(1, cfg.adam_iters + 1):
            if it % cfg.resample_every == 0:
                coll = sample_collocation(cfg.n_collocation, device, torch.float32)
            l_pde, l_ic, l_bc = compute_losses(model, coll, ic, bc, nu)
            loss = cfg.w_pde * l_pde + cfg.w_ic * l_ic + cfg.w_bc * l_bc
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % cfg.log_every == 0 or it == 1:
                logger.log(
                    phase="adam",
                    iter=it,
                    time_s=round(adam_timer.elapsed(), 2),
                    loss=loss.item(),
                    loss_pde=l_pde.item(),
                    loss_ic=l_ic.item(),
                    loss_bc=l_bc.item(),
                )
    adam_seconds = adam_timer.seconds

    # ---- Phase 2: L-BFGS (optionally float64) -------------------------------
    dtype = torch.float64 if cfg.lbfgs_float64 else torch.float32
    if dtype == torch.float64:
        model = model.double()
    ic, bc = make_fixed_points(cfg, device, dtype)
    coll = sample_collocation(cfg.n_collocation, device, dtype)
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=cfg.lbfgs_iters,
        history_size=cfg.lbfgs_history,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-11,
        tolerance_change=1e-13,
    )
    state = {"n": 0, "last": (0.0, 0.0, 0.0)}

    def closure():
        opt.zero_grad()
        l_pde, l_ic, l_bc = compute_losses(model, coll, ic, bc, nu)
        loss = cfg.w_pde * l_pde + cfg.w_ic * l_ic + cfg.w_bc * l_bc
        loss.backward()
        state["n"] += 1
        state["last"] = (l_pde.item(), l_ic.item(), l_bc.item())
        if state["n"] % cfg.log_every == 0:
            logger.log(
                phase="lbfgs",
                iter=state["n"],
                time_s=-1,
                loss=loss.item(),
                loss_pde=state["last"][0],
                loss_ic=state["last"][1],
                loss_bc=state["last"][2],
            )
        return loss

    with Timer() as lbfgs_timer:
        if cfg.lbfgs_iters > 0:
            opt.step(closure)
    lbfgs_seconds = lbfgs_timer.seconds
    logger.close()

    # ---- Evaluate -----------------------------------------------------------
    results = evaluate(model, device)
    results.update(
        adam_seconds=round(adam_seconds, 2),
        lbfgs_seconds=round(lbfgs_seconds, 2),
        train_seconds=round(adam_seconds + lbfgs_seconds, 2),
        lbfgs_closure_calls=state["n"],
        num_params=model.num_params(),
        final_loss_pde=state["last"][0],
        final_loss_ic=state["last"][1],
        final_loss_bc=state["last"][2],
    )

    torch.save(model.state_dict(), outdir / "model.pt")
    save_metrics(outdir, asdict(cfg), results)
    make_figures(model, device, outdir)
    return results


@torch.no_grad()
def _predict_grid(model, device, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    dtype = next(model.parameters()).dtype
    X, T = np.meshgrid(x, t)  # (nt, nx)
    xx = torch.tensor(X.reshape(-1, 1), device=device, dtype=dtype)
    tt = torch.tensor(T.reshape(-1, 1), device=device, dtype=dtype)
    u = model(xx, tt).cpu().numpy().reshape(T.shape)
    return u


def evaluate(model, device) -> dict:
    d = np.load(REPO_ROOT / "data" / "burgers_classic.npz")
    x, t, truth = d["x"], d["t"], d["u"]  # truth: (101, 1024)
    pred = _predict_grid(model, device, x, t)
    return {
        "rel_l2": metrics.rel_l2(pred, truth),
        "linf": metrics.linf(pred, truth),
        "shock_dx": metrics.shock_position_error(pred, truth, x, t),
    }


def make_figures(model, device, outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    d = np.load(REPO_ROOT / "data" / "burgers_classic.npz")
    x, t, truth = d["x"], d["t"], d["u"]
    pred = _predict_grid(model, device, x, t)
    err = np.abs(pred - truth)
    t_b = 1.0 / np.pi
    ext = [t[0], t[-1], x[0], x[-1]]

    # 1. Field comparison: truth / pred / |error| (t horizontal, x vertical)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [
        (truth.T, "Truth (exact)", "RdBu_r", dict(vmin=-1, vmax=1)),
        (pred.T, "PINN", "RdBu_r", dict(vmin=-1, vmax=1)),
        (err.T, "|PINN - truth|", "magma", dict()),
    ]
    for a, (fld, title, cmap, kw) in zip(ax, panels, strict=True):
        im = a.imshow(fld, extent=ext, origin="lower", aspect="auto", cmap=cmap, **kw)
        a.axvline(t_b, color="w" if cmap == "magma" else "k", ls="--", lw=1)
        a.set(xlabel="t", ylabel="x", title=title)
        plt.colorbar(im, ax=a)
    fig.suptitle(f"PINN vs truth   rel L2 = {metrics.rel_l2(pred, truth):.2e}")
    fig.tight_layout()
    fig.savefig(figdir / "field_comparison.png", dpi=120)
    plt.close(fig)

    # 2. Snapshots + shock lens
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for tt in [0.0, 0.25, 0.5, 0.75, 1.0]:
        i = int(round(tt * (len(t) - 1)))
        (line,) = ax[0].plot(x, truth[i], lw=2, alpha=0.5)
        ax[0].plot(x, pred[i], "--", color=line.get_color(), label=f"t={tt:.2f}")
    ax[0].set(xlabel="x", ylabel="u", title="Snapshots: truth (solid) vs PINN (dashed)")
    ax[0].legend(fontsize=8)
    i = int(round(0.75 * (len(t) - 1)))
    ax[1].plot(x, truth[i], lw=2, label="truth")
    ax[1].plot(x, pred[i], "--", label="PINN")
    ax[1].set(xlim=(-0.15, 0.15), xlabel="x", ylabel="u", title="Shock lens at t = 0.75")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(figdir / "snapshots.png", dpi=120)
    plt.close(fig)

    # 3. Loss curves
    log = np.genfromtxt(
        outdir / "train_log.csv", delimiter=",", names=True, dtype=None, encoding=None
    )
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for phase, ls in [("adam", "-"), ("lbfgs", "--")]:
        m = log["phase"] == phase
        if m.any():
            off = log["iter"][log["phase"] == "adam"].max() if phase == "lbfgs" else 0
            for col, c in [("loss_pde", "C0"), ("loss_ic", "C1"), ("loss_bc", "C2")]:
                ax.semilogy(
                    log["iter"][m] + off,
                    log[col][m],
                    ls,
                    color=c,
                    label=f"{col[5:]} ({phase})" if phase == "adam" else None,
                )
    ax.set(xlabel="iteration", ylabel="loss", title="PINN training losses")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "loss_curves.png", dpi=120)
    plt.close(fig)


def train_pinn_stream(
    u0: np.ndarray,
    nu: float,
    device: torch.device,
    iters: int = 4000,
    lr: float = 1.0e-3,
    n_collocation: int = 10000,
    n_bc: int = 100,
    resample_every: int = 1000,
    preview_every: int = 100,
    preview_nt: int = 51,
    callback=None,
    should_cancel=None,
):
    """Live, cancellable PINN training on an ARBITRARY IC (Adam only; for the UI).

    Trains u_theta(x, t) with IC + PDE residual + soft periodic BC, streaming
    (iter, losses, preview field (preview_nt, nx)) through ``callback`` and
    stopping when ``should_cancel()`` is true. Returns the trained model; saves
    nothing. Adam-only (no L-BFGS) so it stays fast and interruptible.
    """
    set_seed(0)
    model = PINN().to(device)
    x = x_grid(len(u0))
    xi = torch.tensor(x.reshape(-1, 1), device=device, dtype=torch.float32)
    ui = torch.tensor(np.asarray(u0).reshape(-1, 1), device=device, dtype=torch.float32)
    ti = torch.zeros_like(xi)
    t_bc = torch.linspace(0, 1, n_bc, device=device, dtype=torch.float32).reshape(-1, 1)
    xm1 = (-torch.ones_like(t_bc)).requires_grad_(True)
    xp1 = torch.ones_like(t_bc).requires_grad_(True)
    coll = sample_collocation(n_collocation, device, torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t_prev = np.linspace(0, 1, preview_nt)

    for it in range(1, iters + 1):
        if should_cancel is not None and should_cancel():
            break
        if it % resample_every == 0:
            coll = sample_collocation(n_collocation, device, torch.float32)
        cx, ct = coll
        l_pde = (pde_residual(model, cx, ct, nu) ** 2).mean()
        l_ic = ((model(xi, ti) - ui) ** 2).mean()
        um, up = model(xm1, t_bc), model(xp1, t_bc)
        um_x, up_x = _grad(um, xm1), _grad(up, xp1)
        l_bc = ((um - up) ** 2).mean() + ((um_x - up_x) ** 2).mean()
        loss = l_pde + l_ic + l_bc
        opt.zero_grad()
        loss.backward()
        opt.step()
        if callback is not None and (it % preview_every == 0 or it == iters):
            with torch.no_grad():
                field = _predict_grid(model, device, x, t_prev)  # (preview_nt, nx)
            callback(it, {"pde": l_pde.item(), "ic": l_ic.item(), "bc": l_bc.item()}, field)
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Train the classic-case PINN.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--out", type=Path, default=None, help="checkpoint dir (default checkpoints/<name>)"
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--adam-iters", type=int, default=None)
    p.add_argument("--lbfgs-iters", type=int, default=None)
    p.add_argument("--n-collocation", type=int, default=None)
    args = p.parse_args()

    cfg = PINNConfig.from_yaml(args.config)
    if args.device is not None:
        cfg.device = args.device
    if args.adam_iters is not None:
        cfg.adam_iters = args.adam_iters
    if args.lbfgs_iters is not None:
        cfg.lbfgs_iters = args.lbfgs_iters
    if args.n_collocation is not None:
        cfg.n_collocation = args.n_collocation
    outdir = args.out or (REPO_ROOT / "checkpoints" / cfg.name)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Training {cfg.name} on {cfg.device} -> {outdir}")
    results = train(cfg, outdir)
    print("\nResults:")
    for k, v in results.items():
        print(f"  {k:22s} {v}")


if __name__ == "__main__":
    main()
