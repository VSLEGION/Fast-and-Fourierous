"""Train the PINO (FNO + physics) and provide test-time "Refine with physics".

The PINO is the FNO architecture (§5.2) trained with the PINN's physics loss added:

    L = L_data + lambda_pde * L_pde + lambda_ic * L_ic

L_pde is the Burgers residual on a FINER time grid (256 x 101, querying the model
there), with u_x, u_xx spectral in x and u_t a 2nd-order finite difference in t,
normalized by the truth's own residual level R0 so the finite-difference floor is
not penalized (CLAUDE.md §5.3).

``refine`` is the test-time, instance-wise fine-tuning: starting from the PINO
weights, minimize L_pde + L_ic ONLY (no data) for a given IC. It operates on a
copy and never touches the saved checkpoint -- the bridge between the slow PINN
(physics) and the fast FNO (data).

Usage:
    python -m shocklab.train.pino --config configs/pino.yaml
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from shocklab.data.datasets import load_test, load_train_val, subsample_x
from shocklab.eval import metrics
from shocklab.models.fno import FNO2d
from shocklab.models.layers import rel_l2_loss
from shocklab.models.physics_loss import pde_residual
from shocklab.physics import invariants as inv
from shocklab.physics.ic import random_ic
from shocklab.physics.solver import NU, burgers_solve, x_grid
from shocklab.train.common import REPO_ROOT, CSVLogger, Timer, get_device, save_metrics, set_seed
from shocklab.train.fno import predict_fields

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class PINOConfig:
    name: str = "pino"
    seed: int = 0
    device: str = "cuda"
    modes_x: int = 32
    modes_t: int = 16
    width: int = 64
    n_layers: int = 4
    t_pad_frac: float = 0.25
    init_from_fno: bool = True
    nt_fine: int = 101
    lambda_pde: float = 1.0e-3
    lambda_ic: float = 1.0
    floor_samples: int = 16
    floor_seed: int = 777
    epochs: int = 250
    batch_size: int = 20
    lr: float = 5.0e-4
    weight_decay: float = 0.0
    log_every: int = 10
    refine_steps: int = 300
    refine_lr: float = 5.0e-3

    @classmethod
    def from_yaml(cls, path: Path) -> PINOConfig:
        d = yaml.safe_load(Path(path).read_text())
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in fields})


def build_model(cfg: PINOConfig) -> FNO2d:
    return FNO2d(cfg.modes_x, cfg.modes_t, cfg.width, cfg.n_layers, cfg.t_pad_frac)


def residual_floor(n: int, seed: int, nt_fine: int) -> float:
    """R0 = RMS of the truth's discrete residual at 256 x nt_fine (the FD-in-t floor)."""
    rng = np.random.default_rng(seed)
    U = burgers_solve(random_ic(n, 2048, rng), n_out=nt_fine, dt=4e-4)  # (n, nt, 2048)
    U256 = U[:, :, ::8]
    res = inv.pde_residual(U256, np.linspace(0, 1, nt_fine), NU)
    return float(np.sqrt(np.mean(res**2)))


def refine(
    model: FNO2d,
    u0: torch.Tensor,
    t: torch.Tensor,
    steps: int = 300,
    lr: float = 5.0e-3,
    nu: float = NU,
    lambda_ic: float = 1.0,
    callback: Callable[[int, float, float, torch.Tensor], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[torch.Tensor, list[dict]]:
    """Test-time physics refinement for one (or a batch of) IC(s). Does NOT modify ``model``.

    Minimizes L_pde + lambda_ic * L_ic on a COPY of the weights. u0: (B, nx),
    t: (nt,). Returns (refined field (B, nx, nt), loss history). Stops early if
    ``should_cancel()`` becomes true.
    """
    work = copy.deepcopy(model).train()
    opt = torch.optim.Adam(work.parameters(), lr=lr)
    dt = float((t[1] - t[0]).item())
    history: list[dict] = []
    for step in range(1, steps + 1):
        if should_cancel is not None and should_cancel():
            break
        pred = work(u0, t)  # (B, nx, nt)
        l_pde = (pde_residual(pred, dt, nu) ** 2).mean()
        l_ic = ((pred[..., 0] - u0) ** 2).mean()
        loss = l_pde + lambda_ic * l_ic
        opt.zero_grad()
        loss.backward()
        opt.step()
        rec = {"step": step, "pde": l_pde.item(), "ic": l_ic.item()}
        history.append(rec)
        if callback is not None:
            callback(step, rec["pde"], rec["ic"], pred.detach())
    with torch.no_grad():
        final = work(u0, t)
    return final, history


def train(cfg: PINOConfig, outdir: Path) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    model = build_model(cfg).to(device)
    if cfg.init_from_fno:
        fno_ckpt = REPO_ROOT / "checkpoints" / "fno" / "model.pt"
        model.load_state_dict(torch.load(fno_ckpt, map_location=device))
        print(f"warm-started from {fno_ckpt}")

    R0 = residual_floor(cfg.floor_samples, cfg.floor_seed, cfg.nt_fine)
    print(f"residual floor R0 = {R0:.4e}")

    train_d, val_d = load_train_val()
    nt_fine = cfg.nt_fine
    dt_fine = 1.0 / (nt_fine - 1)
    t_fine = torch.linspace(0, 1, nt_fine, device=device)
    sub = (nt_fine - 1) // (train_d["u"].shape[1] - 1)  # 101 -> 51 is every 2nd
    u_train = torch.as_tensor(train_d["u"], device=device)  # (900, 51, 256)
    u_val = torch.as_tensor(val_d["u"], device=device)
    u0_train, u0_val = u_train[:, 0, :], u_val[:, 0, :]

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    logger = CSVLogger(
        outdir / "train_log.csv",
        ["epoch", "time_s", "train_data", "val_data", "val_resid_rms", "lr"],
    )

    n = u_train.shape[0]
    best_obj = float("inf")
    best_val_data = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    def pino_losses(u0, field):
        pred = model(u0, t_fine)  # (B, nx, 101)
        l_data = rel_l2_loss(pred[..., ::sub].transpose(-1, -2), field)
        l_pde = (pde_residual(pred, dt_fine, NU) ** 2).mean() / R0**2
        l_ic = ((pred[..., 0] - u0) ** 2).mean()
        return l_data, l_pde, l_ic

    with Timer() as timer:
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            perm = torch.randperm(n, device=device)
            running = 0.0
            for i in range(0, n, cfg.batch_size):
                idx = perm[i : i + cfg.batch_size]
                l_data, l_pde, l_ic = pino_losses(u0_train[idx], u_train[idx])
                loss = l_data + cfg.lambda_pde * l_pde + cfg.lambda_ic * l_ic
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += l_data.item() * len(idx)
            sched.step()
            train_data = running / n

            model.eval()
            with torch.no_grad():
                pred_val = model(u0_val, t_fine)
                val_data = rel_l2_loss(pred_val[..., ::sub].transpose(-1, -2), u_val).item()
                val_resid = float(
                    np.sqrt(((pde_residual(pred_val, dt_fine, NU)) ** 2).mean().item())
                )
            # select on the actual objective (data + weighted, normalized physics),
            # so the chosen checkpoint reflects the physics/data trade-off, not data alone
            val_obj = val_data + cfg.lambda_pde * (val_resid / R0) ** 2
            if val_obj < best_obj:
                best_obj = val_obj
                best_val_data = val_data
                best_state = copy.deepcopy(model.state_dict())
            if epoch % cfg.log_every == 0 or epoch == 1:
                logger.log(
                    epoch=epoch,
                    time_s=round(timer.elapsed(), 1),
                    train_data=train_data,
                    val_data=val_data,
                    val_resid_rms=val_resid,
                    lr=opt.param_groups[0]["lr"],
                )
                print(f"epoch {epoch:4d}  data {val_data:.4e}  resid {val_resid:.4e} (R0={R0:.2e})")
    train_seconds = timer.seconds
    logger.close()

    model.load_state_dict(best_state)
    torch.save(best_state, outdir / "model.pt")

    results = evaluate(model, device, cfg, R0)
    results.update(
        best_val_data_rel_l2=best_val_data,
        residual_floor_R0=R0,
        train_seconds=round(train_seconds, 1),
        epochs=cfg.epochs,
    )
    save_metrics(outdir, asdict(cfg), results)
    make_figures(model, device, outdir, cfg)
    return results


def _residual_rms(model, device, u0_256: np.ndarray, nt_fine: int) -> float:
    t = torch.linspace(0, 1, nt_fine, device=device)
    dt = 1.0 / (nt_fine - 1)
    with torch.no_grad():
        pred = model(torch.as_tensor(u0_256, device=device), t)
        return float(np.sqrt(((pde_residual(pred, dt, NU)) ** 2).mean().item()))


def evaluate(model, device, cfg: PINOConfig, R0: float) -> dict:
    test = load_test()
    u256 = subsample_x(test["u"], 4)
    # data error vs FNO
    pred256 = predict_fields(model, device, u256[:, 0, :], test["t"])
    pred1024 = predict_fields(model, device, test["u"][:, 0, :], test["t"])
    rl2_256 = metrics.mean_rel_l2(pred256, u256)
    rl2_1024 = metrics.mean_rel_l2(pred1024, test["u"])

    # residual vs FNO on test ICs (256 x nt_fine)
    fno = build_model(cfg).to(device).eval()
    fno.load_state_dict(
        torch.load(REPO_ROOT / "checkpoints" / "fno" / "model.pt", map_location=device)
    )
    u0_test = u256[:64, 0, :]
    resid_pino = _residual_rms(model, device, u0_test, cfg.nt_fine)
    resid_fno = _residual_rms(fno, device, u0_test, cfg.nt_fine)

    results = {
        "test_rel_l2_256": rl2_256,
        "test_rel_l2_1024": rl2_1024,
        "residual_rms_pino": resid_pino,
        "residual_rms_fno": resid_fno,
        "residual_ratio_pino_over_fno": resid_pino / resid_fno,
        "residual_pino_over_floor": resid_pino / R0,
        "refine": eval_refine_presets(model, device, cfg),
    }
    return results


def eval_refine_presets(model, device, cfg: PINOConfig) -> dict:
    """Refine on the OOD boosted and linear-limit presets; report rel L2 before/after (§6.3)."""
    x2048 = x_grid(2048)
    nt = cfg.nt_fine
    presets = {
        "boosted_c0.4": -np.sin(np.pi * x2048) + 0.4,
        "linear_eps0.05": 0.05 * np.sin(np.pi * x2048),
    }
    t = torch.linspace(0, 1, nt, device=device)
    out = {}
    for name, u0_2048 in presets.items():
        truth = burgers_solve(u0_2048, n_out=nt, dt=4e-4)[0][:, ::8]  # (nt, 256)
        u0_256 = u0_2048[::8].astype(np.float32)
        before = predict_fields(
            model, device, u0_256[None], np.linspace(0, 1, nt).astype(np.float32)
        )[0]
        final, _ = refine(
            model,
            torch.as_tensor(u0_256[None], device=device),
            t,
            steps=cfg.refine_steps,
            lr=cfg.refine_lr,
            lambda_ic=cfg.lambda_ic,
        )
        after = final.transpose(-1, -2).cpu().numpy()[0]
        out[name] = {
            "rel_l2_before": metrics.rel_l2(before, truth),
            "rel_l2_after": metrics.rel_l2(after, truth),
        }
    return out


def make_figures(model, device, outdir: Path, cfg: PINOConfig) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    x2048 = x_grid(2048)
    x256 = x2048[::8]
    nt = cfg.nt_fine
    tnp = np.linspace(0, 1, nt).astype(np.float32)
    t = torch.linspace(0, 1, nt, device=device)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for a, (name, u0_2048) in zip(
        ax,
        [
            ("boosted  u0=-sin(pi x)+0.4", -np.sin(np.pi * x2048) + 0.4),
            ("linear limit  u0=0.05 sin(pi x)", 0.05 * np.sin(np.pi * x2048)),
        ],
        strict=True,
    ):
        truth = burgers_solve(u0_2048, n_out=nt, dt=4e-4)[0][:, ::8]
        u0_256 = u0_2048[::8].astype(np.float32)
        before = predict_fields(model, device, u0_256[None], tnp)[0]
        final, _ = refine(
            model,
            torch.as_tensor(u0_256[None], device=device),
            t,
            steps=cfg.refine_steps,
            lr=cfg.refine_lr,
            lambda_ic=cfg.lambda_ic,
        )
        after = final.transpose(-1, -2).cpu().numpy()[0]
        a.plot(x256, truth[-1], "k", lw=2, label="truth (solver)")
        a.plot(x256, before[-1], "--", label=f"PINO ({metrics.rel_l2(before, truth):.1%})")
        a.plot(x256, after[-1], "-", label=f"refined ({metrics.rel_l2(after, truth):.1%})")
        a.set(xlabel="x", ylabel="u", title=name + ", t = 1")
        a.legend(fontsize=8)
    fig.suptitle("Refine with physics on OOD presets")
    fig.tight_layout()
    fig.savefig(figdir / "refine_presets.png", dpi=120)
    plt.close(fig)

    log = np.genfromtxt(outdir / "train_log.csv", delimiter=",", names=True)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.semilogy(log["epoch"], log["val_data"], label="val data rel L2")
    ax.semilogy(log["epoch"], log["val_resid_rms"], label="val residual RMS")
    ax.set(xlabel="epoch", ylabel="value", title="PINO training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "loss_curves.png", dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Train the PINO (FNO + physics).")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()

    cfg = PINOConfig.from_yaml(args.config)
    if args.device is not None:
        cfg.device = args.device
    if args.epochs is not None:
        cfg.epochs = args.epochs
    outdir = args.out or (REPO_ROOT / "checkpoints" / cfg.name)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Training {cfg.name} on {cfg.device} -> {outdir}")
    results = train(cfg, outdir)
    print("\nResults:")
    for k, v in results.items():
        print(f"  {k:30s} {v}")


if __name__ == "__main__":
    main()
