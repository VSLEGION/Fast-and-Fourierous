"""Train the space-time FNO-2D on the Burgers dataset (CLAUDE.md §5.2).

Relative-L2 loss over the full space-time field, Adam (lr 1e-3) with cosine
decay, ~500 epochs, batch 20, trained on 256 x 51. The checkpoint is selected on
the validation split (last 100 training samples).

Evaluation:
  * test rel L2 at 256 (subsample the 1024 test data),
  * zero-shot at 1024 in x (discretization invariance) + the error ratio,
  * zero-shot at 101 in t on the classic file,
  * inference latency p50/p95 (batch 1, after warmup).

Usage:
    python -m shocklab.train.fno --config configs/fno.yaml
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from shocklab.data.datasets import load_classic, load_test, load_train_val, subsample_x
from shocklab.eval import metrics
from shocklab.eval.benchmark import latency_ms
from shocklab.models.fno import FNO2d
from shocklab.models.layers import rel_l2_loss
from shocklab.train.common import REPO_ROOT, CSVLogger, Timer, get_device, save_metrics, set_seed

# TF32 is fine for the pointwise convs; it does not touch the complex spectral path.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class FNOConfig:
    name: str = "fno"
    seed: int = 0
    device: str = "cuda"
    modes_x: int = 32
    modes_t: int = 16
    width: int = 64
    n_layers: int = 4
    t_pad_frac: float = 0.25
    epochs: int = 500
    batch_size: int = 20
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    log_every: int = 10

    @classmethod
    def from_yaml(cls, path: Path) -> FNOConfig:
        d = yaml.safe_load(Path(path).read_text())
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in fields})


def build_model(cfg: FNOConfig) -> FNO2d:
    return FNO2d(
        modes_x=cfg.modes_x,
        modes_t=cfg.modes_t,
        width=cfg.width,
        n_layers=cfg.n_layers,
        t_pad_frac=cfg.t_pad_frac,
    )


@torch.no_grad()
def predict_fields(model, device, u0: np.ndarray, t: np.ndarray, batch: int = 50) -> np.ndarray:
    """u0: (N, nx); t: (nt,). Returns predicted fields (N, nt, nx)."""
    model.eval()
    tt = torch.as_tensor(t, device=device, dtype=torch.float32)
    outs = []
    for i in range(0, len(u0), batch):
        ub = torch.as_tensor(u0[i : i + batch], device=device, dtype=torch.float32)
        pred = model(ub, tt)  # (b, nx, nt)
        outs.append(pred.transpose(-1, -2).cpu().numpy())  # (b, nt, nx)
    return np.concatenate(outs)


def train(cfg: FNOConfig, outdir: Path) -> dict:
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    model = build_model(cfg).to(device)

    train_d, val_d = load_train_val()
    t_grid = torch.as_tensor(train_d["t"], device=device, dtype=torch.float32)  # (51,)
    u_train = torch.as_tensor(train_d["u"], device=device, dtype=torch.float32)  # (900,51,256)
    u_val = torch.as_tensor(val_d["u"], device=device, dtype=torch.float32)  # (100,51,256)
    u0_train = u_train[:, 0, :]
    u0_val = u_val[:, 0, :]

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    logger = CSVLogger(
        outdir / "train_log.csv", ["epoch", "time_s", "train_rel_l2", "val_rel_l2", "lr"]
    )

    n = u_train.shape[0]
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    with Timer() as timer:
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            perm = torch.randperm(n, device=device)
            running = 0.0
            for i in range(0, n, cfg.batch_size):
                idx = perm[i : i + cfg.batch_size]
                pred = model(u0_train[idx], t_grid).transpose(-1, -2)  # (b, nt, nx)
                loss = rel_l2_loss(pred, u_train[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item() * len(idx)
            sched.step()
            train_rl2 = running / n

            model.eval()
            with torch.no_grad():
                val_pred = model(u0_val, t_grid).transpose(-1, -2)
                val_rl2 = rel_l2_loss(val_pred, u_val).item()
            if val_rl2 < best_val:
                best_val = val_rl2
                best_state = copy.deepcopy(model.state_dict())
            if epoch % cfg.log_every == 0 or epoch == 1:
                logger.log(
                    epoch=epoch,
                    time_s=round(timer.elapsed(), 1),
                    train_rel_l2=train_rl2,
                    val_rel_l2=val_rl2,
                    lr=opt.param_groups[0]["lr"],
                )
                print(f"epoch {epoch:4d}  train {train_rl2:.4e}  val {val_rl2:.4e}")
    train_seconds = timer.seconds
    logger.close()

    model.load_state_dict(best_state)  # select checkpoint on validation
    torch.save(best_state, outdir / "model.pt")

    results = evaluate(model, device)
    results.update(
        best_val_rel_l2=best_val,
        train_seconds=round(train_seconds, 1),
        epochs=cfg.epochs,
        num_params=model.num_params(),
    )
    save_metrics(outdir, asdict(cfg), results)
    make_figures(model, device, outdir)
    return results


def evaluate(model, device) -> dict:
    test = load_test()  # (200, 51, 1024)
    # test at 256 (subsample x by 4)
    u256 = subsample_x(test["u"], 4)  # (200, 51, 256)
    pred256 = predict_fields(model, device, u256[:, 0, :], test["t"])
    rl2_256 = metrics.mean_rel_l2(pred256, u256)
    # zero-shot at 1024
    pred1024 = predict_fields(model, device, test["u"][:, 0, :], test["t"])
    rl2_1024 = metrics.mean_rel_l2(pred1024, test["u"])
    # zero-shot in t: classic (101 snapshots, 1024 points)
    classic = load_classic()
    predc = predict_fields(model, device, classic["u"][None, 0, :], classic["t"])[0]
    rl2_classic = metrics.rel_l2(predc, classic["u"])

    # latency (batch 1, after warmup), at 256 and at 1024
    t_t = torch.as_tensor(test["t"], device=device, dtype=torch.float32)
    lat256 = latency_ms(
        model, torch.as_tensor(u256[:1, 0, :], device=device, dtype=torch.float32), t_t
    )
    lat1024 = latency_ms(
        model, torch.as_tensor(test["u"][:1, 0, :], device=device, dtype=torch.float32), t_t
    )

    return {
        "test_rel_l2_256": rl2_256,
        "test_rel_l2_1024": rl2_1024,
        "superres_ratio_1024_over_256": rl2_1024 / rl2_256,
        "classic_zeroshot_t_rel_l2": rl2_classic,
        "latency_256_ms": lat256,
        "latency_1024_ms": lat1024,
    }


def make_figures(model, device, outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    test = load_test()
    u256 = subsample_x(test["u"], 4)
    x256 = subsample_x(test["x"], 4)
    s = 0  # show test sample 0
    pred = predict_fields(model, device, u256[s : s + 1, 0, :], test["t"])[0]  # (51, 256)
    truth = u256[s]
    ext = [test["t"][0], test["t"][-1], x256[0], x256[-1]]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [
        (truth.T, "Truth (solver)", "RdBu_r", dict(vmin=-1, vmax=1)),
        (pred.T, "FNO", "RdBu_r", dict(vmin=-1, vmax=1)),
        (np.abs(pred - truth).T, "|FNO - truth|", "magma", dict()),
    ]
    for a, (fld, title, cmap, kw) in zip(ax, panels, strict=True):
        im = a.imshow(fld, extent=ext, origin="lower", aspect="auto", cmap=cmap, **kw)
        a.set(xlabel="t", ylabel="x", title=title)
        plt.colorbar(im, ax=a)
    fig.suptitle(f"FNO test sample 0   rel L2 = {metrics.rel_l2(pred, truth):.2e}")
    fig.tight_layout()
    fig.savefig(figdir / "test_field.png", dpi=120)
    plt.close(fig)

    # zero-shot super-resolution: same IC at 256 and 1024, snapshot at t = 1
    pred1024 = predict_fields(model, device, test["u"][s : s + 1, 0, :], test["t"])[0]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(test["x"], test["u"][s, -1], "k", lw=2, label="truth @1024")
    ax.plot(x256, pred[-1], "o", ms=3, label="FNO @256 (trained)")
    ax.plot(test["x"], pred1024[-1], "-", lw=1, label="FNO @1024 (zero-shot)")
    ax.set(xlabel="x", ylabel="u", title="Zero-shot super-resolution, t = 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "superres.png", dpi=120)
    plt.close(fig)

    # training curves
    log = np.genfromtxt(outdir / "train_log.csv", delimiter=",", names=True)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.semilogy(log["epoch"], log["train_rel_l2"], label="train")
    ax.semilogy(log["epoch"], log["val_rel_l2"], label="val")
    ax.set(xlabel="epoch", ylabel="relative L2", title="FNO training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "loss_curves.png", dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Train the space-time FNO-2D.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()

    cfg = FNOConfig.from_yaml(args.config)
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
