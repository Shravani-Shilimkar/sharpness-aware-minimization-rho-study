"""
animate_compare_loss_surface_3d.py

Side-by-side 3D relative loss surfaces for:
    - baseline SGD
    - SAM (e.g., rho_0.05 / seed_42)

Uses:
  - same 2D directions in parameter space
  - shared vertical scaling

Usage example:

python animate_compare_loss_surface_3d.py \
    --baseline_run baseline_sgd \
    --sam_run rho_0.05 \
    --sam_seed seed_42
"""

import argparse
import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from collections import OrderedDict


# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser("Compare 3D loss surfaces: SGD vs SAM")
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--data_dir", type=str, default="data")

    p.add_argument("--baseline_run", type=str, default="baseline_sgd")
    p.add_argument("--sam_run", type=str, required=True,
                   help="e.g., rho_0.05, rho_0.10")
    p.add_argument("--sam_seed", type=str, required=True,
                   help="e.g., seed_42")

    p.add_argument("--radius", type=float, default=1.0,
                   help="radius for 2D slice in parameter space")
    p.add_argument("--grid", type=int, default=41,
                   help="N x N grid resolution")
    p.add_argument("--frames", type=int, default=120,
                   help="number of rotation frames")
    return p.parse_args()


# ---------------- project-specific ----------------

def build_model(device):
    from model import resnet18_cifar
    model = resnet18_cifar().to(device)
    return model


def get_one_batch(device, data_dir, batch_size=256):
    from dataset import get_cifar100_loaders
    train_loader, _ = get_cifar100_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,
        use_aug=True,
    )
    x, y = next(iter(train_loader))
    return x.to(device), y.to(device)


def get_checkpoint_path(run_name, seed, runs_dir):
    if run_name == "baseline_sgd":
        path = os.path.join(runs_dir, "baseline_sgd", "best.pt")
    else:
        if seed is None:
            raise ValueError(f"--sam_seed/seed must be provided for run {run_name}")
        path = os.path.join(runs_dir, run_name, seed, "best.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def remap_shortcut_to_downsample(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if "shortcut" in k:
            k = k.replace("shortcut", "downsample")
        new_sd[k] = v
    return new_sd


def load_model_for_run(run_name, seed, runs_dir, device):
    model = build_model(device)
    ckpt_path = get_checkpoint_path(run_name, seed, runs_dir)
    print(f"Loading checkpoint from: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    state_dict = remap_shortcut_to_downsample(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# ---------------- utils ----------------

def flatten_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def set_params_from_flat(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].view_as(p))
        idx += n


def make_shared_directions(device):
    # any ResNet-18 instance is fine to set dimensionality
    model = build_model(device)
    w = flatten_params(model)
    d1 = torch.randn_like(w)
    d2 = torch.randn_like(w)
    d1 = d1 / (d1.norm() + 1e-8)
    d2 = d2 - (d2 @ d1) * d1
    d2 = d2 / (d2.norm() + 1e-8)
    return d1, d2


@torch.no_grad()
def batch_loss(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits, y).item()


def sample_loss_grid_with_dirs(model, x, y, radius, grid, device, d1, d2):
    base_w = flatten_params(model)

    alphas = np.linspace(-radius, radius, grid)
    betas = np.linspace(-radius, radius, grid)
    Z = np.zeros((grid, grid), dtype=np.float32)

    tmp = copy.deepcopy(model).to(device)
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            w = base_w + a * d1 + b * d2
            set_params_from_flat(tmp, w)
            Z[j, i] = batch_loss(tmp, x, y)

    return alphas, betas, Z


# ---------------- main ----------------

def main():
    args = parse_args()
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device:", device)

    # shared data batch and directions
    x, y = get_one_batch(device, args.data_dir)
    d1, d2 = make_shared_directions(device)

    # load models
    model_sgd = load_model_for_run(
        args.baseline_run, None, args.runs_dir, device
    )
    model_sam = load_model_for_run(
        args.sam_run, args.sam_seed, args.runs_dir, device
    )

    # sample both surfaces on SAME directions
    print("Sampling baseline SGD surface...")
    a, b, Z_sgd_raw = sample_loss_grid_with_dirs(
        model_sgd, x, y, args.radius, args.grid, device, d1, d2
    )

    print(f"Sampling SAM surface ({args.sam_run}/{args.sam_seed})...")
    _, _, Z_sam_raw = sample_loss_grid_with_dirs(
        model_sam, x, y, args.radius, args.grid, device, d1, d2
    )

    # center by own minimum, but normalize with a GLOBAL max
    Z_sgd_centered = Z_sgd_raw - Z_sgd_raw.min()
    Z_sam_centered = Z_sam_raw - Z_sam_raw.min()

    global_max = max(Z_sgd_centered.max(), Z_sam_centered.max())
    global_max = float(global_max) + 1e-12

    Z_sgd = Z_sgd_centered / global_max
    Z_sam = Z_sam_centered / global_max

    A, B = np.meshgrid(a, b)

    fig = plt.figure(figsize=(11, 5))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")

    surf0 = ax0.plot_surface(
        A, B, Z_sgd,
        cmap="coolwarm",
        linewidth=0,
        antialiased=True,
        vmin=0.0,
        vmax=1.0,
    )
    surf1 = ax1.plot_surface(
        A, B, Z_sam,
        cmap="coolwarm",
        linewidth=0,
        antialiased=True,
        vmin=0.0,
        vmax=1.0,
    )

    cbar = fig.colorbar(surf1, ax=[ax0, ax1], shrink=0.7,
                        label="Relative loss (shared scale)")

    ax0.set_xlabel("α (direction 1)")
    ax0.set_ylabel("β (direction 2)")
    ax0.set_zlabel("Relative loss")
    ax0.set_title("Baseline SGD")

    ax1.set_xlabel("α (direction 1)")
    ax1.set_ylabel("β (direction 2)")
    ax1.set_zlabel("Relative loss")
    ax1.set_title(f"SAM ({args.sam_run}, {args.sam_seed})")

    ax0.view_init(elev=35, azim=45)
    ax1.view_init(elev=35, azim=45)

    def init():
        ax0.view_init(elev=35, azim=45)
        ax1.view_init(elev=35, azim=45)
        return surf0, surf1

    def update(frame):
        az = 45 + frame
        ax0.view_init(elev=35, azim=az)
        ax1.view_init(elev=35, azim=az)
        return surf0, surf1

    out = f"loss_surface3d_compare_{args.baseline_run}_vs_{args.sam_run}_{args.sam_seed}.mp4"
    print(f"Saving to {out} ...")
    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=args.frames, interval=80, blit=False
    )
    try:
        anim.save(out, fps=15, dpi=150, extra_args=["-vcodec", "libx264"])
    except (TypeError, RuntimeError):
        anim.save(out, fps=15, dpi=150)
    print("Done.")


if __name__ == "__main__":
    main()
