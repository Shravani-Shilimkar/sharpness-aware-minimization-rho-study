import os
import csv
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn # Need for BasicBlock/ResNet
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- External Dependencies (Defined internally for self-containment) ---

# Re-implementing the ResNet model structure from model.py
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # Note: The original ResNet implementation uses 'shortcut', but for compatibility
            # with standard PyTorch checkpoints, we must handle both 'shortcut' and 'downsample'.
            # The custom implementation is 'shortcut'.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet18_cifar():
    """Returns a ResNet-18 model configured for CIFAR-100 (100 classes)."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100)


# Mock implementation of get_cifar100_loaders (from dataset.py) to avoid import errors
# NOTE: This only provides a DataLoader, it won't download data. Assumes data is present.
import torchvision.transforms as transforms
import torchvision

def get_cifar100_loaders(data_dir, batch_size, num_workers, use_aug):
    """Mocks the function from dataset.py for loss landscape computation."""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Only need the test set for loss landscape visualization
    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Return mock trainloader and real testloader
    return None, testloader

# --- End of External Dependencies ---

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
SAM_HISTORY_FILE = "./logs/sam_rho005_history.pth"
BASELINE_METRICS_FILE = "./runs/baseline_sgd/metrics.csv"
RUNS_DIR = "./runs"
PLOTS_DIR = "./plots"


# ---------------------------------------------------------------------
# Legacy loaders and Plotting (for single/comparison modes)
# ---------------------------------------------------------------------
def load_data(filepath, data_type):
    """Loads history data from either a PyTorch .pth file (legacy SAM) or a CSV file (baseline / new runs)."""
    if not os.path.exists(filepath):
        print(f"Error: History file not found at {filepath}")
        return None, None

    try:
        if data_type == "pth":
            # --- LEGACY SAM FORMAT (.pth) ---
            data = torch.load(filepath, map_location="cpu")
            rho = data.get("rho", "N/A")

            history = {
                "epochs": list(range(1, len(data["test_acc"]) + 1)),
                "train_loss": data["train_loss"],
                "val_loss": data["test_loss"],
                "train_acc_frac": [a / 100 for a in data["train_acc"]],
                "val_acc_frac": [a / 100 for a in data["test_acc"]],
                "train_acc_pct": data["train_acc"],
                "val_acc_pct": data["test_acc"],
                "epoch_time_sec": data.get("runtime_per_epoch", []),
            }
            print(f"Loaded legacy SAM history (PTH) for ρ = {rho}")
            return history, rho

        elif data_type == "csv":
            df = pd.read_csv(filepath)
            rho = 0.0  # baseline

            history = {
                "epochs": df["epoch"].tolist(),
                "train_loss": df["train_loss"].tolist(),
                "val_loss": df["val_loss"].tolist(),
                "train_acc_frac": df["train_acc"].tolist(),
                "val_acc_frac": df["val_acc"].tolist(),
                "epoch_time_sec": df["epoch_time_sec"].tolist(),
            }
            print(f"Loaded CSV history for ρ = {rho}")
            return history, rho

    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None, None

    return None, None


def plot_single_run(history, rho):
    """Generates and saves loss and accuracy plots for a single SAM run (legacy)."""
    if history is None:
        print("Cannot generate plots: history is missing.")
        return

    epochs = history["epochs"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", linestyle="--", color="C0")
    ax1.plot(epochs, history["val_loss"], label="Test Loss", color="C0", linewidth=2)
    ax1.set_title(f"SAM ($\\rho$={rho}) Loss vs. Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.5)

    # Accuracy (percentage)
    ax2.plot(epochs, history["train_acc_pct"], label="Train Acc", linestyle="--", color="C1")
    ax2.plot(epochs, history["val_acc_pct"], label="Test Acc", color="C1", linewidth=2)
    ax2.set_title(f"SAM ($\\rho$={rho}) Accuracy vs. Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(PLOTS_DIR) / "sam_rho005_curves.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved single-run legacy SAM curves to {out_path}")


def plot_comparison(baseline_history, sam_history, baseline_rho, sam_rho):
    """Baseline vs legacy SAM comparison curves."""
    if baseline_history is None or sam_history is None:
        print("Cannot generate comparison: missing data.")
        return

    min_epochs = min(
        len(baseline_history["epochs"]),
        len(sam_history["epochs"]),
    )
    epochs = list(range(1, min_epochs + 1))

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(
        epochs,
        baseline_history["train_loss"][:min_epochs],
        label=f"SGD (ρ={baseline_rho}) Train Loss",
        linestyle="--",
        color="C0",
    )
    plt.plot(
        epochs,
        baseline_history["val_loss"][:min_epochs],
        label=f"SGD (ρ={baseline_rho}) Test Loss",
        color="C0",
    )
    plt.plot(
        epochs,
        sam_history["train_loss"][:min_epochs],
        label=f"SAM (ρ={sam_rho}) Train Loss",
        linestyle="--",
        color="C1",
    )
    plt.plot(
        epochs,
        sam_history["val_loss"][:min_epochs],
        label=f"SAM (ρ={sam_rho}) Test Loss",
        color="C1",
    )
    plt.title("Training and Test Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.5)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(
        epochs,
        [a * 100 for a in baseline_history["train_acc_frac"][:min_epochs]],
        label=f"SGD (ρ={baseline_rho}) Train Acc",
        linestyle="--",
        color="C0",
    )
    plt.plot(
        epochs,
        [a * 100 for a in baseline_history["val_acc_frac"][:min_epochs]],
        label=f"SGD (ρ={baseline_rho}) Test Acc",
        color="C0",
    )
    plt.plot(
        epochs,
        [a * 100 for a in sam_history["train_acc_frac"][:min_epochs]],
        label=f"SAM (ρ={sam_rho}) Train Acc",
        linestyle="--",
        color="C1",
    )
    plt.plot(
        epochs,
        [a * 100 for a in sam_history["val_acc_frac"][:min_epochs]],
        label=f"SAM (ρ={sam_rho}) Test Acc",
        color="C1",
    )
    plt.title("Training and Test Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# ρ-sweep aggregation: ρ vs accuracy, ρ vs time/epoch
# ---------------------------------------------------------------------
def discover_rho_runs(runs_dir: str = RUNS_DIR):
    """
    Scan ./runs and return {rho: [(seed_name, df), ...]}.
    - Baseline SGD is treated as rho=0.0.
    - rho_xx directories contain seed_* subdirectories.
    """
    results = {}

    # Baseline
    baseline_dir = os.path.join(runs_dir, "baseline_sgd")
    metrics_path = os.path.join(baseline_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        # Convert accuracy from % (0-100) to fraction (0-1) if needed, 
        # but CSV should store fraction. We assume fraction for aggregation.
        results.setdefault(0.0, []).append(("baseline", df))
        print(f"Found baseline metrics at {metrics_path}")

    # rho_* runs
    for entry in os.listdir(runs_dir):
        full = os.path.join(runs_dir, entry)
        if not os.path.isdir(full):
            continue
        if not entry.startswith("rho_"):
            continue

        try:
            # Handle directory names like rho_0.05
            rho = float(entry.split("_", 1)[1])
        except ValueError:
            continue

        for seed_dir in os.listdir(full):
            seed_path = os.path.join(full, seed_dir)
            if not os.path.isdir(seed_path):
                continue
            metrics_path = os.path.join(seed_path, "metrics.csv")
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                results.setdefault(rho, []).append((seed_dir, df))
                print(f"Found metrics for ρ={rho}, seed={seed_dir}")
            else:
                print(f"No metrics.csv at {metrics_path}, skipping.")

    return results

def override_with_legacy_rho005(df_summary: pd.DataFrame):
    """
    Override (or insert) the row for rho=0.05 using the finished legacy SAM run
    stored in logs/sam_rho005_history.pth.
    """
    if not os.path.exists(SAM_HISTORY_FILE):
        print("Legacy SAM history file not found; cannot override rho=0.05.")
        return df_summary

    try:
        data = torch.load(SAM_HISTORY_FILE, map_location="cpu")
    except Exception as e:
        print(f"Error loading legacy SAM history for rho=0.05: {e}")
        return df_summary

    # Final test accuracy in the legacy file is in percent (0–100)
    final_test_acc_pct = float(data["test_acc"][-1])
    final_test_acc = final_test_acc_pct / 100.0

    # Epoch runtimes, if present
    times = data.get("runtime_per_epoch", [])
    mean_time = float(np.mean(times)) if len(times) > 0 else np.nan
    std_time = float(np.std(times)) if len(times) > 0 else 0.0

    new_row = {
        "rho": 0.05,
        "num_seeds": 1,
        "mean_acc": final_test_acc,
        "std_acc": 0.0,          # only 1 seed
        "mean_time": mean_time,
        "std_time": std_time,
    }

    # Drop any existing row for rho=0.05, then append the new one
    if "rho" in df_summary.columns:
        # Use np.isclose for float comparison
        mask = np.isclose(df_summary["rho"], 0.05)
        df_summary = df_summary[~mask]

    # Use pd.concat for appending the new row
    df_summary = pd.concat([df_summary, pd.DataFrame([new_row])], ignore_index=True)
    return df_summary


def summarize_rho_stats(runs_dict):
    """
    Build a summary DataFrame for ρ sweep.
    """
    rows = []
    for rho in sorted(runs_dict.keys()):
        dfs = [df for (_, df) in runs_dict[rho]]
        if not dfs:
            continue

        # val_acc column stores fractional accuracy (0-1) from CSV
        final_accs = [df["val_acc"].iloc[-1] for df in dfs]
        mean_times = [df["epoch_time_sec"].mean() for df in dfs]

        rows.append(
            {
                "rho": rho,
                "num_seeds": len(dfs),
                "mean_acc": float(np.mean(final_accs)),
                "std_acc": float(np.std(final_accs)),
                "mean_time": float(np.mean(mean_times)),
                "std_time": float(np.std(mean_times)),
            }
        )

    if not rows:
        return None

    df_summary = pd.DataFrame(rows)

    # 1) Override rho=0.05 with the finished legacy SAM run (if present)
    df_summary = override_with_legacy_rho005(df_summary)

    # 2) Keep ONLY the rhos we care about
    allowed_rhos = [0.0, 0.05, 0.10, 0.20]
    # Use floating point comparison for filtering
    df_summary = df_summary[np.round(df_summary["rho"], 2).isin(allowed_rhos)]

    # Sort nicely by rho
    df_summary = df_summary.sort_values("rho").reset_index(drop=True)

    return df_summary


def plot_rho_summary(df_summary, out_dir=PLOTS_DIR):
    """Plot ρ vs accuracy and ρ vs mean time/epoch with zoomed y-axes."""
    if df_summary is None or df_summary.empty:
        print("No summary data to plot.")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ---------- ρ vs accuracy ----------
    acc_pct = df_summary["mean_acc"] * 100.0
    acc_err = df_summary["std_acc"] * 100.0

    plt.figure(figsize=(6, 5))
    plt.errorbar(
        df_summary["rho"],
        acc_pct,
        yerr=acc_err,
        fmt="-o",
        capsize=4,
    )
    plt.xlabel("ρ")
    plt.ylabel("Final Test Accuracy (%)")
    plt.title("Generalization vs Neighborhood Radius ρ")
    plt.grid(True, alpha=0.5)

    # Zoom y-axis around the accuracy range with a small margin
    ymin = acc_pct.min()
    ymax = acc_pct.max()
    margin = max(1.0, 0.1 * (ymax - ymin))  # at least 1%
    plt.ylim(max(0.0, ymin - margin), min(100.0, ymax + margin))

    acc_path = Path(out_dir) / "rho_vs_accuracy.png"
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.close()
    print(f"Saved ρ vs accuracy plot to {acc_path}")

    # ---------- ρ vs time/epoch ----------
    mean_time = df_summary["mean_time"]
    time_err = df_summary["std_time"]

    plt.figure(figsize=(6, 5))
    plt.errorbar(
        df_summary["rho"],
        mean_time,
        yerr=time_err,
        fmt="-o",
        capsize=4,
    )
    plt.xlabel("ρ")
    plt.ylabel("Mean Seconds per Epoch")
    plt.title("Training Cost vs Neighborhood Radius ρ")
    plt.grid(True, alpha=0.5)

    # Zoom y-axis for time as well
    tmin = mean_time.min()
    tmax = mean_time.max()
    t_margin = max(0.1, 0.1 * (tmax - tmin))
    plt.ylim(max(0.0, tmin - t_margin), tmax + t_margin)

    time_path = Path(out_dir) / "rho_vs_time.png"
    plt.tight_layout()
    plt.savefig(time_path, dpi=200)
    plt.close()
    print(f"Saved ρ vs time/epoch plot to {time_path}")


# ---------------------------------------------------------------------
# Loss landscape plotting
# ---------------------------------------------------------------------
def load_checkpoint_for_rho(rho: float, seed: str = "seed_1"):
    """
    Locate a best checkpoint for a given ρ.
    """
    # Baseline SGD
    if abs(rho - 0.0) < 1e-8:
        path = os.path.join(RUNS_DIR, "baseline_sgd", "best.pt")

    # Special case: finished legacy SAM (ρ=0.05)
    elif abs(rho - 0.05) < 1e-8:
        # Note: Legacy SAM saves checkpoint inside a 'checkpoint' dir, not in runs
        path = os.path.join("checkpoint", "sam_rho005_best_ckpt.pth")

    # All other ρ values use the new runs directory structure
    else:
        # Format directory name to match the expected format 'rho_0.xx'
        dir_name = f"rho_{rho:.2f}".replace('.', '') 
        path = os.path.join(RUNS_DIR, dir_name, seed, "best.pt")

    if not os.path.exists(path):
        print(f"Checkpoint not found for ρ={rho} at {path}")
        return None
    return path


def plot_loss_landscape(
    rho: float,
    seed: str,
    data_dir: str = "./data",
    steps: int = 21,
    scale: float = 0.5,
    out_dir: str = PLOTS_DIR,
):
    """
    Compute a 2D loss landscape slice around the final weights for a given (rho, seed).
    """
    ckpt_path = load_checkpoint_for_rho(rho, seed)
    if ckpt_path is None:
        return

    # Use CPU for higher precision and deterministic behavior
    device = torch.device("cpu")
    print(f"Using device {device} for landscape, loading {ckpt_path}")

    # Model definition is local due to self-containment requirement
    model = resnet18_cifar().to(device)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "net" in ckpt:  # legacy SAM checkpoint format
            state_dict = ckpt["net"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # --- KEY REMAPPING (Crucial for ResNet) ---
    # The custom ResNet uses 'shortcut', but some standard checkpoints might use 'downsample'
    # We use the structure defined locally, which uses 'shortcut'
    
    # We need to map *from* the external format *to* our local format
    # The existing key remapping logic attempts to fix this, but let's ensure
    # the keys in the loaded state_dict match the keys in the model.
    model_keys = model.state_dict().keys()
    
    if any("downsample" in k for k in state_dict.keys()) and not any("shortcut" in k for k in state_dict.keys()):
        # Loaded state uses 'downsample', but our model expects 'shortcut'
        mapped = {}
        for k, v in state_dict.items():
            new_k = k.replace("downsample", "shortcut")
            mapped[new_k] = v
        state_dict = mapped

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        # Print a few key mismatches for debugging
        # print("Model Keys:", set(model_keys) - set(state_dict.keys()))
        # print("State Keys:", set(state_dict.keys()) - set(model_keys))
        return

    model.eval()

    # Data: take one batch from test set
    _, test_loader = get_cifar100_loaders(
        data_dir, batch_size=256, num_workers=0, use_aug=False
    )
    # The loader is expected to exist, but might fail if data isn't downloaded.
    try:
        x, y = next(iter(test_loader))
    except StopIteration:
        print("Error: Test data loader is empty. Cannot compute landscape.")
        return
        
    x, y = x.to(device), y.to(device)

    # Flatten parameters
    params = [p for p in model.parameters() if p.requires_grad]
    w = torch.cat([p.detach().flatten() for p in params])
    w_norm = w.norm().item()

    # Random directions in weight space (d1, d2)
    # Generate and orthonormalize directions
    d1 = torch.randn_like(w)
    d2 = torch.randn_like(w)
    d1 = d1 / (d1.norm() + 1e-12)
    d2 = d2 - torch.dot(d1, d2) * d1
    d2 = d2 / (d2.norm() + 1e-12)

    # Radius of exploration = scale * ‖w‖
    radius = scale * w_norm

    alphas = torch.linspace(-1.0, 1.0, steps)
    betas = torch.linspace(-1.0, 1.0, steps)
    loss_grid = torch.zeros(steps, steps)

    def set_params(vec):
        idx = 0
        with torch.no_grad():
            for p in params:
                numel = p.numel()
                p.copy_(vec[idx : idx + numel].view_as(p))
                idx += numel

    w0 = w.clone()

    print("Computing loss landscape...")
    # 
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            # Move along 2D plane with meaningful radius
            w_perturbed = w0 + radius * (a * d1 + b * d2)
            set_params(w_perturbed)
            with torch.no_grad():
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            loss_grid[i, j] = loss.item()

    # Restore original weights
    set_params(w0)

    # Plot contour
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    A, B = np.meshgrid(alphas.numpy(), betas.numpy())
    Z = loss_grid.numpy()

    plt.figure(figsize=(6, 5))
    cs = plt.contourf(A, B, Z, levels=30)
    plt.colorbar(cs, label="Loss")
    # Mark the center (w0)
    plt.plot(0, 0, 'rx', markersize=10, markeredgewidth=3, label='Final Solution $w^*$')
    plt.xlabel("α (direction 1)")
    plt.ylabel("β (direction 2)")
    plt.title(f"Loss Landscape around solution (ρ={rho}, {seed})")
    plt.legend()
    out_path = Path(out_dir) / f"loss_landscape_rho{rho:.2f}_{seed}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved loss landscape plot to {out_path}")


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Plotting script for SAM experiments."
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Plot only the legacy SAM (rho=0.05) curves.",
    )
    parser.add_argument(
        "--compare-legacy",
        action="store_true",
        help="Plot legacy baseline vs legacy SAM comparison.",
    )
    parser.add_argument(
        "--rho-summary",
        action="store_true",
        help="Plot ρ vs accuracy and ρ vs mean epoch time from ./runs.",
    )
    parser.add_argument(
        "--landscape",
        action="store_true",
        help="Plot a 2D loss landscape slice for a given ρ and seed.",
    )
    parser.add_argument("--landscape-rho", type=float, default=0.0)
    parser.add_argument("--landscape-seed", type=str, default="seed_1")
    parser.add_argument("--data-dir", type=str, default="./data")

    args = parser.parse_args()

    # 1) Legacy behavior
    if args.single_run or args.compare_legacy:
        sam_history, sam_rho = load_data(SAM_HISTORY_FILE, "pth")

        if args.single_run:
            plot_single_run(sam_history, sam_rho)
        else:
            baseline_history, baseline_rho = load_data(
                BASELINE_METRICS_FILE, "csv"
            )
            plot_comparison(baseline_history, sam_history, baseline_rho, sam_rho)

    # 2) ρ-sweep summary (uses ./runs)
    if args.rho_summary:
        runs_dict = discover_rho_runs(RUNS_DIR)
        df_summary = summarize_rho_stats(runs_dict)
        plot_rho_summary(df_summary, PLOTS_DIR)

    # 3) Loss landscape
    if args.landscape:
        # Scale of 1.0 is a reasonable default for weight space visualization
        plot_loss_landscape(
            rho=args.landscape_rho,
            seed=args.landscape_seed,
            data_dir=args.data_dir,
            steps=21,
            scale=1.0, 
            out_dir=PLOTS_DIR,
        )


if __name__ == "__main__":
    main()