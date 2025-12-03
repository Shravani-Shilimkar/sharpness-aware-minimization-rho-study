# import torch
# import matplotlib.pyplot as plt
# import os
# import numpy as np

# # --- Configuration ---
# # IMPORTANT: Update BASELINE_HISTORY_PATH if Aditi's script saved the baseline (rho=0.0) 
# # history file with a different name or path.
# # BASELINE_HISTORY_PATH = './runs/baseline_sgd/baseline_history.pth' 
# BASELINE_HISTORY_PATH = './runs/baseline_sgd/baseline_history.pth'
# SAM_HISTORY_PATH = './logs/sam_rho005_history.pth'
# PLOT_DIR = './plots'

# def load_history(filepath):
#     """Loads training history data from a PyTorch .pth file."""
#     if not os.path.exists(filepath):
#         print(f"Error: History file not found at {filepath}")
#         return None
#     try:
#         data = torch.load(filepath)
#         print(f"Successfully loaded data for ρ = {data.get('rho', 'N/A')}")
#         return data
#     except Exception as e:
#         print(f"Error loading {filepath}: {e}")
#         return None

# def generate_comparison_plots(baseline_data, sam_data):
#     """Generates and saves comparison plots for loss and accuracy."""
    
#     if baseline_data is None or sam_data is None:
#         print("Cannot generate plots: Missing data from one or both runs.")
#         return

#     os.makedirs(PLOT_DIR, exist_ok=True)
#     # Use the length of the shortest history to prevent errors if runs had different epoch counts
#     min_epochs = min(len(baseline_data['train_loss']), len(sam_data['train_loss']))
#     epochs = np.arange(1, min_epochs + 1)
    
#     # --- 1. Loss Comparison Plot ---
#     plt.figure(figsize=(12, 6))
    
#     # Training Loss
#     plt.plot(epochs, baseline_data['train_loss'][:min_epochs], label='SGD Train Loss (ρ=0.0)', linestyle='--')
#     plt.plot(epochs, sam_data['train_loss'][:min_epochs], label='SAM Train Loss (ρ=0.05)', linestyle='-')
    
#     # Validation/Test Loss
#     plt.plot(epochs, baseline_data['test_loss'][:min_epochs], label='SGD Test Loss (ρ=0.0)', linestyle='--')
#     plt.plot(epochs, sam_data['test_loss'][:min_epochs], label='SAM Test Loss (ρ=0.05)', linestyle='-')
    
#     plt.title('Training and Test Loss Comparison (SGD vs. SAM, ρ=0.05)')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss (CrossEntropy)')
#     plt.legend()
#     plt.grid(True, linestyle='dotted', alpha=0.7)
#     loss_plot_path = os.path.join(PLOT_DIR, 'loss_comparison.png')
#     plt.savefig(loss_plot_path)
#     plt.close()
#     print(f"Loss comparison plot saved to {loss_plot_path}")


#     # --- 2. Accuracy Comparison Plot ---
#     plt.figure(figsize=(12, 6))
    
#     # Training Accuracy
#     plt.plot(epochs, baseline_data['train_acc'][:min_epochs], label='SGD Train Accuracy (ρ=0.0)', linestyle='--')
#     plt.plot(epochs, sam_data['train_acc'][:min_epochs], label='SAM Train Accuracy (ρ=0.05)', linestyle='-')

#     # Validation/Test Accuracy
#     plt.plot(epochs, baseline_data['test_acc'][:min_epochs], label='SGD Test Accuracy (ρ=0.0)', linestyle='--')
#     plt.plot(epochs, sam_data['test_acc'][:min_epochs], label='SAM Test Accuracy (ρ=0.05)', linestyle='-')
    
#     plt.title('Training and Test Accuracy Comparison (SGD vs. SAM, ρ=0.05)')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
#     plt.legend()
#     plt.grid(True, linestyle='dotted', alpha=0.7)
#     acc_plot_path = os.path.join(PLOT_DIR, 'accuracy_comparison.png')
#     plt.savefig(acc_plot_path)
#     plt.close()
#     print(f"Accuracy comparison plot saved to {acc_plot_path}")


# def main():
#     # 1. Load data
#     baseline_data = load_history(BASELINE_HISTORY_PATH)
#     sam_data = load_history(SAM_HISTORY_PATH)
    
#     # 2. Generate plots
#     generate_comparison_plots(baseline_data, sam_data)

# if __name__ == '__main__':
#     # Ensure Matplotlib is configured to run without display server (for headless environments)
#     try:
#         plt.switch_backend('Agg')
#     except ImportError:
#         pass 
        
#     main()


# import torch
# import matplotlib.pyplot as plt
# import os
# import pandas as pd # REQUIRED for reading the new baseline metrics.csv
# import csv

# # --- Define File Paths ---
# # This assumes the SAM data for rho=0.05 is still in the legacy PyTorch format
# SAM_HISTORY_FILE = './logs/sam_rho005_history.pth'
# # This assumes baseline data is now saved as a CSV in the run directory
# BASELINE_METRICS_FILE = './runs/baseline_sgd/metrics.csv'

# def load_data(filepath, data_type):
#     """Loads history data from either a PyTorch .pth file (legacy SAM) or a CSV file (new baseline)."""
#     if not os.path.exists(filepath):
#         print(f"Error: History file not found at {filepath}")
#         return None, None

#     try:
#         if data_type == 'pth':
#             # --- LEGACY SAM FORMAT (.pth) ---
#             # Used for existing SAM runs until train_sam.py is updated to use CSV.
#             data = torch.load(filepath)
#             rho = data.get('rho', 'N/A')
            
#             # Map legacy format keys to standardized keys for plotting
#             # Note: Legacy train_sam.py saves acc as % (0-100), we convert to fraction (0-1)
#             history = {
#                 'epochs': list(range(1, len(data['test_acc']) + 1)),
#                 'train_loss': data['train_loss'],
#                 'val_loss': data['test_loss'], 
#                 'train_acc': [a / 100 for a in data['train_acc']], 
#                 'val_acc': [a / 100 for a in data['test_acc']],   
#                 'epoch_time_sec': data['runtime_per_epoch']
#             }
#             print(f"Successfully loaded legacy data (PTH) for ρ = {rho}")
#             return history, rho
        
#         elif data_type == 'csv':
#             # --- NEW BASELINE FORMAT (metrics.csv) ---
#             df = pd.read_csv(filepath)
            
#             # Infer rho for baseline (0.0)
#             rho = 0.0 

#             # Map CSV columns to dictionary keys
#             history = {
#                 'epochs': df['epoch'].tolist(),
#                 'train_loss': df['train_loss'].tolist(),
#                 'val_loss': df['val_loss'].tolist(),
#                 'train_acc': df['train_acc'].tolist(),
#                 'val_acc': df['val_acc'].tolist(),
#                 'epoch_time_sec': df['epoch_time_sec'].tolist()
#             }
#             print(f"Successfully loaded new data (CSV) for ρ = {rho}")
#             return history, rho
        
#     except Exception as e:
#         print(f"Error loading data from {filepath}: {e}")
#         return None, None
        
#     return None, None

# def plot_history(baseline_history, sam_history, baseline_rho, sam_rho):
#     """Generates and displays loss and accuracy plots."""
    
#     # Check if we have data for both runs
#     if baseline_history is None or sam_history is None:
#         print("Cannot generate plots: Data for one or both runs is missing.")
#         return

#     # Use the length of the shortest history to prevent indexing errors if epochs differ
#     min_epochs = min(len(baseline_history['epochs']), len(sam_history['epochs']))
#     epochs = list(range(1, min_epochs + 1))
    
#     # --- LOSS PLOT ---
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     # Baseline plots
#     plt.plot(epochs, baseline_history['train_loss'][:min_epochs], label=f'SGD (ρ={baseline_rho}) Train Loss', linestyle='--')
#     plt.plot(epochs, baseline_history['val_loss'][:min_epochs], label=f'SGD (ρ={baseline_rho}) Test Loss', color='C0')
#     # SAM plots
#     plt.plot(epochs, sam_history['train_loss'][:min_epochs], label=f'SAM (ρ={sam_rho}) Train Loss', linestyle='--')
#     plt.plot(epochs, sam_history['val_loss'][:min_epochs], label=f'SAM (ρ={sam_rho}) Test Loss', color='C1')
    
#     plt.title('Training and Test Loss Comparison')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True, alpha=0.5)

#     # --- ACCURACY PLOT ---
#     plt.subplot(1, 2, 2)
#     # Note: Accuracy is plotted as a percentage (0-100), so we multiply by 100
#     # Baseline plots
#     plt.plot(epochs, [a * 100 for a in baseline_history['train_acc'][:min_epochs]], label=f'SGD (ρ={baseline_rho}) Train Acc', linestyle='--')
#     plt.plot(epochs, [a * 100 for a in baseline_history['val_acc'][:min_epochs]], label=f'SGD (ρ={baseline_rho}) Test Acc', color='C0')
#     # SAM plots
#     plt.plot(epochs, [a * 100 for a in sam_history['train_acc'][:min_epochs]], label=f'SAM (ρ={sam_rho}) Train Acc', linestyle='--')
#     plt.plot(epochs, [a * 100 for a in sam_history['val_acc'][:min_epochs]], label=f'SAM (ρ={sam_rho}) Test Acc', color='C1')
    
#     plt.title('Training and Test Accuracy Comparison')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
#     plt.legend()
#     plt.grid(True, alpha=0.5)

#     plt.tight_layout()
#     plt.show()

# def main():
#     # 1. Load baseline data (Reads the new CSV file)
#     baseline_history, baseline_rho = load_data(BASELINE_METRICS_FILE, 'csv')
    
#     # 2. Load SAM data (Reads the legacy PTH file)
#     sam_history, sam_rho = load_data(SAM_HISTORY_FILE, 'pth')
    
#     # 3. Plot results
#     plot_history(baseline_history, sam_history, baseline_rho, sam_rho)

# if __name__ == '__main__':
#     main()

import os
import csv
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset import get_cifar100_loaders
from model import resnet18_cifar


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
SAM_HISTORY_FILE = "./logs/sam_rho005_history.pth"      # legacy .pth history
BASELINE_METRICS_FILE = "./runs/baseline_sgd/metrics.csv"
RUNS_DIR = "./runs"
PLOTS_DIR = "./plots"


# ---------------------------------------------------------------------
# Legacy loaders 
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
    stored in logs/sam_rho005_history.pth, instead of using the incomplete run
    under runs/rho_0.05.
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
        mask = np.isclose(df_summary["rho"], 0.05)
        df_summary = df_summary[~mask]

    df_summary = pd.concat([df_summary, pd.DataFrame([new_row])], ignore_index=True)
    return df_summary



def summarize_rho_stats(runs_dict):
    """
    Build a summary DataFrame:
    columns: rho, num_seeds, mean_acc, std_acc, mean_time, std_time

    - Aggregates completed runs found under ./runs
    - Then overrides rho=0.05 with the legacy finished SAM run
      from logs/sam_rho005_history.pth
    - Finally filters to the rhos we actually want to plot: {0.0, 0.05, 0.10, 0.20}
    """
    rows = []
    for rho in sorted(runs_dict.keys()):
        dfs = [df for (_, df) in runs_dict[rho]]
        if not dfs:
            continue

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

    # 1) Override rho=0.05 with the finished legacy SAM run
    df_summary = override_with_legacy_rho005(df_summary)

    # 2) Keep ONLY the rhos we care about: 0.0, 0.05, 0.10, 0.20
    allowed_rhos = [0.0, 0.05, 0.10, 0.20]
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
      - ρ = 0.0 : ./runs/baseline_sgd/best.pt         (baseline SGD)
      - ρ = 0.05: ./checkpoint/sam_rho005_best_ckpt.pth (finished legacy SAM run)
      - other ρ : ./runs/rho_xx/seed_YYY/best.pt      (new SAM runs)
    """
    # Baseline SGD
    if abs(rho - 0.0) < 1e-8:
        path = os.path.join(RUNS_DIR, "baseline_sgd", "best.pt")

    # Special case: finished legacy SAM (ρ=0.05)
    elif abs(rho - 0.05) < 1e-8:
        path = os.path.join("checkpoint", "sam_rho005_best_ckpt.pth")

    # All other ρ values use the new runs directory structure
    else:
        dir_name = f"rho_{rho:.2f}"
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
    scale: float = 0.5,   # interpreted as fraction of ‖w‖
    out_dir: str = PLOTS_DIR,
):
    """
    Compute a 2D loss landscape slice around the final weights for a given (rho, seed).
    Uses a single batch of test data for efficiency.
    """
    ckpt_path = load_checkpoint_for_rho(rho, seed)
    if ckpt_path is None:
        return

    # IMPORTANT: use CPU for higher precision and deterministic behavior
    device = torch.device("cpu")
    print(f"Using device {device} for landscape, loading {ckpt_path}")

    # Model
    model = resnet18_cifar().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    # Handle different checkpoint formats:
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "net" in ckpt:  # legacy SAM checkpoint
            state_dict = ckpt["net"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # --- KEY REMAPPING FOR LEGACY RESNET ('shortcut' -> 'downsample') ---
    if any("shortcut" in k for k in state_dict.keys()):
        mapped = {}
        for k, v in state_dict.items():
            new_k = k.replace("shortcut", "downsample")
            mapped[new_k] = v
        state_dict = mapped

    model.load_state_dict(state_dict)
    model.eval()

    # Data: take one batch from test set
    _, test_loader = get_cifar100_loaders(
        data_dir, batch_size=256, num_workers=0, use_aug=False
    )
    x, y = next(iter(test_loader))
    x, y = x.to(device), y.to(device)

    # Flatten parameters
    params = [p for p in model.parameters() if p.requires_grad]
    w = torch.cat([p.detach().flatten() for p in params])
    w_norm = w.norm().item()

    # Random directions in weight space
    d1 = torch.randn_like(w)
    d2 = torch.randn_like(w)

    # Orthonormalize directions
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

    print("Computing loss landscape ...")
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            # move along 2D plane with meaningful radius
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
    plt.xlabel("α (direction 1)")
    plt.ylabel("β (direction 2)")
    plt.title(f"Loss Landscape around solution (ρ={rho}, {seed})")
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
