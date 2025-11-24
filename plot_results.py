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



import torch
import matplotlib.pyplot as plt
import os
import pandas as pd # REQUIRED for reading the new baseline metrics.csv
import csv
import argparse
from pathlib import Path

# --- Define File Paths ---
# This assumes the SAM data for rho=0.05 is still in the legacy PyTorch format
SAM_HISTORY_FILE = './logs/sam_rho005_history.pth'
# This assumes baseline data is now saved as a CSV in the run directory
BASELINE_METRICS_FILE = './runs/baseline_sgd/metrics.csv'
# Output directory for plots
PLOTS_DIR = './plots'

def load_data(filepath, data_type):
    """Loads history data from either a PyTorch .pth file (legacy SAM) or a CSV file (new baseline)."""
    if not os.path.exists(filepath):
        print(f"Error: History file not found at {filepath}")
        return None, None

    try:
        if data_type == 'pth':
            # --- LEGACY SAM FORMAT (.pth) ---
            # Used for existing SAM runs until train_sam.py is updated to use CSV.
            data = torch.load(filepath)
            rho = data.get('rho', 'N/A')
            
            # Map legacy format keys to standardized keys for plotting
            # Note: Legacy train_sam.py saves acc as % (0-100), we convert to fraction (0-1) 
            # for comparison plotting, but keep as % for single-run plotting titles.
            history = {
                'epochs': list(range(1, len(data['test_acc']) + 1)),
                'train_loss': data['train_loss'],
                'val_loss': data['test_loss'], 
                # Store as fraction (0-1) for comparison mode
                'train_acc_frac': [a / 100 for a in data['train_acc']], 
                'val_acc_frac': [a / 100 for a in data['test_acc']],
                # Store as percentage (0-100) for single run mode
                'train_acc_pct': data['train_acc'],
                'val_acc_pct': data['test_acc'],
                'epoch_time_sec': data['runtime_per_epoch']
            }
            print(f"Successfully loaded legacy data (PTH) for ρ = {rho}")
            return history, rho
        
        elif data_type == 'csv':
            # --- NEW BASELINE FORMAT (metrics.csv) ---
            df = pd.read_csv(filepath)
            
            # Infer rho for baseline (0.0)
            rho = 0.0 

            # Map CSV columns to dictionary keys
            history = {
                'epochs': df['epoch'].tolist(),
                'train_loss': df['train_loss'].tolist(),
                'val_loss': df['val_loss'].tolist(),
                # CSV already stores accuracy as fraction (0-1)
                'train_acc_frac': df['train_acc'].tolist(),
                'val_acc_frac': df['val_acc'].tolist(),
                'epoch_time_sec': df['epoch_time_sec'].tolist()
            }
            print(f"Successfully loaded new data (CSV) for ρ = {rho}")
            return history, rho
        
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return None, None
        
    return None, None

def plot_single_run(history, rho):
    """Generates and saves loss and accuracy plots for a single SAM run."""
    
    if history is None:
        print("Cannot generate plots: History data is missing.")
        return

    epochs = history['epochs']
    
    # Create the figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- LOSS PLOT (Left Subplot) ---
    ax1.plot(epochs, history['train_loss'], label='Train Loss', linestyle='--', color='C0')
    ax1.plot(epochs, history['val_loss'], label='Test Loss', color='C0', linewidth=2)
    
    ax1.set_title(f'SAM ($\\rho$={rho}) Loss vs. Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.5)

    # --- ACCURACY PLOT (Right Subplot) ---
    # Use the percentage (0-100) data loaded from the PTH file
    ax2.plot(epochs, history['train_acc_pct'], label='Train Acc', linestyle='--', color='C1')
    ax2.plot(epochs, history['val_acc_pct'], label='Test Acc', color='C1', linewidth=2)
    
    ax2.set_title(f'SAM ($\\rho$={rho}) Accuracy vs. Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.5)

    # Save the figure
    PLOT_FILENAME = 'sam_rho005_curves.png'
    output_path = Path(PLOTS_DIR) / PLOT_FILENAME
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig) # Close the figure to free memory

    print(f"\nSuccessfully saved single run plot to {output_path}")

def plot_comparison(baseline_history, sam_history, baseline_rho, sam_rho):
    """Generates and displays loss and accuracy plots for the comparison run."""
    
    # Check if we have data for both runs
    if baseline_history is None or sam_history is None:
        print("Cannot generate plots: Data for one or both runs is missing.")
        return

    # Use the length of the shortest history to prevent indexing errors if epochs differ
    min_epochs = min(len(baseline_history['epochs']), len(sam_history['epochs']))
    epochs = list(range(1, min_epochs + 1))
    
    # --- LOSS PLOT ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # Baseline plots
    plt.plot(epochs, baseline_history['train_loss'][:min_epochs], label=f'SGD (ρ={baseline_rho}) Train Loss', linestyle='--')
    plt.plot(epochs, baseline_history['val_loss'][:min_epochs], label=f'SGD (ρ={baseline_rho}) Test Loss', color='C0')
    # SAM plots
    plt.plot(epochs, sam_history['train_loss'][:min_epochs], label=f'SAM (ρ={sam_rho}) Train Loss', linestyle='--')
    plt.plot(epochs, sam_history['val_loss'][:min_epochs], label=f'SAM (ρ={sam_rho}) Test Loss', color='C1')
    
    plt.title('Training and Test Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.5)

    # --- ACCURACY PLOT ---
    plt.subplot(1, 2, 2)
    # Note: Accuracy is plotted as a percentage (0-100), so we use the fraction data and multiply by 100
    # Baseline plots
    plt.plot(epochs, [a * 100 for a in baseline_history['train_acc_frac'][:min_epochs]], label=f'SGD (ρ={baseline_rho}) Train Acc', linestyle='--')
    plt.plot(epochs, [a * 100 for a in baseline_history['val_acc_frac'][:min_epochs]], label=f'SGD (ρ={baseline_rho}) Test Acc', color='C0')
    # SAM plots
    plt.plot(epochs, [a * 100 for a in sam_history['train_acc_frac'][:min_epochs]], label=f'SAM (ρ={sam_rho}) Train Acc', linestyle='--')
    plt.plot(epochs, [a * 100 for a in sam_history['val_acc_frac'][:min_epochs]], label=f'SAM (ρ={sam_rho}) Test Acc', color='C1')
    
    plt.title('Training and Test Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show() # Display the comparison plot in the standard view

def main():
    parser = argparse.ArgumentParser(description="Plotting script for SAM and Baseline results.")
    parser.add_argument("--single-run", action="store_true", help="If set, only plots the single SAM (rho=0.05) run.")
    args = parser.parse_args()

    # 1. Load SAM data (Reads the legacy PTH file)
    sam_history, sam_rho = load_data(SAM_HISTORY_FILE, 'pth')

    if args.single_run:
        # 2a. Plot only the SAM run
        plot_single_run(sam_history, sam_rho)
    else:
        # 2b. Load baseline data (Reads the new CSV file)
        baseline_history, baseline_rho = load_data(BASELINE_METRICS_FILE, 'csv')
        
        # 3. Plot comparison results
        plot_comparison(baseline_history, sam_history, baseline_rho, sam_rho)

if __name__ == '__main__':
    main()