# Empirical Analysis of Sharpness-Aware Minimization: Effect of Neighborhood Radius ρ

This repository contains code and experiments for a controlled empirical study on **Sharpness-Aware Minimization (SAM)**. The goal is to understand how the **neighborhood radius ρ** affects generalization and loss landscape geometry when training ResNet-18 on CIFAR-100.

---

## Overview

Sharpness-Aware Minimization minimizes the **worst-case loss** within a neighborhood of the current model parameters. The radius ρ controls how large that neighborhood is:

- ρ = 0.00 → equivalent to standard SGD  
- small ρ → encourages flatter minima and improved generalization  
- large ρ → over-regularization and reduced performance  

We perform a sweep over:

```
ρ ∈ {0.00, 0.01, 0.05, 0.10, 0.20}
```

All other training settings are held constant.

We report:

- CIFAR-100 accuracy vs ρ  
- Training time vs ρ  
- Learning curves  
- Loss landscapes  

---

## Repository Structure

```
.
├── checkpoint/                 # Best model checkpoints
├── data/                       # CIFAR-100 dataset cache
├── logs/                       # CSV logs per epoch
├── plots/                      # Generated figures
│   ├── Accuracy & loss comparison.png
│   ├── rho_vs_accuracy.png
│   ├── rho_vs_time.png
│   ├── sam_rho005_curves.png
│   └── loss_landscape_*.png
├── runs/                       # Run outputs organized by ρ and seed
│   ├── baseline_sgd/
│   ├── rho_0.01/
│   ├── rho_0.05/
    ├── rho_0.10/
│   └── rho_0.20/
├── dataset.py
├── experiments.py
├── model.py
├── plot_results.py
├── sam_optimizer.py
├── train_baseline.py
└── train_sam.py
```

---

## Installation

### Requirements

- Python ≥ 3.9  
- PyTorch ≥ 2.0  
- Torchvision ≥ 0.15  

Install dependencies:

```bash
pip install -r requirements.txt
```

If a requirements file is not provided:

```bash
pip install torch torchvision matplotlib numpy
```

> The code supports Apple Silicon via the MPS backend in PyTorch.

---

## Training

### Baseline (SGD)

Train ResNet-18 with standard SGD (ρ = 0):

```bash
python train_baseline.py
```

Outputs:

- logs/baseline_sgd/
- checkpoint/baseline_sgd/
- runs/baseline_sgd/

---

### SAM Experiments (ρ-sweep)

Train with SAM over multiple ρ values:

```bash
python train_sam.py
```

This script:

- sweeps ρ over {0.00, 0.01, 0.05, 0.10, 0.20}
- runs two seeds per configuration
- logs metrics to CSV
- saves best checkpoints
- organizes runs by ρ and seed

Adjust seeds, epochs, and hyperparameters in experiments.py.

---

## Results and Plots

Generate plots after training:

```bash
python plot_results.py
```

Key plots:

- rho_vs_accuracy.png
- rho_vs_time.png
- sam_rho005_curves.png
- loss_landscape_*.png

---

## Key Findings

- ρ ≈ 0.05 gives the best performance, improving test accuracy by ~1.2% over SGD.
- The relationship between ρ and accuracy is inverted-U shaped.
- SAM training is ~2× slower than SGD due to an extra forward/backward pass.
- Loss landscapes for ρ ∈ [0.05, 0.10] appear wider and flatter than SGD.

---

## Limitations

- Only 100 epochs (not 200) due to runtime constraints.  
- Only two seeds per configuration due to compute limits.  
- No CIFAR-100-C robustness evaluation included.  
- Designed as a research experiment, not a production model.

---

## Reproducibility

- Logs include: epoch, train loss, train acc, test loss, test acc, epoch time.
- Best checkpoints saved by test accuracy.
- Random seeds set for Python, NumPy, and PyTorch.
- Identical hyperparameters across all runs except ρ.

---

## Citation

If referencing SAM, cite:

```
@inproceedings{foret2021sam,
  title={Sharpness-Aware Minimization for Efficiently Improving Generalization},
  author={Foret, Pierre and Kleiner, Ariel and Mobahi, Hossein and Neyshabur, Behnam},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

---

## Acknowledgements

- SAM: Foret et al., ICLR 2021  
- Dataset: CIFAR-100  
- Backbone: ResNet-18  

This project was completed for EE641 — Deep Learning Systems.
