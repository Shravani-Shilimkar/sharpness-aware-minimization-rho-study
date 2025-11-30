import os
import csv
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_cifar100_loaders, get_cifar100c_loaders
from model import resnet18_cifar
from sam_optimizer import SAM   


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        loss_sum += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        total += bs
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate_cifar100c(model, cifar100c_loaders, device):
    """Compute corruption error per corruption and mean corruption error (mCE)."""
    corr_err = {}
    for corr_name, sev_loaders in cifar100c_loaders.items():
        correct, total = 0, 0
        for loader in sev_loaders:
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        corr_err[corr_name] = 1.0 - acc
    mCE = float(np.mean(list(corr_err.values())))
    return corr_err, mCE


def train_one_epoch(model, loader, criterion, optimizer, device, rho: float):
    """
    One epoch of training.

    - If rho == 0.0: standard SGD step().
    - If rho > 0.0: SAM two-step (first_step / second_step).
    """
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    t0 = time.time()

    if rho == 0.0:
        # -------- Baseline SGD --------
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            loss_sum += loss.item() * bs
            correct += (logits.argmax(1) == y).sum().item()
            total += bs

    else:
        # -------- SAM training --------
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            def closure():
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                return loss

            loss = optimizer.step(closure)  # single call, SAM does two passes internally

            bs = x.size(0)
            loss_sum += loss.item() * bs
            correct += (model(x).argmax(1) == y).sum().item()  # or cache logits if you want
            total += bs

    return loss_sum / total, correct / total, time.time() - t0


# ---------------------------------------------------------
# One experiment: given rho and seed
# ---------------------------------------------------------

def run_single(rho: float, seed: int, args, device):
    seed_everything(seed)

    # ---------- output directory ----------
    out_dir = Path(args.out_dir) / f"rho_{rho:.2f}" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # save config
    with open(out_dir / "config.json", "w") as f:
        json.dump({**vars(args), "rho": rho, "seed": seed}, f, indent=2)

    # ---------- data ----------
    train_loader, test_loader = get_cifar100_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_aug=True,
    )

    # ---------- model ----------
    model = resnet18_cifar().to(device)
    criterion = nn.CrossEntropyLoss()

    # ---------- optimizer ----------
    if rho == 0.0:
        # baseline SGD
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        # SAM wrapper around SGD
        # NOTE: if your SAM constructor signature differs, adjust this line.
        optimizer = SAM(
            model.parameters(),
            base_optimizer=optim.SGD,
            rho=rho,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---------- CSV logger ----------
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "train_loss", "train_acc",
             "val_loss", "val_acc", "lr", "epoch_time_sec"]
        )

    best_val_acc = 0.0

    # ---------- training loop ----------
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, sec = train_one_epoch(
            model, train_loader, criterion, optimizer, device, rho
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()
        lr = scheduler.optimizer.param_groups[0]["lr"]

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch, tr_loss, tr_acc, val_loss, val_acc, lr, sec]
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict()},
                out_dir / "best.pt",
            )

        print(
            f"[rho={rho:.2f} seed={seed}] "
            f"Epoch {epoch:03d} | "
            f"TL {tr_loss:.4f} TA {tr_acc:.4f} | "
            f"VL {val_loss:.4f} VA {val_acc:.4f} | "
            f"{sec:.1f}s | lr={lr:.5f}"
        )

    # save final checkpoint
    torch.save(
        {"epoch": args.epochs, "state_dict": model.state_dict()},
        out_dir / "last.pt",
    )

    # ---------- CIFAR-100-C robustness ----------
    if os.path.exists(args.cifar100c_dir):
        cifar100c_loaders = get_cifar100c_loaders(
            args.cifar100c_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        corr_err, mCE = evaluate_cifar100c(model, cifar100c_loaders, device)

        with open(out_dir / "cifar100c_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["corruption", "error"])
            for name, err in corr_err.items():
                writer.writerow([name, err])
            writer.writerow(["mCE", mCE])

        print(f"[rho={rho:.2f} seed={seed}] CIFAR-100-C mCE = {mCE:.4f}")
    else:
        print(
            f"[rho={rho:.2f} seed={seed}] "
            f"Skipping CIFAR-100-C; directory not found: {args.cifar100c_dir}"
        )

    print(f"Done. Saved run to {out_dir}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", type=float, required=True,
                        help="SAM neighborhood size; 0.0 = baseline SGD")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--cifar100c-dir", type=str, default="./cifar100-c")
    parser.add_argument("--out-dir", type=str, default="./runs")

    args = parser.parse_args()
    device = get_device()
    print("Using device:", device)

    run_single(args.rho, args.seed, args, device)


if __name__ == "__main__":
    main()
