"""
model/train.py
==============
Training loop for the DocClean-Net U-Net.

Usage::

    python -m model.train --epochs 50 --batch 8 --lr 1e-3 --data data/synthetic/

Checkpoints are saved to ``checkpoints/best.pt`` (lowest validation loss).
Training log is printed to stdout, one line per epoch.

Split: 90% train / 10% validation, random with fixed seed (reproducible).

Design decisions (do not reopen without reason):
    - CPU-first: no CUDA assumption. Uses ``torch.device`` auto-detection so
      GPU is used automatically when available without requiring the user to
      pass a flag.
    - DataLoader num_workers=0 on Windows (multiprocessing issues with PyTorch
      spawn context); on Linux/macOS uses min(4, cpu_count).
    - Validation uses augment=False so metrics are deterministic epoch-to-epoch.
    - best.pt stores the full state dict, not a traced model, for simplicity
      and compatibility with future fine-tuning.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model.dataset import DocCleanDataset
from model.losses import CombinedLoss
from model.unet import UNet


# ── Constants ─────────────────────────────────────────────────────────────────

SPLIT_SEED     = 42   # fixed seed for train/val split — never change
TRAIN_FRACTION = 0.9
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_NAME = "best.pt"


# ── Training utilities ────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_num_workers() -> int:
    """Return 0 on Windows (avoids DataLoader spawn issues), else min(4, cpus)."""
    if platform.system() == "Windows":
        return 0
    return min(4, os.cpu_count() or 1)


def _make_dataloaders(
    data_dir: Path,
    patch_size: int,
    batch_size: int,
    dataset_seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from a synthetic dataset directory.

    Args:
        data_dir (Path): Root with ``dirty/`` and ``clean/`` subdirectories.
        patch_size (int): Patch size passed to DocCleanDataset.
        batch_size (int): Mini-batch size for training.
        dataset_seed (int): Seed for patch extraction RNG inside the dataset.

    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    full_dataset = DocCleanDataset(
        data_dir, patch_size=patch_size, augment=True, seed=dataset_seed
    )

    n_total = len(full_dataset)
    n_train = max(1, int(n_total * TRAIN_FRACTION))
    n_val   = n_total - n_train

    if n_val == 0:
        raise ValueError(
            f"Dataset has only {n_total} pairs — not enough for a 90/10 split. "
            "Generate more data or lower --val-split."
        )

    train_set, val_set = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SPLIT_SEED),
    )

    # Validation dataset: reuse same paths but disable augmentation.
    # random_split returns a Subset; we can't change augment= on it directly,
    # so we create a second dataset instance (no augmentation) and use the
    # same indices from the split.
    val_dataset_clean = DocCleanDataset(
        data_dir, patch_size=patch_size, augment=False, seed=dataset_seed
    )
    from torch.utils.data import Subset
    val_set = Subset(val_dataset_clean, val_set.indices)

    num_workers = _get_num_workers()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    """Run one full pass over a DataLoader.

    Args:
        model (nn.Module): The U-Net.
        loader (DataLoader): Train or validation loader.
        criterion (CombinedLoss): Loss function.
        optimizer: AdamW optimizer (pass ``None`` for validation).
        device (torch.device): Target device.

    Returns:
        float: Mean loss over all batches in this epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_batches  = 0

    desc = "train" if is_train else "val  "
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for dirty, clean in tqdm(loader, desc=desc, leave=False, unit="batch"):
            dirty = dirty.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            pred = model(dirty)
            loss = criterion(pred, clean)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.detach().item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


def train(
    data_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    patch_size: int,
    dataset_seed: int,
    checkpoint_dir: Path,
) -> None:
    """Full training loop.

    Args:
        data_dir (Path): Root of synthetic dataset (dirty/ + clean/).
        epochs (int): Number of training epochs.
        batch_size (int): Mini-batch size.
        lr (float): Initial learning rate for AdamW.
        patch_size (int): Size of patches extracted from each image.
        dataset_seed (int): RNG seed for patch extraction.
        checkpoint_dir (Path): Directory to save ``best.pt``.
    """
    device = _get_device()
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader = _make_dataloaders(
        data_dir, patch_size, batch_size, dataset_seed
    )
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    print(f"Train pairs: {n_train}  |  Val pairs: {n_val}")
    print(f"Batch size:  {batch_size}  |  Batches/epoch: {len(train_loader)}")

    # ── Model, loss, optimiser ────────────────────────────────────────────────
    model     = UNet(in_channels=1, out_channels=1).to(device)
    criterion = CombinedLoss(alpha=0.7, data_range=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing: lr decays smoothly to ~0 by epoch `epochs`.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 1e-2
    )

    # ── Checkpoint directory ──────────────────────────────────────────────────
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / CHECKPOINT_NAME

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path = checkpoint_dir / "train_log.csv"
    log_fields = ["epoch", "train_loss", "val_loss", "lr", "elapsed_s"]

    best_val_loss = float("inf")

    print(
        f"\n{'Epoch':>6}  {'Train':>10}  {'Val':>10}  "
        f"{'LR':>10}  {'Time':>8}  {'Saved'}"
    )
    print("-" * 58)

    with open(log_path, "w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=log_fields)
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss   = _run_epoch(model, val_loader,   criterion, None,      device)

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            elapsed    = time.time() - t0

            saved = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_path)
                saved = "✓ best.pt"

            print(
                f"{epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  "
                f"{current_lr:>10.2e}  {elapsed:>7.1f}s  {saved}"
            )

            writer.writerow({
                "epoch":     epoch,
                "train_loss": round(train_loss, 8),
                "val_loss":   round(val_loss,   8),
                "lr":         round(current_lr, 10),
                "elapsed_s":  round(elapsed,    2),
            })
            log_file.flush()

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Checkpoint: {best_path}")
    print(f"Log:        {log_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m model.train",
        description="Train the DocClean-Net U-Net on synthetic (dirty, clean) pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data", type=Path, required=True, metavar="DIR",
        help="Root directory containing dirty/ and clean/ subdirectories.",
    )
    p.add_argument(
        "--epochs", type=int, default=50, metavar="N",
        help="Number of training epochs.",
    )
    p.add_argument(
        "--batch", type=int, default=8, metavar="N",
        help="Mini-batch size.",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3, metavar="F",
        help="Initial learning rate for AdamW.",
    )
    p.add_argument(
        "--patch-size", type=int, default=256, metavar="N",
        help="Side length of square patches extracted from each image.",
    )
    p.add_argument(
        "--seed", type=int, default=42, metavar="N",
        help="RNG seed for dataset patch extraction.",
    )
    p.add_argument(
        "--checkpoint-dir", type=Path, default=CHECKPOINT_DIR, metavar="DIR",
        help="Directory to save best.pt and train_log.csv.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if not args.data.is_dir():
        print(f"[ERROR] --data directory not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    train(
        data_dir       = args.data,
        epochs         = args.epochs,
        batch_size     = args.batch,
        lr             = args.lr,
        patch_size     = args.patch_size,
        dataset_seed   = args.seed,
        checkpoint_dir = args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
