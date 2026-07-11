"""
tests/test_train.py
===================
Tests for model/train.py.

Strategy: use a tiny in-memory dataset (8 pairs, 256×256) written to a
tmp directory. All tests run on CPU and complete in seconds.

Marked @pytest.mark.slow tests are excluded from CI with:
    pytest -m "not slow"
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from model.losses import CombinedLoss
from model.train import _get_device, _make_dataloaders, _run_epoch, train
from model.unet import UNet

# ── Fixture: minimal on-disk dataset ─────────────────────────────────────────

N_PAIRS = 8  # enough for a 90/10 split (7 train, 1 val)
IMAGE_SIZE = 256  # exact patch size → only one crop location possible
PATCH_SIZE = 256
BATCH_SIZE = 4


@pytest.fixture(scope="module")
def tiny_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """8 pairs of 256×256 synthetic images written as PNG."""
    root = tmp_path_factory.mktemp("train_data")
    dirty_dir = root / "dirty"
    clean_dir = root / "clean"
    dirty_dir.mkdir()
    clean_dir.mkdir()

    rng = np.random.default_rng(seed=0)
    for i in range(N_PAIRS):
        dirty = rng.integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        clean = rng.integers(0, 256, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        cv2.imwrite(str(dirty_dir / f"dirty_{i:06d}.png"), dirty)
        cv2.imwrite(str(clean_dir / f"clean_{i:06d}.png"), clean)

    return root


# ── _get_device ───────────────────────────────────────────────────────────────


def test_get_device_returns_torch_device() -> None:
    device = _get_device()
    assert isinstance(device, torch.device)


def test_get_device_returns_cpu_or_cuda() -> None:
    device = _get_device()
    assert device.type in ("cpu", "cuda")


# ── _make_dataloaders ─────────────────────────────────────────────────────────


def test_make_dataloaders_returns_two_loaders(tiny_data_dir: Path) -> None:
    train_loader, val_loader = _make_dataloaders(
        tiny_data_dir, PATCH_SIZE, BATCH_SIZE, dataset_seed=42
    )
    assert train_loader is not None
    assert val_loader is not None


def test_make_dataloaders_split_covers_all_pairs(tiny_data_dir: Path) -> None:
    train_loader, val_loader = _make_dataloaders(
        tiny_data_dir, PATCH_SIZE, BATCH_SIZE, dataset_seed=42
    )
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    assert n_train + n_val == N_PAIRS


def test_make_dataloaders_train_is_larger_than_val(tiny_data_dir: Path) -> None:
    train_loader, val_loader = _make_dataloaders(
        tiny_data_dir, PATCH_SIZE, BATCH_SIZE, dataset_seed=42
    )
    assert len(train_loader.dataset) > len(val_loader.dataset)


def test_make_dataloaders_batch_shape_is_correct(tiny_data_dir: Path) -> None:
    train_loader, _ = _make_dataloaders(
        tiny_data_dir, PATCH_SIZE, BATCH_SIZE, dataset_seed=42
    )
    dirty, clean = next(iter(train_loader))
    assert dirty.ndim == 4  # (B, 1, H, W)
    assert dirty.shape[1] == 1  # single channel
    assert dirty.shape[2] == PATCH_SIZE
    assert dirty.shape[3] == PATCH_SIZE
    assert clean.shape == dirty.shape


def test_make_dataloaders_raises_on_missing_data_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _make_dataloaders(tmp_path / "nonexistent", PATCH_SIZE, BATCH_SIZE, 42)


# ── _run_epoch ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cpu_model() -> UNet:
    return UNet().to(torch.device("cpu"))


@pytest.fixture(scope="module")
def train_loader_fixture(tiny_data_dir: Path):
    loader, _ = _make_dataloaders(
        tiny_data_dir, PATCH_SIZE, BATCH_SIZE, dataset_seed=42
    )
    return loader


@pytest.fixture(scope="module")
def val_loader_fixture(tiny_data_dir: Path):
    _, loader = _make_dataloaders(
        tiny_data_dir, PATCH_SIZE, BATCH_SIZE, dataset_seed=42
    )
    return loader


def test_run_epoch_train_returns_positive_float(
    cpu_model: UNet, train_loader_fixture, tiny_data_dir: Path
) -> None:
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(cpu_model.parameters(), lr=1e-3)
    loss = _run_epoch(
        cpu_model, train_loader_fixture, criterion, optimizer, torch.device("cpu")
    )
    assert isinstance(loss, float)
    assert loss > 0.0


def test_run_epoch_val_does_not_update_parameters(
    tiny_data_dir: Path, val_loader_fixture
) -> None:
    """Validation pass must not modify model weights."""
    model = UNet().to(torch.device("cpu"))
    criterion = CombinedLoss()

    params_before = [p.clone() for p in model.parameters()]
    _run_epoch(model, val_loader_fixture, criterion, None, torch.device("cpu"))
    params_after = list(model.parameters())

    for before, after in zip(params_before, params_after):
        assert torch.equal(before, after), "Validation pass modified model weights"


def test_run_epoch_val_returns_finite_loss(cpu_model: UNet, val_loader_fixture) -> None:
    criterion = CombinedLoss()
    loss = _run_epoch(
        cpu_model, val_loader_fixture, criterion, None, torch.device("cpu")
    )
    assert np.isfinite(loss)


# ── train() end-to-end ────────────────────────────────────────────────────────


def test_train_one_epoch_saves_checkpoint(tiny_data_dir: Path, tmp_path: Path) -> None:
    """train() for 1 epoch must create checkpoints/best.pt."""
    ckpt_dir = tmp_path / "checkpoints"
    train(
        data_dir=tiny_data_dir,
        epochs=1,
        batch_size=BATCH_SIZE,
        lr=1e-3,
        patch_size=PATCH_SIZE,
        dataset_seed=42,
        checkpoint_dir=ckpt_dir,
    )
    assert (ckpt_dir / "best.pt").exists()


def test_train_one_epoch_saves_csv_log(tiny_data_dir: Path, tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    train(
        data_dir=tiny_data_dir,
        epochs=1,
        batch_size=BATCH_SIZE,
        lr=1e-3,
        patch_size=PATCH_SIZE,
        dataset_seed=42,
        checkpoint_dir=ckpt_dir,
    )
    log_path = ckpt_dir / "train_log.csv"
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    # Header + 1 data row
    assert len(lines) == 2


def test_train_checkpoint_loadable_into_unet(
    tiny_data_dir: Path, tmp_path: Path
) -> None:
    """Saved state dict must load into a fresh UNet without errors."""
    ckpt_dir = tmp_path / "checkpoints"
    train(
        data_dir=tiny_data_dir,
        epochs=1,
        batch_size=BATCH_SIZE,
        lr=1e-3,
        patch_size=PATCH_SIZE,
        dataset_seed=42,
        checkpoint_dir=ckpt_dir,
    )
    state = torch.load(ckpt_dir / "best.pt", map_location="cpu")
    model = UNet()
    model.load_state_dict(state)  # must not raise
    model.eval()
    x = torch.rand(1, 1, PATCH_SIZE, PATCH_SIZE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, PATCH_SIZE, PATCH_SIZE)


@pytest.mark.slow
def test_train_loss_decreases_over_two_epochs(
    tiny_data_dir: Path, tmp_path: Path
) -> None:
    """Training loss at epoch 2 must be lower than at epoch 1.

    Marked slow because it runs two full epochs. Excluded from CI with
    ``pytest -m "not slow"``.

    Note: with random data (no signal) the model can only overfit.
    Two epochs is enough to confirm gradient flow and optimiser step work.
    """
    import csv

    ckpt_dir = tmp_path / "checkpoints"
    train(
        data_dir=tiny_data_dir,
        epochs=2,
        batch_size=BATCH_SIZE,
        lr=1e-3,
        patch_size=PATCH_SIZE,
        dataset_seed=42,
        checkpoint_dir=ckpt_dir,
    )
    rows = list(csv.DictReader((ckpt_dir / "train_log.csv").read_text().splitlines()))
    loss_e1 = float(rows[0]["train_loss"])
    loss_e2 = float(rows[1]["train_loss"])
    assert (
        loss_e2 < loss_e1
    ), f"Training loss did not decrease: epoch1={loss_e1:.6f}, epoch2={loss_e2:.6f}"


# ── CLI (main) ────────────────────────────────────────────────────────────────


def test_main_cli_runs_one_epoch(tiny_data_dir: Path, tmp_path: Path) -> None:
    """main() called with parsed argv must not raise and must save best.pt."""
    from model.train import main

    ckpt_dir = tmp_path / "ckpt"
    main(
        [
            "--data",
            str(tiny_data_dir),
            "--epochs",
            "1",
            "--batch",
            str(BATCH_SIZE),
            "--lr",
            "1e-3",
            "--patch-size",
            str(PATCH_SIZE),
            "--checkpoint-dir",
            str(ckpt_dir),
        ]
    )
    assert (ckpt_dir / "best.pt").exists()


def test_main_cli_exits_on_missing_data_dir(tmp_path: Path) -> None:
    from model.train import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--data", str(tmp_path / "nonexistent"), "--epochs", "1"])
    assert exc_info.value.code == 1
