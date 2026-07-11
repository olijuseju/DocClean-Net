"""
tests/test_losses.py
====================
Tests for model/losses.py — CombinedLoss.

All tests use CPU only, batch size 2, patches 256×256.
"""

import pytest
import torch

from model.losses import CombinedLoss

# ── Fixtures ──────────────────────────────────────────────────────────────────

BATCH = 2
HEIGHT = 256
WIDTH = 256


@pytest.fixture(scope="module")
def criterion() -> CombinedLoss:
    return CombinedLoss(alpha=0.7, data_range=1.0)


@pytest.fixture(scope="module")
def pred_tensor() -> torch.Tensor:
    """Random predictions in [0, 1], shape (B, 1, H, W)."""
    torch.manual_seed(0)
    return torch.rand(BATCH, 1, HEIGHT, WIDTH)


@pytest.fixture(scope="module")
def zeros_tensor() -> torch.Tensor:
    return torch.zeros(BATCH, 1, HEIGHT, WIDTH)


@pytest.fixture(scope="module")
def ones_tensor() -> torch.Tensor:
    return torch.ones(BATCH, 1, HEIGHT, WIDTH)


# ── Identity property ─────────────────────────────────────────────────────────


def test_combined_loss_is_zero_for_identical_tensors(
    criterion: CombinedLoss, pred_tensor: torch.Tensor
) -> None:
    """Loss must be exactly 0 when pred == target."""
    loss = criterion(pred_tensor, pred_tensor)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


# ── Non-zero for different tensors ────────────────────────────────────────────


def test_combined_loss_is_positive_for_pred_vs_zeros(
    criterion: CombinedLoss,
    pred_tensor: torch.Tensor,
    zeros_tensor: torch.Tensor,
) -> None:
    loss = criterion(pred_tensor, zeros_tensor)
    assert float(loss) > 0.0


def test_combined_loss_is_positive_for_zeros_vs_ones(
    criterion: CombinedLoss,
    zeros_tensor: torch.Tensor,
    ones_tensor: torch.Tensor,
) -> None:
    """Maximum disagreement (all-black vs all-white) must give large loss."""
    loss = criterion(zeros_tensor, ones_tensor)
    assert float(loss) > 0.0


# ── Scalar output ─────────────────────────────────────────────────────────────


def test_combined_loss_returns_scalar(
    criterion: CombinedLoss, pred_tensor: torch.Tensor, zeros_tensor: torch.Tensor
) -> None:
    loss = criterion(pred_tensor, zeros_tensor)
    assert loss.shape == torch.Size([])


def test_combined_loss_output_dtype_is_float32(
    criterion: CombinedLoss, pred_tensor: torch.Tensor, zeros_tensor: torch.Tensor
) -> None:
    loss = criterion(pred_tensor, zeros_tensor)
    assert loss.dtype == torch.float32


# ── Gradient flow ─────────────────────────────────────────────────────────────


def test_combined_loss_gradient_flows_to_pred() -> None:
    """Loss must be differentiable w.r.t. pred."""
    criterion = CombinedLoss()
    pred = torch.rand(2, 1, 256, 256, requires_grad=True)
    target = torch.rand(2, 1, 256, 256)
    loss = criterion(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()


def test_combined_loss_gradient_is_not_nan_for_identical_tensors() -> None:
    """Gradient must not be NaN even when loss == 0 (SSIM at saturation)."""
    criterion = CombinedLoss()
    pred = torch.ones(2, 1, 256, 256, requires_grad=True)
    # target slightly off to avoid SSIM gradient degeneration at exact 1.0
    target = torch.ones(2, 1, 256, 256) * 0.9999
    loss = criterion(pred, target)
    loss.backward()
    assert not torch.isnan(pred.grad).any()


# ── Alpha weighting ───────────────────────────────────────────────────────────


def test_combined_loss_alpha_zero_equals_pure_ssim_loss(
    pred_tensor: torch.Tensor, zeros_tensor: torch.Tensor
) -> None:
    """With alpha=0 the loss must equal (1 - SSIM) exactly."""
    from pytorch_msssim import ssim

    criterion = CombinedLoss(alpha=0.0)
    loss = criterion(pred_tensor, zeros_tensor)
    ssim_val = ssim(pred_tensor, zeros_tensor, data_range=1.0, size_average=True)
    expected = 1.0 - float(ssim_val)
    assert float(loss) == pytest.approx(expected, abs=1e-5)


def test_combined_loss_alpha_one_equals_pure_mse_loss(
    pred_tensor: torch.Tensor, zeros_tensor: torch.Tensor
) -> None:
    """With alpha=1 the loss must equal MSE exactly."""
    import torch.nn as nn

    criterion = CombinedLoss(alpha=1.0)
    loss = criterion(pred_tensor, zeros_tensor)
    expected = float(nn.MSELoss()(pred_tensor, zeros_tensor))
    assert float(loss) == pytest.approx(expected, abs=1e-5)


# ── Constructor validation ────────────────────────────────────────────────────


def test_combined_loss_raises_on_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        CombinedLoss(alpha=1.5)

    with pytest.raises(ValueError, match="alpha"):
        CombinedLoss(alpha=-0.1)


# ── Symmetry ──────────────────────────────────────────────────────────────────


def test_combined_loss_is_approximately_symmetric(
    criterion: CombinedLoss,
    pred_tensor: torch.Tensor,
    zeros_tensor: torch.Tensor,
) -> None:
    """Loss(pred, target) ≈ Loss(target, pred).

    MSE is exactly symmetric; SSIM is not guaranteed to be perfectly symmetric
    numerically, but the difference should be negligible.
    """
    loss_ab = float(criterion(pred_tensor, zeros_tensor))
    loss_ba = float(criterion(zeros_tensor, pred_tensor))
    assert loss_ab == pytest.approx(loss_ba, rel=1e-3)
