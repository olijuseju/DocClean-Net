"""
tests/test_unet.py
==================
Tests for model/unet.py — UNet and ConvBlock.

All tests use CPU only; no checkpoint required.
"""

import pytest
import torch

from model.unet import ConvBlock, UNet


# ── Fixtures ──────────────────────────────────────────────────────────────────

BATCH = 2
PATCH = 256


@pytest.fixture(scope="module")
def model() -> UNet:
    """UNet instance in eval mode on CPU."""
    net = UNet(in_channels=1, out_channels=1)
    net.eval()
    return net


@pytest.fixture(scope="module")
def dummy_input() -> torch.Tensor:
    """Random batch of grayscale patches, float32 in [0, 1]."""
    return torch.rand(BATCH, 1, PATCH, PATCH)


# ── ConvBlock ─────────────────────────────────────────────────────────────────

def test_convblock_output_shape_preserves_spatial_dimensions() -> None:
    """ConvBlock must not change H or W (padding=1 on kernel=3)."""
    block = ConvBlock(in_channels=4, out_channels=8)
    x = torch.rand(2, 4, 64, 64)
    y = block(x)
    assert y.shape == (2, 8, 64, 64)


def test_convblock_output_channels_match_specified() -> None:
    """Output channel count must equal the out_channels argument."""
    for in_ch, out_ch in [(1, 16), (16, 32), (64, 128)]:
        block = ConvBlock(in_ch, out_ch)
        x = torch.rand(1, in_ch, 32, 32)
        assert block(x).shape[1] == out_ch


def test_convblock_does_not_modify_input_tensor() -> None:
    """ConvBlock must be a pure function — no in-place modification of input."""
    block = ConvBlock(8, 8)
    x = torch.rand(1, 8, 32, 32)
    x_copy = x.clone()
    block(x)
    assert torch.allclose(x, x_copy)


# ── UNet — architecture ───────────────────────────────────────────────────────

def test_unet_forward_pass_output_shape(
    model: UNet, dummy_input: torch.Tensor
) -> None:
    """Forward pass must return (B, 1, 256, 256)."""
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (BATCH, 1, PATCH, PATCH)


def test_unet_output_values_in_unit_interval(
    model: UNet, dummy_input: torch.Tensor
) -> None:
    """Sigmoid output must be in [0, 1]."""
    with torch.no_grad():
        out = model(dummy_input)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_unet_output_dtype_is_float32(
    model: UNet, dummy_input: torch.Tensor
) -> None:
    with torch.no_grad():
        out = model(dummy_input)
    assert out.dtype == torch.float32


def test_unet_accepts_batch_size_one() -> None:
    """Single-sample batch must not raise (BatchNorm has special behavior at B=1
    in training mode; eval mode must always work)."""
    net = UNet().eval()
    x = torch.rand(1, 1, PATCH, PATCH)
    with torch.no_grad():
        out = net(x)
    assert out.shape == (1, 1, PATCH, PATCH)


def test_unet_parameter_count_approximately_660k() -> None:
    """Total trainable parameters must be in the range [600k, 720k].

    The exact count depends on BatchNorm affine parameters; ~660k is the target.
    """
    net = UNet()
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert 450_000 <= n_params <= 550_000, (
        f"Expected ~480k parameters, got {n_params:,}"
    )


def test_unet_gradient_flows_through_all_parameters(
    dummy_input: torch.Tensor,
) -> None:
    """Every parameter must receive a gradient after a forward+backward pass."""
    net = UNet().train()
    out = net(dummy_input)
    # Scalar loss: mean over all outputs
    loss = out.mean()
    loss.backward()

    params_without_grad = [
        name for name, p in net.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert params_without_grad == [], (
        f"Parameters with no gradient: {params_without_grad}"
    )


def test_unet_encoder_decoder_skip_connections_preserve_resolution() -> None:
    """Verify intermediate tensors at each decoder stage have the expected shape.

    This test runs a forward pass with hooks to capture intermediate outputs
    and checks that skip connections are correctly aligned.
    """
    net = UNet().eval()
    captured: dict[str, torch.Tensor] = {}

    def _hook(name: str):
        def fn(module: torch.nn.Module, inp: tuple, out: torch.Tensor) -> None:
            captured[name] = out
        return fn

    net.enc1.register_forward_hook(_hook("enc1"))
    net.enc2.register_forward_hook(_hook("enc2"))
    net.enc3.register_forward_hook(_hook("enc3"))
    net.bottleneck.register_forward_hook(_hook("bottleneck"))
    net.dec3.register_forward_hook(_hook("dec3"))
    net.dec2.register_forward_hook(_hook("dec2"))
    net.dec1.register_forward_hook(_hook("dec1"))

    x = torch.rand(1, 1, PATCH, PATCH)
    with torch.no_grad():
        net(x)

    assert captured["enc1"].shape      == (1,  16, 256, 256)
    assert captured["enc2"].shape      == (1,  32, 128, 128)
    assert captured["enc3"].shape      == (1,  64,  64,  64)
    assert captured["bottleneck"].shape == (1, 128,  32,  32)
    assert captured["dec3"].shape      == (1,  64,  64,  64)
    assert captured["dec2"].shape      == (1,  32, 128, 128)
    assert captured["dec1"].shape      == (1,  16, 256, 256)


def test_unet_different_batch_sizes_produce_consistent_output_shape() -> None:
    """Output shape must scale correctly with batch size."""
    net = UNet().eval()
    for b in (1, 4, 8):
        x = torch.rand(b, 1, PATCH, PATCH)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (b, 1, PATCH, PATCH), f"Failed for batch size {b}"
