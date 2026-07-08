#!/usr/bin/env python3
"""
inference/predict.py
====================
Sliding-window inference for DocClean-Net's U-Net.

The U-Net operates on fixed 256x256 grayscale patches. For arbitrary input
resolutions this module slides a window across the image with 50% overlap
(default stride=128) and blends overlapping predictions with a 2D Gaussian
weight window. Patch-border pixels are the model's least reliable
predictions; Gaussian weighting suppresses exactly those, so no visible
seams appear in the reconstruction (uniform averaging still leaves faint
seams).

CLI:
    python -m inference.predict --model checkpoints/best.pt \
        --input scan.png --output result.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from inference.io_utils import _imread, _imwrite
from model.unet import UNet


def _resolve_device(device: str) -> torch.device:
    """Resolve a device string to a torch.device.

    Args:
        device: "auto", "cpu" or "cuda" (or "cuda:0", ...).

    Returns:
        torch.device: resolved device. "auto" picks CUDA when available.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_model(model_path: Path, device: torch.device) -> UNet:
    """Load a trained UNet checkpoint in eval mode.

    Accepts either a raw state_dict (the format model/train.py saves) or a
    dict wrapping it under "model_state_dict" (common convention, kept for
    forward compatibility).

    Args:
        model_path: Path to a .pt checkpoint.
        device: torch.device to place the model on.

    Returns:
        UNet: model in eval mode, on `device`.

    Raises:
        FileNotFoundError: if model_path does not exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _gaussian_window(patch_size: int, sigma: float | None = None) -> np.ndarray:
    """Build a 2D Gaussian weight window for blending overlapping patches.

    Args:
        patch_size: side length of the (square) patch.
        sigma: Gaussian std-dev in pixels. Defaults to patch_size / 4.

    Returns:
        np.ndarray: shape (patch_size, patch_size), dtype float32,
            values in (0, 1], peak 1.0 at the center.
    """
    if sigma is None:
        sigma = patch_size / 4.0
    kernel_1d = cv2.getGaussianKernel(patch_size, sigma)  # (patch_size, 1) float64
    window_2d = kernel_1d @ kernel_1d.T
    # Peak-normalise, not sum-normalise: the final division by weight_sum
    # handles absolute scale; only the relative center-vs-edge weight matters.
    window_2d = window_2d / window_2d.max()
    return window_2d.astype(np.float32)


def _compute_padded_size(size: int, patch_size: int, stride: int) -> int:
    """Smallest padded dimension that sliding windows exactly tile.

    Args:
        size: original dimension (H or W).
        patch_size: sliding window side length.
        stride: sliding window step.

    Returns:
        int: smallest value >= size such that windows of `patch_size`
            stepped by `stride` from 0 end exactly at the returned value.
    """
    if size <= patch_size:
        return patch_size
    n_steps = int(np.ceil((size - patch_size) / stride))
    return patch_size + n_steps * stride



def _estimate_white_point(image_gray: np.ndarray, margin: int = 10) -> int:
    """Estimate the paper level as the histogram mode, minus a margin.

    In a restored document the paper is by far the dominant intensity, so
    the histogram mode is a robust paper estimate. The margin pushes the
    white point slightly below it so faint residual structure riding just
    under the paper level (e.g. blurred grid remnants) clips to white too.

    Args:
        image_gray (np.ndarray): grayscale image, shape (H, W), dtype uint8.
        margin: levels subtracted from the histogram mode.

    Returns:
        int: white point in [1, 255].
    """
    hist, _ = np.histogram(image_gray, bins=256, range=(0, 256))
    mode = int(np.argmax(hist))
    return int(np.clip(mode - margin, 1, 255))


def _apply_white_point(image_gray: np.ndarray, white_point: int) -> np.ndarray:
    """Linear levels stretch: white_point maps to 255, 0 stays 0.

    Ink pixels (near 0) are barely affected; everything at or above the
    white point saturates to pure white, removing the grey cast and any
    faint residual background structure.

    Args:
        image_gray (np.ndarray): grayscale image, shape (H, W), dtype uint8.
        white_point: intensity mapped to 255. Must be >= 1.

    Returns:
        np.ndarray: stretched image, shape (H, W), dtype uint8.

    Raises:
        ValueError: if white_point < 1.
    """
    if white_point < 1:
        raise ValueError("white_point must be >= 1")
    stretched = image_gray.astype(np.float32) * (255.0 / float(white_point))
    return np.clip(stretched, 0.0, 255.0).astype(np.uint8)


def predict_image(
    model_path: str | Path,
    image_path: str | Path,
    patch_size: int = 256,
    stride: int = 128,
    device: str = "auto",
    batch_size: int = 16,
    white_point: int | str | None = "auto",
) -> np.ndarray:
    """Restore a scanned document image with the trained U-Net.

    Sliding-window inference with Gaussian-blended overlap; works on any
    input resolution (not limited to multiples of patch_size).

    Args:
        model_path: Path to the trained checkpoint (e.g. checkpoints/best.pt).
        image_path: Path to the input scan (any OpenCV-readable format,
            loaded as BGR uint8).
        patch_size: Side of the square sliding window; must match the
            resolution the U-Net was trained on (256).
        stride: Step between windows. 128 = 50% overlap, the tested
            sweet spot (see current_status.md, architecture decisions).
        device: "auto", "cpu" or "cuda".
        batch_size: patches per forward pass. Throughput only — the
            result is identical for any value >= 1.
        white_point: post-network levels normalisation. "auto" (default)
            estimates the paper level from the histogram mode and stretches
            it to pure white, removing the grey cast and faint residual
            background structure inherent to MSE-trained sigmoid outputs
            (paper saturates at ~233, never 255). Pass an int for a fixed
            white point, or None to disable and get the raw network output.

    Returns:
        np.ndarray: restored grayscale image, shape (H, W), dtype uint8,
            0 = ink, 255 = paper. Same resolution as the input. This is
            the model's continuous restoration, NOT a thresholded mask.

    Raises:
        FileNotFoundError: if model_path or image_path does not exist.
        ValueError: if patch_size/stride are non-positive or
            stride > patch_size (windows would leave uncovered gaps).
    """
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    if stride > patch_size:
        raise ValueError("stride must be <= patch_size, or windows leave gaps")

    torch_device = _resolve_device(device)
    model = _load_model(Path(model_path), torch_device)

    img_bgr = _imread(Path(image_path), flags=cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # BGR -> grayscale, as in training
    h, w = gray.shape

    padded_h = _compute_padded_size(h, patch_size, stride)
    padded_w = _compute_padded_size(w, patch_size, stride)

    # Reflect padding: zero/constant padding would create a hard artificial
    # edge the model never saw in training and bias border predictions.
    gray_padded = cv2.copyMakeBorder(
        gray, 0, padded_h - h, 0, padded_w - w, borderType=cv2.BORDER_REFLECT_101
    )
    gray_norm = gray_padded.astype(np.float32) / 255.0  # same preproc as _to_tensor

    ys = range(0, padded_h - patch_size + 1, stride)
    xs = range(0, padded_w - patch_size + 1, stride)
    positions = [(y, x) for y in ys for x in xs]

    window = _gaussian_window(patch_size)
    output_sum = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight_sum = np.zeros((padded_h, padded_w), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, len(positions), batch_size):
            batch_positions = positions[i : i + batch_size]
            patches = np.stack(
                [gray_norm[y : y + patch_size, x : x + patch_size]
                 for y, x in batch_positions],
                axis=0,
            )
            tensor = torch.from_numpy(patches).unsqueeze(1).to(torch_device)  # (B,1,ps,ps)
            preds = model(tensor).squeeze(1).cpu().numpy()                     # (B,ps,ps)

            for (y, x), pred in zip(batch_positions, preds):
                output_sum[y : y + patch_size, x : x + patch_size] += pred * window
                weight_sum[y : y + patch_size, x : x + patch_size] += window

    blended = output_sum / weight_sum          # weight_sum > 0 everywhere by tiling
    blended = blended[:h, :w]                  # crop padding back off
    result = np.clip(blended * 255.0, 0.0, 255.0).astype(np.uint8)

    if white_point is None:
        return result
    if white_point == "auto":
        return _apply_white_point(result, _estimate_white_point(result))
    return _apply_white_point(result, int(white_point))


def main() -> None:
    """CLI entry point: python -m inference.predict ..."""
    parser = argparse.ArgumentParser(
        prog="inference.predict",
        description="Run DocClean-Net U-Net inference on a scanned document.",
    )
    parser.add_argument("--model", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--input", required=True, help="Input scan image")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--white-point", default="auto",
        help='Levels normalisation: "auto" (default), an int 1-255, or "off"',
    )
    args = parser.parse_args()

    if args.white_point == "off":
        white_point: int | str | None = None
    elif args.white_point == "auto":
        white_point = "auto"
    else:
        white_point = int(args.white_point)

    try:
        result = predict_image(
            model_path=args.model,
            image_path=args.input,
            patch_size=args.patch_size,
            stride=args.stride,
            device=args.device,
            batch_size=args.batch_size,
            white_point=white_point,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    _imwrite(Path(args.output), result)
    print(f"Saved: {args.output}  ({result.shape[1]}x{result.shape[0]} px)")


if __name__ == "__main__":
    main()
