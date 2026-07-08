#!/usr/bin/env python3
"""
scripts/visualize_results.py
============================
Quick visual comparison of one scan: original | classic pipeline | U-Net.

Produces a side-by-side PNG figure. With --crop, adds a second row zoomed
into a region of interest — useful for inspecting stroke thickness and
faint-detail preservation up close.

Usage (from the repo root):
    python scripts/visualize_results.py --input data/real_test/21.png \
        --model checkpoints/best.pt
    python scripts/visualize_results.py --input scan.png \
        --model checkpoints/best.pt --crop 400 600 500 500 -o comparison.png
"""

import argparse
import contextlib
import io
import sys
import tempfile
from pathlib import Path

# scripts/ is not a package: put the repo root on sys.path so that
# `python scripts/visualize_results.py` works from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import matplotlib

matplotlib.use("Agg")  # headless backend: works in CI and over SSH
import matplotlib.pyplot as plt
import numpy as np

from classic_pipeline.digitize_notebook import digitize
from inference.io_utils import _imread, _imwrite
from inference.predict import predict_image


def _crop_region(
    image: np.ndarray, x: int, y: int, w: int, h: int
) -> np.ndarray:
    """Crop a region, clamped to the image bounds.

    Args:
        image (np.ndarray): grayscale or BGR image, shape (H, W[, 3]).
        x, y: top-left corner of the crop.
        w, h: crop width and height.

    Returns:
        np.ndarray: cropped view (same dtype/channels as input).

    Raises:
        ValueError: if the clamped region is empty.
    """
    img_h, img_w = image.shape[:2]
    x0 = max(0, min(x, img_w - 1))
    y0 = max(0, min(y, img_h - 1))
    x1 = max(x0 + 1, min(x + w, img_w))
    y1 = max(y0 + 1, min(y + h, img_h))
    region = image[y0:y1, x0:x1]
    if region.size == 0:
        raise ValueError(f"Empty crop region: x={x} y={y} w={w} h={h}")
    return region


def build_comparison_figure(
    original_bgr: np.ndarray,
    classic_gray: np.ndarray,
    dl_gray: np.ndarray,
    crop: tuple[int, int, int, int] | None = None,
) -> plt.Figure:
    """Build the original|classic|DL comparison figure.

    Args:
        original_bgr (np.ndarray): input scan, shape (H, W, 3), dtype uint8.
        classic_gray (np.ndarray): classic pipeline output, shape (H, W),
            dtype uint8.
        dl_gray (np.ndarray): U-Net output, shape (H, W), dtype uint8.
        crop: optional (x, y, w, h) region; adds a second zoomed row.

    Returns:
        plt.Figure: the assembled figure (caller saves/closes it).
    """
    n_rows = 2 if crop is not None else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 7 * n_rows))
    axes = np.atleast_2d(axes)

    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    panels = [
        (original_rgb, "Original", None),
        (classic_gray, "Classic pipeline", "gray"),
        (dl_gray, "DocClean-Net (U-Net)", "gray"),
    ]
    for ax, (img, title, cmap) in zip(axes[0], panels):
        ax.imshow(img, cmap=cmap, vmin=0 if cmap else None,
                  vmax=255 if cmap else None)
        ax.set_title(title)
        ax.axis("off")

    if crop is not None:
        x, y, w, h = crop
        for ax, (img, title, cmap) in zip(axes[1], panels):
            ax.imshow(_crop_region(img, x, y, w, h), cmap=cmap,
                      vmin=0 if cmap else None, vmax=255 if cmap else None)
            ax.set_title(f"{title} — crop ({x},{y},{w}x{h})")
            ax.axis("off")
        # Rectangle on the full-size row marking the zoomed region.
        for ax in axes[0]:
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                       edgecolor="red", linewidth=1.5))

    fig.tight_layout()
    return fig


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Side-by-side comparison: original | classic | U-Net.",
    )
    parser.add_argument("--input", "-i", required=True, help="Input scan image")
    parser.add_argument("--model", "-m", required=True, help="Checkpoint (.pt)")
    parser.add_argument(
        "--output", "-o", default="comparison.png",
        help="Output figure path (default: comparison.png)",
    )
    parser.add_argument(
        "--crop", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
        help="Optional zoom region; adds a second row to the figure",
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = parser.parse_args()

    input_path = Path(args.input)
    try:
        original = _imread(input_path)

        print("Running classic pipeline ...")
        with tempfile.TemporaryDirectory() as tmp:
            classic_path = Path(tmp) / "classic.png"
            with contextlib.redirect_stdout(io.StringIO()):
                digitize(str(input_path), str(classic_path))
            classic = _imread(classic_path, flags=cv2.IMREAD_GRAYSCALE)

        print("Running U-Net inference ...")
        dl = predict_image(args.model, input_path, device=args.device)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    crop = tuple(args.crop) if args.crop else None
    fig = build_comparison_figure(original, classic, dl, crop=crop)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
