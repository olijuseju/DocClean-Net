#!/usr/bin/env python3
"""
scripts/thicken_strokes.py
===========================
Standalone post-processing for already-restored documents: removes small
speckle noise (isolated ink dots) and/or thickens strokes via morphological
erosion.

Both operations act on the U-Net's *output* image — they don't touch
predict.py, the model, or the white-point normalisation, so they can be
iterated on independently and re-run on an existing result.png in seconds
instead of re-running full inference.

Order matters: denoising runs before thickening by default, so isolated
dots are erased while they are still small rather than being grown into
bigger blobs first.

Convention: 0 = ink, 255 = paper (grayscale, uint8). Because ink is dark,
*eroding* the image (shrinking bright regions) is what visually *thickens*
the dark strokes.

CLI:
    python scripts/thicken_strokes.py --input result.png --output clean.png \\
        --amount 1 --min-dot-area 3
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference.io_utils import _imread, _imwrite


def thicken_strokes(image_gray: np.ndarray, amount: int = 1) -> np.ndarray:
    """Thicken dark strokes via morphological erosion.

    Ink is dark-on-light (0=ink, 255=paper), so eroding the bright paper
    region is what grows the dark stroke by `amount` pixels per side.

    Args:
        image_gray (np.ndarray): grayscale image, shape (H, W), dtype uint8.
        amount: kernel radius in pixels. 0 = no-op. 1 -> 3x3 elliptical
            kernel (subtle, safe default). 2+ risks fusing fine details
            (small text, closely-spaced lines) — verify visually before
            using higher values.

    Returns:
        np.ndarray: same shape/dtype, strokes thickened by ~`amount` px.

    Raises:
        ValueError: if amount < 0.
    """
    if amount < 0:
        raise ValueError("amount must be >= 0")
    if amount == 0:
        return image_gray.copy()
    ksize = 2 * amount + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.erode(image_gray, kernel)


def remove_small_dots(
    image_gray: np.ndarray, min_area: int = 3, ink_threshold: int = 128
) -> np.ndarray:
    """Erase small isolated ink components (speckle noise), same technique
    as the classic pipeline's connected-component cleanup (design decision:
    CC-based, not morphological opening — opening would also erode genuine
    thin strokes, not just isolated dots).

    Args:
        image_gray (np.ndarray): grayscale image, shape (H, W), dtype uint8,
            0=ink, 255=paper.
        min_area: components with area (in pixels) strictly below this are
            erased (painted to paper white). Components at or above this
            size are kept untouched.
        ink_threshold: gray level below which a pixel counts as ink, for
            building the binary mask connected components are computed on.

    Returns:
        np.ndarray: same shape/dtype, small isolated components removed.

    Raises:
        ValueError: if min_area < 1.
    """
    if min_area < 1:
        raise ValueError("min_area must be >= 1")

    binary = (image_gray < ink_threshold).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    result = image_gray.copy()
    for label in range(1, n_labels):  # label 0 is the background, skip it
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            result[labels == label] = 255
    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Clean up a restored document: remove speckle noise and/or thicken strokes.",
    )
    parser.add_argument("--input", "-i", required=True, help="Input grayscale PNG")
    parser.add_argument("--output", "-o", required=True, help="Output path")
    parser.add_argument(
        "--amount",
        "-a",
        type=int,
        default=1,
        help="Stroke thickening kernel radius in px (default: 1; 0 disables). "
        "2+ may fuse fine details.",
    )
    parser.add_argument(
        "--min-dot-area",
        type=int,
        default=3,
        help="Remove ink components smaller than this, in px (default: 3; "
        "0 disables denoising).",
    )
    parser.add_argument(
        "--ink-threshold",
        type=int,
        default=128,
        help="Gray level below which a pixel counts as ink (default: 128).",
    )
    args = parser.parse_args()

    try:
        img = _imread(Path(args.input), flags=cv2.IMREAD_GRAYSCALE)
        result = img
        if args.min_dot_area > 0:
            result = remove_small_dots(
                result, min_area=args.min_dot_area, ink_threshold=args.ink_threshold
            )
        if args.amount > 0:
            result = thicken_strokes(result, amount=args.amount)
        _imwrite(Path(args.output), result)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Saved: {args.output}  (min_dot_area={args.min_dot_area}, amount={args.amount})"
    )


if __name__ == "__main__":
    main()
