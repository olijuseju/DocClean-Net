#!/usr/bin/env python3
"""
scripts/boost_black.py
=======================
Standalone post-processing: pushes ink toward pure black in images already
restored by the GUI/pipeline. Independent of run_pipeline.py and
thicken_strokes.py by design — run it as a separate finishing pass on any
result.png, no re-inference needed.

Why this is needed: the U-Net is trained with MSE, which never lets the
output sigmoid fully saturate at either end. The paper-white side of this
is already fixed by inference.predict's white-point normalisation
(anchored at black=0, stretches paper up to 255). This script is the
mirror fix for the black end: a levels stretch anchored at white=255 that
pushes the ink level down to 0. Same technique, opposite anchor — no
retraining involved.

Trade-off to know before using it (flagging explicitly, not hidden): this
is a monotonic stretch across the WHOLE grayscale range, so midtones
(e.g. faint pencil, which the model deliberately keeps gray rather than
removing — see project_context.md "Model contract") get darkened too,
proportionally less than true ink but not zero. If you need pencil visually
untouched, keep --black-point conservative (low) or skip this script for
those images.

Convention: 0 = ink, 255 = paper (grayscale, uint8), same as the rest of
the pipeline.

CLI:
    # single file, auto-estimated black point
    python scripts/boost_black.py --input result.png --output result_black.png

    # whole folder, manual black point
    python scripts/boost_black.py --input cleaned/ --output cleaned_black/ \\
        --black-point 45

    # preview the estimate without writing anything
    python scripts/boost_black.py --input result.png --output out.png --dry-run
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

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_DEFAULT_INK_THRESHOLD = 128
_DEFAULT_MARGIN = 10


def estimate_black_point(
    image_gray: np.ndarray,
    ink_threshold: int = _DEFAULT_INK_THRESHOLD,
    margin: int = _DEFAULT_MARGIN,
) -> int:
    """Estimate the ink level as the histogram mode of dark pixels, plus a margin.

    Mirrors inference.predict._estimate_white_point for the black end. The
    white-point estimator takes the mode of the WHOLE histogram (paper
    dominates any document image); here the histogram is restricted to
    pixels already classified as ink (< ink_threshold) so the paper peak
    never leaks into the estimate.

    Args:
        image_gray (np.ndarray): grayscale image, shape (H, W), dtype uint8.
        ink_threshold: gray level below which a pixel counts as ink.
            Pixels at or above this are excluded from the histogram.
        margin: levels added to the histogram mode, pushed toward paper so
            ink pixels sitting just above the mode also clip to black.

    Returns:
        int: black point in [0, 254]. Falls back to 0 (no-op) if the image
            has no pixels below ink_threshold — nothing to estimate from.
    """
    dark = image_gray[image_gray < ink_threshold]
    if dark.size == 0:
        return 0
    hist, _ = np.histogram(dark, bins=256, range=(0, 256))
    mode = int(np.argmax(hist))
    return int(np.clip(mode + margin, 0, 254))


def apply_black_point(image_gray: np.ndarray, black_point: int) -> np.ndarray:
    """Linear levels stretch: black_point maps to 0, 255 stays 255.

    Mirrors inference.predict._apply_white_point for the opposite end.
    Paper pixels (near 255) are barely affected; ink at or below
    black_point saturates to pure black (0). Midtones between black_point
    and 255 are darkened proportionally — see the module docstring for the
    faint-pencil trade-off.

    Args:
        image_gray (np.ndarray): grayscale image, shape (H, W), dtype uint8.
        black_point: intensity mapped to 0. Must be in [0, 254]. 0 is a
            no-op (nothing to stretch).

    Returns:
        np.ndarray: stretched image, shape (H, W), dtype uint8.

    Raises:
        ValueError: if black_point is outside [0, 254].
    """
    if not 0 <= black_point <= 254:
        raise ValueError("black_point must be in [0, 254]")
    if black_point == 0:
        return image_gray.copy()
    scale = 255.0 / (255.0 - float(black_point))
    stretched = (image_gray.astype(np.float32) - float(black_point)) * scale
    return np.clip(stretched, 0.0, 255.0).astype(np.uint8)


def boost_black(
    image_gray: np.ndarray,
    black_point: int | str | None = "auto",
    ink_threshold: int = _DEFAULT_INK_THRESHOLD,
    margin: int = _DEFAULT_MARGIN,
) -> np.ndarray:
    """Estimate (if needed) and apply the black-point stretch in one call.

    Args:
        image_gray (np.ndarray): grayscale image, shape (H, W), dtype uint8.
        black_point: "auto" (default) to estimate via estimate_black_point,
            an int in [0, 254] to set it manually, or None to disable
            (returns a copy unchanged).
        ink_threshold: forwarded to estimate_black_point. Ignored if
            black_point is not "auto".
        margin: forwarded to estimate_black_point. Ignored if black_point
            is not "auto".

    Returns:
        np.ndarray: same shape/dtype, black-point stretched.
    """
    if black_point is None:
        return image_gray.copy()
    if black_point == "auto":
        black_point = estimate_black_point(image_gray, ink_threshold, margin)
    return apply_black_point(image_gray, int(black_point))


def _discover_images(directory: Path) -> list[Path]:
    """List image files directly inside `directory` (non-recursive).

    Args:
        directory: folder to scan.

    Returns:
        list[Path]: sorted image paths (may be empty).
    """
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def run(
    input_path: str | Path,
    output_path: str | Path,
    black_point: int | str | None = "auto",
    ink_threshold: int = _DEFAULT_INK_THRESHOLD,
    margin: int = _DEFAULT_MARGIN,
    dry_run: bool = False,
) -> list[Path]:
    """Apply the black-point boost to a single file or every image in a directory.

    If `input_path` is a file, `output_path` is treated as the destination
    file (parent dirs created as needed). If `input_path` is a directory,
    `output_path` is treated as the destination directory, and every
    discovered image is processed with the same output filename.

    Args:
        input_path: Input image file, or a directory of images.
        output_path: Output file path, or output directory.
        black_point: forwarded to boost_black.
        ink_threshold: forwarded to boost_black.
        margin: forwarded to boost_black.
        dry_run: if True, nothing is written; the estimated/used black
            point is printed for each image instead.

    Returns:
        list[Path]: output file paths written, in processing order (empty
            if dry_run is True).

    Raises:
        FileNotFoundError: if input_path does not exist, or a directory
            input contains no images.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if input_path.is_dir():
        images = _discover_images(input_path)
        if not images:
            raise FileNotFoundError(f"No images found in directory: {input_path}")
        if not dry_run:
            output_path.mkdir(parents=True, exist_ok=True)
        targets = [(img, output_path / img.name) for img in images]
    else:
        targets = [(input_path, output_path)]

    written: list[Path] = []
    for src, dst in targets:
        img = _imread(src, flags=cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] unreadable, skipping: {src}", file=sys.stderr)
            continue

        used_bp = black_point
        if black_point == "auto":
            used_bp = estimate_black_point(img, ink_threshold, margin)

        if dry_run:
            print(f"{src.name}: black_point={used_bp}")
            continue

        result = boost_black(img, black_point, ink_threshold, margin)
        _imwrite(dst, result)
        print(f"{src.name}: black_point={used_bp} -> {dst}")
        written.append(dst)

    return written


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Push ink toward pure black in already-restored images "
        "(levels stretch anchored at white=255). Accepts a single image "
        "or a directory (batch mode).",
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input image file, or a directory"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output file path, or output directory"
    )
    parser.add_argument(
        "--black-point",
        default="auto",
        help='"auto" (default), an int in [0, 254], or "off" to disable.',
    )
    parser.add_argument(
        "--ink-threshold",
        type=int,
        default=_DEFAULT_INK_THRESHOLD,
        help=f"Gray level below which a pixel counts as ink for the "
        f"auto-estimate (default: {_DEFAULT_INK_THRESHOLD}).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=_DEFAULT_MARGIN,
        help=f"Levels added to the histogram mode in the auto-estimate "
        f"(default: {_DEFAULT_MARGIN}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the black point that would be used, write nothing.",
    )
    args = parser.parse_args()

    if args.black_point == "off":
        black_point: int | str | None = None
    elif args.black_point == "auto":
        black_point = "auto"
    else:
        try:
            black_point = int(args.black_point)
        except ValueError:
            print(
                "[ERROR] --black-point must be 'auto', 'off', or an int",
                file=sys.stderr,
            )
            sys.exit(1)
        if not 0 <= black_point <= 254:
            print("[ERROR] --black-point int must be in [0, 254]", file=sys.stderr)
            sys.exit(1)

    try:
        run(
            args.input,
            args.output,
            black_point=black_point,
            ink_threshold=args.ink_threshold,
            margin=args.margin,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
