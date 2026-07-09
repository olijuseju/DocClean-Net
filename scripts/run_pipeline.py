#!/usr/bin/env python3
"""
scripts/run_pipeline.py
========================
End-to-end orchestrator: U-Net inference + post-processing (speckle
denoise + stroke thickening) in a single command.

Wraps inference.predict.predict_image() and the two post-processing
functions in scripts/thicken_strokes.py (remove_small_dots, then
thicken_strokes — in that order: denoising while dots are still small is
what makes it effective, see thicken_strokes.py docstring). No new logic
duplicated here; this module is a thin pipeline glue.

Accepts either a single image file or a directory: a directory processes
every image in it (non-recursive) and mirrors the file names into the
output directory.

CLI:
    # single file
    python scripts/run_pipeline.py --model checkpoints/best.pt \\
        --input scan.png --output result_final.png

    # whole folder
    python scripts/run_pipeline.py --model checkpoints/best.pt \\
        --input data/real_test/ --output cleaned/
"""

import argparse
import sys
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference.io_utils import _imwrite
from inference.predict import predict_image
from scripts.thicken_strokes import remove_small_dots, thicken_strokes

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def process_image(
    model_path: str | Path,
    image_path: str | Path,
    device: str = "auto",
    white_point: int | str | None = "auto",
    min_dot_area: int = 5,
    ink_threshold: int = 128,
    thicken_amount: int = 1,
) -> "cv2.Mat":
    """Run U-Net inference followed by denoise + thicken post-processing.

    Args:
        model_path: Path to the trained checkpoint (e.g. checkpoints/best.pt).
        image_path: Path to the input scan.
        device: "auto", "cpu" or "cuda".
        white_point: forwarded to predict_image (see its docstring).
            "auto" (default), an int, or None to disable.
        min_dot_area: components with area (px) strictly below this are
            erased. 0 disables denoising.
        ink_threshold: gray level below which a pixel counts as ink.
        thicken_amount: erosion kernel radius in px. 0 disables thickening.

    Returns:
        np.ndarray: final restored + post-processed grayscale image,
            shape (H, W), dtype uint8, 0=ink, 255=paper.
    """
    result = predict_image(
        model_path, image_path, device=device, white_point=white_point
    )
    if min_dot_area > 0:
        result = remove_small_dots(
            result, min_area=min_dot_area, ink_threshold=ink_threshold
        )
    if thicken_amount > 0:
        result = thicken_strokes(result, amount=thicken_amount)
    return result


def _discover_images(directory: Path) -> list[Path]:
    """List image files directly inside `directory` (non-recursive).

    Args:
        directory: folder to scan.

    Returns:
        list[Path]: sorted image paths (may be empty).
    """
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def run_pipeline(
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    device: str = "auto",
    white_point: int | str | None = "auto",
    min_dot_area: int = 3,
    ink_threshold: int = 128,
    thicken_amount: int = 1,
) -> list[Path]:
    """Run the full pipeline on a single file or every image in a directory.

    If `input_path` is a file, `output_path` is treated as the destination
    file (parent dirs created as needed). If `input_path` is a directory,
    `output_path` is treated as the destination directory, and every
    discovered image is processed with the same output filename.

    Args:
        model_path: Path to the trained checkpoint.
        input_path: Input image file, or a directory of images.
        output_path: Output file path, or output directory.
        device: "auto", "cpu" or "cuda".
        white_point: forwarded to predict_image.
        min_dot_area: forwarded to remove_small_dots. 0 disables it.
        ink_threshold: forwarded to remove_small_dots.
        thicken_amount: forwarded to thicken_strokes. 0 disables it.

    Returns:
        list[Path]: output file paths written, in processing order.

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
        output_path.mkdir(parents=True, exist_ok=True)
        targets = [(img, output_path / img.name) for img in images]
    else:
        targets = [(input_path, output_path)]

    written: list[Path] = []
    for src, dst in targets:
        print(f"Processing {src.name} ...")
        result = process_image(
            model_path, src,
            device=device, white_point=white_point,
            min_dot_area=min_dot_area, ink_threshold=ink_threshold,
            thicken_amount=thicken_amount,
        )
        _imwrite(dst, result)
        written.append(dst)

    return written


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="U-Net inference + denoise + thicken, in a single command. "
                     "Accepts a single image or a directory (batch mode).",
    )
    parser.add_argument("--model", "-m", required=True, help="Checkpoint (.pt)")
    parser.add_argument("--input", "-i", required=True,
                         help="Input image file, or a directory of images")
    parser.add_argument("--output", "-o", required=True,
                         help="Output file (single mode) or directory (batch mode)")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument(
        "--white-point", default="auto",
        help='Levels normalisation: "auto" (default), an int 1-255, or "off"',
    )
    parser.add_argument(
        "--min-dot-area", type=int, default=3,
        help="Remove ink components smaller than this, in px (default: 3; 0 disables)",
    )
    parser.add_argument("--ink-threshold", type=int, default=128)
    parser.add_argument(
        "--thicken-amount", type=int, default=1,
        help="Stroke thickening kernel radius in px (default: 1; 0 disables)",
    )
    args = parser.parse_args()

    if args.white_point == "off":
        white_point: int | str | None = None
    elif args.white_point == "auto":
        white_point = "auto"
    else:
        white_point = int(args.white_point)

    try:
        written = run_pipeline(
            model_path=args.model,
            input_path=args.input,
            output_path=args.output,
            device=args.device,
            white_point=white_point,
            min_dot_area=args.min_dot_area,
            ink_threshold=args.ink_threshold,
            thicken_amount=args.thicken_amount,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Done. {len(written)} file(s) written.")


if __name__ == "__main__":
    main()
