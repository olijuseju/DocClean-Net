#!/usr/bin/env python3
"""
inference/benchmark.py
======================
DL (U-Net) vs classic pipeline comparison on real scans in data/real_test/.

There is no ground-truth clean image for real scans, so metrics follow the
scheme fixed in current_status.md ("Classic used as reference"):

    - ssim / psnr ............ DL output vs classic output (classic = reference)
    - brisque_classic / _dl .. no-reference quality, computed per method
                               (lower is better)
    - ink_classic_pct / _dl .. % of pixels classified as ink (< INK_THRESHOLD)
    - time_classic_ms / _dl .. wall-clock per image

BRISQUE note: the Phase 3 spec attributed BRISQUE to scikit-image, which
does not implement it. We use `piq` (pure-PyTorch implementation) — no
libsvm binaries, no opencv-contrib, reuses the torch dependency.

Outputs:
    <output-dir>/metrics.csv
    <output-dir>/benchmark_metrics.png   (per-image metric bars)
    <output-dir>/benchmark_summary.png   (mean-per-method table)

CLI:
    python -m inference.benchmark --model checkpoints/best.pt \
        --test-dir data/real_test/
"""

import argparse
import contextlib
import csv
import io
import sys
import tempfile
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # headless backend: no display available in CI/servers
import matplotlib.pyplot as plt
import numpy as np
import piq
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from classic_pipeline.digitize_notebook import digitize
from inference.io_utils import _imread
from inference.predict import predict_image

# A pixel darker than this is counted as ink. The classic pipeline output is
# strictly binary {0, 255}; the U-Net output is continuous, so the midpoint
# gives a fair, symmetric cut for both.
INK_THRESHOLD = 128

CSV_COLUMNS = [
    "image",
    "ssim",
    "psnr",
    "brisque_classic",
    "brisque_dl",
    "ink_classic_pct",
    "ink_dl_pct",
    "time_classic_ms",
    "time_dl_ms",
]

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _discover_images(test_dir: Path) -> list[Path]:
    """List image files in test_dir, sorted by name.

    Args:
        test_dir: directory containing real test scans.

    Returns:
        list[Path]: image paths (may be empty).
    """
    return sorted(
        p for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _run_classic(image_path: Path, tmp_dir: Path) -> tuple[np.ndarray, float]:
    """Run the classic pipeline on one image, timed.

    digitize() only offers a file-based API and prints progress; we redirect
    stdout rather than modify the (frozen) classic pipeline.

    Args:
        image_path: input scan.
        tmp_dir: writable directory for the intermediate output file.

    Returns:
        tuple:
            - np.ndarray: classic result, shape (H, W), dtype uint8,
              binary {0=ink, 255=paper}.
            - float: elapsed milliseconds.
    """
    out_path = tmp_dir / f"classic_{image_path.stem}.png"
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        digitize(str(image_path), str(out_path))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    result = _imread(out_path, flags=cv2.IMREAD_GRAYSCALE)
    return result, elapsed_ms


def _run_dl(model_path: Path, image_path: Path, device: str) -> tuple[np.ndarray, float]:
    """Run U-Net sliding-window inference on one image, timed.

    Args:
        model_path: trained checkpoint.
        image_path: input scan.
        device: "auto", "cpu" or "cuda".

    Returns:
        tuple:
            - np.ndarray: DL result, shape (H, W), dtype uint8, continuous
              grayscale (0=ink, 255=paper).
            - float: elapsed milliseconds (includes model load — reported
              per image so both methods are measured cold, like a user
              running the CLI once).
    """
    t0 = time.perf_counter()
    result = predict_image(model_path, image_path, device=device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result, elapsed_ms


def _brisque_score(image_gray: np.ndarray) -> float:
    """No-reference BRISQUE quality score (lower = better).

    Args:
        image_gray (np.ndarray): grayscale image, shape (H, W), dtype uint8.

    Returns:
        float: BRISQUE score. NaN if the computation fails (e.g. image too
            small for the multiscale analysis).
    """
    tensor = (
        torch.from_numpy(image_gray.astype(np.float32) / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)  # (1, 1, H, W)
    )
    try:
        return float(piq.brisque(tensor, data_range=1.0))
    except (RuntimeError, ValueError, AssertionError):
        # piq asserts on degenerate inputs (e.g. near-constant binary images,
        # which the classic pipeline produces for stroke-free paper). BRISQUE
        # is undefined there; NaN is excluded from the summary means.
        return float("nan")


def _ink_coverage_pct(image_gray: np.ndarray) -> float:
    """Percentage of pixels classified as ink (darker than INK_THRESHOLD).

    Args:
        image_gray (np.ndarray): grayscale image, shape (H, W), dtype uint8.

    Returns:
        float: ink pixels / total pixels * 100.
    """
    return float((image_gray < INK_THRESHOLD).mean() * 100.0)


def benchmark_image(
    model_path: Path, image_path: Path, tmp_dir: Path, device: str
) -> dict[str, float | str]:
    """Compute all Phase 3 metrics for a single image.

    Args:
        model_path: trained checkpoint.
        image_path: input scan.
        tmp_dir: writable dir for the classic pipeline's file output.
        device: torch device string.

    Returns:
        dict: one row keyed by CSV_COLUMNS.
    """
    classic, time_classic = _run_classic(image_path, tmp_dir)
    dl, time_dl = _run_dl(model_path, image_path, device)

    # Classic output is the reference (see module docstring).
    ssim = float(structural_similarity(classic, dl, data_range=255))
    psnr = float(peak_signal_noise_ratio(classic, dl, data_range=255))

    return {
        "image": image_path.name,
        "ssim": round(ssim, 4),
        "psnr": round(psnr, 2),
        "brisque_classic": round(_brisque_score(classic), 2),
        "brisque_dl": round(_brisque_score(dl), 2),
        "ink_classic_pct": round(_ink_coverage_pct(classic), 2),
        "ink_dl_pct": round(_ink_coverage_pct(dl), 2),
        "time_classic_ms": round(time_classic, 1),
        "time_dl_ms": round(time_dl, 1),
    }


def _write_csv(rows: list[dict[str, float | str]], csv_path: Path) -> None:
    """Write metric rows to CSV. Header is written even with zero rows.

    Args:
        rows: list of metric dicts (keys = CSV_COLUMNS).
        csv_path: destination file.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _plot_metrics(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    """Save per-image bar plots and a summary table as PNG figures.

    Args:
        rows: list of metric dicts (non-empty).
        output_dir: destination directory.
    """
    names = [str(r["image"]) for r in rows]
    x = np.arange(len(names))
    width = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("DocClean-Net — DL vs classic pipeline (real scans)")

    # SSIM / PSNR: DL measured against the classic reference.
    ax = axes[0][0]
    ax.bar(x, [float(r["ssim"]) for r in rows], color="#4c72b0")
    ax.set_title("SSIM (DL vs classic reference)")
    ax.set_ylim(0.0, 1.0)

    ax = axes[0][1]
    ax.bar(x - width / 2, [float(r["brisque_classic"]) for r in rows],
           width, label="classic", color="#dd8452")
    ax.bar(x + width / 2, [float(r["brisque_dl"]) for r in rows],
           width, label="DL", color="#55a868")
    ax.set_title("BRISQUE (lower = better)")
    ax.legend()

    ax = axes[1][0]
    ax.bar(x - width / 2, [float(r["ink_classic_pct"]) for r in rows],
           width, label="classic", color="#dd8452")
    ax.bar(x + width / 2, [float(r["ink_dl_pct"]) for r in rows],
           width, label="DL", color="#55a868")
    ax.set_title("Ink coverage (%)")
    ax.legend()

    ax = axes[1][1]
    ax.bar(x - width / 2, [float(r["time_classic_ms"]) for r in rows],
           width, label="classic", color="#dd8452")
    ax.bar(x + width / 2, [float(r["time_dl_ms"]) for r in rows],
           width, label="DL", color="#55a868")
    ax.set_title("Inference time (ms)")
    ax.legend()

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_metrics.png", dpi=150)
    plt.close(fig)

    # Summary table: mean per method (finite values only, PSNR can be inf).
    def _mean(key: str) -> float:
        values = np.array([float(r[key]) for r in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else float("nan")

    table_rows = [
        ["SSIM (DL vs classic)", f"{_mean('ssim'):.4f}", "—"],
        ["PSNR dB (DL vs classic)", f"{_mean('psnr'):.2f}", "—"],
        ["BRISQUE", f"{_mean('brisque_dl'):.2f}", f"{_mean('brisque_classic'):.2f}"],
        ["Ink coverage %", f"{_mean('ink_dl_pct'):.2f}", f"{_mean('ink_classic_pct'):.2f}"],
        ["Time ms", f"{_mean('time_dl_ms'):.1f}", f"{_mean('time_classic_ms'):.1f}"],
    ]
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=["Metric", "DL (U-Net)", "Classic"],
        loc="center",
        cellLoc="center",
    )
    table.scale(1.0, 1.4)
    ax.set_title(f"Summary — mean over {len(rows)} image(s)")
    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_summary.png", dpi=150)
    plt.close(fig)


def run_benchmark(
    model_path: str | Path,
    test_dir: str | Path,
    output_dir: str | Path = "benchmark_results",
    device: str = "auto",
) -> list[dict[str, float | str]]:
    """Benchmark DL vs classic pipeline over every image in test_dir.

    Args:
        model_path: trained checkpoint (e.g. checkpoints/best.pt).
        test_dir: directory of real scans. May be empty — a header-only
            metrics.csv is still produced and no plots are drawn.
        output_dir: where metrics.csv and plot PNGs are written.
        device: "auto", "cpu" or "cuda".

    Returns:
        list[dict]: one metrics row per image (empty list if no images).

    Raises:
        FileNotFoundError: if test_dir does not exist.
    """
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    images = _discover_images(test_dir)
    rows: list[dict[str, float | str]] = []

    if not images:
        print(f"[WARN] No images found in {test_dir} — writing empty metrics.csv")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for image_path in images:
                print(f"Benchmarking {image_path.name} ...")
                rows.append(
                    benchmark_image(Path(model_path), image_path, tmp_dir, device)
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "metrics.csv")
    if rows:
        _plot_metrics(rows, output_dir)

    print(f"Done. Results in {output_dir}/")
    return rows


def main() -> None:
    """CLI entry point: python -m inference.benchmark ..."""
    parser = argparse.ArgumentParser(
        prog="inference.benchmark",
        description="Benchmark DocClean-Net U-Net vs classic pipeline.",
    )
    parser.add_argument("--model", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--test-dir", required=True, help="Directory of real scans")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = parser.parse_args()

    try:
        run_benchmark(
            model_path=args.model,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            device=args.device,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
