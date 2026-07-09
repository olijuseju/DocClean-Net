"""
Tests for inference/io_utils.py, inference/predict.py and
inference/benchmark.py.

Fast tests use a freshly-initialized (untrained) UNet saved to a tmp
checkpoint: shape/dtype/range contracts don't depend on training. The
end-to-end tests that need checkpoints/best.pt (gitignored, distributed
via GitHub Releases) are marked slow and skip when it's absent.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from inference.benchmark import (
    CSV_COLUMNS,
    _brisque_score,
    _ink_coverage_pct,
    run_benchmark,
)
from inference.io_utils import _imread, _imwrite
from inference.predict import (
    _compute_padded_size,
    _gaussian_window,
    predict_image,
)
from model.unet import UNet


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def untrained_checkpoint(tmp_output_dir: Path) -> Path:
    """State_dict of a freshly-initialized UNet — no training needed."""
    model = UNet(in_channels=1, out_channels=1)
    ckpt_path = tmp_output_dir / "untrained.pt"
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path


@pytest.fixture()
def sample_scan_path(tmp_output_dir: Path, noisy_gray_image: np.ndarray) -> Path:
    """A 64x64 noisy paper-like scan written to disk as BGR PNG."""
    bgr = cv2.cvtColor(noisy_gray_image, cv2.COLOR_GRAY2BGR)
    path = tmp_output_dir / "scan.png"
    _imwrite(path, bgr)
    return path


def _noisy_bgr(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Paper-like BGR image of arbitrary size (mean 245, sigma 4)."""
    noise = rng.normal(loc=245.0, scale=4.0, size=(h, w))
    gray = np.clip(noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ── io_utils ──────────────────────────────────────────────────────────────────


def test_imwrite_then_imread_roundtrip_preserves_shape_and_values(
    tmp_output_dir: Path, white_image_bgr: np.ndarray
) -> None:
    path = tmp_output_dir / "roundtrip.png"
    _imwrite(path, white_image_bgr)
    loaded = _imread(path)
    assert loaded.shape == white_image_bgr.shape
    assert np.array_equal(loaded, white_image_bgr)  # PNG is lossless


def test_imread_raises_filenotfounderror_for_missing_path(
    tmp_output_dir: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        _imread(tmp_output_dir / "does_not_exist.png")


def test_imwrite_and_imread_handle_unicode_path(
    tmp_output_dir: Path, white_image_bgr: np.ndarray
) -> None:
    # Mirrors real filenames like "Escáner_20230219__19_.png".
    path = tmp_output_dir / "Escáner_ñoño_20230219.png"
    _imwrite(path, white_image_bgr)
    loaded = _imread(path)
    assert loaded.shape == white_image_bgr.shape


# ── predict helpers ───────────────────────────────────────────────────────────


def test_gaussian_window_peak_at_center_and_normalized() -> None:
    window = _gaussian_window(patch_size=64)
    assert window.shape == (64, 64)
    assert window.dtype == np.float32
    assert np.isclose(window.max(), 1.0, atol=1e-6)
    assert window[32, 32] > window[0, 0]


@pytest.mark.parametrize(
    "size,patch_size,stride,expected",
    [
        (256, 256, 128, 256),  # exact fit
        (100, 256, 128, 256),  # smaller than one patch -> pad to patch_size
        (300, 256, 128, 384),  # 256 + ceil(44/128)*128
        (512, 256, 128, 512),  # multiple of stride, exact tiling
    ],
)
def test_compute_padded_size(
    size: int, patch_size: int, stride: int, expected: int
) -> None:
    assert _compute_padded_size(size, patch_size, stride) == expected


# ── predict_image ─────────────────────────────────────────────────────────────


def test_predict_image_output_shape_matches_input(
    untrained_checkpoint: Path, sample_scan_path: Path
) -> None:
    result = predict_image(untrained_checkpoint, sample_scan_path, device="cpu")
    input_img = _imread(sample_scan_path)
    assert result.shape == input_img.shape[:2]


def test_predict_image_output_is_uint8_in_valid_range(
    untrained_checkpoint: Path, sample_scan_path: Path
) -> None:
    result = predict_image(untrained_checkpoint, sample_scan_path, device="cpu")
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


@pytest.mark.parametrize("h,w", [(137, 211), (50, 60), (300, 260)])
def test_predict_image_handles_sizes_not_multiple_of_patch(
    untrained_checkpoint: Path,
    tmp_output_dir: Path,
    rng: np.random.Generator,
    h: int,
    w: int,
) -> None:
    path = tmp_output_dir / f"scan_{h}x{w}.png"
    _imwrite(path, _noisy_bgr(h, w, rng))
    result = predict_image(untrained_checkpoint, path, device="cpu")
    assert result.shape == (h, w)


def test_predict_image_batch_size_does_not_change_result(
    untrained_checkpoint: Path, tmp_output_dir: Path, rng: np.random.Generator
) -> None:
    # Batching is a throughput knob only — results must be bit-identical.
    path = tmp_output_dir / "scan_batching.png"
    _imwrite(path, _noisy_bgr(300, 300, rng))
    r1 = predict_image(untrained_checkpoint, path, device="cpu", batch_size=1)
    r16 = predict_image(untrained_checkpoint, path, device="cpu", batch_size=16)
    assert np.array_equal(r1, r16)


def test_predict_image_raises_filenotfounderror_for_missing_model(
    sample_scan_path: Path, tmp_output_dir: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        predict_image(tmp_output_dir / "missing.pt", sample_scan_path, device="cpu")


def test_predict_image_raises_filenotfounderror_for_missing_image(
    untrained_checkpoint: Path, tmp_output_dir: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        predict_image(
            untrained_checkpoint, tmp_output_dir / "missing.png", device="cpu"
        )


def test_predict_image_raises_valueerror_when_stride_exceeds_patch_size(
    untrained_checkpoint: Path, sample_scan_path: Path
) -> None:
    with pytest.raises(ValueError):
        predict_image(
            untrained_checkpoint, sample_scan_path, stride=300, device="cpu"
        )


# ── benchmark helpers ─────────────────────────────────────────────────────────


def test_ink_coverage_pct_on_known_image() -> None:
    img = np.full((10, 10), 255, dtype=np.uint8)
    img[:5, :] = 0  # top half ink
    assert _ink_coverage_pct(img) == pytest.approx(50.0)


def test_brisque_score_returns_finite_float_on_natural_image(
    rng: np.random.Generator,
) -> None:
    noise = rng.normal(loc=128.0, scale=30.0, size=(256, 256))
    img = np.clip(noise, 0, 255).astype(np.uint8)
    score = _brisque_score(img)
    assert isinstance(score, float)
    assert np.isfinite(score)


# ── run_benchmark ─────────────────────────────────────────────────────────────


def test_run_benchmark_empty_test_dir_writes_header_only_csv(
    untrained_checkpoint: Path, tmp_output_dir: Path
) -> None:
    empty_dir = tmp_output_dir / "real_test_empty"
    empty_dir.mkdir()
    out_dir = tmp_output_dir / "results"

    rows = run_benchmark(
        untrained_checkpoint, empty_dir, output_dir=out_dir, device="cpu"
    )

    assert rows == []
    csv_path = out_dir / "metrics.csv"
    assert csv_path.exists()
    header = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert header == [",".join(CSV_COLUMNS)]


def test_run_benchmark_missing_test_dir_raises_filenotfounderror(
    untrained_checkpoint: Path, tmp_output_dir: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        run_benchmark(
            untrained_checkpoint,
            tmp_output_dir / "nope",
            output_dir=tmp_output_dir / "results",
            device="cpu",
        )


@pytest.mark.slow
def test_run_benchmark_generates_csv_with_correct_columns(
    untrained_checkpoint: Path, tmp_output_dir: Path, rng: np.random.Generator
) -> None:
    """Full benchmark on one small synthetic scan (classic pipeline + UNet).

    Marked slow: the classic pipeline's inpainting step takes a few seconds
    even on a 300x300 image.
    """
    test_dir = tmp_output_dir / "real_test"
    test_dir.mkdir()
    _imwrite(test_dir / "sample.png", _noisy_bgr(300, 300, rng))
    out_dir = tmp_output_dir / "results"

    rows = run_benchmark(
        untrained_checkpoint, test_dir, output_dir=out_dir, device="cpu"
    )

    assert len(rows) == 1
    assert list(rows[0].keys()) == CSV_COLUMNS
    csv_lines = (out_dir / "metrics.csv").read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines[0] == ",".join(CSV_COLUMNS)
    assert len(csv_lines) == 2
    assert (out_dir / "benchmark_metrics.png").exists()
    assert (out_dir / "benchmark_summary.png").exists()


@pytest.mark.slow
def test_predict_image_with_real_checkpoint_on_real_scan() -> None:
    """End-to-end sanity check with the trained checkpoint, if present."""
    model_path = Path("checkpoints/best.pt")
    if not model_path.exists():
        pytest.skip("checkpoints/best.pt not present locally")

    test_images = sorted(Path("data/real_test").glob("*.png"))
    if not test_images:
        pytest.skip("no images in data/real_test/")

    result = predict_image(model_path, test_images[0], device="auto")
    img = _imread(test_images[0])
    assert result.shape == img.shape[:2]
    assert result.dtype == np.uint8


if __name__ == "__main__":
    pass


# ── white point normalisation ─────────────────────────────────────────────────


def test_estimate_white_point_returns_mode_minus_margin() -> None:
    from inference.predict import _estimate_white_point

    img = np.full((100, 100), 233, dtype=np.uint8)  # paper-dominant image
    assert _estimate_white_point(img, margin=10) == 223


def test_apply_white_point_saturates_paper_and_preserves_ink() -> None:
    from inference.predict import _apply_white_point

    img = np.full((10, 10), 233, dtype=np.uint8)
    img[0, 0] = 0     # pure ink
    img[0, 1] = 224   # residual at/above the white point -> clips to white
    img[0, 2] = 220   # residual just below -> stretches to near-white
    out = _apply_white_point(img, white_point=223)
    assert out[5, 5] == 255   # paper -> pure white
    assert out[0, 1] == 255
    assert out[0, 2] >= 250   # visually indistinguishable from white
    assert out[0, 0] == 0     # ink untouched


def test_apply_white_point_rejects_invalid_value() -> None:
    from inference.predict import _apply_white_point

    with pytest.raises(ValueError):
        _apply_white_point(np.zeros((4, 4), dtype=np.uint8), white_point=0)


def test_predict_image_white_point_none_returns_raw_output(
    untrained_checkpoint: Path, sample_scan_path: Path
) -> None:
    raw = predict_image(
        untrained_checkpoint, sample_scan_path, device="cpu", white_point=None
    )
    auto = predict_image(
        untrained_checkpoint, sample_scan_path, device="cpu", white_point="auto"
    )
    # auto stretches: its max must reach 255 and be >= the raw maximum.
    assert auto.max() == 255
    assert auto.max() >= raw.max()


# ── scripts/visualize_results.py ──────────────────────────────────────────────
# scripts/ is not a package (no __init__.py, excluded from coverage), so we
# load it by file path for testing.


def _load_visualize_module():
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / "visualize_results.py"
    spec = importlib.util.spec_from_file_location("visualize_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_crop_region_clamps_to_image_bounds() -> None:
    viz = _load_visualize_module()
    img = np.zeros((100, 100), dtype=np.uint8)
    region = viz._crop_region(img, x=80, y=80, w=50, h=50)  # overflows bounds
    assert region.shape == (20, 20)


def test_build_comparison_figure_without_crop_has_three_panels(
    rng: np.random.Generator,
) -> None:
    viz = _load_visualize_module()
    original = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    gray = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    fig = viz.build_comparison_figure(original, gray, gray, crop=None)
    assert len(fig.axes) == 3
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_build_comparison_figure_with_crop_has_six_panels(
    rng: np.random.Generator,
) -> None:
    viz = _load_visualize_module()
    original = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    gray = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
    fig = viz.build_comparison_figure(original, gray, gray, crop=(10, 10, 30, 30))
    assert len(fig.axes) == 6
    import matplotlib.pyplot as plt

    plt.close(fig)


# ── scripts/thicken_strokes.py ────────────────────────────────────────────────


def _load_thicken_module():
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / "thicken_strokes.py"
    spec = importlib.util.spec_from_file_location("thicken_strokes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_thicken_strokes_amount_zero_is_noop() -> None:
    viz = _load_thicken_module()
    img = np.full((20, 20), 255, dtype=np.uint8)
    img[10, 10] = 0
    out = viz.thicken_strokes(img, amount=0)
    assert np.array_equal(out, img)
    assert out is not img  # returns a copy, not the same array


def test_thicken_strokes_grows_a_single_ink_pixel() -> None:
    viz = _load_thicken_module()
    img = np.full((21, 21), 255, dtype=np.uint8)
    img[10, 10] = 0
    out = viz.thicken_strokes(img, amount=1)
    ink_pixels = int((out < 255).sum())
    assert ink_pixels > 1  # a single dark pixel spreads to its neighbors


def test_thicken_strokes_larger_amount_spreads_further() -> None:
    viz = _load_thicken_module()
    img = np.full((41, 41), 255, dtype=np.uint8)
    img[20, 20] = 0
    out1 = viz.thicken_strokes(img, amount=1)
    out2 = viz.thicken_strokes(img, amount=2)
    assert (out2 < 255).sum() > (out1 < 255).sum()


def test_thicken_strokes_rejects_negative_amount() -> None:
    viz = _load_thicken_module()
    with pytest.raises(ValueError):
        viz.thicken_strokes(np.zeros((5, 5), dtype=np.uint8), amount=-1)


def test_thicken_strokes_output_shape_and_dtype_preserved() -> None:
    viz = _load_thicken_module()
    img = np.full((30, 40), 255, dtype=np.uint8)
    out = viz.thicken_strokes(img, amount=1)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_remove_small_dots_erases_isolated_single_pixel() -> None:
    viz = _load_thicken_module()
    img = np.full((20, 20), 255, dtype=np.uint8)
    img[10, 10] = 0  # 1px isolated dot
    out = viz.remove_small_dots(img, min_area=3)
    assert out[10, 10] == 255
    assert (out == 255).all()


def test_remove_small_dots_keeps_components_at_or_above_min_area() -> None:
    viz = _load_thicken_module()
    img = np.full((20, 20), 255, dtype=np.uint8)
    img[10:13, 10] = 0  # 3px vertical component, area == min_area
    out = viz.remove_small_dots(img, min_area=3)
    assert (out[10:13, 10] == 0).all()


def test_remove_small_dots_removes_component_below_min_area_only() -> None:
    viz = _load_thicken_module()
    img = np.full((20, 20), 255, dtype=np.uint8)
    img[10, 10] = 0            # 1px dot, should be removed
    img[15:18, 15] = 0         # 3px stroke, should survive
    out = viz.remove_small_dots(img, min_area=3)
    assert out[10, 10] == 255
    assert (out[15:18, 15] == 0).all()


def test_remove_small_dots_rejects_invalid_min_area() -> None:
    viz = _load_thicken_module()
    with pytest.raises(ValueError):
        viz.remove_small_dots(np.zeros((5, 5), dtype=np.uint8), min_area=0)


def test_remove_small_dots_output_shape_and_dtype_preserved() -> None:
    viz = _load_thicken_module()
    img = np.full((30, 40), 255, dtype=np.uint8)
    out = viz.remove_small_dots(img, min_area=3)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


# ── scripts/run_pipeline.py ───────────────────────────────────────────────────


def _load_run_pipeline_module():
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_process_image_returns_uint8_grayscale(
    untrained_checkpoint: Path, sample_scan_path: Path
) -> None:
    rp = _load_run_pipeline_module()
    result = rp.process_image(untrained_checkpoint, sample_scan_path, device="cpu")
    assert result.dtype == np.uint8
    assert result.ndim == 2


def test_process_image_amount_zero_and_min_dot_area_zero_skip_postprocessing(
    untrained_checkpoint: Path, sample_scan_path: Path, monkeypatch
) -> None:
    rp = _load_run_pipeline_module()
    calls = {"denoise": 0, "thicken": 0}

    def fake_denoise(img, min_area, ink_threshold):
        calls["denoise"] += 1
        return img

    def fake_thicken(img, amount):
        calls["thicken"] += 1
        return img

    monkeypatch.setattr(rp, "remove_small_dots", fake_denoise)
    monkeypatch.setattr(rp, "thicken_strokes", fake_thicken)

    rp.process_image(
        untrained_checkpoint, sample_scan_path, device="cpu",
        min_dot_area=0, thicken_amount=0,
    )
    assert calls == {"denoise": 0, "thicken": 0}


def test_run_pipeline_single_file_writes_one_output(
    untrained_checkpoint: Path, sample_scan_path: Path, tmp_output_dir: Path
) -> None:
    rp = _load_run_pipeline_module()
    out_path = tmp_output_dir / "single_result.png"

    written = rp.run_pipeline(
        untrained_checkpoint, sample_scan_path, out_path, device="cpu"
    )

    assert written == [out_path]
    assert out_path.exists()


def test_run_pipeline_directory_processes_every_image(
    untrained_checkpoint: Path, tmp_output_dir: Path, rng: np.random.Generator
) -> None:
    rp = _load_run_pipeline_module()
    in_dir = tmp_output_dir / "batch_in"
    in_dir.mkdir()
    for name in ["a.png", "b.png", "c.png"]:
        _imwrite(in_dir / name, _noisy_bgr(64, 64, rng))
    out_dir = tmp_output_dir / "batch_out"

    written = rp.run_pipeline(untrained_checkpoint, in_dir, out_dir, device="cpu")

    assert len(written) == 3
    assert {p.name for p in written} == {"a.png", "b.png", "c.png"}
    for p in written:
        assert p.exists()


def test_run_pipeline_raises_filenotfounderror_for_missing_input(
    untrained_checkpoint: Path, tmp_output_dir: Path
) -> None:
    rp = _load_run_pipeline_module()
    with pytest.raises(FileNotFoundError):
        rp.run_pipeline(
            untrained_checkpoint, tmp_output_dir / "nope.png",
            tmp_output_dir / "out.png", device="cpu",
        )


def test_run_pipeline_raises_filenotfounderror_for_empty_directory(
    untrained_checkpoint: Path, tmp_output_dir: Path
) -> None:
    rp = _load_run_pipeline_module()
    empty_dir = tmp_output_dir / "empty_in"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        rp.run_pipeline(
            untrained_checkpoint, empty_dir, tmp_output_dir / "out", device="cpu",
        )
