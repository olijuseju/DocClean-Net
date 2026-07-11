"""
tests/test_dataset.py
=====================
Tests for model/dataset.py — DocCleanDataset.

All tests use a small synthetic dataset written to a tmp directory.
No real images from data/synthetic/ are required.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from model.dataset import DocCleanDataset, _extract_patch, _imread, _to_tensor

# ── Fixtures ──────────────────────────────────────────────────────────────────

IMAGE_H = 512
IMAGE_W = 512
PATCH_SIZE = 256
N_PAIRS = 6  # small enough to keep tests fast


@pytest.fixture(scope="module")
def synthetic_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a minimal on-disk dataset with N_PAIRS (dirty, clean) pairs.

    Images are 512×512 BGR uint8 with random content, written as PNG.
    The fixture uses module scope so it is created once per test session.
    """
    root = tmp_path_factory.mktemp("synthetic")
    dirty_dir = root / "dirty"
    clean_dir = root / "clean"
    dirty_dir.mkdir()
    clean_dir.mkdir()

    rng = np.random.default_rng(seed=99)

    for i in range(N_PAIRS):
        dirty = rng.integers(0, 256, (IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
        clean = rng.integers(0, 256, (IMAGE_H, IMAGE_W, 3), dtype=np.uint8)

        dirty_name = f"dirty_{i:06d}.png"
        clean_name = f"clean_{i:06d}.png"

        cv2.imwrite(str(dirty_dir / dirty_name), dirty)
        cv2.imwrite(str(clean_dir / clean_name), clean)

    return root


@pytest.fixture(scope="module")
def dataset_no_aug(synthetic_data_dir: Path) -> DocCleanDataset:
    """Dataset with augmentation disabled (deterministic output)."""
    return DocCleanDataset(
        synthetic_data_dir, patch_size=PATCH_SIZE, augment=False, seed=0
    )


@pytest.fixture(scope="module")
def dataset_aug(synthetic_data_dir: Path) -> DocCleanDataset:
    """Dataset with augmentation enabled."""
    return DocCleanDataset(
        synthetic_data_dir, patch_size=PATCH_SIZE, augment=True, seed=0
    )


# ── Construction and length ───────────────────────────────────────────────────


def test_dataset_length_matches_number_of_pairs(
    dataset_no_aug: DocCleanDataset,
) -> None:
    assert len(dataset_no_aug) == N_PAIRS


def test_dataset_raises_if_dirty_dir_missing(tmp_path: Path) -> None:
    (tmp_path / "clean").mkdir()
    with pytest.raises(FileNotFoundError, match="dirty/"):
        DocCleanDataset(tmp_path)


def test_dataset_raises_if_clean_dir_missing(tmp_path: Path) -> None:
    (tmp_path / "dirty").mkdir()
    with pytest.raises(FileNotFoundError, match="clean/"):
        DocCleanDataset(tmp_path)


def test_dataset_raises_if_no_matching_pairs(tmp_path: Path) -> None:
    (tmp_path / "dirty").mkdir()
    (tmp_path / "clean").mkdir()
    # Write a dirty file with no matching clean counterpart
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "dirty" / "dirty_000000.png"), img)
    with pytest.raises(ValueError, match="No matching"):
        DocCleanDataset(tmp_path)


def test_dataset_repr_contains_key_info(dataset_no_aug: DocCleanDataset) -> None:
    r = repr(dataset_no_aug)
    assert "DocCleanDataset" in r
    assert str(N_PAIRS) in r
    assert str(PATCH_SIZE) in r


# ── __getitem__ output contract ───────────────────────────────────────────────


def test_dataset_getitem_returns_two_tensors(
    dataset_no_aug: DocCleanDataset,
) -> None:
    item = dataset_no_aug[0]
    assert isinstance(item, tuple) and len(item) == 2
    assert isinstance(item[0], torch.Tensor)
    assert isinstance(item[1], torch.Tensor)


def test_dataset_getitem_output_shape_is_patch_size(
    dataset_no_aug: DocCleanDataset,
) -> None:
    dirty, clean = dataset_no_aug[0]
    assert dirty.shape == (1, PATCH_SIZE, PATCH_SIZE)
    assert clean.shape == (1, PATCH_SIZE, PATCH_SIZE)


def test_dataset_getitem_output_dtype_is_float32(
    dataset_no_aug: DocCleanDataset,
) -> None:
    dirty, clean = dataset_no_aug[0]
    assert dirty.dtype == torch.float32
    assert clean.dtype == torch.float32


def test_dataset_getitem_output_values_in_unit_interval(
    dataset_no_aug: DocCleanDataset,
) -> None:
    for i in range(N_PAIRS):
        dirty, clean = dataset_no_aug[i]
        assert float(dirty.min()) >= 0.0, f"dirty min < 0 at index {i}"
        assert float(dirty.max()) <= 1.0, f"dirty max > 1 at index {i}"
        assert float(clean.min()) >= 0.0, f"clean min < 0 at index {i}"
        assert float(clean.max()) <= 1.0, f"clean max > 1 at index {i}"


def test_dataset_getitem_all_indices_accessible(
    dataset_no_aug: DocCleanDataset,
) -> None:
    """No index in [0, len) should raise."""
    for i in range(len(dataset_no_aug)):
        dirty, clean = dataset_no_aug[i]
        assert dirty.shape == (1, PATCH_SIZE, PATCH_SIZE)


# ── Patch extraction ──────────────────────────────────────────────────────────


def test_extract_patch_output_shape_matches_patch_size() -> None:
    img_a = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
    img_b = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed=0)
    pa, pb = _extract_patch(img_a, img_b, PATCH_SIZE, rng)
    assert pa.shape == (PATCH_SIZE, PATCH_SIZE, 3)
    assert pb.shape == (PATCH_SIZE, PATCH_SIZE, 3)


def test_extract_patch_same_crop_location_for_both_images() -> None:
    """Dirty and clean patches must be cropped at the same (top, left)."""
    # Fill each image with its own unique constant so we can check alignment.
    dirty = np.full((IMAGE_H, IMAGE_W, 3), fill_value=100, dtype=np.uint8)
    clean = np.full((IMAGE_H, IMAGE_W, 3), fill_value=200, dtype=np.uint8)

    # Plant a marker at a known location in dirty only.
    dirty[128:256, 128:256] = 42

    # Run many crops; the marker region should never bleed into clean patch.
    for _ in range(20):
        rng2 = np.random.default_rng(seed=7)
        pd, pc = _extract_patch(dirty.copy(), clean.copy(), PATCH_SIZE, rng2)
        # Clean patch must be uniformly 200 (no dirty content)
        assert (pc == 200).all(), "Patches were not extracted at the same location"


def test_extract_patch_does_not_modify_input_arrays() -> None:
    dirty = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
    clean = np.ones((IMAGE_H, IMAGE_W, 3), dtype=np.uint8) * 128
    dirty_copy = dirty.copy()
    clean_copy = clean.copy()
    rng = np.random.default_rng(seed=0)
    _extract_patch(dirty, clean, PATCH_SIZE, rng)
    np.testing.assert_array_equal(dirty, dirty_copy)
    np.testing.assert_array_equal(clean, clean_copy)


# ── _to_tensor ────────────────────────────────────────────────────────────────


def test_to_tensor_output_shape_adds_channel_dimension() -> None:
    img = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
    t = _to_tensor(img)
    assert t.shape == (1, PATCH_SIZE, PATCH_SIZE)


def test_to_tensor_normalises_255_to_one() -> None:
    img = np.full((64, 64, 3), fill_value=255, dtype=np.uint8)
    t = _to_tensor(img)
    assert float(t.max()) == pytest.approx(1.0, abs=1e-6)


def test_to_tensor_normalises_0_to_zero() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    t = _to_tensor(img)
    assert float(t.min()) == pytest.approx(0.0, abs=1e-6)


def test_to_tensor_output_dtype_is_float32() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    assert _to_tensor(img).dtype == torch.float32


def test_to_tensor_converts_bgr_to_grayscale_correctly() -> None:
    """Pure-blue BGR pixel (255, 0, 0) must not produce the same gray value
    as pure-red (0, 0, 255) — confirms channel order is handled correctly."""
    blue_img = np.zeros((4, 4, 3), dtype=np.uint8)
    blue_img[:, :, 0] = 255  # B channel

    red_img = np.zeros((4, 4, 3), dtype=np.uint8)
    red_img[:, :, 2] = 255  # R channel

    t_blue = _to_tensor(blue_img)
    t_red = _to_tensor(red_img)
    # OpenCV BGR2GRAY weights: Y = 0.114*B + 0.587*G + 0.299*R
    # blue → ~29/255 ≈ 0.114, red → ~76/255 ≈ 0.299 — must differ
    assert not torch.allclose(t_blue, t_red), (
        "Blue and red pixels produced the same grayscale value — "
        "BGR→GRAY conversion may be wrong."
    )


# ── _imread ───────────────────────────────────────────────────────────────────


def test_imread_loads_image_with_correct_shape(tmp_path: Path) -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    path = tmp_path / "test.png"
    cv2.imwrite(str(path), img)
    loaded = _imread(path)
    assert loaded.shape == (64, 64, 3)
    assert loaded.dtype == np.uint8


def test_imread_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _imread(tmp_path / "nonexistent.png")


# ── Augmentation flag ─────────────────────────────────────────────────────────


def test_dataset_augmentation_disabled_is_deterministic(
    dataset_no_aug: DocCleanDataset,
) -> None:
    """Same index accessed twice must return identical tensors when aug=False."""
    d1, c1 = dataset_no_aug[0]
    d2, c2 = dataset_no_aug[0]
    assert torch.equal(d1, d2)
    assert torch.equal(c1, c2)


def test_dataset_augmentation_enabled_does_not_crash(
    dataset_aug: DocCleanDataset,
) -> None:
    """Smoke test: augmented dataset must not raise on any index."""
    for i in range(len(dataset_aug)):
        dirty, clean = dataset_aug[i]
        assert dirty.shape == (1, PATCH_SIZE, PATCH_SIZE)
        assert clean.shape == (1, PATCH_SIZE, PATCH_SIZE)
