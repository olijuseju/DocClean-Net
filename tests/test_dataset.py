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

from data.generate_dataset import _sample_archetype_grid_params, generate_pair
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


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5.1 — generate_pair con mezcla domain-robust
# ══════════════════════════════════════════════════════════════════════════════


class TestGeneratePairDomainRobust:

    def test_generate_pair_output_shapes_and_dtype(self) -> None:
        dirty, clean = generate_pair(idx=0, size=128, seed=123)
        assert dirty.shape == (128, 128, 3)
        assert clean.shape == (128, 128, 3)
        assert dirty.dtype == np.uint8
        assert clean.dtype == np.uint8

    def test_generate_pair_deterministic_for_same_idx_and_seed(self) -> None:
        d_a, c_a = generate_pair(idx=3, size=128, seed=42)
        d_b, c_b = generate_pair(idx=3, size=128, seed=42)
        np.testing.assert_array_equal(d_a, d_b)
        np.testing.assert_array_equal(c_a, c_b)

    def test_generate_pair_robust_prob_zero_matches_legacy_distribution(self) -> None:
        """Con domain_robust_prob=0.0 el par es idéntico salvo por la moneda
        inicial del rng — se verifica que ningún par de un lote presente
        rasgos exclusivos del régimen robust (papel sombreado < 150)."""
        for idx in range(6):
            dirty, clean = generate_pair(
                idx=idx, size=128, seed=99, domain_robust_prob=0.0
            )
            # papel del clean nunca sombreado: percentil alto sobre 200
            assert float(np.percentile(clean, 90)) > 200.0
            # dirty sin iluminación: su percentil alto también se mantiene alto
            assert float(np.percentile(dirty, 90)) > 195.0

    def test_generate_pair_clean_target_is_never_shaded(self) -> None:
        """La iluminación y las manchas se aplican SOLO al dirty: el clean
        conserva papel brillante incluso con domain_robust_prob=1.0."""
        for idx in range(8):
            _, clean = generate_pair(idx=idx, size=128, seed=7, domain_robust_prob=1.0)
            assert float(np.percentile(clean, 90)) > 195.0

    def test_generate_pair_robust_batch_contains_shaded_dirty_samples(self) -> None:
        """Con domain_robust_prob=1.0 e iluminación al 70%, un lote de pares
        contiene al menos uno con papel sombreado (p25 del dirty por debajo
        del soporte v1.0)."""
        shaded = 0
        for idx in range(12):
            dirty, _ = generate_pair(
                idx=idx, size=128, seed=1234, domain_robust_prob=1.0
            )
            if float(np.percentile(dirty, 25)) < 150.0:
                shaded += 1
        assert shaded >= 1

    def test_generate_pair_robust_prob_changes_output_for_same_seed(self) -> None:
        """La moneda robust consume el rng: prob 0.0 y 1.0 divergen (documenta
        que un dataset v1.1 no es byte-idéntico a uno v1.0 con misma seed)."""
        d_legacy, _ = generate_pair(idx=2, size=128, seed=42, domain_robust_prob=0.0)
        d_robust, _ = generate_pair(idx=2, size=128, seed=42, domain_robust_prob=1.0)
        assert not np.array_equal(d_legacy, d_robust)


class TestMilimetradoArchetype:

    def test_archetype_grid_params_always_dense_dark_opaque(self) -> None:
        """El arquetipo correlaciona los tres ejes en el régimen duro:
        spacing [10,20), gris [60,100), opacity [0.85,1.0], blend opaco."""
        rng = np.random.default_rng(3)
        for _ in range(50):
            kw = _sample_archetype_grid_params(rng)
            assert 10 <= kw["spacing"] < 20
            assert kw["opaque_lines"] is True
            assert 0.85 <= kw["opacity"] <= 1.0
            g = kw["color_bgr"][1]
            assert 60 <= g < 100
            assert kw["color_bgr"][2] == g  # gris casi acromático
            assert kw["color_bgr"][0] >= g  # sesgo azul no negativo

    def test_generate_pair_batch_covers_comic_failure_combo(self) -> None:
        """Sanity de cobertura: en un lote robust, una fracción sustancial
        de pares presenta el combo del fallo real (línea de degradación
        oscura y densa). Con muestreo independiente esto era ~1%; el
        arquetipo lo garantiza."""
        import cv2

        hard = 0
        n = 40
        for idx in range(n):
            dirty, clean = generate_pair(
                idx=idx, size=128, seed=51, domain_robust_prob=1.0
            )
            gd = cv2.cvtColor(dirty, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gc = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mask = (gc - gd) > 8
            if mask.sum() < 100:
                continue
            dark = float(np.percentile(gd[mask], 25)) <= 115
            dense = float(mask.mean()) > 0.10
            if dark and dense:
                hard += 1
        assert hard >= n * 0.10


class TestRealBackgroundPairs:

    @pytest.fixture()
    def bg_dir(self, tmp_path: Path) -> Path:
        """Directorio con 3 tiles cuadrados de 'fondo real' sintetizado."""
        rng = np.random.default_rng(0)
        for i in range(3):
            tile = np.full((256, 256, 3), 235, dtype=np.uint8)
            for y in range(0, 256, 14):  # cuadrícula densa tipo real
                cv2.line(tile, (0, y), (255, y), (120, 100, 100), 1)
                cv2.line(tile, (y, 0), (y, 255), (120, 100, 100), 1)
            noise = rng.integers(-4, 5, size=tile.shape, dtype=np.int16)
            tile = np.clip(tile.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(tmp_path / f"bg_{i}.png"), tile)
        return tmp_path

    def _paths(self, bg_dir: Path) -> tuple[str, ...]:
        return tuple(str(q) for q in sorted(bg_dir.glob("*.png")))

    def test_real_bg_pair_side_never_exceeds_tile_side(self, bg_dir: Path) -> None:
        """Con tiles de 256 y size 512 el par sale a 256: nunca se
        reescala hacia arriba (conservación del spacing real)."""
        paths = self._paths(bg_dir)
        for idx in range(6):
            dirty, clean = generate_pair(
                idx=idx, size=512, seed=5, real_bg_paths=paths, real_bg_prob=1.0
            )
            assert dirty.shape == (256, 256, 3)
            assert clean.shape == (256, 256, 3)
            assert dirty.dtype == np.uint8

    def test_real_bg_pair_clean_has_no_grid(self, bg_dir: Path) -> None:
        """El clean usa papel sintético: sin la cuadrícula del tile real."""
        paths = self._paths(bg_dir)
        dirty, clean = generate_pair(
            idx=1, size=256, seed=9, real_bg_paths=paths, real_bg_prob=1.0
        )
        gd = cv2.cvtColor(dirty, cv2.COLOR_BGR2GRAY)
        gc = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        band_dirty = ((gd > 90) & (gd < 190)).mean()
        band_clean = ((gc > 90) & (gc < 190)).mean()
        assert band_dirty > band_clean + 0.02

    def test_real_bg_pair_deterministic(self, bg_dir: Path) -> None:
        paths = self._paths(bg_dir)
        a = generate_pair(
            idx=2, size=256, seed=3, real_bg_paths=paths, real_bg_prob=1.0
        )
        b = generate_pair(
            idx=2, size=256, seed=3, real_bg_paths=paths, real_bg_prob=1.0
        )
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_real_bg_prob_zero_ignores_tiles(self, bg_dir: Path) -> None:
        paths = self._paths(bg_dir)
        dirty, _ = generate_pair(
            idx=0, size=512, seed=42, real_bg_paths=paths, real_bg_prob=0.0
        )
        assert dirty.shape == (512, 512, 3)

    def test_real_bg_empty_paths_falls_back_to_synthetic(self) -> None:
        dirty, _ = generate_pair(
            idx=0, size=256, seed=42, real_bg_paths=(), real_bg_prob=1.0
        )
        assert dirty.shape == (256, 256, 3)
