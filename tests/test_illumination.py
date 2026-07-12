"""
tests/test_illumination.py
==========================
Tests para data/generators/illumination.py (Phase 5.1a).

Fixtures reutilizadas de conftest.py:
  - rng : np.random.Generator con semilla fija (scope=session)
"""

import numpy as np
import pytest

from data.generators.illumination import _MODES, apply_illumination

# ══════════════════════════════════════════════════════════════════════════════
# apply_illumination
# ══════════════════════════════════════════════════════════════════════════════


class TestApplyIllumination:

    def test_apply_illumination_output_shape_matches_input(
        self, rng: np.random.Generator
    ) -> None:
        image = np.full((80, 64, 3), 230, dtype=np.uint8)
        result = apply_illumination(image, rng)
        assert result.shape == (80, 64, 3)

    def test_apply_illumination_output_dtype_is_uint8(
        self, rng: np.random.Generator
    ) -> None:
        image = np.full((64, 64, 3), 230, dtype=np.uint8)
        result = apply_illumination(image, rng)
        assert result.dtype == np.uint8

    def test_apply_illumination_does_not_modify_input_in_place(
        self, rng: np.random.Generator
    ) -> None:
        image = np.full((64, 64, 3), 230, dtype=np.uint8)
        backup = image.copy()
        apply_illumination(image, rng)
        assert np.array_equal(image, backup)

    def test_apply_illumination_never_brightens_any_pixel(
        self, rng: np.random.Generator
    ) -> None:
        """La ganancia es ≤ 1.0 en todas partes: solo puede oscurecer."""
        image = np.full((64, 64, 3), 230, dtype=np.uint8)
        for mode in _MODES:
            result = apply_illumination(image, rng, mode=mode)
            assert int(result.max()) <= 230, f"mode={mode} aclaró píxeles"

    def test_apply_illumination_darkest_region_respects_min_gain(
        self, rng: np.random.Generator
    ) -> None:
        """Con min_gain=0.5 sobre papel 200, el mínimo queda cerca de 100."""
        image = np.full((96, 96, 3), 200, dtype=np.uint8)
        result = apply_illumination(image, rng, mode="linear", min_gain=0.5)
        assert int(result.min()) >= 99  # 200 * 0.5, con margen de redondeo
        assert int(result.min()) <= 110

    def test_apply_illumination_brightest_region_keeps_original_level(
        self, rng: np.random.Generator
    ) -> None:
        """El campo está normalizado: la zona mejor iluminada conserva brillo."""
        image = np.full((96, 96, 3), 200, dtype=np.uint8)
        result = apply_illumination(image, rng, mode="linear", min_gain=0.5)
        assert int(result.max()) >= 198

    def test_apply_illumination_black_ink_stays_black(
        self, rng: np.random.Generator
    ) -> None:
        """La ganancia multiplicativa no puede alterar píxeles de tinta (0)."""
        image = np.full((64, 64, 3), 230, dtype=np.uint8)
        image[30:34, :, :] = 0  # trazo horizontal negro
        result = apply_illumination(image, rng, min_gain=0.4)
        assert int(result[30:34, :, :].max()) == 0

    def test_apply_illumination_produces_spatial_gradient(
        self, rng: np.random.Generator
    ) -> None:
        """El campo no es constante: papel uniforme debe salir no uniforme."""
        image = np.full((96, 96, 3), 220, dtype=np.uint8)
        result = apply_illumination(image, rng, mode="linear", min_gain=0.5)
        gray = result.mean(axis=2)
        assert float(gray.max() - gray.min()) > 50.0

    def test_apply_illumination_all_modes_run_without_error(
        self, rng: np.random.Generator
    ) -> None:
        image = np.full((64, 64, 3), 220, dtype=np.uint8)
        for mode in _MODES:
            result = apply_illumination(image, rng, mode=mode)
            assert result.shape == image.shape

    def test_apply_illumination_deterministic_with_same_seed(self) -> None:
        image = np.full((64, 64, 3), 220, dtype=np.uint8)
        a = apply_illumination(image, np.random.default_rng(7))
        b = apply_illumination(image, np.random.default_rng(7))
        assert np.array_equal(a, b)

    def test_apply_illumination_raises_on_grayscale_input(
        self, rng: np.random.Generator
    ) -> None:
        gray = np.full((64, 64), 220, dtype=np.uint8)
        with pytest.raises(ValueError):
            apply_illumination(gray, rng)

    def test_apply_illumination_raises_on_invalid_mode(
        self, rng: np.random.Generator
    ) -> None:
        image = np.full((64, 64, 3), 220, dtype=np.uint8)
        with pytest.raises(ValueError):
            apply_illumination(image, rng, mode="disco")

    def test_apply_illumination_raises_on_min_gain_out_of_range(
        self, rng: np.random.Generator
    ) -> None:
        image = np.full((64, 64, 3), 220, dtype=np.uint8)
        with pytest.raises(ValueError):
            apply_illumination(image, rng, min_gain=0.0)
        with pytest.raises(ValueError):
            apply_illumination(image, rng, min_gain=1.5)

    def test_apply_illumination_corner_mode_darkens_one_corner_most(
        self, rng: np.random.Generator
    ) -> None:
        """En modo corner, alguna esquina queda claramente más oscura que el
        centro (patrón de sombra de esquina)."""
        image = np.full((96, 96, 3), 220, dtype=np.uint8)
        result = apply_illumination(image, rng, mode="corner", min_gain=0.4)
        gray = result.mean(axis=2)
        corners = [
            float(gray[:12, :12].mean()),
            float(gray[:12, -12:].mean()),
            float(gray[-12:, :12].mean()),
            float(gray[-12:, -12:].mean()),
        ]
        center = float(gray[40:56, 40:56].mean())
        assert min(corners) < center - 20.0
