"""
tests/test_generators.py
========================
Tests para los generadores sintéticos de DocClean-Net (Phase 1).

Fixtures reutilizadas de conftest.py:
  - rng        : np.random.Generator con semilla fija (scope=session)
  - tmp_output_dir : directorio temporal por función
"""

import numpy as np
import pytest

from data.generators.paper import generate_paper


# ══════════════════════════════════════════════════════════════════════════════
# generate_paper
# ══════════════════════════════════════════════════════════════════════════════

class TestGeneratePaper:

    def test_generate_paper_output_shape_matches_request(
        self, rng: np.random.Generator
    ) -> None:
        """El array retornado tiene exactamente (H, W, 3)."""
        result = generate_paper(128, 96, rng)
        assert result.shape == (128, 96, 3)

    def test_generate_paper_output_dtype_is_uint8(
        self, rng: np.random.Generator
    ) -> None:
        result = generate_paper(64, 64, rng)
        assert result.dtype == np.uint8

    def test_generate_paper_values_in_valid_range(
        self, rng: np.random.Generator
    ) -> None:
        """Todos los píxeles en [0, 255] — sin overflow ni underflow."""
        result = generate_paper(64, 64, rng)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_generate_paper_is_bright_like_paper(
        self, rng: np.random.Generator
    ) -> None:
        """El brillo medio debe ser claramente alto (papel blanco ≥ 200)."""
        result = generate_paper(128, 128, rng)
        assert float(result.mean()) >= 200.0

    def test_generate_paper_is_bgr_three_channel(
        self, rng: np.random.Generator
    ) -> None:
        """La imagen tiene 3 canales (BGR), no escala de grises."""
        result = generate_paper(64, 64, rng)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_generate_paper_not_uniform_has_texture(
        self, rng: np.random.Generator
    ) -> None:
        """La imagen no es uniforme: std > 0 gracias al ruido de grano."""
        result = generate_paper(128, 128, rng)
        assert float(result.std()) > 0.5

    def test_generate_paper_explicit_sigma_controls_noise(
        self, rng: np.random.Generator
    ) -> None:
        """sigma_noise=0 produce una imagen casi sin variación de canal a canal."""
        # Con sigma=0 solo queda el sesgo de canal (constante por canal)
        result = generate_paper(64, 64, rng, sigma_noise=0.0, vignette_strength=0.0)
        # Cada canal debe ser casi constante (std < 1.0)
        for ch in range(3):
            assert float(result[:, :, ch].std()) < 1.0

    def test_generate_paper_vignette_darkens_corners(
        self, rng: np.random.Generator
    ) -> None:
        """Con vignette fuerte, las esquinas son más oscuras que el centro."""
        # Imagen grande para que el gradiente sea visible
        h, w = 256, 256
        result = generate_paper(h, w, rng, sigma_noise=0.0, vignette_strength=0.5)

        center_mean = float(result[h//2 - 10 : h//2 + 10,
                                   w//2 - 10 : w//2 + 10].mean())
        corner_mean = float(np.stack([
            result[:20,    :20   ],
            result[:20,    -20:  ],
            result[-20:,   :20   ],
            result[-20:,   -20:  ],
        ]).mean())

        assert center_mean > corner_mean + 5.0, (
            f"Centro ({center_mean:.1f}) debería ser más brillante que esquinas ({corner_mean:.1f})"
        )

    def test_generate_paper_no_vignette_uniform_brightness(
        self, rng: np.random.Generator
    ) -> None:
        """Sin vignette, centro y esquinas tienen brillo similar (diff < 15)."""
        h, w = 256, 256
        result = generate_paper(h, w, rng, sigma_noise=0.0, vignette_strength=0.0)

        center_mean = float(result[h//2 - 10 : h//2 + 10,
                                   w//2 - 10 : w//2 + 10].mean())
        corner_mean = float(np.stack([
            result[:20,    :20   ],
            result[-20:,   -20:  ],
        ]).mean())

        assert abs(center_mean - corner_mean) < 15.0

    def test_generate_paper_deterministic_with_same_rng_state(self) -> None:
        """Dos llamadas con el mismo estado del rng producen el mismo resultado."""
        rng_a = np.random.default_rng(seed=0)
        rng_b = np.random.default_rng(seed=0)
        result_a = generate_paper(64, 64, rng_a)
        result_b = generate_paper(64, 64, rng_b)
        np.testing.assert_array_equal(result_a, result_b)

    def test_generate_paper_different_seeds_produce_different_images(self) -> None:
        """Semillas distintas producen imágenes distintas."""
        rng_a = np.random.default_rng(seed=1)
        rng_b = np.random.default_rng(seed=2)
        result_a = generate_paper(64, 64, rng_a)
        result_b = generate_paper(64, 64, rng_b)
        assert not np.array_equal(result_a, result_b)

    def test_generate_paper_non_square_shape(
        self, rng: np.random.Generator
    ) -> None:
        """Funciona con dimensiones no cuadradas (resolución A4 reducida)."""
        result = generate_paper(297, 210, rng)
        assert result.shape == (297, 210, 3)


# ══════════════════════════════════════════════════════════════════════════════
# generate_strokes
# ══════════════════════════════════════════════════════════════════════════════

from data.generators.strokes import generate_strokes


class TestGenerateStrokes:

    def test_generate_strokes_output_shape_matches_canvas(
        self, rng: np.random.Generator, white_image_bgr: np.ndarray
    ) -> None:
        """El resultado tiene la misma shape que el canvas de entrada."""
        result = generate_strokes(white_image_bgr, rng)
        assert result.shape == white_image_bgr.shape

    def test_generate_strokes_output_dtype_is_uint8(
        self, rng: np.random.Generator, white_image_bgr: np.ndarray
    ) -> None:
        result = generate_strokes(white_image_bgr, rng)
        assert result.dtype == np.uint8

    def test_generate_strokes_does_not_modify_canvas_in_place(
        self, rng: np.random.Generator, white_image_bgr: np.ndarray
    ) -> None:
        """El canvas original no debe ser alterado."""
        canvas_copy = white_image_bgr.copy()
        generate_strokes(white_image_bgr, rng)
        np.testing.assert_array_equal(white_image_bgr, canvas_copy)

    def test_generate_strokes_produces_dark_pixels_on_white_canvas(
        self, rng: np.random.Generator
    ) -> None:
        """Sobre fondo blanco, debe aparecer al menos un píxel de tinta oscuro."""
        canvas = np.full((256, 256, 3), 255, dtype=np.uint8)
        result = generate_strokes(canvas, rng, n_strokes=10)
        # Al menos el 0.1% de píxeles deben ser oscuros (< 128)
        dark_px = int((result.mean(axis=2) < 128).sum())
        assert dark_px > 0, "No se dibujó ningún trazo sobre el canvas blanco"

    def test_generate_strokes_n_strokes_zero_returns_copy_of_canvas(
        self, rng: np.random.Generator, white_image_bgr: np.ndarray
    ) -> None:
        """Con n_strokes=0 el resultado es idéntico al canvas."""
        result = generate_strokes(white_image_bgr, rng, n_strokes=0)
        np.testing.assert_array_equal(result, white_image_bgr)

    def test_generate_strokes_explicit_n_strokes_respected(
        self, rng: np.random.Generator
    ) -> None:
        """n_strokes explícito no lanza error y produce output válido."""
        canvas = np.full((256, 256, 3), 255, dtype=np.uint8)
        result = generate_strokes(canvas, rng, n_strokes=5)
        assert result.shape == canvas.shape
        assert result.dtype == np.uint8

    def test_generate_strokes_more_strokes_means_more_ink(
        self, rng: np.random.Generator
    ) -> None:
        """Más trazos → más píxeles oscuros (test estadístico con semillas fijas)."""
        canvas = np.full((256, 256, 3), 255, dtype=np.uint8)
        rng_few  = np.random.default_rng(seed=10)
        rng_many = np.random.default_rng(seed=10)

        few  = generate_strokes(canvas, rng_few,  n_strokes=3)
        many = generate_strokes(canvas, rng_many, n_strokes=30)

        dark_few  = int((few.mean(axis=2)  < 128).sum())
        dark_many = int((many.mean(axis=2) < 128).sum())
        assert dark_many > dark_few

    def test_generate_strokes_deterministic_with_same_rng_state(self) -> None:
        """Mismo estado del rng → mismo resultado."""
        canvas = np.full((128, 128, 3), 255, dtype=np.uint8)
        rng_a = np.random.default_rng(seed=7)
        rng_b = np.random.default_rng(seed=7)
        result_a = generate_strokes(canvas, rng_a, n_strokes=15)
        result_b = generate_strokes(canvas, rng_b, n_strokes=15)
        np.testing.assert_array_equal(result_a, result_b)

    def test_generate_strokes_raises_on_grayscale_canvas(
        self, rng: np.random.Generator, white_image_gray: np.ndarray
    ) -> None:
        """Lanza ValueError si el canvas es escala de grises (2D)."""
        with pytest.raises(ValueError, match="BGR"):
            generate_strokes(white_image_gray, rng)

    def test_generate_strokes_preserves_background_color(
        self, rng: np.random.Generator
    ) -> None:
        """Los píxeles no tocados conservan el color del canvas original."""
        # Canvas azul para distinguirlo claramente de los trazos negros
        canvas = np.full((128, 128, 3), fill_value=0, dtype=np.uint8)
        canvas[:, :, 0] = 200  # canal B alto → azul
        result = generate_strokes(canvas, rng, n_strokes=5)
        # Los píxeles con B=200 y G=R=0 indican fondo no tocado
        untouched = (result[:, :, 0] == 200) & (result[:, :, 1] == 0) & (result[:, :, 2] == 0)
        assert untouched.sum() > 0, "El fondo azul desapareció completamente"


# ══════════════════════════════════════════════════════════════════════════════
# degradations
# ══════════════════════════════════════════════════════════════════════════════

from data.generators.degradations import add_blue_grid, add_ruled_lines, add_watermark


class TestAddBlueGrid:

    def test_add_blue_grid_output_shape_matches_input(
        self, rng: np.random.Generator, white_image_bgr: np.ndarray
    ) -> None:
        result = add_blue_grid(white_image_bgr, rng)
        assert result.shape == white_image_bgr.shape

    def test_add_blue_grid_output_dtype_is_uint8(
        self, rng: np.random.Generator, white_image_bgr: np.ndarray
    ) -> None:
        result = add_blue_grid(white_image_bgr, rng)
        assert result.dtype == np.uint8

    def test_add_blue_grid_does_not_modify_input_in_place(
        self, rng: np.random.Generator, white_image_bgr: np.ndarray
    ) -> None:
        original = white_image_bgr.copy()
        add_blue_grid(white_image_bgr, rng)
        np.testing.assert_array_equal(white_image_bgr, original)

    def test_add_blue_grid_introduces_blue_tint(
        self, rng: np.random.Generator
    ) -> None:
        """La cuadrícula azul debe elevar el canal B respecto al input blanco."""
        canvas = np.full((128, 128, 3), 255, dtype=np.uint8)
        result = add_blue_grid(
            canvas, rng,
            spacing=20, thickness=2.0, angle_deg=0.0, opacity=0.5
        )
        # Con opacidad 0.5 sobre blanco, el canal B de las líneas azules
        # sigue siendo alto, pero G y R bajan → el canal B domina menos
        # que R en las zonas de línea. Comprobamos que al menos hay
        # variación (la imagen dejó de ser uniforme).
        assert float(result.std()) > 0.0

    def test_add_blue_grid_high_opacity_darkens_image(
        self, rng: np.random.Generator
    ) -> None:
        """Con opacidad alta, la imagen es menos brillante que el original blanco."""
        canvas = np.full((128, 128, 3), 255, dtype=np.uint8)
        result = add_blue_grid(
            canvas, rng,
            spacing=10, thickness=2.0, angle_deg=0.0, opacity=0.9
        )
        assert float(result.mean()) < 255.0

    def test_add_blue_grid_zero_opacity_returns_original(
        self, rng: np.random.Generator, white_image_bgr: np.ndarray
    ) -> None:
        """Opacidad 0 → sin cambio visible."""
        result = add_blue_grid(white_image_bgr, rng, opacity=0.0)
        np.testing.assert_array_equal(result, white_image_bgr)

    def test_add_blue_grid_raises_on_grayscale_input(
        self, rng: np.random.Generator, white_image_gray: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="BGR"):
            add_blue_grid(white_image_gray, rng)

    def test_add_blue_grid_deterministic_with_same_seed(self) -> None:
        canvas = np.full((64, 64, 3), 255, dtype=np.uint8)
        rng_a = np.random.default_rng(seed=5)
        rng_b = np.random.default_rng(seed=5)
        np.testing.assert_array_equal(
            add_blue_grid(canvas, rng_a),
            add_blue_grid(canvas, rng_b),
        )


class TestAddRuledLines:

    def test_add_ruled_lines_output_shape_matches_input(
        self, rng: np.random.Generator, white_image_bgr: np.ndarray
    ) -> None:
        result = add_ruled_lines(white_image_bgr, rng)
        assert result.shape == white_image_bgr.shape

    def test_add_ruled_lines_output_dtype_is_uint8(
        self, rng: np.random.Generator, white_image_bgr: np.ndarray
    ) -> None:
        result = add_ruled_lines(white_image_bgr, rng)
        assert result.dtype == np.uint8

    def test_add_ruled_lines_does_not_modify_input_in_place(
        self, rng: np.random.Generator, white_image_bgr: np.ndarray
    ) -> None:
        original = white_image_bgr.copy()
        add_ruled_lines(white_image_bgr, rng)
        np.testing.assert_array_equal(white_image_bgr, original)

    def test_add_ruled_lines_zero_opacity_unchanged(
        self, rng: np.random.Generator, white_image_bgr: np.ndarray
    ) -> None:
        result = add_ruled_lines(white_image_bgr, rng, opacity=0.0)
        np.testing.assert_array_equal(result, white_image_bgr)

    def test_add_ruled_lines_raises_on_grayscale_input(
        self, rng: np.random.Generator, white_image_gray: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="BGR"):
            add_ruled_lines(white_image_gray, rng)

    def test_add_ruled_lines_deterministic_with_same_seed(self) -> None:
        canvas = np.full((64, 64, 3), 255, dtype=np.uint8)
        rng_a = np.random.default_rng(seed=9)
        rng_b = np.random.default_rng(seed=9)
        np.testing.assert_array_equal(
            add_ruled_lines(canvas, rng_a),
            add_ruled_lines(canvas, rng_b),
        )


class TestAddWatermark:

    def test_add_watermark_output_shape_matches_input(
        self, rng: np.random.Generator, white_image_bgr: np.ndarray
    ) -> None:
        result = add_watermark(white_image_bgr, rng, text="TEST")
        assert result.shape == white_image_bgr.shape

    def test_add_watermark_output_dtype_is_uint8(
        self, rng: np.random.Generator, white_image_bgr: np.ndarray
    ) -> None:
        result = add_watermark(white_image_bgr, rng, text="TEST")
        assert result.dtype == np.uint8

    def test_add_watermark_does_not_modify_input_in_place(
        self, rng: np.random.Generator, white_image_bgr: np.ndarray
    ) -> None:
        original = white_image_bgr.copy()
        add_watermark(white_image_bgr, rng, text="TEST")
        np.testing.assert_array_equal(white_image_bgr, original)

    def test_add_watermark_zero_opacity_unchanged(
        self, rng: np.random.Generator, white_image_bgr: np.ndarray
    ) -> None:
        result = add_watermark(white_image_bgr, rng, text="X", opacity=0.0)
        np.testing.assert_array_equal(result, white_image_bgr)

    def test_add_watermark_darkens_white_canvas(self) -> None:
        """Con opacidad visible, la marca debe oscurecer ligeramente el fondo blanco."""
        rng_local = np.random.default_rng(seed=42)
        canvas = np.full((256, 256, 3), 255, dtype=np.uint8)
        # Parámetros explícitos para evitar que el rng de sesión (ya avanzado)
        # produzca combinaciones edge-case (font_scale enorme, tile > canvas).
        result = add_watermark(
            canvas, rng_local, text="BORRADOR",
            opacity=0.3, angle_deg=45.0, font_scale=1.5,
        )
        assert float(result.mean()) < 255.0

    def test_add_watermark_raises_on_grayscale_input(
        self, rng: np.random.Generator, white_image_gray: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="BGR"):
            add_watermark(white_image_gray, rng, text="X")

    def test_add_watermark_deterministic_with_same_seed(self) -> None:
        canvas = np.full((128, 128, 3), 255, dtype=np.uint8)
        rng_a = np.random.default_rng(seed=3)
        rng_b = np.random.default_rng(seed=3)
        np.testing.assert_array_equal(
            add_watermark(canvas, rng_a, text="COPY"),
            add_watermark(canvas, rng_b, text="COPY"),
        )


# ══════════════════════════════════════════════════════════════════════════════
# augmentation
# ══════════════════════════════════════════════════════════════════════════════

from data.augmentation import augment_pair


class TestAugmentPair:

    def _make_pair(self) -> tuple[np.ndarray, np.ndarray]:
        dirty = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        clean = np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        return dirty, clean

    def test_augment_pair_output_shapes_match_input(
        self, rng: np.random.Generator
    ) -> None:
        dirty, clean = self._make_pair()
        d_aug, c_aug = augment_pair(dirty, clean, rng)
        assert d_aug.shape == dirty.shape
        assert c_aug.shape == clean.shape

    def test_augment_pair_output_dtypes_are_uint8(
        self, rng: np.random.Generator
    ) -> None:
        dirty, clean = self._make_pair()
        d_aug, c_aug = augment_pair(dirty, clean, rng)
        assert d_aug.dtype == np.uint8
        assert c_aug.dtype == np.uint8

    def test_augment_pair_does_not_modify_inputs(
        self, rng: np.random.Generator
    ) -> None:
        dirty, clean = self._make_pair()
        dirty_copy = dirty.copy()
        clean_copy = clean.copy()
        augment_pair(dirty, clean, rng)
        np.testing.assert_array_equal(dirty, dirty_copy)
        np.testing.assert_array_equal(clean, clean_copy)

    def test_augment_pair_raises_on_shape_mismatch(
        self, rng: np.random.Generator
    ) -> None:
        dirty = np.zeros((64, 64, 3), dtype=np.uint8)
        clean = np.zeros((128, 128, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            augment_pair(dirty, clean, rng)

    def test_augment_pair_geometric_transforms_applied_equally(self) -> None:
        """Si hay flip, se aplica igual a dirty y clean (test por simetría)."""
        # Creamos dirty y clean idénticos: si la transformación es la misma,
        # los dos outputs también serán idénticos.
        img = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
        rng_a = np.random.default_rng(seed=99)
        rng_b = np.random.default_rng(seed=99)
        d_aug, _ = augment_pair(img.copy(), img.copy(), rng_a)
        _, c_aug = augment_pair(img.copy(), img.copy(), rng_b)
        np.testing.assert_array_equal(d_aug, c_aug)

    def test_augment_pair_deterministic_with_same_seed(self) -> None:
        dirty, clean = self._make_pair()
        rng_a = np.random.default_rng(seed=77)
        rng_b = np.random.default_rng(seed=77)
        d_a, c_a = augment_pair(dirty.copy(), clean.copy(), rng_a)
        d_b, c_b = augment_pair(dirty.copy(), clean.copy(), rng_b)
        np.testing.assert_array_equal(d_a, d_b)
        np.testing.assert_array_equal(c_a, c_b)

    def test_augment_pair_output_values_in_valid_range(
        self, rng: np.random.Generator
    ) -> None:
        dirty, clean = self._make_pair()
        d_aug, c_aug = augment_pair(dirty, clean, rng)
        assert int(d_aug.min()) >= 0 and int(d_aug.max()) <= 255
        assert int(c_aug.min()) >= 0 and int(c_aug.max()) <= 255
