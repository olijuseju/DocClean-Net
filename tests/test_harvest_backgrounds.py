"""
tests/test_harvest_backgrounds.py
=================================
Tests para scripts/harvest_backgrounds.py (Phase 5.1.2).

Fixtures reutilizadas de conftest.py:
  - rng : np.random.Generator con semilla fija (scope=session)
"""

import cv2
import numpy as np

from scripts.harvest_backgrounds import harvest_page, tile_is_clean_background


def _grid_tile(
    side: int = 256,
    spacing: int = 14,
    line_bgr: tuple[int, int, int] = (120, 100, 100),
    paper: int = 235,
) -> np.ndarray:
    """Tile sintético de papel+cuadrícula, sin tinta."""
    tile = np.full((side, side, 3), paper, dtype=np.uint8)
    for y in range(0, side, spacing):
        cv2.line(tile, (0, y), (side - 1, y), line_bgr, 1)
        cv2.line(tile, (y, 0), (y, side - 1), line_bgr, 1)
    return tile


class TestTileIsCleanBackground:

    def test_accepts_pure_grid_tile(self) -> None:
        assert tile_is_clean_background(_grid_tile()) is True

    def test_rejects_tile_with_ink_stroke(self) -> None:
        tile = _grid_tile()
        cv2.line(tile, (30, 40), (200, 180), (20, 20, 20), 3, cv2.LINE_AA)
        assert tile_is_clean_background(tile) is False

    def test_rejects_tile_with_small_ink_fragment(self) -> None:
        """El test de concentración detecta blobs compactos aunque el
        recuento total de píxeles residuales sea bajo. El fragmento se
        coloca en mitad de celda; un fragmento que solape una línea de
        grid es ambiguo por diseño y lo cubre el veto manual de la hoja
        de contacto."""
        tile = _grid_tile(spacing=14)
        cv2.circle(tile, (119, 119), 3, (30, 30, 30), thickness=-1)
        assert tile_is_clean_background(tile) is False

    def test_rejects_dark_scanner_border(self) -> None:
        tile = _grid_tile()
        tile[:, :90, :] = 25  # franja negra de marco de escáner
        assert tile_is_clean_background(tile) is False

    def test_rejects_plain_paper_without_grid(self) -> None:
        """Sin estructura de cuadrícula no es un fondo de libreta válido."""
        tile = np.full((256, 256, 3), 235, dtype=np.uint8)
        assert tile_is_clean_background(tile) is False

    def test_tolerates_scattered_scanner_noise(self, rng: np.random.Generator) -> None:
        tile = _grid_tile().astype(np.int16)
        noise = rng.integers(-5, 6, size=tile.shape, dtype=np.int16)
        tile = np.clip(tile + noise, 0, 255).astype(np.uint8)
        assert tile_is_clean_background(tile) is True


class TestHarvestPage:

    def test_harvest_page_finds_clean_region_and_skips_drawn_region(self) -> None:
        """Página sintética mitad limpia / mitad dibujada: los tiles
        aceptados provienen solo de la mitad limpia."""
        page = _grid_tile(side=1024, spacing=14)
        rng = np.random.default_rng(2)
        for _ in range(60):  # 'dibujo' denso en la mitad derecha
            p1 = (int(rng.integers(540, 1000)), int(rng.integers(20, 1000)))
            p2 = (int(rng.integers(540, 1000)), int(rng.integers(20, 1000)))
            cv2.line(page, p1, p2, (10, 10, 10), 2, cv2.LINE_AA)

        kept = harvest_page(page, tile=256, stride=64, margin=32)
        assert len(kept) >= 1
        for _, x, _ in kept:
            assert x + 256 <= 560  # ningún tile invade la mitad dibujada

    def test_harvest_page_dedupes_overlapping_positions(self) -> None:
        page = _grid_tile(side=768, spacing=14)
        kept = harvest_page(page, tile=256, stride=32, margin=32)
        for i, (y1, x1, _) in enumerate(kept):
            for y2, x2, _ in kept[i + 1 :]:
                assert abs(y1 - y2) >= 128 or abs(x1 - x2) >= 128
