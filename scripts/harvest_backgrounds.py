"""
scripts/harvest_backgrounds.py
==============================
Cosecha tiles cuadrados de fondo SIN TINTA a partir de páginas de libreta
escaneadas, para la composición de fondos reales del dataset de Phase 5
(``python -m data.generate_dataset --real-bg-dir ...``).

Criterio de aceptación: los píxeles oscuros de un fondo válido pertenecen
casi exclusivamente al raster periódico de la cuadrícula (filas/columnas que
cruzan el tile completo). Cualquier mancha compacta fuera del raster (tinta,
taladro de anillas, borde de escáner) invalida el tile. Funciona sobre
páginas dibujadas (rendimiento bajo, ~1-2% de posiciones) y mucho mejor
sobre páginas en blanco.

El script emite una HOJA DE CONTACTO (contact_sheet.jpg) en el directorio de
salida: verifícala visualmente y borra a mano cualquier tile contaminado
antes de usar el directorio en el generador.

Uso (PowerShell)::

    python -m scripts.harvest_backgrounds `
        --input ruta\\a\\escaneos `
        --output data\\real_backgrounds `
        --tile 256 --stride 64
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from inference.io_utils import _imread, _imwrite

_DEFAULT_TILE = 256
_DEFAULT_STRIDE = 64
_DEFAULT_MARGIN = 120  # borde exterior descartado (marco de escáner, anillas)
_MIN_KEEP_DISTANCE = 128  # separación mínima entre tiles aceptados (dedup)

# Umbrales del filtro (calibrados sobre las 16 páginas reales de Phase 5)
_PAPER_MIN = 210  # papel más oscuro = sombra fuerte o marco -> descartar
_DARK_OFFSET = 60  # oscuro = gris < papel - offset
_GRID_LINE_FRAC = 0.45  # una línea de grid cruza >=45% del tile
_GRID_BAND_DILATION = 3  # +-px alrededor de una fila/columna de línea fuerte:
#                          cubre los bordes anti-aliased de líneas gruesas de
#                          alto DPI (~7 px) sin permitir que trazos densos
#                          fabriquen su propia banda (solo excusan las líneas
#                          que cruzan >=45% del tile)
_MIN_GRID_LINES = 3  # cuadrícula real: varias líneas por eje en el tile
_MAX_RESIDUAL_PX = 60  # píxeles oscuros fuera del raster tolerados (ruido)
_MAX_RESIDUAL_BLOB = 14  # máx. oscuro fuera de raster en ventana 16x16:
#                          un trazo es compacto; el ruido está disperso
_LINE_INTENSITY_MIN = 68  # líneas más oscuras = boli/rotulador, no grid


def tile_is_clean_background(
    tile_bgr: np.ndarray,
    dark_offset: int = _DARK_OFFSET,
) -> bool:
    """Decide si un tile es solo papel+cuadrícula, sin tinta ni artefactos.

    Parameters
    ----------
    tile_bgr : np.ndarray
        Tile BGR, shape (T, T, 3), dtype uint8.
    dark_offset : int
        Umbral de oscuridad: oscuro = gris < papel - dark_offset. El default
        (60) es seguro sobre páginas dibujadas; en páginas EN BLANCO
        conocidas puede bajarse (p. ej. 40) para capturar cuadrículas de
        bajo contraste sin riesgo, porque el filtro solo tiene que vetar
        artefactos (taladros, bordes, sombras), no tinta.

    Returns
    -------
    bool
        True si el tile puede usarse como fondo real de entrenamiento.
    """
    g = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    paper = float(np.percentile(g, 85))
    if paper < _PAPER_MIN:
        return False

    dark = g < paper - dark_offset

    row_frac = dark.mean(axis=1)
    col_frac = dark.mean(axis=0)
    grid_rows = row_frac > _GRID_LINE_FRAC
    grid_cols = col_frac > _GRID_LINE_FRAC
    if grid_rows.sum() < _MIN_GRID_LINES or grid_cols.sum() < _MIN_GRID_LINES:
        return False

    # Banda del raster: solo filas/columnas de línea FUERTE, dilatadas
    # +-_GRID_BAND_DILATION px. La dilatación cubre los bordes anti-aliased
    # de líneas gruesas (alto DPI) y la rotación leve; exigir línea fuerte
    # impide que trazos densos se auto-excusen fabricando banda.
    kernel = [1] * (2 * _GRID_BAND_DILATION + 1)
    rows_band = np.convolve(grid_rows.astype(int), kernel, mode="same") > 0
    cols_band = np.convolve(grid_cols.astype(int), kernel, mode="same") > 0
    raster = rows_band[:, None] | cols_band[None, :]

    residual = dark & ~raster
    if int(residual.sum()) > _MAX_RESIDUAL_PX:
        return False

    # Test de concentración: un fragmento de trazo es un blob compacto;
    # el ruido de escáner está disperso. Suma de residual en ventanas 16x16.
    if residual.any():
        res_f = residual.astype(np.float32)
        window = cv2.boxFilter(res_f, ddepth=-1, ksize=(16, 16), normalize=False)
        if float(window.max()) > _MAX_RESIDUAL_BLOB:
            return False

    # Intensidad de línea coherente con cuadrícula impresa/azul, no con boli
    line_pixels = g[dark & raster]
    if line_pixels.size and float(np.median(line_pixels)) < _LINE_INTENSITY_MIN:
        return False

    return True


def harvest_page(
    page_bgr: np.ndarray,
    tile: int,
    stride: int,
    margin: int,
    dark_offset: int = _DARK_OFFSET,
) -> list[tuple[int, int, np.ndarray]]:
    """Extrae los tiles de fondo limpio de una página escaneada.

    Parameters
    ----------
    page_bgr : np.ndarray
        Página BGR, shape (H, W, 3), dtype uint8.
    tile : int
        Lado del tile en píxeles.
    stride : int
        Paso del barrido en píxeles.
    margin : int
        Margen exterior descartado en píxeles.
    dark_offset : int
        Ver tile_is_clean_background.

    Returns
    -------
    list[tuple[int, int, np.ndarray]]
        Lista de (y, x, tile_bgr) aceptados, deduplicados a una separación
        mínima de _MIN_KEEP_DISTANCE píxeles.
    """
    h, w = page_bgr.shape[:2]
    kept: list[tuple[int, int, np.ndarray]] = []
    for y in range(margin, h - tile - margin, stride):
        for x in range(margin, w - tile - margin, stride):
            candidate = page_bgr[y : y + tile, x : x + tile]
            if not tile_is_clean_background(candidate, dark_offset=dark_offset):
                continue
            if all(
                abs(y - ky) >= _MIN_KEEP_DISTANCE or abs(x - kx) >= _MIN_KEEP_DISTANCE
                for ky, kx, _ in kept
            ):
                kept.append((y, x, candidate))
    return kept


def _write_contact_sheet(tiles: list[np.ndarray], out_path: Path) -> None:
    """Escribe una hoja de contacto para verificación visual manual.

    Parameters
    ----------
    tiles : list[np.ndarray]
        Tiles BGR aceptados (todos del mismo lado).
    out_path : Path
        Ruta del PNG de salida.
    """
    if not tiles:
        return
    cols = 6
    side = tiles[0].shape[0]
    blank = np.full((side, side, 3), 255, dtype=np.uint8)
    padded = tiles + [blank] * ((-len(tiles)) % cols)
    rows = [np.hstack(padded[i : i + cols]) for i in range(0, len(padded), cols)]
    sheet = np.vstack(rows)
    scale = min(1.0, 1536.0 / sheet.shape[1])
    if scale < 1.0:
        sheet = cv2.resize(
            sheet, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    _imwrite(out_path, sheet)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cosecha tiles de fondo sin tinta de páginas escaneadas."
    )
    parser.add_argument("--input", type=str, required=True, help="Dir con escaneos")
    parser.add_argument("--output", type=str, required=True, help="Dir de tiles")
    parser.add_argument("--tile", type=int, default=_DEFAULT_TILE)
    parser.add_argument("--stride", type=int, default=_DEFAULT_STRIDE)
    parser.add_argument("--margin", type=int, default=_DEFAULT_MARGIN)
    parser.add_argument(
        "--dark-offset",
        type=int,
        default=_DARK_OFFSET,
        help=(
            "Umbral de oscuridad (papel - offset). 60 seguro en páginas "
            "dibujadas; 40 para páginas EN BLANCO de bajo contraste."
        ),
    )
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    if not in_dir.is_dir():
        print(f"[ERROR] --input no existe: {in_dir}", file=sys.stderr)
        sys.exit(1)
    if args.tile < 64 or args.stride < 1:
        print("[ERROR] --tile >= 64 y --stride >= 1", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    pages = sorted(
        q for q in in_dir.iterdir() if q.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not pages:
        print(f"[ERROR] sin imágenes en {in_dir}", file=sys.stderr)
        sys.exit(1)

    all_tiles: list[np.ndarray] = []
    for page_path in pages:
        page = _imread(page_path)
        if page is None:
            print(f"[WARN] ilegible, se salta: {page_path.name}", file=sys.stderr)
            continue
        kept = harvest_page(
            page, args.tile, args.stride, args.margin, dark_offset=args.dark_offset
        )
        for y, x, t in kept:
            name = f"bg_{page_path.stem}_y{y:05d}_x{x:05d}.png"
            _imwrite(out_dir / name, t)
            all_tiles.append(t)
        print(f"{page_path.name}: {len(kept)} tiles")

    _write_contact_sheet(all_tiles, out_dir / "contact_sheet.jpg")
    print(f"\nTotal: {len(all_tiles)} tiles en {out_dir}")
    print("Verifica contact_sheet.jpg y borra a mano cualquier tile contaminado.")
    print("(la hoja es .jpg a propósito: el generador solo carga .png)")


if __name__ == "__main__":
    main()
