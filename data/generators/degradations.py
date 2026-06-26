"""
data/generators/degradations.py
================================
Añade degradaciones estructuradas sobre imágenes limpias para generar
las imágenes "sucias" (dirty) del dataset de entrenamiento de DocClean-Net.

Funciones públicas:
    add_blue_grid(image, rng)          — cuadrícula azul de cuaderno
    add_ruled_lines(image, rng)        — líneas horizontales de papel rayado
    add_watermark(image, rng, text)    — marca de agua diagonal semitransparente

Convenciones:
    - Todas las funciones reciben y retornan BGR uint8, shape (H, W, 3).
    - El array de entrada NO se modifica in-place; se retorna una copia.
    - rng: np.random.Generator siempre explícito, nunca estado global.
"""

from __future__ import annotations

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# add_blue_grid
# ─────────────────────────────────────────────────────────────────────────────

def add_blue_grid(
    image: np.ndarray,
    rng: np.random.Generator,
    spacing: int | None = None,
    thickness: float | None = None,
    angle_deg: float | None = None,
    opacity: float | None = None,
) -> np.ndarray:
    """Superpone una cuadrícula azul al estilo de cuaderno de colegio español.

    Spacing calibrado para cuadernos de 5mm estándar escaneados a ~200-300dpi
    (rango [28, 45]px). El blend es selectivo: solo mezcla en los píxeles de
    línea, preservando el negro de los trazos del dibujo.

    Parameters
    ----------
    image : np.ndarray
        Imagen BGR, shape (H, W, 3), dtype uint8.
    rng : np.random.Generator
        Generador de aleatoriedad.
    spacing : int | None
        Separación entre líneas en píxeles. Si es None, se muestrea en [28, 45].
    thickness : float | None
        Grosor de línea en píxeles. Si es None, se muestrea en [0.5, 2.0].
    angle_deg : float | None
        Ángulo de rotación en grados. Si es None, se muestrea en [-3.0, 3.0].
    opacity : float | None
        Opacidad en [0.0, 1.0]. Si es None, se muestrea en [0.2, 0.7].

    Returns
    -------
    np.ndarray
        Imagen BGR, shape (H, W, 3), dtype uint8.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image debe ser BGR (H, W, 3), recibido shape={image.shape}")

    h, w = image.shape[:2]

    if spacing is None:
        spacing = int(rng.integers(28, 46))
    if thickness is None:
        thickness = float(rng.uniform(0.5, 2.0))
    if angle_deg is None:
        angle_deg = float(rng.uniform(-3.0, 3.0))
    if opacity is None:
        opacity = float(rng.uniform(0.2, 0.7))

    # Color azul de cuaderno: B alto, G medio, R bajo
    b_val = int(rng.integers(160, 210))
    g_val = int(rng.integers(120, 170))
    r_val = int(rng.integers(80,  130))
    grid_color = (b_val, g_val, r_val)

    grid_layer = _render_grid(h, w, spacing, thickness, angle_deg, grid_color)

    return _blend_lines_only(image, grid_layer, opacity)


def _render_grid(
    h: int,
    w: int,
    spacing: int,
    thickness: float,
    angle_deg: float,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Renderiza una cuadrícula rotada sobre fondo blanco.

    Dibuja sobre un canvas ampliado para evitar bordes vacíos al rotar,
    luego recorta al tamaño original.

    Returns
    -------
    np.ndarray
        Imagen BGR, shape (H, W, 3), dtype uint8. Fondo blanco, líneas de color.
    """
    diag = int(np.ceil(np.hypot(h, w)))
    pad  = diag
    ch, cw = h + pad * 2, w + pad * 2

    canvas = np.full((ch, cw, 3), 255, dtype=np.uint8)
    thick_px = max(1, int(round(thickness)))

    y = spacing
    while y < ch:
        cv2.line(canvas, (0, y), (cw, y), color, thick_px, lineType=cv2.LINE_AA)
        y += spacing

    x = spacing
    while x < cw:
        cv2.line(canvas, (x, 0), (x, ch), color, thick_px, lineType=cv2.LINE_AA)
        x += spacing

    cx, cy = cw // 2, ch // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        canvas, M, (cw, ch),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    y0 = (ch - h) // 2
    x0 = (cw - w) // 2
    return rotated[y0:y0 + h, x0:x0 + w]


# ─────────────────────────────────────────────────────────────────────────────
# add_ruled_lines
# ─────────────────────────────────────────────────────────────────────────────

def add_ruled_lines(
    image: np.ndarray,
    rng: np.random.Generator,
    spacing: int | None = None,
    thickness: int | None = None,
    opacity: float | None = None,
    color_bgr: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Superpone líneas horizontales de papel rayado.

    Parameters
    ----------
    image : np.ndarray
        Imagen BGR, shape (H, W, 3), dtype uint8.
    rng : np.random.Generator
        Generador de aleatoriedad.
    spacing : int | None
        Separación en píxeles. Si es None, se muestrea en [18, 40].
    thickness : int | None
        Grosor en píxeles. Si es None, se muestrea en [1, 2].
    opacity : float | None
        Opacidad en [0.0, 1.0]. Si es None, se muestrea en [0.15, 0.60].
    color_bgr : tuple[int, int, int] | None
        Color BGR. Si es None, se genera un azul de cuaderno.

    Returns
    -------
    np.ndarray
        Imagen BGR, shape (H, W, 3), dtype uint8.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image debe ser BGR (H, W, 3), recibido shape={image.shape}")

    h, w = image.shape[:2]

    if spacing is None:
        spacing = int(rng.integers(18, 41))
    if thickness is None:
        thickness = int(rng.integers(1, 3))
    if opacity is None:
        opacity = float(rng.uniform(0.15, 0.60))
    if color_bgr is None:
        b_val = int(rng.integers(150, 210))
        g_val = int(rng.integers(110, 165))
        r_val = int(rng.integers(70,  120))
        color_bgr = (b_val, g_val, r_val)

    layer = np.full((h, w, 3), 255, dtype=np.uint8)
    y = spacing
    while y < h:
        cv2.line(layer, (0, y), (w, y), color_bgr, thickness, lineType=cv2.LINE_AA)
        y += spacing

    return _blend_lines_only(layer_img=layer, base=image, opacity=opacity)


# ─────────────────────────────────────────────────────────────────────────────
# add_watermark
# ─────────────────────────────────────────────────────────────────────────────

def add_watermark(
    image: np.ndarray,
    rng: np.random.Generator,
    text: str,
    opacity: float | None = None,
    angle_deg: float | None = None,
    font_scale: float | None = None,
) -> np.ndarray:
    """Superpone una marca de agua de texto diagonal semitransparente.

    Parameters
    ----------
    image : np.ndarray
        Imagen BGR, shape (H, W, 3), dtype uint8.
    rng : np.random.Generator
        Generador de aleatoriedad.
    text : str
        Texto de la marca de agua.
    opacity : float | None
        Opacidad en [0.0, 1.0]. Si es None, se muestrea en [0.08, 0.30].
    angle_deg : float | None
        Ángulo en grados. Si es None, se muestrea en [30.0, 50.0].
    font_scale : float | None
        Escala de fuente OpenCV. Si es None, se muestrea en [1.5, 3.0].

    Returns
    -------
    np.ndarray
        Imagen BGR, shape (H, W, 3), dtype uint8.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image debe ser BGR (H, W, 3), recibido shape={image.shape}")

    h, w = image.shape[:2]

    if opacity is None:
        opacity = float(rng.uniform(0.08, 0.30))
    if angle_deg is None:
        angle_deg = float(rng.uniform(30.0, 50.0))
    if font_scale is None:
        font_scale = float(rng.uniform(1.5, 3.0))

    shade = int(rng.integers(80, 160))
    color_bgr = (shade, shade, shade)

    layer = _render_watermark_tile(h, w, text, angle_deg, font_scale, color_bgr)

    return _blend_lines_only(image, layer, opacity)


def _render_watermark_tile(
    h: int,
    w: int,
    text: str,
    angle_deg: float,
    font_scale: float,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    """Renderiza el texto en mosaico rotado sobre fondo blanco.

    Returns
    -------
    np.ndarray
        Imagen BGR, shape (H, W, 3), dtype uint8, fondo blanco.
    """
    font      = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(font_scale * 1.5))

    (tw, t_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_h = t_h + baseline

    pad       = max(tw, text_h) + 20
    tile_size = tw + pad * 2
    tile      = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)

    tx = pad
    ty = tile_size // 2 + text_h // 2
    cv2.putText(tile, text, (tx, ty), font, font_scale, color_bgr, thickness,
                lineType=cv2.LINE_AA)

    M = cv2.getRotationMatrix2D((tile_size // 2, tile_size // 2), angle_deg, 1.0)
    rotated_tile = cv2.warpAffine(
        tile, M, (tile_size, tile_size),
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255),
    )

    # Reducir si el tile es mayor que el destino para garantizar tesela visible
    if tile_size > min(h, w):
        scale    = min(h, w) / tile_size * 0.5
        new_size = max(16, int(tile_size * scale))
        rotated_tile = cv2.resize(rotated_tile, (new_size, new_size),
                                  interpolation=cv2.INTER_LINEAR)
        tile_size = new_size

    reps_y = h // tile_size + 2
    reps_x = w // tile_size + 2
    tiled  = np.tile(rotated_tile, (reps_y, reps_x, 1))

    return tiled[:h, :w]


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad compartida
# ─────────────────────────────────────────────────────────────────────────────

def _blend_lines_only(
    base: np.ndarray,
    layer_img: np.ndarray,
    opacity: float,
) -> np.ndarray:
    """Mezcla `layer_img` sobre `base` solo donde hay línea (no en fondo blanco).

    Detecta los píxeles de línea como aquellos donde algún canal de `layer_img`
    es < 250. El fondo blanco de la capa no afecta a los trazos negros del dibujo,
    evitando que los trazos se vuelvan grises con opacidades altas.

    Parameters
    ----------
    base : np.ndarray
        Imagen de fondo BGR, shape (H, W, 3), dtype uint8.
    layer_img : np.ndarray
        Capa BGR, shape (H, W, 3), dtype uint8, con fondo blanco y líneas de color.
    opacity : float
        Peso de la capa en [0.0, 1.0] solo en los píxeles de línea.

    Returns
    -------
    np.ndarray
        Imagen mezclada BGR, shape (H, W, 3), dtype uint8.
    """
    base_f  = base.astype(np.float32)
    layer_f = layer_img.astype(np.float32)

    # Máscara: píxeles con línea = algún canal < 250
    line_mask = (layer_img.min(axis=2) < 250).astype(np.float32)[:, :, np.newaxis]

    result = base_f * (1.0 - line_mask * opacity/2) + layer_f * (line_mask * opacity/2)
    return np.clip(result, 0.0, 255.0).astype(np.uint8)
