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
    color_bgr: tuple[int, int, int] | None = None,
    opaque_lines: bool = False,
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
    color_bgr : tuple[int, int, int] | None
        Color BGR de las líneas. Si es None, azul de cuaderno muestreado
        (comportamiento histórico). Grises oscuros casi acromáticos, p. ej.
        (72, 60, 60), simulan papel milimetrado impreso — modo de fallo
        real medido en Phase 5 (grid gris ≈85, BGR (72, 57, 57)).
    opaque_lines : bool
        Si es False (default), blend histórico con peso efectivo opacity/2
        (máx. 0.35) — cuadrícula azul translúcida de cuaderno. Si es True,
        blend de solo-oscurecimiento con peso efectivo = opacity (hasta
        1.0): las líneas alcanzan su color puro sobre el papel, pero nunca
        aclaran tinta más oscura donde la cruzan — modelo físico de la
        cuadrícula impresa con la tinta dibujada por encima.

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

    if color_bgr is None:
        # Color azul de cuaderno: B alto, G medio, R bajo
        b_val = int(rng.integers(160, 210))
        g_val = int(rng.integers(120, 170))
        r_val = int(rng.integers(80, 130))
        grid_color = (b_val, g_val, r_val)
    else:
        grid_color = color_bgr

    grid_layer = _render_grid(h, w, spacing, thickness, angle_deg, grid_color)

    if opaque_lines:
        return _blend_lines_darken(image, grid_layer, opacity)
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
    pad = diag
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
        canvas,
        M,
        (cw, ch),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    y0 = (ch - h) // 2
    x0 = (cw - w) // 2
    return rotated[y0 : y0 + h, x0 : x0 + w]


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
        r_val = int(rng.integers(70, 120))
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(font_scale * 1.5))

    (tw, t_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_h = t_h + baseline

    pad = max(tw, text_h) + 20
    tile_size = tw + pad * 2
    tile = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)

    tx = pad
    ty = tile_size // 2 + text_h // 2
    cv2.putText(
        tile,
        text,
        (tx, ty),
        font,
        font_scale,
        color_bgr,
        thickness,
        lineType=cv2.LINE_AA,
    )

    M = cv2.getRotationMatrix2D((tile_size // 2, tile_size // 2), angle_deg, 1.0)
    rotated_tile = cv2.warpAffine(
        tile,
        M,
        (tile_size, tile_size),
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255),
    )

    # Reducir si el tile es mayor que el destino para garantizar tesela visible
    if tile_size > min(h, w):
        scale = min(h, w) / tile_size * 0.5
        new_size = max(16, int(tile_size * scale))
        rotated_tile = cv2.resize(
            rotated_tile, (new_size, new_size), interpolation=cv2.INTER_LINEAR
        )
        tile_size = new_size

    reps_y = h // tile_size + 2
    reps_x = w // tile_size + 2
    tiled = np.tile(rotated_tile, (reps_y, reps_x, 1))

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
    base_f = base.astype(np.float32)
    layer_f = layer_img.astype(np.float32)

    # Máscara: píxeles con línea = algún canal < 250
    line_mask = (layer_img.min(axis=2) < 250).astype(np.float32)[:, :, np.newaxis]

    result = base_f * (1.0 - line_mask * opacity / 2) + layer_f * (
        line_mask * opacity / 2
    )
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def _blend_lines_darken(
    base: np.ndarray,
    layer_img: np.ndarray,
    opacity: float,
) -> np.ndarray:
    """Mezcla `layer_img` sobre `base` con peso completo, solo oscureciendo.

    A diferencia de _blend_lines_only (peso efectivo opacity/2), aquí el peso
    efectivo es `opacity` sin atenuar, de modo que con opacity=1.0 las líneas
    alcanzan su color puro sobre el papel — necesario para cuadrículas
    impresas oscuras (papel milimetrado, gris ≈70-120). El resultado se
    limita con np.minimum(base, mezcla): la cuadrícula nunca aclara tinta
    más oscura que ella en los cruces, replicando la física real (la tinta
    se dibuja por encima de la cuadrícula impresa).

    Parameters
    ----------
    base : np.ndarray
        Imagen de fondo BGR, shape (H, W, 3), dtype uint8.
    layer_img : np.ndarray
        Capa BGR, shape (H, W, 3), dtype uint8, fondo blanco y líneas de color.
    opacity : float
        Peso de la capa en [0.0, 1.0] en los píxeles de línea.

    Returns
    -------
    np.ndarray
        Imagen mezclada BGR, shape (H, W, 3), dtype uint8.
    """
    base_f = base.astype(np.float32)
    layer_f = layer_img.astype(np.float32)

    line_mask = (layer_img.min(axis=2) < 250).astype(np.float32)[:, :, np.newaxis]

    blended = base_f * (1.0 - line_mask * opacity) + layer_f * (line_mask * opacity)
    result = np.minimum(base_f, blended)
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# add_stain
# ─────────────────────────────────────────────────────────────────────────────


def add_stain(
    image: np.ndarray,
    rng: np.random.Generator,
    n_blobs: int | None = None,
    strength: float | None = None,
    color_bgr: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Añade manchas irregulares translúcidas (café, agua, humedad).

    Genera blobs elípticos aleatorios, los difumina con un kernel grande
    para obtener bordes suaves de baja frecuencia, y los aplica como
    atenuación multiplicativa hacia el color de la mancha. Al ser
    multiplicativa, la tinta negra permanece intacta y el papel se tiñe —
    el comportamiento observado en los escaneos reales con daño de agua
    (demo_6_edge_case del set de fallo de Phase 5).

    Parameters
    ----------
    image : np.ndarray
        Imagen BGR, shape (H, W, 3), dtype uint8.
    rng : np.random.Generator
        Generador de aleatoriedad con semilla.
    n_blobs : int | None
        Número de manchas. Si es None, se muestrea en [1, 4].
    strength : float | None
        Intensidad máxima de la mancha en [0.0, 1.0] (1.0 = el papel alcanza
        el color puro de la mancha en el centro del blob). Si es None, se
        muestrea en [0.25, 0.70].
    color_bgr : tuple[int, int, int] | None
        Color BGR de la mancha. Si es None, marrón-grisáceo muestreado
        (B más bajo, R más alto — tono cálido de café/humedad).

    Returns
    -------
    np.ndarray
        Imagen BGR, shape (H, W, 3), dtype uint8.

    Raises
    ------
    ValueError
        Si image no es BGR (H, W, 3).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image debe ser BGR (H, W, 3), recibido shape={image.shape}")

    h, w = image.shape[:2]

    if n_blobs is None:
        n_blobs = int(rng.integers(1, 5))
    if strength is None:
        strength = float(rng.uniform(0.25, 0.70))
    if color_bgr is None:
        base_tone = int(rng.integers(100, 165))
        warm_r = int(rng.integers(10, 35))
        warm_g = int(rng.integers(0, 15))
        color_bgr = (base_tone, base_tone + warm_g, base_tone + warm_r)

    if n_blobs <= 0 or strength <= 0.0:
        return image.copy()

    # Máscara de blobs elípticos sobre lienzo binario
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_blobs):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        ax = int(rng.integers(max(4, w // 12), max(5, w // 3)))
        ay = int(rng.integers(max(4, h // 12), max(5, h // 3)))
        angle = float(rng.uniform(0.0, 180.0))
        cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, 1.0, thickness=-1)

    # Blur grande: bordes de mancha suaves, sin contornos duros
    k = max(3, (min(h, w) // 4) | 1)
    alpha = cv2.GaussianBlur(mask, (k, k), 0)
    peak = float(alpha.max())
    if peak < 1e-6:
        return image.copy()
    alpha = (alpha / peak) * strength  # (H, W) en [0, strength]

    # Atenuación multiplicativa hacia el color de la mancha:
    # gain_c = 1 - alpha * (1 - color_c/255). Papel blanco -> color; tinta 0 -> 0.
    color_f = np.array(color_bgr, dtype=np.float32) / 255.0
    gain = 1.0 - alpha[:, :, np.newaxis] * (1.0 - color_f[np.newaxis, np.newaxis, :])
    result = image.astype(np.float32) * gain
    return np.clip(result, 0.0, 255.0).astype(np.uint8)
