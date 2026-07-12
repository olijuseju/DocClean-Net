"""
data/generators/strokes.py
==========================
Genera trazos sintéticos (ground truth limpio) sobre un canvas de papel.

Dibuja una combinación aleatoria de curvas Bézier cúbicas, segmentos de línea,
elipses y marcos rectangulares (viñetas de cómic), simulando dibujos a
bolígrafo sobre papel de cuaderno con densidad alta.

Convenciones:
  - Imágenes BGR uint8 a lo largo de todo el módulo.
  - El canvas recibido NO se modifica in-place; se retorna una copia.
  - "Negro" = (0, 0, 0) en BGR. El fondo hereda el color del canvas.
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Constantes de diseño ──────────────────────────────────────────────────────

# Rango de número de primitivas por imagen — subido para simular cómics densos
_MIN_STROKES = 20
_MAX_STROKES = 60

# Grosor de trazo en píxeles (bolígrafo fino)
_THICKNESS_MIN = 1
_THICKNESS_MAX = 3

# Subdivisión de curvas Bézier: más puntos = curva más suave
_BEZIER_STEPS = 80

# Color de tinta: negro puro en BGR
_INK_COLOR = (0, 0, 0)


# ── API pública ───────────────────────────────────────────────────────────────


def generate_strokes(
    canvas: np.ndarray,
    rng: np.random.Generator,
    n_strokes: int | None = None,
    ink_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Dibuja trazos sintéticos en negro sobre el canvas recibido.

    Genera una mezcla aleatoria de curvas Bézier cúbicas, segmentos de línea,
    elipses y marcos rectangulares (viñetas), con densidad alta para simular
    dibujos de cómic a bolígrafo sobre papel de cuaderno.

    El canvas original no se modifica. Se retorna una copia con los trazos.

    Parameters
    ----------
    canvas : np.ndarray
        Imagen de fondo BGR, shape (H, W, 3), dtype uint8.
        Típicamente la salida de generate_paper().
    rng : np.random.Generator
        Generador de aleatoriedad con semilla.
    n_strokes : int | None
        Número total de primitivas a dibujar. Si es None, se muestrea
        aleatoriamente en [_MIN_STROKES, _MAX_STROKES]. Útil en tests
        para fijar la cantidad sin modificar el rng.
    ink_color : tuple[int, int, int] | None
        Color BGR de los trazos. Si es None, negro puro (0, 0, 0) —
        comportamiento histórico. Grises claros (p. ej. (140, 140, 140))
        simulan lápiz tenue, uno de los modos de fallo reales de Phase 5.

    Returns
    -------
    np.ndarray
        Imagen BGR, shape (H, W, 3), dtype uint8.
        Fondo = canvas original. Trazos = ink_color (negro por defecto).
        Este array es el ground truth limpio (clean image) del par de entrenamiento.
    """
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError(
            f"canvas debe ser BGR (H, W, 3), recibido shape={canvas.shape}"
        )

    if ink_color is None:
        ink_color = _INK_COLOR

    result = canvas.copy()
    h, w = result.shape[:2]

    if n_strokes is None:
        n_strokes = int(rng.integers(_MIN_STROKES, _MAX_STROKES + 1))

    # Distribución: 40% Bézier, 25% líneas, 15% elipses, 20% viñetas
    # Las viñetas (marcos rectangulares) son críticas para cómics: el modelo
    # debe aprender que las líneas rectas largas negras NO son cuadrícula.
    primitive_weights = np.array([0.40, 0.25, 0.15, 0.20])
    counts = _distribute(n_strokes, primitive_weights, rng)

    n_bezier, n_lines, n_ellipses, n_panels = counts

    for _ in range(n_bezier):
        _draw_bezier(result, h, w, rng, ink_color)

    for _ in range(n_lines):
        _draw_line(result, h, w, rng, ink_color)

    for _ in range(n_ellipses):
        _draw_ellipse(result, h, w, rng, ink_color)

    for _ in range(n_panels):
        _draw_panel(result, h, w, rng, ink_color)

    return result


# ── Primitivas internas ───────────────────────────────────────────────────────


def _distribute(
    total: int,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    """Distribuye `total` unidades según pesos normalizados.

    Usa el método de residuos para que la suma siempre sea exactamente `total`.

    Parameters
    ----------
    total : int
        Número total a repartir.
    weights : np.ndarray
        Pesos relativos, shape (N,). No necesitan sumar 1.
    rng : np.random.Generator
        Usado para desempatar residuos aleatorios.

    Returns
    -------
    list[int]
        Lista de enteros de longitud N que suma `total`.
    """
    weights = weights / weights.sum()
    raw = weights * total
    counts = raw.astype(int).tolist()
    remainder = total - sum(counts)

    residuals = raw - np.array(counts, dtype=float)
    indices = np.argsort(residuals)[::-1]
    for i in range(remainder):
        counts[int(indices[i % len(indices)])] += 1

    return counts


def _random_thickness(rng: np.random.Generator) -> int:
    """Muestrea grosor de trazo en [_THICKNESS_MIN, _THICKNESS_MAX]."""
    return int(rng.integers(_THICKNESS_MIN, _THICKNESS_MAX + 1))


def _draw_bezier(
    img: np.ndarray,
    h: int,
    w: int,
    rng: np.random.Generator,
    color: tuple[int, int, int] = _INK_COLOR,
) -> None:
    """Dibuja una curva Bézier cúbica sobre img (in-place).

    Los 4 puntos de control se muestrean dentro de la imagen con un margen
    del 5% para que los trazos no queden cortados en el borde.
    """
    margin_y = max(1, int(h * 0.05))
    margin_x = max(1, int(w * 0.05))

    pts = rng.integers(
        low=[margin_x, margin_y],
        high=[w - margin_x, h - margin_y],
        size=(4, 2),
    )

    thickness = _random_thickness(rng)
    t_vals = np.linspace(0.0, 1.0, _BEZIER_STEPS)
    curve_pts = _cubic_bezier(pts, t_vals)

    for i in range(len(curve_pts) - 1):
        p1 = (int(curve_pts[i, 0]), int(curve_pts[i, 1]))
        p2 = (int(curve_pts[i + 1, 0]), int(curve_pts[i + 1, 1]))
        cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)


def _cubic_bezier(
    control_pts: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Evalúa la curva Bézier cúbica en los valores de t dados.

    B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3

    Parameters
    ----------
    control_pts : np.ndarray
        Puntos de control, shape (4, 2). Orden: [x, y].
    t : np.ndarray
        Valores de parámetro en [0, 1], shape (N,).

    Returns
    -------
    np.ndarray
        Puntos de la curva, shape (N, 2). Dtype float64.
    """
    p0, p1, p2, p3 = control_pts.astype(float)
    mt = 1.0 - t
    curve = (
        mt[:, None] ** 3 * p0
        + 3 * mt[:, None] ** 2 * t[:, None] * p1
        + 3 * mt[:, None] * t[:, None] ** 2 * p2
        + t[:, None] ** 3 * p3
    )
    return curve


def _draw_line(
    img: np.ndarray,
    h: int,
    w: int,
    rng: np.random.Generator,
    color: tuple[int, int, int] = _INK_COLOR,
) -> None:
    """Dibuja un segmento de línea recto sobre img (in-place).

    Longitud máxima limitada al 60% de la diagonal para que no se confunda
    visualmente con cuadrícula residual durante la inspección manual.
    """
    max_len_sq = (0.6 * np.hypot(h, w)) ** 2

    for _ in range(10):
        x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x2, y2 = int(rng.integers(0, w)), int(rng.integers(0, h))
        if (x2 - x1) ** 2 + (y2 - y1) ** 2 <= max_len_sq:
            break

    thickness = _random_thickness(rng)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)


def _draw_ellipse(
    img: np.ndarray,
    h: int,
    w: int,
    rng: np.random.Generator,
    color: tuple[int, int, int] = _INK_COLOR,
) -> None:
    """Dibuja la elipse (solo contorno) sobre img (in-place)."""
    margin = 8
    cx = int(rng.integers(margin, w - margin))
    cy = int(rng.integers(margin, h - margin))

    max_ax = min(cx, w - cx, cy, h - cy) - 2
    max_ax = max(4, max_ax)

    axis_a = int(rng.integers(4, max(5, max_ax)))
    axis_b = int(rng.integers(4, max(5, max_ax)))
    angle = float(rng.uniform(0.0, 180.0))

    thickness = _random_thickness(rng)
    cv2.ellipse(
        img,
        (cx, cy),
        (axis_a, axis_b),
        angle,
        0,
        360,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def _draw_panel(
    img: np.ndarray,
    h: int,
    w: int,
    rng: np.random.Generator,
    color: tuple[int, int, int] = _INK_COLOR,
) -> None:
    """Dibuja un marco rectangular de viñeta de cómic sobre img (in-place).

    Las viñetas son rectángulos con líneas rectas y largas en negro — la
    primitiva más difícil de distinguir de la cuadrícula residual. Incluirlas
    en el ground truth fuerza al modelo a aprender esa distinción.

    El marco ocupa entre el 15% y el 60% del área de la imagen, con posición
    aleatoria dentro de los límites. Grosor entre 1 y 3px, igual que el resto
    de trazos.
    """
    margin = max(4, int(min(h, w) * 0.05))

    # Coordenadas del rectángulo: dos esquinas opuestas
    x1 = int(rng.integers(margin, w - margin))
    y1 = int(rng.integers(margin, h - margin))

    # Tamaño mínimo: 15% del lado; máximo: 60%
    min_size = int(min(h, w) * 0.15)
    max_size = int(min(h, w) * 0.60)
    pw = int(rng.integers(min_size, max(min_size + 1, max_size)))
    ph = int(rng.integers(min_size, max(min_size + 1, max_size)))

    x2 = min(w - margin, x1 + pw)
    y2 = min(h - margin, y1 + ph)

    if x2 <= x1 or y2 <= y1:
        return

    thickness = _random_thickness(rng)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
