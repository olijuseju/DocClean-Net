"""
data/generators/illumination.py
================================
Campos de iluminación no uniforme para el dataset sintético de DocClean-Net.

Simula las condiciones de captura que un escáner plano NO produce pero una
app de escaneo por cámara de móvil sí: gradientes de sombra direccionales,
sombras de esquina, caída radial de luz y campos de iluminación suaves e
irregulares.

El campo es una ganancia multiplicativa por píxel en [min_gain, 1.0] aplicada
por igual a los 3 canales. Al ser multiplicativa es segura para la tinta: los
píxeles negros (0) permanecen negros, y el papel se oscurece de forma
espacialmente suave — exactamente el artefacto medido en los escaneos reales
que fallan (papel en sombra ≈ gris 84-156 frente a papel nominal ≈ 246).

Uso previsto (Phase 5): se aplica SOLO a la imagen dirty del par de
entrenamiento, dejando el target limpio sin sombrear, de modo que la red
aprende explícitamente a normalizar el nivel de fondo.
"""

import cv2
import numpy as np

# Modos de iluminación disponibles. El muestreo aleatorio elige entre ellos
# con probabilidad uniforme.
_MODES = ("linear", "corner", "radial", "smooth")

# Rango de muestreo de min_gain. El suelo 0.40 cubre la sombra real más
# oscura medida en el set de fallo (papel 94 / papel nominal 227 ≈ 0.41).
_MIN_GAIN_LOW = 0.40
_MIN_GAIN_HIGH = 0.85

# Resolución base del campo "smooth" antes de reescalar. Baja a propósito:
# la iluminación real es de muy baja frecuencia espacial.
_SMOOTH_GRID = 6


def apply_illumination(
    image: np.ndarray,
    rng: np.random.Generator,
    mode: str | None = None,
    min_gain: float | None = None,
) -> np.ndarray:
    """Aplica un campo de iluminación no uniforme multiplicativo a la imagen.

    Parameters
    ----------
    image : np.ndarray
        Imagen BGR, shape (H, W, 3), dtype uint8.
    rng : np.random.Generator
        Generador de aleatoriedad con semilla.
    mode : str | None
        Uno de {"linear", "corner", "radial", "smooth"}. Si es None, se
        muestrea uniformemente entre los cuatro.
        - "linear": gradiente direccional (sombra tipo foto de móvil).
        - "corner": sombra concentrada en una esquina aleatoria.
        - "radial": caída de luz radial (viñeteado agresivo generalizado).
        - "smooth": campo suave irregular (iluminación desigual sin
          estructura dominante).
    min_gain : float | None
        Ganancia mínima del campo, en (0.0, 1.0]. La zona más oscura de la
        imagen queda multiplicada por este valor. Si es None, se muestrea
        en [0.40, 0.85].

    Returns
    -------
    np.ndarray
        Imagen BGR, shape (H, W, 3), dtype uint8. La zona mejor iluminada
        conserva el brillo original (ganancia 1.0).

    Raises
    ------
    ValueError
        Si image no es BGR (H, W, 3), si mode no es válido, o si min_gain
        está fuera de (0.0, 1.0].
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image debe ser BGR (H, W, 3), recibido shape={image.shape}")

    if mode is None:
        mode = str(rng.choice(_MODES))
    if mode not in _MODES:
        raise ValueError(f"mode debe ser uno de {_MODES}, recibido {mode!r}")

    if min_gain is None:
        min_gain = float(rng.uniform(_MIN_GAIN_LOW, _MIN_GAIN_HIGH))
    if not 0.0 < min_gain <= 1.0:
        raise ValueError(f"min_gain debe estar en (0.0, 1.0], recibido {min_gain}")

    h, w = image.shape[:2]

    if mode == "linear":
        field = _field_linear(h, w, rng)
    elif mode == "corner":
        field = _field_corner(h, w, rng)
    elif mode == "radial":
        field = _field_radial(h, w, rng)
    else:
        field = _field_smooth(h, w, rng)

    gain = min_gain + (1.0 - min_gain) * field  # (H, W) float32 en [min_gain, 1]
    result = image.astype(np.float32) * gain[:, :, np.newaxis]
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


# ── Campos internos ───────────────────────────────────────────────────────────
# Cada helper devuelve un campo float32 (H, W) normalizado a [0, 1] con
# max == 1.0 (garantiza que la zona mejor iluminada conserve el brillo).


def _normalize_field(field: np.ndarray) -> np.ndarray:
    """Normaliza un campo float32 a [0, 1]. Campos degenerados → todo 1.0."""
    fmin = float(field.min())
    fmax = float(field.max())
    if fmax - fmin < 1e-6:
        return np.ones_like(field, dtype=np.float32)
    return ((field - fmin) / (fmax - fmin)).astype(np.float32)


def _field_linear(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Gradiente direccional: 1.0 en un extremo, 0.0 en el opuesto.

    La dirección se muestrea uniformemente en [0, 2π). Es la forma de la
    sombra medida en el escaneo real de Google Drive Scan (limitation_1).
    """
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = xx * np.cos(theta) + yy * np.sin(theta)
    return _normalize_field(proj)


def _field_corner(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Sombra de esquina: 0.0 en una esquina aleatoria, 1.0 en la opuesta.

    El exponente > 1 concentra la oscuridad cerca de la esquina, dejando el
    resto de la página casi sin afectar (patrón típico de página doblada o
    sombra de la mano al fotografiar).
    """
    corner_y = float(rng.choice([0, h - 1]))
    corner_x = float(rng.choice([0, w - 1]))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.hypot(yy - corner_y, xx - corner_x)
    field = _normalize_field(dist)
    exponent = float(rng.uniform(1.0, 2.5))
    return np.power(field, 1.0 / exponent).astype(np.float32)


def _field_radial(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Caída radial: 1.0 en un centro jitterizado, 0.0 en el borde más lejano.

    Generaliza el viñeteado del escáner con centros descentrados y caídas
    más agresivas que las de generate_paper (que limita a 20%).
    """
    cy = float(rng.uniform(0.25 * h, 0.75 * h))
    cx = float(rng.uniform(0.25 * w, 0.75 * w))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.hypot(yy - cy, xx - cx)
    exponent = float(rng.uniform(1.0, 2.0))
    falloff = np.power(_normalize_field(dist), exponent)
    return (1.0 - falloff).astype(np.float32)


def _field_smooth(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Campo suave irregular: ruido de baja resolución reescalado y difuminado.

    Modela iluminación desigual sin estructura dominante (varias fuentes de
    luz, papel ligeramente ondulado).
    """
    coarse = rng.uniform(0.0, 1.0, size=(_SMOOTH_GRID, _SMOOTH_GRID)).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    # Blur proporcional al tamaño para eliminar cualquier resto de estructura
    # de la rejilla de baja resolución. Kernel impar obligatorio en cv2.
    k = max(3, (min(h, w) // 8) | 1)
    field = cv2.GaussianBlur(field, (k, k), 0)
    return _normalize_field(field)
