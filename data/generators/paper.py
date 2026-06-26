"""
data/generators/paper.py
========================
Genera fondos de papel sintético para el dataset de entrenamiento de DocClean-Net.

El fondo resultante simula papel de cuaderno escaneado: ligeramente azulado-grisáceo
(característico del papel de cuaderno español), con ruido de grano y vignetting
opcional de escáner.
"""

import cv2
import numpy as np


def generate_paper(
    h: int,
    w: int,
    rng: np.random.Generator,
    sigma_noise: float | None = None,
    vignette_strength: float | None = None,
) -> np.ndarray:
    """Genera un fondo de papel de cuaderno sintético con tono azulado-grisáceo.

    Simula papel de cuaderno escaneado: base ligeramente azulada (B > R, G > R),
    grano de papel (ruido gaussiano), y vignetting del escáner en los bordes.

    Parameters
    ----------
    h : int
        Alto de la imagen en píxeles.
    w : int
        Ancho de la imagen en píxeles.
    rng : np.random.Generator
        Generador de aleatoriedad con semilla.
    sigma_noise : float | None
        Desviación estándar del ruido gaussiano. Si es None, se muestrea en [2.0, 5.0].
    vignette_strength : float | None
        Intensidad del vignetting en [0.0, 1.0]. Si es None, se decide
        aleatoriamente: ~40% de probabilidad, fuerza en [0.05, 0.20].

    Returns
    -------
    np.ndarray
        Imagen BGR, shape (H, W, 3), dtype uint8.
        Tono base: papel de cuaderno ligeramente azulado (B≈225-240, G≈220-235, R≈210-225).
    """
    # ── 1. Base con tono de papel de cuaderno ────────────────────────────────
    # El papel de cuaderno español tiene un sesgo azulado-grisáceo perceptible
    # en el escaneo. Modelamos esto con B > G > R con diferencias pequeñas.
    # brightness base: ligeramente por debajo de 255 (papel no es blanco puro)
    brightness = float(rng.uniform(215.0, 242.0))

    # Sesgo de canal: B más alto, R más bajo — simula papel azulado
    # Rangos calibrados sobre el escaneo real de referencia
    b_bias = float(rng.uniform(5.0, 15.0))   # B es el más alto
    g_bias = float(rng.uniform(2.0,  8.0))   # G intermedio
    r_bias = 0.0                              # R es la referencia (más bajo)

    paper = np.empty((h, w, 3), dtype=np.float32)
    paper[:, :, 0] = brightness + b_bias   # canal B
    paper[:, :, 1] = brightness + g_bias   # canal G
    paper[:, :, 2] = brightness + r_bias   # canal R

    # ── 2. Ruido gaussiano (grano de papel) ──────────────────────────────────
    if sigma_noise is None:
        sigma_noise = float(rng.uniform(2.0, 5.0))

    grain = rng.normal(loc=0.0, scale=sigma_noise, size=(h, w, 3)).astype(np.float32)
    paper += grain

    # ── 3. Vignetting ────────────────────────────────────────────────────────
    apply_vignette = (
        vignette_strength is not None and vignette_strength > 0.0
    ) or (
        vignette_strength is None and rng.random() < 0.4
    )

    if apply_vignette:
        if vignette_strength is None:
            vignette_strength = float(rng.uniform(0.05, 0.20))
        paper = _apply_vignette(paper, h, w, vignette_strength)

    return np.clip(paper, 0.0, 255.0).astype(np.uint8)


def _apply_vignette(
    paper: np.ndarray,
    h: int,
    w: int,
    strength: float,
) -> np.ndarray:
    """Oscurece los bordes simulando el vignetting del escáner.

    Parameters
    ----------
    paper : np.ndarray
        Imagen float32, shape (H, W, 3). Modificada in-place.
    h, w : int
        Dimensiones de la imagen.
    strength : float
        Fracción de brillo perdida en el borde extremo (0.0–1.0).

    Returns
    -------
    np.ndarray
        Misma referencia que `paper`, modificada.
    """
    y_coords = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    x_coords = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    xv, yv = np.meshgrid(x_coords, y_coords)

    dist_sq = xv ** 2 + yv ** 2
    sigma_v = 1.4
    vignette_mask = np.exp(-dist_sq / (2.0 * sigma_v ** 2))
    vignette_mask = 1.0 - strength * (1.0 - vignette_mask)
    vignette_mask = vignette_mask[:, :, np.newaxis]

    paper *= vignette_mask
    return paper
