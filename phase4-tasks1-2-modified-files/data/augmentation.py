"""
data/augmentation.py
====================
Transforms de augmentación sincronizados para pares (dirty, clean).

La misma transformación geométrica se aplica a ambas imágenes del par para
que sigan siendo correspondientes. Las transformaciones fotométricas solo se
aplican a la imagen dirty (simular variaciones de escáner que no afectan al
ground truth limpio).

Función pública principal:
    augment_pair(dirty, clean, rng) -> tuple[np.ndarray, np.ndarray]

Convenciones:
    - Imágenes BGR uint8, shape (H, W, 3).
    - dirty y clean deben tener la misma shape al entrar.
    - La shape de salida es idéntica a la de entrada.
    - rng: np.random.Generator siempre explícito.
"""

from __future__ import annotations

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────


def augment_pair(
    dirty: np.ndarray,
    clean: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica augmentación sincronizada a un par (dirty, clean).

    Transforms geométricos (aplicados a ambas imágenes por igual):
        - Flip horizontal  (50% de probabilidad)
        - Flip vertical    (20% de probabilidad)
        - Rotación leve    (±5°, 40% de probabilidad)

    Transforms fotométricos (solo a dirty, simulan variaciones del escáner):
        - Ruido gaussiano aditivo  (50% de probabilidad, σ en [1, 4])
        - Ajuste de brillo global  (50% de probabilidad, ±15 niveles)

    Parameters
    ----------
    dirty : np.ndarray
        Imagen sucia BGR, shape (H, W, 3), dtype uint8.
    clean : np.ndarray
        Imagen limpia BGR, shape (H, W, 3), dtype uint8.
    rng : np.random.Generator
        Generador de aleatoriedad con semilla.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Par (dirty_aug, clean_aug), ambas BGR uint8, misma shape que la entrada.

    Raises
    ------
    ValueError
        Si dirty y clean no tienen la misma shape.
    """
    if dirty.shape != clean.shape:
        raise ValueError(
            f"dirty y clean deben tener la misma shape: "
            f"dirty={dirty.shape}, clean={clean.shape}"
        )

    d = dirty.copy()
    c = clean.copy()

    # ── Transforms geométricos (sincronizados) ────────────────────────────────
    if rng.random() < 0.5:
        d, c = _flip_horizontal(d), _flip_horizontal(c)

    if rng.random() < 0.2:
        d, c = _flip_vertical(d), _flip_vertical(c)

    if rng.random() < 0.4:
        angle = float(rng.uniform(-5.0, 5.0))
        d, c = _rotate(d, angle), _rotate(c, angle)

    # ── Transforms fotométricos (solo dirty) ─────────────────────────────────
    if rng.random() < 0.5:
        sigma = float(rng.uniform(1.0, 4.0))
        d = _add_gaussian_noise(d, sigma, rng)

    if rng.random() < 0.5:
        delta = float(rng.uniform(-15.0, 15.0))
        d = _adjust_brightness(d, delta)

    return d, c


# ─────────────────────────────────────────────────────────────────────────────
# Transforms geométricos internos
# ─────────────────────────────────────────────────────────────────────────────


def _flip_horizontal(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


def _flip_vertical(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 0)


def _rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rota la imagen alrededor de su centro, rellenando con blanco.

    Ángulos pequeños (±5°) preservan casi toda la imagen original.
    El relleno blanco (255) es coherente con el fondo de papel.
    """
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Transforms fotométricos internos
# ─────────────────────────────────────────────────────────────────────────────


def _add_gaussian_noise(
    img: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Añade ruido gaussiano aditivo a la imagen.

    Parameters
    ----------
    img : np.ndarray
        BGR uint8, shape (H, W, 3).
    sigma : float
        Desviación estándar del ruido.
    rng : np.random.Generator
        Fuente de aleatoriedad.

    Returns
    -------
    np.ndarray
        BGR uint8, shape (H, W, 3), con ruido añadido y valores recortados a [0, 255].
    """
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    result = img.astype(np.float32) + noise
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def _adjust_brightness(img: np.ndarray, delta: float) -> np.ndarray:
    """Sube o baja el brillo global en `delta` niveles.

    Parameters
    ----------
    img : np.ndarray
        BGR uint8, shape (H, W, 3).
    delta : float
        Offset de brillo. Positivo = más brillante, negativo = más oscuro.

    Returns
    -------
    np.ndarray
        BGR uint8, shape (H, W, 3), valores recortados a [0, 255].
    """
    result = img.astype(np.float32) + delta
    return np.clip(result, 0.0, 255.0).astype(np.uint8)
