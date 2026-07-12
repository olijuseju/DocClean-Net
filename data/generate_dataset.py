"""
data/generate_dataset.py
========================
Script CLI para generar el dataset sintético de DocClean-Net.

Para cada muestra genera:
    1. Fondo de papel (generate_paper)
    2. Trazos encima → imagen limpia / ground truth (generate_strokes)
    3. Degradación aleatoria sobre la imagen limpia → imagen sucia (dirty)
       La degradación se elige aleatoriamente entre:
         - cuadrícula azul (50%)
         - líneas horizontales (25%)
         - cuadrícula azul + líneas (15%)
         - marca de agua (10%)

Salida:
    output_dir/dirty/dirty_XXXXXX.png
    output_dir/clean/clean_XXXXXX.png

Uso:
    python generate_dataset.py --n 5000 --output data/synthetic/
    python generate_dataset.py --n 100  --output data/synthetic/ --seed 0
    python generate_dataset.py --n 5000 --output data/synthetic/ --size 256 --workers 4
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Imports relativos: funcionan cuando el paquete está instalado o cuando
# se ejecuta desde la raíz del repo con `python data/generate_dataset.py`.
from data.generators.degradations import (
    add_blue_grid,
    add_ruled_lines,
    add_stain,
    add_watermark,
)
from data.generators.illumination import apply_illumination
from data.generators.paper import generate_paper
from data.generators.strokes import generate_strokes

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_SIZE = 512  # píxeles — lado de cada imagen cuadrada
_DEFAULT_N = 5000
_DEFAULT_SEED = 42
_DEFAULT_WORKERS = 1

# Fracción de pares generados con el muestreo "domain-robust" de Phase 5
# (iluminación no uniforme, cuadrículas oscuras/densas, tinta tenue, manchas).
# El resto usa la distribución histórica v1.0 para no regresionar los casos
# que ya funcionaban.
_DEFAULT_DOMAIN_ROBUST_PROB = 0.5

# Muestreo domain-robust — rangos calibrados sobre el set real de fallo de
# Phase 5 (18 escaneos: cómics en milimetrado, foto con sombra, apuntes con
# manchas). Medidas de referencia: grid gris ≈85 BGR (72,57,57), spacing
# 12-16 px, lápiz gris 64-92, papel en sombra 84-156.
_ROBUST_INK_FAINT_PROB = 0.40  # prob. de tinta tenue (vs. negro)
_ROBUST_INK_GRAY_RANGE = (60, 160)
_ROBUST_GRID_DARK_PROB = 0.50  # prob. de cuadrícula impresa oscura (vs. azul)
_ROBUST_GRID_GRAY_RANGE = (60, 125)
_ROBUST_GRID_BLUE_BIAS_MAX = 16  # ligera dominante azul del milimetrado real
_ROBUST_GRID_SPACING_RANGE = (10, 46)
_ROBUST_GRID_OPACITY_RANGE = (0.55, 1.0)
_ROBUST_ILLUMINATION_PROB = 0.70
_ROBUST_STAIN_PROB = 0.25

# Pesos de degradación: [blue_grid, ruled_lines, grid+lines, watermark]
_DEGRADATION_WEIGHTS = np.array([0.50, 0.25, 0.15, 0.10])

_WATERMARK_TEXTS = [
    "BORRADOR",
    "DRAFT",
    "CONFIDENCIAL",
    "SAMPLE",
    "REVISIÓN",
    "COPIA",
    "COPY",
    "VOID",
]


# ─────────────────────────────────────────────────────────────────────────────
# Generación de un par individual
# ─────────────────────────────────────────────────────────────────────────────


def generate_pair(
    idx: int,
    size: int,
    seed: int,
    domain_robust_prob: float = _DEFAULT_DOMAIN_ROBUST_PROB,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera un par (dirty, clean) para el índice `idx`.

    Cada par usa su propio rng derivado de seed + idx para reproducibilidad
    total: regenerar el índice N siempre produce el mismo par.

    Con probabilidad `domain_robust_prob` el par se genera con el muestreo
    "domain-robust" de Phase 5: tinta posiblemente tenue (lápiz), cuadrícula
    posiblemente oscura/densa (papel milimetrado impreso), iluminación no
    uniforme y manchas — aplicadas SOLO a la imagen dirty, de modo que el
    target permanece limpio y bien iluminado y la red aprende a normalizar.

    Parameters
    ----------
    idx : int
        Índice de la muestra (0-based). Determina la semilla del rng.
    size : int
        Lado de la imagen cuadrada en píxeles.
    seed : int
        Semilla base del dataset. El rng del par es seed + idx.
    domain_robust_prob : float
        Probabilidad en [0.0, 1.0] de usar el muestreo domain-robust.
        0.0 reproduce exactamente la distribución v1.0.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (dirty, clean) — ambas BGR uint8, shape (size, size, 3).
    """
    rng = np.random.default_rng(seed=seed + idx)

    robust = bool(rng.random() < domain_robust_prob)

    paper = generate_paper(size, size, rng)

    ink_color: tuple[int, int, int] | None = None
    if robust and rng.random() < _ROBUST_INK_FAINT_PROB:
        gray = int(rng.integers(*_ROBUST_INK_GRAY_RANGE))
        ink_color = (gray, gray, gray)

    clean = generate_strokes(paper, rng, ink_color=ink_color)
    dirty = _apply_random_degradation(clean, rng, robust=robust)

    if robust and rng.random() < _ROBUST_STAIN_PROB:
        dirty = add_stain(dirty, rng)
    if robust and rng.random() < _ROBUST_ILLUMINATION_PROB:
        dirty = apply_illumination(dirty, rng)

    return dirty, clean


def _sample_robust_grid_params(
    rng: np.random.Generator,
) -> dict:
    """Muestrea parámetros de cuadrícula del régimen domain-robust.

    Con prob. _ROBUST_GRID_DARK_PROB produce una cuadrícula impresa oscura
    (gris casi acromático con ligera dominante azul, blend opaco); en el
    resto de casos, azul histórica pero con el spacing extendido hacia
    densidades de milimetrado (10 px).

    Returns
    -------
    dict
        Kwargs para add_blue_grid: spacing, opacity, color_bgr, opaque_lines.
    """
    spacing = int(rng.integers(*_ROBUST_GRID_SPACING_RANGE))

    if rng.random() < _ROBUST_GRID_DARK_PROB:
        gray = int(rng.integers(*_ROBUST_GRID_GRAY_RANGE))
        blue_bias = int(rng.integers(0, _ROBUST_GRID_BLUE_BIAS_MAX))
        color = (min(255, gray + blue_bias), gray, gray)
        opacity = float(rng.uniform(*_ROBUST_GRID_OPACITY_RANGE))
        return {
            "spacing": spacing,
            "opacity": opacity,
            "color_bgr": color,
            "opaque_lines": True,
        }

    return {
        "spacing": spacing,
        "opacity": None,
        "color_bgr": None,
        "opaque_lines": False,
    }


def _apply_random_degradation(
    image: np.ndarray,
    rng: np.random.Generator,
    robust: bool = False,
) -> np.ndarray:
    """Elige y aplica una degradación aleatoria según _DEGRADATION_WEIGHTS.

    Parameters
    ----------
    image : np.ndarray
        Imagen limpia BGR, shape (H, W, 3), dtype uint8.
    rng : np.random.Generator
        Fuente de aleatoriedad.
    robust : bool
        Si es True, las cuadrículas usan el muestreo domain-robust de
        Phase 5 (spacing denso, colores oscuros, blend opaco). Las demás
        degradaciones no cambian.

    Returns
    -------
    np.ndarray
        Imagen degradada BGR, shape (H, W, 3), dtype uint8.
    """
    weights = _DEGRADATION_WEIGHTS / _DEGRADATION_WEIGHTS.sum()
    choice = int(rng.choice(len(weights), p=weights))

    grid_kwargs: dict = _sample_robust_grid_params(rng) if robust else {}

    if choice == 0:
        return add_blue_grid(image, rng, **grid_kwargs)
    elif choice == 1:
        return add_ruled_lines(image, rng)
    elif choice == 2:
        img = add_blue_grid(image, rng, **grid_kwargs)
        return add_ruled_lines(img, rng)
    else:
        text = str(rng.choice(_WATERMARK_TEXTS))
        return add_watermark(image, rng, text=text)


# ─────────────────────────────────────────────────────────────────────────────
# Worker para multiprocessing
# ─────────────────────────────────────────────────────────────────────────────


def _worker_init(shared_args: dict) -> None:
    """Inicializa el estado global del worker (evita pasar args en cada tarea)."""
    global _WORKER_ARGS
    _WORKER_ARGS = shared_args


def _worker_task(idx: int) -> int:
    """Genera y guarda el par idx. Retorna idx para confirmar."""
    args = _WORKER_ARGS  # type: ignore[name-defined]
    dirty_dir = Path(args["output_dir"]) / "dirty"
    clean_dir = Path(args["output_dir"]) / "clean"

    dirty, clean = generate_pair(
        idx,
        args["size"],
        args["seed"],
        domain_robust_prob=args["domain_robust_prob"],
    )

    dirty_path = dirty_dir / f"dirty_{idx:06d}.png"
    clean_path = clean_dir / f"clean_{idx:06d}.png"

    cv2.imwrite(str(dirty_path), dirty)
    cv2.imwrite(str(clean_path), clean)

    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────


def generate_dataset(
    n: int,
    output_dir: Path,
    size: int = _DEFAULT_SIZE,
    seed: int = _DEFAULT_SEED,
    workers: int = _DEFAULT_WORKERS,
    domain_robust_prob: float = _DEFAULT_DOMAIN_ROBUST_PROB,
) -> None:
    """Genera `n` pares (dirty, clean) y los guarda en output_dir.

    Parameters
    ----------
    n : int
        Número de pares a generar.
    output_dir : pathlib.Path
        Carpeta de salida. Se crean subdirectorios dirty/ y clean/ si no existen.
    size : int
        Lado de cada imagen cuadrada en píxeles.
    seed : int
        Semilla base para reproducibilidad.
    workers : int
        Número de procesos paralelos (1 = secuencial, sin multiprocessing).
    domain_robust_prob : float
        Fracción esperada de pares con muestreo domain-robust de Phase 5.
        0.0 reproduce la distribución v1.0; el default es 0.5.
    """
    dirty_dir = output_dir / "dirty"
    clean_dir = output_dir / "clean"
    dirty_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(n))
    shared_args = {
        "output_dir": str(output_dir),
        "size": size,
        "seed": seed,
        "domain_robust_prob": domain_robust_prob,
    }

    if workers <= 1:
        # Ejecución secuencial: más simple y debuggeable
        for idx in tqdm(indices, desc="Generating", unit="pair"):
            _worker_init(shared_args)
            _worker_task(idx)
    else:
        with multiprocessing.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(shared_args,),
        ) as pool:
            for _ in tqdm(
                pool.imap_unordered(_worker_task, indices),
                total=n,
                desc="Generating",
                unit="pair",
            ):
                pass

    print(f"\n✓ Dataset generado: {n} pares en {output_dir}")
    print(f"  dirty/ → {dirty_dir}")
    print(f"  clean/ → {clean_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="generate_dataset",
        description="Genera dataset sintético de pares (dirty, clean) para DocClean-Net.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ejemplos:
  python data/generate_dataset.py --n 5000 --output data/synthetic/
  python data/generate_dataset.py --n 100  --output data/synthetic/ --seed 0 --size 256
  python data/generate_dataset.py --n 5000 --output data/synthetic/ --workers 4
        """,
    )
    p.add_argument(
        "--n",
        type=int,
        default=_DEFAULT_N,
        help=f"Número de pares a generar [default: {_DEFAULT_N}]",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directorio de salida (se crea si no existe)",
    )
    p.add_argument(
        "--size",
        type=int,
        default=_DEFAULT_SIZE,
        help=f"Lado de cada imagen cuadrada en píxeles [default: {_DEFAULT_SIZE}]",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=f"Semilla base para reproducibilidad [default: {_DEFAULT_SEED}]",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=_DEFAULT_WORKERS,
        help=f"Procesos paralelos [default: {_DEFAULT_WORKERS}]",
    )
    p.add_argument(
        "--domain-robust-prob",
        type=float,
        default=_DEFAULT_DOMAIN_ROBUST_PROB,
        metavar="P",
        help=(
            "Fracción de pares con muestreo domain-robust de Phase 5 "
            "(iluminación, grids oscuros/densos, tinta tenue, manchas). "
            f"0.0 = distribución v1.0 exacta [default: {_DEFAULT_DOMAIN_ROBUST_PROB}]"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.n <= 0:
        print("[ERROR] --n debe ser > 0", file=sys.stderr)
        sys.exit(1)
    if args.size < 64:
        print("[ERROR] --size debe ser ≥ 64", file=sys.stderr)
        sys.exit(1)
    if args.workers < 1:
        print("[ERROR] --workers debe ser ≥ 1", file=sys.stderr)
        sys.exit(1)
    if not 0.0 <= args.domain_robust_prob <= 1.0:
        print("[ERROR] --domain-robust-prob debe estar en [0.0, 1.0]", file=sys.stderr)
        sys.exit(1)

    print("DocClean-Net — Generador de dataset sintético")
    print(
        f"  n={args.n}, size={args.size}×{args.size}, seed={args.seed}, "
        f"workers={args.workers}, domain_robust_prob={args.domain_robust_prob}"
    )
    print(f"  output: {args.output.resolve()}\n")

    generate_dataset(
        n=args.n,
        output_dir=args.output,
        size=args.size,
        seed=args.seed,
        workers=args.workers,
        domain_robust_prob=args.domain_robust_prob,
    )


if __name__ == "__main__":
    main()
