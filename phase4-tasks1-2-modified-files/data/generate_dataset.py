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
from data.generators.degradations import add_blue_grid, add_ruled_lines, add_watermark
from data.generators.paper import generate_paper
from data.generators.strokes import generate_strokes

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_SIZE = 512  # píxeles — lado de cada imagen cuadrada
_DEFAULT_N = 5000
_DEFAULT_SEED = 42
_DEFAULT_WORKERS = 1

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
) -> tuple[np.ndarray, np.ndarray]:
    """Genera un par (dirty, clean) para el índice `idx`.

    Cada par usa su propio rng derivado de seed + idx para reproducibilidad
    total: regenerar el índice N siempre produce el mismo par.

    Parameters
    ----------
    idx : int
        Índice de la muestra (0-based). Determina la semilla del rng.
    size : int
        Lado de la imagen cuadrada en píxeles.
    seed : int
        Semilla base del dataset. El rng del par es seed + idx.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (dirty, clean) — ambas BGR uint8, shape (size, size, 3).
    """
    rng = np.random.default_rng(seed=seed + idx)

    paper = generate_paper(size, size, rng)
    clean = generate_strokes(paper, rng)
    dirty = _apply_random_degradation(clean, rng)

    return dirty, clean


def _apply_random_degradation(
    image: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Elige y aplica una degradación aleatoria según _DEGRADATION_WEIGHTS.

    Parameters
    ----------
    image : np.ndarray
        Imagen limpia BGR, shape (H, W, 3), dtype uint8.
    rng : np.random.Generator
        Fuente de aleatoriedad.

    Returns
    -------
    np.ndarray
        Imagen degradada BGR, shape (H, W, 3), dtype uint8.
    """
    weights = _DEGRADATION_WEIGHTS / _DEGRADATION_WEIGHTS.sum()
    choice = int(rng.choice(len(weights), p=weights))

    if choice == 0:
        return add_blue_grid(image, rng)
    elif choice == 1:
        return add_ruled_lines(image, rng)
    elif choice == 2:
        img = add_blue_grid(image, rng)
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

    dirty, clean = generate_pair(idx, args["size"], args["seed"])

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

    print("DocClean-Net — Generador de dataset sintético")
    print(
        f"  n={args.n}, size={args.size}×{args.size}, seed={args.seed}, workers={args.workers}"
    )
    print(f"  output: {args.output.resolve()}\n")

    generate_dataset(
        n=args.n,
        output_dir=args.output,
        size=args.size,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
