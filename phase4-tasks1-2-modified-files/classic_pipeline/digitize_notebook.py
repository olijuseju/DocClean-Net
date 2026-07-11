#!/usr/bin/env python3
"""
digitize_notebook.py — v4
==========================
Digitaliza dibujos escaneados de libretas de cuadrícula azul.

Pipeline:
  1. Detecta tintas de color (rojo, verde, etc.) sobre la imagen original.
  2. Construye un canal sintético que aclara la cuadrícula azul sin tocar los trazos.
  3. Extrae la máscara del dibujo mediante umbral adaptativo sobre ese canal.
  4. Detecta cuadrícula residual (líneas largas y rectas) y la repara con inpainting.
  5. Re-extrae la máscara sobre la imagen ya reparada.
  6. Fusiona la máscara del dibujo con las tintas de color detectadas.
  7. Elimina ruido puntual analizando componentes conectados por tamaño y contexto.
  8. Exporta imagen final: papel blanco, trazos negros.

Uso básico:
  python digitize_notebook.py -i escan.png -o digital.png
  python digitize_notebook.py -i escan.png -o digital.png --debug
  python digitize_notebook.py -i escan.png -o digital.png --alpha 5 --c-offset 7

Ajuste fino:
  Queda cuadrícula           → subir --alpha (5-6)  o  --c-offset (7-9)
  Faltan trazos claros       → bajar --alpha (2-3)  o  --c-offset (3-4)
  Mucho ruido de puntos      → subir --noise-area-small (25-30)
  Se pierden detalles finos  → bajar --noise-area-small (10-15)

Requisitos:
  pip install opencv-python-headless numpy

Autor: generado con Claude (Anthropic) — v3
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE I/O SEGUROS CON UNICODE
# cv2.imread / cv2.imwrite fallan silenciosamente en rutas con tildes, espacios
# o caracteres no-ASCII en Windows y algunos Linux. La solución es leer/escribir
# los bytes del archivo manualmente y dejar que OpenCV decodifique el buffer.
# ─────────────────────────────────────────────────────────────────────────────


def _imread(path: str) -> np.ndarray | None:
    """cv2.imread con soporte completo de Unicode en la ruta.

    Args:
        path: Ruta al archivo de imagen (acepta tildes, espacios, etc.)

    Returns:
        Imagen BGR uint8 o None si el archivo no existe / no es una imagen.
    """
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _imwrite(path: str, img: np.ndarray) -> bool:
    """cv2.imwrite con soporte completo de Unicode en la ruta.

    Infiere el formato por la extensión del path (igual que cv2.imwrite).

    Args:
        path: Ruta de destino.
        img:  Imagen BGR o grayscale, dtype uint8.

    Returns:
        True si se guardó correctamente, False en caso contrario.
    """
    ext = Path(path).suffix.lower()
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS POR DEFECTO
# ─────────────────────────────────────────────────────────────────────────────

# Canal sintético
DEFAULT_ALPHA = 4  # Fuerza de supresión azul. Subir → más cuadrícula eliminada.

# Umbral adaptativo
DEFAULT_BLOCK = 25  # Tamaño de vecindario adaptativo (impar, ≥ 15).
DEFAULT_C_OFFSET = (
    5  # Offset del umbral. Subir → más selectivo (menos falsos positivos).
)

# Detección de cuadrícula residual
DEFAULT_GRID_KERNEL = (
    120  # Longitud mínima (px) de línea recta para clasificarla como cuadrícula.
)
DEFAULT_INPAINT_R = 4  # Radio del inpainting TELEA para reparar zonas de cuadrícula.

# Limpieza de ruido por componentes
DEFAULT_NOISE_SMALL = (
    20  # Área (px) por debajo de la cual un componente aislado es ruido seguro.
)
DEFAULT_NOISE_MEDIUM = (
    100  # Área (px) por debajo de la cual se aplica criterio de forma.
)
DEFAULT_NOISE_RADIUS = (
    25  # Radio de influencia (px) de los trazos grandes para proteger vecinos.
)
DEFAULT_NOISE_DENSITY = (
    0.60  # Densidad mínima (area/bbox) para clasificar un mediano como blob=ruido.
)

# Tintas de color: rangos HSV (H 0-180, S 0-255, V 0-255)
# Añadir o quitar entradas para soportar más colores.
COLOR_INK_RANGES = [
    # nombre,   H_lo, H_hi,  S_lo, V_lo, V_hi,  R-G_min, R-B_min   (condición RGB adicional)
    ("rojo", 0, 12, 40, 40, 230, 15, 15),
    ("rojo-mag", 155, 180, 40, 40, 230, 15, 15),
]


# ─────────────────────────────────────────────────────────────────────────────
# PASO 0 — CARGA Y ANÁLISIS
# ─────────────────────────────────────────────────────────────────────────────


def load_image(path: str) -> np.ndarray:
    img = _imread(path)
    if img is None:
        raise FileNotFoundError(f"No se puede abrir: {path}")
    return img


def analyze_image(img: np.ndarray) -> dict:
    b, g, r = cv2.split(img)
    excess = float(b.mean()) - float(r.mean())
    info = {"blue_excess": excess, "shape": img.shape}
    if excess > 5:
        print(f"  Cuadrícula AZUL detectada (B−R = {excess:.1f})")
    else:
        print(f"  Exceso azul bajo (B−R = {excess:.1f}) — puede no ser cuadrícula azul")
    return info


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — DETECCIÓN DE TINTAS DE COLOR
# ─────────────────────────────────────────────────────────────────────────────


def detect_color_inks(img: np.ndarray) -> np.ndarray:
    """
    Detecta tintas de color (rojo, verde, etc.) sobre la imagen ORIGINAL
    antes de cualquier procesado, ya que el canal sintético las eliminaría.

    Estrategia doble:
      - Filtro HSV: captura por tono y saturación.
      - Filtro RGB: el canal dominante debe superar a los otros en al menos
        un umbral (evita falsos positivos en sombras grises desaturadas).

    Retorna máscara uint8 acumulada de todos los colores: 255 = tinta de color.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b_ch, g_ch, r_ch = cv2.split(img)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    combined = np.zeros(img.shape[:2], dtype=np.uint8)

    for name, h_lo, h_hi, s_lo, v_lo, v_hi, rg_min, rb_min in COLOR_INK_RANGES:
        # Máscara HSV
        hsv_mask = cv2.inRange(
            hsv, np.array([h_lo, s_lo, v_lo]), np.array([h_hi, 255, v_hi])
        )

        # Máscara RGB: canal R dominante
        rgb_mask = (
            (r_ch.astype(int) - g_ch.astype(int) > rg_min)
            & (r_ch.astype(int) - b_ch.astype(int) > rb_min)
        ).astype(np.uint8) * 255

        layer = cv2.bitwise_and(hsv_mask, rgb_mask)

        # Dilatación pequeña: rellena huecos internos del trazo
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        layer = cv2.dilate(layer, k, iterations=1)

        n = int(layer.sum() // 255)
        if n > 0:
            print(f"  Tinta '{name}': {n} px detectados")
        combined = cv2.bitwise_or(combined, layer)

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — CANAL SINTÉTICO (supresión de cuadrícula azul)
# ─────────────────────────────────────────────────────────────────────────────


def build_synthetic_channel(img: np.ndarray, alpha: int) -> np.ndarray:
    """
    Canal sintético = gray + alpha × max(B − R, 0)

    La cuadrícula azul (B > R) se aclara proporcionalmente a su azul:
      - Grid (B−R ≈ +11, gray ≈ 185): synthetic ≈ 185 + 4×11 = 229  → casi papel
      - Trazo negro (B−R ≈ 0, gray ≈ 70): synthetic ≈ 70              → igual de oscuro
      - Papel blanco (B−R ≈ +10, gray ≈ 250): synthetic ≈ 290 → 255  → sigue siendo blanco

    Resultado: umbralizar este canal captura todo el dibujo sin la cuadrícula.
    """
    b, g, r = cv2.split(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blue_excess = b.astype(np.float32) - r.astype(np.float32)
    synthetic = gray.astype(np.float32) + alpha * np.clip(blue_excess, 0, None)
    return np.clip(synthetic, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — EXTRACCIÓN DE MÁSCARA DE DIBUJO
# ─────────────────────────────────────────────────────────────────────────────


def extract_mask(synthetic: np.ndarray, block: int, c_offset: int) -> np.ndarray:
    """
    Umbral adaptativo gaussiano sobre el canal sintético.

    El umbral adaptativo ajusta el valor localmente en cada vecindario
    de 'block × block' píxeles, compensando variaciones de iluminación
    del escáner (sombras de encuadernación, vignetting).

    Retorna máscara: 255 = tinta, 0 = papel / cuadrícula.
    """
    synth_blur = cv2.GaussianBlur(synthetic, (5, 5), 0)
    return cv2.adaptiveThreshold(
        synth_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block,
        C=c_offset,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — DETECCIÓN Y ELIMINACIÓN DE CUADRÍCULA RESIDUAL
# ─────────────────────────────────────────────────────────────────────────────


def detect_grid_residual(mask: np.ndarray, grid_kernel: int) -> np.ndarray:
    """
    Extrae de la máscara solo los elementos LARGOS Y PERFECTAMENTE RECTOS,
    que son los únicos que pueden ser cuadrícula (el dibujo es curvo y corto).

    Usa apertura morfológica con kernels muy anchos (1 × grid_kernel y
    grid_kernel × 1): solo sobreviven elementos continuos de esa longitud.
    """
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (grid_kernel, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, grid_kernel))
    return cv2.add(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, kh),
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, kv),
    )


def inpaint_grid(
    img: np.ndarray, grid_mask: np.ndarray, inpaint_r: int
) -> tuple[np.ndarray, int]:
    """
    Repara las zonas de cuadrícula detectadas con el algoritmo TELEA
    (Fast Marching Method): reconstruye los píxeles interpolando desde
    el borde de la región, ponderando por distancia y dirección de gradiente.

    La máscara se dilata ligeramente (3px) para cubrir los bordes del trazo.
    """
    kd = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid_dilated = cv2.dilate(grid_mask, kd, iterations=2)
    n_px = int(grid_dilated.sum() // 255)
    if n_px > 0:
        img_out = cv2.inpaint(
            img, grid_dilated, inpaintRadius=inpaint_r, flags=cv2.INPAINT_TELEA
        )
    else:
        img_out = img
    return img_out, n_px


# ─────────────────────────────────────────────────────────────────────────────
# PASO 5 — LIMPIEZA DE RUIDO POR COMPONENTES CONECTADOS
# ─────────────────────────────────────────────────────────────────────────────


def remove_noise_components(
    ink: np.ndarray,
    area_small: int,
    area_medium: int,
    influence_radius: int,
    density_thresh: float,
) -> tuple[np.ndarray, dict]:
    """
    Elimina ruido puntual inspeccionando cada componente conectado (blob)
    individualmente, sin tocar nunca los trazos grandes del dibujo.

    Criterios:
      PEQUEÑOS (área < area_small, aislados):
        → Eliminados directamente. Ningún trazo de bolígrafo tiene solo
          5-15px de área; son 100% ruido de escáner o textura de papel.

      MEDIANOS (area_small ≤ área < area_medium, aislados):
        → Se analiza la forma: density = area / (w × h).
          Blob compacto y cuadrado (density > thresh, aspect ratio ~1) = ruido.
          Alargado o irregular (fragmento de trazo) = conservar.

      GRANDES (área ≥ area_medium) o CERCANOS a grandes:
        → Nunca eliminados. La zona de influencia (influence_radius px
          alrededor de cada componente grande) protege también los
          extremos de trazo y detalles finos contiguos.

    Retorna (ink limpia, estadísticas).
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]

    # Zona de influencia de trazos grandes
    big_mask = np.zeros_like(ink)
    for i in range(len(areas)):
        if areas[i] >= area_medium:
            big_mask[labels == i + 1] = 255
    r = influence_radius * 2 + 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    big_zone = cv2.dilate(big_mask, k)

    clean_ink = ink.copy()
    stats_out = {"removed_small": 0, "removed_medium": 0, "kept_medium": 0}

    for i in range(len(areas)):
        area = areas[i]
        w = widths[i]
        h = heights[i]
        comp = labels == i + 1
        near_big = big_zone[comp].any()

        if area < area_small and not near_big:
            clean_ink[comp] = 0
            stats_out["removed_small"] += 1

        elif area_small <= area < area_medium and not near_big:
            density = area / max(w * h, 1)
            ar = w / max(h, 1)
            if density > density_thresh and 0.5 < ar < 2.0:
                clean_ink[comp] = 0
                stats_out["removed_medium"] += 1
            else:
                stats_out["kept_medium"] += 1

    return clean_ink, stats_out


# ─────────────────────────────────────────────────────────────────────────────
# DEBUG
# ─────────────────────────────────────────────────────────────────────────────


def save_debug(
    output_path: str,
    img_orig: np.ndarray,
    color_mask: np.ndarray,
    synthetic: np.ndarray,
    mask_rich: np.ndarray,
    grid_mask: np.ndarray,
    mask_final: np.ndarray,
    result: np.ndarray,
) -> None:
    debug_dir = str(Path(output_path).parent / "debug_output")
    os.makedirs(debug_dir, exist_ok=True)

    def to3(im: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im

    h, w = img_orig.shape[:2]
    s = 0.25
    nh, nw = int(h * s), int(w * s)

    items = [
        ("1_original", img_orig),
        ("2_color_mask", color_mask),
        ("3_synthetic", synthetic),
        ("4_mask_raw", mask_rich),
        ("5_grid_mask", grid_mask),
        ("6_mask_final", mask_final),
        ("7_resultado", result),
    ]

    strips = []
    for name, im in items:
        strip = cv2.resize(to3(im), (nw, nh)).copy()
        cv2.putText(strip, name, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220), 2)
        strips.append(strip)
        _imwrite(os.path.join(debug_dir, name + ".png"), to3(im))

    # Comparación en dos filas de 4
    row1 = np.hstack(strips[:4])
    row2 = np.hstack(strips[4:] + [np.zeros_like(strips[0])])  # pad
    comparison = np.vstack([row1, row2])
    _imwrite(os.path.join(debug_dir, "comparison.png"), comparison)
    print(f"  → Debug en: {debug_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────


def digitize(
    input_path: str,
    output_path: str,
    alpha: int = DEFAULT_ALPHA,
    block: int = DEFAULT_BLOCK,
    c_offset: int = DEFAULT_C_OFFSET,
    grid_kernel: int = DEFAULT_GRID_KERNEL,
    inpaint_r: int = DEFAULT_INPAINT_R,
    noise_small: int = DEFAULT_NOISE_SMALL,
    noise_medium: int = DEFAULT_NOISE_MEDIUM,
    noise_radius: int = DEFAULT_NOISE_RADIUS,
    noise_density: float = DEFAULT_NOISE_DENSITY,
    skip_inpaint: bool = False,
    skip_color: bool = False,
    skip_denoise: bool = False,
    debug: bool = False,
) -> None:

    # ── 0. Carga ─────────────────────────────────────────────────────────────
    print("\n[0/6] Cargando imagen...")
    img = load_image(input_path)
    h, w = img.shape[:2]
    print(f"  {w}×{h} px")
    analyze_image(img)

    # ── 1. Tintas de color ───────────────────────────────────────────────────
    print("\n[1/6] Detectando tintas de color...")
    if skip_color:
        color_mask = np.zeros((h, w), dtype=np.uint8)
        print("  Omitido (--skip-color)")
    else:
        color_mask = detect_color_inks(img)
        n_color = int(color_mask.sum() // 255)
        if n_color == 0:
            print("  No se detectaron tintas de color")

    # ── 2. Canal sintético + máscara inicial ─────────────────────────────────
    print(f"\n[2/6] Canal sintético (alpha={alpha}) + máscara adaptativa...")
    synthetic = build_synthetic_channel(img, alpha)
    mask_raw = extract_mask(synthetic, block, c_offset)
    print(f"  Píxeles de tinta detectados: {int(mask_raw.sum()//255):,}")

    # ── 3. Cuadrícula residual + inpainting ──────────────────────────────────
    print("\n[3/6] Cuadrícula residual + inpainting...")
    grid_mask = detect_grid_residual(mask_raw, grid_kernel)

    if skip_inpaint:
        img_clean = img
        print("  Inpainting omitido (--skip-inpaint)")
    else:
        img_clean, n_grid = inpaint_grid(img, grid_mask, inpaint_r)
        if n_grid > 0:
            print(f"  Grid reparado: {n_grid:,} px")
        else:
            print("  Sin cuadrícula residual significativa")

    # ── 4. Re-extraer máscara sobre imagen limpia ─────────────────────────────
    print("\n[4/6] Re-extrayendo máscara sobre imagen reparada...")
    synthetic2 = build_synthetic_channel(img_clean, alpha)
    mask_refined = extract_mask(synthetic2, block, c_offset)

    # Apertura morfológica mínima (elimina píxeles sueltos sin romper trazos)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_OPEN, k_open)

    # ── 5. Fusión con tintas de color ─────────────────────────────────────────
    print("\n[5/6] Fusionando dibujo + tintas de color...")
    mask_combined = cv2.bitwise_or(mask_refined, color_mask)
    n_added = int(
        cv2.bitwise_and(color_mask, cv2.bitwise_not(mask_refined)).sum() // 255
    )
    print(f"  Píxeles de color añadidos al dibujo: {n_added:,}")

    # ── 6. Limpieza de ruido ──────────────────────────────────────────────────
    print("\n[6/6] Limpieza de ruido por componentes conectados...")
    if skip_denoise:
        mask_final = mask_combined
        print("  Omitida (--skip-denoise)")
    else:
        mask_final, st = remove_noise_components(
            mask_combined, noise_small, noise_medium, noise_radius, noise_density
        )
        print(
            f"  Eliminados pequeños (<{noise_small}px aislados):     {st['removed_small']:,}"
        )
        print(
            f"  Eliminados medianos compactos ({noise_small}-{noise_medium}px):  {st['removed_medium']:,}"
        )
        print(f"  Medianos conservados (posible detalle):  {st['kept_medium']:,}")

    # ── Resultado final ────────────────────────────────────────────────────────
    result = 255 - mask_final
    os.makedirs(str(Path(output_path).parent), exist_ok=True)
    _imwrite(output_path, result)

    n_ink = int(mask_final.sum() // 255)
    n_paper = mask_final.size - n_ink
    pct = n_ink / mask_final.size * 100
    print(f"\n✓ Guardado: {output_path}")
    print(f"  Tinta:  {n_ink:,} px ({pct:.1f}%)")
    print(f"  Papel:  {n_paper:,} px ({100-pct:.1f}%)")

    if debug:
        print("\n[DEBUG] Guardando imágenes de diagnóstico...")
        save_debug(
            output_path,
            img,
            color_mask,
            synthetic,
            mask_raw,
            grid_mask,
            mask_final,
            result,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        prog="digitize_notebook",
        description="Digitaliza dibujos de libreta de cuadrícula azul → imagen B&N limpia.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ejemplos:
  %(prog)s -i escan.png -o digital.png
  %(prog)s -i escan.png -o digital.png --debug
  %(prog)s -i escan.png -o digital.png --alpha 5 --c-offset 7
  %(prog)s -i escan.png -o digital.png --skip-inpaint --skip-color

ajuste rápido:
  queda cuadrícula     →  --alpha 5  o  --c-offset 7
  faltan trazos        →  --alpha 3  o  --c-offset 3
  demasiado ruido      →  --noise-area-small 30
  se pierden detalles  →  --noise-area-small 10
        """,
    )

    io = p.add_argument_group("entrada / salida")
    io.add_argument(
        "-i",
        "--input",
        required=True,
        metavar="RUTA",
        help="Imagen de entrada (PNG, JPG, TIFF...)",
    )
    io.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="RUTA",
        help="Imagen de salida (recomendado PNG)",
    )

    synth = p.add_argument_group("canal sintético (supresión de cuadrícula azul)")
    synth.add_argument(
        "--alpha",
        type=int,
        default=DEFAULT_ALPHA,
        metavar="N",
        help=f"Fuerza de supresión azul [default: {DEFAULT_ALPHA}]",
    )
    synth.add_argument(
        "--block",
        type=int,
        default=DEFAULT_BLOCK,
        metavar="N",
        help=f"Bloque umbral adaptativo, impar [default: {DEFAULT_BLOCK}]",
    )
    synth.add_argument(
        "--c-offset",
        type=int,
        default=DEFAULT_C_OFFSET,
        metavar="N",
        help=f"Offset umbral adaptativo [default: {DEFAULT_C_OFFSET}]",
    )

    grid = p.add_argument_group("eliminación de cuadrícula residual")
    grid.add_argument(
        "--grid-kernel",
        type=int,
        default=DEFAULT_GRID_KERNEL,
        metavar="N",
        help=f"Longitud mínima de línea recta (px) [default: {DEFAULT_GRID_KERNEL}]",
    )
    grid.add_argument(
        "--inpaint-r",
        type=int,
        default=DEFAULT_INPAINT_R,
        metavar="N",
        help=f"Radio inpainting TELEA [default: {DEFAULT_INPAINT_R}]",
    )

    noise = p.add_argument_group("limpieza de ruido por componentes")
    noise.add_argument(
        "--noise-area-small",
        type=int,
        default=DEFAULT_NOISE_SMALL,
        metavar="N",
        help=f"Área máxima de componente aislado eliminado sin análisis [default: {DEFAULT_NOISE_SMALL}]",
    )
    noise.add_argument(
        "--noise-area-medium",
        type=int,
        default=DEFAULT_NOISE_MEDIUM,
        metavar="N",
        help=f"Área límite para análisis de forma [default: {DEFAULT_NOISE_MEDIUM}]",
    )
    noise.add_argument(
        "--noise-radius",
        type=int,
        default=DEFAULT_NOISE_RADIUS,
        metavar="N",
        help=f"Radio de influencia de trazos grandes (px) [default: {DEFAULT_NOISE_RADIUS}]",
    )
    noise.add_argument(
        "--noise-density",
        type=float,
        default=DEFAULT_NOISE_DENSITY,
        metavar="F",
        help=f"Densidad mínima para clasificar mediano como blob=ruido [default: {DEFAULT_NOISE_DENSITY}]",
    )

    flags = p.add_argument_group("flags opcionales")
    flags.add_argument(
        "--skip-inpaint",
        action="store_true",
        help="Omitir inpainting de cuadrícula residual (más rápido)",
    )
    flags.add_argument(
        "--skip-color", action="store_true", help="Omitir detección de tintas de color"
    )
    flags.add_argument(
        "--skip-denoise",
        action="store_true",
        help="Omitir limpieza de ruido por componentes",
    )
    flags.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Guardar imágenes de diagnóstico en debug_output/",
    )

    args = p.parse_args()

    # Validaciones
    if args.block % 2 == 0:
        p.error("--block debe ser impar")
    if args.block < 3:
        p.error("--block debe ser ≥ 3")

    try:
        digitize(
            input_path=args.input,
            output_path=args.output,
            alpha=args.alpha,
            block=args.block,
            c_offset=args.c_offset,
            grid_kernel=args.grid_kernel,
            inpaint_r=args.inpaint_r,
            noise_small=args.noise_area_small,
            noise_medium=args.noise_area_medium,
            noise_radius=args.noise_radius,
            noise_density=args.noise_density,
            skip_inpaint=args.skip_inpaint,
            skip_color=args.skip_color,
            skip_denoise=args.skip_denoise,
            debug=args.debug,
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR inesperado] {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
