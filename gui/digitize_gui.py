#!/usr/bin/env python3
"""
digitize_gui.py — v5
=====================
GUI profesional para digitalizar dibujos de libreta de cuadrícula azul.

Mejoras v5:
  · Fix encoding: carga y guarda imágenes con rutas Unicode (tildes, espacios,
    caracteres especiales) usando np.fromfile/cv2.imdecode y buf.tofile.

Mejoras v4:
  · Tema oscuro moderno con ttkbootstrap (darkly)
  · Tooltips robustos con after-delay, nunca se quedan flotando
  · Panel de parámetros con secciones colapsables y orden lógico
  · Toolbar con iconos consistentes
  · Indicador de progreso integrado en la barra de estado
  · Canvas con crosshair dinámico que muestra tamaño del pincel
  · Vista comparada lado a lado (opcional)
  · Correcciones de todos los bugs de la v3

Requisitos:
  pip install opencv-python-headless Pillow numpy ttkbootstrap
  sudo apt install python3-tk  (Linux)

Uso:
  python digitize_gui.py
  python digitize_gui.py -i escan.png
"""

import argparse
import math
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

import cv2
import numpy as np
from PIL import Image, ImageTk


from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS DE I/O SEGUROS CON UNICODE
# cv2.imread / cv2.imwrite fallan silenciosamente con rutas que contienen
# tildes, espacios o caracteres no-ASCII. Se leen/escriben los bytes
# directamente y se delega la codificación/decodificación a OpenCV.
# ══════════════════════════════════════════════════════════════════════════════

def _imread(path: str) -> np.ndarray | None:
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _imwrite(path: str, img: np.ndarray) -> bool:
    ext = Path(path).suffix.lower()
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# COLORES Y CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

C = {
    "bg":        "#1a1d23",
    "panel":     "#22262e",
    "card":      "#2a2f3a",
    "border":    "#363c4a",
    "accent":    "#4a90d9",
    "accent2":   "#5ba85e",
    "warn":      "#e8a838",
    "danger":    "#d9534f",
    "text":      "#e0e4ed",
    "text_dim":  "#7b8399",
    "text_hint": "#4f5568",
    "slider_bg": "#363c4a",
}

FONT_UI    = ("Segoe UI", 9)
FONT_LABEL = ("Segoe UI", 9)
FONT_SMALL = ("Segoe UI", 8)
FONT_MONO  = ("Consolas", 9)
FONT_TITLE = ("Segoe UI", 10, "bold")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE (idéntico al digitize_notebook.py v3)
# ══════════════════════════════════════════════════════════════════════════════

COLOR_INK_RANGES = [
    ("rojo",     0,   12,  40, 40, 230, 15, 15),
    ("rojo-mag", 155, 180, 40, 40, 230, 15, 15),
]

def detect_color_inks(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b_ch, g_ch, r_ch = cv2.split(img)
    combined = np.zeros(img.shape[:2], dtype=np.uint8)
    for (name, h_lo, h_hi, s_lo, v_lo, v_hi, rg_min, rb_min) in COLOR_INK_RANGES:
        hsv_mask = cv2.inRange(hsv, np.array([h_lo, s_lo, v_lo]),
                                    np.array([h_hi, 255,  v_hi]))
        rgb_mask = ((r_ch.astype(int) - g_ch.astype(int) > rg_min) &
                    (r_ch.astype(int) - b_ch.astype(int) > rb_min)
                   ).astype(np.uint8) * 255
        layer = cv2.bitwise_and(hsv_mask, rgb_mask)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        layer = cv2.dilate(layer, k, iterations=1)
        combined = cv2.bitwise_or(combined, layer)
    return combined

def build_synthetic_channel(img, alpha):
    b, g, r = cv2.split(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    be = b.astype(np.float32) - r.astype(np.float32)
    return np.clip(gray.astype(np.float32) + alpha * np.clip(be, 0, None), 0, 255).astype(np.uint8)

def extract_mask(synthetic, block, c_offset):
    blur = cv2.GaussianBlur(synthetic, (5, 5), 0)
    return cv2.adaptiveThreshold(blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, c_offset)

def detect_grid_residual(mask, grid_kernel):
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (grid_kernel, 1))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, grid_kernel))
    return cv2.add(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kh),
                   cv2.morphologyEx(mask, cv2.MORPH_OPEN, kv))

def inpaint_grid(img, grid_mask, inpaint_r):
    kd = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gd = cv2.dilate(grid_mask, kd, iterations=2)
    n  = int(gd.sum() // 255)
    return (cv2.inpaint(img, gd, inpaintRadius=inpaint_r, flags=cv2.INPAINT_TELEA) if n > 0 else img), n

def remove_noise_components(ink, area_small, area_medium, influence_radius, density_thresh):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    areas   = stats[1:, cv2.CC_STAT_AREA]
    widths  = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    big_mask = np.zeros_like(ink)
    for i in range(len(areas)):
        if areas[i] >= area_medium:
            big_mask[labels == i + 1] = 255
    r = influence_radius * 2 + 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    big_zone = cv2.dilate(big_mask, k)
    clean = ink.copy()
    for i in range(len(areas)):
        area = areas[i]; w = widths[i]; h = heights[i]
        comp = (labels == i + 1)
        near = big_zone[comp].any()
        if area < area_small and not near:
            clean[comp] = 0
        elif area_small <= area < area_medium and not near:
            density = area / max(w * h, 1)
            ar = w / max(h, 1)
            if density > density_thresh and 0.5 < ar < 2.0:
                clean[comp] = 0
    return clean

def run_pipeline(img, params, progress_cb=None):
    def p(msg, pct):
        if progress_cb:
            progress_cb(msg, pct)

    p("Detectando tintas de color…", 5)
    color_mask = detect_color_inks(img) if not params["skip_color"] else \
                 np.zeros(img.shape[:2], dtype=np.uint8)

    p("Canal sintético…", 20)
    synthetic = build_synthetic_channel(img, params["alpha"])
    mask_raw  = extract_mask(synthetic, params["block"], params["c_offset"])

    p("Cuadrícula residual…", 40)
    grid_mask = detect_grid_residual(mask_raw, params["grid_kernel"])

    if not params["skip_inpaint"]:
        p("Inpainting…", 55)
        img_clean, _ = inpaint_grid(img, grid_mask, params["inpaint_r"])
    else:
        img_clean = img

    p("Re-extrayendo máscara…", 68)
    synthetic2   = build_synthetic_channel(img_clean, params["alpha"])
    mask_refined = extract_mask(synthetic2, params["block"], params["c_offset"])
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_OPEN, k_open)

    p("Fusionando colores…", 78)
    mask_combined = cv2.bitwise_or(mask_refined, color_mask)

    if not params["skip_denoise"]:
        p("Eliminando ruido…", 88)
        mask_combined = remove_noise_components(
            mask_combined,
            params["noise_small"], params["noise_medium"],
            params["noise_radius"], params["noise_density"])

    p("Listo", 100)
    return 255 - mask_combined


# ══════════════════════════════════════════════════════════════════════════════
# TOOLTIP ROBUSTO  (never gets stuck)
# ══════════════════════════════════════════════════════════════════════════════

class Tooltip:
    """
    Tooltip con delay configurable que se destruye correctamente siempre.
    Usa after() para mostrar y cancela el after() en Leave → nunca se queda.
    """
    _active: "Tooltip | None" = None

    def __init__(self, widget, text, delay=600):
        self._widget  = widget
        self._text    = text
        self._delay   = delay
        self._win     = None
        self._job     = None
        widget.bind("<Enter>",    self._on_enter,  add="+")
        widget.bind("<Leave>",    self._on_leave,  add="+")
        widget.bind("<Button>",   self._on_leave,  add="+")
        widget.bind("<Destroy>",  self._on_destroy, add="+")

    def _on_enter(self, e):
        self._cancel()
        self._job = self._widget.after(self._delay, self._show)

    def _on_leave(self, e=None):
        self._cancel()
        self._hide()

    def _on_destroy(self, e=None):
        self._cancel()
        self._hide()

    def _cancel(self):
        if self._job:
            try:
                self._widget.after_cancel(self._job)
            except Exception:
                pass
            self._job = None

    def _show(self):
        if self._win:
            return
        x = self._widget.winfo_rootx() + 8
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._win = tk.Toplevel(self._widget)
        self._win.wm_overrideredirect(True)
        self._win.wm_geometry(f"+{x}+{y}")
        self._win.attributes("-topmost", True)
        lbl = tk.Label(
            self._win, text=self._text,
            background="#2a2f3a", foreground="#e0e4ed",
            relief="flat", bd=0,
            font=("Segoe UI", 9),
            wraplength=280, justify="left",
            padx=8, pady=5)
        lbl.pack()
        # borde sutil
        self._win.configure(bg="#4a90d9")
        outer = tk.Frame(self._win, bg="#4a90d9", bd=1)
        outer.place(x=0, y=0, relwidth=1, relheight=1)

    def _hide(self):
        if self._win:
            try:
                self._win.destroy()
            except Exception:
                pass
            self._win = None


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN COLAPSABLE
# ══════════════════════════════════════════════════════════════════════════════

class CollapsibleSection(tk.Frame):
    def __init__(self, parent, title, expanded=True, **kw):
        super().__init__(parent, bg=C["panel"], **kw)
        self._expanded = expanded

        # Cabecera clicable
        hdr = tk.Frame(self, bg=C["card"], cursor="hand2")
        hdr.pack(fill="x", pady=(0, 0))

        self._arrow = tk.Label(hdr, text="▾" if expanded else "▸",
                               bg=C["card"], fg=C["accent"],
                               font=("Segoe UI", 10, "bold"))
        self._arrow.pack(side="left", padx=(8, 4), pady=4)

        tk.Label(hdr, text=title, bg=C["card"], fg=C["text"],
                 font=FONT_TITLE, anchor="w"
                 ).pack(side="left", pady=4)

        hdr.bind("<Button-1>", self._toggle)
        self._arrow.bind("<Button-1>", self._toggle)

        # Contenedor del contenido
        self.body = tk.Frame(self, bg=C["panel"])
        if expanded:
            self.body.pack(fill="x", padx=4, pady=(2, 6))

    def _toggle(self, e=None):
        self._expanded = not self._expanded
        self._arrow.configure(text="▾" if self._expanded else "▸")
        if self._expanded:
            self.body.pack(fill="x", padx=4, pady=(2, 6))
        else:
            self.body.pack_forget()


# ══════════════════════════════════════════════════════════════════════════════
# FILA DE PARÁMETRO
# ══════════════════════════════════════════════════════════════════════════════

class ParamRow(tk.Frame):
    """Label + Scale + Spinbox en una fila, con tooltip robusto."""

    def __init__(self, parent, label, var, from_, to,
                 resolution=1, fmt=None, tooltip=None, **kw):
        super().__init__(parent, bg=C["panel"], **kw)
        self._var = var
        self._fmt = fmt or ("{:.2f}" if resolution < 1 else "{:d}")

        # Label
        lbl = tk.Label(self, text=label, bg=C["panel"], fg=C["text_dim"],
                       font=FONT_LABEL, anchor="w", width=13)
        lbl.grid(row=0, column=0, sticky="w", padx=(4, 0))

        # Slider
        self._slider = tk.Scale(
            self, variable=var, from_=from_, to=to,
            resolution=resolution, orient=tk.HORIZONTAL,
            length=110, showvalue=False, bd=0,
            bg=C["panel"], fg=C["text"],
            troughcolor=C["slider_bg"],
            activebackground=C["accent"],
            highlightthickness=0, sliderlength=14,
            relief="flat")
        self._slider.grid(row=0, column=1, padx=(2, 2))

        # Spinbox
        vcmd = (self.register(self._validate), "%P")
        spin_kw = dict(
            textvariable=var, from_=from_, to=to,
            increment=resolution, width=7,
            font=FONT_MONO,
            bg=C["card"], fg=C["text"],
            insertbackground=C["accent"],
            buttonbackground=C["border"],
            relief="flat", bd=1,
            highlightthickness=1,
            highlightbackground=C["border"],
            highlightcolor=C["accent"],
            validate="key", validatecommand=vcmd)
        if resolution < 1:
            spin_kw["format"] = "%.2f"
        self._spin = tk.Spinbox(self, **spin_kw)
        self._spin.grid(row=0, column=2, padx=(0, 4))

        if tooltip:
            for w in (self, lbl, self._slider):
                Tooltip(w, tooltip)

    def _validate(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return value == "" or value == "-"


# ══════════════════════════════════════════════════════════════════════════════
# PANEL DE PARÁMETROS
# ══════════════════════════════════════════════════════════════════════════════

class PanelParams(tk.Frame):
    def __init__(self, parent, app, **kw):
        super().__init__(parent, bg=C["panel"], **kw)
        self.app = app
        self._build()

    def _row(self, parent, label, var, from_, to, res=1, tip=None):
        r = ParamRow(parent, label, var, from_, to, res, tooltip=tip)
        r.pack(fill="x", pady=2)
        return r

    def _sep(self, parent):
        f = tk.Frame(parent, bg=C["border"], height=1)
        f.pack(fill="x", pady=4)

    def _build(self):
        a = self.app

        # ── 1. Canal sintético ───────────────────────────────────────────────
        s1 = CollapsibleSection(self, "① Canal sintético", expanded=True)
        s1.pack(fill="x", pady=(0, 2))
        self._row(s1.body, "alpha", a.p_alpha, 1, 8,
                  tip="Fuerza de supresión de la cuadrícula azul.\n"
                      "↑ Más cuadrícula eliminada  |  ↓ Más trazos recuperados")
        self._row(s1.body, "c_offset", a.p_c_offset, 1, 20,
                  tip="Offset del umbral adaptativo.\n"
                      "↑ Más selectivo (menos capturas, riesgo de perder trazos)\n"
                      "↓ Más permisivo (más capturas, riesgo de capturar cuadrícula)")
        self._row(s1.body, "block", a.p_block, 3, 51, res=2,
                  tip="Tamaño del vecindario adaptativo (siempre impar).\n"
                      "Mayor → menos sensible a variaciones locales de iluminación.")

        # ── 2. Cuadrícula residual ───────────────────────────────────────────
        s2 = CollapsibleSection(self, "② Cuadrícula residual", expanded=True)
        s2.pack(fill="x", pady=(0, 2))
        self._row(s2.body, "grid_kernel", a.p_grid_kernel, 30, 300, res=5,
                  tip="Longitud mínima (px) de línea recta para detectarla como cuadrícula.\n"
                      "↓ Detecta líneas más cortas (más agresivo)")
        self._row(s2.body, "inpaint_r", a.p_inpaint_r, 1, 10,
                  tip="Radio del inpainting TELEA.\n"
                      "Mayor → más suavizado al rellenar zonas reparadas.")

        # ── 3. Ruido ─────────────────────────────────────────────────────────
        s3 = CollapsibleSection(self, "③ Limpieza de ruido", expanded=True)
        s3.pack(fill="x", pady=(0, 2))
        self._row(s3.body, "noise_small", a.p_noise_small, 4, 80,
                  tip="Área máxima (px²) de un componente aislado para eliminarlo directamente.\n"
                      "↑ Elimina puntos más grandes  |  ↓ Conserva más detalles finos")
        self._row(s3.body, "noise_medium", a.p_noise_medium, 20, 400, res=5,
                  tip="Área límite para análisis de forma de componentes medianos.\n"
                      "Los medianos se analizan por densidad antes de eliminarse.")
        self._row(s3.body, "noise_radius", a.p_noise_radius, 5, 100, res=5,
                  tip="Radio (px) de protección alrededor de trazos grandes.\n"
                      "Los componentes pequeños dentro de este radio no se eliminan.")
        self._row(s3.body, "noise_density", a.p_noise_density, 0.30, 0.99, res=0.05,
                  tip="Densidad mínima (area / bbox) para clasificar un componente mediano como ruido.\n"
                      "Un blob compacto y cuadrado tiene densidad alta → ruido.\n"
                      "Un fragmento de trazo tiene densidad baja → conservar.")

        # ── 4. Opciones ───────────────────────────────────────────────────────
        s4 = CollapsibleSection(self, "④ Opciones", expanded=False)
        s4.pack(fill="x", pady=(0, 2))

        checks = [
            (a.p_skip_inpaint, "Omitir inpainting",
             "Más rápido. No repara la cuadrícula residual con TELEA."),
            (a.p_skip_color,   "Omitir detección de color",
             "No detecta tintas de colores (rojo, verde…). Solo negro/azul."),
            (a.p_skip_denoise, "Omitir denoising",
             "No elimina puntitos de ruido. Resultado más rápido pero más sucio."),
        ]
        for var, text, tip in checks:
            f = tk.Frame(s4.body, bg=C["panel"])
            f.pack(fill="x", pady=2)
            cb = tk.Checkbutton(f, text=text, variable=var,
                                bg=C["panel"], fg=C["text"],
                                selectcolor=C["card"],
                                activebackground=C["panel"],
                                activeforeground=C["text"],
                                font=FONT_LABEL,
                                highlightthickness=0)
            cb.pack(side="left", padx=6)
            Tooltip(cb, tip)

        # Auto-procesar
        self._sep(self)
        f2 = tk.Frame(self, bg=C["panel"])
        f2.pack(fill="x", padx=8, pady=4)
        cb2 = tk.Checkbutton(f2, text="Auto-procesar al cambiar parámetros",
                             variable=a.p_auto,
                             bg=C["panel"], fg=C["text_dim"],
                             selectcolor=C["card"],
                             activebackground=C["panel"],
                             font=FONT_SMALL,
                             highlightthickness=0)
        cb2.pack(side="left")
        Tooltip(cb2, "Reprocesa automáticamente 0.8s después de cambiar cualquier parámetro.\n"
                     "Puede ser lento en imágenes grandes.")


# ══════════════════════════════════════════════════════════════════════════════
# CANVAS CON ZOOM, SCROLL Y DIBUJO
# ══════════════════════════════════════════════════════════════════════════════

class DrawingCanvas(tk.Frame):
    MIN_ZOOM = 0.05
    MAX_ZOOM = 5.0

    def __init__(self, parent, app, **kw):
        super().__init__(parent, bg=C["bg"], **kw)
        self.app = app

        self.canvas = tk.Canvas(self, bg="#111418",
                                cursor="crosshair", highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical",   command=self.canvas.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._zoom     = 0.45
        self._img_id   = None
        self._cursor_id = None
        self._tk_img   = None
        self._pan_start = None
        self._last_draw  = None
        self._draw_changed = False

        # ── Eventos ───────────────────────────────────────────────────────────
        c = self.canvas
        c.bind("<ButtonPress-1>",   self._draw_start)
        c.bind("<B1-Motion>",       self._draw_move)
        c.bind("<ButtonRelease-1>", self._draw_end)
        c.bind("<ButtonPress-2>",   self._pan_start_ev)
        c.bind("<B2-Motion>",       self._pan_move)
        c.bind("<ButtonRelease-2>", self._pan_end)
        c.bind("<ButtonPress-3>",   self._pan_start_ev)
        c.bind("<B3-Motion>",       self._pan_move)
        c.bind("<ButtonRelease-3>", self._pan_end)
        c.bind("<Motion>",          self._on_mouse_move)
        c.bind("<Leave>",           self._on_mouse_leave)
        # Zoom: Ctrl+scroll (Linux y Windows)
        c.bind("<Control-MouseWheel>", self._zoom_win)
        c.bind("<Control-Button-4>",   lambda e: self._zoom_step(1.15))
        c.bind("<Control-Button-5>",   lambda e: self._zoom_step(1/1.15))
        # Scroll normal
        c.bind("<MouseWheel>", self._scroll_win)
        c.bind("<Button-4>",   lambda e: c.yview_scroll(-3, "units"))
        c.bind("<Button-5>",   lambda e: c.yview_scroll(3, "units"))

    # ── Render ────────────────────────────────────────────────────────────────

    def refresh(self):
        if self.app.img_original is None:
            return
        pil = self._build_display_image()
        w = max(1, int(pil.width  * self._zoom))
        h = max(1, int(pil.height * self._zoom))
        method = Image.LANCZOS if self._zoom < 1 else Image.NEAREST
        resized = pil.resize((w, h), method)

        # Capa de pintura encima
        overlay = self._paint_overlay(w, h)
        if overlay:
            resized = Image.alpha_composite(resized.convert("RGBA"), overlay).convert("RGB")

        self._tk_img = ImageTk.PhotoImage(resized)
        if self._img_id is None:
            self._img_id = self.canvas.create_image(0, 0, anchor="nw",
                                                     image=self._tk_img)
        else:
            self.canvas.itemconfig(self._img_id, image=self._tk_img)
        self.canvas.configure(scrollregion=(0, 0, w, h))
        self.app.set_status_zoom(self._zoom)

    def _build_display_image(self):
        view = self.app.view_mode.get()
        orig = self.app.img_original
        res  = self.app.img_result

        if view == "original":
            return Image.fromarray(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))

        if res is None:
            return Image.fromarray(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))

        if view == "procesado":
            return Image.fromarray(res)

        # overlay: resultado + original semitransparente
        alpha = self.app.overlay_alpha.get() / 100.0
        orig_f = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB).astype(np.float32)
        res3   = cv2.cvtColor(res,  cv2.COLOR_GRAY2RGB).astype(np.float32)
        blended = res3 * (1 - alpha) + orig_f * alpha
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    def _paint_overlay(self, dw, dh):
        pl = self.app.paint_layer
        if pl is None or not pl.any():
            return None
        oh, ow = pl.shape
        scaled = cv2.resize(pl, (dw, dh), interpolation=cv2.INTER_NEAREST)
        rgba = np.zeros((dh, dw, 4), dtype=np.uint8)
        rgba[scaled == 1]   = [0,   0,   0,   230]
        rgba[scaled == 255] = [255, 255, 255, 230]
        return Image.fromarray(rgba, "RGBA")

    # ── Cursor circular ───────────────────────────────────────────────────────

    def _draw_cursor(self, cx, cy):
        """Dibuja un círculo que representa el pincel en la posición del cursor."""
        if self._cursor_id:
            self.canvas.delete(self._cursor_id)
        r = max(1, int(self.app.brush_size.get() * self._zoom))
        tool = self.app.tool.get()
        color = "#333" if tool == "negro" else "#fff" if tool == "blanco" else "#e88"
        outline = "#aaa"
        self._cursor_id = self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline=outline, fill=color, stipple="gray50",
            width=1, tags="cursor")
        self.canvas.tag_raise("cursor")

    def _on_mouse_move(self, e):
        self._draw_cursor(e.x, e.y)

    def _on_mouse_leave(self, e):
        if self._cursor_id:
            self.canvas.delete(self._cursor_id)
            self._cursor_id = None

    # ── Zoom ──────────────────────────────────────────────────────────────────

    def _zoom_step(self, factor):
        old = self._zoom
        self._zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._zoom * factor))
        if self._zoom != old:
            self.refresh()

    def _zoom_win(self, e):
        self._zoom_step(1.15 if e.delta > 0 else 1/1.15)

    # ── Scroll ────────────────────────────────────────────────────────────────

    def _scroll_win(self, e):
        self.canvas.yview_scroll(-1 if e.delta > 0 else 1, "units")

    # ── Pan ───────────────────────────────────────────────────────────────────

    def _pan_start_ev(self, e):
        self._pan_start = (e.x, e.y)
        self.canvas.configure(cursor="fleur")

    def _pan_move(self, e):
        if self._pan_start:
            dx = self._pan_start[0] - e.x
            dy = self._pan_start[1] - e.y
            self._pan_start = (e.x, e.y)
            self.canvas.xview_scroll(int(dx), "units")
            self.canvas.yview_scroll(int(dy), "units")

    def _pan_end(self, e):
        self._pan_start = None
        self.canvas.configure(cursor="crosshair")

    # ── Dibujo ────────────────────────────────────────────────────────────────

    def _to_orig(self, cx, cy):
        x = int(self.canvas.canvasx(cx) / self._zoom)
        y = int(self.canvas.canvasy(cy) / self._zoom)
        return x, y

    def _draw_start(self, e):
        if self.app.img_original is None:
            return
        self.app.save_undo()
        x, y = self._to_orig(e.x, e.y)
        self._paint_point(x, y)
        self._last_draw = (x, y)
        self._draw_changed = True

    def _draw_move(self, e):
        if self.app.img_original is None or self._last_draw is None:
            return
        x, y   = self._to_orig(e.x, e.y)
        x0, y0 = self._last_draw
        dist    = math.hypot(x - x0, y - y0)
        steps   = max(1, int(dist))
        for i in range(steps + 1):
            t = i / steps
            self._paint_point(int(x0 + (x - x0) * t), int(y0 + (y - y0) * t))
        self._last_draw = (x, y)
        self._draw_cursor(e.x, e.y)
        self.refresh()

    def _draw_end(self, e):
        self._last_draw = None
        if self._draw_changed:
            self.refresh()
            self._draw_changed = False

    def _paint_point(self, cx, cy):
        if self.app.paint_layer is None:
            return
        tool = self.app.tool.get()
        rad  = self.app.brush_size.get()
        h, w = self.app.paint_layer.shape
        y1 = max(0, cy - rad); y2 = min(h, cy + rad + 1)
        x1 = max(0, cx - rad); x2 = min(w, cx + rad + 1)
        yy, xx = np.ogrid[y1:y2, x1:x2]
        circle = (xx - cx)**2 + (yy - cy)**2 <= rad**2
        val = {"negro": 1, "blanco": 255, "borrador": 0}[tool]
        self.app.paint_layer[y1:y2, x1:x2][circle] = val

    # ── Utilidades públicas ───────────────────────────────────────────────────

    def set_zoom(self, z):
        self._zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, z))
        self.refresh()

    def fit_to_window(self):
        if self.app.img_original is None:
            return
        self.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        ih, iw = self.app.img_original.shape[:2]
        if cw > 1 and ch > 1:
            self._zoom = min(cw / iw, ch / ih) * 0.95
        self.refresh()


# ══════════════════════════════════════════════════════════════════════════════
# PANEL DE HERRAMIENTAS (derecho)
# ══════════════════════════════════════════════════════════════════════════════

class PanelTools(tk.Frame):
    def __init__(self, parent, app, **kw):
        super().__init__(parent, bg=C["panel"], **kw)
        self.app = app
        self._build()

    def _section_label(self, text):
        tk.Label(self, text=text, bg=C["card"], fg=C["text"],
                 font=FONT_TITLE, anchor="w"
                 ).pack(fill="x", padx=0, pady=(10, 3), ipady=4, ipadx=8)

    def _build(self):
        a = self.app

        # ── Herramientas ──────────────────────────────────────────────────────
        self._section_label("  Herramienta")
        tool_frame = tk.Frame(self, bg=C["panel"])
        tool_frame.pack(fill="x", padx=6, pady=2)

        TOOLS = [
            ("⬛", "Pincel negro",  "negro",    C["text"],
             "Añade tinta negra sobre el resultado.\nÚsalo para recuperar trazos perdidos."),
            ("⬜", "Pincel blanco", "blanco",   "#aaa",
             "Añade tinta blanca (borra tinta).\nÚsalo para eliminar cuadrícula o ruido restante."),
            ("◌", "Borrador",      "borrador",  C["accent"],
             "Borra la capa de pintura manual, volviendo al resultado del pipeline."),
        ]
        self._tool_btns = {}
        for icon, label, val, icon_color, tip in TOOLS:
            f = tk.Frame(tool_frame, bg=C["panel"])
            f.pack(fill="x", pady=1)
            btn = tk.Button(
                f, text=f" {icon}  {label}",
                command=lambda v=val: self._select_tool(v),
                bg=C["card"], fg=C["text"],
                activebackground=C["accent"],
                activeforeground="white",
                relief="flat", bd=0,
                font=("Segoe UI", 10),
                anchor="w", padx=10, pady=5,
                highlightthickness=1,
                highlightbackground=C["border"],
                highlightcolor=C["accent"])
            btn.pack(fill="x")
            Tooltip(btn, tip)
            self._tool_btns[val] = btn
        self._select_tool("negro")

        # ── Tamaño pincel ─────────────────────────────────────────────────────
        self._section_label("  Tamaño pincel")
        f_size = tk.Frame(self, bg=C["panel"])
        f_size.pack(fill="x", padx=6, pady=2)
        tk.Scale(
            f_size, variable=a.brush_size, from_=1, to=50,
            orient=tk.HORIZONTAL, length=140,
            bg=C["panel"], fg=C["text"],
            troughcolor=C["slider_bg"],
            activebackground=C["accent"],
            highlightthickness=0, sliderlength=14,
            showvalue=True, font=FONT_SMALL
        ).pack(fill="x")

        # ── Vista ─────────────────────────────────────────────────────────────
        self._section_label("  Vista")
        view_frame = tk.Frame(self, bg=C["panel"])
        view_frame.pack(fill="x", padx=6, pady=2)

        VIEWS = [
            ("Original",    "original",  "Muestra la imagen escaneada tal cual."),
            ("Procesado",   "procesado", "Muestra el resultado del pipeline (B&N limpio)."),
            ("Superpuesto", "overlay",   "Muestra el resultado con el original semitransparente\n"
                                         "encima para verificar qué trazos se perdieron."),
        ]
        self._view_btns = {}
        for text, val, tip in VIEWS:
            btn = tk.Button(
                view_frame, text=f"  {text}",
                command=lambda v=val: self._select_view(v),
                bg=C["card"], fg=C["text_dim"],
                activebackground=C["accent"],
                activeforeground="white",
                relief="flat", bd=0,
                font=("Segoe UI", 9),
                anchor="w", padx=8, pady=4,
                highlightthickness=1,
                highlightbackground=C["border"],
                highlightcolor=C["accent"])
            btn.pack(fill="x", pady=1)
            Tooltip(btn, tip)
            self._view_btns[val] = btn
        self._select_view("procesado")

        # Opacidad overlay
        f_ov = tk.Frame(self, bg=C["panel"])
        f_ov.pack(fill="x", padx=6, pady=(2, 0))
        tk.Label(f_ov, text="Opacidad original:", bg=C["panel"],
                 fg=C["text_hint"], font=FONT_SMALL
                 ).pack(anchor="w")
        tk.Scale(
            f_ov, variable=a.overlay_alpha, from_=0, to=80,
            orient=tk.HORIZONTAL, length=140,
            bg=C["panel"], fg=C["text"],
            troughcolor=C["slider_bg"],
            activebackground=C["accent"],
            highlightthickness=0, sliderlength=14,
            showvalue=True, font=FONT_SMALL,
            command=lambda _: a.canvas_view.refresh()
        ).pack(fill="x")

        # ── Edición ───────────────────────────────────────────────────────────
        self._section_label("  Edición")
        f_ed = tk.Frame(self, bg=C["panel"])
        f_ed.pack(fill="x", padx=6, pady=4)

        def mkbtn(parent, text, cmd, tip=None, full=False):
            b = tk.Button(parent, text=text, command=cmd,
                          bg=C["card"], fg=C["text"],
                          activebackground=C["border"],
                          activeforeground=C["text"],
                          relief="flat", bd=0,
                          font=("Segoe UI", 9),
                          padx=6, pady=4)
            if tip:
                Tooltip(b, tip)
            if full:
                b.pack(fill="x", pady=1)
            else:
                b.pack(side="left", padx=2, pady=1)
            return b

        mkbtn(f_ed, "↩ Deshacer", a.undo, "Deshacer último trazo  (Ctrl+Z)")
        mkbtn(f_ed, "↪ Rehacer",  a.redo, "Rehacer  (Ctrl+Y)")

        f_ed2 = tk.Frame(self, bg=C["panel"])
        f_ed2.pack(fill="x", padx=6)
        mkbtn(f_ed2, "✕  Limpiar capa de pintura", a.clear_paint,
              "Borra todas las correcciones manuales.", full=True)

        # ── Zoom rápido ───────────────────────────────────────────────────────
        self._section_label("  Zoom")
        f_z = tk.Frame(self, bg=C["panel"])
        f_z.pack(fill="x", padx=6, pady=4)
        for label, z in [("25%", 0.25), ("50%", 0.5), ("100%", 1.0), ("Ajustar", None)]:
            cmd = (lambda z=z: a.canvas_view.set_zoom(z)) if z else a.canvas_view.fit_to_window
            mkbtn(f_z, label, cmd)

    def _select_tool(self, val):
        self.app.tool.set(val)
        for v, btn in self._tool_btns.items():
            if v == val:
                btn.configure(bg=C["accent"], fg="white",
                              highlightbackground=C["accent"])
            else:
                btn.configure(bg=C["card"], fg=C["text"],
                              highlightbackground=C["border"])

    def _select_view(self, val):
        self.app.view_mode.set(val)
        self.app.canvas_view.refresh()
        for v, btn in self._view_btns.items():
            if v == val:
                btn.configure(bg=C["accent"], fg="white",
                              highlightbackground=C["accent"])
            else:
                btn.configure(bg=C["card"], fg=C["text_dim"],
                              highlightbackground=C["border"])


# ══════════════════════════════════════════════════════════════════════════════
# APLICACIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class App(ttk.Window):
    UNDO_LIMIT = 30

    def __init__(self, initial_image=None):
        super().__init__(themename="darkly")
        self.title("Digitalizador de Libretas  v4")
        self.geometry("1440x900")
        self.minsize(1000, 640)
        self.configure(bg=C["bg"])

        # ── Estado ────────────────────────────────────────────────────────────
        self.img_original: np.ndarray | None = None
        self.img_result:   np.ndarray | None = None
        self.paint_layer:  np.ndarray | None = None
        self._undo_stack = []
        self._redo_stack = []
        self._processing = False
        self._auto_job   = None

        # ── Variables Tk ──────────────────────────────────────────────────────
        self.p_alpha        = tk.IntVar(value=4)
        self.p_block        = tk.IntVar(value=25)
        self.p_c_offset     = tk.IntVar(value=5)
        self.p_grid_kernel  = tk.IntVar(value=120)
        self.p_inpaint_r    = tk.IntVar(value=4)
        self.p_noise_small  = tk.IntVar(value=20)
        self.p_noise_medium = tk.IntVar(value=100)
        self.p_noise_radius = tk.IntVar(value=25)
        self.p_noise_density = tk.DoubleVar(value=0.60)
        self.p_skip_inpaint = tk.BooleanVar(value=False)
        self.p_skip_color   = tk.BooleanVar(value=False)
        self.p_skip_denoise = tk.BooleanVar(value=False)
        self.p_auto         = tk.BooleanVar(value=False)
        self.tool           = tk.StringVar(value="negro")
        self.brush_size     = tk.IntVar(value=5)
        self.view_mode      = tk.StringVar(value="procesado")
        self.overlay_alpha  = tk.IntVar(value=35)

        # Trace para auto-procesar
        for v in (self.p_alpha, self.p_block, self.p_c_offset,
                  self.p_grid_kernel, self.p_inpaint_r,
                  self.p_noise_small, self.p_noise_medium,
                  self.p_noise_radius, self.p_noise_density,
                  self.p_skip_inpaint, self.p_skip_color, self.p_skip_denoise):
            v.trace_add("write", self._param_changed)

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_toolbar()
        self._build_main()
        self._build_statusbar()

        # Atajos
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Control-y>", lambda e: self.redo())
        self.bind("<Control-s>", lambda e: self.save_result())
        self.bind("<Control-o>", lambda e: self.load_image())
        self.bind("<Return>",    lambda e: self.process())

        # Placeholder en canvas
        self._show_placeholder()

        if initial_image:
            self.after(150, lambda: self._load_path(initial_image))

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _build_toolbar(self):
        bar = tk.Frame(self, bg=C["panel"], pady=0)
        bar.pack(fill="x")

        # Línea separadora en la parte inferior
        def mkbtn(text, cmd, accent=False, tip=None):
            bg = C["accent"] if accent else C["card"]
            fg = "white"
            hover_bg = "#3a7bc8" if accent else C["border"]
            b = tk.Button(bar, text=text, command=cmd,
                          bg=bg, fg=fg,
                          activebackground=hover_bg,
                          activeforeground="white",
                          relief="flat", bd=0,
                          font=("Segoe UI", 10),
                          padx=14, pady=8,
                          highlightthickness=0)
            b.pack(side="left", padx=(4 if accent else 1), pady=6)
            if tip:
                Tooltip(b, tip)
            return b

        mkbtn("📂  Abrir",    self.load_image,  tip="Cargar imagen escaneada  (Ctrl+O)")
        self._btn_process = mkbtn(
            "▶  Procesar", self.process, accent=True,
            tip="Ejecutar el pipeline de digitalización  (Enter)")
        mkbtn("💾  Guardar",  self.save_result, tip="Guardar resultado final  (Ctrl+S)")

        # Barra de progreso en la toolbar
        self._progress = ttk.Progressbar(bar, length=200, mode="determinate",
                                         bootstyle="info-striped")
        self._progress.pack(side="left", padx=16, pady=8)
        self._progress["value"] = 0

        tk.Label(bar,
                 text="Ctrl+O  ·  Enter  ·  Ctrl+S  ·  Ctrl+Z/Y  ·  Ctrl+Scroll zoom",
                 bg=C["panel"], fg=C["text_hint"], font=FONT_SMALL
                 ).pack(side="right", padx=12)

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

    # ── Main layout ───────────────────────────────────────────────────────────

    def _build_main(self):
        main = tk.Frame(self, bg=C["bg"])
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # ── Panel izquierdo scrollable ────────────────────────────────────────
        left_w = 310
        left_frame = tk.Frame(main, bg=C["panel"], width=left_w)
        left_frame.grid(row=0, column=0, sticky="nsew")
        left_frame.pack_propagate(False)

        # Título del panel
        tk.Label(left_frame, text="  Parámetros del Pipeline",
                 bg=C["bg"], fg=C["text"],
                 font=("Segoe UI", 11, "bold"),
                 anchor="w"
                 ).pack(fill="x", ipady=8, ipadx=4)
        tk.Frame(left_frame, bg=C["border"], height=1).pack(fill="x")

        # Canvas scrollable para los parámetros
        lc = tk.Canvas(left_frame, bg=C["panel"],
                       highlightthickness=0, width=left_w - 16)
        lvsb = ttk.Scrollbar(left_frame, orient="vertical", command=lc.yview)
        lc.configure(yscrollcommand=lvsb.set)
        lvsb.pack(side="right", fill="y")
        lc.pack(side="left", fill="both", expand=True)

        self.panel_params = PanelParams(lc, self)
        fid = lc.create_window((0, 0), window=self.panel_params, anchor="nw")

        def _cfg_inner(e):
            lc.configure(scrollregion=lc.bbox("all"))
        def _cfg_canvas(e):
            lc.itemconfig(fid, width=e.width)

        self.panel_params.bind("<Configure>", _cfg_inner)
        lc.bind("<Configure>", _cfg_canvas)

        def _scroll_left(e):
            delta = -1 if (getattr(e, "delta", 0) > 0 or getattr(e, "num", 0) == 4) else 1
            lc.yview_scroll(delta, "units")
        lc.bind("<MouseWheel>", _scroll_left)
        lc.bind("<Button-4>",   _scroll_left)
        lc.bind("<Button-5>",   _scroll_left)
        self.panel_params.bind("<MouseWheel>", _scroll_left)

        # ── Canvas central ────────────────────────────────────────────────────
        self.canvas_view = DrawingCanvas(main, self)
        self.canvas_view.grid(row=0, column=1, sticky="nsew", padx=2)

        # ── Panel derecho ─────────────────────────────────────────────────────
        right_w = 210
        right_frame = tk.Frame(main, bg=C["panel"], width=right_w)
        right_frame.grid(row=0, column=2, sticky="nsew")
        right_frame.pack_propagate(False)

        tk.Label(right_frame, text="  Herramientas",
                 bg=C["bg"], fg=C["text"],
                 font=("Segoe UI", 11, "bold"),
                 anchor="w"
                 ).pack(fill="x", ipady=8, ipadx=4)
        tk.Frame(right_frame, bg=C["border"], height=1).pack(fill="x")

        self.panel_tools = PanelTools(right_frame, self)
        self.panel_tools.pack(fill="both", expand=True)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")
        bar = tk.Frame(self, bg=C["bg"], pady=4)
        bar.pack(fill="x", side="bottom")

        self._sv_msg  = tk.StringVar(value="Listo · Abre una imagen con 📂 o Ctrl+O")
        self._sv_img  = tk.StringVar(value="")
        self._sv_zoom = tk.StringVar(value="")
        self._sv_time = tk.StringVar(value="")

        for sv, anchor, w in [
            (self._sv_msg,  "w", 50),
            (self._sv_img,  "w", 20),
            (self._sv_zoom, "e", 8),
            (self._sv_time, "e", 12),
        ]:
            tk.Label(bar, textvariable=sv, bg=C["bg"], fg=C["text_dim"],
                     font=FONT_SMALL, anchor=anchor, width=w
                     ).pack(side="left", padx=10)

    def _show_placeholder(self):
        """Muestra un mensaje centrado en el canvas cuando no hay imagen."""
        c = self.canvas_view.canvas
        self.update_idletasks()
        cw = c.winfo_width() or 800
        ch = c.winfo_height() or 600
        c.create_text(cw // 2, ch // 2,
                      text="📂  Abre una imagen escaneada\n\nCtrl+O  o  botón Abrir",
                      fill=C["text_hint"],
                      font=("Segoe UI", 16),
                      justify="center",
                      tags="placeholder")

    # ── Carga de imagen ───────────────────────────────────────────────────────

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Abrir imagen escaneada",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                       ("Todos los archivos", "*.*")])
        if path:
            self._load_path(path)

    def _load_path(self, path):
        img = _imread(path)
        if img is None:
            messagebox.showerror("Error", f"No se pudo cargar:\n{path}")
            return
        self.img_original = img
        self.img_result   = None
        h, w = img.shape[:2]
        self.paint_layer  = np.zeros((h, w), dtype=np.uint8)
        self._undo_stack.clear()
        self._redo_stack.clear()

        self._sv_img.set(f"📄  {w} × {h} px")
        self._sv_msg.set(f"Imagen cargada · Pulsa ▶ Procesar o Enter")
        self.canvas_view.canvas.delete("placeholder")
        self.panel_tools._select_view("original")
        self.canvas_view.refresh()
        self.after(250, self.canvas_view.fit_to_window)

    # ── Procesado ─────────────────────────────────────────────────────────────

    def _collect_params(self):
        block = self.p_block.get()
        if block % 2 == 0:
            block += 1
        return {
            "alpha":         self.p_alpha.get(),
            "block":         block,
            "c_offset":      self.p_c_offset.get(),
            "grid_kernel":   self.p_grid_kernel.get(),
            "inpaint_r":     self.p_inpaint_r.get(),
            "noise_small":   self.p_noise_small.get(),
            "noise_medium":  self.p_noise_medium.get(),
            "noise_radius":  self.p_noise_radius.get(),
            "noise_density": self.p_noise_density.get(),
            "skip_inpaint":  self.p_skip_inpaint.get(),
            "skip_color":    self.p_skip_color.get(),
            "skip_denoise":  self.p_skip_denoise.get(),
        }

    def process(self):
        if self.img_original is None:
            messagebox.showinfo("Sin imagen",
                                "Carga una imagen primero con 📂 o Ctrl+O.")
            return
        if self._processing:
            return
        self._processing = True
        self._btn_process.configure(state="disabled", text="⏳  Procesando…")
        self._progress["value"] = 0
        params = self._collect_params()
        t0 = time.time()

        def _worker():
            try:
                result = run_pipeline(
                    self.img_original, params,
                    progress_cb=lambda msg, pct: self.after(
                        0, lambda m=msg, p=pct: self._on_progress(m, p)))
                elapsed = time.time() - t0
                self.after(0, lambda: self._on_done(result, elapsed))
            except Exception as ex:
                import traceback
                tb = traceback.format_exc()
                self.after(0, lambda: self._on_error(str(ex), tb))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_progress(self, msg, pct):
        self._sv_msg.set(f"[{pct:3d}%]  {msg}")
        self._progress["value"] = pct

    def _on_done(self, result, elapsed):
        self.img_result = result
        self._processing = False
        self._btn_process.configure(state="normal", text="▶  Procesar")
        self._progress["value"] = 100
        self._sv_msg.set("✓  Procesado correctamente")
        self._sv_time.set(f"⏱  {elapsed:.1f} s")
        self.panel_tools._select_view("procesado")

    def _on_error(self, msg, tb):
        self._processing = False
        self._btn_process.configure(state="normal", text="▶  Procesar")
        self._progress["value"] = 0
        self._sv_msg.set(f"✗  Error: {msg}")
        messagebox.showerror("Error en el pipeline",
                             f"{msg}\n\nTraceback:\n{tb[:600]}")

    def _param_changed(self, *_):
        if not self.p_auto.get() or self._processing or self.img_original is None:
            return
        if self._auto_job:
            self.after_cancel(self._auto_job)
        self._auto_job = self.after(800, self.process)

    # ── Guardado ──────────────────────────────────────────────────────────────

    def save_result(self):
        if self.img_result is None:
            messagebox.showinfo("Sin resultado",
                                "Procesa la imagen primero (▶ Procesar o Enter).")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar resultado",
            defaultextension=".png",
            filetypes=[("PNG sin pérdida", "*.png"),
                       ("TIFF", "*.tiff"),
                       ("Todos", "*.*")])
        if not path:
            return
        ink = 255 - self.img_result.copy()
        if self.paint_layer is not None:
            ink[self.paint_layer == 1]   = 255
            ink[self.paint_layer == 255] = 0
        _imwrite(path, 255 - ink)
        self._sv_msg.set(f"✓  Guardado en {path}")

    # ── Undo / Redo ───────────────────────────────────────────────────────────

    def save_undo(self):
        if self.paint_layer is None:
            return
        self._undo_stack.append(self.paint_layer.copy())
        if len(self._undo_stack) > self.UNDO_LIMIT:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(self.paint_layer.copy())
        self.paint_layer = self._undo_stack.pop()
        self.canvas_view.refresh()

    def redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self.paint_layer.copy())
        self.paint_layer = self._redo_stack.pop()
        self.canvas_view.refresh()

    def clear_paint(self):
        if self.paint_layer is None:
            return
        self.save_undo()
        self.paint_layer[:] = 0
        self.canvas_view.refresh()

    # ── Status helpers ────────────────────────────────────────────────────────

    def set_status_zoom(self, z):
        self._sv_zoom.set(f"🔍  {z*100:.0f}%")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="GUI para digitalizar dibujos de libreta de cuadrícula azul.")
    p.add_argument("-i", "--image", metavar="RUTA",
                   help="Imagen a cargar al inicio (opcional)")
    args = p.parse_args()
    App(initial_image=args.image).mainloop()


if __name__ == "__main__":
    main()
