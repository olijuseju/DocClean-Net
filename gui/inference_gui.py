#!/usr/bin/env python3
"""
gui/inference_gui.py
=====================
GUI interactiva para inferencia DocClean-Net (U-Net) + post-procesado bajo
demanda.

Flujo: cargar checkpoint -> cargar una o varias imágenes -> navegar entre
ellas (Anterior/Siguiente o flechas ←/→) -> ajustar white point /
eliminación de ruido / grosor de trazo con slider o escribiendo el valor ->
pulsar "Aplicar cambios" para recalcular -> guardar (una o todas).

Modelo de estado (importante): cada imagen guarda su ÚLTIMO resultado ya
confirmado (`last_result`). Los sliders solo modifican un estado "en vivo"
que no afecta a la imagen mostrada hasta pulsar "Aplicar cambios". Navegar
entre imágenes muestra el resultado confirmado de cada una, nunca reaplica
el estado en vivo de los sliders — así el botón Aplicar es el único punto
donde el post-procesado cambia, de forma consistente en toda la app.

Rendimiento: la inferencia U-Net (sliding window) es el paso lento; se
ejecuta UNA sola vez por imagen (en un hilo, con barra de progreso) y su
salida cruda (antes del white point) se cachea. Aplicar cambios solo llama
a gui.inference_core.apply_postprocessing() sobre esa caché (~ms). Cambiar
de device invalida las cachés (requiere reinferir).

Zoom y pan: Ctrl+rueda para zoom, arrastrar con clic izquierdo para
desplazar, botones 🔍−/Ajustar/🔍+ en la toolbar. Los paneles "Original" y
"Resultado" están sincronizados al mismo encuadre para comparar.

Requisitos: pip install ttkbootstrap Pillow (ya en requirements.txt)
Uso:
  python -m gui.inference_gui
  python -m gui.inference_gui --model checkpoints/best.pt
"""

import argparse
import json
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import ttkbootstrap as ttk
from PIL import Image, ImageTk

from gui.inference_core import PostprocessParams, apply_postprocessing
from inference.io_utils import _imread, _imwrite
from inference.predict import predict_image

# Optional drag-and-drop support: tkinterdnd2 is not a hard dependency.
# If it's installed the window accepts dropped files; if not, everything
# else works unchanged.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False

# Where the GUI remembers the last-used checkpoint between sessions.
_CONFIG_PATH = Path.home() / ".docclean_gui.json"

# ══════════════════════════════════════════════════════════════════════════
# ESTILO (mismo lenguaje visual que gui/digitize_gui.py)
# ══════════════════════════════════════════════════════════════════════════

C = {
    "bg": "#1a1d23",
    "panel": "#22262e",
    "card": "#2a2f3a",
    "border": "#363c4a",
    "accent": "#4a90d9",
    "accent2": "#5ba85e",
    "warn": "#e8a838",
    "danger": "#d9534f",
    "text": "#e0e4ed",
    "text_dim": "#7b8399",
    "text_hint": "#4f5568",
}

FONT_LABEL = ("Segoe UI", 9)
FONT_SMALL = ("Segoe UI", 8)
FONT_TITLE = ("Segoe UI", 10, "bold")

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Parameters the user is most likely to hit wrong, translated to plain hints.
_TOOLTIPS = {
    "white_auto": "Estima el nivel de papel automáticamente por imagen y lo lleva a "
                  "blanco puro. Desactívalo para fijar un valor manual.",
    "white_manual": "Nivel de gris que se convierte en blanco (255). Más bajo = más "
                    "agresivo (elimina más fondo, puede comerse trazos tenues).",
    "min_dot": "Elimina manchas de tinta más pequeñas que este área en píxeles "
               "(puntitos sueltos). 0 desactiva la limpieza.",
    "ink_threshold": "Nivel de gris por debajo del cual un píxel cuenta como tinta, "
                     "para decidir qué es un puntito. Rara vez hay que tocarlo.",
    "thicken": "Engorda los trazos. 1 = sutil. 2+ puede fusionar detalles finos "
               "(texto pequeño, ojos). 0 desactiva el engorde.",
    "device": "auto elige GPU si hay CUDA, si no CPU. Cambiarlo recalcula la "
              "inferencia de todas las imágenes.",
}


# ══════════════════════════════════════════════════════════════════════════
# Tooltip (portado de gui/digitize_gui.py para consistencia visual)
# ══════════════════════════════════════════════════════════════════════════


class Tooltip:
    """Delayed tooltip that always tears down cleanly.

    Shows after `delay` ms on hover, cancels on leave/click/destroy so it
    never lingers. Ported from gui/digitize_gui.py so both GUIs behave the
    same.
    """

    def __init__(self, widget: tk.Widget, text: str, delay: int = 600) -> None:
        self._widget = widget
        self._text = text
        self._delay = delay
        self._win: tk.Toplevel | None = None
        self._job: str | None = None
        widget.bind("<Enter>", self._on_enter, add="+")
        widget.bind("<Leave>", self._on_leave, add="+")
        widget.bind("<Button>", self._on_leave, add="+")
        widget.bind("<Destroy>", self._on_destroy, add="+")

    def _on_enter(self, _e) -> None:
        self._cancel()
        self._job = self._widget.after(self._delay, self._show)

    def _on_leave(self, _e=None) -> None:
        self._cancel()
        self._hide()

    def _on_destroy(self, _e=None) -> None:
        self._cancel()
        self._hide()

    def _cancel(self) -> None:
        if self._job:
            try:
                self._widget.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None

    def _show(self) -> None:
        if self._win:
            return
        x = self._widget.winfo_rootx() + 8
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._win = tk.Toplevel(self._widget)
        self._win.wm_overrideredirect(True)
        self._win.wm_geometry(f"+{x}+{y}")
        self._win.attributes("-topmost", True)
        self._win.configure(bg=C["accent"])
        tk.Label(
            self._win, text=self._text, background=C["card"], foreground=C["text"],
            relief="flat", bd=0, font=FONT_LABEL, wraplength=280, justify="left",
            padx=8, pady=5,
        ).pack(padx=1, pady=1)

    def _hide(self) -> None:
        if self._win:
            self._win.destroy()
            self._win = None


def _load_config() -> dict:
    """Read the small persisted-settings file, tolerating absence/corruption.

    Returns:
        dict: config mapping, or {} if the file doesn't exist or is invalid.
    """
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return {}


def _save_config(config: dict) -> None:
    """Persist settings, silently ignoring write failures (non-critical).

    Args:
        config: JSON-serialisable settings mapping.
    """
    try:
        _CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    except OSError:
        pass


@dataclass
class ImageEntry:
    """One loaded image plus its cached inference and last confirmed result.

    Attributes:
        path: source file path.
        original_bgr (np.ndarray): input scan, shape (H, W, 3), dtype uint8.
        raw_output (np.ndarray | None): cached U-Net output BEFORE white
            point (predict_image(..., white_point=None)), shape (H, W),
            dtype uint8. None until inference has run for this entry.
        device_used: device string the cache was computed with; if it no
            longer matches the current selection the cache is stale.
        last_result (np.ndarray | None): last post-processed image the user
            confirmed (via "Aplicar cambios" or the first auto-apply after
            inference), shape (H, W), dtype uint8. This is what navigation
            re-displays — never the live slider state.
        applied_params (PostprocessParams | None): the params `last_result`
            was produced with, so the UI can tell whether the live sliders
            currently differ from what's shown.
    """

    path: Path
    original_bgr: np.ndarray
    raw_output: np.ndarray | None = None
    device_used: str | None = None
    last_result: np.ndarray | None = None
    applied_params: PostprocessParams | None = None
    thumbnail: "ImageTk.PhotoImage | None" = None


class ZoomableCanvas(tk.Frame):
    """Canvas with zoom (Ctrl+wheel) and pan (left-drag), matching the
    conventions in gui/digitize_gui.py's drawing canvas (same key bindings,
    same zoom limits), minus the paint layer this app doesn't need.

    Accepts an `on_view_changed` callback invoked after any zoom or pan
    action, so the App can mirror the same view onto a second, linked
    canvas (keeps "Original" and "Resultado" at the same region).
    """

    MIN_ZOOM = 0.05
    MAX_ZOOM = 8.0

    def __init__(self, parent, on_view_changed=None, **kw):
        super().__init__(parent, bg=C["card"], **kw)
        self.on_view_changed = on_view_changed

        self.canvas = tk.Canvas(self, bg="#111418", highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self._on_vscroll)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self._on_hscroll)
        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._zoom = 1.0
        self._array: np.ndarray | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._img_id: int | None = None
        self._pan_start: tuple[int, int] | None = None

        c = self.canvas
        c.bind("<Control-MouseWheel>", self._zoom_win)
        c.bind("<Control-Button-4>", lambda e: self._zoom_step(1.15))
        c.bind("<Control-Button-5>", lambda e: self._zoom_step(1 / 1.15))
        c.bind("<MouseWheel>", self._scroll_win)
        c.bind("<Button-4>", lambda e: self._scroll_units(-3))
        c.bind("<Button-5>", lambda e: self._scroll_units(3))
        c.bind("<ButtonPress-1>", self._pan_start_ev)
        c.bind("<B1-Motion>", self._pan_move)
        c.bind("<ButtonRelease-1>", self._pan_end)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def set_image(self, array: np.ndarray) -> None:
        """Set (or replace) the displayed array and re-render at current zoom.

        Args:
            array (np.ndarray): shape (H, W) or (H, W, 3), dtype uint8.
        """
        self._array = array
        self.refresh()

    def clear(self) -> None:
        """Remove any displayed image (used when the view has nothing to show)."""
        self._array = None
        if self._img_id is not None:
            self.canvas.delete(self._img_id)
            self._img_id = None

    def refresh(self) -> None:
        if self._array is None:
            return
        rgb = (
            cv2.cvtColor(self._array, cv2.COLOR_GRAY2RGB)
            if self._array.ndim == 2
            else cv2.cvtColor(self._array, cv2.COLOR_BGR2RGB)
        )
        h, w = rgb.shape[:2]
        dw, dh = max(1, int(w * self._zoom)), max(1, int(h * self._zoom))
        method = Image.LANCZOS if self._zoom < 1 else Image.NEAREST
        resized = Image.fromarray(rgb).resize((dw, dh), method)

        self._photo = ImageTk.PhotoImage(resized)
        if self._img_id is None:
            self._img_id = self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            self.canvas.itemconfig(self._img_id, image=self._photo)
        self.canvas.configure(scrollregion=(0, 0, dw, dh))

    # ── Zoom ──────────────────────────────────────────────────────────────────

    def _zoom_step(self, factor: float, notify: bool = True) -> None:
        old = self._zoom
        self._zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self._zoom * factor))
        if self._zoom != old:
            self.refresh()
            if notify:
                self._notify()

    def _zoom_win(self, e) -> None:
        self._zoom_step(1.15 if e.delta > 0 else 1 / 1.15)

    def zoom_in(self) -> None:
        self._zoom_step(1.25)

    def zoom_out(self) -> None:
        self._zoom_step(1 / 1.25)

    def fit_to_window(self, notify: bool = True) -> None:
        if self._array is None:
            return
        self.update_idletasks()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        ih, iw = self._array.shape[:2]
        if cw > 1 and ch > 1:
            self._zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, min(cw / iw, ch / ih) * 0.95))
        self.refresh()
        if notify:
            self._notify()

    # ── Scroll ────────────────────────────────────────────────────────────────

    def _scroll_win(self, e) -> None:
        self._scroll_units(-1 if e.delta > 0 else 1)

    def _scroll_units(self, units: int) -> None:
        self.canvas.yview_scroll(units, "units")
        self._notify()

    def _on_vscroll(self, *args) -> None:
        self.canvas.yview(*args)
        self._notify()

    def _on_hscroll(self, *args) -> None:
        self.canvas.xview(*args)
        self._notify()

    # ── Pan ───────────────────────────────────────────────────────────────────

    def _pan_start_ev(self, e) -> None:
        self._pan_start = (e.x, e.y)
        self.canvas.configure(cursor="fleur")

    def _pan_move(self, e) -> None:
        if self._pan_start is None:
            return
        dx, dy = self._pan_start[0] - e.x, self._pan_start[1] - e.y
        self._pan_start = (e.x, e.y)
        self.canvas.xview_scroll(int(dx), "units")
        self.canvas.yview_scroll(int(dy), "units")
        self._notify()

    def _pan_end(self, _e) -> None:
        self._pan_start = None
        self.canvas.configure(cursor="")

    # ── View sync ─────────────────────────────────────────────────────────────

    def _notify(self) -> None:
        if self.on_view_changed:
            self.on_view_changed(self)

    def apply_view(self, zoom: float, x_fraction: float, y_fraction: float) -> None:
        """Mirror another canvas's zoom + scroll onto this one (no re-notify)."""
        if self._zoom != zoom:
            self._zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, zoom))
            self.refresh()
        self.canvas.xview_moveto(x_fraction)
        self.canvas.yview_moveto(y_fraction)


_AppBase = TkinterDnD.Tk if _DND_AVAILABLE else ttk.Window


class App(_AppBase):
    """Main application window."""

    def __init__(self, initial_model: str | None = None) -> None:
        if _DND_AVAILABLE:
            # TkinterDnD.Tk isn't a ttkbootstrap Window, so apply the theme
            # to the default style explicitly.
            super().__init__()
            ttk.Style(theme="darkly")
        else:
            super().__init__(themename="darkly")

        self._config = _load_config()

        self.title("DocClean-Net — Inference Studio")
        self.geometry("1480x880")
        self.minsize(1120, 700)  # usable on a 1366x768 laptop
        self.configure(bg=C["bg"])

        # Last-used checkpoint: CLI arg wins, else the persisted one.
        remembered = self._config.get("last_model")
        if initial_model:
            self.model_path: Path | None = Path(initial_model)
        elif remembered and Path(remembered).exists():
            self.model_path = Path(remembered)
        else:
            self.model_path = None

        self.images: list[ImageEntry] = []
        self.current_index: int = -1
        self._processing = False
        self._dirty = False
        self._flash_job: str | None = None

        self.device_var = tk.StringVar(value="auto")
        self.white_point_auto_var = tk.BooleanVar(value=True)
        self.white_point_value_var = tk.IntVar(value=200)
        self.min_dot_area_var = tk.IntVar(value=3)
        self.ink_threshold_var = tk.IntVar(value=128)
        self.thicken_amount_var = tk.IntVar(value=1)
        self.goto_var = tk.StringVar(value="")

        self._busy_buttons: list[ttk.Button] = []
        self._thumb_widgets: list[tk.Label] = []

        self._build_toolbar()
        self._build_main_area()
        self._build_statusbar()
        self._bind_shortcuts()

        if _DND_AVAILABLE:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_drop)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if self.model_path:
            self._set_status(f"Modelo: {self.model_path.name}")

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        bar = tk.Frame(self, bg=C["panel"], pady=8, padx=10)
        bar.pack(fill="x", side="top")

        b_model = ttk.Button(bar, text="📦 Cargar modelo…", command=self.load_model,
                             bootstyle="secondary")
        b_model.pack(side="left", padx=(0, 8))
        b_imgs = ttk.Button(bar, text="📂 Cargar imágenes…", command=self.load_images,
                            bootstyle="secondary")
        b_imgs.pack(side="left", padx=(0, 16))
        self._busy_buttons += [b_model, b_imgs]

        lbl_device = tk.Label(bar, text="Device:", bg=C["panel"], fg=C["text_dim"],
                              font=FONT_LABEL)
        lbl_device.pack(side="left", padx=(0, 4))
        self._device_combo = ttk.Combobox(
            bar, textvariable=self.device_var, values=["auto", "cpu", "cuda"],
            state="readonly", width=8,
        )
        self._device_combo.pack(side="left", padx=(0, 16))
        self._device_combo.bind("<<ComboboxSelected>>", self._on_device_changed)
        Tooltip(lbl_device, _TOOLTIPS["device"])

        self._btn_prev = ttk.Button(bar, text="◀ Anterior", command=lambda: self.navigate(-1),
                                    bootstyle="secondary-outline")
        self._btn_prev.pack(side="left", padx=(0, 4))
        self._btn_next = ttk.Button(bar, text="Siguiente ▶", command=lambda: self.navigate(1),
                                    bootstyle="secondary-outline")
        self._btn_next.pack(side="left")

        self._sv_counter = tk.StringVar(value="0 / 0")
        tk.Label(bar, textvariable=self._sv_counter, bg=C["panel"], fg=C["text"],
                 font=FONT_TITLE).pack(side="left", padx=(16, 6))

        goto_entry = ttk.Entry(bar, textvariable=self.goto_var, width=4)
        goto_entry.pack(side="left")
        goto_entry.bind("<Return>", self._on_goto)
        b_goto = ttk.Button(bar, text="Ir", width=3, command=self._on_goto,
                            bootstyle="secondary-outline")
        b_goto.pack(side="left", padx=(2, 0))
        Tooltip(goto_entry, "Escribe un número de página y pulsa Enter para saltar a ella.")
        self._busy_buttons.append(b_goto)

        tk.Frame(bar, bg=C["border"], width=1).pack(side="left", fill="y", padx=8)
        ttk.Button(bar, text="🔍−", width=3, command=self._zoom_out,
                   bootstyle="secondary-outline").pack(side="left", padx=(4, 2))
        ttk.Button(bar, text="Ajustar", command=self._zoom_fit,
                   bootstyle="secondary-outline").pack(side="left", padx=2)
        ttk.Button(bar, text="🔍+", width=3, command=self._zoom_in,
                   bootstyle="secondary-outline").pack(side="left", padx=(2, 0))

        self._btn_save = ttk.Button(bar, text="💾 Guardar", command=self.save_current,
                                    bootstyle="success")
        self._btn_save.pack(side="right", padx=(8, 0))
        self._btn_save_all = ttk.Button(bar, text="💾 Guardar todos…", command=self.save_all,
                                        bootstyle="success-outline")
        self._btn_save_all.pack(side="right")
        self._busy_buttons += [self._btn_prev, self._btn_next,
                               self._btn_save, self._btn_save_all]

    def _build_main_area(self) -> None:
        main = tk.Frame(self, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # ── Thumbnail strip (left) ────────────────────────────────────────────
        thumb_card = tk.LabelFrame(main, text="Páginas", bg=C["card"], fg=C["text"],
                                   font=FONT_TITLE, width=132)
        thumb_card.pack(side="left", fill="y", padx=(0, 8))
        thumb_card.pack_propagate(False)

        thumb_canvas = tk.Canvas(thumb_card, bg=C["card"], highlightthickness=0, width=116)
        thumb_scroll = ttk.Scrollbar(thumb_card, orient="vertical",
                                     command=thumb_canvas.yview)
        thumb_canvas.configure(yscrollcommand=thumb_scroll.set)
        thumb_scroll.pack(side="right", fill="y")
        thumb_canvas.pack(side="left", fill="both", expand=True)
        self._thumb_inner = tk.Frame(thumb_canvas, bg=C["card"])
        thumb_canvas.create_window((0, 0), window=self._thumb_inner, anchor="nw")
        self._thumb_inner.bind(
            "<Configure>",
            lambda e: thumb_canvas.configure(scrollregion=thumb_canvas.bbox("all")),
        )
        thumb_canvas.bind("<MouseWheel>",
                          lambda e: thumb_canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))

        # ── Comparison canvases (center) ──────────────────────────────────────
        canvases = tk.Frame(main, bg=C["bg"])
        canvases.pack(side="left", fill="both", expand=True)

        original_card = tk.LabelFrame(canvases, text="Original", bg=C["card"],
                                       fg=C["text"], font=FONT_TITLE)
        original_card.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self._canvas_original = ZoomableCanvas(
            original_card, on_view_changed=self._sync_from_original,
        )
        self._canvas_original.pack(fill="both", expand=True, padx=8, pady=8)

        result_card = tk.LabelFrame(canvases, text="Resultado", bg=C["card"],
                                     fg=C["text"], font=FONT_TITLE)
        result_card.pack(side="left", fill="both", expand=True, padx=(6, 0))
        self._canvas_result = ZoomableCanvas(
            result_card, on_view_changed=self._sync_from_result,
        )
        self._canvas_result.pack(fill="both", expand=True, padx=8, pady=8)

        # ── Parameters (right) ────────────────────────────────────────────────
        panel = tk.Frame(main, bg=C["panel"], width=290)
        panel.pack(side="right", fill="y", padx=(10, 0))
        panel.pack_propagate(False)
        self._build_params_panel(panel)

    def _sync_from_original(self, source: "ZoomableCanvas") -> None:
        self._canvas_result.apply_view(
            source._zoom, source.canvas.xview()[0], source.canvas.yview()[0]
        )

    def _sync_from_result(self, source: "ZoomableCanvas") -> None:
        self._canvas_original.apply_view(
            source._zoom, source.canvas.xview()[0], source.canvas.yview()[0]
        )

    def _build_params_panel(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="Parámetros", bg=C["panel"], fg=C["text"],
                 font=FONT_TITLE).pack(anchor="w", padx=12, pady=(12, 8))

        wp_frame = tk.Frame(parent, bg=C["panel"])
        wp_frame.pack(fill="x", padx=12, pady=(0, 4))
        chk = tk.Checkbutton(
            wp_frame, text="White point automático", variable=self.white_point_auto_var,
            command=self._on_white_point_mode_changed, bg=C["panel"], fg=C["text"],
            selectcolor=C["card"], activebackground=C["panel"], font=FONT_LABEL,
        )
        chk.pack(anchor="w")
        Tooltip(chk, _TOOLTIPS["white_auto"])

        self._entry_white_point = self._slider_labeled(
            parent, "White point (manual)", self.white_point_value_var, 1, 255,
            _TOOLTIPS["white_manual"],
        )
        self._on_white_point_mode_changed()

        self._slider_labeled(parent, "Área mín. puntito (px)", self.min_dot_area_var,
                             0, 20, _TOOLTIPS["min_dot"])
        self._slider_labeled(parent, "Umbral de tinta", self.ink_threshold_var,
                             0, 255, _TOOLTIPS["ink_threshold"])
        self._slider_labeled(parent, "Grosor de trazo", self.thicken_amount_var,
                             0, 5, _TOOLTIPS["thicken"])

        self._btn_apply = ttk.Button(
            parent, text="✓ Aplicar cambios", command=self.apply_changes,
            bootstyle="primary",
        )
        self._btn_apply.pack(fill="x", padx=12, pady=(12, 4))
        self._busy_buttons.append(self._btn_apply)

        self._sv_apply_hint = tk.StringVar(
            value="Ajusta los controles y pulsa aquí para ver el efecto."
        )
        tk.Label(parent, textvariable=self._sv_apply_hint, bg=C["panel"],
                 fg=C["text_hint"], font=FONT_SMALL, wraplength=260,
                 justify="left").pack(anchor="w", padx=12)

        tk.Frame(parent, bg=C["border"], height=1).pack(fill="x", padx=12, pady=12)

        self._btn_process_all = ttk.Button(
            parent, text="⚙ Procesar todas", command=self.process_all,
            bootstyle="info-outline",
        )
        self._btn_process_all.pack(fill="x", padx=12, pady=(0, 4))
        self._busy_buttons.append(self._btn_process_all)
        tk.Label(parent, text="Ejecuta la inferencia en todas las imágenes cargadas "
                              "(necesario antes de «Guardar todos»).",
                 bg=C["panel"], fg=C["text_hint"], font=FONT_SMALL, wraplength=260,
                 justify="left").pack(anchor="w", padx=12)

        self._progress = ttk.Progressbar(parent, mode="determinate", maximum=100)
        self._progress.pack(fill="x", padx=12, pady=(16, 2))
        self._sv_progress = tk.StringVar(value="")
        tk.Label(parent, textvariable=self._sv_progress, bg=C["panel"],
                 fg=C["text_hint"], font=FONT_SMALL).pack(anchor="w", padx=12)

    def _on_white_point_mode_changed(self) -> None:
        """Enable/disable the manual white-point entry; mark state dirty."""
        state = "disabled" if self.white_point_auto_var.get() else "normal"
        self._entry_white_point.configure(state=state)
        self._mark_dirty()

    def _slider_labeled(
        self, parent: tk.Frame, label: str, var: tk.IntVar, lo: int, hi: int,
        tooltip: str,
    ) -> ttk.Entry:
        """Build a labeled slider + numeric entry pair bound to `var`.

        Neither control recomputes the result on change — they update `var`
        and flag the pending-changes state. Recompute happens on Aplicar.

        Args:
            parent: container frame.
            label: field name shown above the control.
            var: shared IntVar, updated live by both slider and entry.
            lo, hi: inclusive bounds for slider and entry clamping.
            tooltip: hover help text.

        Returns:
            ttk.Entry: the numeric entry (so callers can enable/disable it).
        """
        frame = tk.Frame(parent, bg=C["panel"])
        frame.pack(fill="x", padx=12, pady=6)
        header = tk.Frame(frame, bg=C["panel"])
        header.pack(fill="x")
        lbl = tk.Label(header, text=label, bg=C["panel"], fg=C["text_dim"],
                       font=FONT_LABEL)
        lbl.pack(side="left")
        Tooltip(lbl, tooltip)

        row = tk.Frame(frame, bg=C["panel"])
        row.pack(fill="x")

        def _on_slider_move(_evt=None, var=var):
            entry_var.set(str(var.get()))
            self._mark_dirty()

        scale = ttk.Scale(row, from_=lo, to=hi, variable=var, command=_on_slider_move)
        scale.pack(side="left", fill="x", expand=True)

        entry_var = tk.StringVar(value=str(var.get()))

        def _on_entry_commit(_evt=None, var=var, lo=lo, hi=hi):
            try:
                value = int(entry_var.get())
            except ValueError:
                entry_var.set(str(var.get()))
                return
            value = max(lo, min(hi, value))
            var.set(value)
            entry_var.set(str(value))
            self._mark_dirty()

        entry = ttk.Entry(row, textvariable=entry_var, width=5)
        entry.pack(side="left", padx=(8, 0))
        entry.bind("<Return>", _on_entry_commit)
        entry.bind("<FocusOut>", _on_entry_commit)
        Tooltip(entry, tooltip)

        return entry

    def _build_statusbar(self) -> None:
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")
        bar = tk.Frame(self, bg=C["bg"], pady=4)
        bar.pack(fill="x", side="bottom")
        self._sv_msg = tk.StringVar(value="Listo · Carga un modelo y una imagen")
        self._sv_time = tk.StringVar(value="")
        tk.Label(bar, textvariable=self._sv_msg, bg=C["bg"], fg=C["text_dim"],
                 font=FONT_SMALL, anchor="w").pack(side="left", padx=10, fill="x", expand=True)
        tk.Label(bar, textvariable=self._sv_time, bg=C["bg"], fg=C["text_dim"],
                 font=FONT_SMALL, anchor="e").pack(side="right", padx=10)

    def _bind_shortcuts(self) -> None:
        self.bind("<Left>", lambda e: self.navigate(-1))
        self.bind("<Right>", lambda e: self.navigate(1))
        self.bind("<Control-s>", lambda e: self.save_current())
        self.bind("<Return>", lambda e: self.apply_changes())

    # ── Dirty-state feedback ─────────────────────────────────────────────────

    def _mark_dirty(self) -> None:
        """Flag that the live controls differ from the displayed result."""
        self._dirty = True
        if hasattr(self, "_btn_apply"):
            self._btn_apply.configure(bootstyle="warning")
            self._sv_apply_hint.set("● Cambios sin aplicar — pulsa «Aplicar cambios».")

    def _mark_clean(self) -> None:
        self._dirty = False
        if hasattr(self, "_btn_apply"):
            self._btn_apply.configure(bootstyle="primary")
            self._sv_apply_hint.set("Ajusta los controles y pulsa aquí para ver el efecto.")

    # ── Model / image loading ────────────────────────────────────────────────

    def load_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Cargar checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("Todos", "*.*")],
        )
        if not path:
            return
        self.model_path = Path(path)
        self._config["last_model"] = str(self.model_path)
        _save_config(self._config)
        for entry in self.images:
            entry.raw_output = None
            entry.device_used = None
            entry.last_result = None
        self._set_status(f"Modelo cargado: {self.model_path.name}")
        if self.current_index >= 0:
            self._ensure_current_processed()

    def load_images(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Cargar imágenes",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                       ("Todos", "*.*")],
        )
        if not paths:
            return
        self._add_image_paths([Path(p) for p in paths])

    def _add_image_paths(self, paths: list[Path]) -> None:
        """Load and append images (shared by the file dialog and drag-drop).

        Args:
            paths: candidate image paths. Non-images and duplicates are
                skipped; unreadable files raise a warning dialog and are
                skipped without aborting the rest.
        """
        existing = {e.path for e in self.images}
        added = 0
        for path in paths:
            if path in existing or path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            try:
                img = _imread(path)
            except (FileNotFoundError, ValueError) as exc:
                messagebox.showwarning("Imagen no válida",
                                       f"No se pudo cargar {path.name}:\n{exc}")
                continue
            self.images.append(ImageEntry(path=path, original_bgr=img))
            added += 1
        if added == 0:
            return
        if self.current_index < 0:
            self.current_index = 0
        self._rebuild_thumbnails()
        self._refresh_counter()
        self._show_original()
        self._ensure_current_processed()

    # ── Navigation ────────────────────────────────────────────────────────────

    def navigate(self, delta: int) -> None:
        if not self.images or self._processing:
            return
        new_index = self.current_index + delta
        if not (0 <= new_index < len(self.images)):
            return
        self.current_index = new_index
        self._refresh_counter()
        self._highlight_current_thumbnail()
        self._show_original()
        self._ensure_current_processed()

    def _refresh_counter(self) -> None:
        if not self.images:
            self._sv_counter.set("0 / 0")
            return
        entry = self.images[self.current_index]
        # A small marker tells the user which images already have a result.
        done = "✓" if entry.last_result is not None else "…"
        self._sv_counter.set(
            f"{self.current_index + 1} / {len(self.images)} {done} — {entry.path.name}"
        )

    def _zoom_in(self) -> None:
        self._canvas_original.zoom_in()

    def _zoom_out(self) -> None:
        self._canvas_original.zoom_out()

    def _zoom_fit(self) -> None:
        self._canvas_original.fit_to_window()

    def _show_original(self) -> None:
        entry = self.images[self.current_index]
        is_first_display = self._canvas_original._array is None
        self._canvas_original.set_image(entry.original_bgr)
        if is_first_display:
            self._canvas_original.fit_to_window()
        # Show this image's confirmed result, or clear if not computed yet.
        if entry.last_result is not None:
            self._canvas_result.set_image(entry.last_result)
        else:
            self._canvas_result.clear()

    # ── Inference (threaded) ──────────────────────────────────────────────────

    def _resolved_device(self) -> str:
        return self.device_var.get()

    def _on_device_changed(self, _evt=None) -> None:
        for entry in self.images:
            entry.raw_output = None
            entry.device_used = None
            entry.last_result = None
        if self.current_index >= 0:
            self._ensure_current_processed()

    def _ensure_current_processed(self) -> None:
        """Ensure the current image has an inference cache; compute if not."""
        if self.current_index < 0:
            return
        if self.model_path is None:
            self._set_status("Carga un modelo primero (📦 Cargar modelo…)")
            return
        entry = self.images[self.current_index]
        device = self._resolved_device()
        if entry.raw_output is not None and entry.device_used == device:
            # Already inferred: just make sure a confirmed result exists.
            if entry.last_result is None:
                self._apply_and_store(entry)
            return
        self._run_inference([entry], device, then_apply=True)

    def _run_inference(
        self, entries: list[ImageEntry], device: str, then_apply: bool
    ) -> None:
        """Run inference for a list of entries in a background thread.

        Args:
            entries: image entries whose raw_output needs (re)computing.
            device: torch device string.
            then_apply: if True, apply current params and store last_result
                for each entry once inference finishes.
        """
        pending = [e for e in entries
                   if e.raw_output is None or e.device_used != device]
        if not pending:
            for e in entries:
                if e.last_result is None:
                    self._apply_and_store(e)
            self._refresh_current_views()
            return

        self._set_busy(True)
        total = len(pending)
        self._progress.configure(maximum=total, value=0)
        self._sv_progress.set(f"0 / {total}")
        t0 = time.time()

        def _worker() -> None:
            try:
                for i, entry in enumerate(pending, start=1):
                    self.after(0, lambda i=i, e=entry: self._on_inference_progress(i, total, e))
                    raw = predict_image(
                        self.model_path, entry.path, device=device, white_point=None
                    )
                    entry.raw_output = raw
                    entry.device_used = device
                    if then_apply:
                        self.after(0, lambda e=entry: self._apply_and_store(e))
                    self.after(0, lambda i=i, e=entry: self._on_one_done(i, e))
                elapsed = time.time() - t0
                self.after(0, lambda: self._on_inference_done(elapsed, total))
            except Exception as exc:  # noqa: BLE001 - surfaced via dialog
                msg = self._friendly_error(exc)
                self.after(0, lambda: self._on_inference_error(msg))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_inference_progress(self, i: int, total: int, entry: ImageEntry) -> None:
        """Update status line at the START of processing image i (in the UI thread)."""
        self._set_status(f"Inferencia {i}/{total}: {entry.path.name} ...")

    def _on_one_done(self, i: int, entry: ImageEntry) -> None:
        """Advance the determinate bar and refresh this image's thumbnail marker."""
        self._progress.configure(value=i)
        self._sv_progress.set(f"{i} / {int(self._progress.cget('maximum'))}")
        self._update_thumbnail_state(entry)

    def _on_inference_done(self, elapsed: float, count: int) -> None:
        self._set_busy(False)
        self._progress.configure(value=0)
        self._sv_progress.set("")
        per = elapsed / max(1, count)
        self._sv_time.set(f"⏱ {elapsed:.1f} s ({per:.1f} s/img)")
        self._set_status(f"✓ Inferencia completa ({count} imagen(es))")
        self._refresh_current_views()

    def _on_inference_error(self, msg: str) -> None:
        self._set_busy(False)
        self._progress.configure(value=0)
        self._sv_progress.set("")
        self._set_status(f"✗ Error: {msg}")
        messagebox.showerror("Error de inferencia", msg)

    @staticmethod
    def _friendly_error(exc: Exception) -> str:
        """Translate common low-level errors into actionable messages."""
        text = str(exc)
        if "size mismatch" in text or "Missing key" in text or "Unexpected key" in text:
            return ("El checkpoint no coincide con la arquitectura U-Net esperada. "
                    "¿Seguro que es un best.pt de DocClean-Net?")
        if "out of memory" in text.lower():
            return ("Sin memoria en el dispositivo. Prueba con device = cpu, "
                    "o cierra otras aplicaciones que usen la GPU.")
        if "CUDA" in text and "available" in text:
            return "CUDA no está disponible en este equipo. Cambia device a cpu o auto."
        return text

    # ── Post-processing ───────────────────────────────────────────────────────

    def _current_params(self) -> PostprocessParams:
        return PostprocessParams(
            white_point_auto=self.white_point_auto_var.get(),
            white_point_value=self.white_point_value_var.get(),
            min_dot_area=self.min_dot_area_var.get(),
            ink_threshold=self.ink_threshold_var.get(),
            thicken_amount=self.thicken_amount_var.get(),
        )

    def _apply_and_store(self, entry: ImageEntry) -> None:
        """Compute and cache this entry's confirmed result from current params."""
        if entry.raw_output is None:
            return
        params = self._current_params()
        entry.last_result = apply_postprocessing(entry.raw_output, params)
        entry.applied_params = params

    def apply_changes(self) -> None:
        """Apply the current controls to the current image (the only place
        post-processing takes visible effect)."""
        if self.current_index < 0 or self._processing:
            return
        entry = self.images[self.current_index]
        if entry.raw_output is None:
            self._set_status("Espera a que termine la inferencia antes de aplicar.")
            return
        self._apply_and_store(entry)
        self._canvas_result.set_image(entry.last_result)
        self._mark_clean()
        self._refresh_counter()
        self._update_thumbnail_state(entry)
        self._flash_apply_done()

    def _flash_apply_done(self) -> None:
        """Brief green confirmation on the Apply button — post-processing is
        too fast (~ms) for a progress bar to be meaningful, so a short flash
        gives the visual acknowledgement instead."""
        self._set_status("✓ Cambios aplicados")
        self._btn_apply.configure(bootstyle="success")
        if self._flash_job:
            self.after_cancel(self._flash_job)
        self._flash_job = self.after(
            700, lambda: self._btn_apply.configure(bootstyle="primary")
        )

    def _refresh_current_views(self) -> None:
        if self.current_index < 0:
            return
        entry = self.images[self.current_index]
        if entry.last_result is not None:
            self._canvas_result.set_image(entry.last_result)
        self._mark_clean()
        self._refresh_counter()

    def process_all(self) -> None:
        """Run inference on every loaded image (batch), applying current params."""
        if not self.images:
            messagebox.showinfo("Sin imágenes", "Carga imágenes primero.")
            return
        if self.model_path is None:
            messagebox.showinfo("Sin modelo", "Carga un modelo primero.")
            return
        if self._processing:
            return
        self._run_inference(list(self.images), self._resolved_device(), then_apply=True)

    # ── Saving ────────────────────────────────────────────────────────────────

    def save_current(self) -> None:
        if self.current_index < 0:
            messagebox.showinfo("Sin imagen", "Carga una imagen primero.")
            return
        entry = self.images[self.current_index]
        if entry.last_result is None:
            messagebox.showinfo("Sin resultado", "Aplica cambios o espera a la inferencia.")
            return
        path = filedialog.asksaveasfilename(
            title="Guardar resultado", defaultextension=".png",
            initialfile=f"{entry.path.stem}_clean.png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tiff"), ("Todos", "*.*")],
        )
        if not path:
            return
        _imwrite(Path(path), entry.last_result)
        self._set_status(f"✓ Guardado en {path}")

    def save_all(self) -> None:
        if not self.images:
            messagebox.showinfo("Sin imágenes", "Carga imágenes primero.")
            return
        pending = [e for e in self.images if e.last_result is None]
        if pending:
            answer = messagebox.askyesno(
                "Imágenes sin procesar",
                f"{len(pending)} imagen(es) aún no tienen resultado. "
                "¿Procesarlas ahora antes de guardar?",
            )
            if answer:
                self.process_all()
            return
        out_dir = filedialog.askdirectory(title="Carpeta de salida")
        if not out_dir:
            return
        out_dir_path = Path(out_dir)
        for entry in self.images:
            _imwrite(out_dir_path / f"{entry.path.stem}_clean.png", entry.last_result)
        self._set_status(f"✓ {len(self.images)} imagen(es) guardadas en {out_dir}")

    # ── Busy state / lifecycle ───────────────────────────────────────────────

    def _set_busy(self, busy: bool) -> None:
        """Enable/disable action buttons and the device selector during work."""
        self._processing = busy
        state = "disabled" if busy else "normal"
        for btn in self._busy_buttons:
            btn.configure(state=state)
        self._device_combo.configure(state="disabled" if busy else "readonly")

    def _has_unsaved(self) -> bool:
        return any(e.last_result is not None for e in self.images)

    def _on_close(self) -> None:
        if self._processing:
            if not messagebox.askyesno("Procesando",
                                        "Hay una inferencia en curso. ¿Cerrar de todos modos?"):
                return
        self.destroy()

    # ── Go to page ────────────────────────────────────────────────────────────

    def _on_goto(self, _evt=None) -> None:
        """Jump to the 1-based page number typed in the go-to entry."""
        if self._processing:
            return
        raw = self.goto_var.get().strip()
        if not raw:
            return
        try:
            page = int(raw)
        except ValueError:
            self._set_status("Número de página no válido.")
            self.goto_var.set("")
            return
        if not self.images:
            return
        page = max(1, min(len(self.images), page))
        self.current_index = page - 1
        self.goto_var.set("")
        self._refresh_counter()
        self._highlight_current_thumbnail()
        self._show_original()
        self._ensure_current_processed()

    # ── Thumbnails ────────────────────────────────────────────────────────────

    def _make_thumbnail(self, bgr: np.ndarray, size: int = 96) -> ImageTk.PhotoImage:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = size / max(h, w)
        thumb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        return ImageTk.PhotoImage(Image.fromarray(thumb))

    def _rebuild_thumbnails(self) -> None:
        """Recreate the whole thumbnail strip (called after loading images)."""
        for w in self._thumb_widgets:
            w.destroy()
        self._thumb_widgets = []
        for idx, entry in enumerate(self.images):
            if entry.thumbnail is None:
                entry.thumbnail = self._make_thumbnail(entry.original_bgr)
            holder = tk.Frame(self._thumb_inner, bg=C["card"])
            holder.pack(fill="x", padx=4, pady=3)
            lbl = tk.Label(holder, image=entry.thumbnail, bg=C["card"],
                           bd=2, relief="flat", cursor="hand2")
            lbl.image = entry.thumbnail
            lbl.pack()
            lbl.bind("<Button-1>", lambda e, i=idx: self._on_thumb_click(i))
            caption = tk.Label(
                holder, text=f"{idx + 1}", bg=C["card"],
                fg=C["text_dim"], font=FONT_SMALL,
            )
            caption.pack()
            self._thumb_widgets.append(holder)
        self._highlight_current_thumbnail()

    def _update_thumbnail_state(self, entry: ImageEntry) -> None:
        """Refresh the processed marker for one entry's thumbnail caption."""
        if entry not in self.images:
            return
        idx = self.images.index(entry)
        if idx >= len(self._thumb_widgets):
            return
        holder = self._thumb_widgets[idx]
        done = entry.last_result is not None
        for child in holder.winfo_children():
            if isinstance(child, tk.Label) and child.cget("text"):
                child.configure(
                    text=f"{idx + 1} {'✓' if done else ''}".strip(),
                    fg=C["accent2"] if done else C["text_dim"],
                )

    def _highlight_current_thumbnail(self) -> None:
        for idx, holder in enumerate(self._thumb_widgets):
            for child in holder.winfo_children():
                if isinstance(child, tk.Label) and child.cget("image"):
                    child.configure(
                        highlightbackground=C["accent"] if idx == self.current_index else C["card"],
                        highlightthickness=2 if idx == self.current_index else 0,
                        bd=2, relief="solid" if idx == self.current_index else "flat",
                    )

    def _on_thumb_click(self, index: int) -> None:
        if self._processing or index == self.current_index:
            return
        self.current_index = index
        self._refresh_counter()
        self._highlight_current_thumbnail()
        self._show_original()
        self._ensure_current_processed()

    # ── Drag and drop ─────────────────────────────────────────────────────────

    def _on_drop(self, event) -> None:
        """Handle files dropped onto the window (tkinterdnd2 only)."""
        if self._processing:
            return
        paths = self.tk.splitlist(event.data)
        self._add_image_paths([Path(p) for p in paths])

    # ── Status helper ─────────────────────────────────────────────────────────

    def _set_status(self, msg: str) -> None:
        self._sv_msg.set(msg)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GUI interactiva de inferencia DocClean-Net + post-procesado.",
    )
    parser.add_argument("--model", metavar="RUTA", help="Checkpoint a cargar al inicio (opcional)")
    args = parser.parse_args()
    App(initial_model=args.model).mainloop()


if __name__ == "__main__":
    main()
