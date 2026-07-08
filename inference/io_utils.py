"""
inference/io_utils.py
=====================
Unicode-safe image I/O helpers shared across the inference package.

cv2.imread / cv2.imwrite fail silently on Windows when the path contains
non-ASCII characters (e.g. ``Escáner_20230219__19_.png``). These helpers
route through np.fromfile / cv2.imdecode and cv2.imencode / buf.tofile,
which are unaffected by locale-dependent path encoding.

Note: model/dataset.py and classic_pipeline/digitize_notebook.py each have
their own private _imread. Centralising them here is a Phase 4 cleanup task;
duplicating the helper now avoids touching tested Phase 1/2 code.
"""

from pathlib import Path

import cv2
import numpy as np


def _imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image from disk, unicode-safe.

    Args:
        path: Path to the image file.
        flags: OpenCV imread flag. Default IMREAD_COLOR -> BGR.

    Returns:
        np.ndarray: BGR image, shape (H, W, 3), dtype uint8
            (or (H, W) uint8 if flags=cv2.IMREAD_GRAYSCALE).

    Raises:
        FileNotFoundError: if the path does not exist or cannot be decoded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise FileNotFoundError(f"Could not decode image: {path}")
    return img


def _imwrite(path: Path, image: np.ndarray) -> None:
    """Write an image to disk, unicode-safe. Creates parent dirs if needed.

    Args:
        path: Destination path. Extension selects the encoder (.png, .jpg...).
        image (np.ndarray): any OpenCV-encodable array, e.g. shape (H, W)
            or (H, W, 3), dtype uint8.

    Raises:
        ValueError: if encoding fails (e.g. unsupported extension).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Could not encode image with extension '{path.suffix}': {path}")
    buf.tofile(str(path))


if __name__ == "__main__":
    pass
