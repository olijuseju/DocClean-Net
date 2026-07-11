"""
model/dataset.py
================
PyTorch Dataset that reads (dirty, clean) image pairs from the synthetic
dataset generated in Phase 1, extracts random 256×256 patches, and returns
normalised single-channel tensors ready for U-Net training.

Pipeline per sample
-------------------
1. Load dirty and clean images from disk as BGR uint8.
2. Apply augment_pair() for synchronised geometric + photometric transforms.
3. Extract a single random 256×256 patch at the same location from both.
4. Convert BGR → grayscale (cv2.COLOR_BGR2GRAY).
5. Normalise to float32 in [0, 1].
6. Return tensors of shape (1, 256, 256).

Design decisions (do not reopen without reason):
    - Input to the U-Net is raw grayscale, NOT the synthetic B-R channel.
      The model learns its own feature representation.
    - Augmentation happens BEFORE patch extraction so spatial transforms
      do not produce black borders inside the 256×256 window.
    - Patch extraction uses a seeded RNG per __getitem__ call derived from
      the global seed + index, ensuring determinism across epochs when
      DataLoader uses a fixed seed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.augmentation import augment_pair


class DocCleanDataset(Dataset):
    """Dataset of (dirty, clean) synthetic image pairs for U-Net training.

    Args:
        data_dir (str | Path): Root directory containing ``dirty/`` and
            ``clean/`` subdirectories with matching filenames.
        patch_size (int): Side length (pixels) of the square patches to
            extract. Default: 256.
        augment (bool): Whether to apply ``augment_pair()`` before patch
            extraction. Typically ``True`` for training, ``False`` for
            validation. Default: ``True``.
        seed (int): Base seed for patch-location RNG. Combined with the
            sample index to guarantee per-sample determinism. Default: 42.

    Raises:
        FileNotFoundError: If ``dirty/`` or ``clean/`` subdirectory does
            not exist under ``data_dir``.
        ValueError: If no matching pairs are found, or if ``patch_size``
            is larger than the image dimensions.

    Returns (per ``__getitem__``):
        tuple[torch.Tensor, torch.Tensor]:
            - dirty_patch: shape ``(1, patch_size, patch_size)``, float32,
              values in ``[0, 1]``.
            - clean_patch: shape ``(1, patch_size, patch_size)``, float32,
              values in ``[0, 1]``.
    """

    def __init__(
        self,
        data_dir: str | Path,
        patch_size: int = 256,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.seed = seed

        dirty_dir = self.data_dir / "dirty"
        clean_dir = self.data_dir / "clean"

        if not dirty_dir.is_dir():
            raise FileNotFoundError(f"dirty/ directory not found: {dirty_dir}")
        if not clean_dir.is_dir():
            raise FileNotFoundError(f"clean/ directory not found: {clean_dir}")

        # Collect filenames present in BOTH directories (sorted for reproducibility)
        dirty_names = {p.name for p in dirty_dir.glob("*.png")}
        clean_names = {p.name for p in clean_dir.glob("*.png")}

        # Pair names follow the convention: dirty_NNNNNN.png <-> clean_NNNNNN.png
        # Match by replacing the prefix to find the counterpart.
        paired: list[tuple[Path, Path]] = []
        for name in sorted(dirty_names):
            counterpart = name.replace("dirty_", "clean_", 1)
            if counterpart in clean_names:
                paired.append((dirty_dir / name, clean_dir / counterpart))

        if not paired:
            raise ValueError(
                f"No matching dirty/clean pairs found in {self.data_dir}. "
                "Expected filenames like dirty_000000.png / clean_000000.png."
            )

        self._pairs = paired

    # ── Public interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        dirty_path, clean_path = self._pairs[index]

        dirty_bgr = _imread(dirty_path)
        clean_bgr = _imread(clean_path)

        # Validate image dimensions against patch size on first access.
        # (Images are assumed uniform; checking every call would be wasteful.)
        h, w = dirty_bgr.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            raise ValueError(
                f"Image {dirty_path.name} is {w}×{h} px but patch_size="
                f"{self.patch_size}. Image must be at least "
                f"{self.patch_size}×{self.patch_size}."
            )

        # Augmentation on full images BEFORE cropping to avoid border artefacts.
        if self.augment:
            # Deterministic per (epoch, index) when DataLoader seeds workers;
            # non-deterministic otherwise (training with shuffle=True).
            rng = np.random.default_rng(self.seed + index)
            dirty_bgr, clean_bgr = augment_pair(dirty_bgr, clean_bgr, rng)

        # Extract the same random patch from both images.
        patch_rng = np.random.default_rng(self.seed + index + len(self._pairs))
        dirty_patch, clean_patch = _extract_patch(
            dirty_bgr, clean_bgr, self.patch_size, patch_rng
        )

        # BGR → grayscale, normalise to [0, 1], add channel dim.
        dirty_tensor = _to_tensor(dirty_patch)
        clean_tensor = _to_tensor(clean_patch)

        return dirty_tensor, clean_tensor

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"DocCleanDataset("
            f"n_pairs={len(self._pairs)}, "
            f"patch_size={self.patch_size}, "
            f"augment={self.augment})"
        )


# ── Private helpers ───────────────────────────────────────────────────────────


def _imread(path: Path) -> np.ndarray:
    """Load an image from disk, handling non-ASCII paths safely.

    Args:
        path (Path): Image path.

    Returns:
        np.ndarray: BGR image, shape (H, W, 3), dtype uint8.

    Raises:
        FileNotFoundError: If the file cannot be read.
    """
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _extract_patch(
    dirty: np.ndarray,
    clean: np.ndarray,
    patch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a random crop of ``patch_size × patch_size`` from both images
    at the same location.

    Args:
        dirty (np.ndarray): BGR image, shape (H, W, 3), dtype uint8.
        clean (np.ndarray): BGR image, shape (H, W, 3), dtype uint8.
        patch_size (int): Side length of the square patch.
        rng (np.random.Generator): Seeded RNG; must be the same instance for
            both images to guarantee identical crop coordinates.

    Returns:
        tuple[np.ndarray, np.ndarray]: Cropped dirty and clean patches,
            each shape (patch_size, patch_size, 3), dtype uint8.
    """
    h, w = dirty.shape[:2]
    top = int(rng.integers(0, h - patch_size + 1))
    left = int(rng.integers(0, w - patch_size + 1))
    dirty_crop = dirty[top : top + patch_size, left : left + patch_size]
    clean_crop = clean[top : top + patch_size, left : left + patch_size]
    return dirty_crop, clean_crop


def _to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR uint8 image to a normalised single-channel float32 tensor.

    Steps:
        1. BGR → grayscale  (cv2.COLOR_BGR2GRAY)
        2. uint8 [0, 255]  → float32 [0.0, 1.0]
        3. Add channel dim → shape (1, H, W)

    Args:
        image_bgr (np.ndarray): BGR image, shape (H, W, 3), dtype uint8.

    Returns:
        torch.Tensor: shape (1, H, W), dtype float32, values in [0, 1].
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)  # (H, W) uint8
    normalized = gray.astype(np.float32) / 255.0  # (H, W) float32
    return torch.from_numpy(normalized).unsqueeze(0)  # (1, H, W)
