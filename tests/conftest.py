"""
Shared pytest fixtures for DocClean-Net tests.

Fixtures
--------
rng
    A seeded numpy.random.Generator. Use this everywhere instead of
    np.random.xxx to guarantee reproducible synthetic data.

white_image_gray
    Small (64×64) white grayscale image. dtype uint8.
    Useful as a blank canvas for generator tests.

white_image_bgr
    Small (64×64) white BGR image. dtype uint8.
    Useful for pipeline tests that expect a 3-channel input.

noisy_gray_image
    Small (64×64) grayscale image with mild Gaussian noise.
    Simulates a paper background without any strokes.

tmp_output_dir
    Temporary directory (pathlib.Path). Cleaned up after each test.
    Use for any test that writes files to disk.
"""

from pathlib import Path

import numpy as np
import pytest


# ── Random state ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Seeded Generator shared across the entire test session.

    Session scope is intentional: generators must be stateless with
    respect to the rng they receive, so sharing one instance across tests
    exposes any hidden state mutation.
    """
    return np.random.default_rng(seed=42)


# ── Synthetic images ──────────────────────────────────────────────────────────

_IMG_H = 64
_IMG_W = 64


@pytest.fixture(scope="session")
def white_image_gray() -> np.ndarray:
    """White grayscale image, shape (64, 64), dtype uint8."""
    return np.full((_IMG_H, _IMG_W), fill_value=255, dtype=np.uint8)


@pytest.fixture(scope="session")
def white_image_bgr() -> np.ndarray:
    """White BGR image, shape (64, 64, 3), dtype uint8."""
    return np.full((_IMG_H, _IMG_W, 3), fill_value=255, dtype=np.uint8)


@pytest.fixture(scope="session")
def noisy_gray_image(rng: np.random.Generator) -> np.ndarray:
    """Grayscale image with mild Gaussian noise, shape (64, 64), dtype uint8.

    Simulates a paper background (mean=245, sigma=4) without any strokes.
    """
    noise = rng.normal(loc=245.0, scale=4.0, size=(_IMG_H, _IMG_W))
    return np.clip(noise, 0, 255).astype(np.uint8)


# ── Filesystem ────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for tests that write files.

    Uses pytest's built-in tmp_path (function scope) so each test gets
    a clean directory and cleanup is automatic.
    """
    out = tmp_path / "output"
    out.mkdir()
    return out
