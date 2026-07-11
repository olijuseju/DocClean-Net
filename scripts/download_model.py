#!/usr/bin/env python3
"""
scripts/download_model.py
=========================
Downloads the trained U-Net checkpoint (best.pt) from the project's GitHub
Release into checkpoints/best.pt, verifying its integrity via SHA-256.

The checkpoint is gitignored (binary, ~1.9 MB) and published as a Release
asset instead, so a fresh clone needs this one-time step before running
inference, the GUI, or the benchmark:

    python scripts/download_model.py

Idempotent: if checkpoints/best.pt already exists and its SHA-256 matches
the expected hash, nothing is downloaded. A file with a mismatching hash
(corrupt or outdated) is re-downloaded after an explicit warning.

Only stdlib networking (urllib.request) is used — no new dependencies.
"""

import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MODEL_URL = "https://github.com/olijuseju/DocClean-Net/releases/download/v1.0.0/best.pt"
EXPECTED_SHA256 = "7c9913dcaaca3e778a12e86802ab8ff4c3d0f91b9e5a4f88d8095fa1d4c645c9"
EXPECTED_SIZE = 1_961_321  # bytes
DEFAULT_DEST = _REPO_ROOT / "checkpoints" / "best.pt"

_CHUNK_SIZE = 64 * 1024  # 64 KiB read chunks for hashing and downloading


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file, reading in chunks.

    Args:
        path: File to hash. Must exist.

    Returns:
        Lowercase hexadecimal SHA-256 digest (64 characters).
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def is_valid_checkpoint(
    path: Path,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> bool:
    """Check whether a local checkpoint file matches the published one.

    The (cheap) size check runs first so an obviously wrong file is
    rejected without hashing it.

    Args:
        path: Candidate checkpoint file.
        expected_sha256: Expected SHA-256 hex digest (case-insensitive).
            Defaults to the module-level EXPECTED_SHA256 at call time.
        expected_size: Expected file size in bytes. Defaults to the
            module-level EXPECTED_SIZE at call time.

    Returns:
        True if the file exists, has the expected size, and its SHA-256
        digest matches. False otherwise.
    """
    if expected_sha256 is None:
        expected_sha256 = EXPECTED_SHA256
    if expected_size is None:
        expected_size = EXPECTED_SIZE
    if not path.is_file():
        return False
    if path.stat().st_size != expected_size:
        return False
    return compute_sha256(path) == expected_sha256.lower()


def download_file(url: str, dest: Path) -> None:
    """Download a URL to a local path with a tqdm progress bar.

    Downloads to a temporary sibling file (dest + '.part') and renames on
    success, so an interrupted download never leaves a truncated file at
    the final destination.

    Args:
        url: Source URL (HTTPS).
        dest: Final destination path. Parent directories are created.

    Raises:
        urllib.error.URLError: On network failure or HTTP error status.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")

    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        with (
            open(tmp_path, "wb") as out,
            tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc=dest.name,
            ) as bar,
        ):
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))

    tmp_path.replace(dest)


def ensure_checkpoint(dest: Path = DEFAULT_DEST) -> Path:
    """Ensure a verified checkpoint exists at `dest`, downloading if needed.

    Args:
        dest: Where the checkpoint should live
            (default: <repo>/checkpoints/best.pt).

    Returns:
        Path to the verified checkpoint.

    Raises:
        RuntimeError: If the downloaded file fails integrity verification
            (hash/size mismatch — corrupted transfer or tampered asset).
        urllib.error.URLError: On network failure.
    """
    if is_valid_checkpoint(dest):
        print(f"✓ {dest} already present and verified — nothing to do.")
        return dest

    if dest.is_file():
        print(
            f"⚠ {dest} exists but does not match the published checkpoint "
            "(corrupt or outdated). Re-downloading."
        )

    print(f"Downloading {MODEL_URL}")
    download_file(MODEL_URL, dest)

    if not is_valid_checkpoint(dest):
        actual = compute_sha256(dest)
        raise RuntimeError(
            f"Integrity check failed for {dest}.\n"
            f"  expected SHA-256: {EXPECTED_SHA256}\n"
            f"  actual   SHA-256: {actual}\n"
            "The download may be corrupted — delete the file and retry."
        )

    print(f"✓ Downloaded and verified: {dest} ({EXPECTED_SIZE:,} bytes)")
    return dest


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    try:
        ensure_checkpoint()
    except urllib.error.URLError as exc:
        print(f"✗ Network error: {exc}", file=sys.stderr)
        print(
            "Check your connection, or download manually from:\n"
            f"  {MODEL_URL}\n"
            f"and place the file at {DEFAULT_DEST}",
            file=sys.stderr,
        )
        return 1
    except RuntimeError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
