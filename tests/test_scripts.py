"""
tests/test_scripts.py
=====================
Tests for scripts/download_model.py.

Fast tests never touch the network: they exercise hash computation,
checkpoint validation, and the idempotence logic of ensure_checkpoint()
using small local fake files. The real end-to-end download is covered by
a single @pytest.mark.slow test that skips cleanly when offline.
"""

import hashlib
import urllib.error
from pathlib import Path

import pytest

from scripts.download_model import (
    EXPECTED_SHA256,
    EXPECTED_SIZE,
    compute_sha256,
    ensure_checkpoint,
    is_valid_checkpoint,
)


@pytest.fixture()
def fake_checkpoint(tmp_path: Path) -> Path:
    """A small local file standing in for a downloaded checkpoint."""
    path = tmp_path / "fake_best.pt"
    path.write_bytes(b"not a real torch checkpoint, just bytes")
    return path


class TestComputeSha256:
    def test_compute_sha256_matches_hashlib_reference(
        self, fake_checkpoint: Path
    ) -> None:
        """Chunked hashing must equal a one-shot hashlib digest."""
        expected = hashlib.sha256(fake_checkpoint.read_bytes()).hexdigest()
        assert compute_sha256(fake_checkpoint) == expected

    def test_compute_sha256_returns_lowercase_hex(self, fake_checkpoint: Path) -> None:
        """Digest must be 64 lowercase hex characters."""
        digest = compute_sha256(fake_checkpoint)
        assert len(digest) == 64
        assert digest == digest.lower()
        assert all(c in "0123456789abcdef" for c in digest)


class TestIsValidCheckpoint:
    def test_is_valid_checkpoint_accepts_matching_file(
        self, fake_checkpoint: Path
    ) -> None:
        """A file whose size and hash both match is valid."""
        size = fake_checkpoint.stat().st_size
        digest = compute_sha256(fake_checkpoint)
        assert is_valid_checkpoint(
            fake_checkpoint, expected_sha256=digest, expected_size=size
        )

    def test_is_valid_checkpoint_accepts_uppercase_expected_hash(
        self, fake_checkpoint: Path
    ) -> None:
        """Expected hash comparison must be case-insensitive (PowerShell's
        Get-FileHash prints uppercase)."""
        size = fake_checkpoint.stat().st_size
        digest = compute_sha256(fake_checkpoint).upper()
        assert is_valid_checkpoint(
            fake_checkpoint, expected_sha256=digest, expected_size=size
        )

    def test_is_valid_checkpoint_rejects_missing_file(self, tmp_path: Path) -> None:
        """A nonexistent path is never valid."""
        assert not is_valid_checkpoint(tmp_path / "does_not_exist.pt")

    def test_is_valid_checkpoint_rejects_wrong_size(
        self, fake_checkpoint: Path
    ) -> None:
        """Size mismatch fails before any hashing happens."""
        digest = compute_sha256(fake_checkpoint)
        wrong_size = fake_checkpoint.stat().st_size + 1
        assert not is_valid_checkpoint(
            fake_checkpoint, expected_sha256=digest, expected_size=wrong_size
        )

    def test_is_valid_checkpoint_rejects_wrong_hash(
        self, fake_checkpoint: Path
    ) -> None:
        """Correct size but wrong hash must be rejected."""
        size = fake_checkpoint.stat().st_size
        assert not is_valid_checkpoint(
            fake_checkpoint, expected_sha256="0" * 64, expected_size=size
        )


class TestEnsureCheckpointIdempotence:
    def test_ensure_checkpoint_skips_download_when_file_valid(
        self, fake_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the destination already verifies, no network call is made."""
        import scripts.download_model as dm

        monkeypatch.setattr(dm, "EXPECTED_SHA256", compute_sha256(fake_checkpoint))
        monkeypatch.setattr(dm, "EXPECTED_SIZE", fake_checkpoint.stat().st_size)

        def _fail_if_called(url: str, dest: Path) -> None:
            raise AssertionError(
                "download_file() must not be called for a valid local file"
            )

        monkeypatch.setattr(dm, "download_file", _fail_if_called)

        result = dm.ensure_checkpoint(dest=fake_checkpoint)
        assert result == fake_checkpoint

    def test_ensure_checkpoint_redownloads_invalid_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file with a mismatching hash triggers a (mocked) re-download,
        and the freshly written content is then verified."""
        import scripts.download_model as dm

        good_content = b"the real published checkpoint bytes"
        dest = tmp_path / "best.pt"
        dest.write_bytes(b"stale or corrupt content")

        monkeypatch.setattr(
            dm, "EXPECTED_SHA256", hashlib.sha256(good_content).hexdigest()
        )
        monkeypatch.setattr(dm, "EXPECTED_SIZE", len(good_content))

        def _fake_download(url: str, dest_path: Path) -> None:
            dest_path.write_bytes(good_content)

        monkeypatch.setattr(dm, "download_file", _fake_download)

        result = dm.ensure_checkpoint(dest=dest)
        assert result == dest
        assert dest.read_bytes() == good_content

    def test_ensure_checkpoint_raises_when_download_fails_verification(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A downloaded file that still doesn't verify raises RuntimeError."""
        import scripts.download_model as dm

        dest = tmp_path / "best.pt"

        monkeypatch.setattr(dm, "EXPECTED_SHA256", "f" * 64)
        monkeypatch.setattr(dm, "EXPECTED_SIZE", 10)

        def _fake_bad_download(url: str, dest_path: Path) -> None:
            dest_path.write_bytes(b"0123456789")  # right size, wrong hash

        monkeypatch.setattr(dm, "download_file", _fake_bad_download)

        with pytest.raises(RuntimeError, match="Integrity check failed"):
            dm.ensure_checkpoint(dest=dest)


@pytest.mark.slow
def test_download_model_end_to_end_real_release(tmp_path: Path) -> None:
    """Full download from the actual GitHub Release, verified against the
    published hash. Skips (not fails) when the network is unavailable or
    the release asset does not exist yet."""
    dest = tmp_path / "best.pt"
    try:
        result = ensure_checkpoint(dest=dest)
    except urllib.error.URLError as exc:
        pytest.skip(f"Network unavailable or release not published: {exc}")
    assert result.is_file()
    assert result.stat().st_size == EXPECTED_SIZE
    assert compute_sha256(result) == EXPECTED_SHA256
