"""File checksums are streamed in chunks so large FITS files stay out of memory."""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK_SIZE = 1024 * 1024  # 1 MiB


def sha256_file(path: str | Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Return a file's SHA-256 as a lowercase hexadecimal string."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
