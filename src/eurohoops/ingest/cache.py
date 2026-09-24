"""Gzip raw-cache files, written atomically so a partial file is never mistaken for a cached one."""

import gzip
from pathlib import Path


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(payload, mtime=0))
    tmp.replace(path)


def read_cached(path: Path) -> bytes:
    return gzip.decompress(path.read_bytes())
