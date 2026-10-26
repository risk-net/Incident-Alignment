"""Immutable asset fingerprints; deliberately no writes to material datasets."""
from dataclasses import dataclass
from pathlib import Path
import hashlib

@dataclass(frozen=True)
class FileAsset:
    path: Path
    sha256: str

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def verify_sha256(path: str | Path, expected: str) -> FileAsset:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(f'checksum mismatch for {path}: expected={expected}, actual={actual}')
    return FileAsset(Path(path), actual)
