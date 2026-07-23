"""Where a brand's trained artifact is saved and loaded.

The guard talks to this seam instead of touching the filesystem directly, so a
database or object-store backend is a drop-in later: implement the three
``ArtifactStore`` methods and pass it to :class:`~simasia.guard.SimasiaGuard`.
``FileArtifactStore`` is the default and keeps the original on-disk behaviour.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Protocol

import joblib


class ArtifactStore(Protocol):
    """Reads and writes one artifact per brand."""

    def save(self, brand_id: str, artifact: object) -> None: ...
    def load(self, brand_id: str) -> object | None: ...
    def exists(self, brand_id: str) -> bool: ...


def serialize_artifact(artifact: object) -> bytes:
    """Turn an artifact into bytes for storing in a DB blob (or anywhere)."""
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer)
    return buffer.getvalue()


def deserialize_artifact(blob: bytes) -> object:
    """Rebuild an artifact from bytes produced by :func:`serialize_artifact`."""
    return joblib.load(io.BytesIO(blob))


class FileArtifactStore:
    """Save each brand's artifact as ``simasia_<brand_id>_head.joblib``."""

    def __init__(self, artifact_dir: str | os.PathLike[str] = ".") -> None:
        self.artifact_dir = Path(artifact_dir)

    def path_for(self, brand_id: str) -> Path:
        return self.artifact_dir / f"simasia_{brand_id}_head.joblib"

    def save(self, brand_id: str, artifact: object) -> None:
        path = self.path_for(brand_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)

    def load(self, brand_id: str) -> object | None:
        path = self.path_for(brand_id)
        if not path.exists():
            return None
        return joblib.load(path)

    def exists(self, brand_id: str) -> bool:
        return self.path_for(brand_id).exists()
