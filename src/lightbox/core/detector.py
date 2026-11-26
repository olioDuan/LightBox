"""Detector extension protocol and immutable per-photo context (SPEC §4.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from lightbox.core.exif import NormalizedExif

ENGINE_VERSION = "0.5.0"


@dataclass(frozen=True, slots=True)
class DetectedFace:
    """Single detected face with normalized `[0, 1]` bbox and 5-point landmarks."""

    idx: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized in [0, 1]
    landmarks: tuple[tuple[float, float], ...] | None  # 5 points: LE, RE, nose, LM, RM
    det_score: float
    sharpness: float
    area_ratio: float


@dataclass(slots=True)
class PhotoContext:
    """Inputs shared across pure `Detector` implementations for a single photo."""

    photo_id: str
    content_fp: str
    analysis_img: NDArray[np.uint8] | None = None
    loupe_rgb: NDArray[np.uint8] | None = None
    analysis_path: Path | None = None
    loupe_path: Path | None = None
    file_path: Path | None = None
    exif: NormalizedExif | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    faces: tuple[DetectedFace, ...] = ()


@runtime_checkable
class Detector(Protocol):
    """Pure detector interface cached by `(content_fp, version)` (SPEC §4.4)."""

    name: str
    version: str
    requires: set[str]

    def run(self, ctx: PhotoContext) -> dict[str, Any]:
        """Execute deterministic detection on `ctx` and return JSON-serializable results."""
        ...


__all__ = ["DetectedFace", "Detector", "ENGINE_VERSION", "PhotoContext"]
