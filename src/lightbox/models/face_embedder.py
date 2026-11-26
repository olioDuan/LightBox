"""InsightFace antelopev2 (`glintr100.onnx`) 512-d ArcFace face embedder & 5-point alignment (SPEC §6.2, PLAN T18)."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np
from numpy.typing import NDArray

from lightbox.core.faces import (
    ANTELOPEV2_DEFAULT_URL,
    ANTELOPEV2_DIRNAME,
    download_antelopev2_archive,
)
from lightbox.paths import AppPaths, get_app_paths

FACE_EMBEDDING_DIM = 512
FACE_EMBEDDER_MODEL_ID = "insightface_antelopev2"
GLINTR100_FILENAME = "glintr100.onnx"
ARCFACE_INPUT_SIZE = (112, 112)

# Canonical 5-point ArcFace 112x112 reference template (left eye, right eye, nose, left mouth, right mouth)
ARCFACE_DST_LANDMARKS: NDArray[np.float32] = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class FaceEmbedderError(RuntimeError):
    """Raised when local InsightFace glintr100 face embedding cannot be performed."""


@runtime_checkable
class FaceEmbedder(Protocol):
    """Protocol for 512-d face identity embedders (real InsightFace ONNX or test fakes)."""

    model_id: str
    dim: int

    def embed_aligned_faces(
        self,
        aligned_rgb_112: Sequence[NDArray[np.uint8]],
    ) -> NDArray[np.float32]: ...


def _l2_normalize_rows(mat: NDArray[np.float32]) -> NDArray[np.float32]:
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return (arr / norm).astype(np.float32, copy=False) if norm > 0.0 else arr
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return (arr / np.maximum(norms, np.float32(1e-12))).astype(np.float32, copy=False)


def align_face_arcface(
    image_rgb: NDArray[np.uint8],
    bbox: tuple[float, float, float, float],
    landmarks: tuple[tuple[float, float], ...] | None = None,
    *,
    output_size: tuple[int, int] = ARCFACE_INPUT_SIZE,
) -> NDArray[np.uint8]:
    """Align a detected face to the 112x112 ArcFace template via 5-point similarity transform.

    Uses `cv2.estimateAffinePartial2D` when 5 landmarks from T05 are available, falling back
    to a padded bounding-box crop resized to `output_size` otherwise.
    """
    h, w = image_rgb.shape[:2]
    out_w, out_h = output_size
    if h <= 0 or w <= 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    if landmarks is not None and len(landmarks) == 5:
        src_pts = np.array(
            [[pt[0] * float(w), pt[1] * float(h)] for pt in landmarks],
            dtype=np.float32,
        )
        dst_pts = ARCFACE_DST_LANDMARKS.copy()
        if (out_w, out_h) != ARCFACE_INPUT_SIZE:
            dst_pts[:, 0] *= float(out_w) / float(ARCFACE_INPUT_SIZE[0])
            dst_pts[:, 1] *= float(out_h) / float(ARCFACE_INPUT_SIZE[1])

        matrix, _inliers = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.LMEDS
        )
        if matrix is None:
            matrix, _inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC
            )
        if matrix is not None:
            warped = cv2.warpAffine(
                image_rgb,
                matrix,
                (out_w, out_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            return np.asarray(warped, dtype=np.uint8)

    # Fallback: padded bounding-box crop resized to 112x112
    x1, y1, x2, y2 = bbox
    bw = max(1e-4, x2 - x1)
    bh = max(1e-4, y2 - y1)
    pad = 0.15
    px1 = max(0, int(round((x1 - bw * pad) * w)))
    py1 = max(0, int(round((y1 - bh * pad) * h)))
    px2 = min(w, max(px1 + 1, int(round((x2 + bw * pad) * w))))
    py2 = min(h, max(py1 + 1, int(round((y2 + bh * pad) * h))))
    crop = image_rgb[py1:py2, px1:px2]
    if crop.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return np.asarray(resized, dtype=np.uint8)


def find_face_embedder_model_path(app_paths: AppPaths | None = None) -> Path | None:
    """Locate `glintr100.onnx` under `<APP_DATA_DIR>/models/antelopev2/` or `<APP_DATA_DIR>/models/`."""
    paths = app_paths or get_app_paths()
    candidates = (
        paths.models_dir / ANTELOPEV2_DIRNAME / GLINTR100_FILENAME,
        paths.models_dir / GLINTR100_FILENAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def ensure_face_embedder_model(
    app_paths: AppPaths | None = None,
    *,
    allow_download: bool = True,
    download_url: str | None = None,
    timeout_s: float = 120.0,
) -> Path:
    """Return local path to `glintr100.onnx`, downloading `antelopev2.zip` on demand if allowed."""
    existing = find_face_embedder_model_path(app_paths)
    if existing is not None:
        return existing
    if not allow_download:
        paths = app_paths or get_app_paths()
        raise FaceEmbedderError(
            f"InsightFace glintr100.onnx not found under {paths.models_dir} and downloads are disabled."
        )

    paths = app_paths or get_app_paths()
    try:
        download_antelopev2_archive(
            paths,
            download_url=download_url,
            timeout_s=timeout_s,
        )
    except FaceEmbedderError:
        raise
    except Exception as exc:
        raise FaceEmbedderError(
            f"Failed to download InsightFace antelopev2 models "
            f"(check network or set LIGHTBOX_ANTELOPEV2_URL): {exc}"
        ) from exc

    resolved = find_face_embedder_model_path(paths)
    if resolved is None:
        raise FaceEmbedderError(
            f"Downloaded antelopev2 archive did not contain {GLINTR100_FILENAME}."
        )
    return resolved


class InsightFaceEmbedder:
    """Lazy, thread-safe InsightFace `antelopev2/glintr100.onnx` 512-d face identity embedder."""

    model_id: str = FACE_EMBEDDER_MODEL_ID
    dim: int = FACE_EMBEDDING_DIM

    def __init__(
        self,
        app_paths: AppPaths | None = None,
        *,
        model_path: Path | None = None,
        allow_download: bool = True,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> None:
        self.app_paths = app_paths or get_app_paths()
        self._explicit_model_path = model_path
        self.allow_download = allow_download
        self.providers = list(providers)
        self._session: Any = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session
        with self._lock:
            if self._session is not None:
                return self._session
            import onnxruntime as ort

            if self._explicit_model_path is not None:
                if not self._explicit_model_path.is_file():
                    raise FaceEmbedderError(
                        f"Face embedder model not found at {self._explicit_model_path}"
                    )
                model_path = self._explicit_model_path
            else:
                model_path = ensure_face_embedder_model(
                    self.app_paths,
                    allow_download=self.allow_download,
                )

            try:
                sess_options = ort.SessionOptions()
                # Silence verbose ORT VerifyOutputSizes warnings when glintr100 declares static batch=1
                sess_options.log_severity_level = 3
                self._session = ort.InferenceSession(
                    str(model_path),
                    sess_options=sess_options,
                    providers=self.providers,
                )
            except Exception as exc:
                raise FaceEmbedderError(
                    f"Failed to load InsightFace glintr100 ONNX model at {model_path}: {exc}"
                ) from exc
            return self._session

    def embed_aligned_faces(
        self,
        aligned_rgb_112: Sequence[NDArray[np.uint8]],
        *,
        batch_size: int = 16,
    ) -> NDArray[np.float32]:
        """Encode 112x112 RGB aligned face crops into L2-normalized `(N, 512)` float32 vectors."""
        if not aligned_rgb_112:
            return np.zeros((0, self.dim), dtype=np.float32)

        session = self._ensure_session()
        input_meta = session.get_inputs()[0]
        input_name = str(input_meta.name)
        output_name = str(session.get_outputs()[0].name)

        # Respect static batch=1 models cleanly without triggering ORT shape warnings
        input_shape = getattr(input_meta, "shape", None)
        fixed_batch_one = bool(input_shape and input_shape[0] == 1)
        step = 1 if fixed_batch_one else max(1, batch_size)

        chunks: list[NDArray[np.float32]] = []
        for start in range(0, len(aligned_rgb_112), step):
            batch_imgs = aligned_rgb_112[start : start + step]
            prepared: list[NDArray[np.float32]] = []
            for img in batch_imgs:
                if img.shape[:2] != (ARCFACE_INPUT_SIZE[1], ARCFACE_INPUT_SIZE[0]):
                    img = np.asarray(
                        cv2.resize(img, ARCFACE_INPUT_SIZE, interpolation=cv2.INTER_LINEAR),
                        dtype=np.uint8,
                    )
                # ArcFace normalization: RGB float32 `(x - 127.5) / 127.5`, NCHW
                arr = (img.astype(np.float32) - np.float32(127.5)) / np.float32(127.5)
                prepared.append(np.transpose(arr, (2, 0, 1)))
            blob = np.stack(prepared, axis=0).astype(np.float32, copy=False)
            raw = session.run([output_name], {input_name: blob})[0]
            chunks.append(_l2_normalize_rows(np.asarray(raw, dtype=np.float32)))
        return np.concatenate(chunks, axis=0)


_FACE_EMBEDDER_CACHE_LOCK = threading.Lock()
_FACE_EMBEDDER_CACHE: dict[tuple[str, bool, tuple[str, ...]], InsightFaceEmbedder] = {}


def get_shared_face_embedder(
    app_paths: AppPaths | None = None,
    *,
    allow_download: bool = True,
    providers: Sequence[str] = ("CPUExecutionProvider",),
) -> InsightFaceEmbedder:
    """Return a process-wide shared `InsightFaceEmbedder` so `glintr100.onnx` loads at most once."""
    paths = app_paths or get_app_paths()
    key = (str(paths.models_dir.resolve()), allow_download, tuple(providers))
    with _FACE_EMBEDDER_CACHE_LOCK:
        cached = _FACE_EMBEDDER_CACHE.get(key)
        if cached is None:
            cached = InsightFaceEmbedder(
                paths,
                allow_download=allow_download,
                providers=providers,
            )
            _FACE_EMBEDDER_CACHE[key] = cached
        return cached


def clear_shared_face_embedder_cache() -> None:
    """Clear the process-level shared `InsightFaceEmbedder` cache (primarily for tests)."""
    with _FACE_EMBEDDER_CACHE_LOCK:
        _FACE_EMBEDDER_CACHE.clear()


__all__ = [
    "ANTELOPEV2_DEFAULT_URL",
    "ANTELOPEV2_DIRNAME",
    "ARCFACE_DST_LANDMARKS",
    "ARCFACE_INPUT_SIZE",
    "FACE_EMBEDDER_MODEL_ID",
    "FACE_EMBEDDING_DIM",
    "GLINTR100_FILENAME",
    "FaceEmbedder",
    "FaceEmbedderError",
    "InsightFaceEmbedder",
    "align_face_arcface",
    "clear_shared_face_embedder_cache",
    "ensure_face_embedder_model",
    "find_face_embedder_model_path",
    "get_shared_face_embedder",
]
