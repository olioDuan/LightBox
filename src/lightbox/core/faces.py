"""InsightFace antelopev2 (det_10g.onnx) face detection, 5-point landmarks, and ROI sharpness (SPEC §4.2)."""

from __future__ import annotations

import io
import math
import os
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from lightbox.core.detector import ENGINE_VERSION, DetectedFace, PhotoContext
from lightbox.core.metrics import compute_region_sharpness
from lightbox.paths import AppPaths, get_app_paths

ANTELOPEV2_DIRNAME = "antelopev2"
DET_MODEL_FILENAME = "det_10g.onnx"
DEFAULT_DET_INPUT_SIZE = (640, 640)

# Decoupled thresholds (SPEC §4.2, PLAN T05):
# - Persist every valid face with area_ratio >= 0.003 (0.3%) + 5-point landmarks for T18 group photos
# - Only evaluate face out-of-focus on subject faces with area_ratio >= 0.015 (1.5%)
FACE_STORE_MIN_AREA_RATIO = 0.003
FACE_SUBJECT_MIN_AREA_RATIO = 0.015
FACE_DET_SCORE_THRESHOLD = 0.50
FACE_NMS_IOU_THRESHOLD = 0.40

# Subject face out-of-focus comparison thresholds (NEVER escalates to `red`, max `yellow`)
FACE_OOF_ABS_THRESHOLD = 7.0
FACE_OOF_RELATIVE_RATIO = 0.45

# Official InsightFace antelopev2 release archive URL (on-demand user download; never bundled)
ANTELOPEV2_DEFAULT_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
)


@dataclass(frozen=True, slots=True)
class RawFaceDetection:
    """Raw pixel-space face detection before normalization and sharpness scoring."""

    bbox_px: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    landmarks_px: tuple[tuple[float, float], ...] | None  # 5 keypoints in pixels
    det_score: float


class FaceBackend(Protocol):
    """Injectable low-level face detector backend (SCRFD ONNX in production, stub in unit tests)."""

    @property
    def available(self) -> bool: ...

    def detect_raw(self, image_rgb: NDArray[np.uint8]) -> list[RawFaceDetection]: ...


def find_face_model_path(app_paths: AppPaths | None = None) -> Path | None:
    """Locate `det_10g.onnx` under `<APP_DATA_DIR>/models/antelopev2/` or `<APP_DATA_DIR>/models/`."""
    paths = app_paths or get_app_paths()
    candidates = (
        paths.models_dir / ANTELOPEV2_DIRNAME / DET_MODEL_FILENAME,
        paths.models_dir / DET_MODEL_FILENAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def ensure_face_model(
    app_paths: AppPaths | None = None,
    *,
    allow_download: bool = False,
    download_url: str | None = None,
    timeout_s: float = 60.0,
) -> Path | None:
    """Return local path to `det_10g.onnx`, optionally downloading `antelopev2.zip` on demand."""
    existing = find_face_model_path(app_paths)
    if existing is not None:
        return existing
    if not allow_download:
        return None

    paths = app_paths or get_app_paths()
    target_dir = paths.models_dir / ANTELOPEV2_DIRNAME
    target_dir.mkdir(parents=True, exist_ok=True)

    url = download_url or os.environ.get("LIGHTBOX_ANTELOPEV2_URL") or ANTELOPEV2_DEFAULT_URL
    if not url.startswith(("https://", "http://")):
        raise ValueError(f"Unsupported model download URL scheme: {url}")

    req = urllib.request.Request(url, headers={"User-Agent": "lightbox-cull/0.1"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        archive_bytes = resp.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            basename = Path(info.filename).name
            if not basename.endswith(".onnx"):
                continue
            dest = target_dir / basename
            dest.write_bytes(zf.read(info))

    return find_face_model_path(paths)


def _nms_xyxy(
    boxes: NDArray[np.float32],
    scores: NDArray[np.float32],
    iou_threshold: float,
) -> list[int]:
    """Greedy Non-Maximum Suppression over `(N, 4)` `x1,y1,x2,y2` boxes."""
    if boxes.shape[0] == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1 + 1.0) * np.maximum(0.0, y2 - y1 + 1.0)
    order = np.argsort(scores)[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1 + 1.0) * np.maximum(0.0, yy2 - yy1 + 1.0)
        iou = inter / np.maximum(1e-6, areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]
    return keep


def decode_scrfd_outputs(
    outputs: Sequence[NDArray[np.float32]],
    *,
    input_size: tuple[int, int] = DEFAULT_DET_INPUT_SIZE,
    det_scale: float = 1.0,
    score_threshold: float = FACE_DET_SCORE_THRESHOLD,
    nms_iou_threshold: float = FACE_NMS_IOU_THRESHOLD,
) -> list[RawFaceDetection]:
    """Decode InsightFace SCRFD (`det_10g.onnx`) multi-stride score/bbox/kps tensors + NMS."""
    num_outputs = len(outputs)
    if num_outputs == 9:
        fmc = 3
        strides = (8, 16, 32)
        num_anchors = 2
        has_kps = True
    elif num_outputs == 6:
        fmc = 3
        strides = (8, 16, 32)
        num_anchors = 2
        has_kps = False
    else:
        raise ValueError(f"Unexpected SCRFD output count: {num_outputs} (expected 6 or 9)")

    input_w, input_h = input_size
    all_scores: list[NDArray[np.float32]] = []
    all_bboxes: list[NDArray[np.float32]] = []
    all_kps: list[NDArray[np.float32]] = []

    for idx, stride in enumerate(strides):
        scores = np.asarray(outputs[idx], dtype=np.float32).reshape(-1)
        bbox_preds = np.asarray(outputs[idx + fmc], dtype=np.float32).reshape(-1, 4) * float(stride)
        kps_preds = (
            np.asarray(outputs[idx + fmc * 2], dtype=np.float32).reshape(-1, 10) * float(stride)
            if has_kps
            else None
        )

        feat_h = input_h // stride
        feat_w = input_w // stride
        grid_y, grid_x = np.mgrid[:feat_h, :feat_w]
        centers = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2).astype(np.float32) * float(
            stride
        )
        if num_anchors > 1:
            centers = np.repeat(centers, num_anchors, axis=0)

        if centers.shape[0] != scores.shape[0]:
            raise ValueError(
                f"SCRFD stride {stride} anchor mismatch: {centers.shape[0]} vs {scores.shape[0]}"
            )

        pos_inds = np.where(scores >= score_threshold)[0]
        if pos_inds.size == 0:
            continue

        pos_centers = centers[pos_inds]
        pos_bbox = bbox_preds[pos_inds]
        x1 = pos_centers[:, 0] - pos_bbox[:, 0]
        y1 = pos_centers[:, 1] - pos_bbox[:, 1]
        x2 = pos_centers[:, 0] + pos_bbox[:, 2]
        y2 = pos_centers[:, 1] + pos_bbox[:, 3]
        decoded_boxes = (np.stack([x1, y1, x2, y2], axis=-1) / np.float32(det_scale)).astype(
            np.float32, copy=False
        )

        all_scores.append(scores[pos_inds])
        all_bboxes.append(decoded_boxes)

        if kps_preds is not None:
            pos_kps = kps_preds[pos_inds]
            decoded_kps = np.empty((pos_inds.size, 5, 2), dtype=np.float32)
            for kp_idx in range(5):
                decoded_kps[:, kp_idx, 0] = (
                    pos_centers[:, 0] + pos_kps[:, kp_idx * 2]
                ) / float(det_scale)
                decoded_kps[:, kp_idx, 1] = (
                    pos_centers[:, 1] + pos_kps[:, kp_idx * 2 + 1]
                ) / float(det_scale)
            all_kps.append(decoded_kps)

    if not all_bboxes:
        return []

    boxes_cat = np.concatenate(all_bboxes, axis=0)
    scores_cat = np.concatenate(all_scores, axis=0)
    kps_cat = np.concatenate(all_kps, axis=0) if all_kps else None

    keep_indices = _nms_xyxy(boxes_cat, scores_cat, nms_iou_threshold)
    detections: list[RawFaceDetection] = []
    for k in keep_indices:
        box = boxes_cat[k]
        lm_tuple: tuple[tuple[float, float], ...] | None = None
        if kps_cat is not None:
            lm_tuple = tuple((float(pt[0]), float(pt[1])) for pt in kps_cat[k])
        detections.append(
            RawFaceDetection(
                bbox_px=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                landmarks_px=lm_tuple,
                det_score=float(scores_cat[k]),
            )
        )
    return detections


class ScrfdOnnxBackend:
    """ONNX Runtime backend for InsightFace `antelopev2/det_10g.onnx` (CPU EP default)."""

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        app_paths: AppPaths | None = None,
        input_size: tuple[int, int] = DEFAULT_DET_INPUT_SIZE,
        score_threshold: float = FACE_DET_SCORE_THRESHOLD,
        nms_iou_threshold: float = FACE_NMS_IOU_THRESHOLD,
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> None:
        self.model_path = model_path or find_face_model_path(app_paths)
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.providers = list(providers)
        self._session: Any = None

    @property
    def available(self) -> bool:
        return self.model_path is not None and self.model_path.is_file()

    def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session
        if not self.available:
            return None
        import onnxruntime as ort

        assert self.model_path is not None
        self._session = ort.InferenceSession(str(self.model_path), providers=self.providers)
        return self._session

    def detect_raw(self, image_rgb: NDArray[np.uint8]) -> list[RawFaceDetection]:
        session = self._ensure_session()
        if session is None:
            return []

        h, w = image_rgb.shape[:2]
        in_w, in_h = self.input_size
        im_ratio = float(h) / float(w)
        model_ratio = float(in_h) / float(in_w)
        if im_ratio > model_ratio:
            new_h = in_h
            new_w = max(1, int(round(new_h / im_ratio)))
        else:
            new_w = in_w
            new_h = max(1, int(round(new_w * im_ratio)))
        det_scale = float(new_h) / float(h)

        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w, :] = resized

        # SCRFD normalization: RGB float32 `(x - 127.5) / 128.0`, NCHW
        blob = (canvas.astype(np.float32) - 127.5) / 128.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        input_name = str(session.get_inputs()[0].name)
        outputs = session.run(None, {input_name: blob})
        return decode_scrfd_outputs(
            [np.asarray(out, dtype=np.float32) for out in outputs],
            input_size=self.input_size,
            det_scale=det_scale,
            score_threshold=self.score_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
        )


def score_and_filter_faces(
    raw_detections: Sequence[RawFaceDetection],
    analysis_gray: NDArray[np.uint8],
    *,
    noise_sigma: float = 0.0,
    min_area_ratio: float = FACE_STORE_MIN_AREA_RATIO,
    min_det_score: float = FACE_DET_SCORE_THRESHOLD,
) -> tuple[DetectedFace, ...]:
    """Normalize detected faces, compute per-face ROI sharpness, and filter by storage threshold."""
    h, w = analysis_gray.shape[:2]
    frame_area = float(h * w)
    if frame_area <= 0.0:
        return ()

    scored: list[DetectedFace] = []
    for det in raw_detections:
        if det.det_score < min_det_score:
            continue
        x1_px = max(0.0, min(float(w), det.bbox_px[0]))
        y1_px = max(0.0, min(float(h), det.bbox_px[1]))
        x2_px = max(0.0, min(float(w), det.bbox_px[2]))
        y2_px = max(0.0, min(float(h), det.bbox_px[3]))
        bw = x2_px - x1_px
        bh = y2_px - y1_px
        if bw <= 1.0 or bh <= 1.0:
            continue

        area_ratio = (bw * bh) / frame_area
        if area_ratio < min_area_ratio:
            continue

        ix1, iy1 = int(math.floor(x1_px)), int(math.floor(y1_px))
        ix2, iy2 = int(math.ceil(x2_px)), int(math.ceil(y2_px))
        roi = analysis_gray[iy1:iy2, ix1:ix2]
        face_sharp = compute_region_sharpness(roi, noise_sigma=noise_sigma)

        norm_bbox = (
            round(x1_px / float(w), 6),
            round(y1_px / float(h), 6),
            round(x2_px / float(w), 6),
            round(y2_px / float(h), 6),
        )
        norm_lm: tuple[tuple[float, float], ...] | None = None
        if det.landmarks_px is not None:
            norm_lm = tuple(
                (
                    round(max(0.0, min(1.0, pt[0] / float(w))), 6),
                    round(max(0.0, min(1.0, pt[1] / float(h))), 6),
                )
                for pt in det.landmarks_px
            )

        scored.append(
            DetectedFace(
                idx=0,  # assigned after stable sorting
                bbox=norm_bbox,
                landmarks=norm_lm,
                det_score=round(det.det_score, 4),
                sharpness=round(face_sharp, 4),
                area_ratio=round(area_ratio, 6),
            )
        )

    # Sort largest face first, tie-break by score then left-to-right
    scored.sort(key=lambda f: (-f.area_ratio, -f.det_score, f.bbox[0], f.bbox[1]))
    return tuple(
        DetectedFace(
            idx=idx,
            bbox=face.bbox,
            landmarks=face.landmarks,
            det_score=face.det_score,
            sharpness=face.sharpness,
            area_ratio=face.area_ratio,
        )
        for idx, face in enumerate(scored)
    )


def evaluate_subject_faces_oof(
    faces: Sequence[DetectedFace],
    *,
    sharp_p95: float,
    subject_min_area_ratio: float = FACE_SUBJECT_MIN_AREA_RATIO,
) -> tuple[bool, float | None]:
    """Evaluate subject faces (`area_ratio >= 1.5%`) for out-of-focus warning (`yellow` max).

    Small background/group faces (`0.003 <= area_ratio < 0.015`) are stored for T18 people
    classification but never trigger out-of-focus warnings.
    """
    subject_faces = [f for f in faces if f.area_ratio >= subject_min_area_ratio]
    if not subject_faces:
        return False, None

    face_sharp_min = min(f.sharpness for f in subject_faces)
    is_oof = face_sharp_min < FACE_OOF_ABS_THRESHOLD or (
        sharp_p95 >= 12.0 and face_sharp_min < sharp_p95 * FACE_OOF_RELATIVE_RATIO
    )
    return is_oof, round(face_sharp_min, 4)


class FaceDetector:
    """`Detector` implementation wrapping InsightFace `antelopev2/det_10g.onnx`."""

    name: str = "faces"
    version: str = ENGINE_VERSION
    requires: set[str] = {"analysis_img"}

    def __init__(
        self,
        *,
        backend: FaceBackend | None = None,
        app_paths: AppPaths | None = None,
    ) -> None:
        self._backend: FaceBackend = backend or ScrfdOnnxBackend(app_paths=app_paths)

    @property
    def model_available(self) -> bool:
        return self._backend.available

    def run(self, ctx: PhotoContext) -> dict[str, Any]:
        if ctx.analysis_img is None:
            raise ValueError(f"PhotoContext for {ctx.photo_id} missing `analysis_img`")

        if not self._backend.available:
            return {
                "faces": (),
                "face_count": 0,
                "face_sharp_min": None,
                "face_out_of_focus": False,
                "model_available": False,
            }

        if ctx.loupe_rgb is not None:
            image_rgb = ctx.loupe_rgb
            # Scale raw pixel detections from loupe space to analysis space if dimensions differ
            lh, lw = image_rgb.shape[:2]
            ah, aw = ctx.analysis_img.shape[:2]
            raw_dets = self._backend.detect_raw(image_rgb)
            if (lh, lw) != (ah, aw) and lw > 0 and lh > 0:
                sx = float(aw) / float(lw)
                sy = float(ah) / float(lh)
                raw_dets = [
                    RawFaceDetection(
                        bbox_px=(
                            d.bbox_px[0] * sx,
                            d.bbox_px[1] * sy,
                            d.bbox_px[2] * sx,
                            d.bbox_px[3] * sy,
                        ),
                        landmarks_px=(
                            tuple((pt[0] * sx, pt[1] * sy) for pt in d.landmarks_px)
                            if d.landmarks_px is not None
                            else None
                        ),
                        det_score=d.det_score,
                    )
                    for d in raw_dets
                ]
        else:
            image_rgb = np.asarray(
                cv2.cvtColor(ctx.analysis_img, cv2.COLOR_GRAY2RGB), dtype=np.uint8
            )
            raw_dets = self._backend.detect_raw(image_rgb)

        noise_sigma = float(ctx.metrics.get("noise_sigma", 0.0))
        sharp_p95 = float(ctx.metrics.get("sharp_p95", 0.0))
        faces = score_and_filter_faces(raw_dets, ctx.analysis_img, noise_sigma=noise_sigma)
        face_oof, face_sharp_min = evaluate_subject_faces_oof(faces, sharp_p95=sharp_p95)
        ctx.faces = faces
        return {
            "faces": faces,
            "face_count": len(faces),
            "face_sharp_min": face_sharp_min,
            "face_out_of_focus": face_oof,
            "model_available": True,
        }


@lru_cache(maxsize=4)
def get_cached_face_detector(model_path_str: str) -> FaceDetector:
    """Return a process-local cached `FaceDetector` so `det_10g.onnx` loads once per worker."""
    backend = ScrfdOnnxBackend(model_path=Path(model_path_str))
    return FaceDetector(backend=backend)


__all__ = [
    "ANTELOPEV2_DEFAULT_URL",
    "ANTELOPEV2_DIRNAME",
    "DEFAULT_DET_INPUT_SIZE",
    "DET_MODEL_FILENAME",
    "FACE_DET_SCORE_THRESHOLD",
    "FACE_NMS_IOU_THRESHOLD",
    "FACE_OOF_ABS_THRESHOLD",
    "FACE_OOF_RELATIVE_RATIO",
    "FACE_STORE_MIN_AREA_RATIO",
    "FACE_SUBJECT_MIN_AREA_RATIO",
    "FaceBackend",
    "FaceDetector",
    "RawFaceDetection",
    "ScrfdOnnxBackend",
    "decode_scrfd_outputs",
    "ensure_face_model",
    "evaluate_subject_faces_oof",
    "find_face_model_path",
    "get_cached_face_detector",
    "score_and_filter_faces",
]
