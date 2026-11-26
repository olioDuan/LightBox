"""PeopleService: post-cull face embedding, clustering, naming, strict grouping, and SafeFS copy export (SPEC §13(3), PLAN T18)."""

from __future__ import annotations

import hashlib
import io
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from lightbox.core import (
    ENGINE_VERSION,
    FACE_DETECTOR_PIPELINE_VERSION,
    FACE_STORE_MIN_AREA_RATIO,
    FaceDetector,
    PhotoContext,
    compute_content_fp,
    find_face_model_path,
    find_landmark106_model_path,
    load_analysis_gray,
    preview_cache_paths,
)
from lightbox.fs import SafeFS
from lightbox.models import (
    FACE_EMBEDDING_DIM,
    FaceEmbedder,
    InsightFaceEmbedder,
    align_face_arcface,
)
from lightbox.paths import AppPaths, get_app_paths
from lightbox.store import (
    FaceAssignmentRecord,
    FaceEmbeddingRecord,
    FaceRecord,
    LightboxRepository,
    MetricsRecord,
    PersonRecord,
    PhotoRecord,
    SessionRecord,
)

DEFAULT_FACE_COSINE_THRESHOLD = 0.42
SIMILAR_CANDIDATE_MIN_COSINE = 0.22
SIMILAR_CANDIDATE_LIMIT = 4
GROUP_KEY_GROUP_PHOTO = "group_photo"
GROUP_KEY_UNNAMED = "unnamed"
GROUP_KEY_NO_FACE = "no_face"


class PeopleError(ValueError):
    """Domain validation error for people classification / labeling / export."""


@dataclass(frozen=True, slots=True)
class FaceSlotInfo:
    """Per-face assignment state attached to a photo in `/api/people`."""

    content_fp: str
    face_idx: int
    bbox: tuple[float, float, float, float]
    area_ratio: float
    det_score: float | None
    status: str  # 'assigned' | 'unassigned' | 'ignored'
    source: str  # 'user' | 'auto'
    person_id: str | None
    person_name: str | None
    cluster_id: str | None
    confidence: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_fp": self.content_fp,
            "face_idx": self.face_idx,
            "bbox": list(self.bbox),
            "area_ratio": self.area_ratio,
            "det_score": self.det_score,
            "status": self.status,
            "source": self.source,
            "person_id": self.person_id,
            "person_name": self.person_name,
            "cluster_id": self.cluster_id,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
        }


@dataclass(frozen=True, slots=True)
class PeoplePhotoItem:
    """Single kept photo placed in a people classification group (never truncated)."""

    photo_id: str
    session_id: str
    session_name: str
    stem: str
    rel_dir: str
    content_fp: str
    capture_time: float | None
    valid_face_count: int
    faces: tuple[FaceSlotInfo, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "photo_id": self.photo_id,
            "session_id": self.session_id,
            "session_name": self.session_name,
            "stem": self.stem,
            "rel_dir": self.rel_dir,
            "content_fp": self.content_fp,
            "capture_time": self.capture_time,
            "valid_face_count": self.valid_face_count,
            "faces": [f.to_dict() for f in self.faces],
        }


@dataclass(frozen=True, slots=True)
class SimilarFaceCandidate:
    """Potential same-person match surfaced for 1-click visual merge."""

    kind: str  # 'person' | 'cluster'
    target_id: str
    name: str | None
    cover_photo_id: str | None
    cover_face_idx: int | None
    photo_count: int
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target_id": self.target_id,
            "name": self.name,
            "cover_photo_id": self.cover_photo_id,
            "cover_face_idx": self.cover_face_idx,
            "photo_count": self.photo_count,
            "similarity": round(self.similarity, 4),
        }


@dataclass(frozen=True, slots=True)
class PersonGroupInfo:
    """Named single-person group containing 100% of that person's solo kept photos."""

    person_id: str
    group_key: str
    name: str
    is_self: bool
    cover_photo_id: str | None
    cover_face_idx: int | None
    photo_count: int
    group_photo_count: int
    photos: tuple[PeoplePhotoItem, ...]
    similar_candidates: tuple[SimilarFaceCandidate, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "group_key": self.group_key,
            "name": self.name,
            "is_self": self.is_self,
            "cover_photo_id": self.cover_photo_id,
            "cover_face_idx": self.cover_face_idx,
            "photo_count": self.photo_count,
            "group_photo_count": self.group_photo_count,
            "photos": [p.to_dict() for p in self.photos],
            "similar_candidates": [c.to_dict() for c in self.similar_candidates],
        }


@dataclass(frozen=True, slots=True)
class UnnamedClusterInfo:
    """Unnamed face cluster surfaced for one-click naming, merging, or ignoring."""

    cluster_id: str
    cover_photo_id: str
    cover_content_fp: str
    cover_face_idx: int
    face_count: int
    solo_photo_count: int
    faces: tuple[tuple[str, int], ...]
    photos: tuple[PeoplePhotoItem, ...]
    similar_candidates: tuple[SimilarFaceCandidate, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "cover_photo_id": self.cover_photo_id,
            "cover_content_fp": self.cover_content_fp,
            "cover_face_idx": self.cover_face_idx,
            "face_count": self.face_count,
            "solo_photo_count": self.solo_photo_count,
            "faces": [{"content_fp": fp, "face_idx": fi} for fp, fi in self.faces],
            "photos": [p.to_dict() for p in self.photos],
            "similar_candidates": [c.to_dict() for c in self.similar_candidates],
        }


@dataclass(frozen=True, slots=True)
class PeopleGroupsResponse:
    """Complete, untruncated people classification snapshot across kept photos."""

    session_id: str | None
    group_person_id: str | None
    total_kept_photos: int
    indexed_face_count: int
    people: tuple[PersonGroupInfo, ...]
    group_photos: tuple[PeoplePhotoItem, ...]
    group_photo_total_count: int
    unnamed_photos: tuple[PeoplePhotoItem, ...]
    unnamed_clusters: tuple[UnnamedClusterInfo, ...]
    no_face_photos: tuple[PeoplePhotoItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "group_person_id": self.group_person_id,
            "total_kept_photos": self.total_kept_photos,
            "indexed_face_count": self.indexed_face_count,
            "people": [p.to_dict() for p in self.people],
            "group_photos": [p.to_dict() for p in self.group_photos],
            "group_photo_total_count": self.group_photo_total_count,
            "unnamed_photos": [p.to_dict() for p in self.unnamed_photos],
            "unnamed_clusters": [c.to_dict() for c in self.unnamed_clusters],
            "no_face_photos": [p.to_dict() for p in self.no_face_photos],
        }


@dataclass(frozen=True, slots=True)
class PeopleIndexResult:
    """Summary of face embedding extraction + clustering for a session."""

    session_id: str
    kept_photo_count: int
    valid_face_count: int
    newly_embedded_faces: int
    skipped_cached_faces: int
    named_people_count: int
    unnamed_cluster_count: int
    model_id: str
    threshold: float
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "kept_photo_count": self.kept_photo_count,
            "valid_face_count": self.valid_face_count,
            "newly_embedded_faces": self.newly_embedded_faces,
            "skipped_cached_faces": self.skipped_cached_faces,
            "named_people_count": self.named_people_count,
            "unnamed_cluster_count": self.unnamed_cluster_count,
            "model_id": self.model_id,
            "threshold": self.threshold,
            "elapsed_s": self.elapsed_s,
        }


@dataclass(frozen=True, slots=True)
class ExportGroupReport:
    group_key: str
    folder_name: str
    photo_count: int
    file_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_key": self.group_key,
            "folder_name": self.folder_name,
            "photo_count": self.photo_count,
            "file_count": self.file_count,
        }


@dataclass(frozen=True, slots=True)
class PeopleExportResult:
    target_dir: str
    exported_photos: int
    exported_files: int
    groups: tuple[ExportGroupReport, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_dir": self.target_dir,
            "exported_photos": self.exported_photos,
            "exported_files": self.exported_files,
            "groups": [g.to_dict() for g in self.groups],
        }


def _pack_embedding(vec: NDArray[np.float32]) -> bytes:
    arr = np.asarray(vec, dtype="<f4").reshape(-1)
    return arr.tobytes()


def _unpack_embedding(blob: bytes, dim: int = FACE_EMBEDDING_DIM) -> NDArray[np.float32]:
    arr = np.frombuffer(blob, dtype="<f4").astype(np.float32, copy=True)
    if arr.shape != (dim,):
        raise ValueError(f"Expected {dim}-dim face embedding, got shape {arr.shape}.")
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        arr = (arr / norm).astype(np.float32, copy=False)
    return arr


def _normalize_vec(vec: NDArray[np.float32]) -> NDArray[np.float32]:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    return (arr / norm).astype(np.float32, copy=False) if norm > 0.0 else arr


def _build_prototype_bank(
    vecs: Sequence[NDArray[np.float32]],
    *,
    max_samples: int = 24,
) -> list[NDArray[np.float32]]:
    """Build a multi-angle prototype bank (confirmed samples + normalized mean vector).

    Keeping individual user-confirmed samples alongside their mean lets the recognizer
    remember multiple angles/lighting conditions (e.g. front + side profile) after a merge,
    rather than collapsing them into a single averaged vector that misses both extremes.
    """
    if not vecs:
        return []
    if len(vecs) == 1:
        return [vecs[0]]
    if len(vecs) <= max_samples:
        selected = list(vecs)
    else:
        idxs = np.linspace(0, len(vecs) - 1, num=max_samples, dtype=int)
        selected = [vecs[int(i)] for i in idxs]
    mean_arr = np.asarray(np.mean(np.stack(vecs, axis=0), axis=0), dtype=np.float32)
    return [*selected, _normalize_vec(mean_arr)]


def _stable_id(prefix: str, key: str) -> str:
    return f"{prefix}_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}"


def _sanitize_folder_name(name: str) -> str:
    """Sanitize a person or group folder name so it can never traverse outside `target_dir`."""
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", name).strip().strip(".")
    if not cleaned or cleaned in {".", ".."}:
        return "Unnamed"
    return cleaned[:80]


def _special_folder_name(group_key: str, *, lang: str = "zh-CN") -> str:
    is_zh = lang.startswith("zh")
    if group_key == GROUP_KEY_GROUP_PHOTO:
        return "多人合影" if is_zh else "Group Photos"
    if group_key == GROUP_KEY_UNNAMED:
        return "未命名" if is_zh else "Unnamed"
    if group_key == GROUP_KEY_NO_FACE:
        return "无人脸" if is_zh else "No Face"
    return _sanitize_folder_name(group_key)


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def _current_face_detector_tag(app_paths: AppPaths) -> str | None:
    det_path = find_face_model_path(app_paths)
    if det_path is None:
        return None
    det_fp = compute_content_fp(det_path)[:12]
    lm_path = find_landmark106_model_path(app_paths)
    lm_fp = compute_content_fp(lm_path)[:12] if lm_path is not None else "none"
    return f"antelopev2:{det_path.name}:{det_fp}:{FACE_DETECTOR_PIPELINE_VERSION}:lm:{lm_fp}"


def _load_rgb_image_for_photo(
    repo: LightboxRepository,
    app_paths: AppPaths,
    photo: PhotoRecord,
) -> NDArray[np.uint8] | None:
    """Load an EXIF-oriented RGB uint8 image for face alignment (loupe -> grid -> primary asset)."""
    grid_p, loupe_p, _analysis_p = preview_cache_paths(
        app_paths.cache_dir, photo.primary_content_fp
    )
    candidates: list[Path] = []
    if loupe_p.is_file():
        candidates.append(loupe_p)
    if grid_p.is_file():
        candidates.append(grid_p)
    for kind in ("jpg", "heic", "raw"):
        for a in repo.list_assets_for_photo(photo.id):
            if a.kind == kind:
                p = Path(a.abs_path)
                if p.is_file():
                    candidates.append(p)
    for path in candidates:
        try:
            with Image.open(path) as img:
                oriented = ImageOps.exif_transpose(img)
                return np.asarray(oriented.convert("RGB"), dtype=np.uint8)
        except Exception:
            continue
    return None


class PeopleService:
    """Post-cull people classification: face embeddings, clustering, naming, grouping, and copy export."""

    def __init__(
        self,
        repo: LightboxRepository,
        face_embedder: FaceEmbedder,
        *,
        safe_fs: SafeFS | None = None,
        app_paths: AppPaths | None = None,
        default_threshold: float = DEFAULT_FACE_COSINE_THRESHOLD,
    ) -> None:
        self.repo = repo
        self.face_embedder = face_embedder
        self.safe_fs = safe_fs or SafeFS()
        self.app_paths = app_paths or get_app_paths()
        self.default_threshold = default_threshold

    def _list_kept_photos(
        self,
        session_id: str | None = None,
    ) -> list[tuple[PhotoRecord, SessionRecord]]:
        """Return all kept photos (excluding `bucket == 'red'` rejects and quarantined items)."""
        if session_id is not None:
            sess = self.repo.get_session(session_id)
            if sess is None:
                raise PeopleError(f"Unknown session_id: {session_id}")
            sessions = [sess]
        else:
            sessions = self.repo.list_sessions()

        kept: list[tuple[PhotoRecord, SessionRecord]] = []
        for sess in sessions:
            quarantined = self.repo.list_quarantined_photo_ids(sess.id)
            verdicts = self.repo.list_effective_verdicts(sess.id)
            for photo in self.repo.list_photos(sess.id):
                if photo.id in quarantined:
                    continue
                v = verdicts.get(photo.id)
                if v is not None and v.bucket == "red":
                    continue
                kept.append((photo, sess))
        kept.sort(key=lambda pair: (pair[0].rel_dir, pair[0].stem, pair[0].id))
        return kept

    def _maybe_detect_missing_faces(
        self,
        kept_photos: Sequence[tuple[PhotoRecord, SessionRecord]],
    ) -> None:
        """Populate or upgrade face detections when model availability or detector version changed.

        When a photo previously had faces (e.g. a group photo where smaller faces were missed under
        older thresholds), existing `face_idx` slots are preserved via bbox IoU so user labels stay
        intact while newly discovered faces are appended cleanly.
        """
        if not isinstance(self.face_embedder, InsightFaceEmbedder):
            return
        current_tag = _current_face_detector_tag(self.app_paths)
        if current_tag is None:
            return
        detector = FaceDetector(app_paths=self.app_paths)
        if not detector.model_available:
            return

        seen_fps: set[str] = set()
        for photo, _sess in kept_photos:
            fp = photo.primary_content_fp
            if fp in seen_fps:
                continue
            seen_fps.add(fp)
            metrics = self.repo.get_metrics(fp, ENGINE_VERSION)
            face_tag = (metrics.exposure or {}).get("face_model_tag") if metrics else None
            if face_tag == current_tag:
                continue

            _g, loupe_p, analysis_p = preview_cache_paths(self.app_paths.cache_dir, fp)
            if not analysis_p.is_file():
                continue
            gray = load_analysis_gray(analysis_p)
            loupe_rgb = _load_rgb_image_for_photo(self.repo, self.app_paths, photo)
            ctx = PhotoContext(
                photo_id=photo.id,
                content_fp=fp,
                analysis_img=gray,
                loupe_rgb=loupe_rgb,
                analysis_path=analysis_p,
                loupe_path=loupe_p if loupe_p.is_file() else None,
                metrics={
                    "sharp_p95": float(metrics.sharp_p95 or 0.0) if metrics else 0.0,
                    "noise_sigma": float(metrics.noise_sigma or 0.0) if metrics else 0.0,
                },
            )
            face_out = detector.run(ctx)
            existing_faces = self.repo.list_faces(fp)

            final_records: list[FaceRecord]
            if not existing_faces:
                final_records = [
                    FaceRecord(
                        content_fp=fp,
                        idx=f.idx,
                        bbox=f.bbox,
                        landmarks=f.landmarks,
                        det_score=f.det_score,
                        sharpness=f.sharpness,
                        area_ratio=f.area_ratio,
                    )
                    for f in ctx.faces
                ]
                if final_records:
                    self.repo.replace_faces(fp, final_records)
            else:
                # Preserve existing face indices and append any newly discovered faces (e.g. group photos)
                final_records = list(existing_faces)
                next_idx = max(f.idx for f in existing_faces) + 1
                added = False
                for new_f in ctx.faces:
                    if any(_bbox_iou(new_f.bbox, old_f.bbox) >= 0.35 for old_f in existing_faces):
                        continue
                    final_records.append(
                        FaceRecord(
                            content_fp=fp,
                            idx=next_idx,
                            bbox=new_f.bbox,
                            landmarks=new_f.landmarks,
                            det_score=new_f.det_score,
                            sharpness=new_f.sharpness,
                            area_ratio=new_f.area_ratio,
                        )
                    )
                    next_idx += 1
                    added = True
                if added:
                    self.repo.replace_faces(fp, final_records)

            if metrics is not None:
                exp = dict(metrics.exposure or {})
                exp["face_model_tag"] = current_tag
                exp["eyes_closed"] = bool(face_out.get("eyes_closed", False))
                exp["eye_ear_min"] = face_out.get("eye_ear_min")
                self.repo.upsert_metrics(
                    MetricsRecord(
                        content_fp=metrics.content_fp,
                        engine_version=metrics.engine_version,
                        sharp_p95=metrics.sharp_p95,
                        sharp_z=metrics.sharp_z,
                        noise_sigma=metrics.noise_sigma,
                        motion_aniso=metrics.motion_aniso,
                        motion_state=metrics.motion_state,
                        exposure=exp,
                        texture=metrics.texture,
                        face_count=len(final_records),
                        face_sharp_min=metrics.face_sharp_min,
                        scene_tag=metrics.scene_tag,
                    )
                )

    def index_session(
        self,
        session_id: str,
        *,
        threshold: float | None = None,
        batch_size: int = 16,
    ) -> PeopleIndexResult:
        """Extract 512-d ArcFace embeddings for kept photos in `session_id` and run matching/clustering."""
        started = time.monotonic()
        eff_threshold = self.default_threshold if threshold is None else float(threshold)
        kept = self._list_kept_photos(session_id)

        # Ensure model is available before attempting lazy face detection (in case antelopev2.zip provides both)
        if isinstance(self.face_embedder, InsightFaceEmbedder):
            self.face_embedder._ensure_session()
            self._maybe_detect_missing_faces(kept)

        kept_fps = sorted({p.primary_content_fp for p, _ in kept})
        existing_embs = {
            (rec.content_fp, rec.face_idx): rec
            for rec in self.repo.list_face_embeddings(
                self.face_embedder.model_id,
                content_fps=kept_fps,
            )
        }

        pending: list[tuple[str, int, NDArray[np.uint8]]] = []
        valid_face_count = 0
        skipped_cached = 0
        seen_fps: set[str] = set()

        for photo, _sess in kept:
            fp = photo.primary_content_fp
            if fp in seen_fps:
                continue
            seen_fps.add(fp)
            faces = [
                f
                for f in self.repo.list_faces(fp)
                if float(f.area_ratio or 0.0) >= FACE_STORE_MIN_AREA_RATIO
            ]
            if not faces:
                continue
            valid_face_count += len(faces)

            missing_faces = [f for f in faces if (fp, f.idx) not in existing_embs]
            skipped_cached += len(faces) - len(missing_faces)
            if not missing_faces:
                continue

            rgb = _load_rgb_image_for_photo(self.repo, self.app_paths, photo)
            if rgb is None:
                continue
            for face in missing_faces:
                aligned = align_face_arcface(rgb, face.bbox, face.landmarks)
                pending.append((fp, face.idx, aligned))

        newly_embedded = 0
        now = time.time()
        step = max(1, batch_size)
        for start in range(0, len(pending), step):
            chunk = pending[start : start + step]
            crops = [item[2] for item in chunk]
            vectors = self.face_embedder.embed_aligned_faces(crops)
            records = [
                FaceEmbeddingRecord(
                    content_fp=chunk[i][0],
                    face_idx=chunk[i][1],
                    model_id=self.face_embedder.model_id,
                    dim=self.face_embedder.dim,
                    embedding_blob=_pack_embedding(vectors[i]),
                    created_at=now,
                )
                for i in range(len(chunk))
            ]
            self.repo.upsert_face_embeddings(records)
            newly_embedded += len(records)

        unnamed_clusters = self.recluster_and_match(threshold=eff_threshold)
        named_people = self.repo.list_people()

        return PeopleIndexResult(
            session_id=session_id,
            kept_photo_count=len(kept),
            valid_face_count=valid_face_count,
            newly_embedded_faces=newly_embedded,
            skipped_cached_faces=skipped_cached,
            named_people_count=len(named_people),
            unnamed_cluster_count=unnamed_clusters,
            model_id=self.face_embedder.model_id,
            threshold=eff_threshold,
            elapsed_s=round(time.monotonic() - started, 3),
        )

    def recluster_and_match(self, *, threshold: float | None = None) -> int:
        """Match non-user faces against named person prototypes (cross-session) and greedily cluster the rest."""
        eff_threshold = self.default_threshold if threshold is None else float(threshold)
        kept = self._list_kept_photos(session_id=None)
        kept_fps = sorted({p.primary_content_fp for p, _ in kept})
        if not kept_fps:
            return 0

        # Load all embeddings for kept photos (plus any user-labeled prototype faces across all sessions)
        all_embs_list = self.repo.list_face_embeddings(self.face_embedder.model_id)
        emb_by_key: dict[tuple[str, int], NDArray[np.float32]] = {
            (e.content_fp, e.face_idx): _unpack_embedding(e.embedding_blob, self.face_embedder.dim)
            for e in all_embs_list
        }
        all_assignments = {
            (a.content_fp, a.face_idx): a for a in self.repo.list_face_assignments()
        }

        # 1. Build multi-angle prototype bank per named person (authoritatively from `source='user'` samples)
        people = self.repo.list_people()
        proto_ids: list[str] = []
        proto_vecs: list[NDArray[np.float32]] = []
        for person in people:
            user_vecs: list[NDArray[np.float32]] = []
            fallback_vecs: list[NDArray[np.float32]] = []
            for key, asgn in all_assignments.items():
                if asgn.status != "assigned" or asgn.person_id != person.id:
                    continue
                vec = emb_by_key.get(key)
                if vec is None:
                    continue
                if asgn.source == "user":
                    user_vecs.append(vec)
                else:
                    fallback_vecs.append(vec)
            chosen = user_vecs or fallback_vecs
            for pvec in _build_prototype_bank(chosen):
                proto_ids.append(person.id)
                proto_vecs.append(pvec)

        proto_mat = (
            np.stack(proto_vecs, axis=0)
            if proto_vecs
            else np.zeros((0, self.face_embedder.dim), dtype=np.float32)
        )

        # 2. Match every non-user, non-ignored valid face of kept photos against multi-angle prototypes
        now = time.time()
        updates: list[FaceAssignmentRecord] = []
        unmatched: list[tuple[str, int, NDArray[np.float32]]] = []

        for fp in kept_fps:
            faces = self.repo.list_faces(fp)
            for face in faces:
                if float(face.area_ratio or 0.0) < FACE_STORE_MIN_AREA_RATIO:
                    continue
                key = (fp, face.idx)
                existing = all_assignments.get(key)
                if existing is not None and (
                    existing.source == "user" or existing.status == "ignored"
                ):
                    continue
                vec = emb_by_key.get(key)
                if vec is None:
                    continue

                matched_person: str | None = None
                matched_sim = -1.0
                if proto_mat.shape[0] > 0:
                    sims = proto_mat @ vec
                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])
                    if best_sim >= eff_threshold:
                        matched_person = proto_ids[best_idx]
                        matched_sim = best_sim

                if matched_person is not None:
                    updates.append(
                        FaceAssignmentRecord(
                            content_fp=fp,
                            face_idx=face.idx,
                            person_id=matched_person,
                            status="assigned",
                            source="auto",
                            confidence=matched_sim,
                            cluster_id=None,
                            updated_at=now,
                        )
                    )
                else:
                    unmatched.append((fp, face.idx, vec))

        # 3. Greedy cosine threshold clustering in pure numpy over unmatched faces.
        # Pre-seed with any user-merged unnamed clusters (`source='user', status='unassigned'`)
        # using multi-angle sample banks so manually merged clusters stay intact and immediately
        # recall other unmatched faces from any of the merged angles.
        pinned_cluster_vecs: dict[str, list[NDArray[np.float32]]] = {}
        kept_fp_set = set(kept_fps)
        for key, asgn in all_assignments.items():
            if (
                key[0] in kept_fp_set
                and asgn.source == "user"
                and asgn.status == "unassigned"
                and asgn.cluster_id is not None
            ):
                vec = emb_by_key.get(key)
                if vec is not None:
                    pinned_cluster_vecs.setdefault(asgn.cluster_id, []).append(vec)

        clusters_sum: list[NDArray[np.float32]] = []
        clusters_id: list[str] = []
        clusters_pinned_mat: list[NDArray[np.float32] | None] = []
        for pinned_cid in sorted(pinned_cluster_vecs):
            vecs = pinned_cluster_vecs[pinned_cid]
            clusters_id.append(pinned_cid)
            clusters_sum.append(np.sum(np.stack(vecs, axis=0), axis=0).astype(np.float32))
            bank = _build_prototype_bank(vecs)
            clusters_pinned_mat.append(np.stack(bank, axis=0) if bank else None)

        for fp, face_idx, vec in unmatched:
            assigned_cluster: str | None = None
            assigned_sim = 1.0
            if clusters_sum:
                centroids = np.stack([_normalize_vec(s) for s in clusters_sum], axis=0)
                sims = centroids @ vec
                best_i = -1
                best_sim = -1.0
                for idx in range(len(clusters_sum)):
                    cand_sim = float(sims[idx])
                    pmat = clusters_pinned_mat[idx]
                    if pmat is not None:
                        cand_sim = max(cand_sim, float(np.max(pmat @ vec)))
                    if cand_sim > best_sim:
                        best_sim = cand_sim
                        best_i = idx
                if best_i >= 0 and best_sim >= eff_threshold:
                    clusters_sum[best_i] = clusters_sum[best_i] + vec
                    assigned_cluster = clusters_id[best_i]
                    assigned_sim = best_sim
            if assigned_cluster is None:
                assigned_cluster = _stable_id("clu", f"{fp}:{face_idx}")
                clusters_id.append(assigned_cluster)
                clusters_sum.append(vec.copy())
                clusters_pinned_mat.append(None)
                assigned_sim = 1.0

            updates.append(
                FaceAssignmentRecord(
                    content_fp=fp,
                    face_idx=face_idx,
                    person_id=None,
                    status="unassigned",
                    source="auto",
                    confidence=assigned_sim,
                    cluster_id=assigned_cluster,
                    updated_at=now,
                )
            )

        self.repo.upsert_face_assignments(updates)
        return len(clusters_id)

    def _resolve_target_faces(
        self,
        *,
        cluster_id: str | None = None,
        faces: Sequence[tuple[str, int]] | None = None,
        person_id: str | None = None,
    ) -> list[tuple[str, int]]:
        targets: list[tuple[str, int]] = []
        if faces:
            targets.extend((str(fp), int(fi)) for fp, fi in faces)
        if cluster_id:
            for a in self.repo.list_face_assignments():
                if a.cluster_id == cluster_id and a.status == "unassigned":
                    targets.append((a.content_fp, a.face_idx))
        if person_id:
            for a in self.repo.list_face_assignments():
                if a.person_id == person_id and a.status == "assigned":
                    targets.append((a.content_fp, a.face_idx))
        # Deduplicate while preserving deterministic order
        return list(dict.fromkeys(targets))

    def label_name(
        self,
        name: str,
        *,
        cluster_id: str | None = None,
        faces: Sequence[tuple[str, int]] | None = None,
        person_id: str | None = None,
        is_self: bool = False,
        threshold: float | None = None,
    ) -> PersonRecord:
        """Name a cluster or sample faces (or rename an existing person) and propagate across sessions."""
        clean_name = name.strip()
        if not clean_name:
            raise PeopleError("Person name must not be empty.")

        # Case A: Rename an existing person without merging/targeting faces
        if person_id is not None and cluster_id is None and not faces:
            existing_person = self.repo.get_person(person_id)
            if existing_person is None:
                raise PeopleError(f"Unknown person_id: {person_id}")
            collision = self.repo.get_person_by_name(clean_name)
            if collision is not None and collision.id != person_id:
                raise PeopleError(
                    f"Person name {clean_name!r} already exists (use merge to combine them)."
                )
            updated = PersonRecord(
                id=existing_person.id,
                name=clean_name,
                is_self=is_self or existing_person.is_self,
                cover_content_fp=existing_person.cover_content_fp,
                cover_face_idx=existing_person.cover_face_idx,
                created_at=existing_person.created_at,
            )
            self.repo.upsert_person(updated)
            return updated

        targets = self._resolve_target_faces(cluster_id=cluster_id, faces=faces, person_id=person_id)
        if not targets:
            raise PeopleError("No target faces or cluster members found to name.")

        person = self.repo.get_person_by_name(clean_name)
        now = time.time()
        if person is None:
            new_id = _stable_id("per", f"{clean_name}:{now}")
            cover_fp, cover_idx = targets[0]
            person = PersonRecord(
                id=new_id,
                name=clean_name,
                is_self=is_self,
                cover_content_fp=cover_fp,
                cover_face_idx=cover_idx,
                created_at=now,
            )
            self.repo.upsert_person(person)
        elif is_self and not person.is_self:
            person = PersonRecord(
                id=person.id,
                name=person.name,
                is_self=True,
                cover_content_fp=person.cover_content_fp or targets[0][0],
                cover_face_idx=(
                    person.cover_face_idx if person.cover_face_idx is not None else targets[0][1]
                ),
                created_at=person.created_at,
            )
            self.repo.upsert_person(person)

        self.repo.upsert_face_assignments(
            [
                FaceAssignmentRecord(
                    content_fp=fp,
                    face_idx=fi,
                    person_id=person.id,
                    status="assigned",
                    source="user",
                    confidence=1.0,
                    cluster_id=None,
                    updated_at=now,
                )
                for fp, fi in targets
            ]
        )
        self.recluster_and_match(threshold=threshold)
        return person

    def label_merge(
        self,
        *,
        source_cluster_id: str | None = None,
        target_cluster_id: str | None = None,
        source_person_id: str | None = None,
        target_person_id: str | None = None,
        source_cluster_ids: Sequence[str] | None = None,
        source_person_ids: Sequence[str] | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Merge clusters and/or named people into a single target and re-match across angles."""
        now = time.time()
        src_cids = list(
            dict.fromkeys(
                [
                    *(source_cluster_ids or ()),
                    *([source_cluster_id] if source_cluster_id else []),
                ]
            )
        )
        src_pids = list(
            dict.fromkeys(
                [
                    *(source_person_ids or ()),
                    *([source_person_id] if source_person_id else []),
                ]
            )
        )

        # 1. Target is a named person (accepts source person(s) and/or source cluster(s))
        if target_person_id and (src_pids or src_cids):
            dst = self.repo.get_person(target_person_id)
            if dst is None:
                raise PeopleError(f"Unknown target_person_id: {target_person_id}")

            moved_total = 0
            for spid in src_pids:
                if spid == target_person_id:
                    raise PeopleError("Cannot merge a person into themselves.")
                if self.repo.get_person(spid) is None:
                    raise PeopleError(f"Unknown source_person_id: {spid}")
                moved_total += self.repo.reassign_person_faces(spid, target_person_id)
                self.repo.delete_person(spid)

            cluster_faces_to_move: list[tuple[str, int]] = []
            for scid in src_cids:
                targets = self._resolve_target_faces(cluster_id=scid)
                if not targets:
                    raise PeopleError(f"No unassigned faces in cluster {scid!r}.")
                cluster_faces_to_move.extend(targets)

            if cluster_faces_to_move:
                dedup_targets = list(dict.fromkeys(cluster_faces_to_move))
                self.repo.upsert_face_assignments(
                    [
                        FaceAssignmentRecord(
                            content_fp=fp,
                            face_idx=fi,
                            person_id=dst.id,
                            status="assigned",
                            source="user",
                            confidence=1.0,
                            cluster_id=None,
                            updated_at=now,
                        )
                        for fp, fi in dedup_targets
                    ]
                )
                moved_total += len(dedup_targets)

            self.recluster_and_match(threshold=threshold)
            merged_kind = (
                "person_into_person"
                if src_pids and not src_cids
                else "cluster_into_person"
                if src_cids and not src_pids
                else "batch_into_person"
            )
            return {
                "merged_kind": merged_kind,
                "target_person_id": target_person_id,
                "moved_faces": moved_total,
            }

        # 2. Cluster(s) -> Cluster (pin merged angles as user samples and immediately recall similar faces)
        if target_cluster_id and src_cids:
            moved_total = 0
            for scid in src_cids:
                if scid == target_cluster_id:
                    raise PeopleError("Cannot merge a cluster into itself.")
                moved = self.repo.reassign_cluster_faces(scid, target_cluster_id)
                if moved == 0:
                    raise PeopleError(f"No unassigned faces in source cluster {scid!r}.")
                moved_total += moved
            self.recluster_and_match(threshold=threshold)
            return {
                "merged_kind": "cluster_into_cluster",
                "target_cluster_id": target_cluster_id,
                "moved_faces": moved_total,
            }

        raise PeopleError(
            "Merge requires (source_person_id + target_person_id), "
            "(source_cluster_id + target_person_id), or (source_cluster_id + target_cluster_id)."
        )

    def label_split(
        self,
        *,
        faces: Sequence[tuple[str, int]],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Detach mis-grouped face(s) from their current person/cluster into a fresh user-pinned face group."""
        targets = list(dict.fromkeys((str(fp), int(fi)) for fp, fi in faces))
        if not targets:
            raise PeopleError("At least one face is required for split.")
        now = time.time()
        new_cid = _stable_id("clu", f"split:{targets[0][0]}:{targets[0][1]}:{now:.6f}")
        self.repo.upsert_face_assignments(
            [
                FaceAssignmentRecord(
                    content_fp=fp,
                    face_idx=fi,
                    person_id=None,
                    status="unassigned",
                    source="user",
                    confidence=1.0,
                    cluster_id=new_cid,
                    updated_at=now,
                )
                for fp, fi in targets
            ]
        )
        self.recluster_and_match(threshold=threshold)
        return {"cluster_id": new_cid, "moved_faces": len(targets)}

    def label_reassign(
        self,
        *,
        faces: Sequence[tuple[str, int]],
        target_person_id: str | None = None,
        target_cluster_id: str | None = None,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Reassign specific face(s) to another named person or unnamed face group and update prototypes."""
        targets = list(dict.fromkeys((str(fp), int(fi)) for fp, fi in faces))
        if not targets:
            raise PeopleError("At least one face is required for reassign.")
        if bool(target_person_id) == bool(target_cluster_id):
            raise PeopleError(
                "Reassign requires exactly one of target_person_id or target_cluster_id."
            )
        now = time.time()
        if target_person_id:
            dst = self.repo.get_person(target_person_id)
            if dst is None:
                raise PeopleError(f"Unknown target_person_id: {target_person_id}")
            self.repo.upsert_face_assignments(
                [
                    FaceAssignmentRecord(
                        content_fp=fp,
                        face_idx=fi,
                        person_id=dst.id,
                        status="assigned",
                        source="user",
                        confidence=1.0,
                        cluster_id=None,
                        updated_at=now,
                    )
                    for fp, fi in targets
                ]
            )
            self.recluster_and_match(threshold=threshold)
            return {
                "target_person_id": dst.id,
                "target_cluster_id": None,
                "moved_faces": len(targets),
            }

        assert target_cluster_id is not None
        # Also pin existing target cluster members as `source='user'` so reclustering preserves them together
        existing_members = self._resolve_target_faces(cluster_id=target_cluster_id)
        combined = list(dict.fromkeys([*existing_members, *targets]))
        self.repo.upsert_face_assignments(
            [
                FaceAssignmentRecord(
                    content_fp=fp,
                    face_idx=fi,
                    person_id=None,
                    status="unassigned",
                    source="user",
                    confidence=1.0,
                    cluster_id=target_cluster_id,
                    updated_at=now,
                )
                for fp, fi in combined
            ]
        )
        self.recluster_and_match(threshold=threshold)
        return {
            "target_person_id": None,
            "target_cluster_id": target_cluster_id,
            "moved_faces": len(targets),
        }

    def label_ignore(
        self,
        *,
        faces: Sequence[tuple[str, int]] | None = None,
        cluster_id: str | None = None,
    ) -> int:
        """Mark bystander/false-positive face(s) or an entire unnamed cluster as `ignored`."""
        targets = self._resolve_target_faces(cluster_id=cluster_id, faces=faces)
        if not targets:
            raise PeopleError("No target faces or cluster members found to ignore.")
        now = time.time()
        self.repo.upsert_face_assignments(
            [
                FaceAssignmentRecord(
                    content_fp=fp,
                    face_idx=fi,
                    person_id=None,
                    status="ignored",
                    source="user",
                    confidence=1.0,
                    cluster_id=None,
                    updated_at=now,
                )
                for fp, fi in targets
            ]
        )
        return len(targets)

    def get_people_groups(
        self,
        *,
        session_id: str | None = None,
        group_person_id: str | None = None,
    ) -> PeopleGroupsResponse:
        """Compute strict, untruncated people groups over kept photos (SPEC §13(3), PLAN T18).

        Strict per-photo grouping rule over valid faces (`area_ratio >= 0.003` and not `ignored`):
        - 1 valid face & assigned -> that person's single-person group
        - 1 valid face & unassigned -> `unnamed` group
        - >= 2 valid faces -> `group_photo` (`多人合影`), never mixed into single-person groups
        - 0 valid faces -> `no_face` (`无人脸`)
        """
        kept = self._list_kept_photos(session_id=session_id)
        kept_fps = sorted({p.primary_content_fp for p, _ in kept})

        people_records = self.repo.list_people()
        person_by_id = {p.id: p for p in people_records}
        assignments = {
            (a.content_fp, a.face_idx): a
            for a in self.repo.list_face_assignments(content_fps=kept_fps)
        }
        embs = self.repo.list_face_embeddings(self.face_embedder.model_id, content_fps=kept_fps)

        # Map content_fp -> first photo_id for cover avatar resolution
        photo_id_by_fp: dict[str, str] = {}
        for photo, _sess in kept:
            photo_id_by_fp.setdefault(photo.primary_content_fp, photo.id)

        solo_by_person: dict[str, list[PeoplePhotoItem]] = {p.id: [] for p in people_records}
        group_count_by_person: dict[str, int] = {p.id: 0 for p in people_records}
        all_group_photos: list[PeoplePhotoItem] = []
        filtered_group_photos: list[PeoplePhotoItem] = []
        unnamed_photos: list[PeoplePhotoItem] = []
        no_face_photos: list[PeoplePhotoItem] = []

        cluster_faces: dict[str, list[tuple[str, int, str]]] = {}
        cluster_photos: dict[str, list[PeoplePhotoItem]] = {}
        cluster_seen_photo_ids: dict[str, set[str]] = {}

        for photo, sess in kept:
            fp = photo.primary_content_fp
            raw_faces = [
                f
                for f in self.repo.list_faces(fp)
                if float(f.area_ratio or 0.0) >= FACE_STORE_MIN_AREA_RATIO
            ]

            slots: list[FaceSlotInfo] = []
            valid_slots: list[FaceSlotInfo] = []
            for f in raw_faces:
                asgn = assignments.get((fp, f.idx))
                status = asgn.status if asgn is not None else "unassigned"
                source = asgn.source if asgn is not None else "auto"
                pid = asgn.person_id if (asgn is not None and status == "assigned") else None
                pname = person_by_id[pid].name if (pid is not None and pid in person_by_id) else None
                cid = asgn.cluster_id if (asgn is not None and status == "unassigned") else None
                if status == "unassigned" and cid is None:
                    cid = _stable_id("clu", f"{fp}:{f.idx}")
                slot = FaceSlotInfo(
                    content_fp=fp,
                    face_idx=f.idx,
                    bbox=f.bbox,
                    area_ratio=float(f.area_ratio or 0.0),
                    det_score=f.det_score,
                    status=status,
                    source=source,
                    person_id=pid,
                    person_name=pname,
                    cluster_id=cid,
                    confidence=asgn.confidence if asgn is not None else None,
                )
                slots.append(slot)
                if status != "ignored":
                    valid_slots.append(slot)

            item = PeoplePhotoItem(
                photo_id=photo.id,
                session_id=photo.session_id,
                session_name=sess.name,
                stem=photo.stem,
                rel_dir=photo.rel_dir,
                content_fp=fp,
                capture_time=photo.capture_time,
                valid_face_count=len(valid_slots),
                faces=tuple(slots),
            )

            # Track unnamed clusters across all kept photos so users can name/merge/ignore any cluster
            for slot in valid_slots:
                if slot.status == "unassigned" and slot.cluster_id is not None:
                    cid = slot.cluster_id
                    cluster_faces.setdefault(cid, []).append((fp, slot.face_idx, photo.id))
                    seen_set = cluster_seen_photo_ids.setdefault(cid, set())
                    if photo.id not in seen_set:
                        seen_set.add(photo.id)
                        cluster_photos.setdefault(cid, []).append(item)

            # Strict mutually-exclusive photo grouping (SPEC §13(3))
            if len(valid_slots) == 0:
                no_face_photos.append(item)
            elif len(valid_slots) >= 2:
                all_group_photos.append(item)
                present_pids = {s.person_id for s in valid_slots if s.person_id is not None}
                for pid in present_pids:
                    if pid in group_count_by_person:
                        group_count_by_person[pid] += 1
                if group_person_id is None or group_person_id in present_pids:
                    filtered_group_photos.append(item)
            else:
                sole = valid_slots[0]
                if sole.person_id is not None and sole.person_id in solo_by_person:
                    solo_by_person[sole.person_id].append(item)
                else:
                    unnamed_photos.append(item)

        people_groups: list[PersonGroupInfo] = []
        for person in people_records:
            p_photos = solo_by_person.get(person.id, [])
            cover_pid: str | None = None
            cover_fidx: int | None = None
            if person.cover_content_fp and person.cover_content_fp in photo_id_by_fp:
                cover_pid = photo_id_by_fp[person.cover_content_fp]
                cover_fidx = person.cover_face_idx
            elif p_photos:
                cover_pid = p_photos[0].photo_id
                cover_fidx = next(
                    (f.face_idx for f in p_photos[0].faces if f.person_id == person.id),
                    0,
                )
            else:
                # Fallback cover from group photos containing this person
                for gp in all_group_photos:
                    match_face = next((f for f in gp.faces if f.person_id == person.id), None)
                    if match_face is not None:
                        cover_pid = gp.photo_id
                        cover_fidx = match_face.face_idx
                        break

            people_groups.append(
                PersonGroupInfo(
                    person_id=person.id,
                    group_key=f"person:{person.id}",
                    name=person.name,
                    is_self=person.is_self,
                    cover_photo_id=cover_pid,
                    cover_face_idx=cover_fidx,
                    photo_count=len(p_photos),
                    group_photo_count=group_count_by_person.get(person.id, 0),
                    photos=tuple(p_photos),
                )
            )

        unnamed_cluster_infos: list[UnnamedClusterInfo] = []
        for cid, face_entries in cluster_faces.items():
            dedup_faces = list(dict.fromkeys((fp, fi) for fp, fi, _ in face_entries))
            cover_fp, cover_fidx, cover_pid = face_entries[0]
            c_photos = cluster_photos.get(cid, [])
            solo_count = sum(1 for p in c_photos if p.valid_face_count == 1)
            unnamed_cluster_infos.append(
                UnnamedClusterInfo(
                    cluster_id=cid,
                    cover_photo_id=cover_pid,
                    cover_content_fp=cover_fp,
                    cover_face_idx=cover_fidx,
                    face_count=len(dedup_faces),
                    solo_photo_count=solo_count,
                    faces=tuple(dedup_faces),
                    photos=tuple(c_photos),
                )
            )
        unnamed_cluster_infos.sort(key=lambda c: (-c.face_count, c.cluster_id))

        # Compute "possibly same person" suggestions across named people & unnamed clusters
        emb_by_key: dict[tuple[str, int], NDArray[np.float32]] = {}
        for rec in embs:
            try:
                emb_by_key[(rec.content_fp, rec.face_idx)] = _unpack_embedding(
                    rec.embedding_blob, self.face_embedder.dim
                )
            except ValueError:
                continue

        # Entity metadata + multi-sample prototype matrix for pairwise similarity
        entity_entries: list[
            tuple[str, str, str | None, str | None, int | None, int, NDArray[np.float32]]
        ] = []

        for p_grp in people_groups:
            user_vecs: list[NDArray[np.float32]] = []
            fallback_vecs: list[NDArray[np.float32]] = []
            for key, asgn in assignments.items():
                if asgn.status != "assigned" or asgn.person_id != p_grp.person_id:
                    continue
                vec = emb_by_key.get(key)
                if vec is None:
                    continue
                if asgn.source == "user":
                    user_vecs.append(vec)
                else:
                    fallback_vecs.append(vec)
            bank = _build_prototype_bank(user_vecs or fallback_vecs, max_samples=12)
            if bank:
                entity_entries.append(
                    (
                        "person",
                        p_grp.person_id,
                        p_grp.name,
                        p_grp.cover_photo_id,
                        p_grp.cover_face_idx,
                        p_grp.photo_count,
                        np.stack(bank, axis=0),
                    )
                )

        for clu in unnamed_cluster_infos:
            c_vecs = [emb_by_key[key] for key in clu.faces if key in emb_by_key]
            bank = _build_prototype_bank(c_vecs, max_samples=12)
            if bank:
                entity_entries.append(
                    (
                        "cluster",
                        clu.cluster_id,
                        None,
                        clu.cover_photo_id,
                        clu.cover_face_idx,
                        len(clu.photos),
                        np.stack(bank, axis=0),
                    )
                )

        sim_map: dict[tuple[str, str], list[SimilarFaceCandidate]] = {}
        for i, (kind_a, id_a, _na, _cpa, _cfa, _cnta, mat_a) in enumerate(entity_entries):
            cands: list[SimilarFaceCandidate] = []
            for j, (kind_b, id_b, name_b, cpb, cfb, cnt_b, mat_b) in enumerate(entity_entries):
                if i == j:
                    continue
                best_sim = float(np.max(mat_a @ mat_b.T))
                if best_sim >= SIMILAR_CANDIDATE_MIN_COSINE:
                    cands.append(
                        SimilarFaceCandidate(
                            kind=kind_b,
                            target_id=id_b,
                            name=name_b,
                            cover_photo_id=cpb,
                            cover_face_idx=cfb,
                            photo_count=cnt_b,
                            similarity=best_sim,
                        )
                    )
            cands.sort(key=lambda c: (-c.similarity, c.target_id))
            sim_map[(kind_a, id_a)] = cands[:SIMILAR_CANDIDATE_LIMIT]

        people_groups = [
            PersonGroupInfo(
                person_id=p.person_id,
                group_key=p.group_key,
                name=p.name,
                is_self=p.is_self,
                cover_photo_id=p.cover_photo_id,
                cover_face_idx=p.cover_face_idx,
                photo_count=p.photo_count,
                group_photo_count=p.group_photo_count,
                photos=p.photos,
                similar_candidates=tuple(sim_map.get(("person", p.person_id), ())),
            )
            for p in people_groups
        ]

        unnamed_cluster_infos = [
            UnnamedClusterInfo(
                cluster_id=c.cluster_id,
                cover_photo_id=c.cover_photo_id,
                cover_content_fp=c.cover_content_fp,
                cover_face_idx=c.cover_face_idx,
                face_count=c.face_count,
                solo_photo_count=c.solo_photo_count,
                faces=c.faces,
                photos=c.photos,
                similar_candidates=tuple(sim_map.get(("cluster", c.cluster_id), ())),
            )
            for c in unnamed_cluster_infos
        ]

        return PeopleGroupsResponse(
            session_id=session_id,
            group_person_id=group_person_id,
            total_kept_photos=len(kept),
            indexed_face_count=len(embs),
            people=tuple(people_groups),
            group_photos=tuple(filtered_group_photos),
            group_photo_total_count=len(all_group_photos),
            unnamed_photos=tuple(unnamed_photos),
            unnamed_clusters=tuple(unnamed_cluster_infos),
            no_face_photos=tuple(no_face_photos),
        )

    def get_face_crop_bytes(
        self,
        photo_id: str,
        face_idx: int,
        *,
        size: int = 160,
    ) -> bytes:
        """Render a clean EXIF-stripped WebP square avatar crop for a detected face."""
        photo = self.repo.get_photo(photo_id)
        if photo is None:
            raise PeopleError(f"Unknown photo_id: {photo_id}")
        faces = self.repo.list_faces(photo.primary_content_fp)
        face = next((f for f in faces if f.idx == face_idx), None)
        if face is None:
            raise PeopleError(f"Unknown face_idx {face_idx} for photo {photo_id}")

        rgb = _load_rgb_image_for_photo(self.repo, self.app_paths, photo)
        if rgb is None:
            raise FileNotFoundError(f"No readable image file for photo {photo_id}")

        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = face.bbox
        cx = 0.5 * (x1 + x2) * w
        cy = 0.5 * (y1 + y2) * h
        side = max((x2 - x1) * w, (y2 - y1) * h) * 1.45
        half = max(16.0, side * 0.5)
        px1 = max(0, int(round(cx - half)))
        py1 = max(0, int(round(cy - half)))
        px2 = min(w, max(px1 + 1, int(round(cx + half))))
        py2 = min(h, max(py1 + 1, int(round(cy + half))))

        crop = Image.fromarray(rgb[py1:py2, px1:px2], mode="RGB")
        crop = crop.resize((size, size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        crop.save(buf, format="WEBP", quality=85)
        return buf.getvalue()

    def export_groups(
        self,
        target_dir: Path | str,
        *,
        session_id: str | None = None,
        group_keys: Sequence[str] | None = None,
        include_unnamed: bool = False,
        include_no_face: bool = False,
        lang: str = "zh-CN",
    ) -> PeopleExportResult:
        """Export copies of grouped photos via `SafeFS.write_new_copy` (never modifies originals or `plans`)."""
        dest_root = Path(target_dir).expanduser().resolve()

        # Prevent exporting inside any scanned shoot directory so future rescans never re-ingest copies
        for sess in self.repo.list_sessions():
            sess_root = Path(sess.root_path).expanduser().resolve()
            if dest_root == sess_root or dest_root.is_relative_to(sess_root):
                raise PeopleError(
                    f"Refusing to export inside scanned photo folder {sess_root}; "
                    "choose a separate export directory outside your shoot folders."
                )

        groups_resp = self.get_people_groups(session_id=session_id)

        selected_raw: list[tuple[str, str, tuple[PeoplePhotoItem, ...]]] = []
        allowed_keys = set(group_keys) if group_keys is not None else None

        # Reserve special group folder names first so a person named e.g. `多人合影` cannot collide
        special_selected: list[tuple[str, str, tuple[PeoplePhotoItem, ...]]] = []
        if groups_resp.group_photos and (
            allowed_keys is None or GROUP_KEY_GROUP_PHOTO in allowed_keys
        ):
            special_selected.append(
                (
                    GROUP_KEY_GROUP_PHOTO,
                    _special_folder_name(GROUP_KEY_GROUP_PHOTO, lang=lang),
                    groups_resp.group_photos,
                )
            )
        if groups_resp.unnamed_photos and (
            (allowed_keys is not None and GROUP_KEY_UNNAMED in allowed_keys)
            or (allowed_keys is None and include_unnamed)
        ):
            special_selected.append(
                (
                    GROUP_KEY_UNNAMED,
                    _special_folder_name(GROUP_KEY_UNNAMED, lang=lang),
                    groups_resp.unnamed_photos,
                )
            )
        if groups_resp.no_face_photos and (
            (allowed_keys is not None and GROUP_KEY_NO_FACE in allowed_keys)
            or (allowed_keys is None and include_no_face)
        ):
            special_selected.append(
                (
                    GROUP_KEY_NO_FACE,
                    _special_folder_name(GROUP_KEY_NO_FACE, lang=lang),
                    groups_resp.no_face_photos,
                )
            )

        used_folder_names_ci: set[str] = {folder.casefold() for _, folder, _ in special_selected}

        for p_grp in groups_resp.people:
            if not p_grp.photos:
                continue
            if allowed_keys is not None and p_grp.group_key not in allowed_keys:
                continue
            base_folder = _sanitize_folder_name(p_grp.name)
            folder_name = base_folder
            suffix = 2
            while folder_name.casefold() in used_folder_names_ci:
                folder_name = f"{base_folder} ({suffix})"
                suffix += 1
            used_folder_names_ci.add(folder_name.casefold())
            selected_raw.append((p_grp.group_key, folder_name, p_grp.photos))

        selected = [*selected_raw, *special_selected]
        if not selected:
            raise PeopleError("No non-empty people groups selected to export.")

        # Validate destination root safety (forbids DCIM / read-only targets)
        self.safe_fs.mkdir_quarantine(dest_root)

        reports: list[ExportGroupReport] = []
        total_photos = 0
        total_files = 0

        for gkey, folder_name, photos in selected:
            subdir = dest_root / folder_name
            group_files = 0
            group_photos = 0
            for item in photos:
                assets = self.repo.list_assets_for_photo(item.photo_id)
                existing_assets = [
                    (a, Path(a.abs_path)) for a in assets if Path(a.abs_path).is_file()
                ]
                if not existing_assets:
                    continue

                # Determine a single shared suffix for all assets in this atomic photo group so
                # RAW + JPG + LivePhoto + XMP keep matching stems, and skip idempotent repeat exports.
                chosen_targets: list[tuple[Any, Path, Path]] | None = None
                already_exported = False
                suffix_idx = 0
                while True:
                    cand: list[tuple[Any, Path, Path]] = [
                        (
                            a_rec,
                            s_path,
                            subdir
                            / (
                                s_path.name
                                if suffix_idx == 0
                                else f"{s_path.stem}__{suffix_idx}{s_path.suffix}"
                            ),
                        )
                        for a_rec, s_path in existing_assets
                    ]
                    identical = True
                    for a_rec, s_path, d_path in cand:
                        if not d_path.is_file():
                            identical = False
                            break
                        try:
                            if (
                                d_path.stat().st_size != s_path.stat().st_size
                                or compute_content_fp(d_path) != a_rec.content_fp
                            ):
                                identical = False
                                break
                        except OSError:
                            identical = False
                            break
                    if identical:
                        already_exported = True
                        break
                    if not any(d_path.exists() or d_path.is_symlink() for _, _, d_path in cand):
                        chosen_targets = cand
                        break
                    suffix_idx += 1

                if already_exported:
                    continue

                assert chosen_targets is not None
                wrote_any = False
                for _a_rec, src_path, dst_path in chosen_targets:
                    data = self.safe_fs.read(src_path)
                    self.safe_fs.write_new_copy(
                        dst_path,
                        data,
                        allow_suffix=False,
                    )
                    group_files += 1
                    wrote_any = True
                if wrote_any:
                    group_photos += 1
            reports.append(
                ExportGroupReport(
                    group_key=gkey,
                    folder_name=folder_name,
                    photo_count=group_photos,
                    file_count=group_files,
                )
            )
            total_photos += group_photos
            total_files += group_files

        return PeopleExportResult(
            target_dir=str(dest_root),
            exported_photos=total_photos,
            exported_files=total_files,
            groups=tuple(reports),
        )


__all__ = [
    "DEFAULT_FACE_COSINE_THRESHOLD",
    "GROUP_KEY_GROUP_PHOTO",
    "GROUP_KEY_NO_FACE",
    "GROUP_KEY_UNNAMED",
    "SIMILAR_CANDIDATE_MIN_COSINE",
    "ExportGroupReport",
    "FaceSlotInfo",
    "PeopleError",
    "PeopleExportResult",
    "PeopleGroupsResponse",
    "PeopleIndexResult",
    "PeoplePhotoItem",
    "PeopleService",
    "PersonGroupInfo",
    "SimilarFaceCandidate",
    "UnnamedClusterInfo",
]
