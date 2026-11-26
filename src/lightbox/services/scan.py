"""ScanService: orchestrates zero-LLM deterministic scan (ingest -> analyze -> bursts -> verdicts)."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from lightbox.config import AppConfig
from lightbox.core import (
    ENGINE_VERSION,
    BurstCandidate,
    DetectedFace,
    ExifExtractor,
    FaceBackend,
    FaceDetector,
    ImageMetrics,
    NormalizedExif,
    PhotoContext,
    PhotoEvaluationInput,
    SensitivityMode,
    compute_content_fp,
    compute_dhash,
    compute_image_metrics,
    detect_bursts,
    evaluate_session_verdicts,
    find_face_model_path,
    generate_previews,
    get_cached_face_detector,
    load_analysis_gray,
    make_motion_thumb,
    preview_cache_paths,
    scan_directory,
)
from lightbox.fs import is_camera_card_path, is_readonly_path
from lightbox.jobs import JobRunner, ProgressCallback, StageWorkItem, make_cache_key
from lightbox.models import Embedder
from lightbox.paths import AppPaths
from lightbox.services.search import SearchService
from lightbox.store import (
    AssetRecord,
    ExifRecord,
    FaceRecord,
    LightboxRepository,
    MetricsRecord,
    PhotoRecord,
    RootRecord,
    SessionRecord,
    VectorIndex,
)


@dataclass(frozen=True, slots=True)
class _AnalyzeTask:
    photo_id: str
    content_fp: str
    source_path: str
    cache_dir: str
    exif_orientation: int | None
    max_cache_bytes: int | None
    exif: NormalizedExif
    face_model_path: str | None
    face_model_tag: str


@dataclass(frozen=True, slots=True)
class _AnalyzeOutput:
    photo_id: str
    content_fp: str
    preview_w: int
    preview_h: int
    metrics: ImageMetrics
    dhash: int
    faces: tuple[DetectedFace, ...]
    analysis_path: str
    face_model_tag: str


@dataclass(frozen=True, slots=True)
class ScanResult:
    session_id: str
    job_id: str
    root_path: Path
    sensitivity: SensitivityMode
    total_photos: int
    counts_by_bucket: dict[str, int]
    burst_count: int
    cache_hits: int
    skipped_retryable: int
    elapsed_s: float
    status: str
    embedded_count: int = 0


@dataclass(frozen=True, slots=True)
class JobStatusSummary:
    job_id: str
    kind: str
    session_id: str
    status: str
    done: int
    total: int
    spent_usd: float
    paused_reason: str | None
    heartbeat_at: float | None
    stage: str = "analyze"


def _resolve_face_model_info(
    app_paths: AppPaths,
    face_backend: FaceBackend | None,
) -> tuple[str | None, str]:
    """Return `(face_model_path_str, face_model_tag)` for cache invalidation & session reuse."""
    if face_backend is not None:
        if face_backend.available:
            return None, f"custom:{type(face_backend).__qualname__}"
        return None, "none"
    model_path = find_face_model_path(app_paths)
    if model_path is None:
        return None, "none"
    model_fp = compute_content_fp(model_path)[:12]
    return str(model_path.resolve()), f"antelopev2:{model_path.name}:{model_fp}"


def _analyze_photo_worker(task: _AnalyzeTask) -> _AnalyzeOutput:
    """Top-level picklable worker executing preview + metrics + cached ONNX face detection."""
    source_path = Path(task.source_path)
    cache_dir = Path(task.cache_dir)
    preview = generate_previews(
        source_path,
        task.content_fp,
        cache_dir,
        exif_orientation=task.exif_orientation,
        max_cache_bytes=task.max_cache_bytes,
    )
    gray = load_analysis_gray(preview.analysis_path)
    metrics = compute_image_metrics(gray, exif=task.exif)
    dhash = compute_dhash(gray)

    faces: tuple[DetectedFace, ...] = ()
    if task.face_model_path is not None:
        face_detector = get_cached_face_detector(task.face_model_path)
        if face_detector.model_available:
            ctx = PhotoContext(
                photo_id=task.photo_id,
                content_fp=task.content_fp,
                analysis_img=gray,
                analysis_path=preview.analysis_path,
                loupe_path=preview.loupe_path,
                exif=task.exif,
                metrics={"sharp_p95": metrics.sharp_p95, "noise_sigma": metrics.noise_sigma},
            )
            face_detector.run(ctx)
            faces = ctx.faces

    return _AnalyzeOutput(
        photo_id=task.photo_id,
        content_fp=task.content_fp,
        preview_w=preview.preview_w,
        preview_h=preview.preview_h,
        metrics=metrics,
        dhash=dhash,
        faces=faces,
        analysis_path=str(preview.analysis_path),
        face_model_tag=task.face_model_tag,
    )


def _stable_id(prefix: str, key: str) -> str:
    return f"{prefix}_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}"


def _choose_preview_source_path(assets: list[AssetRecord]) -> Path:
    """Prefer JPG/HEIC over RAW for preview extraction when a paired sibling exists (faster)."""
    priority = {"jpg": 0, "heic": 1, "raw": 2}
    stills = [a for a in assets if a.kind in priority]
    chosen = min(stills, key=lambda a: (priority[a.kind], a.rel_path))
    return Path(chosen.abs_path)


def _metrics_record_to_image_metrics(rec: MetricsRecord) -> ImageMetrics:
    exp = rec.exposure or {}
    cell_raw = exp.get("sharpest_cell", [8, 8])
    sharpest_cell = (int(cell_raw[0]), int(cell_raw[1])) if len(cell_raw) == 2 else (8, 8)
    return ImageMetrics(
        sharp_p95=float(rec.sharp_p95 or 0.0),
        noise_sigma=float(rec.noise_sigma or 0.0),
        motion_aniso=float(rec.motion_aniso or 0.0),
        motion_state=rec.motion_state or "unknown",
        texture=float(rec.texture or 0.0),
        exposure=dict(exp),
        sharpest_cell=sharpest_cell,
        panning_detected=bool(exp.get("panning_detected", False)),
        is_extreme_exposure=bool(exp.get("is_extreme_exposure", False)),
        extreme_kind=str(exp["extreme_kind"]) if exp.get("extreme_kind") is not None else None,
        cell_sharpness_p50=float(exp.get("cell_sharpness_p50", 0.0)),
    )


class ScanService:
    """Zero-LLM deterministic folder scan service backed by `JobRunner` and SQLite."""

    def __init__(
        self,
        repo: LightboxRepository,
        app_paths: AppPaths,
        config: AppConfig,
        *,
        job_runner: JobRunner | None = None,
        vector_index: VectorIndex | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.repo = repo
        self.app_paths = app_paths
        self.config = config
        self.job_runner = job_runner or JobRunner(repo)
        self.vector_index = vector_index
        self.embedder = embedder

    def scan_folder(
        self,
        folder_path: Path | str,
        *,
        sensitivity: SensitivityMode | None = None,
        max_workers: int | None = None,
        job_id: str | None = None,
        face_backend: FaceBackend | None = None,
        exif_extractor: ExifExtractor | None = None,
        on_progress: ProgressCallback | None = None,
        worker_override: Any = None,
        enable_embed: bool = False,
        embedder: Embedder | None = None,
    ) -> ScanResult:
        """Scan `folder_path` end-to-end with resumable caching, burst detection, and verdicts."""
        started = time.monotonic()
        root = Path(folder_path).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Scan target is not a directory: {root}")

        sens: SensitivityMode = sensitivity or self.config.sensitivity
        self.app_paths.ensure_dirs()

        # 1. Register root & session
        st_dev = str(os.stat(root).st_dev)
        is_card = is_camera_card_path(root)
        is_ro = is_card or is_readonly_path(root)
        root_id = _stable_id("root", str(root))
        self.repo.upsert_root(
            RootRecord(
                id=root_id,
                path=str(root),
                device_id=st_dev,
                kind="camera_card" if is_card else "disk",
                readonly=is_ro,
            )
        )

        existing_session = self.repo.get_session_by_root_path(str(root))
        session_id = existing_session.id if existing_session is not None else _stable_id("sess", str(root))
        now = time.time()
        self.repo.upsert_session(
            SessionRecord(
                id=session_id,
                root_id=root_id,
                root_path=str(root),
                name=root.name or "session",
                sensitivity=sens,
                engine_version=ENGINE_VERSION,
                created_at=existing_session.created_at if existing_session is not None else now,
                last_scanned_at=existing_session.last_scanned_at if existing_session else None,
            )
        )

        job = self.job_runner.create_or_resume_job(
            kind="scan",
            session_id=session_id,
            job_id=job_id,
            budget_usd=self.config.budget.session_budget_usd,
        )

        # 2. Stage 1: Ingest directory & persist photos / assets / EXIF
        ingest = scan_directory(root, exif_extractor=exif_extractor)
        if on_progress is not None:
            on_progress("ingest", len(ingest.groups), len(ingest.groups))

        photo_exif_map: dict[str, NormalizedExif] = {}
        photo_file_map: dict[str, Path] = {}
        work_items: list[StageWorkItem[_AnalyzeTask]] = []
        max_cache_bytes = int(self.config.cache.preview_max_gb * (1024**3))
        face_model_path, face_model_tag = _resolve_face_model_info(self.app_paths, face_backend)

        for group in ingest.groups:
            photo_id = _stable_id("ph", f"{session_id}:{group.rel_dir}:{group.stem}:{group.primary_content_fp}")
            ex = group.exif
            photo_exif_map[photo_id] = ex
            photo_file_map[photo_id] = group.primary_asset.abs_path

            existing_photo = self.repo.get_photo(photo_id)
            self.repo.upsert_photo(
                PhotoRecord(
                    id=photo_id,
                    session_id=session_id,
                    stem=group.stem,
                    rel_dir=group.rel_dir,
                    primary_content_fp=group.primary_content_fp,
                    cohort=ex.cohort,
                    capture_time=ex.capture_time,
                    camera_serial=ex.camera_serial,
                    camera_model=ex.camera_model,
                    lens_model=ex.lens_model,
                    preview_w=existing_photo.preview_w if existing_photo else None,
                    preview_h=existing_photo.preview_h if existing_photo else None,
                    is_protected=existing_photo.is_protected if existing_photo else False,
                    exempt_reason=existing_photo.exempt_reason if existing_photo else None,
                    is_private=existing_photo.is_private if existing_photo else False,
                    decision_version=existing_photo.decision_version if existing_photo else 1,
                )
            )

            asset_records: list[AssetRecord] = []
            for asset in group.assets:
                rec = AssetRecord(
                    photo_id=photo_id,
                    rel_path=asset.rel_path,
                    abs_path=str(asset.abs_path.resolve()),
                    kind=asset.kind,
                    size_bytes=asset.size_bytes,
                    mtime=asset.mtime,
                    content_fp=asset.content_fp,
                )
                self.repo.upsert_asset(rec)
                asset_records.append(rec)

            self.repo.upsert_exif(
                ExifRecord(
                    photo_id=photo_id,
                    aperture=ex.aperture,
                    shutter_s=ex.shutter_s,
                    iso=ex.iso,
                    focal_eq=ex.focal_eq,
                    ev_comp=ex.ev_comp,
                    ev100=ex.ev100,
                    light_bucket=ex.light_bucket,
                    exposure_program=ex.exposure_program,
                    af_mode=ex.af_mode,
                    flash_fired=ex.flash_fired,
                    capture_local_hour=ex.capture_local_hour,
                    drive_mode=ex.drive_mode,
                    bracket_group=ex.bracket_group,
                    rating=ex.rating,
                    protected=ex.protected,
                    gps_lat=ex.gps_lat,
                    gps_lon=ex.gps_lon,
                    field_confidence=ex.field_confidence,
                )
            )

            preview_src = _choose_preview_source_path(asset_records)
            cache_key = make_cache_key(
                group.primary_content_fp, "analyze", ENGINE_VERSION, face_model_tag
            )
            task = _AnalyzeTask(
                photo_id=photo_id,
                content_fp=group.primary_content_fp,
                source_path=str(preview_src),
                cache_dir=str(self.app_paths.cache_dir),
                exif_orientation=ex.orientation,
                max_cache_bytes=max_cache_bytes,
                exif=ex,
                face_model_path=face_model_path,
                face_model_tag=face_model_tag,
            )
            work_items.append(
                StageWorkItem(photo_id=photo_id, cache_key=cache_key, payload=task)
            )

        # 3. Stage 2: Preview + Metrics + Face Detection via resumable JobRunner
        def _cache_lookup(wi: StageWorkItem[_AnalyzeTask]) -> _AnalyzeOutput | None:
            t = wi.payload
            db_metrics = self.repo.get_metrics(t.content_fp, ENGINE_VERSION)
            if db_metrics is None:
                return None
            # Invalidate analysis cache whenever face model availability/version changes
            stored_face_tag = (db_metrics.exposure or {}).get("face_model_tag")
            if stored_face_tag != t.face_model_tag:
                return None
            grid_p, loupe_p, analysis_p = preview_cache_paths(Path(t.cache_dir), t.content_fp)
            if not (grid_p.is_file() and loupe_p.is_file() and analysis_p.is_file()):
                return None
            with Image.open(loupe_p) as img:
                pw, ph = img.size
            gray = load_analysis_gray(analysis_p)
            dhash = compute_dhash(gray)
            db_faces = self.repo.list_faces(t.content_fp)
            faces = tuple(
                DetectedFace(
                    idx=f.idx,
                    bbox=f.bbox,
                    landmarks=f.landmarks,
                    det_score=float(f.det_score or 0.0),
                    sharpness=float(f.sharpness or 0.0),
                    area_ratio=float(f.area_ratio or 0.0),
                )
                for f in db_faces
            )
            return _AnalyzeOutput(
                photo_id=t.photo_id,
                content_fp=t.content_fp,
                preview_w=pw,
                preview_h=ph,
                metrics=_metrics_record_to_image_metrics(db_metrics),
                dhash=dhash,
                faces=faces,
                analysis_path=str(analysis_p),
                face_model_tag=t.face_model_tag,
            )

        def _persist_analyze_result(wi: StageWorkItem[_AnalyzeTask], out: _AnalyzeOutput) -> None:
            photo = self.repo.get_photo(out.photo_id)
            if photo is not None:
                self.repo.upsert_photo(
                    PhotoRecord(
                        id=photo.id,
                        session_id=photo.session_id,
                        stem=photo.stem,
                        rel_dir=photo.rel_dir,
                        primary_content_fp=photo.primary_content_fp,
                        cohort=photo.cohort,
                        capture_time=photo.capture_time,
                        camera_serial=photo.camera_serial,
                        camera_model=photo.camera_model,
                        lens_model=photo.lens_model,
                        preview_w=out.preview_w,
                        preview_h=out.preview_h,
                        is_protected=photo.is_protected,
                        exempt_reason=photo.exempt_reason,
                        is_private=photo.is_private,
                        decision_version=photo.decision_version,
                    )
                )
            exp_payload = dict(out.metrics.exposure)
            exp_payload["cell_sharpness_p50"] = out.metrics.cell_sharpness_p50
            exp_payload["face_model_tag"] = out.face_model_tag
            self.repo.upsert_metrics(
                MetricsRecord(
                    content_fp=out.content_fp,
                    engine_version=ENGINE_VERSION,
                    sharp_p95=out.metrics.sharp_p95,
                    sharp_z=None,
                    noise_sigma=out.metrics.noise_sigma,
                    motion_aniso=out.metrics.motion_aniso,
                    motion_state=out.metrics.motion_state,
                    exposure=exp_payload,
                    texture=out.metrics.texture,
                    face_count=len(out.faces),
                    face_sharp_min=(
                        min(f.sharpness for f in out.faces) if out.faces else None
                    ),
                )
            )
            self.repo.replace_faces(
                out.content_fp,
                [
                    FaceRecord(
                        content_fp=out.content_fp,
                        idx=f.idx,
                        bbox=f.bbox,
                        landmarks=f.landmarks,
                        det_score=f.det_score,
                        sharpness=f.sharpness,
                        area_ratio=f.area_ratio,
                    )
                    for f in out.faces
                ],
            )

        def _inproc_worker_with_custom_face(task: _AnalyzeTask) -> _AnalyzeOutput:
            out = _analyze_photo_worker(task)
            if face_backend is not None:
                gray = load_analysis_gray(Path(out.analysis_path))
                fd = FaceDetector(backend=face_backend)
                ctx = PhotoContext(
                    photo_id=task.photo_id,
                    content_fp=task.content_fp,
                    analysis_img=gray,
                    exif=task.exif,
                    metrics={
                        "sharp_p95": out.metrics.sharp_p95,
                        "noise_sigma": out.metrics.noise_sigma,
                    },
                )
                fd.run(ctx)
                return _AnalyzeOutput(
                    photo_id=out.photo_id,
                    content_fp=out.content_fp,
                    preview_w=out.preview_w,
                    preview_h=out.preview_h,
                    metrics=out.metrics,
                    dhash=out.dhash,
                    faces=ctx.faces,
                    analysis_path=out.analysis_path,
                    face_model_tag=task.face_model_tag,
                )
            return out

        effective_workers = max_workers if max_workers is not None else min(4, os.cpu_count() or 1)
        use_pool = (
            face_backend is None
            and worker_override is None
            and effective_workers > 1
            and len(work_items) >= 4
        )
        chosen_worker = (
            worker_override
            if worker_override is not None
            else (_analyze_photo_worker if use_pool else _inproc_worker_with_custom_face)
        )

        stage_res = self.job_runner.run_stage(
            job_id=job.id,
            stage="analyze",
            items=work_items,
            worker_fn=chosen_worker,
            cache_lookup_fn=_cache_lookup,
            on_item_result=_persist_analyze_result,
            max_workers=effective_workers,
            use_process_pool=use_pool,
            on_progress=on_progress,
        )

        if stage_res.paused:
            return ScanResult(
                session_id=session_id,
                job_id=job.id,
                root_path=root,
                sensitivity=sens,
                total_photos=len(work_items),
                counts_by_bucket={"red": 0, "yellow": 0, "normal": 0, "exempt": 0},
                burst_count=0,
                cache_hits=stage_res.cache_hits,
                skipped_retryable=stage_res.skipped_retryable,
                elapsed_s=round(time.monotonic() - started, 3),
                status="paused",
            )

        # 4. Stage 3: Burst clustering + inter-frame motion + cohort calibration + verdicts
        burst_candidates: list[BurstCandidate] = []
        for pid, out in stage_res.results.items():
            ex = photo_exif_map[pid]
            gray = load_analysis_gray(Path(out.analysis_path))
            burst_candidates.append(
                BurstCandidate(
                    photo_id=pid,
                    capture_time=ex.capture_time,
                    camera_serial=ex.camera_serial,
                    camera_model=ex.camera_model,
                    cohort=ex.cohort,
                    dhash=out.dhash,
                    sharp_p95=out.metrics.sharp_p95,
                    noise_sigma=out.metrics.noise_sigma,
                    face_sharp_min=(
                        min(f.sharpness for f in out.faces) if out.faces else None
                    ),
                    is_extreme_exposure=out.metrics.is_extreme_exposure,
                    motion_thumb=make_motion_thumb(gray),
                )
            )

        burst_groups, burst_motion_map = detect_bursts(session_id, burst_candidates)
        self.repo.clear_unconfirmed_bursts(session_id)
        existing_confirmed_ids = {
            b.id for b in self.repo.list_bursts(session_id) if b.confirmed
        }
        for grp in burst_groups:
            if grp.burst_id in existing_confirmed_ids:
                continue
            self.repo.upsert_burst(grp.burst_id, session_id, kind=grp.kind, confirmed=False)
            for member in grp.members:
                self.repo.upsert_burst_member(
                    grp.burst_id,
                    member.photo_id,
                    rank=member.rank,
                    is_pick=member.is_pick,
                )

        eval_inputs: list[PhotoEvaluationInput] = []
        for pid, out in stage_res.results.items():
            eval_inputs.append(
                PhotoEvaluationInput(
                    photo_id=pid,
                    content_fp=out.content_fp,
                    metrics=out.metrics,
                    dhash=out.dhash,
                    exif=photo_exif_map[pid],
                    faces=out.faces,
                    file_path=photo_file_map[pid],
                    burst_frame_motion=burst_motion_map.get(pid),
                )
            )

        verdict_outputs = evaluate_session_verdicts(eval_inputs, sensitivity=sens)
        counts: dict[str, int] = {"red": 0, "yellow": 0, "normal": 0, "exempt": 0}

        for vo in verdict_outputs:
            counts[vo.bucket] = counts.get(vo.bucket, 0) + 1
            out = stage_res.results[vo.photo_id]

            # Persist calibrated sharp_z and final motion_state back to metrics (preserving scene_tag)
            existing_metrics = self.repo.get_metrics(out.content_fp, ENGINE_VERSION)
            preserved_scene_tag = existing_metrics.scene_tag if existing_metrics is not None else None
            exp_payload = dict(out.metrics.exposure)
            exp_payload["cell_sharpness_p50"] = out.metrics.cell_sharpness_p50
            exp_payload["face_model_tag"] = out.face_model_tag
            self.repo.upsert_metrics(
                MetricsRecord(
                    content_fp=out.content_fp,
                    engine_version=ENGINE_VERSION,
                    sharp_p95=out.metrics.sharp_p95,
                    sharp_z=vo.sharp_z,
                    noise_sigma=out.metrics.noise_sigma,
                    motion_aniso=out.metrics.motion_aniso,
                    motion_state=vo.motion_state,
                    exposure=exp_payload,
                    texture=out.metrics.texture,
                    face_count=vo.face_count,
                    face_sharp_min=vo.face_sharp_min,
                    scene_tag=preserved_scene_tag,
                )
            )

            # Update photo exemption protection flag if needed
            photo = self.repo.get_photo(vo.photo_id)
            if photo is not None:
                if photo.is_protected != vo.is_protected or photo.exempt_reason != vo.exempt_reason:
                    self.repo.upsert_photo(
                        PhotoRecord(
                            id=photo.id,
                            session_id=photo.session_id,
                            stem=photo.stem,
                            rel_dir=photo.rel_dir,
                            primary_content_fp=photo.primary_content_fp,
                            cohort=photo.cohort,
                            capture_time=photo.capture_time,
                            camera_serial=photo.camera_serial,
                            camera_model=photo.camera_model,
                            lens_model=photo.lens_model,
                            preview_w=photo.preview_w,
                            preview_h=photo.preview_h,
                            is_protected=vo.is_protected,
                            exempt_reason=vo.exempt_reason,
                            is_private=photo.is_private,
                            decision_version=photo.decision_version,
                        )
                    )
                self.repo.upsert_session_fts(
                    photo_id=photo.id,
                    session_name=root.name or "session",
                    rel_dir=photo.rel_dir,
                    stem=photo.stem,
                    scene_tag=preserved_scene_tag or "",
                )

            self.repo.add_verdict(
                photo_id=vo.photo_id,
                source="engine",
                bucket=vo.bucket,
                confidence=vo.confidence,
                reason_codes=vo.reason_codes,
                detail=vo.detail,
                increment_decision_version=False,
            )

        if on_progress is not None:
            on_progress("calibrate", len(verdict_outputs), len(verdict_outputs))

        # 5. Optional Stage 4: L2 SigLIP 2 embedding + zero-text-tower scene tagging (PLAN T11)
        embedded_count = 0
        active_embedder = embedder or self.embedder
        if enable_embed and self.vector_index is not None and active_embedder is not None:
            search_svc = SearchService(
                self.repo,
                self.vector_index,
                active_embedder,
                app_paths=self.app_paths,
            )
            embed_res = search_svc.index_session(session_id, on_progress=on_progress)
            embedded_count = embed_res.indexed_count

        finished_ts = time.time()
        self.repo.upsert_session(
            SessionRecord(
                id=session_id,
                root_id=root_id,
                root_path=str(root),
                name=root.name or "session",
                sensitivity=sens,
                engine_version=ENGINE_VERSION,
                created_at=existing_session.created_at if existing_session is not None else now,
                last_scanned_at=finished_ts,
            )
        )
        self.job_runner.mark_completed(job.id)

        return ScanResult(
            session_id=session_id,
            job_id=job.id,
            root_path=root,
            sensitivity=sens,
            total_photos=len(work_items),
            counts_by_bucket=counts,
            burst_count=len(self.repo.list_bursts(session_id)),
            cache_hits=stage_res.cache_hits,
            skipped_retryable=stage_res.skipped_retryable,
            elapsed_s=round(time.monotonic() - started, 3),
            status="completed",
            embedded_count=embedded_count,
        )

    def get_job_status(self, job_id: str) -> JobStatusSummary:
        job = self.repo.get_job(job_id)
        if job is None:
            raise ValueError(f"Unknown job_id: {job_id}")
        items = self.repo.list_job_items(job_id)
        done = sum(1 for it in items if it.status == "done")
        stage = "done" if job.status == "completed" else (items[0].stage if items else "queued")
        return JobStatusSummary(
            job_id=job.id,
            kind=job.kind,
            session_id=job.session_id,
            status=job.status,
            done=done,
            total=len(items),
            spent_usd=job.spent_usd,
            paused_reason=job.paused_reason,
            heartbeat_at=job.heartbeat_at,
            stage=stage,
        )

    def pause_job(self, job_id: str, *, reason: str = "user_paused") -> JobStatusSummary:
        job = self.repo.get_job(job_id)
        if job is None:
            raise ValueError(f"Unknown job_id: {job_id}")
        if job.status in {"queued", "running"}:
            self.job_runner.mark_paused(job_id, reason)
        return self.get_job_status(job_id)

    def resume_job(
        self,
        job_id: str,
        *,
        max_workers: int | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> ScanResult:
        job = self.repo.get_job(job_id)
        if job is None:
            raise ValueError(f"Unknown job_id: {job_id}")
        session = self.repo.get_session(job.session_id)
        if session is None:
            raise ValueError(f"Unknown session_id for job {job_id}: {job.session_id}")
        sens: SensitivityMode = (
            "standard" if session.sensitivity == "standard" else "conservative"
        )
        return self.scan_folder(
            Path(session.root_path),
            sensitivity=sens,
            max_workers=max_workers,
            job_id=job.id,
            on_progress=on_progress,
        )


__all__ = ["JobStatusSummary", "ScanResult", "ScanService"]
