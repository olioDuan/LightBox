"""Core SQLite repositories, effective verdict queries, and optimistic locking (SPEC §3.2–3.3)."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from lightbox.store.db import Database


@dataclass(frozen=True, slots=True)
class RootRecord:
    id: str
    path: str
    device_id: str
    kind: str  # 'disk' | 'camera_card'
    readonly: bool = False


@dataclass(frozen=True, slots=True)
class SessionRecord:
    id: str
    root_id: str
    root_path: str
    name: str
    sensitivity: str
    engine_version: str
    created_at: float
    last_scanned_at: float | None = None


@dataclass(frozen=True, slots=True)
class PhotoRecord:
    id: str
    session_id: str
    stem: str
    rel_dir: str
    primary_content_fp: str
    cohort: str  # 'camera' | 'phone'
    capture_time: float | None = None
    camera_serial: str | None = None
    camera_model: str | None = None
    lens_model: str | None = None
    preview_w: int | None = None
    preview_h: int | None = None
    is_protected: bool = False
    exempt_reason: str | None = None
    is_private: bool = False
    decision_version: int = 1


@dataclass(frozen=True, slots=True)
class AssetRecord:
    photo_id: str
    rel_path: str
    abs_path: str
    kind: str  # 'raw' | 'jpg' | 'heic' | 'live_mov' | 'sidecar'
    size_bytes: int
    mtime: float
    content_fp: str
    id: int | None = None


@dataclass(frozen=True, slots=True)
class ExifRecord:
    photo_id: str
    aperture: float | None = None
    shutter_s: float | None = None
    iso: int | None = None
    focal_eq: float | None = None
    ev_comp: float | None = None
    ev100: float | None = None
    light_bucket: str | None = None
    exposure_program: str | None = None
    af_mode: str | None = None
    flash_fired: bool | None = None
    capture_local_hour: int | None = None
    drive_mode: str | None = None
    bracket_group: str | None = None
    rating: int | None = None
    protected: bool | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    field_confidence: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MetricsRecord:
    content_fp: str
    engine_version: str
    sharp_p95: float | None = None
    sharp_z: float | None = None
    noise_sigma: float | None = None
    motion_aniso: float | None = None
    motion_state: str | None = None  # 'static' | 'moving' | 'panning' | 'unknown'
    exposure: dict[str, Any] | None = None
    texture: float | None = None
    face_count: int | None = None
    face_sharp_min: float | None = None
    scene_tag: str | None = None


@dataclass(frozen=True, slots=True)
class FaceRecord:
    content_fp: str
    idx: int
    bbox: tuple[float, float, float, float]
    landmarks: tuple[tuple[float, float], ...] | None = None
    det_score: float | None = None
    sharpness: float | None = None
    area_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class VerdictRecord:
    id: int
    photo_id: str
    source: str
    bucket: str
    confidence: float
    reason_codes: tuple[str, ...]
    detail: dict[str, Any] | None
    llm_call_id: str | None
    created_at: float


@dataclass(frozen=True, slots=True)
class PlanRecord:
    id: str
    session_id: str
    action_type: str  # 'quarantine_move' | 'export_clean'
    plan_hash: str
    status: str  # 'draft' | 'reviewed' | 'approved' | 'applying' | 'applied' | 'reverted' | 'stale'
    verifier_report_json: str | None = None
    approval_code_hash: str | None = None
    approval_attempts: int = 0
    approval_expires_at: float | None = None
    approved_by: str | None = None
    approved_at: float | None = None
    quarantine_dir: str | None = None


@dataclass(frozen=True, slots=True)
class PlanItemRecord:
    plan_id: str
    photo_id: str
    decision_version: int
    src_paths: tuple[str, ...]
    dst_paths: tuple[str, ...]
    status: str = "pending"  # 'pending' | 'done' | 'reverted'
    done_at: float | None = None


@dataclass(frozen=True, slots=True)
class BurstRecord:
    id: str
    session_id: str
    kind: str  # 'burst' | 'reshoot'
    confirmed: bool = False


@dataclass(frozen=True, slots=True)
class BurstMemberRecord:
    burst_id: str
    photo_id: str
    rank: int
    is_pick: bool = False
    ai_note: str | None = None


@dataclass(frozen=True, slots=True)
class JobRecord:
    id: str
    kind: str  # 'scan' | 'triage_vlm' | 'embed'
    session_id: str
    status: str  # 'queued' | 'running' | 'paused' | 'completed' | 'failed'
    created_at: float
    paused_reason: str | None = None
    budget_usd: float = 0.50
    spent_usd: float = 0.0
    heartbeat_at: float | None = None


@dataclass(frozen=True, slots=True)
class JobItemRecord:
    job_id: str
    photo_id: str
    stage: str
    status: str  # 'pending' | 'done' | 'skipped_retryable' | 'failed'
    attempts: int = 0
    cache_key: str | None = None
    last_error: str | None = None
    updated_at: float | None = None


@dataclass(frozen=True, slots=True)
class OverrideRecord:
    id: int
    photo_id: str
    session_id: str
    from_bucket: str
    to_bucket: str
    reason_tag: str | None
    feature_snapshot: dict[str, Any]
    actor: str
    created_at: float


@dataclass(frozen=True, slots=True)
class LLMCallRecord:
    id: str
    cache_key: str | None
    role: str
    provider: str
    model: str
    in_tokens: int
    out_tokens: int
    cached_tokens: int
    cost_usd: float
    latency_ms: int
    response_json: str | None
    created_at: float
    session_id: str | None = None
    cache_hit: bool = False


@dataclass(frozen=True, slots=True)
class PersonRecord:
    id: str
    name: str
    is_self: bool = False
    cover_content_fp: str | None = None
    cover_face_idx: int | None = None
    created_at: float = 0.0


@dataclass(frozen=True, slots=True)
class FaceEmbeddingRecord:
    content_fp: str
    face_idx: int
    model_id: str
    dim: int
    embedding_blob: bytes
    created_at: float = 0.0


@dataclass(frozen=True, slots=True)
class FaceAssignmentRecord:
    content_fp: str
    face_idx: int
    person_id: str | None
    status: str  # 'assigned' | 'unassigned' | 'ignored'
    source: str  # 'user' | 'auto'
    confidence: float | None = None
    cluster_id: str | None = None
    updated_at: float = 0.0


@dataclass(frozen=True, slots=True)
class DigestOverviewRow:
    total_shots: int
    tech_reject: int
    burst_cull: int
    subjective_reject: int
    kept: int
    uncertain: int
    session_count: int


@dataclass(frozen=True, slots=True)
class DigestBucketRow:
    bucket_key: str
    total: int
    tech_reject: int
    burst_cull: int
    subjective_reject: int
    kept: int
    uncertain: int
    top_reason_code: str | None = None
    top_reason_count: int = 0


@dataclass(frozen=True, slots=True)
class DigestMatrixRow:
    focal_bucket: str
    shutter_bucket: str
    total: int
    tech_reject: int
    burst_cull: int
    subjective_reject: int
    kept: int
    uncertain: int


_DIGEST_ALLOWED_DIMENSIONS: frozenset[str] = frozenset(
    {
        "light_bucket",
        "capture_local_hour",
        "flash_fired",
        "scene_tag",
        "motion_state",
        "lens_model",
        "focal_bucket",
        "shutter_vs_safe",
        "iso_bucket",
    }
)


def _row_to_root(row: sqlite3.Row) -> RootRecord:
    return RootRecord(
        id=str(row["id"]),
        path=str(row["path"]),
        device_id=str(row["device_id"]),
        kind=str(row["kind"]),
        readonly=bool(row["readonly"]),
    )


def _row_to_session(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        id=str(row["id"]),
        root_id=str(row["root_id"]),
        root_path=str(row["root_path"]),
        name=str(row["name"]),
        sensitivity=str(row["sensitivity"]),
        engine_version=str(row["engine_version"]),
        created_at=float(row["created_at"]),
        last_scanned_at=float(row["last_scanned_at"]) if row["last_scanned_at"] is not None else None,
    )


def _row_to_photo(row: sqlite3.Row) -> PhotoRecord:
    return PhotoRecord(
        id=str(row["id"]),
        session_id=str(row["session_id"]),
        stem=str(row["stem"]),
        rel_dir=str(row["rel_dir"]),
        primary_content_fp=str(row["primary_content_fp"]),
        cohort=str(row["cohort"]),
        capture_time=float(row["capture_time"]) if row["capture_time"] is not None else None,
        camera_serial=str(row["camera_serial"]) if row["camera_serial"] is not None else None,
        camera_model=str(row["camera_model"]) if row["camera_model"] is not None else None,
        lens_model=str(row["lens_model"]) if row["lens_model"] is not None else None,
        preview_w=int(row["preview_w"]) if row["preview_w"] is not None else None,
        preview_h=int(row["preview_h"]) if row["preview_h"] is not None else None,
        is_protected=bool(row["is_protected"]),
        exempt_reason=str(row["exempt_reason"]) if row["exempt_reason"] is not None else None,
        is_private=bool(row["is_private"]),
        decision_version=int(row["decision_version"]),
    )


def _row_to_asset(row: sqlite3.Row) -> AssetRecord:
    return AssetRecord(
        id=int(row["id"]),
        photo_id=str(row["photo_id"]),
        rel_path=str(row["rel_path"]),
        abs_path=str(row["abs_path"]),
        kind=str(row["kind"]),
        size_bytes=int(row["size_bytes"]),
        mtime=float(row["mtime"]),
        content_fp=str(row["content_fp"]),
    )


def _row_to_verdict(row: sqlite3.Row) -> VerdictRecord:
    reasons_raw = json.loads(str(row["reason_codes_json"]))
    detail_raw = json.loads(str(row["detail_json"])) if row["detail_json"] is not None else None
    return VerdictRecord(
        id=int(row["id"]),
        photo_id=str(row["photo_id"]),
        source=str(row["source"]),
        bucket=str(row["bucket"]),
        confidence=float(row["confidence"]),
        reason_codes=tuple(str(r) for r in reasons_raw),
        detail=detail_raw,
        llm_call_id=str(row["llm_call_id"]) if row["llm_call_id"] is not None else None,
        created_at=float(row["created_at"]),
    )


def _row_to_plan(row: sqlite3.Row) -> PlanRecord:
    return PlanRecord(
        id=str(row["id"]),
        session_id=str(row["session_id"]),
        action_type=str(row["action_type"]),
        plan_hash=str(row["plan_hash"]),
        status=str(row["status"]),
        verifier_report_json=(
            str(row["verifier_report_json"]) if row["verifier_report_json"] is not None else None
        ),
        approval_code_hash=(
            str(row["approval_code_hash"]) if row["approval_code_hash"] is not None else None
        ),
        approval_attempts=int(row["approval_attempts"]),
        approval_expires_at=(
            float(row["approval_expires_at"]) if row["approval_expires_at"] is not None else None
        ),
        approved_by=str(row["approved_by"]) if row["approved_by"] is not None else None,
        approved_at=float(row["approved_at"]) if row["approved_at"] is not None else None,
        quarantine_dir=str(row["quarantine_dir"]) if row["quarantine_dir"] is not None else None,
    )


def _row_to_plan_item(row: sqlite3.Row) -> PlanItemRecord:
    return PlanItemRecord(
        plan_id=str(row["plan_id"]),
        photo_id=str(row["photo_id"]),
        decision_version=int(row["decision_version"]),
        src_paths=tuple(str(p) for p in json.loads(str(row["src_paths_json"]))),
        dst_paths=tuple(str(p) for p in json.loads(str(row["dst_paths_json"]))),
        status=str(row["status"]),
        done_at=float(row["done_at"]) if row["done_at"] is not None else None,
    )


def _row_to_burst(row: sqlite3.Row) -> BurstRecord:
    return BurstRecord(
        id=str(row["id"]),
        session_id=str(row["session_id"]),
        kind=str(row["kind"]),
        confirmed=bool(row["confirmed"]),
    )


def _row_to_burst_member(row: sqlite3.Row) -> BurstMemberRecord:
    return BurstMemberRecord(
        burst_id=str(row["burst_id"]),
        photo_id=str(row["photo_id"]),
        rank=int(row["rank"]),
        is_pick=bool(row["is_pick"]),
        ai_note=str(row["ai_note"]) if row["ai_note"] is not None else None,
    )


def _row_to_job(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        id=str(row["id"]),
        kind=str(row["kind"]),
        session_id=str(row["session_id"]),
        status=str(row["status"]),
        created_at=float(row["created_at"]),
        paused_reason=str(row["paused_reason"]) if row["paused_reason"] is not None else None,
        budget_usd=float(row["budget_usd"]),
        spent_usd=float(row["spent_usd"]),
        heartbeat_at=float(row["heartbeat_at"]) if row["heartbeat_at"] is not None else None,
    )


def _row_to_job_item(row: sqlite3.Row) -> JobItemRecord:
    return JobItemRecord(
        job_id=str(row["job_id"]),
        photo_id=str(row["photo_id"]),
        stage=str(row["stage"]),
        status=str(row["status"]),
        attempts=int(row["attempts"]),
        cache_key=str(row["cache_key"]) if row["cache_key"] is not None else None,
        last_error=str(row["last_error"]) if row["last_error"] is not None else None,
        updated_at=float(row["updated_at"]) if row["updated_at"] is not None else None,
    )


def _row_to_override(row: sqlite3.Row) -> OverrideRecord:
    return OverrideRecord(
        id=int(row["id"]),
        photo_id=str(row["photo_id"]),
        session_id=str(row["session_id"]),
        from_bucket=str(row["from_bucket"]),
        to_bucket=str(row["to_bucket"]),
        reason_tag=str(row["reason_tag"]) if row["reason_tag"] is not None else None,
        feature_snapshot=json.loads(str(row["feature_snapshot_json"])),
        actor=str(row["actor"]),
        created_at=float(row["created_at"]),
    )


def _row_to_llm_call(row: sqlite3.Row) -> LLMCallRecord:
    return LLMCallRecord(
        id=str(row["id"]),
        cache_key=str(row["cache_key"]) if row["cache_key"] is not None else None,
        role=str(row["role"]),
        provider=str(row["provider"]),
        model=str(row["model"]),
        in_tokens=int(row["in_tokens"] or 0),
        out_tokens=int(row["out_tokens"] or 0),
        cached_tokens=int(row["cached_tokens"] or 0),
        cost_usd=float(row["cost_usd"] or 0.0),
        latency_ms=int(row["latency_ms"] or 0),
        response_json=str(row["response_json"]) if row["response_json"] is not None else None,
        created_at=float(row["created_at"]),
        session_id=str(row["session_id"]) if row["session_id"] is not None else None,
        cache_hit=bool(row["cache_hit"]),
    )


def _row_to_person(row: sqlite3.Row) -> PersonRecord:
    return PersonRecord(
        id=str(row["id"]),
        name=str(row["name"]),
        is_self=bool(row["is_self"]),
        cover_content_fp=(
            str(row["cover_content_fp"]) if row["cover_content_fp"] is not None else None
        ),
        cover_face_idx=int(row["cover_face_idx"]) if row["cover_face_idx"] is not None else None,
        created_at=float(row["created_at"]),
    )


def _row_to_face_embedding(row: sqlite3.Row) -> FaceEmbeddingRecord:
    return FaceEmbeddingRecord(
        content_fp=str(row["content_fp"]),
        face_idx=int(row["face_idx"]),
        model_id=str(row["model_id"]),
        dim=int(row["dim"]),
        embedding_blob=bytes(row["embedding_blob"]),
        created_at=float(row["created_at"]),
    )


def _row_to_face_assignment(row: sqlite3.Row) -> FaceAssignmentRecord:
    return FaceAssignmentRecord(
        content_fp=str(row["content_fp"]),
        face_idx=int(row["face_idx"]),
        person_id=str(row["person_id"]) if row["person_id"] is not None else None,
        status=str(row["status"]),
        source=str(row["source"]),
        confidence=float(row["confidence"]) if row["confidence"] is not None else None,
        cluster_id=str(row["cluster_id"]) if row["cluster_id"] is not None else None,
        updated_at=float(row["updated_at"]),
    )


class LightboxRepository:
    """Repository exposing typed persistence operations over Lightbox SQLite schema."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # --- Roots & Sessions ---

    def upsert_root(self, root: RootRecord) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO roots(id, path, device_id, kind, readonly)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    path = excluded.path,
                    device_id = excluded.device_id,
                    kind = excluded.kind,
                    readonly = excluded.readonly
                """,
                (root.id, root.path, root.device_id, root.kind, int(root.readonly)),
            )

    def get_root(self, root_id: str) -> RootRecord | None:
        row = self.db.conn.execute("SELECT * FROM roots WHERE id = ?", (root_id,)).fetchone()
        return _row_to_root(row) if row is not None else None

    def upsert_session(self, session: SessionRecord) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO sessions(
                    id, root_id, root_path, name, sensitivity, engine_version,
                    created_at, last_scanned_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    sensitivity = excluded.sensitivity,
                    engine_version = excluded.engine_version,
                    last_scanned_at = excluded.last_scanned_at
                """,
                (
                    session.id,
                    session.root_id,
                    session.root_path,
                    session.name,
                    session.sensitivity,
                    session.engine_version,
                    session.created_at,
                    session.last_scanned_at,
                ),
            )

    def get_session(self, session_id: str) -> SessionRecord | None:
        row = self.db.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _row_to_session(row) if row is not None else None

    def get_session_by_root_path(self, root_path: str) -> SessionRecord | None:
        row = self.db.conn.execute(
            "SELECT * FROM sessions WHERE root_path = ?", (root_path,)
        ).fetchone()
        return _row_to_session(row) if row is not None else None

    def list_sessions(self) -> list[SessionRecord]:
        rows = self.db.conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        return [_row_to_session(row) for row in rows]

    # --- Photos, Assets & EXIF ---

    def upsert_photo(self, photo: PhotoRecord) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO photos(
                    id, session_id, stem, rel_dir, primary_content_fp,
                    capture_time, camera_serial, camera_model, lens_model,
                    cohort, preview_w, preview_h, is_protected, exempt_reason,
                    is_private, decision_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    stem = excluded.stem,
                    rel_dir = excluded.rel_dir,
                    primary_content_fp = excluded.primary_content_fp,
                    capture_time = excluded.capture_time,
                    camera_serial = excluded.camera_serial,
                    camera_model = excluded.camera_model,
                    lens_model = excluded.lens_model,
                    cohort = excluded.cohort,
                    preview_w = excluded.preview_w,
                    preview_h = excluded.preview_h,
                    is_protected = excluded.is_protected,
                    exempt_reason = excluded.exempt_reason,
                    is_private = excluded.is_private,
                    decision_version = CASE
                        WHEN photos.is_protected != excluded.is_protected
                        THEN photos.decision_version + 1
                        ELSE photos.decision_version
                    END
                """,
                (
                    photo.id,
                    photo.session_id,
                    photo.stem,
                    photo.rel_dir,
                    photo.primary_content_fp,
                    photo.capture_time,
                    photo.camera_serial,
                    photo.camera_model,
                    photo.lens_model,
                    photo.cohort,
                    photo.preview_w,
                    photo.preview_h,
                    int(photo.is_protected),
                    photo.exempt_reason,
                    int(photo.is_private),
                    photo.decision_version,
                ),
            )

    def get_photo(self, photo_id: str) -> PhotoRecord | None:
        row = self.db.conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
        return _row_to_photo(row) if row is not None else None

    def list_photos(self, session_id: str) -> list[PhotoRecord]:
        rows = self.db.conn.execute(
            "SELECT * FROM photos WHERE session_id = ? ORDER BY rel_dir, stem, id",
            (session_id,),
        ).fetchall()
        return [_row_to_photo(row) for row in rows]

    def get_photo_decision_versions(self, photo_ids: Sequence[str]) -> dict[str, int]:
        if not photo_ids:
            return {}
        placeholders = ",".join("?" for _ in photo_ids)
        rows = self.db.conn.execute(
            f"SELECT id, decision_version FROM photos WHERE id IN ({placeholders})",
            tuple(photo_ids),
        ).fetchall()
        return {str(row["id"]): int(row["decision_version"]) for row in rows}

    def upsert_asset(self, asset: AssetRecord) -> int:
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO assets(
                    photo_id, rel_path, abs_path, kind, size_bytes, mtime, content_fp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(abs_path) DO UPDATE SET
                    photo_id = excluded.photo_id,
                    rel_path = excluded.rel_path,
                    kind = excluded.kind,
                    size_bytes = excluded.size_bytes,
                    mtime = excluded.mtime,
                    content_fp = excluded.content_fp
                """,
                (
                    asset.photo_id,
                    asset.rel_path,
                    asset.abs_path,
                    asset.kind,
                    asset.size_bytes,
                    asset.mtime,
                    asset.content_fp,
                ),
            )
            row = cur.execute(
                "SELECT id FROM assets WHERE abs_path = ?", (asset.abs_path,)
            ).fetchone()
            assert row is not None
            return int(row["id"])

    def list_assets_for_photo(self, photo_id: str) -> list[AssetRecord]:
        rows = self.db.conn.execute(
            "SELECT * FROM assets WHERE photo_id = ? ORDER BY rel_path, id",
            (photo_id,),
        ).fetchall()
        return [_row_to_asset(row) for row in rows]

    def upsert_exif(self, exif: ExifRecord) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO exif_data(
                    photo_id, aperture, shutter_s, iso, focal_eq, ev_comp,
                    ev100, light_bucket, exposure_program, af_mode, flash_fired,
                    capture_local_hour, drive_mode, bracket_group, rating,
                    protected, gps_lat, gps_lon, field_confidence_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(photo_id) DO UPDATE SET
                    aperture = excluded.aperture,
                    shutter_s = excluded.shutter_s,
                    iso = excluded.iso,
                    focal_eq = excluded.focal_eq,
                    ev_comp = excluded.ev_comp,
                    ev100 = excluded.ev100,
                    light_bucket = excluded.light_bucket,
                    exposure_program = excluded.exposure_program,
                    af_mode = excluded.af_mode,
                    flash_fired = excluded.flash_fired,
                    capture_local_hour = excluded.capture_local_hour,
                    drive_mode = excluded.drive_mode,
                    bracket_group = excluded.bracket_group,
                    rating = excluded.rating,
                    protected = excluded.protected,
                    gps_lat = excluded.gps_lat,
                    gps_lon = excluded.gps_lon,
                    field_confidence_json = excluded.field_confidence_json
                """,
                (
                    exif.photo_id,
                    exif.aperture,
                    exif.shutter_s,
                    exif.iso,
                    exif.focal_eq,
                    exif.ev_comp,
                    exif.ev100,
                    exif.light_bucket,
                    exif.exposure_program,
                    exif.af_mode,
                    int(exif.flash_fired) if exif.flash_fired is not None else None,
                    exif.capture_local_hour,
                    exif.drive_mode,
                    exif.bracket_group,
                    exif.rating,
                    int(exif.protected) if exif.protected is not None else None,
                    exif.gps_lat,
                    exif.gps_lon,
                    json.dumps(exif.field_confidence, ensure_ascii=False, sort_keys=True),
                ),
            )

    def get_exif(self, photo_id: str) -> ExifRecord | None:
        row = self.db.conn.execute(
            "SELECT * FROM exif_data WHERE photo_id = ?", (photo_id,)
        ).fetchone()
        if row is None:
            return None
        conf_raw = (
            json.loads(str(row["field_confidence_json"]))
            if row["field_confidence_json"] is not None
            else {}
        )
        return ExifRecord(
            photo_id=str(row["photo_id"]),
            aperture=float(row["aperture"]) if row["aperture"] is not None else None,
            shutter_s=float(row["shutter_s"]) if row["shutter_s"] is not None else None,
            iso=int(row["iso"]) if row["iso"] is not None else None,
            focal_eq=float(row["focal_eq"]) if row["focal_eq"] is not None else None,
            ev_comp=float(row["ev_comp"]) if row["ev_comp"] is not None else None,
            ev100=float(row["ev100"]) if row["ev100"] is not None else None,
            light_bucket=str(row["light_bucket"]) if row["light_bucket"] is not None else None,
            exposure_program=(
                str(row["exposure_program"]) if row["exposure_program"] is not None else None
            ),
            af_mode=str(row["af_mode"]) if row["af_mode"] is not None else None,
            flash_fired=bool(row["flash_fired"]) if row["flash_fired"] is not None else None,
            capture_local_hour=(
                int(row["capture_local_hour"]) if row["capture_local_hour"] is not None else None
            ),
            drive_mode=str(row["drive_mode"]) if row["drive_mode"] is not None else None,
            bracket_group=str(row["bracket_group"]) if row["bracket_group"] is not None else None,
            rating=int(row["rating"]) if row["rating"] is not None else None,
            protected=bool(row["protected"]) if row["protected"] is not None else None,
            gps_lat=float(row["gps_lat"]) if row["gps_lat"] is not None else None,
            gps_lon=float(row["gps_lon"]) if row["gps_lon"] is not None else None,
            field_confidence={str(k): str(v) for k, v in conf_raw.items()},
        )

    # --- Metrics & Faces ---

    def upsert_metrics(self, metrics: MetricsRecord) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO metrics(
                    content_fp, engine_version, sharp_p95, sharp_z, noise_sigma,
                    motion_aniso, motion_state, exposure_json, texture,
                    face_count, face_sharp_min, scene_tag
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(content_fp, engine_version) DO UPDATE SET
                    sharp_p95 = excluded.sharp_p95,
                    sharp_z = excluded.sharp_z,
                    noise_sigma = excluded.noise_sigma,
                    motion_aniso = excluded.motion_aniso,
                    motion_state = excluded.motion_state,
                    exposure_json = excluded.exposure_json,
                    texture = excluded.texture,
                    face_count = excluded.face_count,
                    face_sharp_min = excluded.face_sharp_min,
                    scene_tag = excluded.scene_tag
                """,
                (
                    metrics.content_fp,
                    metrics.engine_version,
                    metrics.sharp_p95,
                    metrics.sharp_z,
                    metrics.noise_sigma,
                    metrics.motion_aniso,
                    metrics.motion_state,
                    json.dumps(metrics.exposure, ensure_ascii=False)
                    if metrics.exposure is not None
                    else None,
                    metrics.texture,
                    metrics.face_count,
                    metrics.face_sharp_min,
                    metrics.scene_tag,
                ),
            )

    def get_metrics(self, content_fp: str, engine_version: str) -> MetricsRecord | None:
        row = self.db.conn.execute(
            "SELECT * FROM metrics WHERE content_fp = ? AND engine_version = ?",
            (content_fp, engine_version),
        ).fetchone()
        if row is None:
            return None
        return MetricsRecord(
            content_fp=str(row["content_fp"]),
            engine_version=str(row["engine_version"]),
            sharp_p95=float(row["sharp_p95"]) if row["sharp_p95"] is not None else None,
            sharp_z=float(row["sharp_z"]) if row["sharp_z"] is not None else None,
            noise_sigma=float(row["noise_sigma"]) if row["noise_sigma"] is not None else None,
            motion_aniso=float(row["motion_aniso"]) if row["motion_aniso"] is not None else None,
            motion_state=str(row["motion_state"]) if row["motion_state"] is not None else None,
            exposure=(
                json.loads(str(row["exposure_json"])) if row["exposure_json"] is not None else None
            ),
            texture=float(row["texture"]) if row["texture"] is not None else None,
            face_count=int(row["face_count"]) if row["face_count"] is not None else None,
            face_sharp_min=(
                float(row["face_sharp_min"]) if row["face_sharp_min"] is not None else None
            ),
            scene_tag=str(row["scene_tag"]) if row["scene_tag"] is not None else None,
        )

    def replace_faces(self, content_fp: str, faces: Sequence[FaceRecord]) -> None:
        with self.db.transaction() as cur:
            cur.execute("DELETE FROM faces WHERE content_fp = ?", (content_fp,))
            for face in faces:
                cur.execute(
                    """
                    INSERT INTO faces(
                        content_fp, idx, bbox_json, landmarks_json,
                        det_score, sharpness, area_ratio
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        content_fp,
                        face.idx,
                        json.dumps(list(face.bbox)),
                        json.dumps([list(pt) for pt in face.landmarks])
                        if face.landmarks is not None
                        else None,
                        face.det_score,
                        face.sharpness,
                        face.area_ratio,
                    ),
                )

    def list_faces(self, content_fp: str) -> list[FaceRecord]:
        rows = self.db.conn.execute(
            "SELECT * FROM faces WHERE content_fp = ? ORDER BY idx ASC",
            (content_fp,),
        ).fetchall()
        result: list[FaceRecord] = []
        for row in rows:
            bbox_raw = json.loads(str(row["bbox_json"]))
            lm_raw = (
                json.loads(str(row["landmarks_json"]))
                if row["landmarks_json"] is not None
                else None
            )
            landmarks = (
                tuple((float(pt[0]), float(pt[1])) for pt in lm_raw)
                if lm_raw is not None
                else None
            )
            result.append(
                FaceRecord(
                    content_fp=str(row["content_fp"]),
                    idx=int(row["idx"]),
                    bbox=(
                        float(bbox_raw[0]),
                        float(bbox_raw[1]),
                        float(bbox_raw[2]),
                        float(bbox_raw[3]),
                    ),
                    landmarks=landmarks,
                    det_score=float(row["det_score"]) if row["det_score"] is not None else None,
                    sharpness=float(row["sharpness"]) if row["sharpness"] is not None else None,
                    area_ratio=float(row["area_ratio"]) if row["area_ratio"] is not None else None,
                )
            )
        return result

    # --- Append-Only Verdicts & Effective Verdict View ---

    def add_verdict(
        self,
        *,
        photo_id: str,
        source: str,
        bucket: str,
        confidence: float,
        reason_codes: Sequence[str],
        detail: dict[str, Any] | None = None,
        llm_call_id: str | None = None,
        created_at: float | None = None,
        increment_decision_version: bool | None = None,
    ) -> VerdictRecord:
        """Append a verdict row and atomically bump `photos.decision_version` on overrides."""
        ts = created_at if created_at is not None else time.time()
        should_bump = (
            increment_decision_version
            if increment_decision_version is not None
            else (source != "engine")
        )
        reasons_json = json.dumps(list(reason_codes), ensure_ascii=False)
        detail_json = json.dumps(detail, ensure_ascii=False) if detail is not None else None

        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO verdicts(
                    photo_id, source, bucket, confidence,
                    reason_codes_json, detail_json, llm_call_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    photo_id,
                    source,
                    bucket,
                    confidence,
                    reasons_json,
                    detail_json,
                    llm_call_id,
                    ts,
                ),
            )
            verdict_id = int(cur.lastrowid or 0)
            if should_bump:
                cur.execute(
                    "UPDATE photos SET decision_version = decision_version + 1 WHERE id = ?",
                    (photo_id,),
                )

        return VerdictRecord(
            id=verdict_id,
            photo_id=photo_id,
            source=source,
            bucket=bucket,
            confidence=confidence,
            reason_codes=tuple(reason_codes),
            detail=detail,
            llm_call_id=llm_call_id,
            created_at=ts,
        )

    def get_effective_verdict(self, photo_id: str) -> VerdictRecord | None:
        row = self.db.conn.execute(
            "SELECT * FROM effective_verdicts WHERE photo_id = ?", (photo_id,)
        ).fetchone()
        return _row_to_verdict(row) if row is not None else None

    def list_effective_verdicts(self, session_id: str) -> dict[str, VerdictRecord]:
        rows = self.db.conn.execute(
            """
            SELECT ev.*
            FROM effective_verdicts ev
            JOIN photos p ON p.id = ev.photo_id
            WHERE p.session_id = ?
            """,
            (session_id,),
        ).fetchall()
        return {str(row["photo_id"]): _row_to_verdict(row) for row in rows}

    def list_verdict_history(self, photo_id: str) -> list[VerdictRecord]:
        rows = self.db.conn.execute(
            "SELECT * FROM verdicts WHERE photo_id = ? ORDER BY id ASC",
            (photo_id,),
        ).fetchall()
        return [_row_to_verdict(row) for row in rows]

    # --- Bursts ---

    def upsert_burst(
        self, burst_id: str, session_id: str, kind: str = "burst", confirmed: bool = False
    ) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO bursts(id, session_id, kind, confirmed)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind = excluded.kind,
                    confirmed = excluded.confirmed
                """,
                (burst_id, session_id, kind, int(confirmed)),
            )

    def upsert_burst_member(
        self,
        burst_id: str,
        photo_id: str,
        rank: int,
        is_pick: bool = False,
        ai_note: str | None = None,
    ) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO burst_members(burst_id, photo_id, rank, is_pick, ai_note)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(burst_id, photo_id) DO UPDATE SET
                    rank = excluded.rank,
                    is_pick = excluded.is_pick,
                    ai_note = excluded.ai_note
                """,
                (burst_id, photo_id, rank, int(is_pick), ai_note),
            )

    def confirm_burst(self, burst_id: str, pick_photo_id: str) -> None:
        """Mark burst confirmed with chosen pick and atomically bump member decision_versions."""
        with self.db.transaction() as cur:
            cur.execute("UPDATE bursts SET confirmed = 1 WHERE id = ?", (burst_id,))
            cur.execute(
                "UPDATE burst_members SET is_pick = CASE WHEN photo_id = ? THEN 1 ELSE 0 END WHERE burst_id = ?",
                (pick_photo_id, burst_id),
            )
            cur.execute(
                """
                UPDATE photos
                SET decision_version = decision_version + 1
                WHERE id IN (SELECT photo_id FROM burst_members WHERE burst_id = ?)
                """,
                (burst_id,),
            )

    def get_burst(self, burst_id: str) -> BurstRecord | None:
        row = self.db.conn.execute("SELECT * FROM bursts WHERE id = ?", (burst_id,)).fetchone()
        return _row_to_burst(row) if row is not None else None

    def list_bursts(self, session_id: str) -> list[BurstRecord]:
        rows = self.db.conn.execute(
            "SELECT * FROM bursts WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [_row_to_burst(row) for row in rows]

    def list_burst_members(self, burst_id: str) -> list[BurstMemberRecord]:
        rows = self.db.conn.execute(
            "SELECT * FROM burst_members WHERE burst_id = ? ORDER BY rank ASC, photo_id ASC",
            (burst_id,),
        ).fetchall()
        return [_row_to_burst_member(row) for row in rows]

    def clear_unconfirmed_bursts(self, session_id: str) -> None:
        """Remove unconfirmed burst drafts before re-clustering on rescan."""
        with self.db.transaction() as cur:
            cur.execute(
                """
                DELETE FROM burst_members
                WHERE burst_id IN (
                    SELECT id FROM bursts WHERE session_id = ? AND confirmed = 0
                )
                """,
                (session_id,),
            )
            cur.execute(
                "DELETE FROM bursts WHERE session_id = ? AND confirmed = 0",
                (session_id,),
            )

    def list_unconfirmed_burst_non_pick_photo_ids(self, session_id: str) -> set[str]:
        """Return photo IDs that are non-pick members of any unconfirmed burst in `session_id`."""
        rows = self.db.conn.execute(
            """
            SELECT bm.photo_id
            FROM burst_members bm
            JOIN bursts b ON b.id = bm.burst_id
            WHERE b.session_id = ? AND b.confirmed = 0 AND bm.is_pick = 0
            """,
            (session_id,),
        ).fetchall()
        return {str(row["photo_id"]) for row in rows}

    # --- Jobs & Job Items ---

    def create_job(self, job: JobRecord) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO jobs(
                    id, kind, session_id, status, paused_reason,
                    budget_usd, spent_usd, heartbeat_at, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.kind,
                    job.session_id,
                    job.status,
                    job.paused_reason,
                    job.budget_usd,
                    job.spent_usd,
                    job.heartbeat_at,
                    job.created_at,
                ),
            )

    def get_job(self, job_id: str) -> JobRecord | None:
        row = self.db.conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row is not None else None

    def list_jobs(self, session_id: str | None = None) -> list[JobRecord]:
        if session_id is None:
            rows = self.db.conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        else:
            rows = self.db.conn.execute(
                "SELECT * FROM jobs WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,),
            ).fetchall()
        return [_row_to_job(row) for row in rows]

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        paused_reason: str | None = None,
        clear_paused_reason: bool = False,
        spent_usd: float | None = None,
        heartbeat_at: float | None = None,
    ) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                """
                UPDATE jobs
                SET status = COALESCE(?, status),
                    paused_reason = CASE
                        WHEN ? = 1 THEN NULL
                        ELSE COALESCE(?, paused_reason)
                    END,
                    spent_usd = COALESCE(?, spent_usd),
                    heartbeat_at = COALESCE(?, heartbeat_at)
                WHERE id = ?
                """,
                (
                    status,
                    1 if clear_paused_reason else 0,
                    paused_reason,
                    spent_usd,
                    heartbeat_at,
                    job_id,
                ),
            )

    def upsert_job_item(self, item: JobItemRecord) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO job_items(
                    job_id, photo_id, stage, status, attempts,
                    cache_key, last_error, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id, photo_id, stage) DO UPDATE SET
                    status = excluded.status,
                    attempts = excluded.attempts,
                    cache_key = COALESCE(excluded.cache_key, job_items.cache_key),
                    last_error = excluded.last_error,
                    updated_at = excluded.updated_at
                """,
                (
                    item.job_id,
                    item.photo_id,
                    item.stage,
                    item.status,
                    item.attempts,
                    item.cache_key,
                    item.last_error,
                    item.updated_at if item.updated_at is not None else time.time(),
                ),
            )

    def get_job_item(self, job_id: str, photo_id: str, stage: str) -> JobItemRecord | None:
        row = self.db.conn.execute(
            "SELECT * FROM job_items WHERE job_id = ? AND photo_id = ? AND stage = ?",
            (job_id, photo_id, stage),
        ).fetchone()
        return _row_to_job_item(row) if row is not None else None

    def list_job_items(
        self,
        job_id: str,
        *,
        stage: str | None = None,
    ) -> list[JobItemRecord]:
        if stage is None:
            rows = self.db.conn.execute(
                "SELECT * FROM job_items WHERE job_id = ? ORDER BY stage, photo_id",
                (job_id,),
            ).fetchall()
        else:
            rows = self.db.conn.execute(
                "SELECT * FROM job_items WHERE job_id = ? AND stage = ? ORDER BY photo_id",
                (job_id, stage),
            ).fetchall()
        return [_row_to_job_item(row) for row in rows]

    # --- Overrides ---

    def add_override(
        self,
        *,
        photo_id: str,
        session_id: str,
        from_bucket: str,
        to_bucket: str,
        feature_snapshot: dict[str, Any],
        actor: str = "user",
        reason_tag: str | None = None,
        created_at: float | None = None,
    ) -> OverrideRecord:
        ts = created_at if created_at is not None else time.time()
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO overrides(
                    photo_id, session_id, from_bucket, to_bucket,
                    reason_tag, feature_snapshot_json, actor, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    photo_id,
                    session_id,
                    from_bucket,
                    to_bucket,
                    reason_tag,
                    json.dumps(feature_snapshot, ensure_ascii=False, sort_keys=True),
                    actor,
                    ts,
                ),
            )
            oid = int(cur.lastrowid or 0)
        return OverrideRecord(
            id=oid,
            photo_id=photo_id,
            session_id=session_id,
            from_bucket=from_bucket,
            to_bucket=to_bucket,
            reason_tag=reason_tag,
            feature_snapshot=feature_snapshot,
            actor=actor,
            created_at=ts,
        )

    def list_overrides(self, session_id: str | None = None) -> list[OverrideRecord]:
        if session_id is None:
            rows = self.db.conn.execute("SELECT * FROM overrides ORDER BY id ASC").fetchall()
        else:
            rows = self.db.conn.execute(
                "SELECT * FROM overrides WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [_row_to_override(row) for row in rows]

    # --- Plans & Plan Items ---

    def create_plan(self, plan: PlanRecord, items: Sequence[PlanItemRecord]) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO plans(
                    id, session_id, action_type, plan_hash, status,
                    verifier_report_json, approval_code_hash, approval_attempts,
                    approval_expires_at, approved_by, approved_at, quarantine_dir
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan.id,
                    plan.session_id,
                    plan.action_type,
                    plan.plan_hash,
                    plan.status,
                    plan.verifier_report_json,
                    plan.approval_code_hash,
                    plan.approval_attempts,
                    plan.approval_expires_at,
                    plan.approved_by,
                    plan.approved_at,
                    plan.quarantine_dir,
                ),
            )
            for item in items:
                cur.execute(
                    """
                    INSERT INTO plan_items(
                        plan_id, photo_id, decision_version,
                        src_paths_json, dst_paths_json, status, done_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.plan_id,
                        item.photo_id,
                        item.decision_version,
                        json.dumps(list(item.src_paths), ensure_ascii=False),
                        json.dumps(list(item.dst_paths), ensure_ascii=False),
                        item.status,
                        item.done_at,
                    ),
                )

    def get_plan(self, plan_id: str) -> PlanRecord | None:
        row = self.db.conn.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
        return _row_to_plan(row) if row is not None else None

    def list_plans(self, session_id: str | None = None) -> list[PlanRecord]:
        if session_id is None:
            rows = self.db.conn.execute("SELECT * FROM plans ORDER BY rowid DESC").fetchall()
        else:
            rows = self.db.conn.execute(
                "SELECT * FROM plans WHERE session_id = ? ORDER BY rowid DESC",
                (session_id,),
            ).fetchall()
        return [_row_to_plan(row) for row in rows]

    def update_plan_status(
        self,
        plan_id: str,
        status: str,
        *,
        approved_by: str | None = None,
        approved_at: float | None = None,
        quarantine_dir: str | None = None,
    ) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                """
                UPDATE plans
                SET status = ?,
                    approved_by = COALESCE(?, approved_by),
                    approved_at = COALESCE(?, approved_at),
                    quarantine_dir = COALESCE(?, quarantine_dir)
                WHERE id = ?
                """,
                (status, approved_by, approved_at, quarantine_dir, plan_id),
            )

    def list_plan_items(self, plan_id: str) -> list[PlanItemRecord]:
        rows = self.db.conn.execute(
            "SELECT * FROM plan_items WHERE plan_id = ? ORDER BY photo_id",
            (plan_id,),
        ).fetchall()
        return [_row_to_plan_item(row) for row in rows]

    def update_plan_item(
        self,
        plan_id: str,
        photo_id: str,
        *,
        status: str,
        dst_paths: Sequence[str] | None = None,
        done_at: float | None = None,
    ) -> None:
        with self.db.transaction() as cur:
            if dst_paths is not None:
                cur.execute(
                    """
                    UPDATE plan_items
                    SET status = ?, dst_paths_json = ?, done_at = ?
                    WHERE plan_id = ? AND photo_id = ?
                    """,
                    (
                        status,
                        json.dumps(list(dst_paths), ensure_ascii=False),
                        done_at,
                        plan_id,
                        photo_id,
                    ),
                )
            else:
                cur.execute(
                    """
                    UPDATE plan_items
                    SET status = ?, done_at = ?
                    WHERE plan_id = ? AND photo_id = ?
                    """,
                    (status, done_at, plan_id, photo_id),
                )

    def list_quarantined_photo_ids(self, session_id: str) -> set[str]:
        """Return photo IDs already moved by a quarantine plan (`plan_items.status = 'done'`)."""
        rows = self.db.conn.execute(
            """
            SELECT DISTINCT pi.photo_id
            FROM plan_items pi
            JOIN plans p ON p.id = pi.plan_id
            WHERE p.session_id = ?
              AND p.action_type = 'quarantine_move'
              AND pi.status = 'done'
            """,
            (session_id,),
        ).fetchall()
        return {str(row["photo_id"]) for row in rows}

    def list_applying_plan_photo_ids(self, session_id: str) -> set[str]:
        """Return photo IDs belonging to an in-flight `applying` quarantine plan."""
        rows = self.db.conn.execute(
            """
            SELECT DISTINCT pi.photo_id
            FROM plan_items pi
            JOIN plans p ON p.id = pi.plan_id
            WHERE p.session_id = ?
              AND p.action_type = 'quarantine_move'
              AND p.status = 'applying'
              AND pi.status != 'reverted'
            """,
            (session_id,),
        ).fetchall()
        return {str(row["photo_id"]) for row in rows}

    def supersede_unapplied_overlapping_plans(
        self,
        session_id: str,
        photo_ids: Sequence[str],
        *,
        exclude_plan_id: str | None = None,
    ) -> list[str]:
        """Mark older `draft`/`reviewed`/`approved` plans overlapping `photo_ids` as `stale`."""
        if not photo_ids:
            return []
        placeholders = ",".join("?" for _ in photo_ids)
        params: list[Any] = [session_id, *photo_ids]
        exclude_clause = ""
        if exclude_plan_id is not None:
            exclude_clause = "AND p.id != ?"
            params.append(exclude_plan_id)

        with self.db.transaction() as cur:
            rows = cur.execute(
                f"""
                SELECT DISTINCT p.id
                FROM plans p
                JOIN plan_items pi ON pi.plan_id = p.id
                WHERE p.session_id = ?
                  AND p.action_type = 'quarantine_move'
                  AND p.status IN ('draft', 'reviewed', 'approved')
                  AND pi.photo_id IN ({placeholders})
                  {exclude_clause}
                ORDER BY p.id ASC
                """,
                tuple(params),
            ).fetchall()
            superseded_ids = [str(r["id"]) for r in rows]
            for old_id in superseded_ids:
                cur.execute(
                    """
                    UPDATE plans
                    SET status = 'stale',
                        approval_code_hash = NULL,
                        approval_expires_at = NULL
                    WHERE id = ?
                    """,
                    (old_id,),
                )
        return superseded_ids

    def update_plan_verifier_report(self, plan_id: str, report_json: str) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                "UPDATE plans SET verifier_report_json = ? WHERE id = ?",
                (report_json, plan_id),
            )

    def set_plan_approval_code(
        self,
        plan_id: str,
        *,
        code_hash: str,
        expires_at: float,
    ) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                """
                UPDATE plans
                SET approval_code_hash = ?,
                    approval_attempts = 0,
                    approval_expires_at = ?
                WHERE id = ?
                """,
                (code_hash, expires_at, plan_id),
            )

    def record_plan_approval_failure(
        self,
        plan_id: str,
        *,
        max_attempts: int = 5,
    ) -> tuple[int, bool]:
        """Increment failed CLI code attempts; invalidate code hash once limit is reached."""
        with self.db.transaction() as cur:
            cur.execute(
                "UPDATE plans SET approval_attempts = approval_attempts + 1 WHERE id = ?",
                (plan_id,),
            )
            row = cur.execute(
                "SELECT approval_attempts FROM plans WHERE id = ?",
                (plan_id,),
            ).fetchone()
            attempts = int(row["approval_attempts"]) if row is not None else max_attempts
            locked = attempts >= max_attempts
            if locked:
                cur.execute(
                    "UPDATE plans SET approval_code_hash = NULL, approval_expires_at = NULL WHERE id = ?",
                    (plan_id,),
                )
        return attempts, locked

    def clear_plan_approval_code(self, plan_id: str) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                "UPDATE plans SET approval_code_hash = NULL, approval_expires_at = NULL WHERE id = ?",
                (plan_id,),
            )

    # --- FTS5 ---

    def upsert_session_fts(
        self,
        *,
        photo_id: str,
        session_name: str,
        rel_dir: str,
        stem: str,
        scene_tag: str = "",
    ) -> None:
        with self.db.transaction() as cur:
            cur.execute("DELETE FROM session_fts WHERE photo_id = ?", (photo_id,))
            cur.execute(
                """
                INSERT INTO session_fts(photo_id, session_name, rel_dir, stem, scene_tag)
                VALUES (?, ?, ?, ?, ?)
                """,
                (photo_id, session_name, rel_dir, stem, scene_tag),
            )

    def search_session_fts(self, query: str, *, limit: int = 20) -> list[str]:
        rows = self.db.conn.execute(
            "SELECT photo_id FROM session_fts WHERE session_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, limit),
        ).fetchall()
        return [str(row["photo_id"]) for row in rows]

    # --- Audit Logs ---

    def add_audit_log(
        self,
        *,
        surface: str,
        tool_name: str,
        hook_decision: str,
        args_redacted: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        summary: str | None = None,
        ts: float | None = None,
    ) -> int:
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO audit_logs(
                    ts, surface, tool_name, args_redacted_json,
                    hook_decision, duration_ms, summary
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts if ts is not None else time.time(),
                    surface,
                    tool_name,
                    json.dumps(args_redacted, ensure_ascii=False) if args_redacted is not None else None,
                    hook_decision,
                    duration_ms,
                    summary,
                ),
            )
            return int(cur.lastrowid or 0)

    # --- LLM Calls & Cost Ledger ---

    def get_llm_call_by_cache_key(self, cache_key: str) -> LLMCallRecord | None:
        row = self.db.conn.execute(
            "SELECT * FROM llm_calls WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        return _row_to_llm_call(row) if row is not None else None

    def add_llm_call(self, record: LLMCallRecord) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO llm_calls(
                    id, cache_key, role, provider, model,
                    in_tokens, out_tokens, cached_tokens, cost_usd,
                    latency_ms, response_json, created_at, session_id, cache_hit
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.cache_key,
                    record.role,
                    record.provider,
                    record.model,
                    record.in_tokens,
                    record.out_tokens,
                    record.cached_tokens,
                    record.cost_usd,
                    record.latency_ms,
                    record.response_json,
                    record.created_at,
                    record.session_id,
                    int(record.cache_hit),
                ),
            )

    def sum_llm_cost(
        self,
        *,
        session_id: str | None = None,
        since_ts: float | None = None,
    ) -> float:
        clauses: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if since_ts is not None:
            clauses.append("created_at >= ?")
            params.append(since_ts)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self.db.conn.execute(
            f"SELECT COALESCE(SUM(cost_usd), 0.0) AS total FROM llm_calls {where}",
            tuple(params),
        ).fetchone()
        return float(row["total"]) if row is not None else 0.0

    def list_llm_calls(
        self,
        *,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[LLMCallRecord]:
        clamped = max(1, min(limit, 1000))
        if session_id is None:
            rows = self.db.conn.execute(
                "SELECT * FROM llm_calls ORDER BY created_at DESC LIMIT ?",
                (clamped,),
            ).fetchall()
        else:
            rows = self.db.conn.execute(
                "SELECT * FROM llm_calls WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, clamped),
            ).fetchall()
        return [_row_to_llm_call(row) for row in rows]

    # --- People & Face Classification (T18) ---

    def upsert_person(self, person: PersonRecord) -> None:
        with self.db.transaction() as cur:
            cur.execute(
                """
                INSERT INTO people(
                    id, name, is_self, cover_content_fp, cover_face_idx, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    is_self = excluded.is_self,
                    cover_content_fp = COALESCE(excluded.cover_content_fp, people.cover_content_fp),
                    cover_face_idx = COALESCE(excluded.cover_face_idx, people.cover_face_idx)
                """,
                (
                    person.id,
                    person.name,
                    int(person.is_self),
                    person.cover_content_fp,
                    person.cover_face_idx,
                    person.created_at or time.time(),
                ),
            )

    def get_person(self, person_id: str) -> PersonRecord | None:
        row = self.db.conn.execute(
            "SELECT * FROM people WHERE id = ?", (person_id,)
        ).fetchone()
        return _row_to_person(row) if row is not None else None

    def get_person_by_name(self, name: str) -> PersonRecord | None:
        row = self.db.conn.execute(
            "SELECT * FROM people WHERE name = ?", (name,)
        ).fetchone()
        return _row_to_person(row) if row is not None else None

    def list_people(self) -> list[PersonRecord]:
        rows = self.db.conn.execute(
            "SELECT * FROM people ORDER BY is_self DESC, name COLLATE NOCASE ASC, id ASC"
        ).fetchall()
        return [_row_to_person(row) for row in rows]

    def delete_person(self, person_id: str) -> None:
        with self.db.transaction() as cur:
            cur.execute("DELETE FROM people WHERE id = ?", (person_id,))

    def upsert_face_embeddings(self, records: Sequence[FaceEmbeddingRecord]) -> None:
        if not records:
            return
        now = time.time()
        with self.db.transaction() as cur:
            for rec in records:
                cur.execute(
                    """
                    INSERT INTO face_embeddings(
                        content_fp, face_idx, model_id, dim, embedding_blob, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(content_fp, face_idx, model_id) DO UPDATE SET
                        dim = excluded.dim,
                        embedding_blob = excluded.embedding_blob,
                        created_at = excluded.created_at
                    """,
                    (
                        rec.content_fp,
                        rec.face_idx,
                        rec.model_id,
                        rec.dim,
                        rec.embedding_blob,
                        rec.created_at or now,
                    ),
                )

    def list_face_embeddings(
        self,
        model_id: str,
        *,
        content_fps: Sequence[str] | None = None,
    ) -> list[FaceEmbeddingRecord]:
        if content_fps is None:
            rows = self.db.conn.execute(
                """
                SELECT * FROM face_embeddings
                WHERE model_id = ?
                ORDER BY content_fp ASC, face_idx ASC
                """,
                (model_id,),
            ).fetchall()
            return [_row_to_face_embedding(row) for row in rows]

        unique_fps = sorted(set(content_fps))
        if not unique_fps:
            return []
        result: list[FaceEmbeddingRecord] = []
        for start in range(0, len(unique_fps), 500):
            chunk = unique_fps[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.db.conn.execute(
                f"""
                SELECT * FROM face_embeddings
                WHERE model_id = ? AND content_fp IN ({placeholders})
                ORDER BY content_fp ASC, face_idx ASC
                """,
                (model_id, *chunk),
            ).fetchall()
            result.extend(_row_to_face_embedding(row) for row in rows)
        return result

    def upsert_face_assignments(self, records: Sequence[FaceAssignmentRecord]) -> None:
        if not records:
            return
        now = time.time()
        with self.db.transaction() as cur:
            for rec in records:
                cur.execute(
                    """
                    INSERT INTO face_assignments(
                        content_fp, face_idx, person_id, status, source,
                        confidence, cluster_id, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(content_fp, face_idx) DO UPDATE SET
                        person_id = excluded.person_id,
                        status = excluded.status,
                        source = excluded.source,
                        confidence = excluded.confidence,
                        cluster_id = excluded.cluster_id,
                        updated_at = excluded.updated_at
                    """,
                    (
                        rec.content_fp,
                        rec.face_idx,
                        rec.person_id,
                        rec.status,
                        rec.source,
                        rec.confidence,
                        rec.cluster_id,
                        rec.updated_at or now,
                    ),
                )

    def list_face_assignments(
        self,
        *,
        content_fps: Sequence[str] | None = None,
    ) -> list[FaceAssignmentRecord]:
        if content_fps is None:
            rows = self.db.conn.execute(
                "SELECT * FROM face_assignments ORDER BY content_fp ASC, face_idx ASC"
            ).fetchall()
            return [_row_to_face_assignment(row) for row in rows]

        unique_fps = sorted(set(content_fps))
        if not unique_fps:
            return []
        result: list[FaceAssignmentRecord] = []
        for start in range(0, len(unique_fps), 500):
            chunk = unique_fps[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.db.conn.execute(
                f"""
                SELECT * FROM face_assignments
                WHERE content_fp IN ({placeholders})
                ORDER BY content_fp ASC, face_idx ASC
                """,
                tuple(chunk),
            ).fetchall()
            result.extend(_row_to_face_assignment(row) for row in rows)
        return result

    def reassign_person_faces(self, source_person_id: str, target_person_id: str) -> int:
        now = time.time()
        with self.db.transaction() as cur:
            cur.execute(
                """
                UPDATE face_assignments
                SET person_id = ?, updated_at = ?
                WHERE person_id = ?
                """,
                (target_person_id, now, source_person_id),
            )
            return int(cur.rowcount or 0)

    def reassign_cluster_faces(self, source_cluster_id: str, target_cluster_id: str) -> int:
        now = time.time()
        with self.db.transaction() as cur:
            cur.execute(
                """
                UPDATE face_assignments
                SET cluster_id = ?, source = 'user', updated_at = ?
                WHERE cluster_id = ? AND status = 'unassigned'
                """,
                (target_cluster_id, now, source_cluster_id),
            )
            moved = int(cur.rowcount or 0)
            if moved > 0:
                # Also pin existing target cluster members as user-merged so future reclustering
                # preserves the merged cluster as a user constraint instead of splitting it apart.
                cur.execute(
                    """
                    UPDATE face_assignments
                    SET source = 'user', updated_at = ?
                    WHERE cluster_id = ? AND status = 'unassigned'
                    """,
                    (now, target_cluster_id),
                )
            return moved

    # --- Shooting Digest SQL Aggregations (T17, SPEC §13(2)) ---

    def _build_digest_base_cte(
        self,
        *,
        tech_reason_codes: Sequence[str],
        burst_reason_codes: Sequence[str],
        session_id: str | None = None,
        min_capture_ts: float | None = None,
        max_capture_ts: float | None = None,
    ) -> tuple[str, list[Any]]:
        tech_json = json.dumps(sorted(set(tech_reason_codes)))
        burst_json = json.dumps(sorted(set(burst_reason_codes)))

        where_clauses: list[str] = []
        where_params: list[Any] = []
        if session_id is not None:
            where_clauses.append("p.session_id = ?")
            where_params.append(session_id)
        if min_capture_ts is not None:
            where_clauses.append("p.capture_time IS NOT NULL AND p.capture_time >= ?")
            where_params.append(min_capture_ts)
        if max_capture_ts is not None:
            where_clauses.append("p.capture_time IS NOT NULL AND p.capture_time <= ?")
            where_params.append(max_capture_ts)
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        cte_sql = f"""
        WITH digest_raw AS (
            SELECT
                p.id AS photo_id,
                p.session_id AS session_id,
                COALESCE(NULLIF(TRIM(e.light_bucket), ''), 'unknown') AS light_bucket,
                CASE
                    WHEN e.capture_local_hour IS NULL
                         OR e.capture_local_hour < 0
                         OR e.capture_local_hour > 23 THEN 'unknown'
                    WHEN e.capture_local_hour BETWEEN 5 AND 8 THEN '05_08_dawn'
                    WHEN e.capture_local_hour BETWEEN 9 AND 11 THEN '09_11_morning'
                    WHEN e.capture_local_hour BETWEEN 12 AND 16 THEN '12_16_afternoon'
                    WHEN e.capture_local_hour BETWEEN 17 AND 19 THEN '17_19_evening'
                    WHEN e.capture_local_hour BETWEEN 20 AND 23 THEN '20_23_night'
                    ELSE '00_04_late_night'
                END AS capture_local_hour,
                CASE
                    WHEN e.flash_fired = 1 THEN 'fired'
                    WHEN e.flash_fired = 0 THEN 'no_flash'
                    ELSE 'unknown'
                END AS flash_fired,
                COALESCE(NULLIF(TRIM(m.scene_tag), ''), 'unknown') AS scene_tag,
                COALESCE(NULLIF(TRIM(m.motion_state), ''), 'unknown') AS motion_state,
                COALESCE(NULLIF(TRIM(p.lens_model), ''), 'unknown') AS lens_model,
                CASE
                    WHEN e.focal_eq IS NULL OR e.focal_eq <= 0 THEN 'unknown'
                    WHEN e.focal_eq < 24.0 THEN 'ultra_wide'
                    WHEN e.focal_eq <= 35.0 THEN 'wide_24_35'
                    WHEN e.focal_eq <= 55.0 THEN 'standard_36_55'
                    WHEN e.focal_eq <= 105.0 THEN 'portrait_56_105'
                    WHEN e.focal_eq <= 200.0 THEN 'tele_106_200'
                    ELSE 'super_tele_200p'
                END AS focal_bucket,
                CASE
                    WHEN e.shutter_s IS NULL OR e.shutter_s <= 0 THEN 'unknown'
                    WHEN e.shutter_s <= 0.00105 THEN 'fast_1000'
                    WHEN e.shutter_s <= 0.00405 THEN 'fast_250_1000'
                    WHEN e.shutter_s <= 0.017 THEN 'mid_60_250'
                    WHEN e.shutter_s < 1.0 THEN 'slow_handheld'
                    ELSE 'long_1s_plus'
                END AS shutter_bucket,
                CASE
                    WHEN e.shutter_s IS NULL OR e.shutter_s <= 0
                         OR e.focal_eq IS NULL OR e.focal_eq <= 0 THEN 'unknown'
                    WHEN (e.shutter_s * e.focal_eq) <= 1.02 THEN 'at_or_faster_safe'
                    WHEN (e.shutter_s * e.focal_eq) <= 2.04 THEN 'slower_up_to_1_stop'
                    ELSE 'slower_over_1_stop'
                END AS shutter_vs_safe,
                CASE
                    WHEN e.iso IS NULL OR e.iso <= 0 THEN 'unknown'
                    WHEN e.iso <= 200 THEN 'iso_le_200'
                    WHEN e.iso <= 800 THEN 'iso_201_800'
                    WHEN e.iso <= 3200 THEN 'iso_801_3200'
                    WHEN e.iso <= 6400 THEN 'iso_3201_6400'
                    ELSE 'iso_gt_6400'
                END AS iso_bucket,
                CASE
                    WHEN p.is_protected = 1
                         OR COALESCE(ev.bucket, 'normal') IN ('normal', 'exempt') THEN 'kept'
                    WHEN ev.bucket = 'yellow' THEN 'uncertain'
                    WHEN ev.bucket = 'red'
                         AND EXISTS (
                             SELECT 1 FROM json_each(COALESCE(ev.reason_codes_json, '[]')) r
                             WHERE r.value IN (SELECT value FROM json_each(?))
                         )
                    THEN 'tech_reject'
                    WHEN ev.bucket = 'red'
                         AND EXISTS (
                             SELECT 1 FROM json_each(COALESCE(ev.reason_codes_json, '[]')) r
                             WHERE r.value IN (SELECT value FROM json_each(?))
                         )
                    THEN 'burst_cull'
                    ELSE 'subjective_reject'
                END AS outcome,
                COALESCE(ev.reason_codes_json, '[]') AS reason_codes_json
            FROM photos p
            JOIN sessions s ON s.id = p.session_id
            LEFT JOIN exif_data e ON e.photo_id = p.id
            LEFT JOIN metrics m
                ON m.content_fp = p.primary_content_fp
                AND m.engine_version = s.engine_version
            LEFT JOIN effective_verdicts ev ON ev.photo_id = p.id
            {where_sql}
        ),
        digest_base AS (
            SELECT
                dr.*,
                CASE
                    WHEN dr.outcome = 'tech_reject' THEN (
                        SELECT r.value
                        FROM json_each(dr.reason_codes_json) r
                        WHERE r.value IN (SELECT value FROM json_each(?))
                        ORDER BY r.key ASC
                        LIMIT 1
                    )
                    ELSE NULL
                END AS primary_tech_reason
            FROM digest_raw dr
        )
        """
        params: list[Any] = [tech_json, burst_json, *where_params, tech_json]
        return cte_sql, params

    def query_digest_overview(
        self,
        *,
        tech_reason_codes: Sequence[str],
        burst_reason_codes: Sequence[str],
        session_id: str | None = None,
        min_capture_ts: float | None = None,
        max_capture_ts: float | None = None,
    ) -> DigestOverviewRow:
        cte_sql, params = self._build_digest_base_cte(
            tech_reason_codes=tech_reason_codes,
            burst_reason_codes=burst_reason_codes,
            session_id=session_id,
            min_capture_ts=min_capture_ts,
            max_capture_ts=max_capture_ts,
        )
        row = self.db.conn.execute(
            f"""
            {cte_sql}
            SELECT
                COUNT(*) AS total_shots,
                COALESCE(SUM(CASE WHEN outcome = 'tech_reject' THEN 1 ELSE 0 END), 0) AS tech_reject,
                COALESCE(SUM(CASE WHEN outcome = 'burst_cull' THEN 1 ELSE 0 END), 0) AS burst_cull,
                COALESCE(SUM(CASE WHEN outcome = 'subjective_reject' THEN 1 ELSE 0 END), 0) AS subjective_reject,
                COALESCE(SUM(CASE WHEN outcome = 'kept' THEN 1 ELSE 0 END), 0) AS kept,
                COALESCE(SUM(CASE WHEN outcome = 'uncertain' THEN 1 ELSE 0 END), 0) AS uncertain,
                COUNT(DISTINCT session_id) AS session_count
            FROM digest_base
            """,
            tuple(params),
        ).fetchone()
        if row is None:
            return DigestOverviewRow(0, 0, 0, 0, 0, 0, 0)
        return DigestOverviewRow(
            total_shots=int(row["total_shots"]),
            tech_reject=int(row["tech_reject"]),
            burst_cull=int(row["burst_cull"]),
            subjective_reject=int(row["subjective_reject"]),
            kept=int(row["kept"]),
            uncertain=int(row["uncertain"]),
            session_count=int(row["session_count"]),
        )

    def query_digest_tech_reasons(
        self,
        *,
        tech_reason_codes: Sequence[str],
        burst_reason_codes: Sequence[str],
        session_id: str | None = None,
        min_capture_ts: float | None = None,
        max_capture_ts: float | None = None,
    ) -> list[tuple[str, int]]:
        cte_sql, params = self._build_digest_base_cte(
            tech_reason_codes=tech_reason_codes,
            burst_reason_codes=burst_reason_codes,
            session_id=session_id,
            min_capture_ts=min_capture_ts,
            max_capture_ts=max_capture_ts,
        )
        rows = self.db.conn.execute(
            f"""
            {cte_sql}
            SELECT primary_tech_reason AS reason_code, COUNT(*) AS cnt
            FROM digest_base
            WHERE outcome = 'tech_reject' AND primary_tech_reason IS NOT NULL
            GROUP BY primary_tech_reason
            ORDER BY cnt DESC, primary_tech_reason ASC
            """,
            tuple(params),
        ).fetchall()
        return [(str(r["reason_code"]), int(r["cnt"])) for r in rows]

    def query_digest_dimension(
        self,
        dimension: str,
        *,
        tech_reason_codes: Sequence[str],
        burst_reason_codes: Sequence[str],
        session_id: str | None = None,
        min_capture_ts: float | None = None,
        max_capture_ts: float | None = None,
    ) -> list[DigestBucketRow]:
        if dimension not in _DIGEST_ALLOWED_DIMENSIONS:
            raise ValueError(f"Unsupported digest dimension: {dimension!r}")
        cte_sql, params = self._build_digest_base_cte(
            tech_reason_codes=tech_reason_codes,
            burst_reason_codes=burst_reason_codes,
            session_id=session_id,
            min_capture_ts=min_capture_ts,
            max_capture_ts=max_capture_ts,
        )
        agg_rows = self.db.conn.execute(
            f"""
            {cte_sql}
            SELECT
                {dimension} AS bucket_key,
                COUNT(*) AS total,
                SUM(CASE WHEN outcome = 'tech_reject' THEN 1 ELSE 0 END) AS tech_reject,
                SUM(CASE WHEN outcome = 'burst_cull' THEN 1 ELSE 0 END) AS burst_cull,
                SUM(CASE WHEN outcome = 'subjective_reject' THEN 1 ELSE 0 END) AS subjective_reject,
                SUM(CASE WHEN outcome = 'kept' THEN 1 ELSE 0 END) AS kept,
                SUM(CASE WHEN outcome = 'uncertain' THEN 1 ELSE 0 END) AS uncertain
            FROM digest_base
            GROUP BY {dimension}
            ORDER BY total DESC, bucket_key ASC
            """,
            tuple(params),
        ).fetchall()

        reason_rows = self.db.conn.execute(
            f"""
            {cte_sql}
            SELECT
                {dimension} AS bucket_key,
                primary_tech_reason AS reason_code,
                COUNT(*) AS cnt
            FROM digest_base
            WHERE outcome = 'tech_reject' AND primary_tech_reason IS NOT NULL
            GROUP BY {dimension}, primary_tech_reason
            ORDER BY bucket_key ASC, cnt DESC, primary_tech_reason ASC
            """,
            tuple(params),
        ).fetchall()

        top_reason_by_bucket: dict[str, tuple[str, int]] = {}
        for rr in reason_rows:
            bk = str(rr["bucket_key"])
            if bk not in top_reason_by_bucket:
                top_reason_by_bucket[bk] = (str(rr["reason_code"]), int(rr["cnt"]))

        result: list[DigestBucketRow] = []
        for row in agg_rows:
            bk = str(row["bucket_key"])
            top_r = top_reason_by_bucket.get(bk)
            result.append(
                DigestBucketRow(
                    bucket_key=bk,
                    total=int(row["total"]),
                    tech_reject=int(row["tech_reject"]),
                    burst_cull=int(row["burst_cull"]),
                    subjective_reject=int(row["subjective_reject"]),
                    kept=int(row["kept"]),
                    uncertain=int(row["uncertain"]),
                    top_reason_code=top_r[0] if top_r is not None else None,
                    top_reason_count=top_r[1] if top_r is not None else 0,
                )
            )
        return result

    def query_digest_focal_shutter_matrix(
        self,
        *,
        tech_reason_codes: Sequence[str],
        burst_reason_codes: Sequence[str],
        session_id: str | None = None,
        min_capture_ts: float | None = None,
        max_capture_ts: float | None = None,
    ) -> list[DigestMatrixRow]:
        cte_sql, params = self._build_digest_base_cte(
            tech_reason_codes=tech_reason_codes,
            burst_reason_codes=burst_reason_codes,
            session_id=session_id,
            min_capture_ts=min_capture_ts,
            max_capture_ts=max_capture_ts,
        )
        rows = self.db.conn.execute(
            f"""
            {cte_sql}
            SELECT
                focal_bucket,
                shutter_bucket,
                COUNT(*) AS total,
                SUM(CASE WHEN outcome = 'tech_reject' THEN 1 ELSE 0 END) AS tech_reject,
                SUM(CASE WHEN outcome = 'burst_cull' THEN 1 ELSE 0 END) AS burst_cull,
                SUM(CASE WHEN outcome = 'subjective_reject' THEN 1 ELSE 0 END) AS subjective_reject,
                SUM(CASE WHEN outcome = 'kept' THEN 1 ELSE 0 END) AS kept,
                SUM(CASE WHEN outcome = 'uncertain' THEN 1 ELSE 0 END) AS uncertain
            FROM digest_base
            WHERE focal_bucket != 'unknown' AND shutter_bucket != 'unknown'
            GROUP BY focal_bucket, shutter_bucket
            ORDER BY total DESC, focal_bucket ASC, shutter_bucket ASC
            """,
            tuple(params),
        ).fetchall()
        return [
            DigestMatrixRow(
                focal_bucket=str(r["focal_bucket"]),
                shutter_bucket=str(r["shutter_bucket"]),
                total=int(r["total"]),
                tech_reject=int(r["tech_reject"]),
                burst_cull=int(r["burst_cull"]),
                subjective_reject=int(r["subjective_reject"]),
                kept=int(r["kept"]),
                uncertain=int(r["uncertain"]),
            )
            for r in rows
        ]


__all__ = [
    "AssetRecord",
    "BurstMemberRecord",
    "BurstRecord",
    "DigestBucketRow",
    "DigestMatrixRow",
    "DigestOverviewRow",
    "ExifRecord",
    "FaceAssignmentRecord",
    "FaceEmbeddingRecord",
    "FaceRecord",
    "JobItemRecord",
    "JobRecord",
    "LLMCallRecord",
    "LightboxRepository",
    "MetricsRecord",
    "OverrideRecord",
    "PersonRecord",
    "PhotoRecord",
    "PlanItemRecord",
    "PlanRecord",
    "RootRecord",
    "SessionRecord",
    "VerdictRecord",
]
