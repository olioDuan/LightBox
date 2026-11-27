"""Unified ToolRegistry shared by MCP Server and future built-in Agent (SPEC §7.3, §8.2, §10.2)."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lightbox.i18n import (
    REASON_USER_REJECT,
    REASON_USER_RESCUE,
    REASON_USER_UNSURE,
    render_reasons,
)
from lightbox.services import (
    ApprovalCodeError,
    BucketName,
    EmbedderError,
    LightboxServices,
    MissingAPIKeyError,
    ModelError,
    PlanError,
    SafeFSError,
)

RiskLevel = Literal[
    "READ",
    "READ_COST",
    "COMPUTE_COST",
    "ANNOTATE",
    "WRITE_NEW",
    "MOVE",
    "MOVE_REVERT",
]

_SENSITIVE_ARG_FRAGMENTS: tuple[str, ...] = (
    "code",
    "token",
    "secret",
    "password",
    "api_key",
    "apikey",
)
_FORBIDDEN_URL_FRAGMENTS: tuple[str, ...] = (
    "token=",
    "auth/launch",
    "csrf=",
    "secret=",
    "api_key=",
)


def assert_tokenless_url(url: str) -> str:
    """Assert that a generated UI URL contains no authentication token (SPEC §8.1, §10.2(3))."""
    lowered = url.lower()
    for forbidden in _FORBIDDEN_URL_FRAGMENTS:
        if forbidden in lowered:
            raise ValueError(f"Generated URL violates tokenless invariant: {url!r}")
    return url


def redact_tool_args(arguments: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive fields before writing audit logs."""
    redacted: dict[str, Any] = {}
    for key, value in arguments.items():
        if any(frag in key.lower() for frag in _SENSITIVE_ARG_FRAGMENTS):
            redacted[key] = "***REDACTED***"
        elif isinstance(value, str) and len(value) > 240:
            redacted[key] = value[:237] + "..."
        elif isinstance(value, list) and len(value) > 20:
            redacted[key] = [*value[:20], f"...(+{len(value) - 20} more)"]
        else:
            redacted[key] = value
    return redacted


def _truncate_untrusted(value: str | None, max_len: int = 160) -> str | None:
    if value is None:
        return None
    cleaned = value.replace("\r", " ").replace("\n", " ").strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3] + "..."


def _format_meta(ctx: ToolContext, value: str | None, *, max_len: int = 160) -> str | None:
    """Truncate untrusted metadata and wrap with `<untrusted_metadata>` on the Agent surface."""
    truncated = _truncate_untrusted(value, max_len=max_len)
    if truncated is None:
        return None
    if ctx.surface.startswith("agent"):
        safe = truncated.replace("<", "&lt;").replace(">", "&gt;")
        return f"<untrusted_metadata>{safe}</untrusted_metadata>"
    return truncated


@dataclass(frozen=True, slots=True)
class ToolImagePayload:
    """Binary EXIF-stripped image payload attached to a tool result."""

    data: bytes
    mime_type: str = "image/webp"
    label: str = ""


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Normalized execution result shared across MCP Server and Agent surfaces."""

    data: dict[str, Any]
    images: tuple[ToolImagePayload, ...] = ()
    is_error: bool = False
    error_message: str | None = None
    hook_decision: Literal["ALLOW", "DENY", "ASK"] | None = None

    @classmethod
    def ok(
        cls,
        data: dict[str, Any],
        *,
        images: tuple[ToolImagePayload, ...] = (),
    ) -> ToolExecutionResult:
        return cls(
            data=data,
            images=images,
            is_error=False,
            error_message=None,
            hook_decision="ALLOW",
        )

    @classmethod
    def error(
        cls,
        message: str,
        *,
        data: dict[str, Any] | None = None,
        hook_decision: Literal["ALLOW", "DENY", "ASK"] | None = None,
    ) -> ToolExecutionResult:
        payload = dict(data or {})
        payload.setdefault("error", message)
        if hook_decision is not None:
            payload.setdefault("hook_decision", hook_decision)
        return cls(
            data=payload,
            images=(),
            is_error=True,
            error_message=message,
            hook_decision=hook_decision,
        )


@dataclass(slots=True)
class ToolContext:
    """Runtime execution context passed into every ToolRegistry handler."""

    services: LightboxServices
    surface: str = "mcp:default"
    base_url_resolver: Callable[[], str] | None = None
    progress_callback: Callable[[str, int, int], None] | None = None
    elicitation_approver: Callable[[str, int, str | None], bool] | None = None
    lang: str = "zh-CN"
    session_id: str | None = None
    agent_session_id: str | None = None
    viewed_photo_count: int = 0
    max_viewed_photos: int = 40
    saved_sets: dict[str, list[str]] = field(default_factory=dict)
    on_update_plan: Callable[[list[dict[str, Any]]], None] | None = None
    budget_ask_approved: bool = False
    session_budget_override_usd: float | None = None
    bypass_monthly_budget: bool = False

    def review_url(self, session_id: str) -> str:
        base = self._resolve_base_url()
        return assert_tokenless_url(f"{base}/sessions/{quote(session_id, safe='')}/review")

    def approve_url(self, session_id: str, plan_id: str) -> str:
        base = self._resolve_base_url()
        return assert_tokenless_url(
            f"{base}/sessions/{quote(session_id, safe='')}/review?plan_id={quote(plan_id, safe='')}"
        )

    def _resolve_base_url(self) -> str:
        if self.base_url_resolver is not None:
            return self.base_url_resolver().rstrip("/")
        return "http://127.0.0.1:8765"


# ---------------------------------------------------------------------------
# Strict Pydantic v2 Input Models (SPEC §8.2)
# ---------------------------------------------------------------------------


class ScanFolderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folder_path: str = Field(min_length=1, description="Absolute or user-expanded local photo folder path.")
    sensitivity: Literal["conservative", "standard"] = Field(
        default="conservative",
        description="Deterministic culling sensitivity preset.",
    )
    enable_embed: bool = Field(
        default=False,
        description="Enable local L2 SigLIP 2 vision embedding and scene tagging.",
    )
    wait_seconds: int = Field(
        default=15,
        ge=0,
        le=300,
        description="Max seconds to wait synchronously before returning a running job_id.",
    )


class JobStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1, description="Background scan or triage job ID.")


class SessionSummaryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, description="Scanned session ID.")


class ListPhotosInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, description="Scanned session ID.")
    bucket: Literal["red", "yellow", "normal", "exempt", "unreviewed", "kept"] | None = Field(
        default=None,
        description="Optional bucket filter.",
    )
    reason: str | None = Field(default=None, description="Optional reason code filter.")
    sort: Literal["uncertainty", "name"] = Field(
        default="uncertainty",
        description="Sort order; default ascending confidence (most uncertain first).",
    )
    limit: int = Field(default=30, ge=1, le=30, description="Page size (max 30 for compact handles).")
    cursor: str | None = Field(default=None, description="Opaque pagination cursor from previous response.")
    save_as_set: bool = Field(
        default=False,
        description="When true on agent surface, save returned photo_ids under a compact server-side set_id.",
    )


class GetPhotoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    photo_id: str = Field(min_length=1, description="Logical photo ID.")
    image: Literal["none", "thumb", "face_crop", "sharpest_crop"] = Field(
        default="none",
        description="Optional single EXIF-stripped image attachment (max 768px WebP).",
    )


class CompareBurstInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_id: str = Field(min_length=1, description="Burst group ID.")


class SetVerdictInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    photo_ids: list[str] = Field(min_length=1, max_length=100, description="Target photo IDs.")
    verdict: Literal["keep", "reject", "unsure"] = Field(
        description="Suggested verdict (never moves or deletes files).",
    )
    reason: str | None = Field(default=None, max_length=240, description="Optional human-readable rationale.")


class RecommendBurstPickInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_id: str = Field(min_length=1, description="Burst group ID.")
    photo_id: str = Field(min_length=1, description="Recommended pick photo ID in the burst.")
    reason: str = Field(min_length=1, max_length=300, description="Plain-language reason for recommending this frame.")


class TriageUncertainInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, description="Scanned session ID.")
    max_photos: int = Field(default=50, ge=1, le=500, description="Maximum uncertain photos to include.")
    dry_run: bool = Field(default=True, description="When true, estimate scope & cost without calling cloud VLM.")


class SearchPhotosInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Search text (filename, scene, camera, lens, or reason).")
    session_id: str | None = Field(default=None, description="Optional session scope.")
    date_from: str | None = Field(default=None, description="Optional ISO date lower bound (YYYY-MM-DD).")
    camera: str | None = Field(default=None, description="Optional camera model substring filter.")
    limit: int = Field(default=20, ge=1, le=20, description="Maximum matches (max 20).")


class FindSimilarInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    photo_id: str = Field(min_length=1, description="Anchor photo ID.")
    limit: int = Field(default=12, ge=1, le=12, description="Maximum similar photos (max 12).")


class CreatePlanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, description="Scanned session ID.")
    action: Literal["quarantine_move", "export_clean"] = Field(
        default="quarantine_move",
        description="Plan action type ('quarantine_move' supported; 'export_clean' activates in T16).",
    )
    include_bursts: Literal["confirmed_only", "none"] = Field(
        default="confirmed_only",
        description="Whether to include confirmed burst non-picks; unconfirmed bursts are always excluded.",
    )
    photo_ids: list[str] | None = Field(
        default=None,
        description="Optional explicit photo IDs; omitted drafts all eligible obvious rejects.",
    )


class ApplyPlanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=1, description="Execution plan ID to apply.")
    confirmation_code: str | None = Field(
        default=None,
        min_length=6,
        max_length=6,
        description="Optional 6-digit one-time code issued by `lightbox approve <plan_id>`.",
    )


class RestorePlanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(min_length=1, description="Applied quarantine plan ID to restore.")
    photo_ids: list[str] | None = Field(
        default=None,
        description="Optional subset of photo IDs to restore; omitted restores the entire batch.",
    )


class ShootingDigestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(
        default=None,
        description="Optional session ID; omitted summarizes across all scanned sessions.",
    )
    date_from: str | None = Field(
        default=None,
        description="Optional ISO start date filter (YYYY-MM-DD).",
    )
    date_to: str | None = Field(
        default=None,
        description="Optional ISO end date filter (YYYY-MM-DD).",
    )
    min_samples: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Minimum sample count required for a condition highlight or stable setup.",
    )


class ViewPhotosInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    photo_ids: list[str] = Field(
        min_length=1,
        max_length=6,
        description="Up to 6 photo IDs to inspect visually (cumulative cap 40 per agent session).",
    )
    mode: Literal["face_crop", "sharpest_crop", "thumb"] = Field(
        default="sharpest_crop",
        description="EXIF-stripped crop or thumbnail mode to attach for visual inspection.",
    )


class TodoItemInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=40)
    title: str = Field(min_length=1, max_length=160)
    status: Literal["pending", "in_progress", "done", "skipped"] = "pending"


class UpdatePlanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    todos: list[TodoItemInput] = Field(
        min_length=1,
        max_length=20,
        description="Ordered checklist of agent execution steps persisted to SQLite.",
    )


class GetMemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProposeRuleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_text: str = Field(
        min_length=4,
        max_length=240,
        description="Concise text preference rule proposed for human approval (never auto-activated).",
    )
    rationale: str | None = Field(
        default=None,
        max_length=280,
        description="Short evidence summary explaining why this rule is proposed.",
    )
    evidence_ids: list[int] = Field(
        default_factory=list,
        max_length=20,
        description="Optional override IDs supporting this proposal.",
    )


# ---------------------------------------------------------------------------
# Tool Handlers (all strictly routed through LightboxServices)
# ---------------------------------------------------------------------------


def _serialize_summary(ctx: ToolContext, session_id: str) -> dict[str, Any]:
    summary = ctx.services.review.get_session_summary(session_id)
    preview_note = (
        f"{summary.low_res_preview_count} photos have embedded previews under 2000px; verify fine focus in Web UI."
        if summary.low_res_preview_count > 0
        else "All scanned previews are full resolution (>= 2000px) or standard JPG/HEIC."
    )
    summary_dict: dict[str, Any] = {
        "session_id": summary.session.id,
        "name": _format_meta(ctx, summary.session.name),
        "sensitivity": summary.session.sensitivity,
        "total_photos": summary.total_photos,
        "active_total_photos": summary.active_total_photos,
        "counts_by_bucket": summary.counts_by_bucket,
        "active_counts_by_bucket": summary.active_counts_by_bucket,
        "actionable_reject_count": summary.actionable_reject_count,
        "top_reasons": [
            {
                "code": code,
                "count": cnt,
                "label": render_reasons([code], lang=ctx.lang)[0],
            }
            for code, cnt in summary.top_reasons
        ],
        "camera_distribution": summary.camera_distribution,
        "low_res_preview_count": summary.low_res_preview_count,
        "preview_resolution_note": preview_note,
        "burst_count": summary.burst_count,
        "unconfirmed_burst_count": summary.unconfirmed_burst_count,
        "quarantined_count": summary.quarantined_count,
        "review_url": ctx.review_url(summary.session.id),
    }
    # Never expose raw absolute host paths into the built-in Agent LLM context (OutboundPrivacyHook)
    if not ctx.surface.startswith("agent"):
        summary_dict["root_path"] = summary.session.root_path
    return summary_dict


def _handle_scan_folder(ctx: ToolContext, args: ScanFolderInput) -> ToolExecutionResult:
    folder = Path(args.folder_path).expanduser().resolve()
    if not folder.is_dir():
        return ToolExecutionResult.error(f"Scan target is not an existing directory: {folder}")

    progress_state: dict[str, Any] = {"stage": "ingest", "done": 0, "total": 0}
    result_box: dict[str, Any] = {}
    error_box: list[Exception] = []
    done_evt = threading.Event()

    def _on_progress(stage: str, done: int, total: int) -> None:
        progress_state["stage"] = stage
        progress_state["done"] = done
        progress_state["total"] = total
        if ctx.progress_callback is not None:
            ctx.progress_callback(stage, done, total)

    def _worker() -> None:
        try:
            with LightboxServices(
                app_paths=ctx.services.app_paths,
                embedder=ctx.services.embedder,
                migrate=False,
            ) as bg_svcs:
                scan_res = bg_svcs.scan.scan_folder(
                    folder,
                    sensitivity=args.sensitivity,
                    max_workers=1,
                    enable_embed=args.enable_embed,
                    on_progress=_on_progress,
                )
                result_box["scan"] = scan_res
        except Exception as exc:  # pragma: no cover - defensive capture
            error_box.append(exc)
        finally:
            done_evt.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    finished = done_evt.wait(timeout=float(args.wait_seconds))

    if error_box:
        return ToolExecutionResult.error(f"Scan failed: {error_box[0]}")

    if finished and "scan" in result_box:
        scan_res = result_box["scan"]
        summary_payload = _serialize_summary(ctx, scan_res.session_id)
        payload: dict[str, Any] = {
            "job_id": scan_res.job_id,
            "session_id": scan_res.session_id,
            "status": scan_res.status,
            "progress": {
                "stage": "done",
                "done": scan_res.total_photos,
                "total": scan_res.total_photos,
            },
            "summary": summary_payload,
            "review_url": ctx.review_url(scan_res.session_id),
        }
        if args.enable_embed:
            payload["embedded_count"] = scan_res.embedded_count
        return ToolExecutionResult.ok(payload)

    # Still running after `wait_seconds`: return job handle for polling via lightbox_job_status
    sessions = ctx.services.review.list_sessions()
    matched = next((s for s in sessions if Path(s.root_path).resolve() == folder), None)
    session_id = matched.id if matched is not None else ""
    review_url = ctx.review_url(session_id) if session_id else ctx._resolve_base_url()
    return ToolExecutionResult.ok(
        {
            "job_id": None,
            "session_id": session_id or None,
            "status": "running",
            "progress": dict(progress_state),
            "review_url": assert_tokenless_url(review_url),
            "note": (
                f"Scan is still running after wait_seconds={args.wait_seconds}; "
                "poll lightbox_job_status or check lightbox://sessions once ready."
            ),
        }
    )


def _handle_job_status(ctx: ToolContext, args: JobStatusInput) -> ToolExecutionResult:
    try:
        st = ctx.services.scan.get_job_status(args.job_id)
    except ValueError as exc:
        return ToolExecutionResult.error(str(exc))
    remaining = max(0, st.total - st.done)
    eta_s: float | None = 0.0 if st.status == "completed" else (
        round(remaining * 0.25, 1) if st.total > 0 else None
    )
    return ToolExecutionResult.ok(
        {
            "job_id": st.job_id,
            "kind": st.kind,
            "session_id": st.session_id,
            "status": st.status,
            "stage": st.stage,
            "done": st.done,
            "total": st.total,
            "eta_s": eta_s,
            "spent_usd": st.spent_usd,
            "paused_reason": st.paused_reason,
        }
    )


def _handle_session_summary(ctx: ToolContext, args: SessionSummaryInput) -> ToolExecutionResult:
    try:
        return ToolExecutionResult.ok(_serialize_summary(ctx, args.session_id))
    except ValueError as exc:
        return ToolExecutionResult.error(str(exc))


def _handle_list_photos(ctx: ToolContext, args: ListPhotosInput) -> ToolExecutionResult:
    try:
        # Validate session exists first
        ctx.services.review.get_session_summary(args.session_id)
        page, next_cursor = ctx.services.review.list_photos_page(
            args.session_id,
            bucket=args.bucket,
            reason=args.reason,
            sort=args.sort,
            limit=args.limit,
            cursor=args.cursor,
        )
    except ValueError as exc:
        return ToolExecutionResult.error(str(exc))

    items: list[dict[str, Any]] = []
    for it in page:
        rendered = render_reasons(it.reason_codes, lang=ctx.lang)
        items.append(
            {
                "photo_id": it.photo_id,
                "stem": _format_meta(ctx, it.stem),
                "rel_dir": _format_meta(ctx, it.rel_dir),
                "bucket": it.bucket,
                "source": it.source,
                "confidence": round(it.confidence, 4),
                "reason": "; ".join(rendered) if rendered else "",
                "reason_codes": list(it.reason_codes),
                "exif_brief": ctx.services.review.format_exif_brief(it.photo_id),
                "burst_id": it.burst_id,
                "decision_version": it.decision_version,
                "is_protected": it.is_protected,
            }
        )
    resp_data: dict[str, Any] = {
        "session_id": args.session_id,
        "items": items,
        "next_cursor": next_cursor,
        "review_url": ctx.review_url(args.session_id),
    }
    if args.save_as_set:
        set_id = f"set_{len(ctx.saved_sets) + 1:02d}"
        ctx.saved_sets[set_id] = [it["photo_id"] for it in items]
        resp_data["set_id"] = set_id
        resp_data["set_size"] = len(items)
    return ToolExecutionResult.ok(resp_data)


def _handle_get_photo(ctx: ToolContext, args: GetPhotoInput) -> ToolExecutionResult:
    try:
        detail = ctx.services.review.get_photo_detail(args.photo_id)
    except ValueError as exc:
        return ToolExecutionResult.error(str(exc))

    burst_id = ctx.services.burst.find_burst_for_photo(detail.photo.session_id, detail.photo.id)
    eff = detail.effective_verdict
    exif = detail.exif
    metrics = detail.metrics

    payload: dict[str, Any] = {
        "photo_id": detail.photo.id,
        "session_id": detail.photo.session_id,
        "stem": _format_meta(ctx, detail.photo.stem),
        "rel_dir": _format_meta(ctx, detail.photo.rel_dir),
        "cohort": detail.photo.cohort,
        "camera_model": _format_meta(ctx, detail.photo.camera_model),
        "lens_model": _format_meta(ctx, detail.photo.lens_model),
        "preview_w": detail.photo.preview_w,
        "preview_h": detail.photo.preview_h,
        "is_protected": detail.photo.is_protected,
        "exempt_reason": detail.photo.exempt_reason,
        "decision_version": detail.photo.decision_version,
        "burst_id": burst_id,
        "assets": [
            {"rel_path": _format_meta(ctx, a.rel_path), "kind": a.kind, "size_bytes": a.size_bytes}
            for a in detail.assets
        ],
        "exif_brief": ctx.services.review.format_exif_brief(detail.photo.id),
        "exif": (
            {
                "aperture": exif.aperture,
                "shutter_s": exif.shutter_s,
                "iso": exif.iso,
                "focal_eq": exif.focal_eq,
                "ev_comp": exif.ev_comp,
                "ev100": exif.ev100,
                "light_bucket": exif.light_bucket,
                "exposure_program": exif.exposure_program,
                "af_mode": exif.af_mode,
                "flash_fired": exif.flash_fired,
                "capture_local_hour": exif.capture_local_hour,
            }
            if exif is not None
            else None
        ),
        "metrics": (
            {
                "sharp_p95": metrics.sharp_p95,
                "sharp_z": metrics.sharp_z,
                "noise_sigma": metrics.noise_sigma,
                "motion_aniso": metrics.motion_aniso,
                "motion_state": metrics.motion_state,
                "face_count": metrics.face_count,
                "face_sharp_min": metrics.face_sharp_min,
                "scene_tag": metrics.scene_tag,
            }
            if metrics is not None
            else None
        ),
        "faces": [
            {
                "idx": f.idx,
                "sharpness": f.sharpness,
                "area_ratio": f.area_ratio,
                "det_score": f.det_score,
            }
            for f in detail.faces
        ],
        "verdict": (
            {
                "bucket": eff.bucket,
                "source": eff.source,
                "confidence": eff.confidence,
                "reason_codes": list(eff.reason_codes),
                "reasons": render_reasons(eff.reason_codes, lang=ctx.lang),
            }
            if eff is not None
            else None
        ),
        "image_mode": args.image,
    }

    images: tuple[ToolImagePayload, ...] = ()
    if args.image != "none":
        try:
            img_bytes = ctx.services.review.get_photo_mcp_image_bytes(
                args.photo_id,
                mode=args.image,
            )
            images = (
                ToolImagePayload(
                    data=img_bytes,
                    mime_type="image/webp",
                    label=f"{detail.photo.id}:{args.image}",
                ),
            )
        except (ValueError, FileNotFoundError) as exc:
            return ToolExecutionResult.error(f"Failed to load image for {args.photo_id}: {exc}")

    return ToolExecutionResult.ok(payload, images=images)


def _handle_compare_burst(ctx: ToolContext, args: CompareBurstInput) -> ToolExecutionResult:
    try:
        detail = ctx.services.burst.get_burst_detail(args.group_id)
    except ValueError as exc:
        return ToolExecutionResult.error(str(exc))

    member_ids = [m.photo_id for m in detail.members]
    frames: list[dict[str, Any]] = []
    for idx, m in enumerate(detail.members, start=1):
        pd = ctx.services.review.get_photo_detail(m.photo_id)
        frames.append(
            {
                "frame_no": idx,
                "photo_id": m.photo_id,
                "stem": _format_meta(ctx, pd.photo.stem),
                "rank": m.rank,
                "is_pick": m.is_pick,
                "ai_note": m.ai_note,
                "bucket": pd.effective_verdict.bucket if pd.effective_verdict else "normal",
                "sharp_p95": pd.metrics.sharp_p95 if pd.metrics else None,
                "face_count": pd.metrics.face_count if pd.metrics else 0,
                "face_sharp_min": pd.metrics.face_sharp_min if pd.metrics else None,
                "exif_brief": ctx.services.review.format_exif_brief(m.photo_id),
            }
        )

    images: list[ToolImagePayload] = []
    if member_ids:
        try:
            sheet_bytes, face_strip_bytes = ctx.services.review.build_burst_compare_images(
                member_ids
            )
            images.append(
                ToolImagePayload(
                    data=sheet_bytes,
                    mime_type="image/webp",
                    label=f"{args.group_id}:contact_sheet",
                )
            )
            if face_strip_bytes is not None:
                images.append(
                    ToolImagePayload(
                        data=face_strip_bytes,
                        mime_type="image/webp",
                        label=f"{args.group_id}:face_crop_strip",
                    )
                )
        except (ValueError, FileNotFoundError):
            pass

    return ToolExecutionResult.ok(
        {
            "group_id": detail.burst.id,
            "session_id": detail.burst.session_id,
            "kind": detail.burst.kind,
            "confirmed": detail.burst.confirmed,
            "has_face_strip": len(images) > 1,
            "frames": frames,
            "review_url": ctx.review_url(detail.burst.session_id),
        },
        images=tuple(images),
    )


def _handle_set_verdict(ctx: ToolContext, args: SetVerdictInput) -> ToolExecutionResult:
    bucket_map: dict[str, BucketName] = {
        "keep": "normal",
        "reject": "red",
        "unsure": "yellow",
    }
    reason_map: dict[str, str] = {
        "keep": REASON_USER_RESCUE,
        "reject": REASON_USER_REJECT,
        "unsure": REASON_USER_UNSURE,
    }
    target_bucket: BucketName = bucket_map[args.verdict]
    source_tag = ctx.surface if ctx.surface.startswith("mcp:") else "agent"

    updated: list[dict[str, Any]] = []
    latest_version = 1
    for pid in args.photo_ids:
        try:
            verdict = ctx.services.review.set_verdict(
                pid,
                bucket=target_bucket,
                source=source_tag,
                confidence=0.85,
                reason_codes=(reason_map[args.verdict],),
                reason_tag=args.reason,
                actor=source_tag,
                detail={"reason": args.reason, "suggested_verdict": args.verdict}
                if args.reason
                else {"suggested_verdict": args.verdict},
            )
            pd = ctx.services.review.get_photo_detail(pid)
        except ValueError as exc:
            return ToolExecutionResult.error(str(exc))
        latest_version = pd.photo.decision_version
        updated.append(
            {
                "photo_id": pid,
                "bucket": verdict.bucket,
                "source": verdict.source,
                "decision_version": pd.photo.decision_version,
            }
        )

    return ToolExecutionResult.ok(
        {
            "updated": updated,
            "decision_version": latest_version,
            "note": "Recorded non-destructive suggestion in SQLite only; no files were moved.",
        }
    )


def _handle_recommend_burst_pick(
    ctx: ToolContext,
    args: RecommendBurstPickInput,
) -> ToolExecutionResult:
    try:
        updated = ctx.services.burst.recommend_burst_pick(
            args.group_id,
            args.photo_id,
            note=args.reason,
        )
    except ValueError as exc:
        return ToolExecutionResult.error(str(exc))

    return ToolExecutionResult.ok(
        {
            "group_id": updated.burst.id,
            "pick_photo_id": args.photo_id,
            "reason": args.reason,
            "confirmed": updated.burst.confirmed,
            "review_url": ctx.review_url(updated.burst.session_id),
        }
    )


def _handle_triage_uncertain(
    ctx: ToolContext,
    args: TriageUncertainInput,
) -> ToolExecutionResult:
    if args.dry_run:
        try:
            est = ctx.services.triage.estimate_triage(
                args.session_id,
                max_photos=args.max_photos,
            )
        except (ValueError, ModelError) as exc:
            return ToolExecutionResult.error(str(exc))

        return ToolExecutionResult.ok(
            {
                "session_id": est.session_id,
                "dry_run": True,
                "count": est.total_units,
                "yellow_count": est.yellow_count,
                "unconfirmed_burst_count": est.burst_count,
                "total_yellow_available": est.total_yellow_available,
                "total_bursts_available": est.total_bursts_available,
                "skipped_bursts_due_to_limit": est.skipped_bursts_due_to_limit,
                "note": est.note,
                "est_usd": est.est_usd,
                "profile": est.profile,
                "model_alias": est.model_alias,
                "model_id": est.model_id,
                "provider": est.provider,
                "vlm_backend_ready": est.vlm_backend_ready,
                "session_spent_usd": est.session_spent_usd,
                "session_budget_usd": est.session_budget_usd,
                "monthly_spent_usd": est.monthly_spent_usd,
                "monthly_budget_usd": est.monthly_budget_usd,
                "review_url": ctx.review_url(args.session_id),
            }
        )

    try:
        res = ctx.services.triage.run_triage_sync(
            args.session_id,
            max_photos=args.max_photos,
            session_budget_usd=ctx.session_budget_override_usd,
            bypass_monthly_budget=ctx.bypass_monthly_budget,
            on_progress=ctx.progress_callback,
        )
    except (ValueError, MissingAPIKeyError, ModelError) as exc:
        return ToolExecutionResult.error(
            str(exc),
            data={
                "session_id": args.session_id,
                "review_url": ctx.review_url(args.session_id),
            },
        )

    return ToolExecutionResult.ok(
        {
            "job_id": res.job_id,
            "session_id": res.session_id,
            "dry_run": False,
            "status": res.status,
            "paused_reason": res.paused_reason,
            "yellow_total": res.yellow_total,
            "burst_total": res.burst_total,
            "reviewed_yellow": res.reviewed_yellow,
            "reviewed_bursts": res.reviewed_bursts,
            "failed_yellow": res.failed_yellow,
            "failed_bursts": res.failed_bursts,
            "failed_count": res.failed_count,
            "cache_hits": res.cache_hits,
            "paid_calls": res.paid_calls,
            "fresh_calls": res.fresh_calls,
            "spent_usd": res.spent_usd,
            "model_id": res.model_id,
            "provider": res.provider,
            "review_url": ctx.review_url(args.session_id),
        }
    )


def _handle_search_photos(ctx: ToolContext, args: SearchPhotosInput) -> ToolExecutionResult:
    try:
        res = ctx.services.search.search_photos(
            args.query,
            session_id=args.session_id,
            date_from=args.date_from,
            camera=args.camera,
            limit=args.limit,
            lang=ctx.lang,
        )
    except (ValueError, EmbedderError) as exc:
        return ToolExecutionResult.error(str(exc))

    matches = [m.to_dict() for m in res.matches]
    review_session = args.session_id or (matches[0]["session_id"] if matches else None)
    review_url = (
        ctx.review_url(str(review_session))
        if review_session
        else assert_tokenless_url(ctx._resolve_base_url())
    )
    return ToolExecutionResult.ok(
        {
            "query": res.query,
            "mode": res.mode,
            "semantic_ready": res.semantic_ready,
            "indexed_count": res.indexed_count,
            "total_candidates": res.total_candidates,
            "matches": matches,
            "review_url": review_url,
        }
    )


def _handle_find_similar(ctx: ToolContext, args: FindSimilarInput) -> ToolExecutionResult:
    try:
        res = ctx.services.search.find_similar(
            args.photo_id,
            limit=args.limit,
            lang=ctx.lang,
        )
    except (ValueError, EmbedderError) as exc:
        return ToolExecutionResult.error(str(exc))

    return ToolExecutionResult.ok(
        {
            "anchor_id": res.anchor_id,
            "session_id": res.session_id,
            "mode": res.mode,
            "semantic_ready": res.semantic_ready,
            "similar": [m.to_dict() for m in res.similar],
            "review_url": ctx.review_url(res.session_id),
        }
    )


def _handle_create_plan(ctx: ToolContext, args: CreatePlanInput) -> ToolExecutionResult:
    if args.action == "export_clean":
        return ToolExecutionResult.error(
            "Clean privacy export plans (`export_clean`) are an optional module activated in T16."
        )

    try:
        if args.photo_ids is not None:
            draft = ctx.services.plan.create_plan(args.session_id, args.photo_ids)
        else:
            draft = ctx.services.plan.draft_obvious_rejects(
                args.session_id,
                include_confirmed_burst_non_picks=(args.include_bursts == "confirmed_only"),
            )
    except (PlanError, SafeFSError, ValueError) as exc:
        return ToolExecutionResult.error(str(exc))

    # Run two-layer Independent Verifier before returning plan (SPEC §7.5, PLAN T12)
    from lightbox.agent.verifier import IndependentVerifier

    draft = IndependentVerifier(ctx.services).verify_and_finalize_draft_sync(
        args.session_id,
        draft,
    )

    if draft.plan is None:
        return ToolExecutionResult.ok(
            {
                "plan_id": None,
                "plan_hash": None,
                "status": "empty",
                "count": 0,
                "verifier_report": draft.verifier_report,
                "approve_url": ctx.review_url(args.session_id),
                "note": "No eligible reject photos found for a quarantine plan.",
            }
        )

    plan_payload: dict[str, Any] = {
        "plan_id": draft.plan.id,
        "plan_hash": draft.plan.plan_hash,
        "status": draft.plan.status,
        "count": draft.selected_count,
        "tech_reject_count": draft.tech_reject_count,
        "burst_non_pick_count": draft.burst_non_pick_count,
        "manual_reject_count": draft.manual_reject_count,
        "high_ratio_warning": draft.high_ratio_warning,
        "verifier_report": draft.verifier_report,
        "approve_url": ctx.approve_url(args.session_id, draft.plan.id),
        "cli_approve_command": f"lightbox approve {draft.plan.id}",
    }
    if ctx.surface.startswith("agent"):
        plan_payload["quarantine_folder_name"] = (
            Path(draft.plan.quarantine_dir).name if draft.plan.quarantine_dir else None
        )
    else:
        plan_payload["quarantine_dir"] = draft.plan.quarantine_dir
    return ToolExecutionResult.ok(plan_payload)


def _handle_apply_plan(ctx: ToolContext, args: ApplyPlanInput) -> ToolExecutionResult:
    try:
        plan, items = ctx.services.plan.get_plan(args.plan_id)
    except PlanError as exc:
        return ToolExecutionResult.error(str(exc))

    approve_url = ctx.approve_url(plan.session_id, plan.id)

    if plan.status in {"applied", "reverted"}:
        return ToolExecutionResult.error(
            f"Plan {plan.id} cannot be applied because its status is {plan.status!r}.",
            data={"plan_id": plan.id, "status": plan.status},
        )
    if plan.status == "stale":
        return ToolExecutionResult.error(
            f"Plan {plan.id} is stale because underlying photo verdicts changed; draft a fresh plan first.",
            data={"plan_id": plan.id, "status": "stale", "approve_url": approve_url},
        )

    # Enforce strict human approval gate (SPEC §8.2, §10.2(3))
    if plan.status != "approved":
        if args.confirmation_code:
            try:
                ctx.services.plan.verify_and_approve_with_code(
                    plan.id,
                    args.confirmation_code,
                    approved_by=f"{ctx.surface}:code",
                )
            except (ApprovalCodeError, PlanError) as exc:
                return ToolExecutionResult.error(
                    f"Approval code verification failed: {exc}",
                    data={"plan_id": plan.id, "approve_url": approve_url},
                )
        elif ctx.elicitation_approver is not None:
            approved_by_elicit = ctx.elicitation_approver(
                plan.id,
                len(items),
                plan.quarantine_dir,
            )
            if not approved_by_elicit:
                return ToolExecutionResult.error(
                    "Human approval was declined via MCP Elicitation. Plan was NOT applied.",
                    data={"plan_id": plan.id, "status": plan.status, "approve_url": approve_url},
                )
            try:
                ctx.services.plan.approve_plan(plan.id, approved_by=f"{ctx.surface}:elicitation")
            except PlanError as exc:
                return ToolExecutionResult.error(str(exc))
        else:
            return ToolExecutionResult.error(
                "Human approval is required before applying a quarantine plan. "
                f"Open {approve_url} in the Lightbox Web UI to review and approve, "
                f"or run `lightbox approve {plan.id}` in an interactive terminal to obtain a 6-digit code.",
                data={
                    "plan_id": plan.id,
                    "plan_hash": plan.plan_hash,
                    "status": plan.status,
                    "count": len(items),
                    "approve_url": approve_url,
                    "cli_approve_command": f"lightbox approve {plan.id}",
                },
            )

    try:
        res = ctx.services.plan.apply_plan(plan.id)
    except (PlanError, SafeFSError) as exc:
        return ToolExecutionResult.error(str(exc), data={"plan_id": plan.id, "approve_url": approve_url})

    return ToolExecutionResult.ok(
        {
            "plan_id": res.plan_id,
            "plan_hash": res.plan_hash,
            "status": res.status,
            "moved_count": res.moved_photo_count,
            "moved_file_count": res.moved_file_count,
            "quarantine_dir": str(res.quarantine_dir),
            "restore_hint": (
                f"All moved files are reversible via lightbox_restore_plan(plan_id={res.plan_id!r}) "
                f"or RESTORE_README.txt inside {res.quarantine_dir}."
            ),
        }
    )


def _handle_restore_plan(ctx: ToolContext, args: RestorePlanInput) -> ToolExecutionResult:
    try:
        res = ctx.services.plan.restore_plan(args.plan_id, photo_ids=args.photo_ids)
    except (PlanError, SafeFSError) as exc:
        return ToolExecutionResult.error(str(exc))

    return ToolExecutionResult.ok(
        {
            "plan_id": res.plan_id,
            "restored_count": len(res.restored_photo_ids),
            "restored_photo_ids": list(res.restored_photo_ids),
            "restored_file_count": res.restored_file_count,
            "plan_status": res.plan_status,
        }
    )


def _handle_shooting_digest(
    ctx: ToolContext,
    args: ShootingDigestInput,
) -> ToolExecutionResult:
    try:
        report = ctx.services.digest.generate_digest(
            session_id=args.session_id,
            date_from=args.date_from,
            date_to=args.date_to,
            min_samples=args.min_samples,
            lang=ctx.lang,
        )
    except ValueError as exc:
        return ToolExecutionResult.error(str(exc))

    return ToolExecutionResult.ok(report.to_dict())


def _handle_view_photos(
    ctx: ToolContext,
    args: ViewPhotosInput,
) -> ToolExecutionResult:
    """Attach up to 6 EXIF-stripped crops/thumbnails (cumulative <= 40 per agent session)."""
    requested = len(args.photo_ids)
    if ctx.viewed_photo_count + requested > ctx.max_viewed_photos:
        remaining = max(0, ctx.max_viewed_photos - ctx.viewed_photo_count)
        return ToolExecutionResult.error(
            f"Agent visual budget exceeded: requested {requested} photo(s), but only {remaining} "
            f"of {ctx.max_viewed_photos} remaining in this session. Direct the user to Web UI review_url."
        )

    from lightbox.models import strip_metadata_and_encode

    items: list[dict[str, Any]] = []
    images: list[ToolImagePayload] = []
    for pid in args.photo_ids:
        try:
            detail = ctx.services.review.get_photo_detail(pid)
            raw_bytes = ctx.services.review.get_photo_mcp_image_bytes(pid, mode=args.mode)
        except (ValueError, FileNotFoundError) as exc:
            return ToolExecutionResult.error(f"Failed to load photo {pid}: {exc}")
        clean_bytes, mime = strip_metadata_and_encode(raw_bytes, max_edge=512, fmt="JPEG", quality=84)
        eff = detail.effective_verdict
        items.append(
            {
                "photo_id": pid,
                "stem": _format_meta(ctx, detail.photo.stem),
                "bucket": eff.bucket if eff else "normal",
                "confidence": round(eff.confidence, 4) if eff else 0.5,
                "reason_codes": list(eff.reason_codes) if eff else [],
                "mode": args.mode,
            }
        )
        images.append(ToolImagePayload(data=clean_bytes, mime_type=mime, label=f"{pid}:{args.mode}"))

    ctx.viewed_photo_count += len(items)
    return ToolExecutionResult.ok(
        {
            "viewed": items,
            "viewed_total": ctx.viewed_photo_count,
            "viewed_limit": ctx.max_viewed_photos,
        },
        images=tuple(images),
    )


def _handle_update_plan(
    ctx: ToolContext,
    args: UpdatePlanInput,
) -> ToolExecutionResult:
    """Persist structured agent todos into SQLite `agent_sessions.plan_json` and emit SSE update."""
    todos = [t.model_dump() for t in args.todos]
    if ctx.agent_session_id:
        ctx.services.repo.update_agent_session(ctx.agent_session_id, plan_todos=todos)
    if ctx.on_update_plan is not None:
        ctx.on_update_plan(todos)
    return ToolExecutionResult.ok({"todos": todos, "count": len(todos)})


def _handle_get_memory(
    ctx: ToolContext,
    args: GetMemoryInput,
) -> ToolExecutionResult:
    """Read active user preferences from `<APP_DATA_DIR>/preferences.md`."""
    _ = args
    pref_path = ctx.services.app_paths.preferences_file
    if pref_path.is_file():
        content = pref_path.read_text(encoding="utf-8").strip()
    else:
        content = ""
    return ToolExecutionResult.ok(
        {
            "exists": bool(content),
            "preferences": content[:3000],
        }
    )


def _handle_propose_rule(
    ctx: ToolContext,
    args: ProposeRuleInput,
) -> ToolExecutionResult:
    """Propose a text preference rule into `memory_rules_meta` with `status='proposed'` (never auto-activates)."""
    from lightbox.agent.memory import MemoryManager, serialize_memory_rule

    try:
        rec = MemoryManager(ctx.services).propose_manual_rule(
            args.rule_text,
            rationale=args.rationale,
            evidence_ids=args.evidence_ids,
        )
    except ValueError as exc:
        return ToolExecutionResult.error(str(exc))

    return ToolExecutionResult.ok(
        {
            "rule": serialize_memory_rule(rec),
            "note": (
                "Rule recorded as status='proposed' only; it will NOT take effect until explicitly "
                "accepted by the photographer in Insights -> Memory."
            ),
        }
    )


# ---------------------------------------------------------------------------
# ToolSpec & ToolRegistry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Unified tool specification shared by MCP Server and built-in Agent."""

    name: str
    mcp_name: str | None
    description: str
    risk_level: RiskLevel
    agent_allowed: bool
    input_model: type[BaseModel]
    handler: Callable[[ToolContext, Any], ToolExecutionResult]
    read_only_hint: bool = False
    destructive_hint: bool = False
    idempotent_hint: bool = False

    def json_schema(self) -> dict[str, Any]:
        return self.input_model.model_json_schema()


PreExecuteHook = Callable[[ToolSpec, dict[str, Any], ToolContext], tuple[Literal["allow", "deny", "ask"], str | None]]
PostExecuteHook = Callable[[ToolSpec, dict[str, Any], ToolExecutionResult, ToolContext], None]


class ToolRegistry:
    """Single source of truth for tool schemas, risk tiers, argument validation, and execution."""

    def __init__(
        self,
        specs: Collection[ToolSpec] | None = None,
        *,
        attach_hooks: bool = True,
    ) -> None:
        self._by_name: dict[str, ToolSpec] = {}
        self._by_mcp_name: dict[str, ToolSpec] = {}
        self._pre_hooks: list[PreExecuteHook] = []
        self._post_hooks: list[PostExecuteHook] = []
        self._audit_hook_attached: bool = False
        for spec in sorted(specs or _build_default_specs(), key=lambda s: s.name):
            self.register(spec)
        if attach_hooks:
            from lightbox.agent.hooks import attach_default_hooks

            attach_default_hooks(registry=self)

    def mark_audit_hook_attached(self) -> None:
        """Prevent duplicate audit writes when `AuditLogHook` handles `post_tool_use` logging."""
        self._audit_hook_attached = True

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._by_name:
            raise ValueError(f"Duplicate tool name: {spec.name}")
        if spec.mcp_name is not None:
            if spec.mcp_name in self._by_mcp_name:
                raise ValueError(f"Duplicate MCP tool name: {spec.mcp_name}")
            self._by_mcp_name[spec.mcp_name] = spec
        self._by_name[spec.name] = spec

    def add_pre_hook(self, hook: PreExecuteHook) -> None:
        self._pre_hooks.append(hook)

    def add_post_hook(self, hook: PostExecuteHook) -> None:
        self._post_hooks.append(hook)

    def get(self, name_or_mcp_name: str) -> ToolSpec:
        spec = self._by_name.get(name_or_mcp_name) or self._by_mcp_name.get(name_or_mcp_name)
        if spec is None:
            raise KeyError(f"Unknown Lightbox tool: {name_or_mcp_name!r}")
        return spec

    def list_specs(self) -> list[ToolSpec]:
        """Return all registered tools in stable alphabetical order (Prompt Caching safe)."""
        return [self._by_name[k] for k in sorted(self._by_name)]

    def mcp_specs(self) -> list[ToolSpec]:
        """Return the 14 MCP-exposed tools in deterministic order."""
        return [self._by_mcp_name[k] for k in sorted(self._by_mcp_name)]

    def agent_schemas(
        self,
        allowed_tools: Collection[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Export LLM tool schemas for built-in Agent, strictly excluding non-agent tools like `apply_plan`."""
        allowed_set = set(allowed_tools) if allowed_tools is not None else None
        out: list[dict[str, Any]] = []
        for spec in self.list_specs():
            if not spec.agent_allowed:
                continue
            if allowed_set is not None and spec.name not in allowed_set:
                continue
            out.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "risk_level": spec.risk_level,
                    "input_schema": spec.json_schema(),
                }
            )
        return out

    def validate_args(self, name_or_mcp_name: str, arguments: dict[str, Any]) -> BaseModel:
        spec = self.get(name_or_mcp_name)
        return spec.input_model.model_validate(arguments)

    def execute(
        self,
        name_or_mcp_name: str,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> ToolExecutionResult:
        """Validate arguments, invoke hooks, execute handler, verify URLs, and record audit log."""
        started = time.monotonic()
        try:
            spec = self.get(name_or_mcp_name)
        except KeyError as exc:
            return ToolExecutionResult.error(str(exc))

        redacted = redact_tool_args(arguments)

        # Agent surface hard invariant: never execute disallowed tools (SPEC §7.1, §7.3)
        if ctx.surface.startswith("agent") and not spec.agent_allowed:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            ctx.services.system.record_audit_log(
                surface=ctx.surface,
                tool_name=spec.name,
                hook_decision="deny",
                args_redacted=redacted,
                duration_ms=elapsed_ms,
                summary="Tool forbidden on agent surface",
            )
            return ToolExecutionResult.error(
                f"Tool {spec.name!r} is forbidden on the agent surface (SPEC §7.1/§7.3)."
            )

        # MCP surface hard invariant: never execute agent-only tools over MCP (SPEC §7.3, §8.2)
        if ctx.surface.startswith("mcp:") and spec.mcp_name is None:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            ctx.services.system.record_audit_log(
                surface=ctx.surface,
                tool_name=spec.name,
                hook_decision="deny",
                args_redacted=redacted,
                duration_ms=elapsed_ms,
                summary="Tool not exposed on MCP surface",
            )
            return ToolExecutionResult.error(
                f"Tool {spec.name!r} is not exposed on the MCP surface (SPEC §7.3/§8.2)."
            )

        tool_log_name = spec.mcp_name or spec.name

        # Ensure OutboundPrivacyHook is attached on the active ModelRouter
        from lightbox.agent.hooks import attach_default_hooks

        attach_default_hooks(router=ctx.services.model_router)

        for pre_hook in self._pre_hooks:
            decision, reason = pre_hook(spec, arguments, ctx)
            if decision != "allow":
                elapsed_ms = int((time.monotonic() - started) * 1000)
                ctx.services.system.record_audit_log(
                    surface=ctx.surface,
                    tool_name=spec.name,
                    hook_decision=decision,
                    args_redacted=redacted,
                    duration_ms=elapsed_ms,
                    summary=reason or f"Blocked by pre-hook ({decision})",
                )
                upper_decision: Literal["ALLOW", "DENY", "ASK"] = (
                    "ASK" if decision == "ask" else "DENY"
                )
                return ToolExecutionResult.error(
                    reason or f"Blocked by hook ({decision}).",
                    hook_decision=upper_decision,
                )

        try:
            validated = spec.input_model.model_validate(arguments)
        except ValidationError as exc:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            msg = f"Invalid arguments for {tool_log_name}: {exc.errors()}"
            ctx.services.system.record_audit_log(
                surface=ctx.surface,
                tool_name=tool_log_name,
                hook_decision="deny",
                args_redacted=redacted,
                duration_ms=elapsed_ms,
                summary="Validation error",
            )
            return ToolExecutionResult.error(msg)

        result = spec.handler(ctx, validated)

        # Defense-in-depth: assert no URL field in result payload leaks auth tokens
        for url_key in ("review_url", "approve_url"):
            val = result.data.get(url_key)
            if isinstance(val, str):
                assert_tokenless_url(val)
        summary_sub = result.data.get("summary")
        if isinstance(summary_sub, dict):
            for url_key in ("review_url", "approve_url"):
                val = summary_sub.get(url_key)
                if isinstance(val, str):
                    assert_tokenless_url(val)

        for post_hook in self._post_hooks:
            post_hook(spec, arguments, result, ctx)

        if not self._audit_hook_attached:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            decision_str = "deny" if result.is_error else "allow"
            summary_text = (
                result.error_message
                if result.is_error
                else json.dumps(
                    {k: v for k, v in result.data.items() if k not in ("items", "frames", "matches")},
                    ensure_ascii=False,
                )[:240]
            )
            ctx.services.system.record_audit_log(
                surface=ctx.surface,
                tool_name=tool_log_name,
                hook_decision=decision_str,
                args_redacted=redacted,
                duration_ms=elapsed_ms,
                summary=summary_text,
            )
        return result


def _build_default_specs() -> list[ToolSpec]:
    """Construct the 14 canonical Lightbox business tools (SPEC §7.3 & §8.2)."""
    return [
        ToolSpec(
            name="scan_folder",
            mcp_name="lightbox_scan_folder",
            description=(
                "Scan a local photo folder with Lightbox's deterministic zero-LLM engine "
                "(RAW/JPG/HEIC/Live Photo pairing, EXIF, sharpness, face focus, bursts, creative guards). "
                "Returns summary + tokenless review_url if completed within wait_seconds, or a job_id to poll."
            ),
            risk_level="ANNOTATE",
            agent_allowed=True,
            input_model=ScanFolderInput,
            handler=_handle_scan_folder,
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
        ),
        ToolSpec(
            name="job_status",
            mcp_name="lightbox_job_status",
            description="Poll status, stage, progress counts, ETA, spent cost, and pause reason of a background job.",
            risk_level="READ",
            agent_allowed=True,
            input_model=JobStatusInput,
            handler=_handle_job_status,
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
        ),
        ToolSpec(
            name="get_session_summary",
            mcp_name="lightbox_session_summary",
            description=(
                "Get session bucket counts (red/yellow/normal/exempt), top reasons, camera distribution, "
                "low-resolution preview notice, burst counts, and tokenless Web UI review_url."
            ),
            risk_level="READ",
            agent_allowed=True,
            input_model=SessionSummaryInput,
            handler=_handle_session_summary,
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
        ),
        ToolSpec(
            name="query_photos",
            mcp_name="lightbox_list_photos",
            description=(
                "List compact photo handles for a session with bucket/reason filtering and cursor pagination "
                "(limit <= 30, default sorted by uncertainty ascending)."
            ),
            risk_level="READ",
            agent_allowed=True,
            input_model=ListPhotosInput,
            handler=_handle_list_photos,
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
        ),
        ToolSpec(
            name="get_photo",
            mcp_name="lightbox_get_photo",
            description=(
                "Inspect a single photo's metadata, sharpness/face metrics, verdict history, and optionally "
                "attach one EXIF-stripped image (thumb, face_crop, or sharpest_crop, max 768px WebP)."
            ),
            risk_level="READ_COST",
            agent_allowed=True,
            input_model=GetPhotoInput,
            handler=_handle_get_photo,
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
        ),
        ToolSpec(
            name="compare_burst",
            mcp_name="lightbox_compare_burst",
            description=(
                "Compare frames in a burst group with per-frame sharpness/face metrics plus numbered "
                "thumbnail contact sheet and 1:1 horizontal face crop strip (EXIF-stripped)."
            ),
            risk_level="READ_COST",
            agent_allowed=True,
            input_model=CompareBurstInput,
            handler=_handle_compare_burst,
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
        ),
        ToolSpec(
            name="set_verdict",
            mcp_name="lightbox_set_verdict",
            description=(
                "Record a non-destructive keep/reject/unsure verdict suggestion in SQLite "
                "(increments decision_version; never moves or deletes files; cannot reject exempt photos)."
            ),
            risk_level="ANNOTATE",
            agent_allowed=True,
            input_model=SetVerdictInput,
            handler=_handle_set_verdict,
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
        ),
        ToolSpec(
            name="recommend_burst_pick",
            mcp_name="lightbox_recommend_burst_pick",
            description="Record a recommended pick photo and plain-language rationale for a burst group without confirming deletion.",
            risk_level="ANNOTATE",
            agent_allowed=True,
            input_model=RecommendBurstPickInput,
            handler=_handle_recommend_burst_pick,
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
        ),
        ToolSpec(
            name="run_triage",
            mcp_name="lightbox_triage_uncertain",
            description=(
                "Estimate (dry_run=true) or trigger Lightbox concurrent VLM batch triage on uncertain (yellow) "
                "photos and unconfirmed bursts."
            ),
            risk_level="COMPUTE_COST",
            agent_allowed=True,
            input_model=TriageUncertainInput,
            handler=_handle_triage_uncertain,
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
        ),
        ToolSpec(
            name="search_photos",
            mcp_name="lightbox_search_photos",
            description="Search scanned photos using EXIF/session/date filters and local SigLIP 2 bilingual semantic vector similarity (with FTS5/EXIF fallback).",
            risk_level="READ",
            agent_allowed=True,
            input_model=SearchPhotosInput,
            handler=_handle_search_photos,
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
        ),
        ToolSpec(
            name="find_similar",
            mcp_name="lightbox_find_similar",
            description="Find visually similar photos via local SigLIP 2 vector cosine similarity (with burst/dHash fallback), excluding the anchor photo itself.",
            risk_level="READ",
            agent_allowed=True,
            input_model=FindSimilarInput,
            handler=_handle_find_similar,
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
        ),
        ToolSpec(
            name="create_plan",
            mcp_name="lightbox_create_plan",
            description=(
                "Draft an immutable quarantine execution plan bound to plan_hash and run deterministic Verifier checks. "
                "Never moves files; returns a tokenless approve_url for human review."
            ),
            risk_level="ANNOTATE",
            agent_allowed=True,
            input_model=CreatePlanInput,
            handler=_handle_create_plan,
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
        ),
        ToolSpec(
            name="apply_plan",
            mcp_name="lightbox_apply_plan",
            description=(
                "Execute an approved same-volume quarantine move plan via SafeFS. "
                "Strictly requires prior human approval (Web UI approval, MCP Elicitation, or 6-digit CLI code)."
            ),
            risk_level="MOVE",
            agent_allowed=False,
            input_model=ApplyPlanInput,
            handler=_handle_apply_plan,
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
        ),
        ToolSpec(
            name="restore_plan",
            mcp_name="lightbox_restore_plan",
            description="Restore an entire applied quarantine plan or selected photo_ids back to their original paths.",
            risk_level="MOVE_REVERT",
            agent_allowed=False,
            input_model=RestorePlanInput,
            handler=_handle_restore_plan,
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
        ),
        ToolSpec(
            name="shooting_digest",
            mcp_name=None,
            description=(
                "Generate a deterministic SQL-backed shooting digest (daily report) separating technical "
                "rejects from burst non-picks, with counts, ratios, single-dimension slices, and focal x shutter setups."
            ),
            risk_level="READ",
            agent_allowed=True,
            input_model=ShootingDigestInput,
            handler=_handle_shooting_digest,
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
        ),
        ToolSpec(
            name="view_photos",
            mcp_name=None,
            description=(
                "Visually inspect up to 6 EXIF-stripped photo crops or thumbnails in the main Agent context "
                "(cumulative cap 40 photos per agent session)."
            ),
            risk_level="READ_COST",
            agent_allowed=True,
            input_model=ViewPhotosInput,
            handler=_handle_view_photos,
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
        ),
        ToolSpec(
            name="update_plan",
            mcp_name=None,
            description="Persist and stream the Agent's explicit todo checklist for the current task.",
            risk_level="ANNOTATE",
            agent_allowed=True,
            input_model=UpdatePlanInput,
            handler=_handle_update_plan,
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
        ),
        ToolSpec(
            name="get_memory",
            mcp_name=None,
            description="Read active user aesthetic preferences from preferences.md.",
            risk_level="READ",
            agent_allowed=True,
            input_model=GetMemoryInput,
            handler=_handle_get_memory,
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
        ),
        ToolSpec(
            name="propose_rule",
            mcp_name=None,
            description=(
                "Propose a new text culling preference rule for human approval (status='proposed'; "
                "never activates until explicitly accepted into preferences.md)."
            ),
            risk_level="ANNOTATE",
            agent_allowed=True,
            input_model=ProposeRuleInput,
            handler=_handle_propose_rule,
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
        ),
    ]


__all__ = [
    "ApplyPlanInput",
    "CompareBurstInput",
    "CreatePlanInput",
    "FindSimilarInput",
    "GetPhotoInput",
    "JobStatusInput",
    "ListPhotosInput",
    "PostExecuteHook",
    "PreExecuteHook",
    "RecommendBurstPickInput",
    "RestorePlanInput",
    "RiskLevel",
    "ScanFolderInput",
    "SearchPhotosInput",
    "SessionSummaryInput",
    "SetVerdictInput",
    "ShootingDigestInput",
    "ToolContext",
    "ToolExecutionResult",
    "ToolImagePayload",
    "ToolRegistry",
    "ToolSpec",
    "TriageUncertainInput",
    "assert_tokenless_url",
    "redact_tool_args",
]
