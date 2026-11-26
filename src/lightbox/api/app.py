"""FastAPI local backend, REST routes, SSE event stream, and static `web_dist/` hosting (SPEC §9.2)."""

from __future__ import annotations

import json
import os
import secrets
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lightbox import __version__
from lightbox.api.events import EventBus
from lightbox.api.security import (
    LocalSecurityMiddleware,
    SessionAuthState,
    set_session_cookie,
)
from lightbox.i18n import render_reasons
from lightbox.paths import AppPaths, get_app_paths
from lightbox.services import (
    BucketName,
    DestinationExistsError,
    ImageSizeMode,
    LightboxServices,
    PlanError,
    PlanStateError,
    SafeFSError,
    StalePlanError,
    build_demo_dataset,
)

WEB_DIST_DIR = Path(__file__).resolve().parent.parent / "web_dist"

_FALLBACK_LANDING_HTML = """\
<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Lightbox Local Server</title></head>
<body>
  <h1>Lightbox Local API Server</h1>
  <p>Frontend bundle (<code>web_dist/</code>) is built in T08. Local API is active on 127.0.0.1.</p>
</body>
</html>
"""


class ScanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    folder_path: str
    sensitivity: Literal["conservative", "standard"] | None = None
    wait: bool = True
    max_workers: int | None = Field(default=None, ge=1)


class DemoBootstrapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_dir: str | None = None
    sensitivity: Literal["conservative", "standard"] = "conservative"


class VerdictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    photo_ids: list[str] = Field(min_length=1)
    bucket: BucketName
    reason_tag: str | None = None
    reason_codes: list[str] | None = None


class UndoVerdictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    photo_id: str


class ConfirmBurstRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pick_photo_id: str


class CreatePlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    photo_ids: list[str] = Field(min_length=1)
    strategy: Literal["outside_library", "sibling", "subfolder", "custom"] | None = None
    custom_dir: str | None = None


class DraftObviousRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["outside_library", "sibling", "subfolder", "custom"] | None = None
    custom_dir: str | None = None


class ApproveAndApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_hash: str
    compute_sha256: bool = False


class RestorePlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    photo_ids: list[str] | None = None


def _serialize_plan(
    plan: Any,
    items: list[Any] | tuple[Any, ...],
    *,
    session_name: str | None = None,
    photo_reasons: dict[str, tuple[str, tuple[str, ...]]] | None = None,
    lang: str = "zh-CN",
) -> dict[str, Any]:
    report = json.loads(plan.verifier_report_json) if plan.verifier_report_json else None
    serialized_items: list[dict[str, Any]] = []
    for it in items:
        row: dict[str, Any] = {
            "photo_id": it.photo_id,
            "decision_version": it.decision_version,
            "src_paths": list(it.src_paths),
            "dst_paths": list(it.dst_paths),
            "status": it.status,
            "done_at": it.done_at,
        }
        if photo_reasons is not None and it.photo_id in photo_reasons:
            stem, codes = photo_reasons[it.photo_id]
            row["stem"] = stem
            row["reason_codes"] = list(codes)
            row["reason_labels"] = render_reasons(codes, lang=lang)
        serialized_items.append(row)
    return {
        "id": plan.id,
        "session_id": plan.session_id,
        "session_name": session_name,
        "action_type": plan.action_type,
        "plan_hash": plan.plan_hash,
        "status": plan.status,
        "quarantine_dir": plan.quarantine_dir,
        "approved_by": plan.approved_by,
        "approved_at": plan.approved_at,
        "verifier_report": report,
        "item_count": len(items),
        "items": serialized_items,
    }


def _get_svcs(request: Request) -> Iterator[LightboxServices]:
    svcs = LightboxServices(app_paths=request.app.state.app_paths, migrate=False)
    try:
        yield svcs
    finally:
        svcs.close()


SvcsDep = Annotated[LightboxServices, Depends(_get_svcs)]


def create_app(
    *,
    app_paths: AppPaths | None = None,
    port: int = 8765,
    bootstrap_token: str | None = None,
    instance_id: str | None = None,
    web_dist_dir: Path | None = None,
) -> FastAPI:
    """Create the local-only FastAPI app with DNS-rebinding, Cookie, CSRF, REST, and SSE."""
    resolved_paths = app_paths or get_app_paths()
    # Migrate schema once at startup; per-request handles open lightweight WAL connections
    with LightboxServices(app_paths=resolved_paths, migrate=True):
        pass

    auth = SessionAuthState(bootstrap_token=bootstrap_token)
    event_bus = EventBus()
    resolved_instance_id = instance_id if instance_id is not None else secrets.token_hex(8)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            event_bus.close()

    app = FastAPI(
        title="Lightbox Local API",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=_lifespan,
    )
    app.state.app_paths = resolved_paths
    app.state.port = port
    app.state.auth = auth
    app.state.event_bus = event_bus
    app.state.instance_id = resolved_instance_id
    app.add_middleware(LocalSecurityMiddleware, allowed_port=port, auth=auth)

    # --- Local Liveness Probe (Host-checked, zero user data) ---

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, Any]:
        return {
            "service": "lightbox",
            "pid": os.getpid(),
            "instance_id": resolved_instance_id,
        }

    # --- Auth Endpoints ---

    @app.get("/auth/launch")
    def auth_launch(token: Annotated[str, Query(min_length=1)]) -> Response:
        session = auth.consume_bootstrap_token(token)
        if session is None:
            raise HTTPException(status_code=401, detail="Invalid or already-used launch token.")
        resp = RedirectResponse(url="/", status_code=302)
        set_session_cookie(resp, session)
        return resp

    @app.post("/api/auth/session")
    def auth_local_session() -> JSONResponse:
        session = auth.issue_session()
        resp = JSONResponse({"authenticated": True, "csrf_token": session.csrf_token})
        set_session_cookie(resp, session)
        return resp

    @app.get("/api/auth/me")
    def auth_me(request: Request) -> dict[str, Any]:
        return {"authenticated": True, "csrf_token": request.state.csrf_token}

    # --- System & Settings ---

    @app.post("/api/system/pick-folder")
    def pick_folder(svcs: SvcsDep) -> dict[str, Any]:
        chosen = svcs.system.pick_folder()
        return {"path": chosen, "canceled": chosen is None}

    @app.get("/api/settings")
    def get_settings(svcs: SvcsDep) -> dict[str, Any]:
        return svcs.system.get_settings().model_dump()

    @app.put("/api/settings")
    def put_settings(updates: dict[str, Any], svcs: SvcsDep) -> dict[str, Any]:
        try:
            updated = svcs.system.update_settings(updates)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from None
        return updated.model_dump()

    # --- Demo & Scan Jobs ---

    @app.post("/api/demo/bootstrap")
    def demo_bootstrap(svcs: SvcsDep, body: DemoBootstrapRequest | None = None) -> dict[str, Any]:
        req = body or DemoBootstrapRequest()
        demo_dir = (
            Path(req.target_dir).expanduser().resolve()
            if req.target_dir
            else (resolved_paths.root / "demo_shoot").resolve()
        )
        specs = build_demo_dataset(demo_dir)
        res = svcs.scan.scan_folder(
            demo_dir,
            sensitivity=req.sensitivity,
            max_workers=1,
            on_progress=lambda stage, done, total: event_bus.publish(
                "job.progress",
                {"stage": stage, "done": done, "total": total},
            ),
        )
        event_bus.publish(
            "job.progress",
            {
                "job_id": res.job_id,
                "status": res.status,
                "stage": "done",
                "done": res.total_photos,
                "total": res.total_photos,
            },
            session_id=res.session_id,
        )
        return {
            "demo_dir": str(demo_dir),
            "generated_photos": len(specs),
            "session_id": res.session_id,
            "job_id": res.job_id,
            "status": res.status,
            "total_photos": res.total_photos,
            "counts_by_bucket": res.counts_by_bucket,
            "burst_count": res.burst_count,
            "elapsed_s": res.elapsed_s,
        }

    @app.post("/api/sessions/scan")
    def scan_session(body: ScanRequest, svcs: SvcsDep) -> dict[str, Any]:
        folder = Path(body.folder_path).expanduser().resolve()
        if not folder.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a directory: {folder}")

        def _run_scan_with_events(target_svcs: LightboxServices) -> dict[str, Any]:
            res = target_svcs.scan.scan_folder(
                folder,
                sensitivity=body.sensitivity,
                max_workers=body.max_workers,
                on_progress=lambda stage, done, total: event_bus.publish(
                    "job.progress",
                    {"stage": stage, "done": done, "total": total},
                ),
            )
            event_bus.publish(
                "job.progress",
                {
                    "job_id": res.job_id,
                    "status": res.status,
                    "stage": "done",
                    "done": res.total_photos,
                    "total": res.total_photos,
                },
                session_id=res.session_id,
            )
            return {
                "session_id": res.session_id,
                "job_id": res.job_id,
                "status": res.status,
                "total_photos": res.total_photos,
                "counts_by_bucket": res.counts_by_bucket,
                "burst_count": res.burst_count,
                "cache_hits": res.cache_hits,
                "elapsed_s": res.elapsed_s,
            }

        if body.wait:
            return _run_scan_with_events(svcs)

        def _bg() -> None:
            with LightboxServices(app_paths=resolved_paths, migrate=False) as bg_svcs:
                _run_scan_with_events(bg_svcs)

        threading.Thread(target=_bg, daemon=True).start()
        return {"status": "queued", "folder_path": str(folder)}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, svcs: SvcsDep) -> dict[str, Any]:
        try:
            st = svcs.scan.get_job_status(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        return {
            "job_id": st.job_id,
            "kind": st.kind,
            "session_id": st.session_id,
            "status": st.status,
            "done": st.done,
            "total": st.total,
            "spent_usd": st.spent_usd,
            "paused_reason": st.paused_reason,
            "heartbeat_at": st.heartbeat_at,
        }

    @app.post("/api/jobs/{job_id}/pause")
    def pause_job(job_id: str, svcs: SvcsDep) -> dict[str, Any]:
        try:
            st = svcs.scan.pause_job(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        event_bus.publish(
            "job.progress",
            {"job_id": st.job_id, "status": st.status, "paused_reason": st.paused_reason},
            session_id=st.session_id,
        )
        return {"job_id": st.job_id, "status": st.status, "paused_reason": st.paused_reason}

    @app.post("/api/jobs/{job_id}/resume")
    def resume_job(job_id: str, svcs: SvcsDep) -> dict[str, Any]:
        try:
            res = svcs.scan.resume_job(
                job_id,
                max_workers=1,
                on_progress=lambda stage, done, total: event_bus.publish(
                    "job.progress",
                    {"job_id": job_id, "stage": stage, "done": done, "total": total},
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        event_bus.publish(
            "job.progress",
            {
                "job_id": res.job_id,
                "status": res.status,
                "done": res.total_photos,
                "total": res.total_photos,
            },
            session_id=res.session_id,
        )
        return {
            "job_id": res.job_id,
            "session_id": res.session_id,
            "status": res.status,
            "total_photos": res.total_photos,
            "cache_hits": res.cache_hits,
            "counts_by_bucket": res.counts_by_bucket,
        }

    # --- SSE Event Stream ---

    @app.get("/api/events")
    async def sse_events(
        session_id: Annotated[str | None, Query()] = None,
        once: Annotated[bool, Query()] = False,
    ) -> StreamingResponse:
        async def _gen() -> Any:
            async for chunk in event_bus.subscribe(session_id=session_id):
                yield chunk
                if once:
                    break

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --- Sessions, Photos, Images, Verdicts, Bursts ---

    @app.get("/api/sessions")
    def list_sessions(svcs: SvcsDep) -> dict[str, Any]:
        sessions = svcs.review.list_sessions()
        return {
            "items": [
                {
                    "id": s.id,
                    "name": s.name,
                    "root_path": s.root_path,
                    "sensitivity": s.sensitivity,
                    "engine_version": s.engine_version,
                    "created_at": s.created_at,
                    "last_scanned_at": s.last_scanned_at,
                }
                for s in sessions
            ]
        }

    @app.get("/api/sessions/{session_id}/summary")
    def session_summary(
        session_id: str,
        svcs: SvcsDep,
        lang: Annotated[str, Query()] = "zh-CN",
    ) -> dict[str, Any]:
        try:
            summary = svcs.review.get_session_summary(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        return {
            "session": {
                "id": summary.session.id,
                "name": summary.session.name,
                "root_path": summary.session.root_path,
                "sensitivity": summary.session.sensitivity,
                "last_scanned_at": summary.session.last_scanned_at,
            },
            "total_photos": summary.total_photos,
            "active_total_photos": summary.active_total_photos,
            "counts_by_bucket": summary.counts_by_bucket,
            "active_counts_by_bucket": summary.active_counts_by_bucket,
            "reviewed_count": summary.reviewed_count,
            "unreviewed_count": summary.unreviewed_count,
            "actionable_reject_count": summary.actionable_reject_count,
            "top_reasons": [
                {"code": code, "count": cnt, "label": render_reasons([code], lang=lang)[0]}
                for code, cnt in summary.top_reasons
            ],
            "camera_distribution": summary.camera_distribution,
            "low_res_preview_count": summary.low_res_preview_count,
            "burst_count": summary.burst_count,
            "unconfirmed_burst_count": summary.unconfirmed_burst_count,
            "quarantined_count": summary.quarantined_count,
        }

    @app.post("/api/sessions/{session_id}/open-folder")
    def open_session_folder(session_id: str, svcs: SvcsDep) -> dict[str, Any]:
        try:
            summary = svcs.review.get_session_summary(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        folder = Path(summary.session.root_path)
        if not folder.is_dir():
            raise HTTPException(status_code=404, detail=f"Folder does not exist: {folder}")
        opened = svcs.system.open_folder(folder)
        return {"opened": opened, "path": str(folder)}

    @app.post("/api/sessions/{session_id}/reset-review")
    def reset_session_review(session_id: str, svcs: SvcsDep) -> dict[str, Any]:
        try:
            reset_count = svcs.review.reset_session_review(session_id, actor="web:user")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        return {"session_id": session_id, "reset_count": reset_count}

    @app.post("/api/sessions/{session_id}/auto-screen")
    def auto_screen_session(session_id: str, svcs: SvcsDep) -> dict[str, Any]:
        try:
            summary = svcs.review.get_session_summary(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        folder = Path(summary.session.root_path)
        if folder.is_dir():
            # Fast cache-hit rescan ensures any new photos or newly installed models are analyzed
            sens: Literal["conservative", "standard"] = (
                "standard" if summary.session.sensitivity == "standard" else "conservative"
            )
            svcs.scan.scan_folder(
                folder,
                sensitivity=sens,
                max_workers=1,
            )
        screen = svcs.review.apply_engine_screen(session_id, actor="web:user")
        return {
            "session_id": screen.session_id,
            "total_active": screen.total_active,
            "flagged_red": screen.flagged_red,
            "flagged_yellow": screen.flagged_yellow,
            "unchanged_normal": screen.unchanged_normal,
        }

    @app.get("/api/sessions/{session_id}/photos")
    def session_photos(
        session_id: str,
        svcs: SvcsDep,
        bucket: Annotated[str | None, Query()] = None,
        reason: Annotated[str | None, Query()] = None,
        sort: Annotated[str, Query()] = "uncertainty",
        limit: Annotated[int, Query(ge=1, le=200)] = 30,
        offset: Annotated[int, Query(ge=0)] = 0,
        lang: Annotated[str, Query()] = "zh-CN",
        include_quarantined: Annotated[bool, Query()] = False,
    ) -> dict[str, Any]:
        items = svcs.review.list_photos(
            session_id,
            bucket=bucket,
            reason=reason,
            sort=sort,
            limit=limit,
            offset=offset,
            include_quarantined=include_quarantined,
        )
        return {
            "items": [
                {
                    "photo_id": it.photo_id,
                    "stem": it.stem,
                    "rel_dir": it.rel_dir,
                    "bucket": it.bucket,
                    "source": it.source,
                    "confidence": it.confidence,
                    "reason_codes": list(it.reason_codes),
                    "reason_labels": render_reasons(it.reason_codes, lang=lang),
                    "decision_version": it.decision_version,
                    "is_protected": it.is_protected,
                    "is_quarantined": it.is_quarantined,
                    "burst_id": it.burst_id,
                    "preview_w": it.preview_w,
                    "preview_h": it.preview_h,
                    "face_count": it.face_count,
                    "seq_no": it.seq_no,
                }
                for it in items
            ],
            "offset": offset,
            "limit": limit,
        }

    @app.get("/api/sessions/{session_id}/bursts")
    def session_bursts(session_id: str, svcs: SvcsDep) -> dict[str, Any]:
        bursts = svcs.burst.list_bursts(session_id)
        return {
            "items": [
                {
                    "id": b.burst.id,
                    "kind": b.burst.kind,
                    "confirmed": b.burst.confirmed,
                    "members": [
                        {
                            "photo_id": m.photo_id,
                            "rank": m.rank,
                            "is_pick": m.is_pick,
                            "ai_note": m.ai_note,
                        }
                        for m in b.members
                    ],
                }
                for b in bursts
            ]
        }

    @app.get("/api/photos/{photo_id}/image")
    def photo_image(
        photo_id: str,
        svcs: SvcsDep,
        size: Annotated[ImageSizeMode, Query()] = "grid",
        region: Annotated[str | None, Query()] = None,
    ) -> Response:
        try:
            data = svcs.review.get_photo_image_bytes(photo_id, size=size, region=region)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        return Response(
            content=data,
            media_type="image/webp",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.post("/api/verdicts")
    def set_verdicts(body: VerdictRequest, svcs: SvcsDep) -> dict[str, Any]:
        updated: list[dict[str, Any]] = []
        for pid in body.photo_ids:
            try:
                v = svcs.review.set_verdict(
                    pid,
                    bucket=body.bucket,
                    source="user",
                    reason_codes=body.reason_codes,
                    reason_tag=body.reason_tag,
                    actor="web:user",
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
            detail = svcs.review.get_photo_detail(pid)
            updated.append(
                {
                    "photo_id": pid,
                    "bucket": v.bucket,
                    "reason_codes": list(v.reason_codes),
                    "decision_version": detail.photo.decision_version,
                }
            )
        return {"updated": updated}

    @app.post("/api/verdicts/undo")
    def undo_verdict(body: UndoVerdictRequest, svcs: SvcsDep) -> dict[str, Any]:
        try:
            v = svcs.review.undo_verdict(body.photo_id, actor="web:user")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        if v is None:
            return {"undone": False, "photo_id": body.photo_id}
        detail = svcs.review.get_photo_detail(body.photo_id)
        return {
            "undone": True,
            "photo_id": body.photo_id,
            "bucket": v.bucket,
            "decision_version": detail.photo.decision_version,
        }

    @app.post("/api/bursts/{burst_id}/confirm")
    def confirm_burst(burst_id: str, body: ConfirmBurstRequest, svcs: SvcsDep) -> dict[str, Any]:
        try:
            res = svcs.burst.confirm_burst(burst_id, body.pick_photo_id, actor="web:user")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return {
            "burst_id": res.burst_id,
            "pick_photo_id": res.pick_photo_id,
            "non_pick_photo_ids": list(res.non_pick_photo_ids),
            "confirmed": res.confirmed,
        }

    # --- Plans: Draft / Obvious Rejects / Approve-and-Apply / Restore ---

    @app.get("/api/plans")
    def list_plans(
        svcs: SvcsDep,
        session_id: Annotated[str | None, Query()] = None,
    ) -> dict[str, Any]:
        session_names = {s.id: s.name for s in svcs.review.list_sessions()}
        plans = svcs.plan.list_plans(session_id)
        return {
            "items": [
                _serialize_plan(
                    p,
                    svcs.plan.get_plan(p.id)[1],
                    session_name=session_names.get(p.session_id),
                )
                for p in plans
            ]
        }

    @app.get("/api/plans/{plan_id}")
    def get_plan(plan_id: str, svcs: SvcsDep) -> dict[str, Any]:
        try:
            plan, items = svcs.plan.get_plan(plan_id)
        except PlanError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        return _serialize_plan(plan, items)

    @app.post("/api/sessions/{session_id}/plans")
    def create_plan(session_id: str, body: CreatePlanRequest, svcs: SvcsDep) -> dict[str, Any]:
        try:
            draft = svcs.plan.create_plan(
                session_id,
                body.photo_ids,
                strategy=body.strategy,
                custom_dir=Path(body.custom_dir) if body.custom_dir else None,
            )
        except PlanError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        assert draft.plan is not None
        event_bus.publish(
            "plan.updated",
            {"plan_id": draft.plan.id, "status": draft.plan.status},
            session_id=session_id,
        )
        return {
            "plan": _serialize_plan(draft.plan, draft.items),
            "tech_reject_count": draft.tech_reject_count,
            "burst_non_pick_count": draft.burst_non_pick_count,
            "manual_reject_count": draft.manual_reject_count,
            "high_ratio_warning": draft.high_ratio_warning,
            "superseded_plan_ids": list(draft.superseded_plan_ids),
        }

    @app.post("/api/sessions/{session_id}/plans/obvious-rejects")
    def draft_obvious_rejects(
        session_id: str,
        svcs: SvcsDep,
        body: DraftObviousRequest | None = None,
        lang: Annotated[str, Query()] = "zh-CN",
    ) -> dict[str, Any]:
        req = body or DraftObviousRequest()
        try:
            draft = svcs.plan.draft_obvious_rejects(
                session_id,
                strategy=req.strategy,
                custom_dir=Path(req.custom_dir) if req.custom_dir else None,
            )
        except PlanError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        photo_reasons = (
            svcs.review.get_photo_reasons_map(session_id) if draft.plan is not None else None
        )
        if draft.plan is not None:
            event_bus.publish(
                "plan.updated",
                {"plan_id": draft.plan.id, "status": draft.plan.status},
                session_id=session_id,
            )
        return {
            "plan": (
                _serialize_plan(
                    draft.plan,
                    draft.items,
                    photo_reasons=photo_reasons,
                    lang=lang,
                )
                if draft.plan
                else None
            ),
            "selected_count": draft.selected_count,
            "tech_reject_count": draft.tech_reject_count,
            "burst_non_pick_count": draft.burst_non_pick_count,
            "manual_reject_count": draft.manual_reject_count,
            "high_ratio_warning": draft.high_ratio_warning,
            "excluded_already_quarantined": draft.excluded_already_quarantined,
            "excluded_unconfirmed_burst": draft.excluded_unconfirmed_burst,
            "excluded_in_applying_plan": draft.excluded_in_applying_plan,
            "superseded_plan_ids": list(draft.superseded_plan_ids),
            "verifier_report": draft.verifier_report,
        }

    @app.post("/api/plans/{plan_id}/approve-and-apply")
    def approve_and_apply_plan(
        plan_id: str,
        body: ApproveAndApplyRequest,
        svcs: SvcsDep,
    ) -> dict[str, Any]:
        try:
            plan, _items = svcs.plan.get_plan(plan_id)
        except PlanError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None

        if body.plan_hash != plan.plan_hash:
            raise HTTPException(
                status_code=409,
                detail=f"plan_hash mismatch (expected {plan.plan_hash}, got {body.plan_hash}).",
            )

        try:
            if plan.status in {"draft", "reviewed"}:
                svcs.plan.approve_plan(plan_id, approved_by="web:user")
            result = svcs.plan.apply_plan(
                plan_id,
                resume=(plan.status == "applying"),
                compute_sha256=body.compute_sha256,
            )
        except StalePlanError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except DestinationExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except (PlanStateError, PlanError, SafeFSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        event_bus.publish(
            "plan.updated",
            {"plan_id": result.plan_id, "status": result.status},
            session_id=plan.session_id,
        )
        return {
            "plan_id": result.plan_id,
            "plan_hash": result.plan_hash,
            "status": result.status,
            "quarantine_dir": str(result.quarantine_dir),
            "moved_photo_count": result.moved_photo_count,
            "moved_file_count": result.moved_file_count,
            "manifest_path": str(result.manifest_path),
            "readme_path": str(result.readme_path),
        }

    @app.post("/api/plans/{plan_id}/restore")
    def restore_plan(
        plan_id: str,
        svcs: SvcsDep,
        body: RestorePlanRequest | None = None,
    ) -> dict[str, Any]:
        req = body or RestorePlanRequest()
        try:
            plan, _ = svcs.plan.get_plan(plan_id)
            res = svcs.plan.restore_plan(plan_id, photo_ids=req.photo_ids)
        except (PlanStateError, DestinationExistsError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except PlanError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except SafeFSError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        event_bus.publish(
            "plan.updated",
            {"plan_id": res.plan_id, "status": res.plan_status},
            session_id=plan.session_id,
        )
        return {
            "plan_id": res.plan_id,
            "restored_photo_ids": list(res.restored_photo_ids),
            "restored_file_count": res.restored_file_count,
            "plan_status": res.plan_status,
        }

    @app.post("/api/plans/{plan_id}/open-folder")
    def open_plan_folder(plan_id: str, svcs: SvcsDep) -> dict[str, Any]:
        try:
            plan, _ = svcs.plan.get_plan(plan_id)
        except PlanError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        if not plan.quarantine_dir:
            raise HTTPException(status_code=404, detail="Folder path is not available.")
        folder = Path(plan.quarantine_dir)
        if not folder.is_dir():
            raise HTTPException(status_code=404, detail=f"Folder does not exist: {folder}")
        opened = svcs.system.open_folder(folder)
        return {"opened": opened, "path": str(folder)}

    # --- Static Frontend Hosting (`web_dist/`) ---

    dist = (web_dist_dir or WEB_DIST_DIR).resolve()
    assets_dir = dist / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> Response:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found.")
        index_file = dist / "index.html"
        if index_file.is_file():
            return HTMLResponse(index_file.read_text(encoding="utf-8"))
        return HTMLResponse(_FALLBACK_LANDING_HTML)

    return app


__all__ = ["WEB_DIST_DIR", "create_app"]
