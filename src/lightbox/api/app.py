"""FastAPI local backend, REST routes, SSE event stream, and static `web_dist/` hosting (SPEC §9.2)."""

from __future__ import annotations

import json
import os
import secrets
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from lightbox import __version__
from lightbox.agent import (
    AgentRunner,
    AgentSessionBusyError,
    IndependentVerifier,
    MemoryManager,
    attach_default_hooks,
    serialize_memory_rule,
)
from lightbox.api.events import EventBus
from lightbox.api.security import (
    LocalSecurityMiddleware,
    SessionAuthState,
    set_session_cookie,
)
from lightbox.config import AIProviderName
from lightbox.i18n import render_reasons
from lightbox.paths import AppPaths, get_app_paths
from lightbox.services import (
    BucketName,
    DestinationExistsError,
    EmbedderError,
    EndpointBindingError,
    FaceEmbedderError,
    ImageSizeMode,
    LightboxServices,
    MCPClientId,
    MissingAPIKeyError,
    ModelError,
    PeopleError,
    PlanError,
    PlanStateError,
    ProbeAdapterFactory,
    SafeFSError,
    StalePlanError,
    UnpricedTriageConfirmationRequiredError,
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
    enable_embed: bool = False
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
    skip_llm_verify: bool = False


class DraftObviousRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["outside_library", "sibling", "subfolder", "custom"] | None = None
    custom_dir: str | None = None
    skip_llm_verify: bool = False


class ApproveAndApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_hash: str
    compute_sha256: bool = False


class RestorePlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    photo_ids: list[str] | None = None


class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_photos: int = Field(default=50, ge=1, le=500)
    dry_run: bool = True
    profile: Literal["balanced", "best", "local"] | None = None
    budget_usd: float | None = Field(default=None, ge=0.0)
    wait: bool = False
    confirm_unpriced: bool = False


class TestAIConnectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: AIProviderName | None = None
    agent_model: str | None = None
    triage_model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    is_local_free: bool | None = None


class SaveAIConnectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: AIProviderName
    agent_model: str | None = None
    triage_model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    reviewer: Literal["none", "agent", "cross_family"] = "none"
    is_local_free: bool = False
    input_cost_per_mtok: float | None = Field(default=None, ge=0.0)
    output_cost_per_mtok: float | None = Field(default=None, ge=0.0)
    triage_input_cost_per_mtok: float | None = Field(default=None, ge=0.0)
    triage_output_cost_per_mtok: float | None = Field(default=None, ge=0.0)
    max_session_calls: int = Field(default=20, ge=1, le=200)
    max_triage_photos_unpriced: int = Field(default=20, ge=1, le=500)

    @model_validator(mode="after")
    def _validate_paired_costs(self) -> SaveAIConnectionRequest:
        if (self.input_cost_per_mtok is None) != (self.output_cost_per_mtok is None):
            raise ValueError(
                "input_cost_per_mtok and output_cost_per_mtok must both be set or both omitted."
            )
        if (self.triage_input_cost_per_mtok is None) != (self.triage_output_cost_per_mtok is None):
            raise ValueError(
                "triage_input_cost_per_mtok and triage_output_cost_per_mtok must both be set or both omitted."
            )
        return self


class MCPSnippetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client: MCPClientId


class PeopleIndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class FaceRefInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_fp: str = Field(min_length=1)
    face_idx: int = Field(ge=0)


class PeopleLabelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["name", "merge", "ignore", "split", "reassign"]
    name: str | None = None
    is_self: bool = False
    cluster_id: str | None = None
    person_id: str | None = None
    faces: list[FaceRefInput] | None = None
    source_cluster_id: str | None = None
    source_cluster_ids: list[str] | None = None
    target_cluster_id: str | None = None
    source_person_id: str | None = None
    source_person_ids: list[str] | None = None
    target_person_id: str | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class PeopleExportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_dir: str = Field(min_length=1)
    session_id: str | None = None
    group_keys: list[str] | None = None
    include_unnamed: bool = False
    include_no_face: bool = False
    lang: str = "zh-CN"


class CreateAgentSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    mode: Literal["chat", "supervisor"] = "chat"
    message: str | None = None
    constraints: list[str] = Field(default_factory=list)
    profile: Literal["balanced", "best", "local"] | None = None
    lang: str = "zh-CN"


class SendAgentMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = ""
    approve_ask: bool | None = None
    constraints: list[str] = Field(default_factory=list)
    profile: Literal["balanced", "best", "local"] | None = None
    lang: str = "zh-CN"


class SavePreferencesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str


class ConsolidateMemoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    force: bool = False
    profile: Literal["balanced", "best", "local"] | None = None


class MemoryRuleDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["accept", "reject"]
    rule_text: str | None = Field(default=None, max_length=240)


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
    factory = getattr(request.app.state, "model_router_factory", None)
    if factory is not None:
        svcs.override_model_router(factory(svcs))
    attach_default_hooks(router=svcs.model_router)
    embedder_factory = getattr(request.app.state, "embedder_factory", None)
    if embedder_factory is not None:
        svcs.override_embedder(embedder_factory(svcs))
    face_embedder_factory = getattr(request.app.state, "face_embedder_factory", None)
    if face_embedder_factory is not None:
        svcs.override_face_embedder(face_embedder_factory(svcs))
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
    model_router_factory: Callable[[LightboxServices], Any] | None = None,
    embedder_factory: Callable[[LightboxServices], Any] | None = None,
    face_embedder_factory: Callable[[LightboxServices], Any] | None = None,
    probe_adapter_factory: ProbeAdapterFactory | None = None,
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
    app.state.model_router_factory = model_router_factory
    app.state.embedder_factory = embedder_factory
    app.state.face_embedder_factory = face_embedder_factory
    app.state.probe_adapter_factory = probe_adapter_factory
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
        if "ai" in updates:
            raise HTTPException(
                status_code=422,
                detail="AI connection settings must be managed via /api/ai/connection.",
            )
        try:
            updated = svcs.system.update_settings(updates)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        return updated.model_dump()

    # --- AI Connection, Verifiable Probes, Ollama & Read-only MCP Clients ---

    @app.get("/api/ai/connection")
    def get_ai_connection(svcs: SvcsDep) -> dict[str, Any]:
        return svcs.system.get_ai_connection()

    @app.post("/api/ai/connection/test")
    async def test_ai_connection_route(
        svcs: SvcsDep,
        body: TestAIConnectionRequest | None = None,
    ) -> dict[str, Any]:
        req = body or TestAIConnectionRequest()
        try:
            return await svcs.system.test_ai_connection(
                provider=req.provider,
                agent_model=req.agent_model,
                triage_model=req.triage_model,
                base_url=req.base_url,
                api_key=req.api_key,
                is_local_free=req.is_local_free,
                adapter_factory=probe_adapter_factory,
            )
        except EndpointBindingError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_class": "endpoint_binding_mismatch",
                    "message": str(exc),
                },
            ) from None
        except (MissingAPIKeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.put("/api/ai/connection")
    def put_ai_connection(
        body: SaveAIConnectionRequest,
        svcs: SvcsDep,
    ) -> dict[str, Any]:
        try:
            return svcs.system.save_ai_connection(
                provider=body.provider,
                agent_model=body.agent_model,
                triage_model=body.triage_model,
                base_url=body.base_url,
                api_key=body.api_key,
                reviewer=body.reviewer,
                is_local_free=body.is_local_free,
                input_cost_per_mtok=body.input_cost_per_mtok,
                output_cost_per_mtok=body.output_cost_per_mtok,
                triage_input_cost_per_mtok=body.triage_input_cost_per_mtok,
                triage_output_cost_per_mtok=body.triage_output_cost_per_mtok,
                max_session_calls=body.max_session_calls,
                max_triage_photos_unpriced=body.max_triage_photos_unpriced,
            )
        except EndpointBindingError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_class": "endpoint_binding_mismatch",
                    "message": str(exc),
                },
            ) from None
        except (MissingAPIKeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.delete("/api/ai/connection")
    def delete_ai_connection_route(
        svcs: SvcsDep,
        delete_credential: Annotated[bool, Query()] = True,
    ) -> dict[str, Any]:
        return svcs.system.delete_ai_connection(delete_credential=delete_credential)

    @app.get("/api/ai/ollama/models")
    async def list_ollama_models(
        svcs: SvcsDep,
        base_url: Annotated[str | None, Query()] = None,
    ) -> dict[str, Any]:
        try:
            return await anyio.to_thread.run_sync(
                lambda: svcs.system.discover_ollama_models(base_url=base_url)
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.get("/api/ai/mcp/clients")
    def list_mcp_clients(svcs: SvcsDep) -> dict[str, Any]:
        return svcs.system.detect_mcp_clients()

    @app.post("/api/ai/mcp/snippet")
    def create_mcp_snippet(
        body: MCPSnippetRequest,
        svcs: SvcsDep,
    ) -> dict[str, Any]:
        return svcs.system.generate_mcp_snippet(body.client)

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
                enable_embed=body.enable_embed,
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
                "embedded_count": res.embedded_count,
                "elapsed_s": res.elapsed_s,
            }

        if body.wait:
            return _run_scan_with_events(svcs)

        def _bg() -> None:
            with LightboxServices(app_paths=resolved_paths, migrate=False) as bg_svcs:
                if embedder_factory is not None:
                    bg_svcs.override_embedder(embedder_factory(bg_svcs))
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
        limit: Annotated[int, Query(ge=1, le=2000)] = 30,
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

    # --- Semantic Search & Find Similar (PLAN T11) ---

    @app.get("/api/search")
    async def search_photos(
        svcs: SvcsDep,
        q: Annotated[str, Query()] = "",
        query: Annotated[str | None, Query()] = None,
        session_id: Annotated[str | None, Query()] = None,
        date_from: Annotated[str | None, Query()] = None,
        date_to: Annotated[str | None, Query()] = None,
        camera: Annotated[str | None, Query()] = None,
        light_bucket: Annotated[str | None, Query()] = None,
        focal_min: Annotated[float | None, Query()] = None,
        focal_max: Annotated[float | None, Query()] = None,
        location: Annotated[str | None, Query()] = None,
        ignore_location: Annotated[bool, Query()] = False,
        strictness: Annotated[Literal["strict", "balanced", "relaxed", "all"], Query()] = "balanced",
        min_score: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 24,
        lang: Annotated[str, Query()] = "zh-CN",
    ) -> dict[str, Any]:
        effective_query = q if q.strip() else (query or "")
        try:
            res = await anyio.to_thread.run_sync(
                lambda: svcs.search.search_photos(
                    effective_query,
                    session_id=session_id,
                    date_from=date_from,
                    date_to=date_to,
                    camera=camera,
                    light_bucket=light_bucket,
                    focal_min=focal_min,
                    focal_max=focal_max,
                    location=location,
                    ignore_location=ignore_location,
                    strictness=strictness,
                    min_score=min_score,
                    limit=limit,
                    lang=lang,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except EmbedderError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        return {
            "query": res.query,
            "mode": res.mode,
            "semantic_ready": res.semantic_ready,
            "indexed_count": res.indexed_count,
            "total_candidates": res.total_candidates,
            "cameras": list(res.cameras),
            "locations": list(res.locations),
            "detected_location_key": res.detected_location_key,
            "detected_location_label": res.detected_location_label,
            "effective_visual_query": res.effective_visual_query,
            "strictness": res.strictness,
            "score_threshold": res.score_threshold,
            "matches": [m.to_dict() for m in res.matches],
            "weak_matches": [m.to_dict() for m in res.weak_matches],
        }

    @app.get("/api/photos/{photo_id}/similar")
    async def similar_photos(
        photo_id: str,
        svcs: SvcsDep,
        limit: Annotated[int, Query(ge=1, le=30)] = 12,
        session_id: Annotated[str | None, Query()] = None,
        strictness: Annotated[Literal["strict", "balanced", "relaxed", "all"], Query()] = "balanced",
        min_score: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
        lang: Annotated[str, Query()] = "zh-CN",
    ) -> dict[str, Any]:
        try:
            res = await anyio.to_thread.run_sync(
                lambda: svcs.search.find_similar(
                    photo_id,
                    limit=limit,
                    session_id=session_id,
                    strictness=strictness,
                    min_score=min_score,
                    lang=lang,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except EmbedderError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        return {
            "anchor": res.anchor.to_dict(),
            "anchor_id": res.anchor_id,
            "session_id": res.session_id,
            "mode": res.mode,
            "semantic_ready": res.semantic_ready,
            "strictness": res.strictness,
            "score_threshold": res.score_threshold,
            "similar": [m.to_dict() for m in res.similar],
            "weak_similar": [m.to_dict() for m in res.weak_similar],
        }

    @app.post("/api/sessions/{session_id}/embed")
    async def embed_session(
        session_id: str,
        svcs: SvcsDep,
    ) -> dict[str, Any]:
        try:
            res = await anyio.to_thread.run_sync(
                lambda: svcs.search.index_session(
                    session_id,
                    on_progress=lambda stage, done, total: event_bus.publish(
                        "job.progress",
                        {"stage": stage, "done": done, "total": total},
                        session_id=session_id,
                    ),
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except EmbedderError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        return {
            "session_id": res.session_id,
            "indexed_count": res.indexed_count,
            "skipped_count": res.skipped_count,
            "tagged_count": res.tagged_count,
            "total_photos": res.total_photos,
            "model_id": res.model_id,
            "elapsed_s": res.elapsed_s,
        }

    # --- People Auto-Classification (PLAN T18) ---

    @app.get("/api/photos/{photo_id}/face-crop")
    def photo_face_crop(
        photo_id: str,
        svcs: SvcsDep,
        face_idx: Annotated[int, Query(ge=0)] = 0,
        size: Annotated[int, Query(ge=64, le=320)] = 160,
    ) -> Response:
        try:
            data = svcs.people.get_face_crop_bytes(photo_id, face_idx, size=size)
        except (PeopleError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        return Response(
            content=data,
            media_type="image/webp",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.post("/api/sessions/{session_id}/people/index")
    async def index_session_people(
        session_id: str,
        svcs: SvcsDep,
        body: PeopleIndexRequest | None = None,
    ) -> dict[str, Any]:
        req = body or PeopleIndexRequest()
        try:
            res = await anyio.to_thread.run_sync(
                lambda: svcs.people.index_session(
                    session_id,
                    threshold=req.threshold,
                )
            )
        except PeopleError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except FaceEmbedderError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        return res.to_dict()

    @app.get("/api/people")
    def list_people_groups(
        svcs: SvcsDep,
        session_id: Annotated[str | None, Query()] = None,
        group_person_id: Annotated[str | None, Query()] = None,
    ) -> dict[str, Any]:
        try:
            res = svcs.people.get_people_groups(
                session_id=session_id,
                group_person_id=group_person_id,
            )
        except PeopleError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        return res.to_dict()

    @app.post("/api/people/label")
    def label_people(body: PeopleLabelRequest, svcs: SvcsDep) -> dict[str, Any]:
        face_tuples = (
            [(f.content_fp, f.face_idx) for f in body.faces]
            if body.faces is not None
            else None
        )
        try:
            if body.action == "name":
                if not body.name:
                    raise PeopleError("`name` is required when action is 'name'.")
                person = svcs.people.label_name(
                    body.name,
                    cluster_id=body.cluster_id,
                    faces=face_tuples,
                    person_id=body.person_id,
                    is_self=body.is_self,
                    threshold=body.threshold,
                )
                return {
                    "action": "name",
                    "person": {
                        "id": person.id,
                        "name": person.name,
                        "is_self": person.is_self,
                    },
                }
            if body.action == "merge":
                merged = svcs.people.label_merge(
                    source_cluster_id=body.source_cluster_id,
                    source_cluster_ids=body.source_cluster_ids,
                    target_cluster_id=body.target_cluster_id,
                    source_person_id=body.source_person_id,
                    source_person_ids=body.source_person_ids,
                    target_person_id=body.target_person_id,
                    threshold=body.threshold,
                )
                return {"action": "merge", **merged}
            if body.action == "split":
                split_res = svcs.people.label_split(
                    faces=face_tuples or [],
                    threshold=body.threshold,
                )
                return {"action": "split", **split_res}
            if body.action == "reassign":
                reassign_res = svcs.people.label_reassign(
                    faces=face_tuples or [],
                    target_person_id=body.target_person_id,
                    target_cluster_id=body.target_cluster_id,
                    threshold=body.threshold,
                )
                return {"action": "reassign", **reassign_res}
            ignored_count = svcs.people.label_ignore(
                faces=face_tuples,
                cluster_id=body.cluster_id,
            )
            return {"action": "ignore", "ignored_count": ignored_count}
        except PeopleError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.post("/api/people/export")
    async def export_people_groups(
        body: PeopleExportRequest,
        svcs: SvcsDep,
    ) -> dict[str, Any]:
        try:
            res = await anyio.to_thread.run_sync(
                lambda: svcs.people.export_groups(
                    body.target_dir,
                    session_id=body.session_id,
                    group_keys=body.group_keys,
                    include_unnamed=body.include_unnamed,
                    include_no_face=body.include_no_face,
                    lang=body.lang,
                )
            )
        except PeopleError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except SafeFSError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        svcs.system.record_audit_log(
            surface="web",
            tool_name="people_export",
            hook_decision="allow",
            args_redacted={
                "target_dir": res.target_dir,
                "session_id": body.session_id,
                "group_count": len(res.groups),
            },
            summary=f"Exported {res.exported_photos} photos ({res.exported_files} files)",
        )
        return res.to_dict()

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

    # --- L3 VLM Batch Triage ---

    @app.post("/api/sessions/{session_id}/triage")
    async def triage_session(
        session_id: str,
        svcs: SvcsDep,
        body: TriageRequest | None = None,
    ) -> dict[str, Any]:
        req = body or TriageRequest()
        if req.dry_run:
            try:
                est = svcs.triage.estimate_triage(
                    session_id,
                    max_photos=req.max_photos,
                    profile=req.profile,
                    session_budget_usd=req.budget_usd,
                )
            except ValueError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from None
            except ModelError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
            return {
                "session_id": est.session_id,
                "dry_run": True,
                "yellow_count": est.yellow_count,
                "burst_count": est.burst_count,
                "total_yellow_available": est.total_yellow_available,
                "total_bursts_available": est.total_bursts_available,
                "skipped_bursts_due_to_limit": est.skipped_bursts_due_to_limit,
                "note": est.note,
                "total_units": est.total_units,
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
                "cost_classification": est.cost_classification,
                "requires_unpriced_confirmation": est.requires_unpriced_confirmation,
                "unpriced_max_photos": est.unpriced_max_photos,
            }

        # Validate session & credentials synchronously before spawning worker thread
        try:
            job_id = svcs.triage.create_queued_job(
                session_id,
                profile=req.profile,
                session_budget_usd=req.budget_usd,
                confirm_unpriced=req.confirm_unpriced,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except UnpricedTriageConfirmationRequiredError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except (MissingAPIKeyError, EndpointBindingError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except ModelError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from None

        def _execute_triage_in_thread(target_svcs: LightboxServices) -> dict[str, Any]:
            res = target_svcs.triage.run_triage_sync(
                session_id,
                max_photos=req.max_photos,
                profile=req.profile,
                session_budget_usd=req.budget_usd,
                job_id=job_id,
                confirm_unpriced=req.confirm_unpriced,
                on_progress=lambda stage, done, total: event_bus.publish(
                    "job.progress",
                    {"job_id": job_id, "stage": stage, "done": done, "total": total},
                    session_id=session_id,
                ),
            )
            event_bus.publish(
                "job.progress",
                {
                    "job_id": res.job_id,
                    "status": res.status,
                    "paused_reason": res.paused_reason,
                    "stage": "triage_vlm",
                    "done": res.reviewed_yellow + res.reviewed_bursts + res.failed_count,
                    "total": res.yellow_total + res.burst_total,
                    "failed_count": res.failed_count,
                },
                session_id=session_id,
            )
            return {
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
            }

        if req.wait:
            # Run synchronous Pillow/SQLite triage work off the FastAPI event loop thread
            try:
                return await anyio.to_thread.run_sync(_execute_triage_in_thread, svcs)
            except MissingAPIKeyError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
            except ModelError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from None

        def _bg_triage() -> None:
            with LightboxServices(app_paths=resolved_paths, migrate=False) as bg_svcs:
                if model_router_factory is not None:
                    bg_svcs.override_model_router(model_router_factory(bg_svcs))
                try:
                    _execute_triage_in_thread(bg_svcs)
                except Exception as exc:
                    bg_svcs.job_runner.mark_failed(job_id, f"{type(exc).__name__}: {exc}")
                    event_bus.publish(
                        "job.progress",
                        {
                            "job_id": job_id,
                            "status": "failed",
                            "paused_reason": f"{type(exc).__name__}: {exc}",
                            "stage": "triage_vlm",
                        },
                        session_id=session_id,
                    )

        threading.Thread(target=_bg_triage, daemon=True).start()
        return {
            "job_id": job_id,
            "session_id": session_id,
            "dry_run": False,
            "status": "queued",
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
    def get_plan(
        plan_id: str,
        svcs: SvcsDep,
        lang: Annotated[str, Query()] = "zh-CN",
    ) -> dict[str, Any]:
        try:
            plan, items = svcs.plan.get_plan(plan_id)
        except PlanError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        photo_reasons = svcs.review.get_photo_reasons_map(plan.session_id)
        return _serialize_plan(plan, items, photo_reasons=photo_reasons, lang=lang)

    @app.post("/api/sessions/{session_id}/plans")
    async def create_plan(
        session_id: str,
        body: CreatePlanRequest,
        svcs: SvcsDep,
    ) -> dict[str, Any]:
        custom_dir = Path(body.custom_dir) if body.custom_dir else None

        def _draft_and_verify() -> Any:
            draft = svcs.plan.create_plan(
                session_id,
                body.photo_ids,
                strategy=body.strategy,
                custom_dir=custom_dir,
            )
            return IndependentVerifier(svcs).verify_and_finalize_draft_sync(
                session_id,
                draft,
                skip_llm=body.skip_llm_verify,
                strategy=body.strategy,
                custom_dir=custom_dir,
            )

        try:
            draft = await anyio.to_thread.run_sync(_draft_and_verify)
        except PlanError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        if draft.plan is not None:
            event_bus.publish(
                "plan.updated",
                {"plan_id": draft.plan.id, "status": draft.plan.status},
                session_id=session_id,
            )
        return {
            "plan": _serialize_plan(draft.plan, draft.items) if draft.plan else None,
            "selected_count": draft.selected_count,
            "tech_reject_count": draft.tech_reject_count,
            "burst_non_pick_count": draft.burst_non_pick_count,
            "manual_reject_count": draft.manual_reject_count,
            "high_ratio_warning": draft.high_ratio_warning,
            "superseded_plan_ids": list(draft.superseded_plan_ids),
            "verifier_report": draft.verifier_report,
        }

    @app.post("/api/sessions/{session_id}/plans/obvious-rejects")
    async def draft_obvious_rejects(
        session_id: str,
        svcs: SvcsDep,
        body: DraftObviousRequest | None = None,
        lang: Annotated[str, Query()] = "zh-CN",
    ) -> dict[str, Any]:
        req = body or DraftObviousRequest()
        custom_dir = Path(req.custom_dir) if req.custom_dir else None

        def _draft_and_verify() -> tuple[Any, dict[str, tuple[str, tuple[str, ...]]] | None]:
            draft = svcs.plan.draft_obvious_rejects(
                session_id,
                strategy=req.strategy,
                custom_dir=custom_dir,
            )
            verified = IndependentVerifier(svcs).verify_and_finalize_draft_sync(
                session_id,
                draft,
                skip_llm=req.skip_llm_verify,
                strategy=req.strategy,
                custom_dir=custom_dir,
            )
            reasons = (
                svcs.review.get_photo_reasons_map(session_id)
                if verified.plan is not None
                else None
            )
            return verified, reasons

        try:
            draft, photo_reasons = await anyio.to_thread.run_sync(_draft_and_verify)
        except PlanError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
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

    # --- Single-Main Agent Sessions & Traces (PLAN T13, SPEC §7.1–7.3, §9.2) ---

    @app.get("/api/agent/sessions")
    def list_agent_sessions(
        svcs: SvcsDep,
        session_id: Annotated[str | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=50)] = 20,
    ) -> dict[str, Any]:
        rows = svcs.repo.list_agent_sessions(session_id=session_id, limit=limit)
        return {
            "items": [
                {
                    "agent_session_id": r.id,
                    "session_id": r.session_id,
                    "mode": r.mode,
                    "status": r.status,
                    "stage": str(r.stage_summary.get("stage", "explore")),
                    "step": r.step,
                    "spent_usd": round(r.spent_usd, 6),
                    "latest_user_message": str(r.stage_summary.get("latest_user_message", "")),
                    "created_at": r.created_at,
                }
                for r in rows
            ]
        }

    @app.get("/api/agent/sessions/{agent_session_id}")
    def get_agent_session(agent_session_id: str, svcs: SvcsDep) -> dict[str, Any]:
        runner = AgentRunner(
            svcs,
            base_url_resolver=lambda: f"http://127.0.0.1:{port}",
        )
        try:
            return runner.get_session(agent_session_id).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None

    @app.post("/api/agent/sessions")
    async def create_agent_session(
        body: CreateAgentSessionRequest,
        svcs: SvcsDep,
    ) -> dict[str, Any]:
        def _run() -> dict[str, Any]:
            runner = AgentRunner(
                svcs,
                base_url_resolver=lambda: f"http://127.0.0.1:{port}",
                lang=body.lang,
            )
            res = runner.start_session_sync(
                body.session_id,
                mode=body.mode,
                message=body.message,
                constraints=body.constraints,
                profile=body.profile,
                on_event=lambda _etype, payload: event_bus.publish(
                    "agent.event",
                    payload,
                    session_id=body.session_id,
                ),
            )
            return res.to_dict()

        try:
            return await anyio.to_thread.run_sync(_run)
        except ValueError as exc:
            msg = str(exc)
            status = 404 if msg.startswith("Unknown session_id") else 400
            raise HTTPException(status_code=status, detail=msg) from None

    @app.post("/api/agent/sessions/{agent_session_id}/messages")
    async def send_agent_message(
        agent_session_id: str,
        body: SendAgentMessageRequest,
        svcs: SvcsDep,
    ) -> dict[str, Any]:
        def _run() -> dict[str, Any]:
            runner = AgentRunner(
                svcs,
                base_url_resolver=lambda: f"http://127.0.0.1:{port}",
                lang=body.lang,
            )
            existing = runner.get_session(agent_session_id)
            res = runner.continue_session_sync(
                agent_session_id,
                body.message,
                approve_ask=body.approve_ask,
                constraints=body.constraints,
                profile=body.profile,
                on_event=lambda _etype, payload: event_bus.publish(
                    "agent.event",
                    payload,
                    session_id=existing.session_id,
                ),
            )
            return res.to_dict()

        try:
            return await anyio.to_thread.run_sync(_run)
        except AgentSessionBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except ValueError as exc:
            msg = str(exc)
            status = 404 if msg.startswith("Unknown agent_session_id") else 400
            raise HTTPException(status_code=status, detail=msg) from None

    @app.get("/api/insights/traces")
    def insights_traces(
        svcs: SvcsDep,
        session_id: Annotated[str | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> dict[str, Any]:
        return svcs.system.get_llm_traces(session_id=session_id, limit=limit)

    # --- Text-Rule Memory Loop (PLAN T14, SPEC §7.6, §9.2) ---

    @app.get("/api/memory/preferences")
    def get_memory_preferences(svcs: SvcsDep) -> dict[str, Any]:
        return MemoryManager(svcs).get_state()

    @app.put("/api/memory/preferences")
    def put_memory_preferences(body: SavePreferencesRequest, svcs: SvcsDep) -> dict[str, Any]:
        return MemoryManager(svcs).save_preferences(body.content)

    @app.post("/api/memory/consolidate")
    async def consolidate_memory(
        svcs: SvcsDep,
        body: ConsolidateMemoryRequest | None = None,
    ) -> dict[str, Any]:
        req = body or ConsolidateMemoryRequest()

        def _run() -> dict[str, Any]:
            mgr = MemoryManager(svcs)
            res = mgr.consolidate_sync(force=req.force, profile=req.profile)
            return {
                "result": res.to_dict(),
                "state": mgr.get_state(),
            }

        try:
            out = await anyio.to_thread.run_sync(_run)
        except MissingAPIKeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        except ModelError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from None

        if out["result"]["proposed_rules"]:
            event_bus.publish(
                "memory.proposed",
                {
                    "count": len(out["result"]["proposed_rules"]),
                    "rule_ids": [r["rule_id"] for r in out["result"]["proposed_rules"]],
                },
            )
        return out

    @app.post("/api/memory/rules/{rule_id}/decision")
    def decide_memory_rule(
        rule_id: str,
        body: MemoryRuleDecisionRequest,
        svcs: SvcsDep,
    ) -> dict[str, Any]:
        mgr = MemoryManager(svcs)
        try:
            updated = mgr.decide_rule(
                rule_id,
                decision=body.decision,
                rule_text=body.rule_text,
            )
        except ValueError as exc:
            msg = str(exc)
            status = 404 if msg.startswith("Unknown rule_id") else 400
            raise HTTPException(status_code=status, detail=msg) from None
        return {
            "rule": serialize_memory_rule(updated),
            "state": mgr.get_state(),
        }

    # --- Shooting Digest / Insights (PLAN T17, SPEC §9.2 & §13(2)) ---

    @app.get("/api/insights/digest")
    def insights_digest(
        svcs: SvcsDep,
        session_id: Annotated[str | None, Query()] = None,
        date_from: Annotated[str | None, Query()] = None,
        date_to: Annotated[str | None, Query()] = None,
        min_samples: Annotated[int, Query(ge=1, le=100)] = 5,
        lang: Annotated[str, Query()] = "zh-CN",
    ) -> dict[str, Any]:
        try:
            report = svcs.digest.generate_digest(
                session_id=session_id,
                date_from=date_from,
                date_to=date_to,
                min_samples=min_samples,
                lang=lang,
            )
        except ValueError as exc:
            msg = str(exc)
            status = 404 if msg.startswith("Unknown session_id") else 400
            raise HTTPException(status_code=status, detail=msg) from None
        return report.to_dict()

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
