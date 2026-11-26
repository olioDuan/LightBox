"""Lightbox MCP Server: 14 tools, 4 resources, 3 prompts over stdio or 127.0.0.1 Streamable HTTP (SPEC §8)."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal

import anyio
import mcp.types as types
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field
from starlette.applications import Starlette

from lightbox import __version__
from lightbox.mcp.ui_bridge import LocalUIBridge
from lightbox.paths import AppPaths, get_app_paths
from lightbox.services import LightboxServices, PlanError
from lightbox.tools import (
    ToolContext,
    ToolExecutionResult,
    ToolRegistry,
    assert_tokenless_url,
)

MCPTransport = Literal["stdio", "http"]
DEFAULT_MCP_HTTP_PORT = 8766


class _PlanApprovalElicitation(BaseModel):
    """MCP Elicitation confirmation form for human approval of a quarantine plan."""

    confirm: bool = Field(
        default=False,
        description="Set true to explicitly approve moving these photos to the reversible quarantine folder.",
    )
    photo_count: int = Field(
        ge=1,
        description="Enter the exact number of photo groups shown in the prompt to confirm.",
    )


@dataclass(slots=True)
class LightboxMCPBundle:
    """Container pairing the configured MCPServer instance with its ToolRegistry and LocalUIBridge."""

    server: MCPServer[Any]
    registry: ToolRegistry
    ui_bridge: LocalUIBridge
    app_paths: AppPaths


def _result_to_mcp(result: ToolExecutionResult) -> types.CallToolResult:
    content: list[types.ContentBlock] = [
        types.TextContent(
            type="text",
            text=json.dumps(result.data, ensure_ascii=False, indent=2),
        )
    ]
    for img in result.images:
        content.append(
            types.ImageContent(
                type="image",
                data=base64.b64encode(img.data).decode("ascii"),
                mime_type=img.mime_type,
            )
        )
    return types.CallToolResult(content=content, is_error=result.is_error)


def _client_surface_name(ctx: Context[Any, Any] | None) -> str:
    if ctx is None:
        return "mcp:unknown"
    try:
        params = ctx.session.client_params
        client_info = getattr(params, "client_info", None) or getattr(params, "clientInfo", None)
        if client_info is not None and getattr(client_info, "name", None):
            cleaned = str(client_info.name).strip().lower().replace(" ", "-")[:40]
            return f"mcp:{cleaned}"
    except Exception:
        pass
    return "mcp:client"


def _make_tool_annotations(registry: ToolRegistry, mcp_name: str) -> types.ToolAnnotations:
    spec = registry.get(mcp_name)
    return types.ToolAnnotations(
        read_only_hint=spec.read_only_hint,
        destructive_hint=spec.destructive_hint,
        idempotent_hint=spec.idempotent_hint,
        open_world_hint=False,
    )


def create_mcp_server(
    *,
    app_paths: AppPaths | None = None,
    registry: ToolRegistry | None = None,
    auto_start_ui: bool = True,
    fallback_base_url: str = "http://127.0.0.1:8765",
) -> LightboxMCPBundle:
    """Build the official MCP SDK server exposing all 14 Lightbox tools, 4 resources, and 3 prompts."""
    resolved_paths = app_paths or get_app_paths()
    resolved_paths.ensure_dirs()
    # Ensure SQLite schema is migrated once upfront
    with LightboxServices(app_paths=resolved_paths, migrate=True):
        pass

    reg = registry or ToolRegistry()
    ui_bridge = LocalUIBridge(
        app_paths=resolved_paths,
        auto_start=auto_start_ui,
        fallback_base_url=fallback_base_url,
    )

    @asynccontextmanager
    async def _lifespan(_srv: MCPServer[Any]) -> AsyncIterator[None]:
        try:
            yield
        finally:
            ui_bridge.stop()

    mcp = MCPServer(
        name="lightbox",
        version=__version__,
        instructions=(
            "Lightbox local-first photo culling MCP server. "
            "For local photo folders: start with lightbox_scan_folder, read lightbox://memory/preferences "
            "before subjective culling, direct >8 photo reviews to the tokenless review_url in Web UI, "
            "and never rm/mv photos directly — always draft with lightbox_create_plan and require human approval."
        ),
        lifespan=_lifespan,
    )

    async def _run_tool(
        mcp_name: str,
        arguments: dict[str, Any],
        ctx: Context[Any, Any] | None = None,
        *,
        allow_progress: bool = False,
        allow_elicitation: bool = False,
    ) -> types.CallToolResult:
        surface = _client_surface_name(ctx)
        loop = asyncio.get_running_loop()

        progress_cb = None
        if allow_progress and ctx is not None:
            def _report(stage: str, done: int, total: int) -> None:
                if total <= 0:
                    return
                try:
                    asyncio.run_coroutine_threadsafe(
                        ctx.report_progress(
                            progress=float(done),
                            total=float(total),
                            message=f"{stage}: {done}/{total}",
                        ),
                        loop,
                    )
                except Exception:
                    pass

            progress_cb = _report

        elicitation_approver = None
        if allow_elicitation and ctx is not None:
            caps = None
            try:
                caps = ctx.client_capabilities
            except Exception:
                caps = None
            if caps is not None and getattr(caps, "elicitation", None) is not None:
                def _ask_elicit(plan_id: str, photo_count: int, quarantine_dir: str | None) -> bool:
                    prompt = (
                        f"Approve Lightbox quarantine plan {plan_id}?\n"
                        f"Photo groups to move: {photo_count}\n"
                        f"Reversible destination: {quarantine_dir or 'Lightbox_Quarantine'}\n"
                        "Originals are never deleted and can always be restored."
                    )
                    try:
                        fut = asyncio.run_coroutine_threadsafe(
                            ctx.elicit(prompt, _PlanApprovalElicitation),
                            loop,
                        )
                        elicit_res = fut.result(timeout=120.0)
                        if getattr(elicit_res, "action", None) == "accept":
                            data = getattr(elicit_res, "data", None)
                            if (
                                data is not None
                                and bool(getattr(data, "confirm", False))
                                and int(getattr(data, "photo_count", -1)) == photo_count
                            ):
                                return True
                    except Exception:
                        return False
                    return False

                elicitation_approver = _ask_elicit

        def _execute_sync() -> ToolExecutionResult:
            with LightboxServices(app_paths=resolved_paths, migrate=False) as svcs:
                tctx = ToolContext(
                    services=svcs,
                    surface=surface,
                    base_url_resolver=ui_bridge.ensure_base_url,
                    progress_callback=progress_cb,
                    elicitation_approver=elicitation_approver,
                )
                return reg.execute(mcp_name, arguments, tctx)

        result = await anyio.to_thread.run_sync(_execute_sync)
        return _result_to_mcp(result)

    # -----------------------------------------------------------------------
    # 14 MCP Tools (SPEC §8.2)
    # -----------------------------------------------------------------------

    @mcp.tool(
        name="lightbox_scan_folder",
        description=reg.get("lightbox_scan_folder").description,
        annotations=_make_tool_annotations(reg, "lightbox_scan_folder"),
        structured_output=False,
    )
    async def lightbox_scan_folder(
        folder_path: str,
        sensitivity: Literal["conservative", "standard"] = "conservative",
        enable_embed: bool = False,
        wait_seconds: int = 15,
        ctx: Context[Any, Any] | None = None,
    ) -> types.CallToolResult:
        return await _run_tool(
            "lightbox_scan_folder",
            {
                "folder_path": folder_path,
                "sensitivity": sensitivity,
                "enable_embed": enable_embed,
                "wait_seconds": wait_seconds,
            },
            ctx,
            allow_progress=True,
        )

    @mcp.tool(
        name="lightbox_job_status",
        description=reg.get("lightbox_job_status").description,
        annotations=_make_tool_annotations(reg, "lightbox_job_status"),
        structured_output=False,
    )
    async def lightbox_job_status(
        job_id: str,
        ctx: Context[Any, Any] | None = None,
    ) -> types.CallToolResult:
        return await _run_tool("lightbox_job_status", {"job_id": job_id}, ctx)

    @mcp.tool(
        name="lightbox_session_summary",
        description=reg.get("lightbox_session_summary").description,
        annotations=_make_tool_annotations(reg, "lightbox_session_summary"),
        structured_output=False,
    )
    async def lightbox_session_summary(
        session_id: str,
        ctx: Context[Any, Any] | None = None,
    ) -> types.CallToolResult:
        return await _run_tool("lightbox_session_summary", {"session_id": session_id}, ctx)

    @mcp.tool(
        name="lightbox_list_photos",
        description=reg.get("lightbox_list_photos").description,
        annotations=_make_tool_annotations(reg, "lightbox_list_photos"),
        structured_output=False,
    )
    async def lightbox_list_photos(
        session_id: str,
        bucket: Literal["red", "yellow", "normal", "exempt", "unreviewed", "kept"] | None = None,
        reason: str | None = None,
        sort: Literal["uncertainty", "name"] = "uncertainty",
        limit: int = 30,
        cursor: str | None = None,
        ctx: Context[Any, Any] | None = None,
    ) -> types.CallToolResult:
        return await _run_tool(
            "lightbox_list_photos",
            {
                "session_id": session_id,
                "bucket": bucket,
                "reason": reason,
                "sort": sort,
                "limit": limit,
                "cursor": cursor,
            },
            ctx,
        )

    @mcp.tool(
        name="lightbox_get_photo",
        description=reg.get("lightbox_get_photo").description,
        annotations=_make_tool_annotations(reg, "lightbox_get_photo"),
        structured_output=False,
    )
    async def lightbox_get_photo(
        photo_id: str,
        image: Literal["none", "thumb", "face_crop", "sharpest_crop"] = "none",
        ctx: Context[Any, Any] | None = None,
    ) -> types.CallToolResult:
        return await _run_tool(
            "lightbox_get_photo",
            {"photo_id": photo_id, "image": image},
            ctx,
        )

    @mcp.tool(
        name="lightbox_compare_burst",
        description=reg.get("lightbox_compare_burst").description,
        annotations=_make_tool_annotations(reg, "lightbox_compare_burst"),
        structured_output=False,
    )
    async def lightbox_compare_burst(
        group_id: str,
        ctx: Context[Any, Any] | None = None,
    ) -> types.CallToolResult:
        return await _run_tool("lightbox_compare_burst", {"group_id": group_id}, ctx)

    @mcp.tool(
        name="lightbox_set_verdict",
        description=reg.get("lightbox_set_verdict").description,
        annotations=_make_tool_annotations(reg, "lightbox_set_verdict"),
        structured_output=False,
    )
    async def lightbox_set_verdict(
        photo_ids: list[str],
        verdict: Literal["keep", "reject", "unsure"],
        reason: str | None = None,
        ctx: Context[Any, Any] | None = None,
    ) -> types.CallToolResult:
        return await _run_tool(
            "lightbox_set_verdict",
            {"photo_ids": photo_ids, "verdict": verdict, "reason": reason},
            ctx,
        )

    @mcp.tool(
        name="lightbox_recommend_burst_pick",
        description=reg.get("lightbox_recommend_burst_pick").description,
        annotations=_make_tool_annotations(reg, "lightbox_recommend_burst_pick"),
        structured_output=False,
    )
    async def lightbox_recommend_burst_pick(
        group_id: str,
        photo_id: str,
        reason: str,
        ctx: Context[Any, Any] | None = None,
    ) -> types.CallToolResult:
        return await _run_tool(
            "lightbox_recommend_burst_pick",
            {"group_id": group_id, "photo_id": photo_id, "reason": reason},
            ctx,
        )

    @mcp.tool(
        name="lightbox_triage_uncertain",
        description=reg.get("lightbox_triage_uncertain").description,
        annotations=_make_tool_annotations(reg, "lightbox_triage_uncertain"),
        structured_output=False,
    )
    async def lightbox_triage_uncertain(
        session_id: str,
        max_photos: int = 50,
        dry_run: bool = True,
        ctx: Context[Any, Any] | None = None,
    ) -> types.CallToolResult:
        return await _run_tool(
            "lightbox_triage_uncertain",
            {"session_id": session_id, "max_photos": max_photos, "dry_run": dry_run},
            ctx,
        )

    @mcp.tool(
        name="lightbox_search_photos",
        description=reg.get("lightbox_search_photos").description,
        annotations=_make_tool_annotations(reg, "lightbox_search_photos"),
        structured_output=False,
    )
    async def lightbox_search_photos(
        query: str,
        session_id: str | None = None,
        date_from: str | None = None,
        camera: str | None = None,
        limit: int = 20,
        ctx: Context[Any, Any] | None = None,
    ) -> types.CallToolResult:
        return await _run_tool(
            "lightbox_search_photos",
            {
                "query": query,
                "session_id": session_id,
                "date_from": date_from,
                "camera": camera,
                "limit": limit,
            },
            ctx,
        )

    @mcp.tool(
        name="lightbox_find_similar",
        description=reg.get("lightbox_find_similar").description,
        annotations=_make_tool_annotations(reg, "lightbox_find_similar"),
        structured_output=False,
    )
    async def lightbox_find_similar(
        photo_id: str,
        limit: int = 12,
        ctx: Context[Any, Any] | None = None,
    ) -> types.CallToolResult:
        return await _run_tool(
            "lightbox_find_similar",
            {"photo_id": photo_id, "limit": limit},
            ctx,
        )

    @mcp.tool(
        name="lightbox_create_plan",
        description=reg.get("lightbox_create_plan").description,
        annotations=_make_tool_annotations(reg, "lightbox_create_plan"),
        structured_output=False,
    )
    async def lightbox_create_plan(
        session_id: str,
        action: Literal["quarantine_move", "export_clean"] = "quarantine_move",
        include_bursts: Literal["confirmed_only", "none"] = "confirmed_only",
        photo_ids: list[str] | None = None,
        ctx: Context[Any, Any] | None = None,
    ) -> types.CallToolResult:
        return await _run_tool(
            "lightbox_create_plan",
            {
                "session_id": session_id,
                "action": action,
                "include_bursts": include_bursts,
                "photo_ids": photo_ids,
            },
            ctx,
        )

    @mcp.tool(
        name="lightbox_apply_plan",
        description=reg.get("lightbox_apply_plan").description,
        annotations=_make_tool_annotations(reg, "lightbox_apply_plan"),
        structured_output=False,
    )
    async def lightbox_apply_plan(
        plan_id: str,
        confirmation_code: str | None = None,
        ctx: Context[Any, Any] | None = None,
    ) -> types.CallToolResult:
        return await _run_tool(
            "lightbox_apply_plan",
            {"plan_id": plan_id, "confirmation_code": confirmation_code},
            ctx,
            allow_elicitation=True,
        )

    @mcp.tool(
        name="lightbox_restore_plan",
        description=reg.get("lightbox_restore_plan").description,
        annotations=_make_tool_annotations(reg, "lightbox_restore_plan"),
        structured_output=False,
    )
    async def lightbox_restore_plan(
        plan_id: str,
        photo_ids: list[str] | None = None,
        ctx: Context[Any, Any] | None = None,
    ) -> types.CallToolResult:
        return await _run_tool(
            "lightbox_restore_plan",
            {"plan_id": plan_id, "photo_ids": photo_ids},
            ctx,
        )

    # -----------------------------------------------------------------------
    # 4 MCP Resources (SPEC §8.3)
    # -----------------------------------------------------------------------

    @mcp.resource(
        "lightbox://sessions",
        name="sessions",
        description="List of all scanned Lightbox photo sessions.",
        mime_type="application/json",
    )
    def resource_sessions() -> str:
        with LightboxServices(app_paths=resolved_paths, migrate=False) as svcs:
            sessions = svcs.review.list_sessions()
            base = ui_bridge.ensure_base_url()
            payload = {
                "sessions": [
                    {
                        "session_id": s.id,
                        "name": s.name,
                        "root_path": s.root_path,
                        "sensitivity": s.sensitivity,
                        "created_at": s.created_at,
                        "last_scanned_at": s.last_scanned_at,
                        "review_url": assert_tokenless_url(f"{base}/sessions/{s.id}/review"),
                    }
                    for s in sessions
                ]
            }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @mcp.resource(
        "lightbox://session/{id}/summary",
        name="session_summary",
        description="Detailed JSON summary for a scanned Lightbox session.",
        mime_type="application/json",
    )
    def resource_session_summary(id: str) -> str:
        with LightboxServices(app_paths=resolved_paths, migrate=False) as svcs:
            tctx = ToolContext(
                services=svcs,
                surface="mcp:resource",
                base_url_resolver=ui_bridge.ensure_base_url,
            )
            res = reg.execute("lightbox_session_summary", {"session_id": id}, tctx)
            if res.is_error:
                raise ValueError(res.error_message or f"Unknown session_id: {id}")
            return json.dumps(res.data, ensure_ascii=False, indent=2)

    @mcp.resource(
        "lightbox://memory/preferences",
        name="preferences",
        description="Active user culling preferences from <APP_DATA_DIR>/preferences.md.",
        mime_type="text/markdown",
    )
    def resource_preferences() -> str:
        with LightboxServices(app_paths=resolved_paths, migrate=False) as svcs:
            return svcs.review.read_preferences_markdown()

    @mcp.resource(
        "lightbox://plan/{id}",
        name="plan_detail",
        description="Pending or executed quarantine plan details, items, and Verifier report.",
        mime_type="application/json",
    )
    def resource_plan(id: str) -> str:
        with LightboxServices(app_paths=resolved_paths, migrate=False) as svcs:
            try:
                plan, items = svcs.plan.get_plan(id)
            except PlanError as exc:
                raise ValueError(str(exc)) from exc
            tctx = ToolContext(
                services=svcs,
                surface="mcp:resource",
                base_url_resolver=ui_bridge.ensure_base_url,
            )
            report = json.loads(plan.verifier_report_json) if plan.verifier_report_json else None
            payload = {
                "plan_id": plan.id,
                "session_id": plan.session_id,
                "action_type": plan.action_type,
                "plan_hash": plan.plan_hash,
                "status": plan.status,
                "quarantine_dir": plan.quarantine_dir,
                "approved_by": plan.approved_by,
                "approved_at": plan.approved_at,
                "verifier_report": report,
                "approve_url": tctx.approve_url(plan.session_id, plan.id),
                "item_count": len(items),
                "items": [
                    {
                        "photo_id": it.photo_id,
                        "decision_version": it.decision_version,
                        "src_paths": list(it.src_paths),
                        "dst_paths": list(it.dst_paths),
                        "status": it.status,
                    }
                    for it in items
                ],
            }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------------
    # 3 MCP Prompts (SPEC §8.3)
    # -----------------------------------------------------------------------

    @mcp.prompt(
        name="cull-session",
        description="Scan a local photo folder, review uncertain/burst shots, draft a quarantine plan, and guide human approval.",
    )
    def prompt_cull_session(folder_path: str) -> str:
        return (
            f"Help the photographer cull the local photo folder: {folder_path}\n\n"
            "Follow the Lightbox local-folder SOP strictly:\n"
            "1. Call `lightbox_scan_folder` on this folder first (never write shell/Python scripts to walk the directory).\n"
            "2. Read `lightbox://memory/preferences` before making subjective judgments.\n"
            "3. Inspect summary counts. Review up to 8 uncertain (`yellow`) photos or burst groups via "
            "`lightbox_list_photos` / `lightbox_get_photo` / `lightbox_compare_burst`. If more than 8 photos need manual inspection, "
            "direct the user to the tokenless `review_url` in the local Web UI.\n"
            "4. Record non-destructive suggestions with `lightbox_set_verdict` or `lightbox_recommend_burst_pick`.\n"
            "5. Draft a reversible quarantine plan via `lightbox_create_plan` and present the tokenless `approve_url` "
            "(or `lightbox approve <plan_id>`) for explicit human approval. Never run rm/mv directly."
        )

    @mcp.prompt(
        name="curate-for-social",
        description="Pick N standout, non-duplicate photos from a scanned session matching a requested vibe.",
    )
    def prompt_curate_for_social(
        session_id: str,
        count: int = 3,
        vibe: str = "natural",
    ) -> str:
        return (
            f"Select {count} standout photos from Lightbox session `{session_id}` matching vibe: {vibe}.\n\n"
            "Workflow:\n"
            "1. Read `lightbox://memory/preferences` and `lightbox://session/{id}/summary`.\n"
            "2. Use `lightbox_list_photos` (focus on `normal` / `exempt` / promising `yellow`) and `lightbox_search_photos` "
            "to shortlist candidates.\n"
            "3. Use `lightbox_find_similar` to avoid near-duplicate picks; inspect top candidates via `lightbox_get_photo` "
            "(keep direct image fetches <= 8, otherwise point the user to `review_url`).\n"
            "4. Return the selected filenames/photo IDs, concrete reasons for each pick, and any borderline/uncertain runner-ups."
        )

    @mcp.prompt(
        name="shooting-digest",
        description="Generate an objective shooting digest separating technical failures from burst non-picks.",
    )
    def prompt_shooting_digest(session_id: str | None = None) -> str:
        scope = f"session `{session_id}`" if session_id else "all recent sessions in `lightbox://sessions`"
        return (
            f"Create an objective shooting digest for {scope}.\n\n"
            "Guidelines:\n"
            "1. Read `lightbox://sessions` and `lightbox://session/{id}/summary`.\n"
            "2. Strictly separate technical failures (`motion_blur`, `out_of_focus`, `severe_blur`, `extreme_exposure`) "
            "from qualified burst non-picks (`burst_non_pick`) — never inflate technical failure rates with sharp burst duplicates.\n"
            "3. Report actual counts AND percentages for each bucket and top failure reason.\n"
            "4. Summarize factual patterns neutrally without lecturing or giving preachy tutorials."
        )

    return LightboxMCPBundle(
        server=mcp,
        registry=reg,
        ui_bridge=ui_bridge,
        app_paths=resolved_paths,
    )


def build_mcp_transport_security(port: int) -> TransportSecuritySettings:
    """Strict 127.0.0.1 Host & Origin DNS-rebinding defense for Streamable HTTP (SPEC §8.1, §10.2(3))."""
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[f"127.0.0.1:{port}", f"localhost:{port}"],
        allowed_origins=[f"http://127.0.0.1:{port}", f"http://localhost:{port}"],
    )


def create_streamable_http_app(
    bundle: LightboxMCPBundle,
    *,
    port: int = DEFAULT_MCP_HTTP_PORT,
    streamable_http_path: str = "/mcp",
) -> Starlette:
    """Create the local-only (127.0.0.1) Streamable HTTP Starlette app with strict Origin checks."""
    return bundle.server.streamable_http_app(
        streamable_http_path=streamable_http_path,
        host="127.0.0.1",
        transport_security=build_mcp_transport_security(port),
    )


def run_mcp_server(
    *,
    transport: MCPTransport = "stdio",
    port: int = DEFAULT_MCP_HTTP_PORT,
    app_paths: AppPaths | None = None,
    auto_start_ui: bool = True,
) -> None:
    """Run Lightbox MCP Server over `stdio` (default) or local-only `127.0.0.1` Streamable HTTP."""
    bundle = create_mcp_server(app_paths=app_paths, auto_start_ui=auto_start_ui)
    try:
        if transport == "stdio":
            bundle.server.run(transport="stdio")
        else:
            bundle.server.run(
                transport="streamable-http",
                host="127.0.0.1",
                port=port,
                transport_security=build_mcp_transport_security(port),
            )
    finally:
        bundle.ui_bridge.stop()


__all__ = [
    "DEFAULT_MCP_HTTP_PORT",
    "LightboxMCPBundle",
    "MCPTransport",
    "build_mcp_transport_security",
    "create_mcp_server",
    "create_streamable_http_app",
    "run_mcp_server",
]
