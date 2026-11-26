"""In-memory MCP Client contract tests, Streamable HTTP Origin defense, and LocalUIBridge tests (SPEC §8, PLAN T09)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import mcp.types as types
import pytest
from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams
from PIL import Image
from starlette.testclient import TestClient

from lightbox.api import is_server_responsive, read_server_lock
from lightbox.mcp import (
    LightboxMCPBundle,
    LocalUIBridge,
    create_mcp_server,
    create_streamable_http_app,
)
from lightbox.paths import get_app_paths
from lightbox.services import LightboxServices, build_demo_dataset

EXPECTED_TOOL_NAMES: set[str] = {
    "lightbox_scan_folder",
    "lightbox_job_status",
    "lightbox_session_summary",
    "lightbox_list_photos",
    "lightbox_get_photo",
    "lightbox_compare_burst",
    "lightbox_set_verdict",
    "lightbox_recommend_burst_pick",
    "lightbox_triage_uncertain",
    "lightbox_search_photos",
    "lightbox_find_similar",
    "lightbox_create_plan",
    "lightbox_apply_plan",
    "lightbox_restore_plan",
}


@asynccontextmanager
async def _open_memory_mcp_client(
    bundle: LightboxMCPBundle,
) -> AsyncIterator[ClientSession]:
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        init_opts = bundle.server._lowlevel_server.create_initialization_options()

        async with anyio.create_task_group() as tg:
            tg.start_soon(
                bundle.server._lowlevel_server.run,
                server_read,
                server_write,
                init_opts,
                True,
            )
            async with ClientSession(client_read, client_write) as session:
                await session.initialize()
                try:
                    yield session
                finally:
                    tg.cancel_scope.cancel()


def _parse_first_json(result: types.CallToolResult) -> dict[str, Any]:
    assert result.content, "Expected at least one content block"
    first = result.content[0]
    assert isinstance(first, types.TextContent)
    parsed = json.loads(first.text)
    assert isinstance(parsed, dict)
    return parsed


def _assert_no_token_in_urls(payload: Any) -> None:
    if isinstance(payload, dict):
        for k, v in payload.items():
            if k.endswith("_url") and isinstance(v, str):
                lowered = v.lower()
                assert "token" not in lowered, f"Token leaked in {k}: {v}"
                assert "auth/launch" not in lowered, f"Launch auth leaked in {k}: {v}"
                assert "csrf" not in lowered, f"CSRF leaked in {k}: {v}"
            else:
                _assert_no_token_in_urls(v)
    elif isinstance(payload, list):
        for item in payload:
            _assert_no_token_in_urls(item)


def test_mcp_in_memory_full_contract_and_approval_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify all 14 tools, 4 resources, 3 prompts, tokenless URLs, and approval enforcement."""
    for env_var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr("lightbox.models.secrets._read_from_keyring", lambda _var: None)

    home = tmp_path / "lightbox_home"
    shoot_dir = tmp_path / "demo_shoot"
    build_demo_dataset(shoot_dir)

    paths = get_app_paths(home)
    bundle = create_mcp_server(app_paths=paths, auto_start_ui=False)

    async def _run() -> None:
        async with _open_memory_mcp_client(bundle) as client:
            # 1. Discover all 14 tools + verify MCP annotations
            tools_res = await client.list_tools()
            tool_map = {t.name: t for t in tools_res.tools}
            assert set(tool_map) == EXPECTED_TOOL_NAMES
            assert tool_map["lightbox_session_summary"].annotations is not None
            assert tool_map["lightbox_session_summary"].annotations.read_only_hint is True
            assert tool_map["lightbox_apply_plan"].annotations is not None
            assert tool_map["lightbox_apply_plan"].annotations.destructive_hint is True

            # 2. Discover all 4 Resources (2 static + 2 templates)
            static_res = await client.list_resources()
            tmpl_res = await client.list_resource_templates()
            static_uris = {str(r.uri) for r in static_res.resources}
            tmpl_uris = {str(r.uri_template) for r in tmpl_res.resource_templates}
            assert static_uris == {"lightbox://sessions", "lightbox://memory/preferences"}
            assert tmpl_uris == {"lightbox://session/{id}/summary", "lightbox://plan/{id}"}

            # 3. Discover all 3 Prompts and inspect prompt rendering
            prompts_res = await client.list_prompts()
            prompt_names = {p.name for p in prompts_res.prompts}
            assert prompt_names == {"cull-session", "curate-for-social", "shooting-digest"}
            p_out = await client.get_prompt("cull-session", {"folder_path": str(shoot_dir)})
            assert p_out.messages
            assert "lightbox_scan_folder" in str(p_out.messages[0].content)

            # 4. Scan folder via MCP
            scan_res = await client.call_tool(
                "lightbox_scan_folder",
                {"folder_path": str(shoot_dir), "wait_seconds": 30},
            )
            assert not scan_res.is_error
            scan_data = _parse_first_json(scan_res)
            _assert_no_token_in_urls(scan_data)
            session_id = str(scan_data["session_id"])
            job_id = str(scan_data["job_id"])
            assert scan_data["status"] == "completed"
            assert session_id in scan_data["review_url"]

            # 5. Job status & session summary
            job_res = await client.call_tool("lightbox_job_status", {"job_id": job_id})
            assert not job_res.is_error
            assert _parse_first_json(job_res)["status"] == "completed"

            sum_res = await client.call_tool(
                "lightbox_session_summary", {"session_id": session_id}
            )
            assert not sum_res.is_error
            sum_data = _parse_first_json(sum_res)
            _assert_no_token_in_urls(sum_data)
            assert sum_data["total_photos"] > 0

            # 6. Paginated photo listing & invalid limit rejection
            bad_list = await client.call_tool(
                "lightbox_list_photos", {"session_id": session_id, "limit": 99}
            )
            assert bad_list.is_error

            page1 = await client.call_tool(
                "lightbox_list_photos", {"session_id": session_id, "limit": 3}
            )
            assert not page1.is_error
            page1_data = _parse_first_json(page1)
            assert len(page1_data["items"]) == 3
            assert page1_data["next_cursor"] == "3"

            first_photo_id = str(page1_data["items"][0]["photo_id"])

            # 7. Get photo with EXIF-stripped <= 768px ImageContent
            photo_res = await client.call_tool(
                "lightbox_get_photo",
                {"photo_id": first_photo_id, "image": "thumb"},
            )
            assert not photo_res.is_error
            assert len(photo_res.content) == 2
            img_block = photo_res.content[1]
            assert isinstance(img_block, types.ImageContent)
            assert img_block.mime_type == "image/webp"
            raw_bytes = base64.b64decode(img_block.data)
            with Image.open(io.BytesIO(raw_bytes)) as im:
                assert max(im.size) <= 768
                assert not im.getexif()

            # 8. Burst comparison & non-destructive pick recommendation
            with LightboxServices(app_paths=paths, migrate=False) as svcs:
                bursts = svcs.burst.list_bursts(session_id)
            assert bursts, "Demo dataset should contain at least one burst group"
            burst_id = bursts[0].burst.id
            pick_pid = bursts[0].members[0].photo_id

            cmp_res = await client.call_tool("lightbox_compare_burst", {"group_id": burst_id})
            assert not cmp_res.is_error
            cmp_data = _parse_first_json(cmp_res)
            _assert_no_token_in_urls(cmp_data)
            assert len(cmp_data["frames"]) >= 2
            assert any(isinstance(b, types.ImageContent) for b in cmp_res.content)

            rec_res = await client.call_tool(
                "lightbox_recommend_burst_pick",
                {"group_id": burst_id, "photo_id": pick_pid, "reason": "Sharpest eyes in burst"},
            )
            assert not rec_res.is_error
            assert _parse_first_json(rec_res)["confirmed"] is False

            # 9. Set non-destructive verdict & protect exempt photos
            all_photos_res = await client.call_tool(
                "lightbox_list_photos", {"session_id": session_id, "limit": 30}
            )
            all_items = _parse_first_json(all_photos_res)["items"]
            normal_item = next(it for it in all_items if not it["is_protected"])
            verdict_res = await client.call_tool(
                "lightbox_set_verdict",
                {
                    "photo_ids": [normal_item["photo_id"]],
                    "verdict": "unsure",
                    "reason": "Needs closer look",
                },
            )
            assert not verdict_res.is_error
            assert _parse_first_json(verdict_res)["decision_version"] >= 2

            exempt_items = [it for it in all_items if it["is_protected"]]
            if exempt_items:
                deny_exempt = await client.call_tool(
                    "lightbox_set_verdict",
                    {"photo_ids": [exempt_items[0]["photo_id"]], "verdict": "reject"},
                )
                assert deny_exempt.is_error

            # 10. Triage dry_run & search & find_similar
            triage_dry = await client.call_tool(
                "lightbox_triage_uncertain", {"session_id": session_id, "dry_run": True}
            )
            assert not triage_dry.is_error
            assert _parse_first_json(triage_dry)["dry_run"] is True

            triage_live = await client.call_tool(
                "lightbox_triage_uncertain", {"session_id": session_id, "dry_run": False}
            )
            assert triage_live.is_error

            search_res = await client.call_tool(
                "lightbox_search_photos", {"query": "burst", "session_id": session_id}
            )
            assert not search_res.is_error
            _assert_no_token_in_urls(_parse_first_json(search_res))

            sim_res = await client.call_tool(
                "lightbox_find_similar", {"photo_id": pick_pid, "limit": 6}
            )
            assert not sim_res.is_error
            sim_data = _parse_first_json(sim_res)
            _assert_no_token_in_urls(sim_data)
            assert sim_data["anchor_id"] == pick_pid

            # 11. Draft plan (moves 0 files) -> unapproved apply MUST BE REJECTED!
            plan_res = await client.call_tool(
                "lightbox_create_plan",
                {"session_id": session_id, "action": "quarantine_move"},
            )
            assert not plan_res.is_error
            plan_data = _parse_first_json(plan_res)
            _assert_no_token_in_urls(plan_data)
            plan_id = str(plan_data["plan_id"])
            assert plan_data["count"] > 0

            # Capture original file paths before attempting apply
            with LightboxServices(app_paths=paths, migrate=False) as svcs:
                _plan_rec, plan_items = svcs.plan.get_plan(plan_id)
                src_files = [Path(p) for it in plan_items for p in it.src_paths]
            assert all(f.is_file() for f in src_files)

            # Unapproved call MUST fail cleanly and move 0 files
            unapproved = await client.call_tool("lightbox_apply_plan", {"plan_id": plan_id})
            assert unapproved.is_error
            unapproved_data = _parse_first_json(unapproved)
            _assert_no_token_in_urls(unapproved_data)
            assert all(f.is_file() for f in src_files)

            # 12. Read all 4 Resources via MCP
            sessions_blob = await client.read_resource("lightbox://sessions")
            assert session_id in str(sessions_blob.contents[0])
            prefs_blob = await client.read_resource("lightbox://memory/preferences")
            assert "Preferences" in str(prefs_blob.contents[0])
            sess_blob = await client.read_resource(f"lightbox://session/{session_id}/summary")
            _assert_no_token_in_urls(json.loads(sess_blob.contents[0].text))  # type: ignore[union-attr]
            plan_blob = await client.read_resource(f"lightbox://plan/{plan_id}")
            _assert_no_token_in_urls(json.loads(plan_blob.contents[0].text))  # type: ignore[union-attr]

            # 13. Approve via 6-digit CLI code -> apply_plan succeeds -> restore_plan reverses!
            with LightboxServices(app_paths=paths, migrate=False) as svcs:
                issued = svcs.plan.issue_cli_approval_code(plan_id)

            applied = await client.call_tool(
                "lightbox_apply_plan",
                {"plan_id": plan_id, "confirmation_code": issued.code},
            )
            assert not applied.is_error
            applied_data = _parse_first_json(applied)
            assert applied_data["moved_count"] == len(plan_items)
            assert all(not f.exists() for f in src_files)

            restored = await client.call_tool("lightbox_restore_plan", {"plan_id": plan_id})
            assert not restored.is_error
            restored_data = _parse_first_json(restored)
            assert restored_data["restored_count"] == len(plan_items)
            assert all(f.is_file() for f in src_files)

    asyncio.run(_run())


def test_streamable_http_enforces_origin_and_host_dns_rebinding_defense(tmp_path: Path) -> None:
    """Streamable HTTP MCP endpoint must reject forged Origin and Host headers (SPEC §8.1, §10.2(3))."""
    paths = get_app_paths(tmp_path / "home")
    bundle = create_mcp_server(app_paths=paths, auto_start_ui=False)
    app = create_streamable_http_app(bundle, port=8766)

    with TestClient(app, base_url="http://127.0.0.1:8766") as client:
        # Forged Origin header -> 403 Forbidden
        bad_origin = client.post(
            "/mcp",
            headers={
                "Host": "127.0.0.1:8766",
                "Origin": "https://evil.example",
                "Content-Type": "application/json",
            },
            content="{}",
        )
        assert bad_origin.status_code == 403

        # Forged Host header -> 421 Misdirected Request / 403 Forbidden
        bad_host = client.post(
            "/mcp",
            headers={
                "Host": "evil.example:8766",
                "Content-Type": "application/json",
            },
            content="{}",
        )
        assert bad_host.status_code in {403, 421}


def test_local_ui_bridge_autostarts_and_reuses_server_lock(tmp_path: Path) -> None:
    """LocalUIBridge starts an embedded 127.0.0.1 server on demand, writes server.lock, and reuses it."""
    paths = get_app_paths(tmp_path / "home")
    with LocalUIBridge(app_paths=paths, auto_start=True) as bridge:
        base_url = bridge.ensure_base_url()
        assert base_url.startswith("http://127.0.0.1:")
        assert "token" not in base_url

        lock_info = read_server_lock(app_paths=paths, check_alive=True, verify_http=True)
        assert lock_info is not None
        assert lock_info.base_url == base_url
        assert is_server_responsive(lock_info)

        # A second bridge should detect and reuse the same live server without spawning a new one
        bridge2 = LocalUIBridge(app_paths=paths, auto_start=True)
        assert bridge2.ensure_base_url() == base_url
        assert bridge2.owned_lock_info is None

    # After context exit, server.lock is cleared
    assert read_server_lock(app_paths=paths, check_alive=True, verify_http=False) is None
