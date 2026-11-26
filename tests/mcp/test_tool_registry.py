"""Unit tests for unified ToolRegistry schemas, risk gating, validation, and redaction (SPEC §7.3, §8.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lightbox.services import LightboxServices
from lightbox.tools import (
    ToolContext,
    ToolRegistry,
    assert_tokenless_url,
    redact_tool_args,
)

EXPECTED_MCP_TOOLS: set[str] = {
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


def test_registry_has_exactly_14_mcp_tools_in_deterministic_order() -> None:
    reg = ToolRegistry()
    mcp_specs = reg.mcp_specs()
    assert len(mcp_specs) == 14
    assert {s.mcp_name for s in mcp_specs} == EXPECTED_MCP_TOOLS
    names = [s.name for s in reg.list_specs()]
    assert names == sorted(names)


def test_agent_schemas_strictly_exclude_apply_and_restore_plan() -> None:
    reg = ToolRegistry()
    default_names = {s["name"] for s in reg.agent_schemas()}
    assert "apply_plan" not in default_names
    assert "restore_plan" not in default_names
    assert "create_plan" in default_names

    # Even if caller explicitly requests apply_plan, agent_schemas must refuse to expose it
    forced = {s["name"] for s in reg.agent_schemas(["apply_plan", "restore_plan", "create_plan"])}
    assert forced == {"create_plan"}


def test_agent_surface_cannot_execute_apply_plan(tmp_path: Path) -> None:
    reg = ToolRegistry()
    with LightboxServices.open(tmp_path / "home") as svcs:
        ctx = ToolContext(services=svcs, surface="agent")
        res = reg.execute("apply_plan", {"plan_id": "plan_dummy"}, ctx)
        assert res.is_error is True
        assert "forbidden" in (res.error_message or "").lower()


def test_strict_pydantic_argument_validation() -> None:
    reg = ToolRegistry()
    with pytest.raises(ValidationError):
        reg.validate_args("lightbox_list_photos", {"session_id": "s1", "limit": 50})
    with pytest.raises(ValidationError):
        reg.validate_args("lightbox_search_photos", {"query": "cat", "limit": 25})
    with pytest.raises(ValidationError):
        reg.validate_args("lightbox_find_similar", {"photo_id": "p1", "limit": 20})
    with pytest.raises(ValidationError):
        reg.validate_args("lightbox_session_summary", {"session_id": "s1", "extra_field": 1})
    with pytest.raises(ValidationError):
        reg.validate_args("lightbox_apply_plan", {"plan_id": "p1", "confirmation_code": "123"})


def test_tokenless_url_and_arg_redaction() -> None:
    assert assert_tokenless_url("http://127.0.0.1:8765/sessions/sess_1/review")
    for bad in (
        "http://127.0.0.1:8765/auth/launch?token=abc",
        "http://127.0.0.1:8765/sessions/s1/review?token=abc",
        "http://127.0.0.1:8765/sessions/s1/review?csrf=xyz",
    ):
        with pytest.raises(ValueError, match="tokenless"):
            assert_tokenless_url(bad)

    redacted = redact_tool_args(
        {
            "plan_id": "plan_1",
            "confirmation_code": "654321",
            "api_key": "sk-secret",
            "photo_ids": ["p1", "p2"],
        }
    )
    assert redacted["confirmation_code"] == "***REDACTED***"
    assert redacted["api_key"] == "***REDACTED***"
    assert redacted["plan_id"] == "plan_1"


def test_pre_hook_can_block_execution(tmp_path: Path) -> None:
    reg = ToolRegistry()
    reg.add_pre_hook(lambda spec, _args, _ctx: ("deny", f"hook blocked {spec.name}"))
    with LightboxServices.open(tmp_path / "home") as svcs:
        ctx = ToolContext(services=svcs, surface="mcp:test")
        res = reg.execute("lightbox_session_summary", {"session_id": "sess_1"}, ctx)
        assert res.is_error is True
        assert "hook blocked" in (res.error_message or "")
