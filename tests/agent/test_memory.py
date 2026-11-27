"""Unit & integration tests for the T14 text-rule memory loop (SPEC §7.6, PLAN T14)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from lightbox.agent import AgentRunner, MemoryManager
from lightbox.api import create_app
from lightbox.models import (
    GenerationResult,
    Message,
    ModelCaps,
    ModelRouter,
    ToolSchema,
    Usage,
)
from lightbox.services import LightboxServices, build_demo_dataset
from lightbox.tools import ToolContext, ToolRegistry


class _FakeConsolidateModel:
    id = "claude-sonnet-5-5"
    provider = "anthropic"
    family = "anthropic"
    caps = ModelCaps(vision=True, tools=True, json_schema=True, prompt_cache=True)

    def __init__(self, rules_payload: list[dict[str, Any]]) -> None:
        self.rules_payload = list(rules_payload)
        self.received_messages: list[list[Message]] = []

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        response_schema: type[BaseModel] | None = None,
        max_tokens: int = 2048,
        effort: str | None = None,
    ) -> GenerationResult:
        _ = (tools, max_tokens, effort)
        self.received_messages.append(list(messages))
        payload = {"rules": self.rules_payload}
        parsed: Any = (
            response_schema.model_validate(payload) if response_schema is not None else None
        )
        return GenerationResult(
            text=json.dumps(payload, ensure_ascii=False),
            parsed=parsed,
            usage=Usage(in_tokens=300, out_tokens=90),
            cost_usd=0.001,
            provider=self.provider,
            model=self.id,
        )


class _RecordingAgentModel:
    id = "claude-sonnet-5-5"
    provider = "anthropic"
    family = "anthropic"
    caps = ModelCaps(vision=True, tools=True, json_schema=True, prompt_cache=True)

    def __init__(self) -> None:
        self.received_messages: list[list[Message]] = []

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        response_schema: type[BaseModel] | None = None,
        max_tokens: int = 2048,
        effort: str | None = None,
    ) -> GenerationResult:
        _ = (tools, response_schema, max_tokens, effort)
        self.received_messages.append(list(messages))
        return GenerationResult(
            text="Acknowledged.",
            usage=Usage(in_tokens=100, out_tokens=10),
            cost_usd=0.0,
            provider=self.provider,
            model=self.id,
        )


class _RecordingTriageModel:
    id = "claude-haiku-5-5"
    provider = "anthropic"
    family = "anthropic"
    caps = ModelCaps(vision=True, tools=True, json_schema=True, prompt_cache=True)

    def __init__(self) -> None:
        self.received_messages: list[list[Message]] = []

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        response_schema: type[BaseModel] | None = None,
        max_tokens: int = 2048,
        effort: str | None = None,
    ) -> GenerationResult:
        _ = (tools, max_tokens, effort)
        self.received_messages.append(list(messages))
        payload = {
            "decision": "keep",
            "confidence": 0.85,
            "reason_code": "vlm_keep",
            "rationale": "Matches user style.",
        }
        parsed: Any = (
            response_schema.model_validate(payload) if response_schema is not None else None
        )
        return GenerationResult(
            text=json.dumps(payload),
            parsed=parsed,
            usage=Usage(in_tokens=200, out_tokens=40),
            cost_usd=0.0,
            provider=self.provider,
            model=self.id,
        )


def _clear_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "LIGHTBOX_LOCAL_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("lightbox.models.secrets._read_from_keyring", lambda _var: None)


def test_memory_proposed_never_leaks_and_accepted_injects_everywhere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PLAN T14 core acceptance: proposed rules NEVER enter prompts; accepted rules appear in preferences.md, Agent, VLM, and MCP."""
    _clear_keys(monkeypatch)
    shoot_dir = tmp_path / "shoot"
    build_demo_dataset(shoot_dir)

    with LightboxServices.open(tmp_path / "home") as svcs:
        scan = svcs.scan.scan_folder(shoot_dir, max_workers=1)
        photos = [p for p in svcs.repo.list_photos(scan.session_id) if not p.is_protected]

        # Record 4 genuine overrides (< 5 threshold) -> non-forced consolidate skips cleanly without calling LLM
        for p in photos[:4]:
            cur = svcs.repo.get_effective_verdict(p.id)
            target_bucket = "normal" if (cur is not None and cur.bucket != "normal") else "red"
            svcs.review.set_verdict(
                p.id,
                bucket=target_bucket,
                source="user",
                reason_tag="intentional_blur",
                actor="web:user",
            )

        proposed_text_1 = "Keep intentional slow-shutter street motion blur frames"
        proposed_text_2 = "Reject backlit portraits with blown highlights"
        fake_consolidate = _FakeConsolidateModel(
            [
                {
                    "rule_text": proposed_text_1,
                    "rationale": "User rescued 5 slow-shutter motion frames.",
                    "evidence_ids": [1, 2, 3, 4, 5],
                },
                {
                    "rule_text": proposed_text_2,
                    "rationale": "User rejected bright backlit frames.",
                    "evidence_ids": [3, 5],
                },
            ]
        )
        fake_agent = _RecordingAgentModel()
        fake_vlm = _RecordingTriageModel()
        svcs.override_model_router(
            ModelRouter(
                svcs.repo,
                svcs.app_paths,
                svcs.config,
                adapters={
                    "consolidate": fake_consolidate,
                    "agent": fake_agent,
                    "triage_vlm": fake_vlm,
                },
            )
        )

        mgr = MemoryManager(svcs)
        skipped = asyncio.run(mgr.consolidate(force=False))
        assert skipped.status == "skipped_insufficient_overrides"
        assert skipped.new_override_count == 4
        assert len(fake_consolidate.received_messages) == 0

        # Add 5th genuine override -> hits >= 5 threshold and triggers 1 single text-only LLM call
        cur4 = svcs.repo.get_effective_verdict(photos[4].id)
        target4 = "normal" if (cur4 is not None and cur4.bucket != "normal") else "red"
        svcs.review.set_verdict(
            photos[4].id,
            bucket=target4,
            source="user",
            reason_tag="intentional_blur",
            actor="web:user",
        )
        res = asyncio.run(mgr.consolidate(force=False))
        assert res.status == "proposed"
        assert len(res.proposed_rules) == 2
        assert len(fake_consolidate.received_messages) == 1
        # Strictly zero images sent during consolidation!
        assert all(len(m.images) == 0 for m in fake_consolidate.received_messages[0])
        # condition_dsl_json MUST remain NULL in v1
        assert all(r.condition_dsl_json is None for r in res.proposed_rules)
        assert all(r.status == "proposed" for r in res.proposed_rules)

        # Verify `proposed` rules do NOT appear in preferences.md, MCP resource, Agent system prompt, or VLM prompt!
        assert proposed_text_1 not in svcs.review.read_preferences_markdown()
        assert proposed_text_2 not in svcs.review.read_preferences_markdown()

        asyncio.run(AgentRunner(svcs).start_session(scan.session_id, mode="chat", message="Hi"))
        agent_sys_before = fake_agent.received_messages[-1][0].text
        assert proposed_text_1 not in agent_sys_before
        assert proposed_text_2 not in agent_sys_before

        svcs.review.set_verdict(photos[5].id, bucket="yellow", source="engine")
        svcs.triage.run_triage_sync(scan.session_id, max_photos=1)
        vlm_sys_before = fake_vlm.received_messages[-1][0].text
        assert proposed_text_1 not in vlm_sys_before
        assert proposed_text_2 not in vlm_sys_before

        # Accept rule 1 (with human edit), reject rule 2
        r1, r2 = res.proposed_rules[0], res.proposed_rules[1]
        edited_rule_1 = "Keep intentional slow-shutter street motion blur frames (1/15s or slower)"
        dec1 = mgr.decide_rule(r1.rule_id, decision="accept", rule_text=edited_rule_1)
        dec2 = mgr.decide_rule(r2.rule_id, decision="reject")
        assert dec1.status == "accepted" and dec1.condition_dsl_json is None
        assert dec2.status == "rejected"

        # Accepted rule appears in preferences.md & MCP resource; rejected rule does NOT!
        prefs_md = svcs.review.read_preferences_markdown()
        assert f"- [{r1.rule_id}] {edited_rule_1}" in prefs_md
        assert proposed_text_2 not in prefs_md

        # Accepted rule now injects into BOTH Agent system prompt AND VLM triage prompt!
        asyncio.run(AgentRunner(svcs).start_session(scan.session_id, mode="chat", message="Check again"))
        agent_sys_after = fake_agent.received_messages[-1][0].text
        assert edited_rule_1 in agent_sys_after
        assert proposed_text_2 not in agent_sys_after

        svcs.review.set_verdict(photos[6].id, bucket="yellow", source="engine")
        svcs.triage.run_triage_sync(scan.session_id, max_photos=1)
        vlm_sys_after = fake_vlm.received_messages[-1][0].text
        assert edited_rule_1 in vlm_sys_after
        assert proposed_text_2 not in vlm_sys_after


def test_manual_preferences_edit_takes_effect_and_zero_llm_engine_is_unaffected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PLAN T14 acceptance: manual edit to preferences.md activates immediately; zero-LLM engine stays untouched."""
    _clear_keys(monkeypatch)
    shoot_dir = tmp_path / "shoot"
    build_demo_dataset(shoot_dir)

    # 1. Baseline zero-LLM scan without preferences.md
    with LightboxServices.open(tmp_path / "home_clean") as svcs_clean:
        scan_clean = svcs_clean.scan.scan_folder(shoot_dir, max_workers=1)
        clean_buckets = {
            p.stem: (svcs_clean.repo.get_effective_verdict(p.id).bucket)  # type: ignore[union-attr]
            for p in svcs_clean.repo.list_photos(scan_clean.session_id)
        }

    # 2. Scan identical shoot WITH aggressive manual preference rules in preferences.md
    with LightboxServices.open(tmp_path / "home_with_prefs") as svcs_prefs:
        manual_rule = "Prefer warm candid portraits and never reject grainy high-ISO night shots"
        MemoryManager(svcs_prefs).save_preferences(f"# Lightbox Preferences\n\n- {manual_rule}\n")

        scan_with_prefs = svcs_prefs.scan.scan_folder(shoot_dir, max_workers=1)
        prefs_buckets = {
            p.stem: (svcs_prefs.repo.get_effective_verdict(p.id).bucket)  # type: ignore[union-attr]
            for p in svcs_prefs.repo.list_photos(scan_with_prefs.session_id)
        }
        # Zero-LLM engine verdicts MUST be 100% identical regardless of preferences.md
        assert prefs_buckets == clean_buckets

        # Meanwhile, manual edit takes effect immediately in Agent system prompt & MCP resource!
        assert manual_rule in svcs_prefs.review.read_preferences_markdown()
        fake_agent = _RecordingAgentModel()
        svcs_prefs.override_model_router(
            ModelRouter(svcs_prefs.repo, svcs_prefs.app_paths, svcs_prefs.config, adapters={"agent": fake_agent})
        )
        asyncio.run(
            AgentRunner(svcs_prefs).start_session(
                scan_with_prefs.session_id,
                mode="chat",
                message="Pick top shots.",
            )
        )
        assert manual_rule in fake_agent.received_messages[0][0].text


def test_propose_rule_tool_and_memory_rest_api_endpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Agent `propose_rule` tool + GET/PUT preferences, POST consolidate, POST rules/{id}/decision APIs."""
    _clear_keys(monkeypatch)
    shoot_dir = tmp_path / "api_shoot"
    build_demo_dataset(shoot_dir)
    home_dir = tmp_path / "api_home"

    # 1. Agent `propose_rule` tool writes `proposed` ONLY (never touches preferences.md)
    reg = ToolRegistry()
    assert len(reg.mcp_specs()) == 14  # propose_rule stays agent-only
    with LightboxServices.open(home_dir) as svcs:
        scan = svcs.scan.scan_folder(shoot_dir, max_workers=1)
        photos = [p for p in svcs.repo.list_photos(scan.session_id) if not p.is_protected]
        for p in photos[:5]:
            cur = svcs.repo.get_effective_verdict(p.id)
            target = "normal" if (cur is not None and cur.bucket != "normal") else "red"
            svcs.review.set_verdict(p.id, bucket=target, source="user", reason_tag="good_moment", actor="web:user")

        ctx = ToolContext(services=svcs, surface="agent:chat", session_id=scan.session_id)
        prop_res = reg.execute(
            "propose_rule",
            {"rule_text": "Favor natural smiles over stiff poses", "rationale": "Observed in chat"},
            ctx,
        )
        assert not prop_res.is_error
        assert prop_res.data["rule"]["status"] == "proposed"
        assert "Favor natural smiles" not in svcs.review.read_preferences_markdown()
        app_paths = svcs.app_paths

    # 2. FastAPI Memory endpoints end-to-end
    fake_consolidate = _FakeConsolidateModel(
        [
            {
                "rule_text": "Keep candid slightly soft moments with strong emotion",
                "rationale": "5 rescued good_moment overrides",
                "evidence_ids": [1, 2, 3, 4, 5],
            }
        ]
    )
    app = create_app(
        app_paths=app_paths,
        port=8765,
        model_router_factory=lambda s: ModelRouter(
            s.repo,
            s.app_paths,
            s.config,
            adapters={"consolidate": fake_consolidate},
        ),
    )

    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        auth = client.post(
            "/api/auth/session",
            headers={"Origin": "http://127.0.0.1:8765", "X-Lightbox-Local-Confirm": "1"},
        )
        csrf = auth.json()["csrf_token"]
        headers = {"Origin": "http://127.0.0.1:8765", "X-Lightbox-CSRF": csrf}

        # GET /api/memory/preferences
        get_res = client.get("/api/memory/preferences")
        assert get_res.status_code == 200
        state0 = get_res.json()
        assert state0["ready_to_consolidate"] is True
        assert state0["proposed_count"] == 1

        # POST /api/memory/consolidate
        cons_res = client.post("/api/memory/consolidate", headers=headers, json={"force": False})
        assert cons_res.status_code == 200, cons_res.text
        cons_data = cons_res.json()
        assert cons_data["result"]["status"] == "proposed"
        new_rule_id = cons_data["result"]["proposed_rules"][0]["rule_id"]

        # POST /api/memory/rules/{id}/decision -> accept
        dec_res = client.post(
            f"/api/memory/rules/{new_rule_id}/decision",
            headers=headers,
            json={"decision": "accept"},
        )
        assert dec_res.status_code == 200, dec_res.text
        assert dec_res.json()["rule"]["status"] == "accepted"
        assert new_rule_id in dec_res.json()["state"]["preferences_md"]

        # PUT /api/memory/preferences -> direct manual edit
        custom_md = "# Lightbox Preferences\n\n- Manual rule from textarea\n"
        put_res = client.put(
            "/api/memory/preferences",
            headers=headers,
            json={"content": custom_md},
        )
        assert put_res.status_code == 200
        assert "Manual rule from textarea" in put_res.json()["preferences_md"]
