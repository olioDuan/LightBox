"""Step 2 API & service tests: verifiable probes, error mapping, SQLite state derivation, Ollama, and MCP safety."""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import BaseModel

from lightbox.api import CSRF_HEADER_NAME, create_app
from lightbox.config import ModelSpec
from lightbox.logging import configure_logging
from lightbox.models import (
    GenerationResult,
    Message,
    ModelCaps,
    ModelError,
    ToolCall,
    ToolSchema,
    Usage,
    classify_provider_error,
)
from lightbox.paths import AppPaths, get_app_paths
from lightbox.store import Database, LightboxRepository, LLMCallRecord

TEST_PORT = 8765
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"

_RGB_TO_NAME: dict[tuple[int, int, int], str] = {
    (220, 38, 38): "red",
    (37, 99, 235): "blue",
    (22, 163, 74): "green",
    (234, 179, 8): "yellow",
    (234, 88, 12): "orange",
    (147, 51, 234): "purple",
}


def _nearest_color_name(rgb: tuple[int, int, int]) -> str:
    return min(
        _RGB_TO_NAME.items(),
        key=lambda kv: sum((a - b) ** 2 for a, b in zip(rgb, kv[0], strict=True)),
    )[1]


class _ScriptedProbeAdapter:
    """Configurable fake adapter that inspects actual probe inputs (image pixels, tools, schema)."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        vision_mode: str = "accurate",  # "accurate" | "wrong" | "raise"
        tools_mode: str = "accurate",  # "accurate" | "ignore" | "raise"
        json_mode: str = "accurate",  # "accurate" | "wrong" | "raise"
        error_to_raise: BaseException | None = None,
        call_log: list[tuple[str, str]] | None = None,
    ) -> None:
        self.id = spec.model_id
        self.provider = spec.provider
        self.family = spec.family
        self.caps = ModelCaps(vision=True, tools=True, json_schema=True, prompt_cache=False)
        self.vision_mode = vision_mode
        self.tools_mode = tools_mode
        self.json_mode = json_mode
        self.error_to_raise = error_to_raise
        self.call_log = call_log if call_log is not None else []

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        response_schema: type[BaseModel] | None = None,
        max_tokens: int = 2048,
        effort: str | None = None,
    ) -> GenerationResult:
        del max_tokens, effort
        user_msg = messages[-1]

        # 1. Vision probe
        if user_msg.images:
            self.call_log.append((self.id, "vision"))
            if self.vision_mode == "raise" and self.error_to_raise is not None:
                raise self.error_to_raise
            if self.vision_mode == "wrong":
                return GenerationResult(text="black,white", provider=self.provider, model=self.id)
            with Image.open(io.BytesIO(user_msg.images[0].data)) as im:
                rgb = im.convert("RGB")
                left_px = rgb.getpixel((16, 32))
                right_px = rgb.getpixel((96, 32))
                assert isinstance(left_px, tuple) and isinstance(right_px, tuple)
                left_name = _nearest_color_name((int(left_px[0]), int(left_px[1]), int(left_px[2])))
                right_name = _nearest_color_name((int(right_px[0]), int(right_px[1]), int(right_px[2])))
            return GenerationResult(
                text=f"{left_name},{right_name}",
                usage=Usage(in_tokens=50, out_tokens=4),
                provider=self.provider,
                model=self.id,
            )

        # 2. Tool probe
        if tools:
            self.call_log.append((self.id, "tools"))
            if self.tools_mode == "raise" and self.error_to_raise is not None:
                raise self.error_to_raise
            if self.tools_mode == "ignore":
                return GenerationResult(text="I cannot call tools.", provider=self.provider, model=self.id)
            m = re.search(r'"token"\s*:\s*"([^"]+)"', user_msg.text)
            tok = m.group(1) if m else ""
            return GenerationResult(
                text="",
                tool_calls=(ToolCall(id="tc_1", name="lightbox_probe_echo", arguments={"token": tok}),),
                usage=Usage(in_tokens=40, out_tokens=10),
                provider=self.provider,
                model=self.id,
            )

        # 3. Structured JSON probe
        if response_schema is not None:
            self.call_log.append((self.id, "json_schema"))
            if self.json_mode == "raise" and self.error_to_raise is not None:
                raise self.error_to_raise
            if self.json_mode == "wrong":
                return GenerationResult(text='{"ok": false, "code": "bad"}', parsed=None, provider=self.provider, model=self.id)
            m = re.search(r'code="([^"]+)"', user_msg.text)
            code = m.group(1) if m else ""
            payload = {"ok": True, "code": code}
            parsed = response_schema.model_validate(payload)
            return GenerationResult(
                text=json.dumps(payload),
                parsed=parsed,
                usage=Usage(in_tokens=40, out_tokens=10),
                provider=self.provider,
                model=self.id,
            )

        return GenerationResult(text="ok", provider=self.provider, model=self.id)


def _setup_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    probe_factory: Any = None,
) -> tuple[TestClient, dict[str, str], AppPaths, dict[str, str]]:
    for v in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "LIGHTBOX_CUSTOM_API_KEY",
        "LIGHTBOX_LOCAL_API_KEY",
    ):
        monkeypatch.delenv(v, raising=False)

    fake_kr: dict[str, str] = {}
    monkeypatch.setattr("lightbox.models.secrets._read_from_keyring", lambda acct: fake_kr.get(acct))
    monkeypatch.setattr("lightbox.models.secrets._write_to_keyring", lambda acct, val: fake_kr.__setitem__(acct, val))
    monkeypatch.setattr("lightbox.models.secrets._delete_from_keyring", lambda acct: fake_kr.pop(acct, None))

    paths = get_app_paths(tmp_path / "api_home")
    token = "step2-launch-token"
    app = create_app(
        app_paths=paths,
        port=TEST_PORT,
        bootstrap_token=token,
        probe_adapter_factory=probe_factory,
    )
    client = TestClient(app, base_url=BASE_URL)
    r = client.get(f"/auth/launch?token={token}", follow_redirects=False)
    assert r.status_code == 302
    csrf = str(client.get("/api/auth/me").json()["csrf_token"])
    headers = {CSRF_HEADER_NAME: csrf, "Origin": BASE_URL}
    return client, headers, paths, fake_kr


# ---------------------------------------------------------------------------
# 1. Verifiable probes, dual-model dedup, role degradation & error mapping
# ---------------------------------------------------------------------------


def test_verifiable_probes_role_degradation_and_shared_model_dedup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_log: list[tuple[str, str]] = []

    def _factory(spec: ModelSpec, _api_key: str) -> _ScriptedProbeAdapter:
        # Agent model: vision + json OK, tools FAIL -> agent_ready=False, triage_ready=True!
        if spec.model_id == "gpt-4o":
            return _ScriptedProbeAdapter(spec, tools_mode="ignore", call_log=call_log)
        return _ScriptedProbeAdapter(spec, call_log=call_log)

    client, headers, _paths, _kr = _setup_client(tmp_path, monkeypatch, probe_factory=_factory)

    # A. Dual models (gpt-4o + gpt-4o-mini): tools fail on agent, triage passes independently
    r = client.post(
        "/api/ai/connection/test",
        json={
            "provider": "openai",
            "agent_model": "gpt-4o",
            "triage_model": "gpt-4o-mini",
            "api_key": "sk-test-dual-1111",
        },
        headers=headers,
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["agent_ready"] is False
    assert data["agent_probes"]["vision"]["ok"] is True
    assert data["agent_probes"]["tools"]["ok"] is False
    assert data["agent_probes"]["json_schema"]["ok"] is True
    assert data["triage_ready"] is True
    assert data["triage_probes"]["shared_with_agent"] is False
    assert ("gpt-4o", "vision") in call_log
    assert ("gpt-4o-mini", "vision") in call_log

    # B. Same model for both roles -> shared probes, NO duplicate vision/json calls!
    call_log.clear()
    r_same = client.post(
        "/api/ai/connection/test",
        json={
            "provider": "openai",
            "agent_model": "gpt-4o-mini",
            "triage_model": "gpt-4o-mini",
            "api_key": "sk-test-same-2222",
        },
        headers=headers,
    )
    assert r_same.status_code == 200
    d_same = r_same.json()
    assert d_same["agent_ready"] is True
    assert d_same["triage_ready"] is True
    assert d_same["triage_probes"]["shared_with_agent"] is True
    assert call_log == [
        ("gpt-4o-mini", "vision"),
        ("gpt-4o-mini", "tools"),
        ("gpt-4o-mini", "json_schema"),
    ]


def test_verifiable_vision_probe_rejects_hallucinated_color_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, headers, _paths, _kr = _setup_client(
        tmp_path,
        monkeypatch,
        probe_factory=lambda spec, _k: _ScriptedProbeAdapter(spec, vision_mode="wrong"),
    )
    r = client.post(
        "/api/ai/connection/test",
        json={
            "provider": "anthropic",
            "agent_model": "claude-sonnet-5-5",
            "triage_model": "claude-sonnet-5-5",
            "api_key": "sk-ant-blind-3333",
        },
        headers=headers,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["agent_probes"]["vision"]["ok"] is False
    assert data["agent_ready"] is False
    assert data["triage_ready"] is False
    assert data["overall_ok"] is False


def test_provider_error_mapping_persistent_vs_transient_and_unmapped_fallback() -> None:
    # OpenAI 429 insufficient_quota -> persistent insufficient_balance
    m_quota = classify_provider_error(
        "openai",
        ModelError("Quota exceeded", status_code=429, provider_code="insufficient_quota"),
    )
    assert m_quota.error_class == "insufficient_balance" and m_quota.is_persistent is True

    # OpenAI 429 rate_limit_exceeded -> transient rate_limited
    m_rl = classify_provider_error(
        "openai",
        ModelError("Rate limit", status_code=429, provider_code="rate_limit_exceeded"),
    )
    assert m_rl.error_class == "rate_limited" and m_rl.is_persistent is False

    # Anthropic credit balance too low -> persistent insufficient_balance
    m_ant_bal = classify_provider_error(
        "anthropic",
        ModelError("Your credit balance is too low to access the Anthropic API.", status_code=400),
    )
    assert m_ant_bal.error_class == "insufficient_balance" and m_ant_bal.is_persistent is True

    # Gemini API_KEY_INVALID -> persistent auth_failed
    m_gem = classify_provider_error(
        "gemini",
        ModelError("Bad key", status_code=400, provider_code="API_KEY_INVALID"),
    )
    assert m_gem.error_class == "auth_failed" and m_gem.is_persistent is True

    # Unmapped 418 -> fallback request_failed preserving raw code
    m_other = classify_provider_error(
        "custom",
        ModelError("I am a teapot", status_code=418, provider_code="teapot_err"),
    )
    assert m_other.error_class == "request_failed"
    assert m_other.raw_code == "teapot_err"
    assert m_other.is_persistent is False


# ---------------------------------------------------------------------------
# 2. Fingerprint + SQLite state derivation (persistent error vs transient notice)
# ---------------------------------------------------------------------------


def test_connection_state_derivation_strictly_matches_fingerprint_and_real_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, headers, paths, _kr = _setup_client(
        tmp_path,
        monkeypatch,
        probe_factory=lambda spec, _k: _ScriptedProbeAdapter(spec),
    )

    # 1. Initially unconfigured
    assert client.get("/api/ai/connection").json()["status"] == "unconfigured"

    # 2. Save without testing -> configured_unverified (zero network calls)
    r_save = client.put(
        "/api/ai/connection",
        json={"provider": "openai", "api_key": "sk-live-test-4321"},
        headers=headers,
    )
    assert r_save.status_code == 200, r_save.text
    saved = r_save.json()
    assert saved["status"] == "configured_unverified"
    assert saved["agent_model"] == "gpt-4o"
    assert saved["triage_model"] == "gpt-4o-mini"
    assert saved["review_mode"] == "rules_only"
    assert saved["credential_source"] == "keyring"
    assert saved["key_suffix"] == "4321"
    assert "sk-live-test" not in r_save.text
    fp = saved["fingerprint"]

    # 3. Test saved connection -> becomes connected with verified_at
    r_test = client.post("/api/ai/connection/test", json={}, headers=headers)
    assert r_test.status_code == 200 and r_test.json()["overall_ok"] is True

    st = client.get("/api/ai/connection").json()
    assert st["status"] == "connected"
    assert st["verified_at"] is not None
    assert st["transient_notice"] is None

    # Helper to inject SQLite llm_calls rows
    def _insert_call(
        *,
        call_id: str,
        status: str,
        error_class: str | None,
        conn_fp: str,
        cache_hit: bool,
        ts: float,
    ) -> None:
        with Database(paths.db_file) as db:
            repo = LightboxRepository(db)
            repo.add_llm_call(
                LLMCallRecord(
                    id=call_id,
                    cache_key=None,
                    role="triage_vlm",
                    provider="openai",
                    model="gpt-4o-mini",
                    in_tokens=0,
                    out_tokens=0,
                    cached_tokens=0,
                    cost_usd=0.0,
                    latency_ms=10,
                    response_json=None,
                    created_at=ts,
                    session_id="s1",
                    cache_hit=cache_hit,
                    status=status,
                    error_class=error_class,
                    connection_fingerprint=conn_fp,
                )
            )

    now = time.time() + 1.0

    # 4. Persistent error on a DIFFERENT (old) fingerprint is ignored!
    _insert_call(
        call_id="c_old_fp",
        status="error",
        error_class="auth_failed",
        conn_fp="fp_stale_old_config",
        cache_hit=False,
        ts=now,
    )
    assert client.get("/api/ai/connection").json()["status"] == "connected"

    # 5. Cache-hit error row (if any) is ignored (`cache_hit=False` required)!
    _insert_call(
        call_id="c_cache_hit",
        status="error",
        error_class="auth_failed",
        conn_fp=fp,
        cache_hit=True,
        ts=now + 1.0,
    )
    assert client.get("/api/ai/connection").json()["status"] == "connected"

    # 6. Transient errors (429 rate_limited, network_timeout, local_budget_exceeded) KEEP `connected` + transient_notice!
    for i, transient_cls in enumerate(("rate_limited", "network_timeout", "local_budget_exceeded")):
        _insert_call(
            call_id=f"c_trans_{i}",
            status="error",
            error_class=transient_cls,
            conn_fp=fp,
            cache_hit=False,
            ts=now + 2.0 + i,
        )
        cur = client.get("/api/ai/connection").json()
        assert cur["status"] == "connected"
        assert cur["transient_notice"]["error_class"] == transient_cls

    # 7. Model-specific persistent error on Triage ONLY degrades Triage while Agent remains healthy!
    _insert_call(
        call_id="c_triage_404",
        status="error",
        error_class="model_or_endpoint_unavailable",
        conn_fp=fp,
        cache_hit=False,
        ts=now + 8.0,
    )
    cur_triage_deg = client.get("/api/ai/connection").json()
    assert cur_triage_deg["status"] == "connected"
    assert cur_triage_deg["agent_ready"] is True
    assert cur_triage_deg["agent_error"] is None
    assert cur_triage_deg["triage_ready"] is False
    assert cur_triage_deg["triage_error"]["error_class"] == "model_or_endpoint_unavailable"

    # 8. Account-wide persistent error (insufficient_balance / auth_failed) flips entire connection to `error`!
    _insert_call(
        call_id="c_persist_bal",
        status="error",
        error_class="insufficient_balance",
        conn_fp=fp,
        cache_hit=False,
        ts=now + 10.0,
    )
    cur_err = client.get("/api/ai/connection").json()
    assert cur_err["status"] == "error"
    assert cur_err["agent_ready"] is False
    assert cur_err["triage_ready"] is False
    assert cur_err["error"]["error_class"] == "insufficient_balance"

    # 9. Subsequent real ok call on triage recovers triage (and overall status -> `connected`)!
    _insert_call(
        call_id="c_ok_recover",
        status="ok",
        error_class=None,
        conn_fp=fp,
        cache_hit=False,
        ts=now + 11.0,
    )
    cur_ok = client.get("/api/ai/connection").json()
    assert cur_ok["status"] == "connected"
    assert cur_ok["triage_ready"] is True
    assert cur_ok["transient_notice"] is None

    # 10. Partial pricing (only input_cost_per_mtok without output) rejected with HTTP 422!
    r_half_price = client.put(
        "/api/ai/connection",
        json={"provider": "openai", "input_cost_per_mtok": 2.0},
        headers=headers,
    )
    assert r_half_price.status_code == 422

    # 11. Changing model resets fingerprint -> `configured_unverified` (no network call)
    r_change = client.put(
        "/api/ai/connection",
        json={"provider": "openai", "agent_model": "gpt-4o-mini", "triage_model": "gpt-4o-mini"},
        headers=headers,
    )
    assert r_change.status_code == 200
    assert r_change.json()["status"] == "configured_unverified"
    assert r_change.json()["fingerprint"] != fp


# ---------------------------------------------------------------------------
# 3. DELETE re-resolves remaining credential source & base_url change requires Key
# ---------------------------------------------------------------------------


def test_delete_reresolves_env_credential_and_base_url_change_requires_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, headers, _paths, _kr = _setup_client(
        tmp_path,
        monkeypatch,
        probe_factory=lambda spec, _k: _ScriptedProbeAdapter(spec),
    )

    # Env key exists AND user saves a keyring key on official endpoint
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-7777")
    r_put = client.put(
        "/api/ai/connection",
        json={"provider": "anthropic", "api_key": "sk-ant-kr-8888"},
        headers=headers,
    )
    assert r_put.status_code == 200
    assert r_put.json()["credential_source"] == "keyring"
    assert r_put.json()["can_delete_credential"] is True

    # Trying to change base_url WITHOUT re-entering api_key MUST be rejected!
    r_bad_save = client.put(
        "/api/ai/connection",
        json={"provider": "anthropic", "base_url": "https://other-proxy.example.com"},
        headers=headers,
    )
    assert r_bad_save.status_code == 400
    assert r_bad_save.json()["detail"]["error_class"] == "endpoint_binding_mismatch"

    r_bad_test = client.post(
        "/api/ai/connection/test",
        json={"provider": "anthropic", "base_url": "https://other-proxy.example.com"},
        headers=headers,
    )
    assert r_bad_test.status_code == 400
    assert r_bad_test.json()["detail"]["error_class"] == "endpoint_binding_mismatch"

    # DELETE /api/ai/connection clears keyring key, deactivates [ai], and re-resolves remaining `env` source!
    r_del = client.delete("/api/ai/connection", headers=headers)
    assert r_del.status_code == 200
    del_data = r_del.json()
    assert del_data["status"] == "unconfigured"
    assert del_data["credential_deleted"] is True
    assert del_data["remaining_credential_source"] == "env"
    assert del_data["remaining_key_suffix"] == "7777"
    assert del_data["remaining_can_delete_credential"] is False

    # Re-activating with env key (no api_key passed) reports credential_source="env" & can_delete_credential=False
    r_env = client.put("/api/ai/connection", json={"provider": "anthropic"}, headers=headers)
    assert r_env.status_code == 200
    assert r_env.json()["credential_source"] == "env"
    assert r_env.json()["can_delete_credential"] is False


# ---------------------------------------------------------------------------
# 4. Settings isolation, Ollama discovery, MCP read-only safety & CSRF/Cookie gates
# ---------------------------------------------------------------------------


def test_settings_rejects_ai_and_mcp_routes_enforce_strict_safety(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, headers, paths, _kr = _setup_client(tmp_path, monkeypatch)
    configure_logging(paths, level="INFO", log_to_file=True, log_to_stderr=False)

    # 1. PUT /api/settings explicitly rejects `ai`
    r_bad_settings = client.put(
        "/api/settings",
        json={"ai": {"provider": "openai", "agent_model": "gpt-4o", "triage_model": "gpt-4o-mini"}},
        headers=headers,
    )
    assert r_bad_settings.status_code == 422

    # 2. POST /api/ai/mcp/snippet rejects extra fields (`extra="forbid"`) & invalid client enums
    assert (
        client.post(
            "/api/ai/mcp/snippet",
            json={"client": "claude_code", "command": "rm -rf /"},
            headers=headers,
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/ai/mcp/snippet",
            json={"client": "unknown_client"},
            headers=headers,
        ).status_code
        == 422
    )

    for cid, expected_fmt in (
        ("claude_code", "shell"),
        ("codex", "toml"),
        ("claude_desktop", "json"),
        ("cursor", "json"),
    ):
        r_snip = client.post("/api/ai/mcp/snippet", json={"client": cid}, headers=headers)
        assert r_snip.status_code == 200, r_snip.text
        body = r_snip.json()
        assert body["format"] == expected_fmt
        assert "lightbox" in body["snippet"]

    # 3. GET /api/ai/mcp/clients: read-only, validates paths, never returns or logs raw secrets!
    fake_home = tmp_path / "fake_home"
    valid_bin = fake_home / "bin" / "uv"
    valid_bin.parent.mkdir(parents=True, exist_ok=True)
    valid_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    valid_repo = fake_home / "repo"
    valid_repo.mkdir(parents=True, exist_ok=True)

    # Cursor: configured with valid command + valid --directory, PLUS another server with a secret API key!
    secret_sentinel = "sk-SUPER-SECRET-OTHER-MCP-SERVER-TOKEN-999"
    cursor_cfg = fake_home / ".cursor" / "mcp.json"
    cursor_cfg.parent.mkdir(parents=True, exist_ok=True)
    cursor_cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "lightbox": {
                        "command": str(valid_bin),
                        "args": ["--directory", str(valid_repo), "run", "lightbox", "mcp"],
                    },
                    "other_secret_tool": {
                        "command": "node",
                        "env": {"SECRET_API_KEY": secret_sentinel},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    # Codex: invalid --directory path
    codex_cfg = fake_home / ".codex" / "config.toml"
    codex_cfg.parent.mkdir(parents=True, exist_ok=True)
    codex_cfg.write_text(
        f'[mcp_servers.lightbox]\ncommand = "{valid_bin}"\nargs = ["--directory", "{fake_home / "missing_repo"}", "run", "lightbox", "mcp"]\n',
        encoding="utf-8",
    )

    # Claude Code: malformed JSON -> parse_error and file MUST NOT be modified!
    claude_code_cfg = fake_home / ".claude.json"
    corrupt_original = f'{{"broken": "{secret_sentinel}'
    claude_code_cfg.write_text(corrupt_original, encoding="utf-8")

    monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)
    r_clients = client.get("/api/ai/mcp/clients")
    assert r_clients.status_code == 200
    assert secret_sentinel not in r_clients.text
    assert "other_secret_tool" not in r_clients.text

    by_id = {c["client_id"]: c for c in r_clients.json()["clients"]}
    assert by_id["cursor"]["status"] == "configured"
    assert by_id["cursor"]["command_valid"] is True
    assert by_id["cursor"]["directory_valid"] is True
    assert by_id["codex"]["status"] == "invalid_path"
    assert by_id["codex"]["command_valid"] is True
    assert by_id["codex"]["directory_valid"] is False
    assert by_id["claude_code"]["status"] == "parse_error"
    assert claude_code_cfg.read_text(encoding="utf-8") == corrupt_original

    # Verify log file contains zero leaked external config secret
    if paths.log_file.is_file():
        assert secret_sentinel not in paths.log_file.read_text(encoding="utf-8")

    # 4. Ollama models discovery (/api/tags + /api/show) + confirms Ollama as `known_free`
    monkeypatch.setattr(
        "lightbox.services.ai_connection._default_http_get_json",
        lambda url, **_kw: {"models": [{"name": "qwen2.5vl:7b", "size": 4_500_000_000}]}
        if url == "http://127.0.0.1:11434/api/tags"
        else {},
    )
    monkeypatch.setattr(
        "lightbox.services.ai_connection._default_http_post_json",
        lambda url, body, **_kw: {"model_info": {"qwen2vl.context_length": 32768}}
        if url == "http://127.0.0.1:11434/api/show" and body.get("name") == "qwen2.5vl:7b"
        else {},
    )
    r_ollama = client.get("/api/ai/ollama/models?base_url=http://127.0.0.1:11434")
    assert r_ollama.status_code == 200
    o_data = r_ollama.json()
    assert o_data["ollama_confirmed"] is True
    assert o_data["openai_v1_url"] == "http://127.0.0.1:11434/v1"
    assert o_data["models"][0]["name"] == "qwen2.5vl:7b"
    assert o_data["models"][0]["context_length_limit"] == 32768
    assert "num_ctx" in o_data["context_note"]

    # Saving Ollama connection now classifies as `known_free` because `/api/tags` confirmed it
    r_save_ollama = client.put(
        "/api/ai/connection",
        json={"provider": "ollama", "agent_model": "qwen2.5vl:7b", "triage_model": "qwen2.5vl:7b"},
        headers=headers,
    )
    assert r_save_ollama.status_code == 200
    assert r_save_ollama.json()["cost_classification"] == "known_free"

    # 5. Auth & CSRF security gates on new /api/ai/* routes
    anon = TestClient(client.app, base_url=BASE_URL)
    assert anon.get("/api/ai/connection").status_code == 401
    assert anon.get("/api/ai/mcp/clients").status_code == 401
    assert client.put("/api/ai/connection", json={"provider": "ollama"}).status_code == 403
    assert client.post("/api/ai/connection/test", json={}).status_code == 403
    assert client.delete("/api/ai/connection").status_code == 403
    assert client.post("/api/ai/mcp/snippet", json={"client": "codex"}).status_code == 403


def test_dns_unreachable_probes_untested_validation_ollama_guard_and_claude_code_mcp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket

    # 1. DNS failure classifies as `service_unreachable` (NOT `network_timeout`) with raw_code=None (no `ModelError`)
    dns_root = socket.gaierror(8, "nodename nor servname provided, or not known")
    conn_exc = ConnectionError("Connection error.")
    conn_exc.__cause__ = dns_root
    wrapped_dns = ModelError("OpenAI-compatible request failed (m): Connection error.")
    wrapped_dns.__cause__ = conn_exc

    mapped_dns = classify_provider_error("custom", wrapped_dns)
    assert mapped_dns.error_class == "service_unreachable"
    assert mapped_dns.raw_code is None

    # Timeout still classifies as `network_timeout` and never leaks Python class name
    timeout_exc = ModelError("OpenAI-compatible request failed (m): Request timed out.")
    timeout_exc.__cause__ = TimeoutError("timed out")
    mapped_to = classify_provider_error("custom", timeout_exc)
    assert mapped_to.error_class == "network_timeout"
    assert mapped_to.raw_code is None

    # 2. Unreachable probe aborts early and marks ALL agent & triage probes as `untested`
    call_log: list[tuple[str, str]] = []
    client, headers, _paths, _kr = _setup_client(
        tmp_path,
        monkeypatch,
        probe_factory=lambda spec, _k: _ScriptedProbeAdapter(
            spec,
            vision_mode="raise",
            error_to_raise=wrapped_dns,
            call_log=call_log,
        ),
    )
    r_unreach = client.post(
        "/api/ai/connection/test",
        json={
            "provider": "custom",
            "base_url": "https://nonexistent.invalid/v1",
            "agent_model": "my-agent",
            "triage_model": "my-triage",
            "api_key": "sk-custom-9999",
        },
        headers=headers,
    )
    assert r_unreach.status_code == 200
    d_unreach = r_unreach.json()
    assert d_unreach["overall_ok"] is False
    assert d_unreach["error_class"] == "service_unreachable"
    assert d_unreach["error_code"] is None
    assert d_unreach["agent_probes"]["vision"]["status"] == "untested"
    assert d_unreach["agent_probes"]["tools"]["status"] == "untested"
    assert d_unreach["agent_probes"]["json_schema"]["status"] == "untested"
    assert d_unreach["triage_probes"]["vision"]["status"] == "untested"
    assert d_unreach["triage_probes"]["json_schema"]["status"] == "untested"
    assert call_log == [("my-agent", "vision")]

    # 3. Custom provider without base_url returns structured `missing_base_url` error_class
    r_no_base = client.put(
        "/api/ai/connection",
        json={"provider": "custom", "agent_model": "m1", "api_key": "sk-test-1234"},
        headers=headers,
    )
    assert r_no_base.status_code == 400
    assert r_no_base.json()["detail"]["error_class"] == "missing_base_url"

    # 4. Unconfirmed Ollama cannot bypass `/api/tags` check via `is_local_free=True`
    r_ollama_unconfirmed = client.put(
        "/api/ai/connection",
        json={
            "provider": "ollama",
            "base_url": "http://127.0.0.1:19999",
            "agent_model": "qwen2.5vl:7b",
            "triage_model": "qwen2.5vl:7b",
            "is_local_free": True,
        },
        headers=headers,
    )
    assert r_ollama_unconfirmed.status_code == 200
    assert r_ollama_unconfirmed.json()["is_local_free"] is False
    assert r_ollama_unconfirmed.json()["cost_classification"] == "price_unknown"

    # 5. Claude Code without ~/.claude.json, ~/.claude/, or `claude` binary reports `not_installed`
    empty_home = tmp_path / "empty_home"
    empty_home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: empty_home)
    monkeypatch.setattr("lightbox.services.ai_connection.shutil.which", lambda _cmd: None)
    r_mcp = client.get("/api/ai/mcp/clients")
    assert r_mcp.status_code == 200
    by_id = {c["client_id"]: c for c in r_mcp.json()["clients"]}
    assert by_id["claude_code"]["status"] == "not_installed"
    assert by_id["claude_desktop"]["status"] == "not_installed"
    assert by_id["cursor"]["status"] == "not_installed"
    assert by_id["codex"]["status"] == "not_installed"

    # 6. Claude Code snippet quotes special shell characters with `shlex.quote`
    monkeypatch.setattr(
        "lightbox.services.ai_connection._resolve_mcp_launch_spec",
        lambda: ("/opt/my uv/bin/uv", ["--directory", "/tmp/a$b'c d", "run", "lightbox", "mcp"]),
    )
    r_cc = client.post("/api/ai/mcp/snippet", json={"client": "claude_code"}, headers=headers)
    assert r_cc.status_code == 200
    snip = r_cc.json()["snippet"]
    assert "'/opt/my uv/bin/uv'" in snip
    assert "'/tmp/a$b'\"'\"'c d'" in snip
