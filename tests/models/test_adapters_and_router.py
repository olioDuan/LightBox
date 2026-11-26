"""Unit tests for Anthropic 5.5 / OpenAI-compat adapters, secrets, image sanitization, and ModelRouter (PLAN T10)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx2
import pytest
from PIL import Image
from pydantic import BaseModel, ConfigDict

from lightbox.config import AppConfig, ModelSpec, load_config, load_models_config
from lightbox.logging import configure_logging
from lightbox.models import (
    AnthropicChatModel,
    BudgetExceededError,
    GenerationResult,
    ImageInput,
    Message,
    MissingAPIKeyError,
    ModelCaps,
    ModelRouter,
    OpenAICompatChatModel,
    ToolCall,
    ToolSchema,
    Usage,
    has_api_key,
    resolve_api_key,
    save_secret_to_dotenv,
    strict_json_schema,
)
from lightbox.paths import get_app_paths
from lightbox.services import BurstTriageOutput, YellowTriageOutput
from lightbox.store import Database, LightboxRepository


class _DummyOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: str
    score: float


def _make_jpeg_with_exif_and_gps(width: int = 800, height: int = 600) -> bytes:
    img = Image.new("RGB", (width, height), (64, 128, 192))
    exif = Image.Exif()
    exif[0x010F] = "Canon"  # Make
    exif[0x0110] = "EOS R5"  # Model
    exif[0xA431] = "SN-99999999"  # BodySerialNumber
    exif[0x8825] = {1: "N", 2: (37.0, 46.0, 0.0)}  # GPSInfo
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes(), quality=90)
    data = buf.getvalue()
    with Image.open(io.BytesIO(data)) as verify:
        assert len(verify.getexif()) > 0
    return data


def test_anthropic_55_wire_constraints_exif_stripping_and_refusal() -> None:
    """Verify Anthropic 5.5 HTTP JSON payload constraints, EXIF stripping, and refusal handling."""
    captured_bodies: list[dict[str, Any]] = []

    def _handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_bodies.append(body)

        if len(captured_bodies) == 1:
            return httpx2.Response(
                200,
                json={
                    "id": "msg_01",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-5-5",
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": '{"decision":"keep","score":0.92}'}],
                    "usage": {
                        "input_tokens": 1000,
                        "output_tokens": 25,
                        "cache_read_input_tokens": 200,
                    },
                },
            )
        return httpx2.Response(
            200,
            json={
                "id": "msg_02",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "stop_reason": "refusal",
                "content": [{"type": "text", "text": "Cannot assess this image."}],
                "usage": {"input_tokens": 500, "output_tokens": 10},
            },
        )

    spec = ModelSpec(
        provider="anthropic",
        protocol="anthropic",
        family="anthropic",
        model_id="claude-haiku-5-5",
        default_effort="low",
        input_cost_per_mtok=0.80,
        output_cost_per_mtok=4.00,
        cached_input_cost_per_mtok=0.08,
    )
    img_bytes = _make_jpeg_with_exif_and_gps()
    tools = [
        ToolSchema(
            name="inspect_photo",
            description="Inspect a photo",
            input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
            strict=True,
        )
    ]

    async def _run() -> tuple[GenerationResult, GenerationResult]:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(_handler)) as http:
            adapter = AnthropicChatModel(spec, api_key="sk-ant-test", http_client=http)
            r1 = await adapter.generate(
                [
                    Message(role="system", text="Be conservative."),
                    Message(
                        role="user",
                        text="Review this frame.",
                        images=(ImageInput(data=img_bytes, mime_type="image/jpeg"),),
                    ),
                ],
                tools=tools,
                response_schema=_DummyOutput,
                max_tokens=512,
                effort="low",
            )
            r2 = await adapter.generate(
                [Message(role="user", text="Refusal test")],
                response_schema=_DummyOutput,
                max_tokens=256,
            )
            return r1, r2

    res1, res2 = asyncio.run(_run())

    assert len(captured_bodies) == 2
    wire = captured_bodies[0]

    # SPEC §6.1 / PLAN T10 hard constraints:
    assert wire["output_config"]["effort"] == "low"
    assert wire["output_config"]["format"]["type"] == "json_schema"
    assert wire["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert "temperature" not in wire
    assert "top_p" not in wire
    assert "thinking" not in wire
    assert wire["tool_choice"] == {"type": "auto"}
    assert wire["tool_choice"]["type"] not in {"any", "tool"}

    # Uploaded image must have ZERO EXIF/GPS metadata
    sent_b64 = wire["messages"][0]["content"][0]["source"]["data"]
    with Image.open(io.BytesIO(base64.b64decode(sent_b64))) as sent_im:
        assert len(sent_im.getexif()) == 0

    assert isinstance(res1.parsed, _DummyOutput)
    assert res1.parsed.decision == "keep"
    assert res1.refused is False
    assert res1.usage.cached_tokens == 200
    assert res1.cost_usd > 0.0

    # Refusal must be surfaced cleanly without raising schema validation error
    assert res2.refused is True
    assert res2.stop_reason == "refusal"
    assert res2.parsed is None


def test_openai_compat_mlx_vlm_max_pixels_and_tool_calls() -> None:
    """Verify OpenAI-compat adapter supports base_url override, mlx-vlm max_pixels, and EXIF strip."""
    models_cfg = load_models_config()
    mlx_spec = models_cfg.models["mlx-vlm"]
    assert mlx_spec.max_pixels is not None
    assert mlx_spec.max_pixels <= 1_003_520
    assert models_cfg.profiles["local"].triage_vlm == "mlx-vlm"

    captured: list[tuple[str, dict[str, Any]]] = []

    def _handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.append((str(request.url), body))
        if len(captured) == 1:
            return httpx2.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "model": mlx_spec.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": '```json\n{"decision":"unsure","score":0.51}\n```',
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 800, "completion_tokens": 20},
                },
            )
        return httpx2.Response(
            200,
            json={
                "id": "chatcmpl-2",
                "object": "chat.completion",
                "model": "gemini-2.5-flash",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_photo",
                                        "arguments": '{"photo_id":"ph_1"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 400,
                    "completion_tokens": 15,
                    "prompt_tokens_details": {"cached_tokens": 100},
                },
            },
        )

    oversized_img = _make_jpeg_with_exif_and_gps(2000, 2000)  # 4,000,000 px > 1,003,520
    gemini_spec = models_cfg.models["gemini-2-5-flash"]

    async def _run() -> tuple[GenerationResult, GenerationResult]:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(_handler)) as http:
            mlx_model = OpenAICompatChatModel(mlx_spec, api_key="local-no-key", http_client=http)
            r1 = await mlx_model.generate(
                [
                    Message(
                        role="user",
                        text="Check local VLM resize.",
                        images=(ImageInput(data=oversized_img),),
                    )
                ],
                response_schema=_DummyOutput,
            )
            gem_model = OpenAICompatChatModel(gemini_spec, api_key="gem-key", http_client=http)
            r2 = await gem_model.generate(
                [Message(role="user", text="Call tool")],
                tools=[
                    ToolSchema(
                        name="get_photo",
                        description="Get photo",
                        input_schema={"type": "object", "properties": {"photo_id": {"type": "string"}}},
                    )
                ],
            )
            return r1, r2

    res1, res2 = asyncio.run(_run())

    url1, body1 = captured[0]
    assert url1.startswith("http://127.0.0.1:8080/v1")
    data_url = body1["messages"][0]["content"][1]["image_url"]["url"]
    raw_sent = base64.b64decode(data_url.split(",", 1)[1])
    with Image.open(io.BytesIO(raw_sent)) as sent_im:
        w, h = sent_im.size
        assert (w * h) <= 1_003_520
        assert len(sent_im.getexif()) == 0
    assert isinstance(res1.parsed, _DummyOutput)
    assert res1.parsed.decision == "unsure"
    assert res1.cost_usd == 0.0

    url2, _body2 = captured[1]
    assert "generativelanguage.googleapis.com" in url2
    assert res2.tool_calls == (ToolCall(id="call_1", name="get_photo", arguments={"photo_id": "ph_1"}),)
    assert res2.usage.cached_tokens == 100
    assert res2.cost_usd > 0.0


def test_secrets_precedence_permissions_and_zero_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify keyring -> env -> chmod-600 .env priority and strict permission/redaction checks."""
    paths = get_app_paths(tmp_path / "home")
    paths.ensure_dirs()
    configure_logging(paths, level="INFO", log_to_file=True, log_to_stderr=False)

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("lightbox.models.secrets._read_from_keyring", lambda _var: None)

    assert has_api_key("anthropic", app_paths=paths) is False
    assert has_api_key("local", app_paths=paths) is True
    assert resolve_api_key("local", app_paths=paths) == "local-no-key"

    with pytest.raises(MissingAPIKeyError):
        resolve_api_key("anthropic", app_paths=paths)

    # 3. .env with 0o600 works; loose permissions (0o644) on POSIX rejected
    dotenv = save_secret_to_dotenv("ANTHROPIC_API_KEY", "sk-from-dotenv", paths)
    assert resolve_api_key("anthropic", app_paths=paths) == "sk-from-dotenv"

    if os.name == "posix":
        os.chmod(dotenv, 0o644)
        with pytest.raises(PermissionError, match="chmod 600"):
            resolve_api_key("anthropic", app_paths=paths)
        os.chmod(dotenv, 0o600)

    # 2. Environment overrides .env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    assert resolve_api_key("anthropic", app_paths=paths) == "sk-from-env"

    # 1. Keyring overrides environment
    monkeypatch.setattr(
        "lightbox.models.secrets._read_from_keyring",
        lambda var: "sk-from-keyring" if var == "ANTHROPIC_API_KEY" else None,
    )
    assert resolve_api_key("anthropic", app_paths=paths) == "sk-from-keyring"


class _FakeAdapter:
    def __init__(self, *, model_id: str, cost_usd: float = 0.01, delay_s: float = 0.0) -> None:
        self.id = model_id
        self.provider = "anthropic"
        self.family = "anthropic"
        self.caps = ModelCaps(vision=True, tools=True, json_schema=True, prompt_cache=True)
        self.cost_usd = cost_usd
        self.delay_s = delay_s
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        response_schema: type[BaseModel] | None = None,
        max_tokens: int = 2048,
        effort: str | None = None,
    ) -> GenerationResult:
        del messages, tools, max_tokens, effort
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay_s > 0:
                await asyncio.sleep(self.delay_s)
            text = '{"decision":"keep","score":0.95}'
            parsed = response_schema.model_validate_json(text) if response_schema else None
            return GenerationResult(
                text=text,
                parsed=parsed,
                usage=Usage(in_tokens=500, out_tokens=30),
                cost_usd=self.cost_usd,
                stop_reason="end_turn",
                provider=self.provider,
                model=self.id,
            )
        finally:
            self.in_flight -= 1


def test_router_cache_concurrency_and_budget_enforcement(tmp_path: Path) -> None:
    """Verify SQLite content caching ($0 hit), asyncio.Semaphore cap, and session/monthly budgets."""
    paths = get_app_paths(tmp_path / "home")
    paths.ensure_dirs()
    db = Database(paths.db_file)
    db.migrate()
    repo = LightboxRepository(db)
    cfg = load_config(paths)
    fake = _FakeAdapter(model_id="claude-haiku-5-5", cost_usd=0.05, delay_s=0.02)
    router = ModelRouter(repo, paths, cfg, max_concurrency=2, adapters={"triage_vlm": fake})

    try:
        img = _make_jpeg_with_exif_and_gps(400, 300)
        msg = [Message(role="user", text="Evaluate shot", images=(ImageInput(data=img),))]

        # First call -> paid miss; second identical call -> $0 cache hit
        r1 = asyncio.run(
            router.generate(
                role="triage_vlm",
                messages=msg,
                response_schema=_DummyOutput,
                session_id="sess_1",
                session_budget_usd=0.50,
            )
        )
        r2 = asyncio.run(
            router.generate(
                role="triage_vlm",
                messages=msg,
                response_schema=_DummyOutput,
                session_id="sess_1",
                session_budget_usd=0.50,
            )
        )
        assert r1.cache_hit is False and r1.cost_usd == pytest.approx(0.05)
        assert r2.cache_hit is True and r2.cost_usd == 0.0
        assert fake.calls == 1
        assert repo.sum_llm_cost(session_id="sess_1") == pytest.approx(0.05)

        # Concurrency cap via asyncio.Semaphore(max_concurrency=2)
        async def _burst_calls() -> None:
            await asyncio.gather(
                *[
                    router.generate(
                        role="triage_vlm",
                        messages=[Message(role="user", text=f"Prompt {i}")],
                        response_schema=_DummyOutput,
                        session_id="sess_2",
                        session_budget_usd=1.0,
                    )
                    for i in range(6)
                ]
            )

        asyncio.run(_burst_calls())
        assert fake.max_in_flight <= 2
        assert fake.calls == 7

        # Session budget exceeded
        with pytest.raises(BudgetExceededError) as exc_info:
            asyncio.run(
                router.generate(
                    role="triage_vlm",
                    messages=[Message(role="user", text="Over session budget")],
                    session_id="sess_1",
                    session_budget_usd=0.05,
                )
            )
        assert exc_info.value.scope == "session"

        # Monthly budget exceeded
        tight_cfg = AppConfig.model_validate(
            {
                **cfg.model_dump(),
                "budget": {"session_budget_usd": 10.0, "monthly_budget_usd": 0.20},
            }
        )
        router_monthly = ModelRouter(repo, paths, tight_cfg, adapters={"triage_vlm": fake})
        with pytest.raises(BudgetExceededError) as m_exc:
            asyncio.run(
                router_monthly.generate(
                    role="triage_vlm",
                    messages=[Message(role="user", text="Over monthly budget")],
                    session_id="sess_fresh",
                )
            )
        assert m_exc.value.scope == "monthly"

        # Ensure no raw secret value ever lands in llm_calls
        rows = repo.list_llm_calls(limit=50)
        assert len(rows) >= 8
        for row in rows:
            assert "sk-" not in (row.response_json or "")
            assert time.time() >= row.created_at
    finally:
        db.close()


def test_openai_strict_schema_includes_all_optional_fields() -> None:
    """OpenAI strict json_schema mode requires EVERY property (including nullable ones) in `required`."""
    yellow_schema = strict_json_schema(YellowTriageOutput)
    assert yellow_schema["additionalProperties"] is False
    assert set(yellow_schema["required"]) == set(yellow_schema["properties"].keys())
    assert "reason_code" in yellow_schema["required"]

    burst_schema = strict_json_schema(BurstTriageOutput)
    assert burst_schema["additionalProperties"] is False
    assert set(burst_schema["required"]) == set(burst_schema["properties"].keys())
    assert "eyes_open" in burst_schema["required"]


def test_refusal_is_billed_but_never_cached(tmp_path: Path) -> None:
    """Refused responses record cost_usd in llm_calls but leave cache_key=NULL so retries work."""

    class _RefuseOnceAdapter:
        def __init__(self) -> None:
            self.id = "claude-haiku-5-5"
            self.provider = "anthropic"
            self.family = "anthropic"
            self.caps = ModelCaps(vision=True, tools=True, json_schema=True, prompt_cache=True)
            self.calls = 0

        async def generate(
            self,
            messages: list[Message],
            *,
            tools: list[ToolSchema] | None = None,
            response_schema: type[BaseModel] | None = None,
            max_tokens: int = 2048,
            effort: str | None = None,
        ) -> GenerationResult:
            del messages, tools, max_tokens, effort
            self.calls += 1
            if self.calls == 1:
                return GenerationResult(
                    text="Refused",
                    parsed=None,
                    usage=Usage(in_tokens=400, out_tokens=10),
                    cost_usd=0.01,
                    stop_reason="refusal",
                    refused=True,
                    provider=self.provider,
                    model=self.id,
                )
            text = '{"decision":"keep","score":0.9}'
            parsed = response_schema.model_validate_json(text) if response_schema else None
            return GenerationResult(
                text=text,
                parsed=parsed,
                usage=Usage(in_tokens=400, out_tokens=20),
                cost_usd=0.01,
                stop_reason="end_turn",
                refused=False,
                provider=self.provider,
                model=self.id,
            )

    paths = get_app_paths(tmp_path / "home")
    paths.ensure_dirs()
    db = Database(paths.db_file)
    db.migrate()
    repo = LightboxRepository(db)
    cfg = load_config(paths)
    adapter = _RefuseOnceAdapter()
    router = ModelRouter(repo, paths, cfg, adapters={"triage_vlm": adapter})
    msg = [Message(role="user", text="Retry after refusal")]

    try:
        r1 = asyncio.run(
            router.generate(role="triage_vlm", messages=msg, response_schema=_DummyOutput, session_id="s_ref")
        )
        assert r1.refused is True and r1.cost_usd == pytest.approx(0.01)

        # Call 2 MUST retry (not hit refusal cache)
        r2 = asyncio.run(
            router.generate(role="triage_vlm", messages=msg, response_schema=_DummyOutput, session_id="s_ref")
        )
        assert r2.refused is False and r2.cache_hit is False and adapter.calls == 2
        assert isinstance(r2.parsed, _DummyOutput) and r2.parsed.decision == "keep"

        # Call 3 hits the successful non-refusal cache ($0)
        r3 = asyncio.run(
            router.generate(role="triage_vlm", messages=msg, response_schema=_DummyOutput, session_id="s_ref")
        )
        assert r3.cache_hit is True and r3.cost_usd == 0.0 and adapter.calls == 2
        assert repo.sum_llm_cost(session_id="s_ref") == pytest.approx(0.02)
    finally:
        db.close()


def test_single_jpeg_encode_pass_and_cross_router_budget_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify images are JPEG-encoded once (router only) and two routers share in-flight reservations."""
    import lightbox.models.router as router_mod
    from lightbox.models.images import strip_metadata_and_encode as real_strip

    encode_calls = 0

    def _counting_strip(*args: Any, **kwargs: Any) -> tuple[bytes, str]:
        nonlocal encode_calls
        encode_calls += 1
        return real_strip(*args, **kwargs)

    monkeypatch.setattr(router_mod, "strip_metadata_and_encode", _counting_strip)
    monkeypatch.setattr(
        "lightbox.models.anthropic_adapter.strip_metadata_and_encode",
        _counting_strip,
    )

    captured_b64: list[str] = []

    def _handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_b64.append(body["messages"][0]["content"][0]["source"]["data"])
        return httpx2.Response(
            200,
            json={
                "id": "msg_single",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": '{"decision":"keep","score":0.9}'}],
                "usage": {"input_tokens": 500, "output_tokens": 20},
            },
        )

    paths = get_app_paths(tmp_path / "home")
    paths.ensure_dirs()
    db = Database(paths.db_file)
    db.migrate()
    repo = LightboxRepository(db)
    cfg = load_config(paths)
    spec = load_models_config(paths).models["claude-haiku-5-5"]
    img = _make_jpeg_with_exif_and_gps(640, 480)

    try:
        async def _run_single_encode() -> None:
            async with httpx2.AsyncClient(transport=httpx2.MockTransport(_handler)) as http:
                adapter = AnthropicChatModel(spec, api_key="sk-test", http_client=http)
                router = ModelRouter(repo, paths, cfg, adapters={"triage_vlm": adapter})
                await router.generate(
                    role="triage_vlm",
                    messages=[Message(role="user", text="Check single pass", images=(ImageInput(data=img),))],
                    response_schema=_DummyOutput,
                    session_id="s_enc",
                )

        asyncio.run(_run_single_encode())
        # Router encodes once; AnthropicChatModel sees sanitized=True and does NOT re-encode!
        assert encode_calls == 1
        assert len(captured_b64) == 1
        with Image.open(io.BytesIO(base64.b64decode(captured_b64[0]))) as sent_im:
            assert len(sent_im.getexif()) == 0

        # Two separate ModelRouter instances sharing the same SQLite DB see each other's in-flight reservation
        slow_a = _FakeAdapter(model_id="claude-haiku-5-5", cost_usd=0.001, delay_s=0.08)
        slow_b = _FakeAdapter(model_id="claude-haiku-5-5", cost_usd=0.001, delay_s=0.08)
        router_a = ModelRouter(repo, paths, cfg, adapters={"triage_vlm": slow_a})
        router_b = ModelRouter(repo, paths, cfg, adapters={"triage_vlm": slow_b})
        # Estimate for 1 text-only call on Haiku is ~$0.001025; budget=0.0015 allows only 1 in-flight call total
        async def _run_cross_router() -> list[Any]:
            return await asyncio.gather(
                router_a.generate(
                    role="triage_vlm",
                    messages=[Message(role="user", text="Concurrent A")],
                    session_id="s_shared",
                    session_budget_usd=0.0015,
                ),
                router_b.generate(
                    role="triage_vlm",
                    messages=[Message(role="user", text="Concurrent B")],
                    session_id="s_shared",
                    session_budget_usd=0.0015,
                ),
                return_exceptions=True,
            )

        outcomes = asyncio.run(_run_cross_router())
        errors = [o for o in outcomes if isinstance(o, BudgetExceededError)]
        successes = [o for o in outcomes if isinstance(o, GenerationResult)]
        assert len(successes) == 1
        assert len(errors) == 1
    finally:
        db.close()
