"""AI connection management, verifiable capability probes, Ollama discovery, and read-only MCP client inspection."""

from __future__ import annotations

import io
import json
import os
import secrets
import shutil
import sys
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict

from lightbox.config import (
    DEFAULT_CONFIG_TOML,
    ActiveAIConfig,
    AIProviderName,
    AppConfig,
    ModelSpec,
    load_config,
    load_models_config,
)
from lightbox.config_io import atomic_update_text, atomic_write_text
from lightbox.models import (
    ACTIVE_AI_AGENT_ALIAS,
    ACTIVE_AI_TRIAGE_ALIAS,
    DEFAULT_OLLAMA_BASE_URL,
    OFFICIAL_DEFAULT_ENDPOINTS,
    PERSISTENT_ERROR_CLASSES,
    AnthropicChatModel,
    ChatModel,
    ImageInput,
    Message,
    ModelRouter,
    OpenAICompatChatModel,
    ToolSchema,
    build_effective_models_config,
    classify_provider_error,
    compute_connection_fingerprint,
    delete_provider_secret,
    inspect_provider_secret,
    mask_api_key_suffix,
    normalize_endpoint,
    normalize_ollama_root_url,
    normalize_ollama_v1_url,
    resolve_api_key_with_source,
    save_api_key_bound,
)
from lightbox.paths import AppPaths
from lightbox.store import Database, LightboxRepository

MCPClientId = Literal["claude_desktop", "cursor", "codex", "claude_code"]
ProbeAdapterFactory = Callable[[ModelSpec, str], ChatModel]

AI_PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "agent_model": "claude-sonnet-5-5",
        "triage_model": "claude-haiku-5-5",
        "default_base_url": OFFICIAL_DEFAULT_ENDPOINTS["anthropic"],
        "requires_api_key": True,
    },
    "openai": {
        "agent_model": "gpt-4o",
        "triage_model": "gpt-4o-mini",
        "default_base_url": OFFICIAL_DEFAULT_ENDPOINTS["openai"],
        "requires_api_key": True,
    },
    "gemini": {
        "agent_model": "gemini-2.5-pro",
        "triage_model": "gemini-2.5-flash",
        "default_base_url": OFFICIAL_DEFAULT_ENDPOINTS["gemini"],
        "requires_api_key": True,
    },
    "ollama": {
        "agent_model": "qwen2.5vl:7b",
        "triage_model": "qwen2.5vl:7b",
        "default_base_url": DEFAULT_OLLAMA_BASE_URL,
        "requires_api_key": False,
    },
    "custom": {
        "agent_model": "",
        "triage_model": "",
        "default_base_url": "",
        "requires_api_key": True,
    },
}

_VISION_COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red": (220, 38, 38),
    "blue": (37, 99, 235),
    "green": (22, 163, 74),
    "yellow": (234, 179, 8),
    "orange": (234, 88, 12),
    "purple": (147, 51, 234),
}

_VISION_COLOR_PAIRS: tuple[tuple[str, str], ...] = (
    ("red", "blue"),
    ("blue", "yellow"),
    ("green", "red"),
    ("purple", "yellow"),
    ("orange", "blue"),
    ("green", "purple"),
)


class _ProbeJsonOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    code: str


def _make_verifiable_vision_image(left_color: str, right_color: str) -> bytes:
    """Dynamically synthesize a tiny 128x64 PNG with distinct solid left/right color halves."""
    img = Image.new("RGB", (128, 64), _VISION_COLOR_RGB[left_color])
    right_patch = Image.new("RGB", (64, 64), _VISION_COLOR_RGB[right_color])
    img.paste(right_patch, (64, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _read_connection_state(app_paths: AppPaths) -> dict[str, Any]:
    path = app_paths.ai_connection_state_file
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_connection_state(app_paths: AppPaths, state: dict[str, Any]) -> None:
    payload = json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    atomic_write_text(app_paths, app_paths.ai_connection_state_file, payload, mode=0o600)


def _record_confirmed_ollama_endpoint(app_paths: AppPaths, root_url: str) -> None:
    state = _read_connection_state(app_paths)
    existing = state.get("ollama_confirmed_endpoints")
    endpoints: list[str] = [str(u) for u in existing if isinstance(u, str)] if isinstance(existing, list) else []
    if root_url not in endpoints:
        endpoints.append(root_url)
    state["ollama_confirmed_endpoints"] = endpoints
    _write_connection_state(app_paths, state)


def _default_http_get_json(url: str, *, timeout_s: float = 4.0) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object response.")
        return data


def _default_http_post_json(
    url: str,
    body: dict[str, Any],
    *,
    timeout_s: float = 4.0,
) -> dict[str, Any]:
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object response.")
        return data


def _build_probe_adapter(
    spec: ModelSpec,
    api_key: str,
    *,
    adapter_factory: ProbeAdapterFactory | None = None,
) -> ChatModel:
    if adapter_factory is not None:
        return adapter_factory(spec, api_key)
    if spec.protocol == "anthropic":
        return AnthropicChatModel(spec, api_key=api_key, max_retries=1)
    if spec.protocol == "openai_compat":
        return OpenAICompatChatModel(spec, api_key=api_key, max_retries=1)
    raise ValueError(f"Unsupported chat protocol for probe: {spec.protocol!r}")


async def _run_vision_probe(adapter: ChatModel) -> tuple[bool, str | None, BaseException | None]:
    left_color, right_color = secrets.choice(_VISION_COLOR_PAIRS)
    img_bytes = _make_verifiable_vision_image(left_color, right_color)
    prompt = (
        "This image is split vertically into a LEFT half and a RIGHT half, each filled with one solid color "
        "from the set {red, blue, green, yellow, orange, purple}. "
        "Reply with ONLY two lowercase English color words separated by a comma in the format: LEFT,RIGHT"
    )
    try:
        res = await adapter.generate(
            [
                Message(
                    role="user",
                    text=prompt,
                    images=(ImageInput(data=img_bytes, mime_type="image/png", label="probe_vision"),),
                )
            ],
            max_tokens=64,
        )
    except Exception as exc:
        return False, str(exc), exc

    text_lower = (res.text or "").lower()
    left_pos = text_lower.find(left_color)
    right_pos = text_lower.find(right_color, left_pos + len(left_color) if left_pos >= 0 else 0)
    if left_pos >= 0 and right_pos > left_pos:
        return True, None, None
    return (
        False,
        f"Vision answer mismatch (expected {left_color},{right_color}; got {res.text[:60]!r}).",
        None,
    )


async def _run_tools_probe(adapter: ChatModel) -> tuple[bool, str | None, BaseException | None]:
    probe_token = f"ok_{secrets.token_hex(3)}"
    tool = ToolSchema(
        name="lightbox_probe_echo",
        description="Read-only connectivity probe tool. Call this tool with the exact token string.",
        input_schema={
            "type": "object",
            "properties": {"token": {"type": "string"}},
            "required": ["token"],
            "additionalProperties": False,
        },
        strict=True,
    )
    try:
        # Adapters strictly enforce tool_choice='auto' (SPEC §6.1)
        res = await adapter.generate(
            [
                Message(
                    role="user",
                    text=(
                        f"Call the read-only tool `lightbox_probe_echo` with "
                        f'{{"token": "{probe_token}"}} to confirm tool calling works.'
                    ),
                )
            ],
            tools=[tool],
            max_tokens=128,
        )
    except Exception as exc:
        return False, str(exc), exc

    for tc in res.tool_calls:
        if tc.name == "lightbox_probe_echo" and isinstance(tc.arguments, dict):
            if str(tc.arguments.get("token", "")).strip() == probe_token:
                return True, None, None
    return False, "Model did not return a valid `lightbox_probe_echo` tool call.", None


async def _run_json_probe(adapter: ChatModel) -> tuple[bool, str | None, BaseException | None]:
    expected_code = f"lb_{secrets.token_hex(3)}"
    try:
        res = await adapter.generate(
            [
                Message(
                    role="user",
                    text=(
                        f'Return strict JSON matching the schema with ok=true and code="{expected_code}".'
                    ),
                )
            ],
            response_schema=_ProbeJsonOutput,
            max_tokens=128,
        )
    except Exception as exc:
        return False, str(exc), exc

    parsed = res.parsed
    if isinstance(parsed, _ProbeJsonOutput) and parsed.ok is True and parsed.code == expected_code:
        return True, None, None
    return False, "Structured JSON schema probe output did not match expected values.", None


_AGENT_ROLES: frozenset[str] = frozenset({"agent", "consolidate"})
_TRIAGE_ROLES: frozenset[str] = frozenset({"triage_vlm"})
# Account-level persistent errors affect the shared single API key across both roles.
_ACCOUNT_WIDE_PERSISTENT_ERRORS: frozenset[str] = frozenset(
    {"auth_failed", "insufficient_balance", "endpoint_binding_mismatch"}
)


def _derive_role_state_from_calls(
    *,
    initial_ready: bool,
    calls: list[Any],
    shared_model: bool,
    role_filter: frozenset[str],
) -> tuple[bool, dict[str, Any] | None, dict[str, Any] | None]:
    """Derive `(ready, persistent_error, transient_notice)` for a single role from real SQLite calls."""
    if not initial_ready:
        return (False, None, None)

    latest_persistent = None
    latest_transient = None
    for call in calls:
        err_cls = call.error_class or "request_failed"
        belongs_to_role = call.role in role_filter or shared_model
        affects_account = (
            call.status == "error" and err_cls in _ACCOUNT_WIDE_PERSISTENT_ERRORS
        )
        if not belongs_to_role and not affects_account:
            continue
        if call.status == "ok":
            # A real call succeeded more recently than any older errors -> healthy
            break
        if err_cls in PERSISTENT_ERROR_CLASSES:
            if latest_persistent is None:
                latest_persistent = call
        else:
            if latest_transient is None:
                latest_transient = call

    if latest_persistent is not None:
        err_cls = latest_persistent.error_class or "request_failed"
        return (
            False,
            {
                "error_class": err_cls,
                "error_code": err_cls,
                "message": f"Recent model call failed ({err_cls}).",
                "occurred_at": latest_persistent.created_at,
                "role": latest_persistent.role,
            },
            None,
        )

    transient_notice: dict[str, Any] | None = None
    if latest_transient is not None:
        transient_notice = {
            "error_class": latest_transient.error_class or "request_failed",
            "occurred_at": latest_transient.created_at,
            "role": latest_transient.role,
        }
    return (True, None, transient_notice)


def _derive_connection_status_from_db(
    *,
    repo: LightboxRepository,
    fingerprint: str,
    cached_state: dict[str, Any],
    shared_model: bool = False,
) -> tuple[
    Literal["configured_unverified", "connected", "error"],
    float | None,
    bool,
    bool,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Derive per-role and overall connection state purely from cached verification + SQLite real `llm_calls`."""
    if cached_state.get("fingerprint") != fingerprint:
        return ("configured_unverified", None, False, False, None, None, None, None, None, None)

    verified_at_raw = cached_state.get("verified_at")
    verified_at = float(verified_at_raw) if isinstance(verified_at_raw, (int, float)) else None
    init_agent_ready = bool(cached_state.get("agent_ready", False))
    init_triage_ready = bool(cached_state.get("triage_ready", False))
    agent_probes = (
        dict(cached_state["agent_probes"])
        if isinstance(cached_state.get("agent_probes"), dict)
        else None
    )
    triage_probes = (
        dict(cached_state["triage_probes"])
        if isinstance(cached_state.get("triage_probes"), dict)
        else None
    )

    if not (init_agent_ready or init_triage_ready):
        err_obj = (
            {
                "error_class": cached_state.get("last_error_class") or "request_failed",
                "error_code": cached_state.get("last_error_code"),
                "message": cached_state.get("last_error_message") or "Verification failed.",
            }
            if cached_state.get("last_error_class") or cached_state.get("last_error_message")
            else {
                "error_class": "request_failed",
                "error_code": None,
                "message": "Model capability probes did not pass.",
            }
        )
        return (
            "error",
            verified_at,
            init_agent_ready,
            init_triage_ready,
            agent_probes,
            triage_probes,
            err_obj,
            None,
            err_obj,
            err_obj,
        )

    # Check real (cache_hit = False) calls matching this exact fingerprint since verification
    recent_calls = repo.list_real_llm_calls_by_fingerprint(
        fingerprint,
        since_ts=verified_at,
        limit=100,
    )

    agent_ready, agent_error, agent_transient = _derive_role_state_from_calls(
        initial_ready=init_agent_ready,
        calls=recent_calls,
        shared_model=shared_model,
        role_filter=_AGENT_ROLES,
    )
    triage_ready, triage_error, triage_transient = _derive_role_state_from_calls(
        initial_ready=init_triage_ready,
        calls=recent_calls,
        shared_model=shared_model,
        role_filter=_TRIAGE_ROLES,
    )

    if not agent_ready and not triage_ready:
        top_error = agent_error or triage_error
        return (
            "error",
            verified_at,
            False,
            False,
            agent_probes,
            triage_probes,
            top_error,
            None,
            agent_error,
            triage_error,
        )

    # Pick the newest transient notice across healthy roles
    transient_candidates = [t for t in (agent_transient, triage_transient) if t is not None]
    transient_notice: dict[str, Any] | None = (
        max(transient_candidates, key=lambda t: float(t.get("occurred_at", 0.0)))
        if transient_candidates
        else None
    )

    return (
        "connected",
        verified_at,
        agent_ready,
        triage_ready,
        agent_probes,
        triage_probes,
        None,
        transient_notice,
        agent_error,
        triage_error,
    )


def get_ai_connection_status(app_paths: AppPaths, db: Database) -> dict[str, Any]:
    """Compute the full AI connection snapshot purely locally (zero network calls)."""
    cfg = load_config(app_paths)
    repo = LightboxRepository(db)

    if cfg.ai is None:
        return {
            "status": "unconfigured",
            "configured": False,
            "provider": None,
            "agent_model": None,
            "triage_model": None,
            "base_url": None,
            "reviewer": "none",
            "review_mode": "rules_only",
            "is_local_free": False,
            "input_cost_per_mtok": None,
            "output_cost_per_mtok": None,
            "triage_input_cost_per_mtok": None,
            "triage_output_cost_per_mtok": None,
            "cost_classification": None,
            "triage_cost_classification": None,
            "max_session_calls": 20,
            "max_triage_photos_unpriced": 20,
            "credential_source": "none",
            "can_delete_credential": False,
            "key_suffix": None,
            "fingerprint": None,
            "verified_at": None,
            "agent_ready": False,
            "triage_ready": False,
            "agent_probes": None,
            "triage_probes": None,
            "agent_error": None,
            "triage_error": None,
            "error": None,
            "transient_notice": None,
            "presets": AI_PROVIDER_PRESETS,
        }

    ai = cfg.ai
    insp = inspect_provider_secret(ai.provider, app_paths=app_paths, base_url=ai.base_url)
    try:
        norm_ep = normalize_endpoint(ai.provider, ai.base_url)
    except ValueError:
        norm_ep = ai.base_url or ""

    fingerprint = compute_connection_fingerprint(
        provider=ai.provider,
        base_url=ai.base_url,
        agent_model=ai.agent_model,
        triage_model=ai.triage_model,
        key_suffix=insp.key_suffix,
    )
    router = ModelRouter(repo, app_paths, cfg)
    _a_alias, agent_spec = router.resolve_spec("agent")
    _t_alias, triage_spec = router.resolve_spec("triage_vlm")
    cost_class = router.classify_cost(ACTIVE_AI_AGENT_ALIAS, agent_spec)
    triage_cost_class = router.classify_cost(ACTIVE_AI_TRIAGE_ALIAS, triage_spec)
    review_mode = "rules_only" if (ai.reviewer or "none") == "none" else "two_layer"
    can_delete_cred = insp.source in {"keyring", "dotenv"}

    if insp.endpoint_mismatch:
        status: Literal["unconfigured", "configured_unverified", "connected", "error"] = "error"
        verified_at = None
        agent_ready = False
        triage_ready = False
        agent_probes = None
        triage_probes = None
        error_obj: dict[str, Any] | None = {
            "error_class": "endpoint_binding_mismatch",
            "error_code": "endpoint_binding_mismatch",
            "message": insp.mismatch_reason or "Stored API key is bound to a different endpoint.",
        }
        agent_error = error_obj
        triage_error = error_obj
        transient_notice = None
    elif ai.provider != "ollama" and not insp.has_key:
        status = "error"
        verified_at = None
        agent_ready = False
        triage_ready = False
        agent_probes = None
        triage_probes = None
        error_obj = {
            "error_class": "auth_failed",
            "error_code": "missing_api_key",
            "message": f"Missing API key for provider {ai.provider!r}.",
        }
        agent_error = error_obj
        triage_error = error_obj
        transient_notice = None
    else:
        cached_state = _read_connection_state(app_paths)
        (
            status,
            verified_at,
            agent_ready,
            triage_ready,
            agent_probes,
            triage_probes,
            error_obj,
            transient_notice,
            agent_error,
            triage_error,
        ) = _derive_connection_status_from_db(
            repo=repo,
            fingerprint=fingerprint,
            cached_state=cached_state,
            shared_model=(ai.agent_model == ai.triage_model),
        )

    return {
        "status": status,
        "configured": True,
        "provider": ai.provider,
        "agent_model": ai.agent_model,
        "triage_model": ai.triage_model,
        "base_url": ai.base_url if ai.provider != "ollama" else normalize_ollama_root_url(ai.base_url),
        "normalized_endpoint": norm_ep,
        "reviewer": ai.reviewer,
        "review_mode": review_mode,
        "is_local_free": ai.is_local_free,
        "input_cost_per_mtok": ai.input_cost_per_mtok,
        "output_cost_per_mtok": ai.output_cost_per_mtok,
        "triage_input_cost_per_mtok": ai.triage_input_cost_per_mtok,
        "triage_output_cost_per_mtok": ai.triage_output_cost_per_mtok,
        "cost_classification": cost_class,
        "triage_cost_classification": triage_cost_class,
        "max_session_calls": ai.max_session_calls,
        "max_triage_photos_unpriced": ai.max_triage_photos_unpriced,
        "credential_source": insp.source,
        "can_delete_credential": can_delete_cred,
        "key_suffix": insp.key_suffix,
        "fingerprint": fingerprint,
        "verified_at": verified_at,
        "agent_ready": agent_ready,
        "triage_ready": triage_ready,
        "agent_probes": agent_probes,
        "triage_probes": triage_probes,
        "agent_error": agent_error,
        "triage_error": triage_error,
        "error": error_obj,
        "transient_notice": transient_notice,
        "presets": AI_PROVIDER_PRESETS,
    }


def _resolve_candidate_models(
    provider: AIProviderName,
    agent_model: str | None,
    triage_model: str | None,
) -> tuple[str, str]:
    preset = AI_PROVIDER_PRESETS[provider]
    resolved_agent = (agent_model or "").strip() or str(preset["agent_model"])
    if not resolved_agent:
        raise ValueError(f"agent_model is required for provider {provider!r}.")
    resolved_triage = (triage_model or "").strip() or str(preset["triage_model"]) or resolved_agent
    return resolved_agent, resolved_triage


def _parse_config_text(existing_text: str | None) -> AppConfig:
    raw = (
        tomllib.loads(existing_text)
        if existing_text is not None and existing_text.strip()
        else tomllib.loads(DEFAULT_CONFIG_TOML)
    )
    return AppConfig.model_validate(raw)


def save_ai_connection(
    app_paths: AppPaths,
    db: Database,
    *,
    dump_config_fn: Callable[[AppConfig], str],
    provider: AIProviderName,
    agent_model: str | None = None,
    triage_model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    reviewer: str = "none",
    is_local_free: bool = False,
    input_cost_per_mtok: float | None = None,
    output_cost_per_mtok: float | None = None,
    triage_input_cost_per_mtok: float | None = None,
    triage_output_cost_per_mtok: float | None = None,
    max_session_calls: int = 20,
    max_triage_photos_unpriced: int = 20,
) -> dict[str, Any]:
    """Save `[ai]` connection enforcing strict write order: secret+endpoint binding FIRST, then locked read-modify-write `config.toml`."""
    resolved_agent, resolved_triage = _resolve_candidate_models(provider, agent_model, triage_model)
    cleaned_base = (base_url or "").strip() or None
    if provider == "ollama":
        stored_base_url: str | None = normalize_ollama_root_url(cleaned_base)
    elif provider == "custom":
        stored_base_url = normalize_endpoint("custom", cleaned_base)
    else:
        # Validate base_url if supplied; store canonical URL or None if matching official default
        norm_ep = normalize_endpoint(provider, cleaned_base)
        stored_base_url = (
            None
            if cleaned_base is None or norm_ep == OFFICIAL_DEFAULT_ENDPOINTS.get(provider)
            else norm_ep
        )

    # Validate new ActiveAIConfig upfront (e.g. paired cost check) before writing secrets
    new_ai = ActiveAIConfig(
        provider=provider,
        agent_model=resolved_agent,
        triage_model=resolved_triage,
        base_url=stored_base_url,
        reviewer=reviewer,
        is_local_free=is_local_free,
        input_cost_per_mtok=input_cost_per_mtok,
        output_cost_per_mtok=output_cost_per_mtok,
        triage_input_cost_per_mtok=triage_input_cost_per_mtok,
        triage_output_cost_per_mtok=triage_output_cost_per_mtok,
        max_session_calls=max_session_calls,
        max_triage_photos_unpriced=max_triage_photos_unpriced,
    )

    current_cfg = load_config(app_paths)
    prev_base_url = (
        current_cfg.ai.base_url
        if current_cfg.ai is not None and current_cfg.ai.provider == provider
        else None
    )

    # 1. Write secret + endpoint binding FIRST (or verify existing bound key matches target endpoint)
    if provider != "ollama":
        cleaned_key = (api_key or "").strip()
        if cleaned_key:
            save_api_key_bound(
                provider,
                cleaned_key,
                app_paths,
                base_url=stored_base_url,
                previous_base_url=prev_base_url,
            )
        else:
            # Will raise MissingAPIKeyError or EndpointBindingError if missing or endpoint changed!
            resolve_api_key_with_source(
                provider,
                app_paths=app_paths,
                base_url=stored_base_url,
            )

    # 2. Locked read -> merge -> validate -> atomic write `config.toml` SECOND
    def _merge_ai(existing_text: str | None) -> str:
        latest_cfg = _parse_config_text(existing_text)
        updated_cfg = AppConfig.model_validate({**latest_cfg.model_dump(), "ai": new_ai.model_dump()})
        return dump_config_fn(updated_cfg)

    atomic_update_text(app_paths, app_paths.config_file, _merge_ai)

    return get_ai_connection_status(app_paths, db)


def delete_ai_connection(
    app_paths: AppPaths,
    db: Database,
    *,
    dump_config_fn: Callable[[AppConfig], str],
    delete_credential: bool = True,
) -> dict[str, Any]:
    """Deactivate `[ai]` inside config_io lock, delete non-env stored credentials, and re-resolve remaining source."""
    prev_ai_holder: list[ActiveAIConfig | None] = [None]

    def _clear_ai(existing_text: str | None) -> str:
        latest_cfg = _parse_config_text(existing_text)
        prev_ai_holder[0] = latest_cfg.ai
        updated_cfg = AppConfig.model_validate({**latest_cfg.model_dump(), "ai": None})
        return dump_config_fn(updated_cfg)

    atomic_update_text(app_paths, app_paths.config_file, _clear_ai)
    prev_ai = prev_ai_holder[0]

    credential_deleted = False
    remaining_source = "none"
    remaining_suffix: str | None = None
    remaining_can_delete = False

    if prev_ai is not None:
        if delete_credential and prev_ai.provider != "ollama":
            credential_deleted = delete_provider_secret(
                prev_ai.provider,
                app_paths,
                base_url=prev_ai.base_url,
            )
        # Re-resolve remaining effective credential source for the previous provider
        if prev_ai.provider != "ollama":
            rem = inspect_provider_secret(
                prev_ai.provider,
                app_paths=app_paths,
                base_url=prev_ai.base_url,
            )
            if not rem.has_key and prev_ai.provider in OFFICIAL_DEFAULT_ENDPOINTS:
                rem = inspect_provider_secret(
                    prev_ai.provider,
                    app_paths=app_paths,
                    base_url=None,
                )
            remaining_source = rem.source
            remaining_suffix = rem.key_suffix
            remaining_can_delete = rem.source in {"keyring", "dotenv"}

    snapshot = get_ai_connection_status(app_paths, db)
    snapshot["deactivated_provider"] = prev_ai.provider if prev_ai is not None else None
    snapshot["credential_deleted"] = credential_deleted
    snapshot["remaining_credential_source"] = remaining_source
    snapshot["remaining_key_suffix"] = remaining_suffix
    snapshot["remaining_can_delete_credential"] = remaining_can_delete
    return snapshot


async def test_ai_connection(
    app_paths: AppPaths,
    db: Database,
    *,
    provider: AIProviderName | None = None,
    agent_model: str | None = None,
    triage_model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    is_local_free: bool | None = None,
    adapter_factory: ProbeAdapterFactory | None = None,
    http_get_json: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute verifiable probes for Agent and Triage models, map errors, and persist fingerprint state."""
    current_cfg = load_config(app_paths)
    saved_ai = current_cfg.ai

    eff_provider = provider or (saved_ai.provider if saved_ai is not None else None)
    if eff_provider is None:
        raise ValueError("No provider specified and no active [ai] connection configured.")

    eff_agent_raw = agent_model if agent_model is not None else (
        saved_ai.agent_model if saved_ai is not None and saved_ai.provider == eff_provider else None
    )
    eff_triage_raw = triage_model if triage_model is not None else (
        saved_ai.triage_model if saved_ai is not None and saved_ai.provider == eff_provider else None
    )
    eff_agent, eff_triage = _resolve_candidate_models(eff_provider, eff_agent_raw, eff_triage_raw)

    eff_base_raw = (
        base_url
        if base_url is not None
        else (saved_ai.base_url if saved_ai is not None and saved_ai.provider == eff_provider else None)
    )
    eff_base = (eff_base_raw or "").strip() or None
    eff_is_local_free = (
        is_local_free
        if is_local_free is not None
        else (saved_ai.is_local_free if saved_ai is not None else False)
    )

    # Validate endpoint & resolve secret (enforcing strict endpoint binding!)
    if eff_provider == "ollama":
        ollama_root = normalize_ollama_root_url(eff_base)
        resolved_key = (api_key or "").strip() or "local-no-key"
        key_suffix = mask_api_key_suffix(resolved_key)
        eff_base_norm: str | None = ollama_root
    else:
        norm_ep = normalize_endpoint(eff_provider, eff_base)
        eff_base_norm = (
            None
            if eff_base is None or norm_ep == OFFICIAL_DEFAULT_ENDPOINTS.get(eff_provider)
            else norm_ep
        )
        cleaned_key = (api_key or "").strip()
        if cleaned_key:
            resolved_key = cleaned_key
            key_suffix = mask_api_key_suffix(cleaned_key)
        else:
            secret_rec = resolve_api_key_with_source(
                eff_provider,
                app_paths=app_paths,
                base_url=eff_base_norm,
            )
            resolved_key = secret_rec.api_key
            key_suffix = secret_rec.key_suffix

    ollama_confirmed = False
    if eff_provider == "ollama":
        getter = http_get_json or _default_http_get_json
        ollama_root = normalize_ollama_root_url(eff_base_norm)
        try:
            tags_data = getter(f"{ollama_root}/api/tags")
            if isinstance(tags_data.get("models"), list):
                ollama_confirmed = True
                _record_confirmed_ollama_endpoint(app_paths, ollama_root)
        except Exception:
            ollama_confirmed = False

    # Synthesize effective candidate specs using the unified catalog builder
    candidate_ai = ActiveAIConfig(
        provider=eff_provider,
        agent_model=eff_agent,
        triage_model=eff_triage,
        base_url=eff_base_norm,
        reviewer="none",
        is_local_free=eff_is_local_free,
    )
    candidate_cfg = AppConfig.model_validate(
        {**current_cfg.model_dump(), "ai": candidate_ai.model_dump()}
    )
    base_models = load_models_config(app_paths)
    eff_models, _ = build_effective_models_config(candidate_cfg, base_models)
    agent_spec = eff_models.models[ACTIVE_AI_AGENT_ALIAS]
    triage_spec = eff_models.models[ACTIVE_AI_TRIAGE_ALIAS]

    agent_adapter = _build_probe_adapter(agent_spec, resolved_key, adapter_factory=adapter_factory)

    first_exc: BaseException | None = None
    first_detail: str | None = None

    def _capture_err(detail: str | None, exc: BaseException | None) -> bool:
        nonlocal first_exc, first_detail
        if exc is not None and first_exc is None:
            first_exc = exc
        if detail and first_detail is None:
            first_detail = detail
        if exc is not None:
            mapped = classify_provider_error(eff_provider, exc)
            return mapped.is_persistent
        return False

    # 1. Agent model probes: Vision + Tools + JSON Schema
    a_vis_ok, a_vis_err, a_vis_exc = await _run_vision_probe(agent_adapter)
    abort_agent = _capture_err(a_vis_err, a_vis_exc)

    if abort_agent:
        a_tools_ok, a_tools_err = False, a_vis_err
        a_json_ok, a_json_err = False, a_vis_err
    else:
        a_tools_ok, a_tools_err, a_tools_exc = await _run_tools_probe(agent_adapter)
        abort_agent = _capture_err(a_tools_err, a_tools_exc)
        if abort_agent:
            a_json_ok, a_json_err = False, a_tools_err
        else:
            a_json_ok, a_json_err, a_json_exc = await _run_json_probe(agent_adapter)
            _capture_err(a_json_err, a_json_exc)

    agent_probes = {
        "vision": {"ok": a_vis_ok, "error": a_vis_err},
        "tools": {"ok": a_tools_ok, "error": a_tools_err},
        "json_schema": {"ok": a_json_ok, "error": a_json_err},
    }
    agent_ready = a_vis_ok and a_tools_ok and a_json_ok

    # 2. Triage model probes: share results when model_id matches Agent model!
    shared_with_agent = eff_triage == eff_agent
    if shared_with_agent:
        t_vis_ok, t_vis_err = a_vis_ok, a_vis_err
        t_json_ok, t_json_err = a_json_ok, a_json_err
    else:
        triage_adapter = _build_probe_adapter(
            triage_spec,
            resolved_key,
            adapter_factory=adapter_factory,
        )
        t_vis_ok, t_vis_err, t_vis_exc = await _run_vision_probe(triage_adapter)
        abort_triage = _capture_err(t_vis_err, t_vis_exc)
        if abort_triage:
            t_json_ok, t_json_err = False, t_vis_err
        else:
            t_json_ok, t_json_err, t_json_exc = await _run_json_probe(triage_adapter)
            _capture_err(t_json_err, t_json_exc)

    triage_probes = {
        "vision": {"ok": t_vis_ok, "error": t_vis_err},
        "json_schema": {"ok": t_json_ok, "error": t_json_err},
        "shared_with_agent": shared_with_agent,
    }
    triage_ready = t_vis_ok and t_json_ok
    overall_ok = agent_ready or triage_ready

    mapped_err = classify_provider_error(eff_provider, first_exc) if first_exc is not None else None
    error_class = mapped_err.error_class if mapped_err is not None else (
        None if overall_ok else "capability_probe_failed"
    )
    error_code = mapped_err.raw_code if mapped_err is not None else None
    error_message = (
        mapped_err.message
        if mapped_err is not None
        else (None if overall_ok else first_detail)
    )
    is_persistent_error = mapped_err.is_persistent if mapped_err is not None else False

    fingerprint = compute_connection_fingerprint(
        provider=eff_provider,
        base_url=eff_base_norm,
        agent_model=eff_agent,
        triage_model=eff_triage,
        key_suffix=key_suffix,
    )
    verified_at = time.time()

    # Persist verification fingerprint & probe outcomes atomically (preserving confirmed ollama endpoints)
    existing_state = _read_connection_state(app_paths)
    confirmed_eps = existing_state.get("ollama_confirmed_endpoints")
    endpoints_list = (
        [str(u) for u in confirmed_eps if isinstance(u, str)]
        if isinstance(confirmed_eps, list)
        else []
    )
    if eff_provider == "ollama" and ollama_confirmed:
        o_root = normalize_ollama_root_url(eff_base_norm)
        if o_root not in endpoints_list:
            endpoints_list.append(o_root)

    new_state: dict[str, Any] = {
        "fingerprint": fingerprint,
        "verified_at": verified_at,
        "overall_ok": overall_ok,
        "agent_ready": agent_ready,
        "triage_ready": triage_ready,
        "agent_probes": agent_probes,
        "triage_probes": triage_probes,
        "ollama_confirmed": ollama_confirmed,
        "ollama_root_url": normalize_ollama_root_url(eff_base_norm) if eff_provider == "ollama" else None,
        "ollama_confirmed_endpoints": endpoints_list,
        "last_error_class": error_class,
        "last_error_code": error_code,
        "last_error_message": error_message,
    }
    _write_connection_state(app_paths, new_state)

    router = ModelRouter(LightboxRepository(db), app_paths, candidate_cfg)
    cost_class = router.classify_cost(
        ACTIVE_AI_AGENT_ALIAS,
        agent_spec,
        ollama_confirmed=ollama_confirmed if eff_provider == "ollama" else None,
    )

    return {
        "overall_ok": overall_ok,
        "status": "connected" if overall_ok else "error",
        "provider": eff_provider,
        "agent_model": eff_agent,
        "triage_model": eff_triage,
        "agent_ready": agent_ready,
        "triage_ready": triage_ready,
        "agent_probes": agent_probes,
        "triage_probes": triage_probes,
        "ollama_confirmed": ollama_confirmed,
        "cost_classification": cost_class,
        "fingerprint": fingerprint,
        "verified_at": verified_at,
        "error_class": error_class,
        "error_code": error_code,
        "error_message": error_message,
        "is_persistent_error": is_persistent_error,
    }


def _extract_ollama_context_limit(show_payload: dict[str, Any]) -> int | None:
    model_info = show_payload.get("model_info")
    if isinstance(model_info, dict):
        for k, v in model_info.items():
            if isinstance(k, str) and k.endswith(".context_length") and isinstance(v, int) and v > 0:
                return v
    params = show_payload.get("parameters")
    if isinstance(params, str):
        for line in params.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == "num_ctx":
                try:
                    val = int(parts[1])
                    if val > 0:
                        return val
                except ValueError:
                    pass
    return None


def discover_ollama_models(
    app_paths: AppPaths,
    *,
    base_url: str | None = None,
    http_get_json: Callable[[str], dict[str, Any]] | None = None,
    http_post_json: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Query Ollama `/api/tags` and `/api/show` from its service root URL."""
    root = normalize_ollama_root_url(base_url)
    v1_url = normalize_ollama_v1_url(root)
    getter = http_get_json or _default_http_get_json
    poster = http_post_json or _default_http_post_json

    context_note = (
        "/api/show reports the model's theoretical maximum context length, not the runtime num_ctx. "
        "If truncation occurs during chat or triage, increase Ollama's runtime num_ctx."
    )

    try:
        tags = getter(f"{root}/api/tags")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return {
            "base_url": root,
            "openai_v1_url": v1_url,
            "reachable": False,
            "ollama_confirmed": False,
            "models": [],
            "context_note": context_note,
            "error": str(exc),
        }

    raw_models = tags.get("models")
    if not isinstance(raw_models, list):
        return {
            "base_url": root,
            "openai_v1_url": v1_url,
            "reachable": True,
            "ollama_confirmed": False,
            "models": [],
            "context_note": context_note,
            "error": "Endpoint responded to /api/tags without an Ollama `models` list.",
        }

    _record_confirmed_ollama_endpoint(app_paths, root)

    discovered: list[dict[str, Any]] = []
    for entry in raw_models:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or entry.get("model") or "").strip()
        if not name:
            continue
        size_bytes = entry.get("size")
        ctx_limit: int | None = None
        try:
            show_data = poster(f"{root}/api/show", {"name": name})
            ctx_limit = _extract_ollama_context_limit(show_data)
        except Exception:
            ctx_limit = None
        discovered.append(
            {
                "name": name,
                "size_bytes": int(size_bytes) if isinstance(size_bytes, int) else None,
                "context_length_limit": ctx_limit,
            }
        )

    return {
        "base_url": root,
        "openai_v1_url": v1_url,
        "reachable": True,
        "ollama_confirmed": True,
        "models": discovered,
        "context_note": context_note,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Read-only External MCP Client Detection & Safe Snippet Generation
# (NEVER returns or logs raw external config file contents!)
# ---------------------------------------------------------------------------


def _mcp_client_config_path(
    client: MCPClientId,
    *,
    home_dir: Path | None = None,
    platform_name: str | None = None,
) -> Path:
    home = (home_dir or Path.home()).expanduser()
    plat = platform_name or sys.platform
    if client == "claude_desktop":
        if plat == "darwin":
            return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        if plat == "win32":
            appdata = os.environ.get("APPDATA")
            base = Path(appdata) if appdata else (home / "AppData" / "Roaming")
            return base / "Claude" / "claude_desktop_config.json"
        return home / ".config" / "Claude" / "claude_desktop_config.json"
    if client == "cursor":
        return home / ".cursor" / "mcp.json"
    if client == "codex":
        return home / ".codex" / "config.toml"
    return home / ".claude.json"


def _command_exists(cmd: str) -> bool:
    cleaned = cmd.strip()
    if not cleaned:
        return False
    p = Path(cleaned).expanduser()
    if p.is_absolute() or os.sep in cleaned or (os.altsep and os.altsep in cleaned):
        return p.is_file()
    return shutil.which(cleaned) is not None


def _inspect_mcp_server_entry(entry: dict[str, Any]) -> tuple[bool, bool]:
    """Validate `command` existence and `--directory <dir>` path existence without leaking values."""
    cmd_raw = entry.get("command")
    cmd_ok = isinstance(cmd_raw, str) and _command_exists(cmd_raw)

    args_raw = entry.get("args")
    dir_ok = True
    if isinstance(args_raw, list):
        str_args = [str(a) for a in args_raw]
        for idx, arg in enumerate(str_args):
            if arg == "--directory" and idx + 1 < len(str_args):
                dir_ok = Path(str_args[idx + 1]).expanduser().is_dir()
                break
            if arg.startswith("--directory="):
                dir_ok = Path(arg.split("=", 1)[1]).expanduser().is_dir()
                break
    return cmd_ok, dir_ok


def detect_mcp_clients(
    *,
    home_dir: Path | None = None,
    platform_name: str | None = None,
) -> dict[str, Any]:
    """Read-only inspection of local MCP client configurations (zero raw content returned or logged)."""
    clients: tuple[MCPClientId, ...] = ("claude_desktop", "cursor", "codex", "claude_code")
    results: list[dict[str, Any]] = []

    for cid in clients:
        cfg_path = _mcp_client_config_path(cid, home_dir=home_dir, platform_name=platform_name)
        config_exists = cfg_path.is_file()
        parent_exists = cfg_path.parent.is_dir()

        if not config_exists:
            status = "not_configured" if (parent_exists or cid == "claude_code") else "not_installed"
            results.append(
                {
                    "client_id": cid,
                    "status": status,
                    "config_exists": False,
                    "lightbox_configured": False,
                    "command_valid": False,
                    "directory_valid": False,
                    "config_path": str(cfg_path),
                }
            )
            continue

        # Read-only parse; NEVER overwrite on failure and NEVER return or log parsed content
        try:
            raw_text = cfg_path.read_text(encoding="utf-8")
            parsed: Any = tomllib.loads(raw_text) if cid == "codex" else json.loads(raw_text)
        except Exception:
            results.append(
                {
                    "client_id": cid,
                    "status": "parse_error",
                    "config_exists": True,
                    "lightbox_configured": False,
                    "command_valid": False,
                    "directory_valid": False,
                    "config_path": str(cfg_path),
                }
            )
            continue

        if not isinstance(parsed, dict):
            results.append(
                {
                    "client_id": cid,
                    "status": "parse_error",
                    "config_exists": True,
                    "lightbox_configured": False,
                    "command_valid": False,
                    "directory_valid": False,
                    "config_path": str(cfg_path),
                }
            )
            continue

        servers = parsed.get("mcp_servers") if cid == "codex" else parsed.get("mcpServers")
        lb_entry = servers.get("lightbox") if isinstance(servers, dict) else None
        if not isinstance(lb_entry, dict):
            results.append(
                {
                    "client_id": cid,
                    "status": "not_configured",
                    "config_exists": True,
                    "lightbox_configured": False,
                    "command_valid": False,
                    "directory_valid": False,
                    "config_path": str(cfg_path),
                }
            )
            continue

        cmd_ok, dir_ok = _inspect_mcp_server_entry(lb_entry)
        results.append(
            {
                "client_id": cid,
                "status": "configured" if (cmd_ok and dir_ok) else "invalid_path",
                "config_exists": True,
                "lightbox_configured": True,
                "command_valid": cmd_ok,
                "directory_valid": dir_ok,
                "config_path": str(cfg_path),
            }
        )

    return {"clients": results}


def _resolve_mcp_launch_spec() -> tuple[str, list[str]]:
    """Derive backend-controlled MCP launch `(command, args)` without accepting untrusted input."""
    pkg_src = Path(__file__).resolve().parent.parent.parent  # .../src
    repo_root = pkg_src.parent
    is_dev_checkout = (repo_root / "pyproject.toml").is_file() and (
        repo_root / "src" / "lightbox"
    ).is_dir()

    if is_dev_checkout:
        uv_bin = shutil.which("uv") or "uv"
        return uv_bin, ["--directory", str(repo_root), "run", "lightbox", "mcp", "--transport", "stdio"]

    uvx_bin = shutil.which("uvx") or "uvx"
    return uvx_bin, ["--from", "lightbox-cull", "lightbox", "mcp", "--transport", "stdio"]


def generate_mcp_snippet(
    client: MCPClientId,
    *,
    home_dir: Path | None = None,
    platform_name: str | None = None,
) -> dict[str, Any]:
    """Generate a backend-controlled copy-paste configuration snippet for an external MCP client."""
    cmd, args = _resolve_mcp_launch_spec()
    cfg_path = _mcp_client_config_path(client, home_dir=home_dir, platform_name=platform_name)

    if client == "claude_code":
        quoted_parts = [json.dumps(p) if " " in p else p for p in [cmd, *args]]
        snippet = f"claude mcp add --scope user lightbox -- {' '.join(quoted_parts)}"
        return {
            "client_id": client,
            "format": "shell",
            "config_path": str(cfg_path),
            "restart_required": False,
            "snippet": snippet,
        }

    if client == "codex":
        args_toml = ", ".join(json.dumps(a, ensure_ascii=False) for a in args)
        snippet = (
            "[mcp_servers.lightbox]\n"
            f"command = {json.dumps(cmd, ensure_ascii=False)}\n"
            f"args = [{args_toml}]\n"
        )
        return {
            "client_id": client,
            "format": "toml",
            "config_path": str(cfg_path),
            "restart_required": True,
            "snippet": snippet,
        }

    # claude_desktop / cursor -> JSON mcpServers
    payload = {
        "mcpServers": {
            "lightbox": {
                "command": cmd,
                "args": args,
            }
        }
    }
    return {
        "client_id": client,
        "format": "json",
        "config_path": str(cfg_path),
        "restart_required": True,
        "snippet": json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    }


__all__ = [
    "AI_PROVIDER_PRESETS",
    "MCPClientId",
    "ProbeAdapterFactory",
    "delete_ai_connection",
    "detect_mcp_clients",
    "discover_ollama_models",
    "generate_mcp_snippet",
    "get_ai_connection_status",
    "save_ai_connection",
    "test_ai_connection",
]
