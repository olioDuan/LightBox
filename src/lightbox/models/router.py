"""Thin ModelRouter with global `[ai]` catalog synthesis, cost classification, cache, semaphore, and budget/call guard (SPEC §6.1–6.2)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from lightbox.config import (
    ActiveAIConfig,
    AppConfig,
    ModelsConfig,
    ModelSpec,
    ProfileRoles,
    load_models_config,
)
from lightbox.logging import get_logger
from lightbox.models.anthropic_adapter import AnthropicChatModel
from lightbox.models.images import check_outbound_privacy, strip_metadata_and_encode
from lightbox.models.openai_compat import OpenAICompatChatModel
from lightbox.models.provider_errors import classify_provider_error
from lightbox.models.secrets import (
    OFFICIAL_DEFAULT_ENDPOINTS,
    compute_connection_fingerprint,
    inspect_provider_secret,
    is_official_endpoint,
    normalize_endpoint,
    normalize_ollama_root_url,
    normalize_ollama_v1_url,
    resolve_api_key,
)
from lightbox.models.types import (
    BudgetExceededError,
    CallLimitExceededError,
    ChatModel,
    CostClassification,
    GenerationResult,
    ImageInput,
    Message,
    ModelCallRequest,
    ModelError,
    ModelHookDeniedError,
    PreModelCallHook,
    ToolSchema,
    Usage,
    compute_cost_usd,
    deserialize_cached_payload,
    serialize_cached_payload,
)
from lightbox.paths import AppPaths
from lightbox.store import LightboxRepository, LLMCallRecord

DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_PRICE_UNKNOWN_MAX_SESSION_CALLS = 20
ACTIVE_AI_PROFILE_NAME = "__ai__"
ACTIVE_AI_AGENT_ALIAS = "__ai_agent__"
ACTIVE_AI_TRIAGE_ALIAS = "__ai_triage__"
BUILTIN_FREE_LOCAL_ALIASES: frozenset[str] = frozenset({"siglip2-base", "mlx-vlm", "ollama-vlm"})

logger = get_logger("lightbox.models.router")

# Process-wide in-flight budget and call-count reservation state keyed by resolved DB identity,
# so concurrent FastAPI / MCP requests creating separate ModelRouter instances still share reservations.
_BUDGET_THREAD_LOCK = threading.Lock()
_IN_FLIGHT_BY_DB: dict[str, float] = {}
_IN_FLIGHT_BY_SESSION: dict[tuple[str, str], float] = {}
_IN_FLIGHT_CALLS_BY_SESSION: dict[tuple[str, str], int] = {}


def _db_budget_key(repo: LightboxRepository) -> str:
    raw = repo.db.db_path
    if raw == ":memory:":
        return f":memory:{id(repo.db)}"
    return str(Path(raw).resolve())


def _month_start_ts(now: float | None = None) -> float:
    dt = datetime.fromtimestamp(now if now is not None else time.time(), tz=UTC)
    start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return start.timestamp()


def _find_catalog_spec(
    base_models: ModelsConfig,
    *,
    provider: str,
    model_id: str,
) -> ModelSpec | None:
    """Find a matching built-in catalog ModelSpec by alias or `(provider, model_id)`."""
    direct = base_models.models.get(model_id)
    if direct is not None and direct.provider == provider:
        return direct
    for spec in base_models.models.values():
        if spec.provider == provider and spec.model_id == model_id:
            return spec
    return None


def _synthesize_ai_model_spec(
    ai_cfg: ActiveAIConfig,
    model_id: str,
    base_models: ModelsConfig,
    *,
    role: str = "agent",
) -> ModelSpec:
    """Construct a `ModelSpec` for a role under global `[ai]` configuration."""
    if ai_cfg.provider == "ollama":
        v1_url = normalize_ollama_v1_url(ai_cfg.base_url)
        return ModelSpec(
            provider="local",
            protocol="openai_compat",
            family="local",
            model_id=model_id,
            base_url=v1_url,
            input_cost_per_mtok=0.0,
            output_cost_per_mtok=0.0,
            cached_input_cost_per_mtok=0.0,
            vision=True,
            tools=True,
            json_schema=True,
            prompt_cache=False,
        )

    mapped_provider = ai_cfg.provider
    protocol: str = "anthropic" if mapped_provider == "anthropic" else "openai_compat"
    family_map = {
        "anthropic": "anthropic",
        "openai": "openai",
        "gemini": "google",
        "custom": "custom",
    }
    family = family_map[mapped_provider]

    norm_ep = normalize_endpoint(mapped_provider, ai_cfg.base_url)
    official = is_official_endpoint(mapped_provider, norm_ep)

    # Preserve canonical SDK defaults for official endpoints while carrying explicit custom base_url
    effective_base_url: str | None
    if mapped_provider == "gemini":
        effective_base_url = f"{norm_ep}/" if official else norm_ep
    elif official and ai_cfg.base_url is None:
        effective_base_url = None
    elif official:
        effective_base_url = None if mapped_provider in {"anthropic", "openai"} else norm_ep
    else:
        effective_base_url = norm_ep

    catalog_match = (
        _find_catalog_spec(base_models, provider=mapped_provider, model_id=model_id)
        if official
        else None
    )

    # Resolve role-specific pricing (paired input+output costs)
    if role == "triage" and (
        ai_cfg.triage_input_cost_per_mtok is not None
        and ai_cfg.triage_output_cost_per_mtok is not None
    ):
        in_cost = float(ai_cfg.triage_input_cost_per_mtok)
        out_cost = float(ai_cfg.triage_output_cost_per_mtok)
        cached_cost = 0.0
    elif (
        (role == "agent" or ai_cfg.triage_model == ai_cfg.agent_model)
        and ai_cfg.input_cost_per_mtok is not None
        and ai_cfg.output_cost_per_mtok is not None
    ):
        in_cost = float(ai_cfg.input_cost_per_mtok)
        out_cost = float(ai_cfg.output_cost_per_mtok)
        cached_cost = 0.0
    elif catalog_match is not None:
        in_cost = catalog_match.input_cost_per_mtok
        out_cost = catalog_match.output_cost_per_mtok
        cached_cost = catalog_match.cached_input_cost_per_mtok
    else:
        in_cost = 0.0
        out_cost = 0.0
        cached_cost = 0.0

    default_effort = catalog_match.default_effort if catalog_match is not None else None
    prompt_cache = catalog_match.prompt_cache if catalog_match is not None else (
        mapped_provider == "anthropic" and official
    )

    return ModelSpec(
        provider=mapped_provider,
        protocol="anthropic" if protocol == "anthropic" else "openai_compat",
        family=family,
        model_id=catalog_match.model_id if catalog_match is not None else model_id,
        base_url=effective_base_url,
        default_effort=default_effort,
        max_pixels=catalog_match.max_pixels if catalog_match is not None else None,
        input_cost_per_mtok=in_cost,
        output_cost_per_mtok=out_cost,
        cached_input_cost_per_mtok=cached_cost,
        vision=True,
        tools=True,
        json_schema=True,
        prompt_cache=prompt_cache,
    )


def build_effective_models_config(
    config: AppConfig,
    base_models: ModelsConfig,
) -> tuple[ModelsConfig, str]:
    """Build the unified effective model catalog & default profile name.

    When `config.ai is None`, returns `(base_models, config.model_profile)` unchanged.
    When `config.ai` is active, injects the `__ai__` profile and synthesized model specs.
    """
    if config.ai is None:
        return base_models, config.model_profile

    ai_cfg = config.ai
    agent_spec = _synthesize_ai_model_spec(ai_cfg, ai_cfg.agent_model, base_models, role="agent")
    triage_spec = _synthesize_ai_model_spec(ai_cfg, ai_cfg.triage_model, base_models, role="triage")

    models = dict(base_models.models)
    models[ACTIVE_AI_AGENT_ALIAS] = agent_spec
    models[ACTIVE_AI_TRIAGE_ALIAS] = triage_spec

    reviewer_alias = ai_cfg.reviewer or "none"
    if reviewer_alias == "agent":
        reviewer_alias = ACTIVE_AI_AGENT_ALIAS

    profiles = dict(base_models.profiles)
    profiles[ACTIVE_AI_PROFILE_NAME] = ProfileRoles(
        embedder="siglip2-base",
        triage_vlm=ACTIVE_AI_TRIAGE_ALIAS,
        agent=ACTIVE_AI_AGENT_ALIAS,
        reviewer=reviewer_alias,
        consolidate=ACTIVE_AI_AGENT_ALIAS,
    )
    return ModelsConfig(profiles=profiles, models=models), ACTIVE_AI_PROFILE_NAME


def _is_ollama_endpoint_confirmed(app_paths: AppPaths, base_url: str | None) -> bool:
    """Read local verification cache to check whether `base_url` was confirmed as Ollama via `/api/tags`."""
    state_file = app_paths.ai_connection_state_file
    if not state_file.is_file():
        return False
    try:
        raw = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(raw, dict):
        return False
    target_root = normalize_ollama_root_url(base_url)
    confirmed_list = raw.get("ollama_confirmed_endpoints")
    if isinstance(confirmed_list, list) and target_root in confirmed_list:
        return True
    if raw.get("ollama_confirmed") is True and raw.get("ollama_root_url") == target_root:
        return True
    return False


class ModelRouter:
    """Profile-based router managing adapters, SQLite cache, concurrency, cost classification, and budgets."""

    def __init__(
        self,
        repo: LightboxRepository,
        app_paths: AppPaths,
        config: AppConfig,
        *,
        models_config: ModelsConfig | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        adapters: dict[str, ChatModel] | None = None,
    ) -> None:
        self.repo = repo
        self.app_paths = app_paths
        self.config = config
        base_models = models_config or load_models_config(app_paths)
        self.base_models_config = base_models
        effective_models, default_profile = build_effective_models_config(config, base_models)
        self.models_config = effective_models
        self._default_profile_name = default_profile
        self.max_concurrency = max(1, max_concurrency)
        self._injected_adapters = dict(adapters or {})
        self._cached_adapters: dict[tuple[str, str], ChatModel] = {}
        self._fingerprint_cache: dict[tuple[str, str], str] = {}
        self._loop_semaphores: dict[int, asyncio.Semaphore] = {}
        self._pre_model_call_hooks: list[PreModelCallHook] = []
        self._pre_model_call_hook_names: set[str] = set()
        # Mount outbound privacy guard unconditionally on every ModelRouter instance (SPEC §7.4)
        self.add_pre_model_call_hook(check_outbound_privacy, name="OutboundPrivacyHook")

    def effective_profile_name(self, profile: str | None = None) -> str:
        """Resolve effective profile priority: explicit `profile` -> global `[ai]` -> `config.model_profile`."""
        if profile is not None:
            return profile
        return self._default_profile_name

    def resolve_profile(self, profile: str | None = None) -> tuple[str, ProfileRoles | None]:
        """Return `(profile_name, ProfileRoles | None)` using the unified resolution priority."""
        prof_name = self.effective_profile_name(profile)
        return prof_name, self.models_config.profiles.get(prof_name)

    def add_pre_model_call_hook(
        self,
        hook: PreModelCallHook,
        *,
        name: str | None = None,
    ) -> None:
        """Register an outbound `pre_model_call` hook (deduplicated by `name` or identity)."""
        if name is not None:
            if name in self._pre_model_call_hook_names:
                return
            self._pre_model_call_hook_names.add(name)
        elif hook in self._pre_model_call_hooks:
            return
        self._pre_model_call_hooks.append(hook)

    def has_injected_adapter(self, key: str) -> bool:
        """Return True if a test/custom adapter was injected under `key` (role or model alias)."""
        return key in self._injected_adapters

    def _get_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        key = id(loop)
        sem = self._loop_semaphores.get(key)
        if sem is None:
            sem = asyncio.Semaphore(self.max_concurrency)
            self._loop_semaphores[key] = sem
        return sem

    def resolve_spec(
        self,
        role: str,
        *,
        profile: str | None = None,
        model_alias: str | None = None,
    ) -> tuple[str, ModelSpec]:
        """Resolve `(model_alias, ModelSpec)` for a functional role and profile."""
        if model_alias is not None:
            spec = self.models_config.models.get(model_alias)
            if spec is None:
                raise ModelError(f"Model alias {model_alias!r} missing from models.toml.")
            return model_alias, spec

        prof_name = self.effective_profile_name(profile)
        prof = self.models_config.profiles.get(prof_name)
        if prof is None:
            raise ModelError(f"Unknown model profile: {prof_name!r}")
        if not hasattr(prof, role):
            raise ModelError(f"Unknown model role: {role!r}")

        alias = str(getattr(prof, role))
        if alias == "none":
            raise ModelError(f"Role {role!r} is disabled ('none') under profile {prof_name!r}.")
        spec = self.models_config.models.get(alias)
        if spec is None:
            raise ModelError(f"Model alias {alias!r} (role={role!r}) missing from models.toml.")
        return alias, spec

    def classify_cost(
        self,
        alias: str,
        spec: ModelSpec,
        *,
        ollama_confirmed: bool | None = None,
    ) -> CostClassification:
        """Classify a model's cost semantics strictly at the Router level.

        - `known_paid`: positive token pricing (built-in catalog or user-entered estimates).
        - Outside `[ai]` (legacy `models.toml` path): preserves 100% baseline behavior where
          0-cost models (such as local LM Studio or custom `models.toml` entries) remain uncapped (`known_free`).
        - Under `[ai]`:
          - `known_free`: confirmed Ollama (`/api/tags`) or explicitly flagged `is_local_free`.
          - `price_unknown`: everything else with 0 rates (including localhost/127.0.0.1 proxies by default).
        """
        if (
            spec.input_cost_per_mtok > 0.0
            or spec.output_cost_per_mtok > 0.0
            or spec.cached_input_cost_per_mtok > 0.0
        ):
            return "known_paid"

        if spec.protocol == "local_onnx":
            return "known_free"

        if alias in BUILTIN_FREE_LOCAL_ALIASES and spec.provider == "local":
            return "known_free"

        ai_cfg = self.config.ai
        if ai_cfg is not None and alias in {ACTIVE_AI_AGENT_ALIAS, ACTIVE_AI_TRIAGE_ALIAS}:
            if ai_cfg.is_local_free:
                return "known_free"
            if ai_cfg.provider == "ollama":
                confirmed = (
                    ollama_confirmed
                    if ollama_confirmed is not None
                    else _is_ollama_endpoint_confirmed(self.app_paths, ai_cfg.base_url)
                )
                if confirmed:
                    return "known_free"
            return "price_unknown"

        # Strict no-[ai] baseline compatibility: preserve legacy uncapped behavior for 0-cost models.toml entries
        return "known_free"

    def compute_active_connection_fingerprint(
        self,
        *,
        alias: str,
        spec: ModelSpec,
        profile: str | None = None,
    ) -> str:
        """Compute and cache (per-Router) the connection fingerprint recorded on `llm_calls`."""
        prof_name = self.effective_profile_name(profile)
        cache_key = (prof_name, alias)
        cached_fp = self._fingerprint_cache.get(cache_key)
        if cached_fp is not None:
            return cached_fp

        ai_cfg = self.config.ai
        if ai_cfg is not None and (
            prof_name == ACTIVE_AI_PROFILE_NAME
            or alias in {ACTIVE_AI_AGENT_ALIAS, ACTIVE_AI_TRIAGE_ALIAS}
        ):
            insp = inspect_provider_secret(
                ai_cfg.provider,
                app_paths=self.app_paths,
                base_url=ai_cfg.base_url,
            )
            fp = compute_connection_fingerprint(
                provider=ai_cfg.provider,
                base_url=ai_cfg.base_url,
                agent_model=ai_cfg.agent_model,
                triage_model=ai_cfg.triage_model,
                key_suffix=insp.key_suffix,
            )
        else:
            insp = inspect_provider_secret(
                spec.provider,
                app_paths=self.app_paths,
                base_url=spec.base_url,
            )
            fp = compute_connection_fingerprint(
                provider=spec.provider,
                base_url=spec.base_url or OFFICIAL_DEFAULT_ENDPOINTS.get(spec.provider),
                agent_model=spec.model_id,
                triage_model=spec.model_id,
                key_suffix=insp.key_suffix,
            )

        self._fingerprint_cache[cache_key] = fp
        return fp

    def get_model(
        self,
        role: str,
        *,
        profile: str | None = None,
        model_alias: str | None = None,
    ) -> ChatModel:
        """Instantiate or retrieve a `ChatModel` adapter for `role` or explicit `model_alias`."""
        if model_alias is None and role in self._injected_adapters:
            return self._injected_adapters[role]

        alias, spec = self.resolve_spec(role, profile=profile, model_alias=model_alias)
        if alias in self._injected_adapters:
            return self._injected_adapters[alias]
        if role in self._injected_adapters:
            return self._injected_adapters[role]

        prof_name = self.effective_profile_name(profile)
        cache_id = (prof_name, alias)
        existing = self._cached_adapters.get(cache_id)
        if existing is not None:
            return existing

        api_key = resolve_api_key(
            spec.provider,
            app_paths=self.app_paths,
            base_url=spec.base_url,
        )
        model: ChatModel
        if spec.protocol == "anthropic":
            model = AnthropicChatModel(spec, api_key=api_key, max_retries=3)
        elif spec.protocol == "openai_compat":
            model = OpenAICompatChatModel(spec, api_key=api_key, max_retries=3)
        else:
            raise ModelError(
                f"Model {alias!r} uses protocol {spec.protocol!r}, which is not a ChatModel."
            )
        self._cached_adapters[cache_id] = model
        return model

    def estimate_call_cost(
        self,
        spec: ModelSpec,
        messages: Sequence[Message],
        *,
        max_tokens: int = 2048,
    ) -> float:
        """Conservative pre-call USD estimate using text/image token heuristics."""
        text_chars = sum(len(m.text) for m in messages)
        num_images = sum(len(m.images) for m in messages)
        est_in = max(1, (text_chars // 4) + (num_images * 650))
        est_out = min(max_tokens, 256)
        return compute_cost_usd(spec, Usage(in_tokens=est_in, out_tokens=est_out, cached_tokens=0))

    @staticmethod
    def _sanitize_messages(
        messages: Sequence[Message],
        *,
        max_pixels: int | None,
    ) -> list[Message]:
        sanitized: list[Message] = []
        for msg in messages:
            if not msg.images:
                sanitized.append(msg)
                continue
            clean_imgs: list[ImageInput] = []
            for img in msg.images:
                if img.sanitized:
                    clean_imgs.append(img)
                    continue
                encoded, mime = strip_metadata_and_encode(
                    img.data,
                    max_pixels=max_pixels,
                    fmt="JPEG",
                    quality=88,
                )
                clean_imgs.append(
                    ImageInput(
                        data=encoded,
                        mime_type=mime,
                        label=img.label,
                        sanitized=True,
                    )
                )
            sanitized.append(
                Message(
                    role=msg.role,
                    text=msg.text,
                    images=tuple(clean_imgs),
                    tool_calls=msg.tool_calls,
                    tool_results=msg.tool_results,
                )
            )
        return sanitized

    @staticmethod
    def compute_cache_key(
        *,
        spec: ModelSpec,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None,
        response_schema: type[BaseModel] | None,
        max_tokens: int,
        effort: str | None,
    ) -> str:
        """Compute deterministic content-addressed SHA-256 key over model + messages + image bytes."""
        canonical_msgs: list[dict[str, Any]] = []
        for msg in messages:
            canonical_msgs.append(
                {
                    "role": msg.role,
                    "text": msg.text,
                    "images": [
                        {
                            "sha256": hashlib.sha256(img.data).hexdigest(),
                            "mime": img.mime_type,
                            "label": img.label,
                        }
                        for img in msg.images
                    ],
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in msg.tool_calls
                    ],
                    "tool_results": [
                        {"id": tr.tool_call_id, "content": tr.content, "is_error": tr.is_error}
                        for tr in msg.tool_results
                    ],
                }
            )
        payload = {
            "provider": spec.provider,
            "model_id": spec.model_id,
            "effort": effort or spec.default_effort,
            "max_tokens": max_tokens,
            "schema": response_schema.model_json_schema() if response_schema is not None else None,
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                    "strict": t.strict,
                }
                for t in (tools or ())
            ],
            "messages": canonical_msgs,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _record_error_call(
        self,
        *,
        role: str,
        provider: str,
        model_id: str,
        session_id: str | None,
        error_class: str,
        connection_fingerprint: str,
        latency_ms: int = 0,
    ) -> None:
        """Persist a real failed call with `cache_key=None`, `cost_usd=0.0`, `cache_hit=False`."""
        try:
            self.repo.add_llm_call(
                LLMCallRecord(
                    id=f"llm_{uuid.uuid4().hex[:12]}",
                    cache_key=None,
                    role=role,
                    provider=provider,
                    model=model_id,
                    in_tokens=0,
                    out_tokens=0,
                    cached_tokens=0,
                    cost_usd=0.0,
                    latency_ms=max(0, latency_ms),
                    response_json=None,
                    created_at=time.time(),
                    session_id=session_id,
                    cache_hit=False,
                    status="error",
                    error_class=error_class,
                    connection_fingerprint=connection_fingerprint,
                )
            )
        except Exception:
            # Never mask the primary model error if audit persistence fails
            pass

    async def generate(
        self,
        *,
        role: str,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        response_schema: type[BaseModel] | None = None,
        max_tokens: int = 2048,
        effort: str | None = None,
        profile: str | None = None,
        model_alias: str | None = None,
        session_id: str | None = None,
        session_budget_usd: float | None = None,
        bypass_monthly_budget: bool = False,
        max_session_calls: int | None = None,
    ) -> GenerationResult:
        """Route a request with EXIF stripping, pre_model_call hooks, content cache, cost guard, and concurrency cap."""
        alias, spec = self.resolve_spec(role, profile=profile, model_alias=model_alias)
        conn_fp = self.compute_active_connection_fingerprint(
            alias=alias,
            spec=spec,
            profile=profile,
        )
        sanitized_messages = self._sanitize_messages(messages, max_pixels=spec.max_pixels)

        # 0. Code-level `pre_model_call` privacy/security hooks (SPEC §7.4)
        if self._pre_model_call_hooks:
            session_roots = tuple(s.root_path for s in self.repo.list_sessions())
            hook_req = ModelCallRequest(
                role=role,
                spec=spec,
                messages=tuple(sanitized_messages),
                session_id=session_id,
                session_root_paths=session_roots,
            )
            for hook in self._pre_model_call_hooks:
                decision, reason = hook(hook_req)
                if decision != "ALLOW":
                    raise ModelHookDeniedError(
                        reason or f"Blocked by pre_model_call hook ({decision}).",
                        decision=decision,
                    )

        cache_key = self.compute_cache_key(
            spec=spec,
            messages=sanitized_messages,
            tools=tools,
            response_schema=response_schema,
            max_tokens=max_tokens,
            effort=effort,
        )

        # 1. Content-addressed SQLite cache lookup ($0 cost on hit)
        cached = self.repo.get_llm_call_by_cache_key(cache_key)
        if cached is not None and cached.response_json is not None:
            text, tool_calls, parsed, stop_reason, refused = deserialize_cached_payload(
                cached.response_json,
                response_schema=response_schema,
            )
            hit_id = f"llm_{uuid.uuid4().hex[:12]}"
            self.repo.add_llm_call(
                LLMCallRecord(
                    id=hit_id,
                    cache_key=None,
                    role=role,
                    provider=cached.provider,
                    model=cached.model,
                    in_tokens=cached.in_tokens,
                    out_tokens=cached.out_tokens,
                    cached_tokens=cached.cached_tokens,
                    cost_usd=0.0,
                    latency_ms=0,
                    response_json=None,
                    created_at=time.time(),
                    session_id=session_id,
                    cache_hit=True,
                    status="ok",
                    error_class=None,
                    connection_fingerprint=conn_fp,
                )
            )
            return GenerationResult(
                text=text,
                tool_calls=tool_calls,
                parsed=parsed,
                usage=Usage(
                    in_tokens=cached.in_tokens,
                    out_tokens=cached.out_tokens,
                    cached_tokens=cached.cached_tokens,
                ),
                cost_usd=0.0,
                stop_reason=stop_reason,
                refused=refused,
                provider=cached.provider,
                model=cached.model,
                latency_ms=0,
                cache_hit=True,
                llm_call_id=hit_id,
            )

        # 2. Ensure adapter/API key exists before checking/reserving budget
        try:
            model = self.get_model(role, profile=profile, model_alias=model_alias)
        except Exception as exc:
            mapped = classify_provider_error(spec.provider, exc)
            self._record_error_call(
                role=role,
                provider=spec.provider,
                model_id=spec.model_id,
                session_id=session_id,
                error_class=mapped.error_class,
                connection_fingerprint=conn_fp,
            )
            raise

        semaphore = self._get_semaphore()
        est_cost = self.estimate_call_cost(spec, sanitized_messages, max_tokens=max_tokens)
        cost_class = self.classify_cost(alias, spec)
        is_paid_model = cost_class == "known_paid"

        session_limit = (
            session_budget_usd
            if session_budget_usd is not None
            else self.config.budget.session_budget_usd
        )
        monthly_limit = self.config.budget.monthly_budget_usd
        effective_call_cap = (
            max_session_calls
            if max_session_calls is not None
            else (
                self.config.ai.max_session_calls
                if self.config.ai is not None
                else DEFAULT_PRICE_UNKNOWN_MAX_SESSION_CALLS
            )
        )
        db_key = _db_budget_key(self.repo)

        # 3. Bounded concurrency + cross-router budget / price_unknown call-cap enforcement
        reserved = 0.0
        reserved_call_slot = False
        async with semaphore:
            with _BUDGET_THREAD_LOCK:
                if is_paid_model:
                    if session_id is not None:
                        session_spent = self.repo.sum_llm_cost(session_id=session_id)
                        sess_in_flight = _IN_FLIGHT_BY_SESSION.get((db_key, session_id), 0.0)
                        projected_session = session_spent + sess_in_flight + est_cost
                        if (
                            session_spent + sess_in_flight >= session_limit
                            or projected_session > session_limit
                        ):
                            err = BudgetExceededError(
                                "session",
                                spent_usd=session_spent,
                                budget_usd=session_limit,
                            )
                            self._record_error_call(
                                role=role,
                                provider=spec.provider,
                                model_id=spec.model_id,
                                session_id=session_id,
                                error_class="local_budget_exceeded",
                                connection_fingerprint=conn_fp,
                            )
                            raise err

                    if not bypass_monthly_budget:
                        monthly_spent = self.repo.sum_llm_cost(since_ts=_month_start_ts())
                        db_in_flight = _IN_FLIGHT_BY_DB.get(db_key, 0.0)
                        projected_monthly = monthly_spent + db_in_flight + est_cost
                        if (
                            monthly_spent + db_in_flight >= monthly_limit
                            or projected_monthly > monthly_limit
                        ):
                            err = BudgetExceededError(
                                "monthly",
                                spent_usd=monthly_spent,
                                budget_usd=monthly_limit,
                            )
                            self._record_error_call(
                                role=role,
                                provider=spec.provider,
                                model_id=spec.model_id,
                                session_id=session_id,
                                error_class="local_budget_exceeded",
                                connection_fingerprint=conn_fp,
                            )
                            raise err
                    reserved = est_cost
                    _IN_FLIGHT_BY_DB[db_key] = _IN_FLIGHT_BY_DB.get(db_key, 0.0) + reserved
                    if session_id is not None:
                        _IN_FLIGHT_BY_SESSION[(db_key, session_id)] = (
                            _IN_FLIGHT_BY_SESSION.get((db_key, session_id), 0.0) + reserved
                        )
                elif cost_class == "price_unknown" and session_id is not None:
                    calls_done = self.repo.count_real_llm_calls(session_id=session_id, status="ok")
                    in_flight_calls = _IN_FLIGHT_CALLS_BY_SESSION.get((db_key, session_id), 0)
                    if calls_done + in_flight_calls >= effective_call_cap:
                        err_cap = CallLimitExceededError(
                            session_id=session_id,
                            calls_made=calls_done,
                            max_calls=effective_call_cap,
                        )
                        self._record_error_call(
                            role=role,
                            provider=spec.provider,
                            model_id=spec.model_id,
                            session_id=session_id,
                            error_class="session_call_limit_exceeded",
                            connection_fingerprint=conn_fp,
                        )
                        raise err_cap
                    _IN_FLIGHT_CALLS_BY_SESSION[(db_key, session_id)] = in_flight_calls + 1
                    reserved_call_slot = True

            started = time.monotonic()
            try:
                res = await model.generate(
                    sanitized_messages,
                    tools=tools,
                    response_schema=response_schema,
                    max_tokens=max_tokens,
                    effort=effort or spec.default_effort,
                )
                actual_cost = res.cost_usd
                if (
                    actual_cost == 0.0
                    and is_paid_model
                    and (res.usage.in_tokens or res.usage.out_tokens)
                ):
                    actual_cost = compute_cost_usd(spec, res.usage)

                call_id = f"llm_{uuid.uuid4().hex[:12]}"
                # Never cache refusal responses (cache_key=None) so callers can retry later,
                # while still recording usage and cost_usd accurately in the ledger.
                stored_cache_key = None if res.refused else cache_key
                record = LLMCallRecord(
                    id=call_id,
                    cache_key=stored_cache_key,
                    role=role,
                    provider=res.provider or spec.provider,
                    model=res.model or spec.model_id,
                    in_tokens=res.usage.in_tokens,
                    out_tokens=res.usage.out_tokens,
                    cached_tokens=res.usage.cached_tokens,
                    cost_usd=actual_cost,
                    latency_ms=res.latency_ms,
                    response_json=serialize_cached_payload(res),
                    created_at=time.time(),
                    session_id=session_id,
                    cache_hit=False,
                    status="ok",
                    error_class=None,
                    connection_fingerprint=conn_fp,
                )
                try:
                    self.repo.add_llm_call(record)
                except sqlite3.IntegrityError:
                    # Another concurrent identical call wrote the cache_key first; store audit ledger without unique key
                    self.repo.add_llm_call(
                        LLMCallRecord(
                            id=call_id,
                            cache_key=None,
                            role=record.role,
                            provider=record.provider,
                            model=record.model,
                            in_tokens=record.in_tokens,
                            out_tokens=record.out_tokens,
                            cached_tokens=record.cached_tokens,
                            cost_usd=record.cost_usd,
                            latency_ms=record.latency_ms,
                            response_json=None,
                            created_at=record.created_at,
                            session_id=record.session_id,
                            cache_hit=False,
                            status="ok",
                            error_class=None,
                            connection_fingerprint=conn_fp,
                        )
                    )
            except Exception as exc:
                latency_ms = max(0, int((time.monotonic() - started) * 1000))
                mapped = classify_provider_error(spec.provider, exc)
                self._record_error_call(
                    role=role,
                    provider=spec.provider,
                    model_id=spec.model_id,
                    session_id=session_id,
                    error_class=mapped.error_class,
                    connection_fingerprint=conn_fp,
                    latency_ms=latency_ms,
                )
                raise
            finally:
                if reserved > 0.0 or reserved_call_slot:
                    with _BUDGET_THREAD_LOCK:
                        if reserved > 0.0:
                            rem_db = max(0.0, _IN_FLIGHT_BY_DB.get(db_key, 0.0) - reserved)
                            if rem_db <= 1e-12:
                                _IN_FLIGHT_BY_DB.pop(db_key, None)
                            else:
                                _IN_FLIGHT_BY_DB[db_key] = rem_db
                            if session_id is not None:
                                rem_sess = max(
                                    0.0,
                                    _IN_FLIGHT_BY_SESSION.get((db_key, session_id), 0.0) - reserved,
                                )
                                if rem_sess <= 1e-12:
                                    _IN_FLIGHT_BY_SESSION.pop((db_key, session_id), None)
                                else:
                                    _IN_FLIGHT_BY_SESSION[(db_key, session_id)] = rem_sess
                        if reserved_call_slot and session_id is not None:
                            rem_calls = max(
                                0, _IN_FLIGHT_CALLS_BY_SESSION.get((db_key, session_id), 0) - 1
                            )
                            if rem_calls <= 0:
                                _IN_FLIGHT_CALLS_BY_SESSION.pop((db_key, session_id), None)
                            else:
                                _IN_FLIGHT_CALLS_BY_SESSION[(db_key, session_id)] = rem_calls

        logger.info(
            "llm_call_completed",
            role=role,
            alias=alias,
            provider=record.provider,
            model=record.model,
            in_tokens=record.in_tokens,
            out_tokens=record.out_tokens,
            cost_usd=actual_cost,
            cache_hit=False,
            session_id=session_id,
        )
        return GenerationResult(
            text=res.text,
            tool_calls=res.tool_calls,
            parsed=res.parsed,
            usage=res.usage,
            cost_usd=actual_cost,
            stop_reason=res.stop_reason,
            refused=res.refused,
            provider=record.provider,
            model=record.model,
            latency_ms=res.latency_ms,
            cache_hit=False,
            llm_call_id=call_id,
        )


__all__ = [
    "ACTIVE_AI_AGENT_ALIAS",
    "ACTIVE_AI_PROFILE_NAME",
    "ACTIVE_AI_TRIAGE_ALIAS",
    "BUILTIN_FREE_LOCAL_ALIASES",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_PRICE_UNKNOWN_MAX_SESSION_CALLS",
    "ModelRouter",
    "build_effective_models_config",
]
