"""Thin ModelRouter with profile resolution, content cache, concurrency semaphore, and budget guard (SPEC §6.1–6.2)."""

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

from lightbox.config import AppConfig, ModelsConfig, ModelSpec, load_models_config
from lightbox.logging import get_logger
from lightbox.models.anthropic_adapter import AnthropicChatModel
from lightbox.models.images import strip_metadata_and_encode
from lightbox.models.openai_compat import OpenAICompatChatModel
from lightbox.models.secrets import resolve_api_key
from lightbox.models.types import (
    BudgetExceededError,
    ChatModel,
    GenerationResult,
    ImageInput,
    Message,
    ModelError,
    ToolSchema,
    Usage,
    compute_cost_usd,
    deserialize_cached_payload,
    serialize_cached_payload,
)
from lightbox.paths import AppPaths
from lightbox.store import LightboxRepository, LLMCallRecord

DEFAULT_MAX_CONCURRENCY = 4
logger = get_logger("lightbox.models.router")

# Process-wide in-flight budget reservation state keyed by resolved DB identity,
# so concurrent FastAPI / MCP requests creating separate ModelRouter instances still share reservations.
_BUDGET_THREAD_LOCK = threading.Lock()
_IN_FLIGHT_BY_DB: dict[str, float] = {}
_IN_FLIGHT_BY_SESSION: dict[tuple[str, str], float] = {}


def _db_budget_key(repo: LightboxRepository) -> str:
    raw = repo.db.db_path
    if raw == ":memory:":
        return f":memory:{id(repo.db)}"
    return str(Path(raw).resolve())


def _month_start_ts(now: float | None = None) -> float:
    dt = datetime.fromtimestamp(now if now is not None else time.time(), tz=UTC)
    start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return start.timestamp()


class ModelRouter:
    """Profile-based router managing adapters, SQLite cache, concurrency, and budgets."""

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
        self.models_config = models_config or load_models_config(app_paths)
        self.max_concurrency = max(1, max_concurrency)
        self._injected_adapters = dict(adapters or {})
        self._cached_adapters: dict[tuple[str, str], ChatModel] = {}
        self._loop_semaphores: dict[int, asyncio.Semaphore] = {}

    def _get_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        key = id(loop)
        sem = self._loop_semaphores.get(key)
        if sem is None:
            sem = asyncio.Semaphore(self.max_concurrency)
            self._loop_semaphores[key] = sem
        return sem

    def resolve_spec(self, role: str, *, profile: str | None = None) -> tuple[str, ModelSpec]:
        """Resolve `(model_alias, ModelSpec)` for a functional role and profile."""
        prof_name = profile or self.config.model_profile
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

    def get_model(self, role: str, *, profile: str | None = None) -> ChatModel:
        """Instantiate or retrieve a `ChatModel` adapter for `role`."""
        if role in self._injected_adapters:
            return self._injected_adapters[role]

        alias, spec = self.resolve_spec(role, profile=profile)
        if alias in self._injected_adapters:
            return self._injected_adapters[alias]

        prof_name = profile or self.config.model_profile
        cache_id = (prof_name, alias)
        existing = self._cached_adapters.get(cache_id)
        if existing is not None:
            return existing

        api_key = resolve_api_key(spec.provider, app_paths=self.app_paths)
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
        session_id: str | None = None,
        session_budget_usd: float | None = None,
    ) -> GenerationResult:
        """Route a request with EXIF stripping, content cache, budget checks, and concurrency cap."""
        alias, spec = self.resolve_spec(role, profile=profile)
        sanitized_messages = self._sanitize_messages(messages, max_pixels=spec.max_pixels)
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
        model = self.get_model(role, profile=profile)
        semaphore = self._get_semaphore()
        est_cost = self.estimate_call_cost(spec, sanitized_messages, max_tokens=max_tokens)
        is_paid_model = (
            spec.input_cost_per_mtok > 0.0
            or spec.output_cost_per_mtok > 0.0
            or spec.cached_input_cost_per_mtok > 0.0
        )

        session_limit = (
            session_budget_usd
            if session_budget_usd is not None
            else self.config.budget.session_budget_usd
        )
        monthly_limit = self.config.budget.monthly_budget_usd
        db_key = _db_budget_key(self.repo)

        # 3. Bounded concurrency + cross-router budget enforcement via SQLite SUM(cost_usd)
        reserved = 0.0
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
                            raise BudgetExceededError(
                                "session",
                                spent_usd=session_spent,
                                budget_usd=session_limit,
                            )

                    monthly_spent = self.repo.sum_llm_cost(since_ts=_month_start_ts())
                    db_in_flight = _IN_FLIGHT_BY_DB.get(db_key, 0.0)
                    projected_monthly = monthly_spent + db_in_flight + est_cost
                    if (
                        monthly_spent + db_in_flight >= monthly_limit
                        or projected_monthly > monthly_limit
                    ):
                        raise BudgetExceededError(
                            "monthly",
                            spent_usd=monthly_spent,
                            budget_usd=monthly_limit,
                        )
                    reserved = est_cost
                    _IN_FLIGHT_BY_DB[db_key] = db_in_flight + reserved
                    if session_id is not None:
                        _IN_FLIGHT_BY_SESSION[(db_key, session_id)] = (
                            _IN_FLIGHT_BY_SESSION.get((db_key, session_id), 0.0) + reserved
                        )

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
                        )
                    )
            finally:
                if reserved > 0.0:
                    with _BUDGET_THREAD_LOCK:
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


__all__ = ["DEFAULT_MAX_CONCURRENCY", "ModelRouter"]
