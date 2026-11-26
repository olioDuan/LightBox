"""Core model protocol, message/result types, schema helpers, and cost calculation (SPEC §6.1)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast, runtime_checkable

from pydantic import BaseModel

from lightbox.config import ModelSpec

ProviderName = Literal["anthropic", "openai", "gemini", "local"]


class ModelError(RuntimeError):
    """Base error for model adapter or routing failures."""


class MissingAPIKeyError(ModelError):
    """Raised when a required cloud provider API key is not configured."""


class BudgetExceededError(ModelError):
    """Raised before making a paid model call when session or monthly budget is exhausted."""

    def __init__(
        self,
        scope: Literal["session", "monthly"],
        *,
        spent_usd: float,
        budget_usd: float,
    ) -> None:
        self.scope = scope
        self.spent_usd = spent_usd
        self.budget_usd = budget_usd
        super().__init__(
            f"budget_exceeded ({scope}): spent=${spent_usd:.4f} >= limit=${budget_usd:.4f}"
        )


@dataclass(frozen=True, slots=True)
class ModelCaps:
    """Capability flags for a configured model."""

    vision: bool = True
    tools: bool = True
    json_schema: bool = True
    prompt_cache: bool = False


@dataclass(frozen=True, slots=True)
class ImageInput:
    """In-memory image payload attached to a `Message` (sanitized before outbound send)."""

    data: bytes
    mime_type: str = "image/jpeg"
    label: str = ""
    sanitized: bool = False


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Normalized tool invocation returned by a model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    """Tool execution output paired to a prior assistant `ToolCall`."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """Provider-agnostic tool declaration passed to `ChatModel.generate`."""

    name: str
    description: str
    input_schema: dict[str, Any]
    strict: bool = False


@dataclass(frozen=True, slots=True)
class Message:
    """Normalized conversation turn across Anthropic and OpenAI-compatible adapters."""

    role: Literal["system", "user", "assistant"]
    text: str = ""
    images: tuple[ImageInput, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResultBlock, ...] = ()


@dataclass(frozen=True, slots=True)
class Usage:
    """Normalized token usage counts."""

    in_tokens: int = 0
    out_tokens: int = 0
    cached_tokens: int = 0


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Normalized model generation result."""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    parsed: Any = None
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    stop_reason: str | None = None
    refused: bool = False
    provider: str = ""
    model: str = ""
    latency_ms: int = 0
    cache_hit: bool = False
    llm_call_id: str | None = None


@runtime_checkable
class ChatModel(Protocol):
    """Thin chat/vision model protocol implemented by all adapters (SPEC §6.1)."""

    id: str
    provider: str
    family: str
    caps: ModelCaps

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        response_schema: type[BaseModel] | None = None,
        max_tokens: int = 2048,
        effort: str | None = None,
    ) -> GenerationResult: ...


def strict_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Build a strict JSON Schema with `additionalProperties: false` on all object nodes."""
    raw = model_cls.model_json_schema()
    return cast(dict[str, Any], _enforce_strict_object_schema(raw))


def _enforce_strict_object_schema(node: Any) -> Any:
    if isinstance(node, list):
        return [_enforce_strict_object_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, val in node.items():
        if key in {"$defs", "definitions", "properties"} and isinstance(val, dict):
            out[key] = {k: _enforce_strict_object_schema(v) for k, v in val.items()}
        elif key in {"anyOf", "oneOf", "allOf"} and isinstance(val, list):
            out[key] = [_enforce_strict_object_schema(item) for item in val]
        elif key == "items":
            out[key] = _enforce_strict_object_schema(val)
        else:
            out[key] = val

    if out.get("type") == "object" or "properties" in out:
        out["additionalProperties"] = False
        if "properties" in out and isinstance(out["properties"], dict):
            # OpenAI strict json_schema requires EVERY property key in `required`
            # (optional fields remain nullable via Pydantic's `anyOf: [..., {"type": "null"}]`).
            out["required"] = list(out["properties"].keys())
    return out


def compute_cost_usd(spec: ModelSpec, usage: Usage) -> float:
    """Compute USD cost from `models.toml` per-million-token pricing."""
    uncached_in = max(0, usage.in_tokens - usage.cached_tokens)
    in_cost = (uncached_in / 1_000_000.0) * spec.input_cost_per_mtok
    cached_cost = (max(0, usage.cached_tokens) / 1_000_000.0) * spec.cached_input_cost_per_mtok
    out_cost = (max(0, usage.out_tokens) / 1_000_000.0) * spec.output_cost_per_mtok
    return round(in_cost + cached_cost + out_cost, 8)


def strip_json_markdown_fences(raw_text: str) -> str:
    """Extract raw JSON payload if a local model wraps JSON inside markdown fences."""
    cleaned = raw_text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def serialize_cached_payload(result: GenerationResult) -> str:
    """Serialize response text, tool calls, and refusal state for `llm_calls.response_json`."""
    payload = {
        "text": result.text,
        "stop_reason": result.stop_reason,
        "refused": result.refused,
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in result.tool_calls
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def deserialize_cached_payload(
    response_json: str,
    *,
    response_schema: type[BaseModel] | None = None,
) -> tuple[str, tuple[ToolCall, ...], Any, str | None, bool]:
    """Rehydrate a cached `llm_calls.response_json` entry."""
    data = json.loads(response_json)
    text = str(data.get("text", ""))
    stop_reason = data.get("stop_reason")
    refused = bool(data.get("refused", False))
    tool_calls = tuple(
        ToolCall(
            id=str(tc["id"]),
            name=str(tc["name"]),
            arguments=dict(tc.get("arguments", {})),
        )
        for tc in data.get("tool_calls", [])
    )
    parsed: Any = None
    if response_schema is not None and not refused and text.strip():
        parsed = response_schema.model_validate_json(strip_json_markdown_fences(text))
    return text, tool_calls, parsed, stop_reason, refused


__all__ = [
    "BudgetExceededError",
    "ChatModel",
    "GenerationResult",
    "ImageInput",
    "Message",
    "MissingAPIKeyError",
    "ModelCaps",
    "ModelError",
    "ProviderName",
    "ToolCall",
    "ToolResultBlock",
    "ToolSchema",
    "Usage",
    "compute_cost_usd",
    "deserialize_cached_payload",
    "serialize_cached_payload",
    "strict_json_schema",
    "strip_json_markdown_fences",
]
