"""Anthropic Messages API adapter strictly adhering to Claude 5.5 constraints (SPEC §6.1)."""

from __future__ import annotations

import base64
import time
from typing import Any, cast

import anthropic
from pydantic import BaseModel

from lightbox.config import ModelSpec
from lightbox.models.images import strip_metadata_and_encode
from lightbox.models.provider_errors import extract_sdk_error_meta
from lightbox.models.types import (
    GenerationResult,
    Message,
    ModelCaps,
    ModelError,
    ToolCall,
    ToolSchema,
    Usage,
    compute_cost_usd,
    strict_json_schema,
    strip_json_markdown_fences,
)


class AnthropicChatModel:
    """Native `anthropic.AsyncAnthropic` adapter respecting SPEC §6.1 5.5 API constraints."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        api_key: str,
        max_retries: int = 3,
        http_client: Any = None,
    ) -> None:
        self.spec = spec
        self.id: str = spec.model_id
        self.provider: str = spec.provider
        self.family: str = spec.family
        self.caps = ModelCaps(
            vision=spec.vision,
            tools=spec.tools,
            json_schema=spec.json_schema,
            prompt_cache=spec.prompt_cache,
        )
        self._api_key = api_key
        self._max_retries = max_retries
        self._http_client = http_client

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        response_schema: type[BaseModel] | None = None,
        max_tokens: int = 2048,
        effort: str | None = None,
    ) -> GenerationResult:
        system_parts: list[str] = []
        wire_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                if msg.text.strip():
                    system_parts.append(msg.text)
                continue

            blocks: list[dict[str, Any]] = []
            if msg.role == "user":
                for tr in msg.tool_results:
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tr.tool_call_id,
                            "content": tr.content,
                            "is_error": tr.is_error,
                        }
                    )
                for img in msg.images:
                    if img.sanitized:
                        clean_bytes, media_type = img.data, img.mime_type
                    else:
                        clean_bytes, media_type = strip_metadata_and_encode(
                            img.data,
                            max_pixels=self.spec.max_pixels,
                            fmt="JPEG",
                        )
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64.b64encode(clean_bytes).decode("ascii"),
                            },
                        }
                    )
                if msg.text:
                    blocks.append({"type": "text", "text": msg.text})
            else:
                if msg.text:
                    blocks.append({"type": "text", "text": msg.text})
                for tc in msg.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )

            if not blocks:
                blocks.append({"type": "text", "text": ""})
            wire_messages.append({"role": msg.role, "content": blocks})

        kwargs: dict[str, Any] = {
            "model": self.spec.model_id,
            "max_tokens": max_tokens,
            "messages": wire_messages,
        }
        if system_parts:
            system_text = "\n\n".join(system_parts)
            if self.spec.prompt_cache:
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                kwargs["system"] = system_text

        # SPEC §6.1: Use output_config.format and output_config.effort;
        # NEVER send thinking={"type": "disabled"}, temperature, top_p, or forced tool_choice.
        output_config: dict[str, Any] = {}
        resolved_effort = effort or self.spec.default_effort
        if resolved_effort is not None:
            output_config["effort"] = resolved_effort
        if response_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": strict_json_schema(response_schema),
            }
        if output_config:
            kwargs["output_config"] = output_config

        if tools:
            wire_tools: list[dict[str, Any]] = []
            for tool in sorted(tools, key=lambda t: t.name):
                entry: dict[str, Any] = {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                if tool.strict:
                    entry["strict"] = True
                wire_tools.append(entry)
            if self.spec.prompt_cache and wire_tools:
                wire_tools[-1]["cache_control"] = {"type": "ephemeral"}
            kwargs["tools"] = wire_tools
            kwargs["tool_choice"] = {"type": "auto"}

        started = time.monotonic()
        try:
            if self._http_client is not None:
                client = anthropic.AsyncAnthropic(
                    api_key=self._api_key,
                    base_url=self.spec.base_url,
                    max_retries=self._max_retries,
                    http_client=self._http_client,
                )
                response = await client.messages.create(**cast(Any, kwargs))
            else:
                async with anthropic.AsyncAnthropic(
                    api_key=self._api_key,
                    base_url=self.spec.base_url,
                    max_retries=self._max_retries,
                ) as client:
                    response = await client.messages.create(**cast(Any, kwargs))
        except anthropic.AnthropicError as exc:
            status_code, provider_code = extract_sdk_error_meta(exc)
            raise ModelError(
                f"Anthropic request failed ({self.id}): {exc}",
                status_code=status_code,
                provider_code=provider_code,
            ) from exc

        latency_ms = max(0, int((time.monotonic() - started) * 1000))
        stop_reason = getattr(response, "stop_reason", None)
        refused = stop_reason == "refusal"

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in getattr(response, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(str(getattr(block, "text", "")))
            elif btype == "tool_use":
                raw_input = getattr(block, "input", {}) or {}
                tool_calls.append(
                    ToolCall(
                        id=str(getattr(block, "id", "")),
                        name=str(getattr(block, "name", "")),
                        arguments=dict(raw_input) if isinstance(raw_input, dict) else {},
                    )
                )

        text = "\n".join(part for part in text_parts if part)
        usage_obj = getattr(response, "usage", None)
        raw_in = int(getattr(usage_obj, "input_tokens", 0) or 0)
        cached_in = int(getattr(usage_obj, "cache_read_input_tokens", 0) or 0)
        out_tokens = int(getattr(usage_obj, "output_tokens", 0) or 0)
        usage = Usage(
            in_tokens=raw_in + cached_in,
            out_tokens=out_tokens,
            cached_tokens=cached_in,
        )
        cost_usd = compute_cost_usd(self.spec, usage)

        parsed: Any = None
        if response_schema is not None and not refused:
            if not text.strip():
                raise ModelError(f"Anthropic model {self.id} returned empty structured JSON output.")
            try:
                parsed = response_schema.model_validate_json(strip_json_markdown_fences(text))
            except Exception as exc:
                raise ModelError(
                    f"Failed to validate structured output from {self.id}: {exc}"
                ) from exc

        return GenerationResult(
            text=text,
            tool_calls=tuple(tool_calls),
            parsed=parsed,
            usage=usage,
            cost_usd=cost_usd,
            stop_reason=str(stop_reason) if stop_reason is not None else None,
            refused=refused,
            provider=self.provider,
            model=self.id,
            latency_ms=latency_ms,
        )


__all__ = ["AnthropicChatModel"]
