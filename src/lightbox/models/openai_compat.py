"""OpenAI-compatible adapter for OpenAI, Gemini OpenAI-compat, Ollama, and mlx-vlm (SPEC §6.1)."""

from __future__ import annotations

import base64
import json
import time
from typing import Any, cast

import openai
from pydantic import BaseModel

from lightbox.config import ModelSpec
from lightbox.models.images import strip_metadata_and_encode
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


class OpenAICompatChatModel:
    """`openai.AsyncOpenAI` adapter supporting OpenAI, Gemini, Ollama, and local `mlx-vlm`."""

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
        del effort  # OpenAI-compat endpoints do not share Anthropic's output_config.effort contract
        wire_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                wire_messages.append({"role": "system", "content": msg.text})
                continue

            if msg.role == "user" and msg.tool_results:
                for tr in msg.tool_results:
                    wire_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.tool_call_id,
                            "content": tr.content,
                        }
                    )
                if not msg.text and not msg.images:
                    continue

            if msg.role == "assistant" and msg.tool_calls:
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.text or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
                wire_messages.append(assistant_msg)
                continue

            if msg.images:
                parts: list[dict[str, Any]] = []
                if msg.text:
                    parts.append({"type": "text", "text": msg.text})
                for img in msg.images:
                    if img.sanitized:
                        clean_bytes, media_type = img.data, img.mime_type
                    else:
                        clean_bytes, media_type = strip_metadata_and_encode(
                            img.data,
                            max_pixels=self.spec.max_pixels,
                            fmt="JPEG",
                        )
                    b64 = base64.b64encode(clean_bytes).decode("ascii")
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{b64}"},
                        }
                    )
                wire_messages.append({"role": msg.role, "content": parts})
            else:
                wire_messages.append({"role": msg.role, "content": msg.text})

        kwargs: dict[str, Any] = {
            "model": self.spec.model_id,
            "messages": wire_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                        **({"strict": True} if t.strict else {}),
                    },
                }
                for t in tools
            ]
            kwargs["tool_choice"] = "auto"

        if response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "strict": True,
                    "schema": strict_json_schema(response_schema),
                },
            }

        started = time.monotonic()
        try:
            if self._http_client is not None:
                client = openai.AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=self.spec.base_url,
                    max_retries=self._max_retries,
                    http_client=self._http_client,
                )
                response = await client.chat.completions.create(**cast(Any, kwargs))
            else:
                async with openai.AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=self.spec.base_url,
                    max_retries=self._max_retries,
                ) as client:
                    response = await client.chat.completions.create(**cast(Any, kwargs))
        except openai.OpenAIError as exc:
            raise ModelError(f"OpenAI-compatible request failed ({self.id}): {exc}") from exc

        latency_ms = max(0, int((time.monotonic() - started) * 1000))
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ModelError(f"Model {self.id} returned no completion choices.")

        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        message_obj = getattr(choice, "message", None)
        refusal_text = getattr(message_obj, "refusal", None) if message_obj is not None else None
        refused = bool(refusal_text) or finish_reason == "content_filter"

        raw_content = getattr(message_obj, "content", None) if message_obj is not None else None
        text = str(raw_content or "")
        if not text and refusal_text:
            text = str(refusal_text)

        tool_calls: list[ToolCall] = []
        raw_tcs = getattr(message_obj, "tool_calls", None) if message_obj is not None else None
        for tc in raw_tcs or []:
            fn = getattr(tc, "function", None)
            fn_name = str(getattr(fn, "name", "") or "")
            raw_args = str(getattr(fn, "arguments", "") or "{}")
            try:
                parsed_args = json.loads(raw_args)
                args_dict = dict(parsed_args) if isinstance(parsed_args, dict) else {}
            except json.JSONDecodeError:
                args_dict = {}
            tool_calls.append(
                ToolCall(
                    id=str(getattr(tc, "id", "")),
                    name=fn_name,
                    arguments=args_dict,
                )
            )

        usage_obj = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0)
        details = getattr(usage_obj, "prompt_tokens_details", None)
        cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
        usage = Usage(
            in_tokens=prompt_tokens,
            out_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )
        cost_usd = compute_cost_usd(self.spec, usage)

        parsed: Any = None
        if response_schema is not None and not refused:
            if not text.strip():
                raise ModelError(f"Model {self.id} returned empty structured JSON output.")
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
            stop_reason=str(finish_reason) if finish_reason is not None else None,
            refused=refused,
            provider=self.provider,
            model=self.id,
            latency_ms=latency_ms,
        )


__all__ = ["OpenAICompatChatModel"]
