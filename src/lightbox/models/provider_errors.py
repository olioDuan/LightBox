"""Per-provider error classification distinguishing persistent vs transient failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lightbox.models.types import (
    BudgetExceededError,
    CallLimitExceededError,
    EndpointBindingError,
    MissingAPIKeyError,
    ModelError,
)

# Only these persistent error classes downgrade an active `connected` state to `error`.
PERSISTENT_ERROR_CLASSES: frozenset[str] = frozenset(
    {
        "auth_failed",
        "insufficient_balance",
        "model_or_endpoint_unavailable",
        "endpoint_binding_mismatch",
    }
)

# These transient error classes keep `connected` intact and surface a temporary notice instead.
TRANSIENT_ERROR_CLASSES: frozenset[str] = frozenset(
    {
        "rate_limited",
        "network_timeout",
        "local_budget_exceeded",
        "session_call_limit_exceeded",
    }
)

# Per-provider code tables: provider_code (lowercase) -> normalized error_class
_PROVIDER_CODE_MAP: dict[str, dict[str, str]] = {
    "openai": {
        "invalid_api_key": "auth_failed",
        "authentication_error": "auth_failed",
        "permission_error": "auth_failed",
        "insufficient_quota": "insufficient_balance",
        "billing_hard_limit_reached": "insufficient_balance",
        "account_deactivated": "auth_failed",
        "model_not_found": "model_or_endpoint_unavailable",
        "not_found_error": "model_or_endpoint_unavailable",
        "rate_limit_exceeded": "rate_limited",
        "rate_limit_error": "rate_limited",
        "server_error": "network_timeout",
        "timeout": "network_timeout",
    },
    "anthropic": {
        "authentication_error": "auth_failed",
        "permission_error": "auth_failed",
        "billing_error": "insufficient_balance",
        "not_found_error": "model_or_endpoint_unavailable",
        "rate_limit_error": "rate_limited",
        "overloaded_error": "network_timeout",
        "api_error": "network_timeout",
    },
    "gemini": {
        "api_key_invalid": "auth_failed",
        "unauthenticated": "auth_failed",
        "permission_denied": "auth_failed",
        "billing_disabled": "insufficient_balance",
        "not_found": "model_or_endpoint_unavailable",
        "model_not_found": "model_or_endpoint_unavailable",
        "resource_exhausted": "rate_limited",
        "rate_limit_exceeded": "rate_limited",
        "unavailable": "network_timeout",
        "deadline_exceeded": "network_timeout",
    },
    "custom": {
        "invalid_api_key": "auth_failed",
        "authentication_error": "auth_failed",
        "unauthorized": "auth_failed",
        "insufficient_quota": "insufficient_balance",
        "insufficient_balance": "insufficient_balance",
        "billing_error": "insufficient_balance",
        "model_not_found": "model_or_endpoint_unavailable",
        "not_found_error": "model_or_endpoint_unavailable",
        "rate_limit_exceeded": "rate_limited",
        "rate_limit_error": "rate_limited",
        "too_many_requests": "rate_limited",
        "overloaded_error": "network_timeout",
    },
}


@dataclass(frozen=True, slots=True)
class MappedProviderError:
    """Normalized error metadata derived from an adapter/provider exception."""

    error_class: str
    is_persistent: bool
    raw_code: str | None = None
    status_code: int | None = None
    message: str = ""


def _extract_provider_code_from_body(body: Any) -> str | None:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            for key in ("code", "type", "status"):
                val = err.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        for key in ("code", "type", "status"):
            val = body.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def extract_sdk_error_meta(exc: BaseException) -> tuple[int | None, str | None]:
    """Extract `(status_code, provider_code)` safely from Anthropic/OpenAI SDK exceptions."""
    status_raw = getattr(exc, "status_code", None)
    status_code = int(status_raw) if isinstance(status_raw, int) else None
    code_raw = getattr(exc, "code", None)
    provider_code = str(code_raw).strip() if isinstance(code_raw, str) and code_raw.strip() else None
    if provider_code is None:
        provider_code = _extract_provider_code_from_body(getattr(exc, "body", None))
    if provider_code is None:
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                body = resp.json()
                provider_code = _extract_provider_code_from_body(body)
            except Exception:
                pass
    return status_code, provider_code


def classify_provider_error(provider: str, exc: BaseException) -> MappedProviderError:
    """Map an adapter/routing exception into a persistent or transient normalized error class."""
    if isinstance(exc, BudgetExceededError):
        return MappedProviderError(
            error_class="local_budget_exceeded",
            is_persistent=False,
            raw_code=f"local_budget_{exc.scope}",
            message=str(exc),
        )
    if isinstance(exc, CallLimitExceededError):
        return MappedProviderError(
            error_class="session_call_limit_exceeded",
            is_persistent=False,
            raw_code="session_call_limit_exceeded",
            message=str(exc),
        )
    if isinstance(exc, EndpointBindingError):
        return MappedProviderError(
            error_class="endpoint_binding_mismatch",
            is_persistent=True,
            raw_code="endpoint_binding_mismatch",
            message=str(exc),
        )
    if isinstance(exc, MissingAPIKeyError):
        return MappedProviderError(
            error_class="auth_failed",
            is_persistent=True,
            raw_code="missing_api_key",
            message=str(exc),
        )

    status_code: int | None = None
    provider_code: str | None = None
    preset_class: str | None = None
    if isinstance(exc, ModelError):
        status_code = exc.status_code
        provider_code = exc.provider_code
        preset_class = exc.error_class

    if status_code is None or provider_code is None:
        cause = exc.__cause__ if exc.__cause__ is not None else exc
        c_status, c_code = extract_sdk_error_meta(cause)
        status_code = status_code if status_code is not None else c_status
        provider_code = provider_code or c_code

    raw_code = provider_code or (str(status_code) if status_code is not None else type(exc).__name__)

    if preset_class:
        return MappedProviderError(
            error_class=preset_class,
            is_persistent=preset_class in PERSISTENT_ERROR_CLASSES,
            raw_code=raw_code,
            status_code=status_code,
            message=str(exc),
        )

    table_key = provider if provider in _PROVIDER_CODE_MAP else "custom"
    table = _PROVIDER_CODE_MAP[table_key]
    msg_lower = str(exc).lower()

    # 1. Explicit provider error code lookup
    if provider_code:
        mapped = table.get(provider_code.strip().lower())
        if mapped is not None:
            return MappedProviderError(
                error_class=mapped,
                is_persistent=mapped in PERSISTENT_ERROR_CLASSES,
                raw_code=raw_code,
                status_code=status_code,
                message=str(exc),
            )

    # 2. Message heuristics for provider-specific billing/balance errors that share 400/429
    if any(
        token in msg_lower
        for token in (
            "credit balance is too low",
            "insufficient_quota",
            "insufficient balance",
            "exceeded your current quota",
            "billing_disabled",
            "billing hard limit",
        )
    ):
        return MappedProviderError(
            error_class="insufficient_balance",
            is_persistent=True,
            raw_code=raw_code,
            status_code=status_code,
            message=str(exc),
        )

    # 3. HTTP status code mapping
    if status_code in {401, 403}:
        return MappedProviderError(
            error_class="auth_failed",
            is_persistent=True,
            raw_code=raw_code,
            status_code=status_code,
            message=str(exc),
        )
    if status_code == 402:
        return MappedProviderError(
            error_class="insufficient_balance",
            is_persistent=True,
            raw_code=raw_code,
            status_code=status_code,
            message=str(exc),
        )
    if status_code == 404:
        return MappedProviderError(
            error_class="model_or_endpoint_unavailable",
            is_persistent=True,
            raw_code=raw_code,
            status_code=status_code,
            message=str(exc),
        )
    if status_code == 429:
        return MappedProviderError(
            error_class="rate_limited",
            is_persistent=False,
            raw_code=raw_code,
            status_code=status_code,
            message=str(exc),
        )
    if status_code in {408, 502, 503, 504, 529}:
        return MappedProviderError(
            error_class="network_timeout",
            is_persistent=False,
            raw_code=raw_code,
            status_code=status_code,
            message=str(exc),
        )

    # 4. Connection / timeout exception type or text heuristics
    exc_chain_names = {type(exc).__name__.lower()}
    if exc.__cause__ is not None:
        exc_chain_names.add(type(exc.__cause__).__name__.lower())
    if any(
        any(tok in name for tok in ("timeout", "connection", "connecterror", "network"))
        for name in exc_chain_names
    ) or any(tok in msg_lower for tok in ("timed out", "connection error", "connection refused", "connect error")):
        return MappedProviderError(
            error_class="network_timeout",
            is_persistent=False,
            raw_code=raw_code,
            status_code=status_code,
            message=str(exc),
        )

    if any(tok in msg_lower for tok in ("model_not_found", "model not found", "does not exist")):
        return MappedProviderError(
            error_class="model_or_endpoint_unavailable",
            is_persistent=True,
            raw_code=raw_code,
            status_code=status_code,
            message=str(exc),
        )
    if any(tok in msg_lower for tok in ("invalid api key", "incorrect api key", "unauthorized", "authentication")):
        return MappedProviderError(
            error_class="auth_failed",
            is_persistent=True,
            raw_code=raw_code,
            status_code=status_code,
            message=str(exc),
        )

    # 5. Unmapped fallback: `request_failed` with original error code
    return MappedProviderError(
        error_class="request_failed",
        is_persistent=False,
        raw_code=raw_code,
        status_code=status_code,
        message=str(exc),
    )


__all__ = [
    "MappedProviderError",
    "PERSISTENT_ERROR_CLASSES",
    "TRANSIENT_ERROR_CLASSES",
    "classify_provider_error",
    "extract_sdk_error_meta",
]
