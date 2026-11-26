"""Structured JSON logging initialization with secret redaction (SPEC §3.1, §10.2(6))."""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

from lightbox.paths import AppPaths, get_app_paths

REDACTED = "[REDACTED]"
SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "authorization",
    "cookie",
    "csrf",
)


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in SENSITIVE_KEY_FRAGMENTS)


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: (REDACTED if _is_sensitive_key(str(k)) else _redact_value(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def redact_secrets(
    _logger: Any,
    _method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Structlog processor that redacts keys likely to hold credentials or tokens."""
    for key in list(event_dict.keys()):
        if _is_sensitive_key(str(key)):
            event_dict[key] = REDACTED
        else:
            event_dict[key] = _redact_value(event_dict[key])
    return event_dict


def configure_logging(
    paths: AppPaths | None = None,
    *,
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_stderr: bool = False,
) -> None:
    """Configure stdlib + structlog JSON logging to `<APP_DATA_DIR>/logs/lightbox.log`."""
    resolved = paths or get_app_paths()
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = []
    if log_to_file:
        resolved.ensure_dirs()
        file_handler = logging.FileHandler(resolved.log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)

    if log_to_stderr or not handlers:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(numeric_level)
        handlers.append(stream_handler)

    root_logger = logging.getLogger("lightbox")
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)
    root_logger.propagate = False
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_secrets,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str = "lightbox") -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger under the `lightbox` namespace."""
    logger_name = name if name.startswith("lightbox") else f"lightbox.{name}"
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(logger_name)
    return logger


__all__ = ["REDACTED", "configure_logging", "get_logger", "redact_secrets"]
