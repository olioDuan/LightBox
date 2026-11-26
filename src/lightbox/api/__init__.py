"""Lightbox local FastAPI backend, security middleware, SSE hub, and server lock (SPEC §8.1, §9.2)."""

from __future__ import annotations

from lightbox.api.app import WEB_DIST_DIR, create_app
from lightbox.api.events import EventBus, ServerEvent
from lightbox.api.security import (
    CSRF_HEADER_NAME,
    LOCAL_CONFIRM_HEADER,
    SESSION_COOKIE_NAME,
    AuthSession,
    LocalSecurityMiddleware,
    SessionAuthState,
)
from lightbox.api.server_lock import (
    HEALTHZ_PATH,
    LOCAL_HOST,
    ServerLockInfo,
    clear_server_lock,
    find_free_local_port,
    is_pid_alive,
    is_server_responsive,
    read_server_lock,
    write_server_lock,
)

__all__ = [
    "AuthSession",
    "CSRF_HEADER_NAME",
    "EventBus",
    "HEALTHZ_PATH",
    "LOCAL_CONFIRM_HEADER",
    "LOCAL_HOST",
    "LocalSecurityMiddleware",
    "SESSION_COOKIE_NAME",
    "ServerEvent",
    "ServerLockInfo",
    "SessionAuthState",
    "WEB_DIST_DIR",
    "clear_server_lock",
    "create_app",
    "find_free_local_port",
    "is_pid_alive",
    "is_server_responsive",
    "read_server_lock",
    "write_server_lock",
]
