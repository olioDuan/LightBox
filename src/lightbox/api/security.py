"""Local-only authentication, DNS rebinding defense, Cookie & CSRF middleware (SPEC §10.2(3))."""

from __future__ import annotations

import hmac
import secrets
import threading
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

SESSION_COOKIE_NAME = "lightbox_session"
CSRF_HEADER_NAME = "x-lightbox-csrf"
LOCAL_CONFIRM_HEADER = "x-lightbox-local-confirm"

_MUTATING_METHODS: frozenset[str] = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_PUBLIC_API_PATHS: frozenset[str] = frozenset({"/api/auth/session"})


@dataclass(frozen=True, slots=True)
class AuthSession:
    session_id: str
    csrf_token: str


class SessionAuthState:
    """In-memory session store with one-time launch token and per-session CSRF token."""

    def __init__(self, *, bootstrap_token: str | None = None) -> None:
        self._lock = threading.Lock()
        self._bootstrap_token: str | None = (
            bootstrap_token if bootstrap_token is not None else secrets.token_urlsafe(32)
        )
        self._sessions: dict[str, str] = {}  # session_id -> csrf_token

    @property
    def bootstrap_token(self) -> str | None:
        with self._lock:
            return self._bootstrap_token

    def issue_session(self) -> AuthSession:
        sid = secrets.token_urlsafe(32)
        csrf = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[sid] = csrf
        return AuthSession(session_id=sid, csrf_token=csrf)

    def consume_bootstrap_token(self, token: str) -> AuthSession | None:
        """Consume the one-time launch token and mint a fresh session; replay returns None."""
        candidate = token.strip()
        if not candidate:
            return None
        with self._lock:
            expected = self._bootstrap_token
            if expected is None or not hmac.compare_digest(candidate, expected):
                return None
            self._bootstrap_token = None
            sid = secrets.token_urlsafe(32)
            csrf = secrets.token_urlsafe(32)
            self._sessions[sid] = csrf
            return AuthSession(session_id=sid, csrf_token=csrf)

    def get_csrf_for_session(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        with self._lock:
            return self._sessions.get(session_id)

    def verify_csrf(self, session_id: str | None, csrf_header: str | None) -> bool:
        expected = self.get_csrf_for_session(session_id)
        if not expected or not csrf_header:
            return False
        return hmac.compare_digest(csrf_header.strip(), expected)


def set_session_cookie(response: Response, session: AuthSession) -> None:
    """Attach an `HttpOnly + SameSite=Strict` session cookie."""
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session.session_id,
        httponly=True,
        samesite="strict",
        secure=False,  # Localhost 127.0.0.1 HTTP only
        path="/",
    )


class LocalSecurityMiddleware(BaseHTTPMiddleware):
    """Enforces 127.0.0.1 Host/Origin DNS-rebinding defense, Cookie auth, and CSRF checks.

    Intentionally emits NO CORS headers (`Access-Control-Allow-Origin` is never set).
    """

    def __init__(self, app: object, *, allowed_port: int, auth: SessionAuthState) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.allowed_port = allowed_port
        self.allowed_hosts = {f"127.0.0.1:{allowed_port}"}
        if allowed_port == 80:
            self.allowed_hosts.add("127.0.0.1")
        self.allowed_origin = f"http://127.0.0.1:{allowed_port}"
        self.auth = auth

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        # 1. Strict Host header check (DNS rebinding defense)
        host = (request.headers.get("host") or "").strip().lower()
        if host not in self.allowed_hosts:
            return JSONResponse(
                status_code=403,
                content={"detail": f"Forbidden Host header: {host!r}"},
            )

        # 2. Strict Origin header check whenever present
        origin = request.headers.get("origin")
        if origin is not None and origin.strip().lower() != self.allowed_origin:
            return JSONResponse(
                status_code=403,
                content={"detail": f"Forbidden Origin header: {origin!r}"},
            )

        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)

        # 3. Local SameSite=Strict fallback landing session endpoint (SPEC §12 #8)
        if path in _PUBLIC_API_PATHS:
            if request.method in _MUTATING_METHODS:
                # Require custom header + matching Origin to defeat cross-site simple POSTs
                if (
                    origin is None
                    or request.headers.get(LOCAL_CONFIRM_HEADER, "").strip() != "1"
                ):
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "Local session confirmation header/Origin required."},
                    )
            return await call_next(request)

        # 4. ALL /api/* read & write endpoints require a valid HttpOnly session Cookie
        sid = request.cookies.get(SESSION_COOKIE_NAME)
        csrf_expected = self.auth.get_csrf_for_session(sid)
        if csrf_expected is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required (missing or invalid session cookie)."},
            )

        # 5. ALL write endpoints additionally require matching X-Lightbox-CSRF header
        if request.method in _MUTATING_METHODS:
            csrf_provided = request.headers.get(CSRF_HEADER_NAME)
            if not self.auth.verify_csrf(sid, csrf_provided):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Missing or invalid X-Lightbox-CSRF header."},
                )

        request.state.session_id = sid
        request.state.csrf_token = csrf_expected
        return await call_next(request)


__all__ = [
    "AuthSession",
    "CSRF_HEADER_NAME",
    "LOCAL_CONFIRM_HEADER",
    "LocalSecurityMiddleware",
    "SESSION_COOKIE_NAME",
    "SessionAuthState",
    "set_session_cookie",
]
