"""Local Web UI server coordination for MCP Server via `<APP_DATA_DIR>/server.lock` (SPEC §8.1)."""

from __future__ import annotations

import os
import secrets
import threading
import time
from typing import Any

import uvicorn

from lightbox.api import (
    LOCAL_HOST,
    ServerLockInfo,
    clear_server_lock,
    create_app,
    find_free_local_port,
    is_server_responsive,
    read_server_lock,
    write_server_lock,
)
from lightbox.paths import AppPaths, get_app_paths
from lightbox.tools import assert_tokenless_url


class LocalUIBridge:
    """Detects a running local Lightbox Web UI via `server.lock` or auto-starts one on 127.0.0.1."""

    def __init__(
        self,
        *,
        app_paths: AppPaths | None = None,
        auto_start: bool = True,
        fallback_base_url: str = "http://127.0.0.1:8765",
    ) -> None:
        self.app_paths = app_paths or get_app_paths()
        self.auto_start = auto_start
        self.fallback_base_url = fallback_base_url.rstrip("/")
        self._lock = threading.Lock()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._owned_info: ServerLockInfo | None = None

    def ensure_base_url(self) -> str:
        """Return a tokenless `http://127.0.0.1:<port>` base URL, starting the local UI if needed."""
        with self._lock:
            existing = read_server_lock(
                app_paths=self.app_paths,
                check_alive=True,
                verify_http=True,
                timeout_s=0.4,
            )
            if existing is not None:
                return assert_tokenless_url(existing.base_url)

            if not self.auto_start:
                return assert_tokenless_url(self.fallback_base_url)

            info = self._start_embedded_server()
            return assert_tokenless_url(info.base_url)

    def _start_embedded_server(self) -> ServerLockInfo:
        self.app_paths.ensure_dirs()
        port = find_free_local_port(LOCAL_HOST)
        instance_id = secrets.token_hex(8)
        fastapi_app = create_app(
            app_paths=self.app_paths,
            port=port,
            instance_id=instance_id,
        )
        config = uvicorn.Config(
            fastapi_app,
            host=LOCAL_HOST,
            port=port,
            log_level="error",
            timeout_graceful_shutdown=1,
        )
        server = uvicorn.Server(config)

        def _serve() -> None:
            try:
                server.run()
            finally:
                fastapi_app.state.event_bus.close()

        thread = threading.Thread(target=_serve, name="lightbox-mcp-ui", daemon=True)
        thread.start()

        info = write_server_lock(
            port,
            app_paths=self.app_paths,
            host=LOCAL_HOST,
            pid=os.getpid(),
            instance_id=instance_id,
        )

        # Wait briefly for embedded uvicorn socket to accept /healthz
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if is_server_responsive(info, timeout_s=0.25):
                break
            time.sleep(0.05)

        self._server = server
        self._thread = thread
        self._owned_info = info
        return info

    def stop(self) -> None:
        """Shut down the embedded local UI server if this bridge started it."""
        with self._lock:
            server = self._server
            thread = self._thread
            owned = self._owned_info
            self._server = None
            self._thread = None
            self._owned_info = None

        if server is not None:
            server.should_exit = True
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        if owned is not None:
            clear_server_lock(app_paths=self.app_paths, only_if_pid=owned.pid)

    @property
    def owned_lock_info(self) -> ServerLockInfo | None:
        return self._owned_info

    def __enter__(self) -> LocalUIBridge:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()


__all__ = ["LocalUIBridge"]
