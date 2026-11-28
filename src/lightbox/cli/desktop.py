"""Native desktop window launcher (`pywebview`) with service ownership, log redirection, and safe exit."""

from __future__ import annotations

import html
import os
import secrets
import subprocess
import sys
import threading
import time
import traceback
from io import TextIOBase
from pathlib import Path
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

DEFAULT_SHUTDOWN_TIMEOUT_S = 8.0
DEFAULT_SERVER_READY_TIMEOUT_S = 5.0
DEFAULT_WINDOW_TITLE = "Lightbox"
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 840


class DesktopServerStartError(RuntimeError):
    """Raised when the owned embedded local API server cannot start or become responsive."""

    def __init__(self, message: str, *, log_path: Path | None = None) -> None:
        super().__init__(message)
        self.log_path = log_path


def _import_webview() -> Any:
    """Lazy-import `webview` (pywebview) so headless CI without pywebview imports this module cleanly."""
    import webview

    return webview


def _resolve_folder_dialog_const(webview_mod: Any) -> Any:
    """Prefer `webview.FileDialog.FOLDER` (pywebview 6.x) before legacy `webview.FOLDER_DIALOG` (5.x)."""
    file_dialog = getattr(webview_mod, "FileDialog", None)
    if file_dialog is not None and hasattr(file_dialog, "FOLDER"):
        return file_dialog.FOLDER
    return getattr(webview_mod, "FOLDER_DIALOG", 20)


def _try_acquire_advisory_lock(lock_path: Path) -> Any | None:
    """Try acquiring a non-blocking exclusive advisory lock; returns open handle on success, else None."""
    lock_file = lock_path.open("a+b")
    try:
        if sys.platform == "win32":
            import msvcrt

            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_file
    except OSError:
        try:
            lock_file.close()
        except OSError:
            pass
        return None


def _release_advisory_lock(lock_file: Any | None) -> None:
    if lock_file is None:
        return
    try:
        if sys.platform == "win32":
            import msvcrt

            try:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        try:
            lock_file.close()
        except OSError:
            pass


class DesktopLogHandle:
    """Line-buffered log stream wrapper that holds the primary-instance advisory lock until closed."""

    def __init__(
        self,
        stream: TextIOBase,
        *,
        path: Path,
        is_primary: bool,
        lock_file: Any | None = None,
    ) -> None:
        self._stream = stream
        self.path = path
        self.is_primary = is_primary
        self.encoding: str = getattr(stream, "encoding", "utf-8") or "utf-8"
        self.errors: str | None = getattr(stream, "errors", None)
        self._lock_file = lock_file

    @property
    def closed(self) -> bool:
        return bool(self._stream.closed)

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self._stream.fileno()

    def write(self, s: str) -> int:
        return self._stream.write(s)

    def writelines(self, lines: Any) -> None:
        self._stream.writelines(lines)

    def flush(self) -> None:
        if not self._stream.closed:
            self._stream.flush()

    def close(self) -> None:
        try:
            if not self._stream.closed:
                self._stream.flush()
                self._stream.close()
        finally:
            lock_file = self._lock_file
            self._lock_file = None
            _release_advisory_lock(lock_file)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def redirect_desktop_logs(app_paths: AppPaths) -> DesktopLogHandle:
    """Redirect stdout/stderr to `<APP_DATA_DIR>/logs/desktop.log` safely across multiple instances.

    - Primary instance (acquires `.desktop_log.lock`): copies existing `desktop.log` into `desktop.prev.log`
      without any file deletion and opens `desktop.log`.
    - Secondary concurrent instance (lock already held): leaves `desktop.log` and `desktop.prev.log`
      untouched and writes to `desktop-<pid>.log` instead.
    """
    app_paths.ensure_dirs()
    lock_path = app_paths.logs_dir / ".desktop_log.lock"
    lock_file = _try_acquire_advisory_lock(lock_path)

    if lock_file is not None:
        log_path = app_paths.desktop_log_file
        prev_path = app_paths.desktop_prev_log_file
        # Zero-delete single-backup rotation: overwrite `desktop.prev.log` with previous `desktop.log` content
        if log_path.is_file():
            try:
                prev_path.write_bytes(log_path.read_bytes())
            except OSError:
                pass
        is_primary = True
    else:
        log_path = app_paths.logs_dir / f"desktop-{os.getpid()}.log"
        is_primary = False

    raw_handle = log_path.open("w", encoding="utf-8", buffering=1)
    handle = DesktopLogHandle(raw_handle, path=log_path, is_primary=is_primary, lock_file=lock_file)
    sys.stdout = handle
    sys.stderr = handle
    return handle


def _is_interactive_cli(orig_stdout: Any, orig_stderr: Any) -> bool:
    """Return True only when launched from an interactive terminal/CLI rather than a desktop icon/shortcut."""
    for stream in (orig_stdout, orig_stderr):
        if stream is None:
            continue
        isatty_fn = getattr(stream, "isatty", None)
        if callable(isatty_fn):
            try:
                if bool(isatty_fn()):
                    return True
            except Exception:
                pass
    return False


def _show_native_error_dialog(title: str, message: str) -> None:
    """Best-effort OS-native alert dialog for console-less launches when no webview window can be shown."""
    try:
        if sys.platform == "darwin":
            escaped_title = title.replace("\\", "\\\\").replace('"', '\\"')
            escaped_msg = message.replace("\\", "\\\\").replace('"', '\\"')
            script = (
                f'display alert "{escaped_title}" message "{escaped_msg}" '
                'as critical buttons {"OK"} default button "OK"'
            )
            subprocess.run(["osascript", "-e", script], check=False, timeout=15)
            return
        if sys.platform == "win32":
            import ctypes

            # MB_OK (0x0) | MB_ICONERROR (0x10)
            ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)
            return
        # Linux desktop fallback: zenity / kdialog if present
        for cmd in (
            ["zenity", "--error", f"--title={title}", f"--text={message}"],
            ["kdialog", "--error", message, "--title", title],
        ):
            try:
                proc = subprocess.run(cmd, check=False, timeout=15)
                if proc.returncode == 0:
                    return
            except FileNotFoundError:
                continue
    except Exception:
        pass


def _build_startup_error_html(error_detail: str, log_path: Path) -> str:
    """Render an inline HTML error page when the embedded local server fails to start."""
    safe_err = html.escape(error_detail or "Unknown startup error")
    safe_log = html.escape(str(log_path))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Lightbox 启动失败</title>
  <style>
    body {{
      margin: 0;
      padding: 40px 24px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #fafafa;
      color: #18181b;
      line-height: 1.6;
    }}
    .card {{
      max-width: 560px;
      margin: 0 auto;
      background: #ffffff;
      border: 1px solid #e4e4e7;
      border-radius: 12px;
      padding: 24px 28px;
    }}
    h1 {{ font-size: 18px; margin: 0 0 10px 0; }}
    p {{ font-size: 14px; color: #52525b; margin: 0 0 12px 0; }}
    code, pre {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      background: #f4f4f5;
      color: #18181b;
      border-radius: 6px;
    }}
    code {{ padding: 2px 6px; word-break: break-all; }}
    pre {{ padding: 10px 12px; overflow-x: auto; white-space: pre-wrap; margin: 12px 0 0 0; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Lightbox 本地服务启动失败</h1>
    <p>桌面窗口已尝试切换本地端口重试，但仍未能连上本地服务。请关闭窗口后重试，或检查日志文件以排查原因：</p>
    <p>日志位置：<code>{safe_log}</code></p>
    <pre>{safe_err}</pre>
  </div>
</body>
</html>"""


class DesktopJsApi:
    """Minimal JS bridge exposed to `pywebview` (`window.pywebview.api`)."""

    def __init__(self, controller: DesktopAppController) -> None:
        self._controller = controller

    def pick_folder(self) -> dict[str, Any]:
        """Open native OS folder picker attached to the desktop window."""
        return self._controller.pick_folder()

    def reconnect(self) -> dict[str, Any]:
        """Start a fresh local server and reload the desktop window after sustained disconnect."""
        return self._controller.reconnect()

    def get_lifecycle_status(self) -> dict[str, Any]:
        """Report current server ownership and in-flight file operation count."""
        return {
            "owns_server": self._controller.owns_server,
            "in_flight_file_ops": self._controller.in_flight_file_ops,
        }


class DesktopAppController:
    """Manages local FastAPI/Uvicorn ownership, native dialog bridge, reconnect, and safe exit."""

    def __init__(
        self,
        *,
        app_paths: AppPaths | None = None,
        port: int = 0,
        shutdown_timeout_s: float = DEFAULT_SHUTDOWN_TIMEOUT_S,
        ready_timeout_s: float = DEFAULT_SERVER_READY_TIMEOUT_S,
        log_path: Path | None = None,
    ) -> None:
        self.app_paths = app_paths or get_app_paths()
        self.requested_port = port
        self.shutdown_timeout_s = max(0.05, shutdown_timeout_s)
        self.ready_timeout_s = max(0.1, ready_timeout_s)
        self.log_path = log_path or self.app_paths.desktop_log_file

        self._lock = threading.Lock()
        self._owns_server = False
        self._owned_lock: ServerLockInfo | None = None
        self._server: uvicorn.Server | None = None
        self._server_thread: threading.Thread | None = None
        self._fastapi_app: Any | None = None
        self._window: Any | None = None
        self._folder_dialog_const: Any = 20
        self._base_url: str | None = None
        self._pending_close = False

    @property
    def owns_server(self) -> bool:
        return self._owns_server

    @property
    def owned_lock_info(self) -> ServerLockInfo | None:
        return self._owned_lock

    @property
    def base_url(self) -> str | None:
        return self._base_url

    @property
    def in_flight_file_ops(self) -> int:
        app = self._fastapi_app
        if not self._owns_server or app is None:
            return 0
        lifecycle = getattr(app.state, "lifecycle", None)
        if lifecycle is None:
            return 0
        return int(lifecycle.in_flight_file_ops)

    def attach_window(self, window: Any, *, folder_dialog: Any = 20) -> None:
        """Bind the active `pywebview` window handle (or test double)."""
        self._window = window
        self._folder_dialog_const = folder_dialog

    def _start_single_attempt(
        self,
        *,
        port: int,
        ready_timeout_s: float,
    ) -> tuple[str, ServerLockInfo, uvicorn.Server, threading.Thread, Any]:
        self.app_paths.ensure_dirs()
        chosen_port = port if port > 0 else find_free_local_port(LOCAL_HOST)
        launch_token = secrets.token_urlsafe(32)
        instance_id = secrets.token_hex(8)
        fastapi_app = create_app(
            app_paths=self.app_paths,
            port=chosen_port,
            bootstrap_token=launch_token,
            instance_id=instance_id,
        )

        class _DesktopUvicornServer(uvicorn.Server):
            def install_signal_handlers(self) -> None:
                # Embedded server runs on a background thread; signals are handled on the GUI thread.
                return

        config = uvicorn.Config(
            fastapi_app,
            host=LOCAL_HOST,
            port=chosen_port,
            log_level="warning",
            timeout_graceful_shutdown=1,
        )
        server = _DesktopUvicornServer(config)
        serve_errors: list[BaseException] = []

        def _serve() -> None:
            try:
                server.run()
            except BaseException as exc:  # pragma: no cover - captured for diagnostics
                serve_errors.append(exc)
            finally:
                fastapi_app.state.event_bus.close()

        thread = threading.Thread(target=_serve, name="lightbox-desktop-server", daemon=True)
        thread.start()

        info = write_server_lock(
            chosen_port,
            app_paths=self.app_paths,
            host=LOCAL_HOST,
            pid=os.getpid(),
            instance_id=instance_id,
        )

        ready = False
        deadline = time.monotonic() + ready_timeout_s
        while time.monotonic() < deadline:
            if is_server_responsive(info, timeout_s=0.25):
                ready = True
                break
            if not thread.is_alive():
                break
            time.sleep(0.05)

        if not ready:
            server.should_exit = True
            fastapi_app.state.event_bus.close()
            if thread.is_alive():
                thread.join(timeout=1.0)
            clear_server_lock(
                app_paths=self.app_paths,
                only_if_pid=info.pid,
                only_if_instance_id=info.instance_id,
            )
            detail = (
                f"Local server failed on {LOCAL_HOST}:{chosen_port}: {serve_errors[0]}"
                if serve_errors
                else f"Local server on {LOCAL_HOST}:{chosen_port} did not become responsive within {ready_timeout_s:.1f}s."
            )
            raise DesktopServerStartError(detail, log_path=self.log_path)

        # Internal one-time launch URL (never printed, logged, or returned to MCP)
        launch_url = f"http://{LOCAL_HOST}:{chosen_port}/auth/launch?token={launch_token}"
        return launch_url, info, server, thread, fastapi_app

    def _start_owned_server(self, *, port: int = 0) -> tuple[str, ServerLockInfo]:
        attempt_ports = [port, 0] if port > 0 else [0, 0]
        last_error: BaseException | None = None

        for attempt_port in attempt_ports:
            try:
                launch_url, info, server, thread, fastapi_app = self._start_single_attempt(
                    port=attempt_port,
                    ready_timeout_s=self.ready_timeout_s,
                )
            except Exception as exc:
                last_error = exc
                print(f"Lightbox desktop server start attempt failed (port={attempt_port}): {exc}")
                continue

            lifecycle = getattr(fastapi_app.state, "lifecycle", None)
            if lifecycle is not None and hasattr(lifecycle, "on_file_ops_drained"):
                lifecycle.on_file_ops_drained(self._on_file_ops_drained)

            with self._lock:
                self._owns_server = True
                self._owned_lock = info
                self._server = server
                self._server_thread = thread
                self._fastapi_app = fastapi_app
                self._base_url = info.base_url
                self._pending_close = False

            return launch_url, info

        if isinstance(last_error, DesktopServerStartError):
            raise last_error
        raise DesktopServerStartError(
            f"Unable to start local Lightbox server after retry: {last_error}",
            log_path=self.log_path,
        ) from last_error

    def ensure_server_and_url(self) -> tuple[str, bool]:
        """Reuse a healthy existing local server or start an owned embedded Uvicorn instance.

        Returns `(initial_window_url, started_new_server)`.
        """
        existing = read_server_lock(
            app_paths=self.app_paths,
            check_alive=True,
            verify_http=True,
            timeout_s=0.5,
        )
        if existing is not None:
            with self._lock:
                self._owns_server = False
                self._owned_lock = None
                self._base_url = existing.base_url
            return existing.base_url, False

        launch_url, _info = self._start_owned_server(port=self.requested_port)
        return launch_url, True

    def pick_folder(self) -> dict[str, Any]:
        """Invoke `window.create_file_dialog(webview.FileDialog.FOLDER)` natively."""
        win = self._window
        if win is None or not hasattr(win, "create_file_dialog"):
            return {"path": None, "canceled": True}
        result = win.create_file_dialog(self._folder_dialog_const)
        chosen: str | None = None
        if isinstance(result, str) and result.strip():
            chosen = result.strip()
        elif isinstance(result, (list, tuple)) and result:
            first = str(result[0]).strip()
            chosen = first or None
        return {"path": chosen, "canceled": chosen is None}

    def reconnect(self) -> dict[str, Any]:
        """Restart/start an owned local server on a fresh port and navigate the window to it."""
        if self._owns_server:
            self.shutdown(timeout_s=min(self.shutdown_timeout_s, 3.0))
        launch_url, info = self._start_owned_server(port=0)
        win = self._window
        if win is not None and hasattr(win, "load_url"):
            win.load_url(launch_url)
        return {"ok": True, "url": info.base_url}

    def _on_file_ops_drained(self) -> None:
        """Automatically complete a previously blocked window close once in-flight file moves finish."""
        with self._lock:
            if not self._pending_close or self.in_flight_file_ops > 0:
                return
            self._pending_close = False
            win = self._window

        if win is not None and hasattr(win, "destroy"):

            def _destroy_window() -> None:
                try:
                    win.destroy()
                except Exception:
                    pass

            threading.Thread(
                target=_destroy_window,
                name="lightbox-deferred-close",
                daemon=True,
            ).start()

    def on_closing(self) -> bool:
        """Unified close handler for window `×` and macOS `Cmd+Q`.

        Runs synchronously on the GUI main thread (`windowShouldClose_` on macOS):
        - Never calls `win.evaluate_js(...)` synchronously (dispatches on a background thread to avoid Cocoa deadlock).
        - Never blocks on `shutdown()` (returns `True` immediately so cleanup runs after `webview.start()` returns).
        - If file moves are in flight, records pending close and auto-completes closing when moves drain to 0.
        """
        if self.in_flight_file_ops > 0:
            with self._lock:
                self._pending_close = True
                win = self._window

            # Re-check in case file ops finished concurrently right before `_pending_close` was set
            if self.in_flight_file_ops == 0:
                with self._lock:
                    self._pending_close = False
                return True

            if win is not None and hasattr(win, "evaluate_js"):

                def _notify_close_blocked() -> None:
                    try:
                        win.evaluate_js(
                            "window.dispatchEvent(new CustomEvent('lightbox:close-blocked'))"
                        )
                    except Exception:
                        pass

                threading.Thread(
                    target=_notify_close_blocked,
                    name="lightbox-notify-close-blocked",
                    daemon=True,
                ).start()
            return False

        return True

    def shutdown(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Gracefully stop only the server owned by this desktop process with a strict timeout ceiling."""
        with self._lock:
            owns = self._owns_server
            server = self._server
            thread = self._server_thread
            app = self._fastapi_app
            owned_lock = self._owned_lock
            self._owns_server = False
            self._server = None
            self._server_thread = None
            self._fastapi_app = None
            self._owned_lock = None
            self._pending_close = False

        if not owns:
            return {"owned": False, "workers_finished": True, "timed_out": False}

        eff_timeout = self.shutdown_timeout_s if timeout_s is None else max(0.05, timeout_s)
        workers_finished = True
        if app is not None:
            lifecycle = getattr(app.state, "lifecycle", None)
            if lifecycle is not None:
                workers_finished = bool(lifecycle.request_shutdown(timeout_s=eff_timeout))
            event_bus = getattr(app.state, "event_bus", None)
            if event_bus is not None:
                event_bus.close()

        if server is not None:
            server.should_exit = True
        if thread is not None and thread.is_alive():
            thread.join(timeout=min(2.0, eff_timeout))

        if owned_lock is not None:
            clear_server_lock(
                app_paths=self.app_paths,
                only_if_pid=owned_lock.pid,
                only_if_instance_id=owned_lock.instance_id,
            )

        return {
            "owned": True,
            "workers_finished": workers_finished,
            "timed_out": not workers_finished,
        }


def _resolve_desktop_icon_path(app_paths: AppPaths) -> str | None:
    """Resolve optional desktop icon file (from macOS .app bundle env or `<APP_DATA_DIR>/assets/`)."""
    env_icon = os.environ.get("LIGHTBOX_APP_ICON", "").strip()
    if env_icon and Path(env_icon).is_file():
        return env_icon
    candidates = [
        app_paths.root / "assets" / "LightBox.icns",
        app_paths.root / "assets" / "lightbox.png",
        Path.home() / "Applications" / "LightBox.app" / "Contents" / "Resources" / "LightBox.icns",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _start_webview(webview_mod: Any, *, icon_path: str | None = None) -> None:
    """Start pywebview event loop, forwarding `icon` when supported by the backend/version."""
    if icon_path:
        try:
            webview_mod.start(icon=icon_path)
            return
        except TypeError:
            pass
    webview_mod.start()


def _show_startup_error(webview_mod: Any | None, error_detail: str, log_path: Path) -> None:
    """Display startup failure inline in a pywebview window, falling back to a native OS dialog."""
    if webview_mod is not None:
        try:
            webview_mod.create_window(
                f"{DEFAULT_WINDOW_TITLE} - 启动失败",
                html=_build_startup_error_html(error_detail, log_path),
                width=640,
                height=420,
                min_size=(480, 320),
            )
            _start_webview(webview_mod)
            return
        except Exception:
            traceback.print_exc()

    _show_native_error_dialog(
        f"{DEFAULT_WINDOW_TITLE} 启动失败",
        f"{error_detail}\n\n请查看日志：{log_path}",
    )


def main(port: int = 0) -> None:
    """Unified desktop entry point for `lightbox-app`, `lightbox app`, and `python -m lightbox.cli app`."""
    paths = get_app_paths()
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_handle = redirect_desktop_logs(paths)
    log_path = log_handle.path

    webview_mod: Any | None = None
    controller: DesktopAppController | None = None

    try:
        try:
            webview_mod = _import_webview()
        except ImportError:
            install_hint = (
                "pywebview is not installed. Install desktop extras with "
                "`uv pip install 'lightbox-cull[desktop]'` (on Linux also install system GTK "
                "bindings such as `python3-gi gir1.2-webkit2-4.1` or `PySide6`)."
            )
            if _is_interactive_cli(orig_stdout, orig_stderr):
                msg = f"{install_hint} Falling back to browser UI (`lightbox ui`)."
                print(msg)
                if orig_stderr is not None:
                    try:
                        orig_stderr.write(msg + "\n")
                        orig_stderr.flush()
                    except OSError:
                        pass
                # Keep stdout/stderr redirected to desktop.log on CLI fallback as well
                from lightbox.cli.app import ui as cli_ui

                cli_ui(port=port, no_browser=False)
            else:
                print(f"{install_hint} Non-interactive launch detected; exiting without headless server.")
                _show_native_error_dialog(
                    f"{DEFAULT_WINDOW_TITLE} 需要桌面依赖",
                    f"{install_hint}\n\n日志位置：{log_path}",
                )
            return

        controller = DesktopAppController(app_paths=paths, port=port, log_path=log_path)
        initial_url, started_new = controller.ensure_server_and_url()
        print(
            f"Lightbox desktop window attached to {controller.base_url} "
            f"(owns_server={started_new})"
        )

        js_api = DesktopJsApi(controller)
        folder_dialog = _resolve_folder_dialog_const(webview_mod)
        window = webview_mod.create_window(
            DEFAULT_WINDOW_TITLE,
            url=initial_url,
            width=DEFAULT_WINDOW_WIDTH,
            height=DEFAULT_WINDOW_HEIGHT,
            min_size=(960, 640),
            js_api=js_api,
        )
        controller.attach_window(window, folder_dialog=folder_dialog)
        if hasattr(window, "events") and hasattr(window.events, "closing"):
            window.events.closing += controller.on_closing

        _start_webview(webview_mod, icon_path=_resolve_desktop_icon_path(paths))
    except Exception as exc:
        traceback.print_exc()
        if controller is not None:
            controller.shutdown()
            controller = None
        _show_startup_error(webview_mod, str(exc), log_path)
    finally:
        if controller is not None:
            controller.shutdown()
        try:
            log_handle.flush()
            log_handle.close()
        except OSError:
            pass
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr


__all__ = [
    "DEFAULT_SERVER_READY_TIMEOUT_S",
    "DEFAULT_SHUTDOWN_TIMEOUT_S",
    "DesktopAppController",
    "DesktopJsApi",
    "DesktopLogHandle",
    "DesktopServerStartError",
    "main",
    "redirect_desktop_logs",
]
