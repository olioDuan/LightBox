"""Step 4 unit tests: desktop launcher, lazy webview import, log redirection, ownership, lifecycle & PID-safe healing."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from typer.testing import CliRunner

from lightbox.api import (
    CSRF_HEADER_NAME,
    LOCAL_HOST,
    create_app,
    is_pid_alive,
    read_server_lock,
    write_server_lock,
)
from lightbox.api.app import AppLifecycleController
from lightbox.api.events import EventBus
from lightbox.cli import app as cli_app
from lightbox.cli.desktop import (
    DesktopAppController,
    DesktopJsApi,
    DesktopServerStartError,
    _resolve_folder_dialog_const,
    redirect_desktop_logs,
)
from lightbox.cli.desktop import main as desktop_main
from lightbox.jobs import JobRunner
from lightbox.models import GenerationResult, Message, ModelCaps, ModelRouter, Usage
from lightbox.paths import get_app_paths
from lightbox.process_utils import get_process_create_time, is_process_owner_alive
from lightbox.services import LightboxServices
from lightbox.services.triage import YellowTriageOutput
from lightbox.store import JobRecord

REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeEventSlot:
    def __init__(self) -> None:
        self.handlers: list[Any] = []

    def __iadd__(self, fn: Any) -> _FakeEventSlot:
        self.handlers.append(fn)
        return self


class _FakeEvents:
    def __init__(self) -> None:
        self.closing = _FakeEventSlot()


class _FakeWebviewWindow:
    """In-memory fake pywebview Window for testing closing guard, folder dialog, and reconnect."""

    def __init__(self, *, folder_result: Any = None) -> None:
        self.events = _FakeEvents()
        self.folder_result = folder_result
        self.dialog_calls: list[Any] = []
        self.loaded_urls: list[str] = []
        self.js_calls: list[str] = []
        self.destroyed = threading.Event()

    def create_file_dialog(self, dialog_type: Any) -> Any:
        self.dialog_calls.append(dialog_type)
        return self.folder_result

    def load_url(self, url: str) -> None:
        self.loaded_urls.append(url)

    def evaluate_js(self, script: str) -> None:
        self.js_calls.append(script)

    def destroy(self) -> None:
        self.destroyed.set()


class _MainThreadLoopFakeWindow:
    """Simulates pywebview's macOS Cocoa main-thread dispatch + blocking semaphore in `evaluate_js`.

    When `evaluate_js(script)` is called, it enqueues work onto the simulated GUI main thread
    and blocks waiting on a per-call semaphore (`threading.Event`). Therefore, calling
    `evaluate_js` synchronously from inside `on_closing` on the GUI main thread immediately
    deadlocks until timeout — matching real Cocoa `windowShouldClose_` semantics.
    """

    def __init__(self, *, js_timeout_s: float = 2.0) -> None:
        self.events = _FakeEvents()
        self._js_timeout_s = js_timeout_s
        self._queue: queue.Queue[tuple[str, threading.Event]] = queue.Queue()
        self.js_calls: list[str] = []
        self.destroyed = threading.Event()
        self._lock = threading.Lock()

    def evaluate_js(self, script: str) -> None:
        done = threading.Event()
        self._queue.put((script, done))
        if not done.wait(timeout=self._js_timeout_s):
            raise TimeoutError("Deadlock: evaluate_js waited for main thread loop")

    def pump_once(self, *, timeout_s: float = 0.5) -> bool:
        try:
            script, done = self._queue.get(timeout=timeout_s)
        except queue.Empty:
            return False
        with self._lock:
            self.js_calls.append(script)
        done.set()
        return True

    def destroy(self) -> None:
        self.destroyed.set()


class _FakeTTYStream:
    def __init__(self, *, tty: bool) -> None:
        self._tty = tty
        self.written: list[str] = []

    def isatty(self) -> bool:
        return self._tty

    def write(self, s: str) -> int:
        self.written.append(s)
        return len(s)

    def flush(self) -> None:
        return


# ---------------------------------------------------------------------------
# 1. Packaging, CLI entrypoints, lazy webview import, and console-less log redirect
# ---------------------------------------------------------------------------


def test_pyproject_cli_entries_and_lazy_webview_import() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = pyproject["project"]["dependencies"]
    assert any(d.startswith("psutil") for d in deps), "psutil must be in main dependencies"
    assert any(
        d.startswith("pywebview")
        for d in pyproject["project"]["optional-dependencies"]["desktop"]
    )
    assert pyproject["project"]["gui-scripts"]["lightbox-app"] == "lightbox.cli.desktop:main"

    # CLI `lightbox app --help` registered
    runner = CliRunner()
    res = runner.invoke(cli_app, ["app", "--help"])
    assert res.exit_code == 0
    plain = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", res.output)
    assert "--port" in plain

    # `python -m lightbox.cli --help` works via __main__.py
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    proc = subprocess.run(
        [sys.executable, "-m", "lightbox.cli", "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "app" in proc.stdout


def test_redirect_desktop_logs_handles_none_stdio_rotation_and_second_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = get_app_paths(tmp_path / "desk_logs")
    orig_out, orig_err = sys.stdout, sys.stderr
    try:
        # Simulate Windows pythonw.exe / gui-scripts where stdout & stderr are None
        monkeypatch.setattr(sys, "stdout", None)
        monkeypatch.setattr(sys, "stderr", None)

        h1 = redirect_desktop_logs(paths)
        try:
            assert h1.is_primary is True
            assert h1.path == paths.desktop_log_file
            print("first-run-stdout")
            sys.stderr.write("first-run-stderr\n")
            h1.flush()

            # Concurrent second desktop instance MUST NOT overwrite or rotate primary desktop.log
            h_second = redirect_desktop_logs(paths)
            try:
                assert h_second.is_primary is False
                assert h_second.path != paths.desktop_log_file
                assert h_second.path.name == f"desktop-{os.getpid()}.log"
                print("second-instance-line")
                h_second.flush()
            finally:
                h_second.close()

            # Primary desktop.log is intact and desktop.prev.log was not prematurely touched
            assert "first-run-stdout" in paths.desktop_log_file.read_text(encoding="utf-8")
            assert "second-instance-line" not in paths.desktop_log_file.read_text(encoding="utf-8")
            assert "second-instance-line" in h_second.path.read_text(encoding="utf-8")
            assert not paths.desktop_prev_log_file.exists()
        finally:
            h1.close()

        # After first instance exits, next launch acquires primary lock and rotates cleanly
        h2 = redirect_desktop_logs(paths)
        try:
            assert h2.is_primary is True
            assert h2.path == paths.desktop_log_file
            print("next-run-stdout")
            h2.flush()
        finally:
            h2.close()

        assert "first-run-stdout" in paths.desktop_prev_log_file.read_text(encoding="utf-8")
        assert "next-run-stdout" in paths.desktop_log_file.read_text(encoding="utf-8")
        assert "first-run-stdout" not in paths.desktop_log_file.read_text(encoding="utf-8")
    finally:
        sys.stdout = orig_out
        sys.stderr = orig_err


def test_desktop_main_missing_pywebview_cli_fallback_vs_icon_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lightbox.cli.desktop._import_webview",
        lambda: (_ for _ in ()).throw(ImportError("No module named 'webview'")),
    )
    cli_app_mod = importlib.import_module("lightbox.cli.app")

    # Case A: Interactive CLI (TTY) falls back to `lightbox ui` while keeping stdout/stderr on desktop.log
    cli_home = tmp_path / "fallback_cli"
    monkeypatch.setenv("LIGHTBOX_HOME", str(cli_home))
    ui_calls: list[tuple[int, bool, bool]] = []
    alerts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "lightbox.cli.desktop._show_native_error_dialog",
        lambda title, msg: alerts.append((title, msg)),
    )

    def _fake_cli_ui(*, port: int = 0, no_browser: bool = False) -> None:
        # Verify stdio remained redirected to the log handle (never restored to None)
        ui_calls.append((port, no_browser, sys.stdout is not None and sys.stderr is not None))

    monkeypatch.setattr(cli_app_mod, "ui", _fake_cli_ui)
    monkeypatch.setattr(sys, "stdout", _FakeTTYStream(tty=True))
    monkeypatch.setattr(sys, "stderr", _FakeTTYStream(tty=True))

    desktop_main(port=9123)
    assert ui_calls == [(9123, False, True)]
    assert alerts == []
    log_text = get_app_paths(cli_home).desktop_log_file.read_text(encoding="utf-8")
    assert "pywebview is not installed" in log_text

    # Case B: Icon/shortcut launch (console-less: stdout/stderr None or non-TTY) shows native dialog and exits
    icon_home = tmp_path / "fallback_icon"
    monkeypatch.setenv("LIGHTBOX_HOME", str(icon_home))
    ui_calls.clear()
    alerts.clear()
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    desktop_main(port=9124)
    assert ui_calls == [], "Non-interactive icon launch must NOT spawn an orphaned headless server"
    assert len(alerts) == 1
    assert "lightbox-cull[desktop]" in alerts[0][1]


def test_resolve_folder_dialog_const_prefers_pywebview6_enum() -> None:
    class _V6FileDialog:
        FOLDER = 99

    class _V6Webview:
        FileDialog = _V6FileDialog

        @property
        def FOLDER_DIALOG(self) -> int:
            raise AssertionError("Must not access deprecated FOLDER_DIALOG when FileDialog.FOLDER exists")

    class _V5Webview:
        FOLDER_DIALOG = 20

    assert _resolve_folder_dialog_const(_V6Webview()) == 99
    assert _resolve_folder_dialog_const(_V5Webview()) == 20


# ---------------------------------------------------------------------------
# 2. Cross-platform PID liveness (no os.kill) & PID-reuse-safe job healing
# ---------------------------------------------------------------------------


def test_is_pid_alive_uses_psutil_and_never_calls_os_kill_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lightbox.api.server_lock as server_lock_mod
    import lightbox.process_utils as process_utils_mod

    # Neither our process_utils nor server_lock may directly call os.kill (unsafe on Windows!)
    for mod in (process_utils_mod, server_lock_mod):
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "kill"
            ):
                raise AssertionError(f"Forbidden os.kill call found in {mod.__name__}")

    # Simulate Windows-safe psutil probe without touching os.kill at all
    def _forbidden_kill(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("os.kill must NOT be called when psutil handles PID probing!")

    class _FakeProc:
        def __init__(self, pid: int) -> None:
            self._pid = pid

        def is_running(self) -> bool:
            return self._pid == 4242

        def status(self) -> str:
            return "running"

    monkeypatch.setattr(os, "kill", _forbidden_kill)
    monkeypatch.setattr(process_utils_mod.psutil, "pid_exists", lambda p: p == 4242)
    monkeypatch.setattr(process_utils_mod.psutil, "Process", _FakeProc)

    assert is_pid_alive(4242) is True
    assert is_pid_alive(-1) is False
    assert is_pid_alive(0) is False
    assert is_pid_alive(999_999_999) is False


def test_reconcile_orphaned_jobs_respects_live_owner_and_catches_pid_reuse(
    tmp_path: Path,
) -> None:
    paths = get_app_paths(tmp_path / "orphan_home")
    my_pid = os.getpid()
    my_started = get_process_create_time(my_pid)
    assert my_started is not None
    assert is_process_owner_alive(my_pid, my_started) is True
    assert is_process_owner_alive(my_pid, my_started - 3600.0) is False

    with LightboxServices(app_paths=paths) as svcs:
        now = time.time()
        from lightbox.store import RootRecord, SessionRecord

        svcs.repo.upsert_root(RootRecord(id="r1", path=str(tmp_path), device_id="dev1", kind="disk"))
        svcs.repo.upsert_session(
            SessionRecord(
                id="s1",
                root_id="r1",
                root_path=str(tmp_path),
                name="s1",
                sensitivity="conservative",
                engine_version="0.5.0",
                created_at=now,
            )
        )

        # 1. Live job owned by current live process with matching create_time (even with old heartbeat!)
        svcs.repo.create_job(
            JobRecord(
                id="job_live_long",
                kind="triage_vlm",
                session_id="s1",
                status="running",
                created_at=now - 600.0,
                heartbeat_at=now - 500.0,
                owner_pid=my_pid,
                owner_pid_started_at=my_started,
                owner_instance_id="inst_live",
            )
        )
        # 2. Recycled PID: PID exists (my_pid) but create_time does NOT match!
        svcs.repo.create_job(
            JobRecord(
                id="job_pid_recycled",
                kind="scan",
                session_id="s1",
                status="running",
                created_at=now - 10.0,
                heartbeat_at=now - 2.0,
                owner_pid=my_pid,
                owner_pid_started_at=my_started - 9999.0,
                owner_instance_id="inst_old",
            )
        )
        # 3. Dead PID: non-existent PID
        svcs.repo.create_job(
            JobRecord(
                id="job_dead_pid",
                kind="triage_vlm",
                session_id="s1",
                status="queued",
                created_at=now - 5.0,
                heartbeat_at=now - 1.0,
                owner_pid=999_999_999,
                owner_pid_started_at=now - 20.0,
                owner_instance_id="inst_dead",
            )
        )

        runner = JobRunner(svcs.repo)
        healed = runner.reconcile_orphaned_jobs(current_instance_id="inst_live")
        assert set(healed) == {"job_pid_recycled", "job_dead_pid"}

        j_live = svcs.repo.get_job("job_live_long")
        j_recycled = svcs.repo.get_job("job_pid_recycled")
        j_dead = svcs.repo.get_job("job_dead_pid")
        assert j_live is not None and j_live.status == "running"
        assert j_recycled is not None and j_recycled.status == "paused"
        assert j_recycled.paused_reason == "owner_process_exited"
        assert j_dead is not None and j_dead.status == "paused"
        assert j_dead.paused_reason == "owner_process_exited"


def test_reconnect_does_not_pause_job_while_old_instance_worker_is_still_alive(
    tmp_path: Path,
) -> None:
    paths = get_app_paths(tmp_path / "dual_worker_home")
    my_pid = os.getpid()
    my_started = get_process_create_time(my_pid)

    app_old = create_app(app_paths=paths, port=18801, instance_id="inst_old_live")
    with LightboxServices(app_paths=paths, migrate=False) as svcs:
        from lightbox.store import RootRecord, SessionRecord

        now = time.time()
        svcs.repo.upsert_root(RootRecord(id="r_dw", path=str(tmp_path), device_id="dev_dw", kind="disk"))
        svcs.repo.upsert_session(
            SessionRecord(
                id="s_dw",
                root_id="r_dw",
                root_path=str(tmp_path),
                name="s_dw",
                sensitivity="conservative",
                engine_version="0.5.0",
                created_at=now,
            )
        )
        svcs.repo.create_job(
            JobRecord(
                id="job_still_running",
                kind="triage_vlm",
                session_id="s_dw",
                status="running",
                created_at=now,
                heartbeat_at=now,
                owner_pid=my_pid,
                owner_pid_started_at=my_started,
                owner_instance_id="inst_old_live",
            )
        )

    worker_started = threading.Event()
    release_worker = threading.Event()

    def _long_worker() -> None:
        worker_started.set()
        release_worker.wait(timeout=10.0)

    worker_thread = app_old.state.lifecycle.spawn_daemon_worker(
        name="test-old-instance-worker",
        target=_long_worker,
    )
    assert worker_started.wait(timeout=1.0)
    try:
        # Simulate reconnect spinning up a new app instance while old worker outlived shutdown timeout
        create_app(app_paths=paths, port=18802, instance_id="inst_new_reconnect")
        with LightboxServices(app_paths=paths, migrate=False) as svcs_check:
            j = svcs_check.repo.get_job("job_still_running")
            assert j is not None
            assert j.status == "running", "Must NOT mark job paused while old worker is still alive"
    finally:
        release_worker.set()
        worker_thread.join(timeout=2.0)

    # Once old worker truly exits, subsequent instance startup reconciles it cleanly
    create_app(app_paths=paths, port=18803, instance_id="inst_after_exit")
    with LightboxServices(app_paths=paths, migrate=False) as svcs_check2:
        j2 = svcs_check2.repo.get_job("job_still_running")
        assert j2 is not None
        assert j2.status == "paused"
        assert j2.paused_reason == "owner_instance_restarted"


# ---------------------------------------------------------------------------
# 3. Daemon triage worker + cooperative stop_event + bounded shutdown timeout
# ---------------------------------------------------------------------------


def test_triage_daemon_worker_cooperative_stop_and_bounded_shutdown_timeout(
    tmp_path: Path,
) -> None:
    # A. Stuck daemon worker does NOT hang request_shutdown past timeout_s
    bus = EventBus()
    lc = AppLifecycleController(bus, instance_id="inst_lc")
    release_stuck = threading.Event()
    started = threading.Event()

    def _stuck_worker() -> None:
        started.set()
        release_stuck.wait(timeout=10.0)

    t = lc.spawn_daemon_worker(name="test-stuck-triage", target=_stuck_worker)
    assert t.daemon is True
    assert started.wait(timeout=1.0)

    t0 = time.monotonic()
    finished = lc.request_shutdown(timeout_s=0.15)
    elapsed = time.monotonic() - t0
    assert lc.stop_event.is_set() is True
    assert finished is False
    assert elapsed < 1.5
    release_stuck.set()
    t.join(timeout=1.0)

    # B. Triage respects cooperative `stop_event` and pauses cleanly with `shutdown_requested`
    shoot = tmp_path / "shoot_stop"
    shoot.mkdir()
    for i in range(4):
        im = Image.new("RGB", (128, 128), (80 + i * 20, 90, 100))
        im.save(shoot / f"IMG_{i:02d}.jpg", format="JPEG")

    stop_ev = threading.Event()
    calls_seen = 0

    class _CooperativeStopAdapter:
        id = "fake-triage"
        provider = "openai"
        family = "openai"
        caps = ModelCaps(vision=True, tools=True, json_schema=True, prompt_cache=False)

        async def generate(self, messages: list[Message], **kwargs: Any) -> GenerationResult:
            nonlocal calls_seen
            del messages, kwargs
            calls_seen += 1
            stop_ev.set()
            parsed = YellowTriageOutput(
                decision="keep", confidence=0.9, reason_code="vlm_keep", rationale="sharp"
            )
            return GenerationResult(
                text=parsed.model_dump_json(),
                parsed=parsed,
                usage=Usage(in_tokens=10, out_tokens=10),
                provider="openai",
                model="fake-triage",
            )

    paths = get_app_paths(tmp_path / "triage_stop_home")
    with LightboxServices(app_paths=paths) as svcs:
        scan_res = svcs.scan.scan_folder(shoot, max_workers=1)
        for p in svcs.repo.list_photos(scan_res.session_id):
            svcs.review.set_verdict(p.id, bucket="yellow", source="engine")

        router = ModelRouter(
            svcs.repo,
            paths,
            svcs.config,
            adapters={"triage_vlm": _CooperativeStopAdapter()},
            max_concurrency=1,
        )
        svcs.override_model_router(router)
        res = svcs.triage.run_triage_sync(
            scan_res.session_id,
            stop_event=stop_ev,
            owner_instance_id="inst_stop",
        )
        assert res.status == "paused"
        assert res.paused_reason == "shutdown_requested"
        assert calls_seen == 1
        job = svcs.repo.get_job(res.job_id)
        assert job is not None
        assert job.status == "paused"
        assert job.owner_pid == os.getpid()
        assert job.owner_instance_id == "inst_stop"


# ---------------------------------------------------------------------------
# 4. Main-thread loop fake window: zero-deadlock on_closing, non-blocking close,
#    deferred auto-close, dialog, reconnect, and startup failure error page
# ---------------------------------------------------------------------------


def test_on_closing_main_thread_no_deadlock_auto_resume_and_fast_close(
    tmp_path: Path,
) -> None:
    """Verifies Cocoa-style main-thread `on_closing` never deadlocks on `evaluate_js`,
    auto-completes closing when file ops finish, and never blocks the GUI thread on shutdown."""
    paths = get_app_paths(tmp_path / "main_loop_home")
    controller = DesktopAppController(app_paths=paths, port=0, shutdown_timeout_s=0.5)
    controller.ensure_server_and_url()
    win = _MainThreadLoopFakeWindow(js_timeout_s=1.5)
    controller.attach_window(win, folder_dialog=42)

    lifecycle = controller._fastapi_app.state.lifecycle
    stuck_release = threading.Event()
    try:
        # 1. With in-flight file op: calling on_closing() directly on the simulated main thread
        #    MUST return False immediately (if evaluate_js ran synchronously here it would deadlock!)
        with lifecycle.track_file_op():
            t0 = time.monotonic()
            allowed = controller.on_closing()
            elapsed = time.monotonic() - t0
            assert allowed is False
            assert elapsed < 0.2, f"on_closing blocked main thread for {elapsed:.2f}s"
            # Now yield the simulated main thread to pump its event queue -> JS arrives cleanly!
            assert win.pump_once(timeout_s=1.0) is True
            assert any("lightbox:close-blocked" in js for js in win.js_calls)
            assert not win.destroyed.is_set()

        # 2. As soon as file op exits (in_flight_file_ops -> 0), deferred close triggers win.destroy()!
        assert win.destroyed.wait(timeout=1.0) is True

        # 3. With a stuck daemon worker (and 0 file ops), on_closing() MUST return True immediately
        #    without blocking the GUI main thread on shutdown()!
        lifecycle.spawn_daemon_worker(
            name="test-stuck-main-thread-check",
            target=lambda: stuck_release.wait(timeout=10.0),
        )
        t1 = time.monotonic()
        allowed_now = controller.on_closing()
        close_elapsed = time.monotonic() - t1
        assert allowed_now is True
        assert close_elapsed < 0.2, f"on_closing froze main thread for {close_elapsed:.2f}s"
        # Server cleanup happens AFTER webview.start() returns, in controller.shutdown()
        assert controller.owns_server is True
        res = controller.shutdown(timeout_s=0.15)
        assert res["owned"] is True
        assert res["timed_out"] is True
        assert controller.owns_server is False
        assert read_server_lock(app_paths=paths, check_alive=True) is None
    finally:
        stuck_release.set()
        controller.shutdown(timeout_s=0.2)


def test_desktop_controller_fake_window_dialog_and_reconnect(
    tmp_path: Path,
) -> None:
    paths = get_app_paths(tmp_path / "desktop_ctrl_home")
    controller = DesktopAppController(app_paths=paths, port=0, shutdown_timeout_s=0.2)

    initial_url, started_new = controller.ensure_server_and_url()
    assert started_new is True
    assert controller.owns_server is True
    assert "/auth/launch?token=" in initial_url
    first_lock = controller.owned_lock_info
    assert first_lock is not None

    win = _FakeWebviewWindow(folder_result=("/Users/tester/Photos",))
    controller.attach_window(win, folder_dialog=42)
    js_api = DesktopJsApi(controller)

    try:
        # Native folder dialog bridge
        picked = js_api.pick_folder()
        assert picked == {"path": "/Users/tester/Photos", "canceled": False}
        assert win.dialog_calls == [42]

        # Reconnect starts fresh server on new port and navigates window via load_url
        reconn = js_api.reconnect()
        assert reconn["ok"] is True
        assert "token=" not in str(reconn["url"])
        assert len(win.loaded_urls) == 1
        assert "/auth/launch?token=" in win.loaded_urls[0]
        second_lock = controller.owned_lock_info
        assert second_lock is not None
        assert second_lock.instance_id != first_lock.instance_id
    finally:
        controller.shutdown()


def test_desktop_controller_reuses_foreign_server_without_killing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = get_app_paths(tmp_path / "foreign_home")
    foreign_lock = write_server_lock(
        18765,
        app_paths=paths,
        host=LOCAL_HOST,
        pid=os.getpid(),
        instance_id="foreign_mcp_inst",
    )
    monkeypatch.setattr("lightbox.api.server_lock.is_server_responsive", lambda info, **_kw: True)

    controller = DesktopAppController(app_paths=paths)
    url, started_new = controller.ensure_server_and_url()
    assert started_new is False
    assert controller.owns_server is False
    assert url == foreign_lock.base_url

    # Closing desktop window + shutdown MUST NOT clear foreign server.lock
    assert controller.on_closing() is True
    assert controller.shutdown()["owned"] is False
    still_there = read_server_lock(app_paths=paths, check_alive=False)
    assert still_there is not None
    assert still_there.instance_id == "foreign_mcp_inst"


def test_desktop_startup_retries_on_failure_and_shows_inline_error_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A. Controller retries once on a new port when server never becomes responsive, then raises
    paths = get_app_paths(tmp_path / "start_fail_home")
    controller = DesktopAppController(app_paths=paths, port=19999, ready_timeout_s=0.15)
    attempts: list[int] = []

    def _fail_attempt(*, port: int, ready_timeout_s: float) -> Any:
        del ready_timeout_s
        attempts.append(port)
        raise DesktopServerStartError(f"port {port} busy", log_path=paths.desktop_log_file)

    monkeypatch.setattr(controller, "_start_single_attempt", _fail_attempt)
    with pytest.raises(DesktopServerStartError) as exc_info:
        controller._start_owned_server(port=19999)
    assert attempts == [19999, 0]
    assert exc_info.value.log_path == paths.desktop_log_file

    # B. desktop.main() catches startup failure (e.g. create_app crash) and renders inline error HTML
    err_home = tmp_path / "main_err_home"
    monkeypatch.setenv("LIGHTBOX_HOME", str(err_home))
    created_windows: list[dict[str, Any]] = []
    started_count = 0

    class _FakeWebviewModule:
        class FileDialog:
            FOLDER = 20

        @staticmethod
        def create_window(title: str, **kwargs: Any) -> _FakeWebviewWindow:
            created_windows.append({"title": title, **kwargs})
            return _FakeWebviewWindow()

        @staticmethod
        def start() -> None:
            nonlocal started_count
            started_count += 1

    monkeypatch.setattr("lightbox.cli.desktop._import_webview", lambda: _FakeWebviewModule)
    monkeypatch.setattr(
        "lightbox.cli.desktop.create_app",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("simulated sqlite init failure")),
    )

    desktop_main(port=0)
    assert started_count == 1
    assert len(created_windows) == 1
    err_html = str(created_windows[0].get("html", ""))
    assert "启动失败" in err_html
    assert "simulated sqlite init failure" in err_html
    assert str(get_app_paths(err_home).desktop_log_file) in err_html


# ---------------------------------------------------------------------------
# 5. Crash during apply_plan -> restart -> apply_plan(resume=True) completes safely
# ---------------------------------------------------------------------------


def test_apply_plan_resume_after_mid_apply_crash(tmp_path: Path) -> None:
    shoot = tmp_path / "shoot_resume"
    shoot.mkdir()
    for i in range(2):
        Image.new("RGB", (64, 64), (10 + i * 50, 20, 30)).save(
            shoot / f"R_{i}.jpg", format="JPEG"
        )

    paths = get_app_paths(tmp_path / "resume_home")
    with LightboxServices(app_paths=paths) as svcs:
        scan_res = svcs.scan.scan_folder(shoot, max_workers=1)
        photos = svcs.repo.list_photos(scan_res.session_id)
        pids = [p.id for p in photos]
        for pid in pids:
            svcs.review.set_verdict(pid, bucket="red", source="user")

        draft = svcs.plan.create_plan(
            scan_res.session_id,
            pids,
            strategy="subfolder",
        )
        assert draft.plan is not None
        plan, items = draft.plan, draft.items
        svcs.plan.approve_plan(plan.id, approved_by="test")

        # Simulate mid-apply crash: move item 0 manually via SafeFS, mark plan='applying' & item 0='done', leave item 1='pending'
        q_dir = Path(plan.quarantine_dir or "")
        q_dir.mkdir(parents=True, exist_ok=True)
        first_item = items[0]
        moved_dsts: list[str] = []
        for src_s, dst_s in zip(first_item.src_paths, first_item.dst_paths, strict=True):
            svcs.safe_fs.rename_same_volume(Path(src_s), Path(dst_s))
            moved_dsts.append(dst_s)
        svcs.repo.update_plan_status(plan.id, "applying")
        svcs.repo.update_plan_item(
            plan.id,
            first_item.photo_id,
            status="done",
            dst_paths=moved_dsts,
            done_at=time.time(),
        )

    # Fresh service container simulating restart after force-kill: resume completes item 1 cleanly
    with LightboxServices(app_paths=paths) as svcs2:
        res = svcs2.plan.apply_plan(plan.id, resume=True)
        assert res.status == "applied"
        assert res.moved_photo_count == 2
        plan_after, items_after = svcs2.plan.get_plan(plan.id)
        assert plan_after.status == "applied"
        assert all(it.status == "done" for it in items_after)
        for it in items_after:
            for dst_s in it.dst_paths:
                assert Path(dst_s).is_file()


def test_healthz_reports_in_flight_file_ops_and_create_app_heals_orphans(
    tmp_path: Path,
) -> None:
    from fastapi.testclient import TestClient

    paths = get_app_paths(tmp_path / "healthz_home")
    with LightboxServices(app_paths=paths) as svcs:
        from lightbox.store import RootRecord, SessionRecord

        now = time.time()
        svcs.repo.upsert_root(RootRecord(id="r_h", path=str(tmp_path), device_id="dev_h", kind="disk"))
        svcs.repo.upsert_session(
            SessionRecord(
                id="s_h",
                root_id="r_h",
                root_path=str(tmp_path),
                name="s_h",
                sensitivity="conservative",
                engine_version="0.5.0",
                created_at=now,
            )
        )
        svcs.repo.create_job(
            JobRecord(
                id="job_orphan_startup",
                kind="scan",
                session_id="s_h",
                status="running",
                created_at=now,
                heartbeat_at=now,
                owner_pid=999_999_999,
                owner_pid_started_at=now - 10.0,
                owner_instance_id="dead_inst",
            )
        )

    app = create_app(app_paths=paths, port=8765, bootstrap_token="tok-h")
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    hz = client.get("/healthz")
    assert hz.status_code == 200
    body = hz.json()
    assert body["service"] == "lightbox"
    assert body["in_flight_file_ops"] == 0
    assert CSRF_HEADER_NAME  # sanity import check

    # Startup reconciliation healed the orphaned job to `paused`
    with LightboxServices(app_paths=paths, migrate=False) as svcs_check:
        healed = svcs_check.repo.get_job("job_orphan_startup")
        assert healed is not None
        assert healed.status == "paused"
        assert healed.paused_reason == "owner_process_exited"
