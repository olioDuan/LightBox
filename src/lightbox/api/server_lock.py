"""Local server process lock (`<APP_DATA_DIR>/server.lock`) with live HTTP `/healthz` verification (SPEC §3.1, §8.1)."""

from __future__ import annotations

import http.client
import json
import os
import secrets
import socket
import time
from dataclasses import dataclass
from pathlib import Path

from lightbox.paths import AppPaths, get_app_paths

LOCAL_HOST = "127.0.0.1"
HEALTHZ_PATH = "/healthz"


@dataclass(frozen=True, slots=True)
class ServerLockInfo:
    pid: int
    host: str
    port: int
    started_at: float
    instance_id: str = ""

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def is_pid_alive(pid: int) -> bool:
    """Return True if `pid` refers to an active OS process."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def is_server_responsive(info: ServerLockInfo, *, timeout_s: float = 0.5) -> bool:
    """Probe `http://127.0.0.1:<port>/healthz` to confirm the lock belongs to a live Lightbox server.

    Prevents stale locks from blocking startup when an OS recycles a crashed process's PID.
    """
    conn = http.client.HTTPConnection(info.host, info.port, timeout=timeout_s)
    try:
        conn.request("GET", HEALTHZ_PATH, headers={"Host": f"{info.host}:{info.port}"})
        resp = conn.getresponse()
        if resp.status != 200:
            return False
        body = json.loads(resp.read(4096).decode("utf-8"))
    except (OSError, http.client.HTTPException, json.JSONDecodeError, UnicodeDecodeError):
        return False
    finally:
        conn.close()

    if not isinstance(body, dict):
        return False
    if body.get("service") != "lightbox":
        return False
    if int(body.get("pid", 0)) != info.pid:
        return False
    if info.instance_id and str(body.get("instance_id", "")) != info.instance_id:
        return False
    return True


def find_free_local_port(host: str = LOCAL_HOST) -> int:
    """Bind an ephemeral socket on `127.0.0.1` to select an available local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def write_server_lock(
    port: int,
    *,
    app_paths: AppPaths | None = None,
    host: str = LOCAL_HOST,
    pid: int | None = None,
    instance_id: str | None = None,
) -> ServerLockInfo:
    """Write active local UI server metadata to `<APP_DATA_DIR>/server.lock`."""
    paths = app_paths or get_app_paths()
    paths.ensure_dirs()
    info = ServerLockInfo(
        pid=pid if pid is not None else os.getpid(),
        host=host,
        port=port,
        started_at=time.time(),
        instance_id=instance_id if instance_id is not None else secrets.token_hex(8),
    )
    payload = {
        "pid": info.pid,
        "host": info.host,
        "port": info.port,
        "started_at": info.started_at,
        "instance_id": info.instance_id,
    }
    paths.server_lock_file.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return info


def read_server_lock(
    *,
    app_paths: AppPaths | None = None,
    check_alive: bool = True,
    verify_http: bool = True,
    timeout_s: float = 0.5,
) -> ServerLockInfo | None:
    """Read `<APP_DATA_DIR>/server.lock`, returning None if absent, dead, or not responding to `/healthz`."""
    paths = app_paths or get_app_paths()
    lock_path: Path = paths.server_lock_file
    if not lock_path.is_file():
        return None
    try:
        raw = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None

    pid = int(raw.get("pid", 0))
    port = int(raw.get("port", 0))
    host = str(raw.get("host", LOCAL_HOST))
    started_at = float(raw.get("started_at", 0.0))
    instance_id = str(raw.get("instance_id", ""))
    if pid <= 0 or port <= 0 or host != LOCAL_HOST:
        return None

    info = ServerLockInfo(
        pid=pid,
        host=host,
        port=port,
        started_at=started_at,
        instance_id=instance_id,
    )
    if check_alive:
        if not is_pid_alive(pid):
            return None
        if verify_http and not is_server_responsive(info, timeout_s=timeout_s):
            return None
    return info


def clear_server_lock(
    *,
    app_paths: AppPaths | None = None,
    only_if_pid: int | None = None,
) -> None:
    """Mark `<APP_DATA_DIR>/server.lock` as stopped without deleting files (no-unlink gate)."""
    paths = app_paths or get_app_paths()
    lock_path = paths.server_lock_file
    if not lock_path.is_file():
        return
    if only_if_pid is not None:
        current = read_server_lock(app_paths=paths, check_alive=False)
        if current is not None and current.pid != only_if_pid:
            return
    lock_path.write_text(
        json.dumps({"pid": 0, "host": LOCAL_HOST, "port": 0, "stopped_at": time.time()}) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "HEALTHZ_PATH",
    "LOCAL_HOST",
    "ServerLockInfo",
    "clear_server_lock",
    "find_free_local_port",
    "is_pid_alive",
    "is_server_responsive",
    "read_server_lock",
    "write_server_lock",
]
