"""Cross-platform process liveness and PID-reuse-safe ownership helpers (zero `os.kill(pid, 0)`)."""

from __future__ import annotations

import os

import psutil

DEFAULT_CREATE_TIME_TOLERANCE_S = 2.0


def is_pid_alive(pid: int) -> bool:
    """Return True if `pid` refers to an active non-zombie OS process (cross-platform, safe on Windows)."""
    if pid <= 0:
        return False
    if not psutil.pid_exists(pid):
        return False
    try:
        proc = psutil.Process(pid)
        return bool(proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        # Process exists, owned by another user / elevated privilege
        return True
    except psutil.Error:
        return False


def get_process_create_time(pid: int | None = None) -> float | None:
    """Return OS process start timestamp (epoch seconds) for PID-reuse detection."""
    target_pid = os.getpid() if pid is None else pid
    if target_pid <= 0:
        return None
    try:
        return float(psutil.Process(target_pid).create_time())
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error, OSError):
        return None


def is_process_owner_alive(
    owner_pid: int | None,
    owner_pid_started_at: float | None,
    *,
    tolerance_s: float = DEFAULT_CREATE_TIME_TOLERANCE_S,
) -> bool:
    """Verify whether the exact process `(owner_pid, owner_pid_started_at)` is still alive.

    Protects against OS PID recycling: if `owner_pid` exists but its `create_time()` differs
    from `owner_pid_started_at` by more than `tolerance_s`, returns False.
    """
    if owner_pid is None or owner_pid <= 0:
        return False
    if not psutil.pid_exists(owner_pid):
        return False
    try:
        proc = psutil.Process(owner_pid)
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return False
        if owner_pid_started_at is not None:
            actual_started = float(proc.create_time())
            if abs(actual_started - owner_pid_started_at) > tolerance_s:
                return False
        return True
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        # Cannot inspect create_time of an elevated/foreign process; assume alive to avoid false pause
        return True
    except (psutil.Error, OSError):
        return False


__all__ = [
    "DEFAULT_CREATE_TIME_TOLERANCE_S",
    "get_process_create_time",
    "is_pid_alive",
    "is_process_owner_alive",
]
