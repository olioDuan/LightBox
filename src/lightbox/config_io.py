"""Atomic, locked configuration file I/O strictly confined to `<APP_DATA_DIR>`.

This module is the sole module whitelisted for `os.replace` outside `SafeFS` / `cache_lru`.
It enforces:
1. Runtime path boundary guard: target must reside strictly inside `app_paths.root`.
2. Full read -> transform -> temp-write -> fsync -> atomic `os.replace` locking across both
   threads (in-process lock) and processes (`fcntl.flock` on POSIX / `msvcrt.locking` on Windows).
3. Mode preservation (and explicit `0o600` creation for secrets like `.env` before writing content).
4. Bounded retry on transient `PermissionError` during `os.replace` (Windows AV / indexer locks).
5. Zero hard-deletion calls (`unlink`/`remove` are never used).
"""

from __future__ import annotations

import os
import stat
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from lightbox.paths import AppPaths

_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.RLock] = {}

DEFAULT_REPLACE_RETRIES = 5
DEFAULT_REPLACE_RETRY_DELAY_S = 0.03


def _get_thread_lock(lock_key: str) -> threading.RLock:
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(lock_key)
        if lock is None:
            lock = threading.RLock()
            _THREAD_LOCKS[lock_key] = lock
        return lock


def _assert_inside_app_data_dir(app_paths: AppPaths, target: Path) -> Path:
    root_resolved = app_paths.root.expanduser().resolve()
    target_resolved = target.expanduser().resolve()
    if target_resolved == root_resolved or not target_resolved.is_relative_to(root_resolved):
        raise ValueError(
            f"Refusing atomic config write outside <APP_DATA_DIR> ({root_resolved}): {target_resolved}"
        )
    return target_resolved


@contextmanager
def _exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    """Acquire an exclusive thread + OS advisory lock for the full read-modify-replace cycle."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(lock_path.resolve())
    tlock = _get_thread_lock(key)
    with tlock:
        fh = lock_path.open("a+b")
        try:
            if sys.platform == "win32":
                import msvcrt

                fh.seek(0)
                if fh.tell() == 0:
                    fh.write(b"\0")
                    fh.flush()
                    fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            elif os.name == "posix":
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if sys.platform == "win32":
                    import msvcrt

                    fh.seek(0)
                    try:
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                elif os.name == "posix":
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def _write_temp_file(tmp_path: Path, content: str, *, mode: int | None) -> None:
    """Write `content` into `tmp_path` with desired POSIX permissions and flush to disk."""
    encoded = content.encode("utf-8")
    if os.name == "posix":
        create_mode = mode if mode is not None else 0o600
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, create_mode)
        try:
            if mode is not None:
                os.fchmod(fd, mode)
            with os.fdopen(fd, "wb") as fh:
                fh.write(encoded)
                fh.flush()
                os.fsync(fh.fileno())
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
    else:
        with tmp_path.open("wb") as fh:
            fh.write(encoded)
            fh.flush()
            os.fsync(fh.fileno())


def _replace_with_retry(
    src: Path,
    dst: Path,
    *,
    max_retries: int = DEFAULT_REPLACE_RETRIES,
    retry_delay_s: float = DEFAULT_REPLACE_RETRY_DELAY_S,
) -> None:
    attempts = max(1, max_retries)
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(retry_delay_s * (attempt + 1))


def atomic_update_text(
    app_paths: AppPaths,
    target: Path,
    transform: Callable[[str | None], str],
    *,
    mode: int | None = None,
    max_replace_retries: int = DEFAULT_REPLACE_RETRIES,
    retry_delay_s: float = DEFAULT_REPLACE_RETRY_DELAY_S,
) -> Path:
    """Atomically read, transform, and replace a text file inside `<APP_DATA_DIR>`.

    The lock covers the entire read -> transform -> write temp -> fsync -> os.replace sequence.
    If `transform` or disk write fails, the existing target file remains completely untouched.
    """
    app_paths.ensure_dirs()
    target_resolved = _assert_inside_app_data_dir(app_paths, target)
    target_resolved.parent.mkdir(parents=True, exist_ok=True)

    lock_path = app_paths.root.expanduser().resolve() / ".config_io.lock"
    with _exclusive_file_lock(lock_path):
        existing_text: str | None = None
        effective_mode = mode
        if target_resolved.is_file():
            if effective_mode is None and os.name == "posix":
                effective_mode = stat.S_IMODE(target_resolved.stat().st_mode)
            existing_text = target_resolved.read_text(encoding="utf-8")
        elif effective_mode is None and target_resolved.name == ".env":
            effective_mode = 0o600
        elif effective_mode is None and os.name == "posix":
            effective_mode = 0o644

        new_text = transform(existing_text)
        tmp_path = target_resolved.parent / f".{target_resolved.name}.tmp"
        _write_temp_file(tmp_path, new_text, mode=effective_mode)
        _replace_with_retry(
            tmp_path,
            target_resolved,
            max_retries=max_replace_retries,
            retry_delay_s=retry_delay_s,
        )
        if effective_mode is not None and os.name == "posix":
            os.chmod(target_resolved, effective_mode)
    return target_resolved


def atomic_write_text(
    app_paths: AppPaths,
    target: Path,
    content: str,
    *,
    mode: int | None = None,
    max_replace_retries: int = DEFAULT_REPLACE_RETRIES,
    retry_delay_s: float = DEFAULT_REPLACE_RETRY_DELAY_S,
) -> Path:
    """Atomically write `content` to `target` inside `<APP_DATA_DIR>`."""
    return atomic_update_text(
        app_paths,
        target,
        lambda _existing: content,
        mode=mode,
        max_replace_retries=max_replace_retries,
        retry_delay_s=retry_delay_s,
    )


__all__ = [
    "DEFAULT_REPLACE_RETRIES",
    "DEFAULT_REPLACE_RETRY_DELAY_S",
    "atomic_update_text",
    "atomic_write_text",
]
