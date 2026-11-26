"""SafeFS: no-delete filesystem primitive layer (SPEC §10.2(1)(2)(4)).

Exposes strictly 4 primitives:
- `read`
- `mkdir_quarantine`
- `rename_same_volume`
- `write_new_copy`

Enforces same-volume (`st_dev`) checks, `DCIM/` camera card & read-only volume
protection, and non-overwriting renames (`renamex_np(RENAME_EXCL)` on macOS,
check-then-rename with collision suffix elsewhere).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import stat
import sys
from pathlib import Path
from typing import Any


class SafeFSError(OSError):
    """Base error raised by SafeFS invariant violations."""


class CrossVolumeError(SafeFSError):
    """Raised when a move/quarantine operation crosses filesystem mounts (`st_dev`)."""


class ReadOnlyVolumeError(SafeFSError):
    """Raised when targeting a `DCIM/` camera card or read-only volume."""


class DestinationExistsError(SafeFSError):
    """Raised when a non-overwriting write/rename collides and suffixes are disallowed."""


_RENAME_EXCL = 0x00000004
_LIBC_RENAMEX_NP: Any = None
_LIBC_CHECKED = False


def _get_macos_renamex_np() -> Any:
    """Lazily resolve macOS `libc.renamex_np` for atomic `RENAME_EXCL` renames."""
    global _LIBC_RENAMEX_NP, _LIBC_CHECKED
    if _LIBC_CHECKED:
        return _LIBC_RENAMEX_NP
    _LIBC_CHECKED = True
    if sys.platform != "darwin":
        return None
    libc_path = ctypes.util.find_library("c")
    if not libc_path:
        return None
    libc = ctypes.CDLL(libc_path, use_errno=True)
    fn = getattr(libc, "renamex_np", None)
    if fn is not None:
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        fn.restype = ctypes.c_int
        _LIBC_RENAMEX_NP = fn
    return _LIBC_RENAMEX_NP


def is_camera_card_path(path: Path) -> bool:
    """Return True if `path` is inside or directly roots a `DCIM/` camera card structure."""
    resolved = path.expanduser().resolve()
    if any(part.upper() == "DCIM" for part in resolved.parts):
        return True
    if resolved.is_dir():
        try:
            for child in resolved.iterdir():
                if child.name.upper() == "DCIM" and child.is_dir():
                    return True
        except OSError:
            pass
    return False


def _nearest_existing_ancestor(path: Path) -> Path:
    current = path.expanduser()
    while not current.exists():
        parent = current.parent
        if parent == current:
            break
        current = parent
    return current


def is_readonly_path(path: Path) -> bool:
    """Return True if the filesystem or directory containing `path` is read-only."""
    target = _nearest_existing_ancestor(path)
    if hasattr(os, "statvfs"):
        try:
            vfs = os.statvfs(target)
            st_rdonly = getattr(os, "ST_RDONLY", 1)
            if bool(vfs.f_flag & st_rdonly):
                return True
        except OSError:
            pass
    if target.is_dir():
        if not os.access(target, os.W_OK):
            return True
    elif target.exists():
        mode = target.stat().st_mode
        if not bool(mode & stat.S_IWUSR):
            return True
        if not os.access(target.parent, os.W_OK):
            return True
    return False


def find_mount_root(path: Path) -> Path:
    """Walk upward from `path` to locate the filesystem mount root."""
    current = _nearest_existing_ancestor(path).resolve()
    dev = current.stat().st_dev
    while True:
        parent = current.parent
        if parent == current:
            return current
        try:
            if parent.stat().st_dev != dev:
                return current
        except OSError:
            return current
        current = parent


def _with_collision_suffix(dst: Path, index: int) -> Path:
    return dst.with_name(f"{dst.stem}__{index}{dst.suffix}")


def _rename_excl(src: Path, dst: Path) -> None:
    """Atomically rename `src` -> `dst` without overwriting an existing `dst`."""
    renamex = _get_macos_renamex_np()
    if renamex is not None:
        ctypes.set_errno(0)
        rc = int(renamex(os.fsencode(src), os.fsencode(dst), _RENAME_EXCL))
        if rc == 0:
            return
        err = ctypes.get_errno()
        if err == errno.EEXIST:
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(dst))
        raise OSError(err, os.strerror(err), str(src), None, str(dst))

    # Non-macOS fallback: check destination absence before rename inside exclusive dir.
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(dst))
    os.rename(src, dst)


class SafeFS:
    """Restricted filesystem facade enforcing Lightbox safety invariants (SPEC §10.2)."""

    def read(self, path: Path) -> bytes:
        """Read and return raw bytes from `path`."""
        return Path(path).read_bytes()

    def mkdir_quarantine(self, directory: Path) -> Path:
        """Create a quarantine directory after verifying it is writable and not a camera card."""
        target = Path(directory).expanduser().resolve()
        if is_camera_card_path(target):
            raise ReadOnlyVolumeError(f"Refusing to create quarantine inside DCIM path: {target}")
        if is_readonly_path(target):
            raise ReadOnlyVolumeError(f"Target quarantine parent is read-only: {target}")
        target.mkdir(parents=True, exist_ok=True)
        return target

    def rename_same_volume(
        self,
        src: Path,
        dst: Path,
        *,
        allow_suffix: bool = True,
    ) -> Path:
        """Move `src` to `dst` on the same volume without ever overwriting an existing file."""
        src_path = Path(src).expanduser().resolve()
        dst_path = Path(dst).expanduser().resolve()

        if not src_path.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(src_path))
        if is_camera_card_path(src_path) or is_camera_card_path(dst_path):
            raise ReadOnlyVolumeError(
                f"Refusing to move files on DCIM camera card: {src_path} -> {dst_path}"
            )
        if is_readonly_path(src_path.parent) or is_readonly_path(dst_path.parent):
            raise ReadOnlyVolumeError(
                f"Refusing to move files on read-only volume: {src_path} -> {dst_path}"
            )

        src_dev = os.stat(src_path).st_dev
        dst_anchor = _nearest_existing_ancestor(dst_path.parent)
        dst_dev = os.stat(dst_anchor).st_dev
        if src_dev != dst_dev:
            raise CrossVolumeError(
                f"Cross-volume move forbidden (st_dev {src_dev} != {dst_dev}): "
                f"{src_path} -> {dst_path}"
            )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if os.stat(dst_path.parent).st_dev != src_dev:
            raise CrossVolumeError(
                f"Cross-volume move forbidden after mkdir: {src_path} -> {dst_path}"
            )

        candidate = dst_path
        suffix_idx = 1
        while True:
            try:
                _rename_excl(src_path, candidate)
                return candidate
            except FileExistsError:
                if not allow_suffix:
                    raise DestinationExistsError(
                        f"Refusing to overwrite existing file: {candidate}"
                    ) from None
                candidate = _with_collision_suffix(dst_path, suffix_idx)
                suffix_idx += 1

    def write_new_copy(
        self,
        dst: Path,
        data: bytes,
        *,
        allow_suffix: bool = False,
    ) -> Path:
        """Create a brand-new file at `dst` using `O_CREAT | O_EXCL` (never overwrites)."""
        dst_path = Path(dst).expanduser().resolve()
        if is_camera_card_path(dst_path):
            raise ReadOnlyVolumeError(f"Refusing to write inside DCIM path: {dst_path}")
        if is_readonly_path(dst_path.parent):
            raise ReadOnlyVolumeError(f"Destination parent is read-only: {dst_path.parent}")

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY

        candidate = dst_path
        suffix_idx = 1
        while True:
            try:
                fd = os.open(candidate, flags, 0o644)
            except FileExistsError:
                if not allow_suffix:
                    raise DestinationExistsError(
                        f"Refusing to overwrite existing file: {candidate}"
                    ) from None
                candidate = _with_collision_suffix(dst_path, suffix_idx)
                suffix_idx += 1
                continue

            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            return candidate


__all__ = [
    "CrossVolumeError",
    "DestinationExistsError",
    "ReadOnlyVolumeError",
    "SafeFS",
    "SafeFSError",
    "find_mount_root",
    "is_camera_card_path",
    "is_readonly_path",
]
