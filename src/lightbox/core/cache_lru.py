"""Whitelisted LRU eviction strictly scoped to `<APP_DATA_DIR>/cache/` (SPEC §4.1, §10.2(1)).

This is the ONLY module whitelisted by `scripts/check_no_delete.py` to call
`.unlink()`. Every unlink is guarded at runtime by `is_relative_to(cache_root)`
and symlink checks so user original photos can never be touched.
"""

from __future__ import annotations

from pathlib import Path


def enforce_cache_limit(cache_dir: Path, max_bytes: int) -> list[Path]:
    """Evict least-recently-used files inside `cache_dir` until total size <= `max_bytes`."""
    if max_bytes < 0:
        raise ValueError(f"max_bytes must be >= 0, got {max_bytes}")

    raw_dir = Path(cache_dir).expanduser()
    if raw_dir.is_symlink():
        raise ValueError(f"Refusing to run cache LRU on symlinked directory: {raw_dir}")
    if not raw_dir.exists():
        return []

    cache_root = raw_dir.resolve()
    if cache_root == cache_root.parent:
        raise ValueError(f"Refusing to run cache LRU on filesystem root: {cache_root}")

    entries: list[tuple[float, float, int, Path]] = []
    total_bytes = 0

    for item in cache_root.rglob("*"):
        if item.is_symlink() or not item.is_file():
            continue
        resolved = item.resolve()
        if not resolved.is_relative_to(cache_root):
            continue
        try:
            st = resolved.stat()
        except OSError:
            continue
        total_bytes += st.st_size
        entries.append((st.st_atime, st.st_mtime, st.st_size, resolved))

    if total_bytes <= max_bytes:
        return []

    # Sort oldest accessed/modified first (LRU)
    entries.sort(key=lambda row: (row[0], row[1], str(row[3])))

    evicted: list[Path] = []
    for _atime, _mtime, size_bytes, candidate in entries:
        if total_bytes <= max_bytes:
            break
        # Mandatory runtime containment guard before deleting cached artifact
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(cache_root):
            raise ValueError(f"Refusing to delete path outside cache root: {candidate}")
        try:
            candidate.unlink()
        except FileNotFoundError:
            continue
        total_bytes -= size_bytes
        evicted.append(candidate)

    return evicted


__all__ = ["enforce_cache_limit"]
