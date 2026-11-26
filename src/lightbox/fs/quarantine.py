"""Same-volume out-of-library quarantine directory resolution & restore artifacts (SPEC §3.1, §10.2)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from lightbox.fs.safe_fs import (
    CrossVolumeError,
    ReadOnlyVolumeError,
    SafeFS,
    find_mount_root,
    is_camera_card_path,
    is_readonly_path,
)

QUARANTINE_DIR_NAME = "Lightbox_Quarantine"
SUBFOLDER_TRASH_DIR_NAME = "待删除废片"
MANIFEST_FILENAME = "manifest.json"
RESTORE_README_FILENAME = "RESTORE_README.txt"

_SAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_slug(name: str) -> str:
    cleaned = _SAFE_SLUG_RE.sub("_", name.strip()).strip("._-")
    return cleaned or "session"


def _is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def resolve_quarantine_dir(
    session_root: Path,
    session_name: str,
    short_id: str,
    *,
    strategy: str = "outside_library",
    custom_dir: Path | None = None,
    user_home: Path | None = None,
) -> Path:
    """Resolve a same-volume, out-of-library quarantine directory path without creating it."""
    src_root = Path(session_root).expanduser().resolve()
    if not src_root.exists():
        raise FileNotFoundError(f"Session root does not exist: {src_root}")
    if is_camera_card_path(src_root):
        raise ReadOnlyVolumeError(f"Camera card / DCIM root is read-only: {src_root}")
    if is_readonly_path(src_root):
        raise ReadOnlyVolumeError(f"Session root volume is read-only: {src_root}")

    src_dev = os.stat(src_root).st_dev
    folder_name = f"{_sanitize_slug(session_name)}_{_sanitize_slug(short_id)}"

    if strategy == "subfolder":
        return src_root / SUBFOLDER_TRASH_DIR_NAME

    if strategy == "outside_library":
        home = (user_home or Path.home()).expanduser().resolve()
        if home.exists() and os.stat(home).st_dev == src_dev:
            base = home / QUARANTINE_DIR_NAME
        else:
            mount_root = find_mount_root(src_root)
            if is_readonly_path(mount_root):
                raise ReadOnlyVolumeError(
                    f"External volume root is not writable for quarantine: {mount_root}"
                )
            base = mount_root / QUARANTINE_DIR_NAME
    elif strategy == "sibling":
        base = src_root.parent / QUARANTINE_DIR_NAME
    elif strategy == "custom":
        if custom_dir is None:
            raise ValueError("`custom_dir` must be provided when strategy='custom'.")
        base = Path(custom_dir).expanduser().resolve()
    else:
        raise ValueError(f"Unknown quarantine strategy: {strategy!r}")

    if _is_subpath(base, src_root):
        raise ValueError(
            f"Quarantine directory must be outside the photo library root: {base} inside {src_root}"
        )

    anchor = base
    while not anchor.exists() and anchor.parent != anchor:
        anchor = anchor.parent
    if os.stat(anchor).st_dev != src_dev:
        raise CrossVolumeError(
            f"Quarantine base {base} is on a different volume than {src_root}."
        )
    if is_readonly_path(anchor):
        raise ReadOnlyVolumeError(f"Quarantine base directory is read-only: {base}")

    return base / folder_name


def build_restore_readme(
    *,
    plan_id: str,
    plan_hash: str,
    session_root: Path,
    quarantine_dir: Path,
    item_count: int,
) -> str:
    """Build bilingual (zh-CN + en-US) manual & CLI restore instructions (SPEC §10.2(2))."""
    return (
        "========================================================================\n"
        "Lightbox 隔离区还原说明 / Lightbox Quarantine Restore Guide\n"
        "========================================================================\n"
        f"Plan ID        : {plan_id}\n"
        f"Plan Hash      : {plan_hash}\n"
        f"Session Root   : {session_root}\n"
        f"Quarantine Dir : {quarantine_dir}\n"
        f"Photo Groups   : {item_count}\n"
        "\n"
        "[中文说明]\n"
        "本目录存放由 Lightbox 安全隔离移出的照片原文件（Lightbox 永不硬删原片）。\n"
        "1. 推荐通过 Lightbox Web UI（/insights 页面）或 MCP 工具 `lightbox_restore_plan` 一键整批或单张还原。\n"
        "2. 即使脱离 Lightbox 软件，您也可以查看同目录下的 `manifest.json`，其中完整记录了每个文件\n"
        "   的原始路径（`src_path`）、预期/当前隔离路径（`dst_path`）、文件大小（`size_bytes`）与\n"
        "   内容指纹（`content_fp`），按原始相对路径将文件移回原拍摄目录即可 100% 恢复。\n"
        "\n"
        "[English Guide]\n"
        "This directory holds original photo files safely quarantined by Lightbox\n"
        "(Lightbox never hard-deletes original photos).\n"
        "1. Restore one-click via the Lightbox Web UI (/insights) or the MCP tool\n"
        "   `lightbox_restore_plan` (full batch or individual photos).\n"
        "2. Even without Lightbox installed, inspect `manifest.json` in this folder:\n"
        "   every moved file's original `src_path`, quarantine `dst_path`, `size_bytes`,\n"
        "   and `content_fp` are recorded so you can move files back manually with zero loss.\n"
    )


def write_quarantine_artifacts(
    safe_fs: SafeFS,
    quarantine_dir: Path,
    manifest_payload: dict[str, Any],
    readme_text: str,
    *,
    manifest_filename: str = MANIFEST_FILENAME,
    write_readme: bool = True,
) -> tuple[Path, Path]:
    """Write manifest JSON and optional bilingual `RESTORE_README.txt` via SafeFS (no overwrite)."""
    manifest_bytes = (
        json.dumps(manifest_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    target_manifest = quarantine_dir / manifest_filename
    if target_manifest.exists():
        manifest_path = safe_fs.write_new_copy(
            target_manifest,
            manifest_bytes,
            allow_suffix=True,
        )
    else:
        manifest_path = safe_fs.write_new_copy(
            target_manifest,
            manifest_bytes,
            allow_suffix=False,
        )

    readme_target = quarantine_dir / RESTORE_README_FILENAME
    if write_readme and not readme_target.exists():
        readme_path = safe_fs.write_new_copy(
            readme_target,
            readme_text.encode("utf-8"),
            allow_suffix=False,
        )
    else:
        readme_path = readme_target
    return manifest_path, readme_path


__all__ = [
    "MANIFEST_FILENAME",
    "QUARANTINE_DIR_NAME",
    "RESTORE_README_FILENAME",
    "build_restore_readme",
    "resolve_quarantine_dir",
    "write_quarantine_artifacts",
]
