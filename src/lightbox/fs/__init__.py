"""Lightbox safe filesystem, quarantine resolution, and immutable execution plans (SPEC §10.2)."""

from __future__ import annotations

from lightbox.fs.plans import (
    ApplyPlanResult,
    PlanError,
    PlanExecutor,
    PlanStateError,
    RestorePlanResult,
    StalePlanError,
    compute_plan_hash,
)
from lightbox.fs.quarantine import (
    MANIFEST_FILENAME,
    QUARANTINE_DIR_NAME,
    RESTORE_README_FILENAME,
    build_restore_readme,
    resolve_quarantine_dir,
    write_quarantine_artifacts,
)
from lightbox.fs.safe_fs import (
    CrossVolumeError,
    DestinationExistsError,
    ReadOnlyVolumeError,
    SafeFS,
    SafeFSError,
    find_mount_root,
    is_camera_card_path,
    is_readonly_path,
)

__all__ = [
    "ApplyPlanResult",
    "CrossVolumeError",
    "DestinationExistsError",
    "MANIFEST_FILENAME",
    "PlanError",
    "PlanExecutor",
    "PlanStateError",
    "QUARANTINE_DIR_NAME",
    "RESTORE_README_FILENAME",
    "ReadOnlyVolumeError",
    "RestorePlanResult",
    "SafeFS",
    "SafeFSError",
    "StalePlanError",
    "build_restore_readme",
    "compute_plan_hash",
    "find_mount_root",
    "is_camera_card_path",
    "is_readonly_path",
    "resolve_quarantine_dir",
    "write_quarantine_artifacts",
]
