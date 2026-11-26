"""Immutable execution plans with plan_hash optimistic locking, WAL progress, and restore (SPEC §3.3, §10.2)."""

from __future__ import annotations

import hashlib
import json
import struct
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lightbox.fs.quarantine import (
    MANIFEST_FILENAME,
    RESTORE_README_FILENAME,
    build_restore_readme,
    resolve_quarantine_dir,
    write_quarantine_artifacts,
)
from lightbox.fs.safe_fs import SafeFS
from lightbox.store.repos import LightboxRepository, PlanItemRecord, PlanRecord

_MUTABLE_STALE_STATUSES: frozenset[str] = frozenset({"draft", "reviewed", "approved"})
_FP_CHUNK_SIZE = 64 * 1024


class PlanError(RuntimeError):
    """Base error for execution plan operations."""


class StalePlanError(PlanError):
    """Raised when a photo's `decision_version` or protection status changed after drafting."""


class PlanStateError(PlanError):
    """Raised when attempting an operation not allowed in the plan's current state."""


@dataclass(frozen=True, slots=True)
class ApplyPlanResult:
    plan_id: str
    plan_hash: str
    status: str
    quarantine_dir: Path
    moved_photo_count: int
    moved_file_count: int
    manifest_path: Path
    readme_path: Path


@dataclass(frozen=True, slots=True)
class RestorePlanResult:
    plan_id: str
    restored_photo_ids: tuple[str, ...]
    restored_file_count: int
    plan_status: str


def compute_plan_hash(entries: Sequence[tuple[str, int, Sequence[str]]]) -> str:
    """Compute canonical `sha256(sorted([(photo_id, decision_version, dst_paths)]))` (SPEC §3.3)."""
    canonical = sorted(
        [
            [photo_id, int(decision_version), list(dst_paths)]
            for photo_id, decision_version, dst_paths in entries
        ],
        key=lambda row: str(row[0]),
    )
    payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(safe_fs: SafeFS, path: Path) -> str:
    return hashlib.sha256(safe_fs.read(path)).hexdigest()


def _fast_content_fp_and_size(path: Path) -> tuple[str | None, int | None]:
    """Compute lightweight `(size + head_64KB + tail_64KB)` fingerprint without reading whole RAWs."""
    try:
        size = path.stat().st_size
        hasher = hashlib.sha256()
        hasher.update(struct.pack(">Q", size))
        with path.open("rb") as fh:
            if size <= _FP_CHUNK_SIZE * 2:
                hasher.update(fh.read())
            else:
                hasher.update(fh.read(_FP_CHUNK_SIZE))
                fh.seek(size - _FP_CHUNK_SIZE)
                hasher.update(fh.read(_FP_CHUNK_SIZE))
        return hasher.hexdigest(), size
    except OSError:
        return None, None


class PlanExecutor:
    """Draft, verify, execute (with WAL crash-recovery), and restore quarantine plans."""

    def __init__(self, repo: LightboxRepository, safe_fs: SafeFS | None = None) -> None:
        self.repo = repo
        self.safe_fs = safe_fs or SafeFS()

    def create_quarantine_plan(
        self,
        session_id: str,
        photo_ids: Sequence[str],
        *,
        strategy: str = "outside_library",
        custom_dir: Path | None = None,
        user_home: Path | None = None,
        plan_id: str | None = None,
    ) -> tuple[PlanRecord, list[PlanItemRecord]]:
        """Draft an immutable quarantine plan binding all atomic asset pairs and decision versions."""
        session = self.repo.get_session(session_id)
        if session is None:
            raise PlanError(f"Unknown session_id: {session_id}")

        unique_ids = sorted(set(photo_ids))
        if not unique_ids:
            raise PlanError("Cannot create an empty quarantine plan.")

        pid = plan_id or f"plan_{uuid.uuid4().hex[:12]}"
        short_id = pid[-8:]
        session_root = Path(session.root_path).expanduser().resolve()
        quarantine_dir = resolve_quarantine_dir(
            session_root=session_root,
            session_name=session.name,
            short_id=short_id,
            strategy=strategy,
            custom_dir=custom_dir,
            user_home=user_home,
        )

        items: list[PlanItemRecord] = []
        hash_entries: list[tuple[str, int, list[str]]] = []

        for photo_id in unique_ids:
            photo = self.repo.get_photo(photo_id)
            if photo is None:
                raise PlanError(f"Unknown photo_id: {photo_id}")
            if photo.session_id != session_id:
                raise PlanError(f"Photo {photo_id} does not belong to session {session_id}")
            if photo.is_protected:
                raise PlanError(f"Refusing to include protected/exempt photo in plan: {photo_id}")

            assets = self.repo.list_assets_for_photo(photo_id)
            if not assets:
                raise PlanError(f"Photo {photo_id} has no physical assets recorded.")

            src_paths: list[str] = []
            dst_paths: list[str] = []
            for asset in assets:
                src = Path(asset.abs_path).expanduser().resolve()
                rel = Path(asset.rel_path)
                if rel.is_absolute() or ".." in rel.parts:
                    raise PlanError(f"Invalid asset relative path: {asset.rel_path}")
                dst = (quarantine_dir / rel).resolve()
                src_paths.append(str(src))
                dst_paths.append(str(dst))

            items.append(
                PlanItemRecord(
                    plan_id=pid,
                    photo_id=photo_id,
                    decision_version=photo.decision_version,
                    src_paths=tuple(src_paths),
                    dst_paths=tuple(dst_paths),
                    status="pending",
                )
            )
            hash_entries.append((photo_id, photo.decision_version, dst_paths))

        plan_hash = compute_plan_hash(hash_entries)
        plan = PlanRecord(
            id=pid,
            session_id=session_id,
            action_type="quarantine_move",
            plan_hash=plan_hash,
            status="draft",
            quarantine_dir=str(quarantine_dir),
        )
        self.repo.create_plan(plan, items)
        return plan, items

    def _mark_stale_if_mutable(self, plan: PlanRecord) -> None:
        """Only transition mutable pre-execution plans (`draft`/`reviewed`/`approved`) to `stale`."""
        if plan.status in _MUTABLE_STALE_STATUSES:
            self.repo.update_plan_status(plan.id, "stale")

    def verify_plan_freshness(
        self,
        plan_id: str,
        *,
        check_hash: bool = True,
    ) -> tuple[PlanRecord, list[PlanItemRecord]]:
        """Validate optimistic locks (`decision_version`, `is_protected`, `plan_hash`).

        Only marks the plan `stale` when it is still in `draft`/`reviewed`/`approved` status;
        already-applied or in-flight plans are never overwritten to `stale`.
        """
        plan = self.repo.get_plan(plan_id)
        if plan is None:
            raise PlanError(f"Unknown plan_id: {plan_id}")
        items = self.repo.list_plan_items(plan_id)
        if not items:
            raise PlanError(f"Plan {plan_id} contains no items.")

        for item in items:
            photo = self.repo.get_photo(item.photo_id)
            if photo is None or photo.decision_version != item.decision_version:
                self._mark_stale_if_mutable(plan)
                cur_ver = photo.decision_version if photo is not None else None
                raise StalePlanError(
                    f"Plan {plan_id} is stale: photo {item.photo_id} decision_version "
                    f"changed ({item.decision_version} -> {cur_ver})."
                )
            if photo.is_protected:
                self._mark_stale_if_mutable(plan)
                raise StalePlanError(
                    f"Plan {plan_id} is stale: photo {item.photo_id} is now protected/exempt."
                )

        if check_hash:
            recomputed = compute_plan_hash(
                [(item.photo_id, item.decision_version, item.dst_paths) for item in items]
            )
            if recomputed != plan.plan_hash:
                self._mark_stale_if_mutable(plan)
                raise StalePlanError(
                    f"Plan {plan_id} hash mismatch ({recomputed} != {plan.plan_hash})."
                )

        return plan, items

    def approve_plan(self, plan_id: str, *, approved_by: str = "user") -> PlanRecord:
        """Mark a fresh `draft`/`reviewed` plan as approved by a human actor."""
        plan = self.repo.get_plan(plan_id)
        if plan is None:
            raise PlanError(f"Unknown plan_id: {plan_id}")
        if plan.status not in {"draft", "reviewed"}:
            raise PlanStateError(f"Cannot approve plan {plan_id} in status {plan.status!r}.")

        self.verify_plan_freshness(plan_id, check_hash=True)
        now = time.time()
        self.repo.update_plan_status(
            plan_id,
            "approved",
            approved_by=approved_by,
            approved_at=now,
        )
        updated = self.repo.get_plan(plan_id)
        assert updated is not None
        return updated

    def _asset_meta_lookup(self, photo_id: str) -> dict[str, tuple[str, int]]:
        """Map resolved `abs_path` -> `(content_fp, size_bytes)` from DB assets table."""
        lookup: dict[str, tuple[str, int]] = {}
        for asset in self.repo.list_assets_for_photo(photo_id):
            key = str(Path(asset.abs_path).expanduser().resolve())
            lookup[key] = (asset.content_fp, asset.size_bytes)
        return lookup

    def _build_manifest_item(
        self,
        item: PlanItemRecord,
        *,
        dst_paths: Sequence[str],
        asset_meta: dict[str, tuple[str, int]],
        compute_sha256: bool,
    ) -> dict[str, Any]:
        files_payload: list[dict[str, Any]] = []
        for src_s, dst_s in zip(item.src_paths, dst_paths, strict=True):
            meta = asset_meta.get(src_s)
            if meta is not None:
                content_fp: str | None = meta[0]
                size_bytes: int | None = meta[1]
            else:
                probe_path = Path(dst_s) if Path(dst_s).exists() else Path(src_s)
                content_fp, size_bytes = _fast_content_fp_and_size(probe_path)

            entry: dict[str, Any] = {
                "src_path": src_s,
                "dst_path": dst_s,
                "content_fp": content_fp,
                "size_bytes": size_bytes,
            }
            if compute_sha256 and Path(dst_s).exists():
                entry["sha256"] = _sha256_file(self.safe_fs, Path(dst_s))
            files_payload.append(entry)

        return {
            "photo_id": item.photo_id,
            "decision_version": item.decision_version,
            "files": files_payload,
        }

    def apply_plan(
        self,
        plan_id: str,
        *,
        resume: bool = False,
        compute_sha256: bool = False,
    ) -> ApplyPlanResult:
        """Execute an approved quarantine plan with pre-written manifest, WAL commit, and crash recovery.

        By default (`compute_sha256=False`), `manifest.json` records lightweight
        `content_fp` + `size_bytes` without full-file SHA-256 reads so batch-moving
        thousands of RAW files completes in seconds.
        """
        plan = self.repo.get_plan(plan_id)
        if plan is None:
            raise PlanError(f"Unknown plan_id: {plan_id}")
        if plan.status == "stale":
            raise StalePlanError(f"Plan {plan_id} is stale and cannot be applied.")

        allowed_statuses = {"approved", "applying"} if resume else {"approved"}
        if plan.status not in allowed_statuses:
            raise PlanStateError(
                f"Plan {plan_id} must be approved before applying (current status={plan.status!r})."
            )

        # Verify optimistic locks & protection flags before touching disk.
        # On resume, skip plan_hash re-check because a prior partial run may have
        # recorded a collision-suffixed dst_path in `plan_items`.
        plan, items = self.verify_plan_freshness(plan_id, check_hash=(not resume))
        if plan.quarantine_dir is None:
            raise PlanError(f"Plan {plan_id} has no quarantine_dir configured.")

        session = self.repo.get_session(plan.session_id)
        if session is None:
            raise PlanError(f"Session {plan.session_id} not found for plan {plan_id}.")

        quarantine_dir = self.safe_fs.mkdir_quarantine(Path(plan.quarantine_dir))
        if plan.status != "applying":
            self.repo.update_plan_status(plan_id, "applying", quarantine_dir=str(quarantine_dir))

        asset_meta_by_photo = {
            item.photo_id: self._asset_meta_lookup(item.photo_id) for item in items
        }

        # Pre-write expected `manifest.json` and bilingual `RESTORE_README.txt` BEFORE moving
        # the first file so even an immediate crash leaves full manual restore metadata on disk.
        readme_text = build_restore_readme(
            plan_id=plan.id,
            plan_hash=plan.plan_hash,
            session_root=Path(session.root_path),
            quarantine_dir=quarantine_dir,
            item_count=len(items),
        )
        manifest_name = MANIFEST_FILENAME
        expected_manifest_path = quarantine_dir / manifest_name
        if not resume and expected_manifest_path.exists():
            # Shared subfolder (e.g. `strategy="subfolder"`): keep each batch's manifest distinct
            manifest_name = f"manifest_{plan.id}.json"
            expected_manifest_path = quarantine_dir / manifest_name
        readme_path = quarantine_dir / RESTORE_README_FILENAME
        if not expected_manifest_path.exists():
            expected_items = [
                self._build_manifest_item(
                    item,
                    dst_paths=item.dst_paths,
                    asset_meta=asset_meta_by_photo[item.photo_id],
                    compute_sha256=False,
                )
                for item in items
            ]
            expected_payload: dict[str, Any] = {
                "schema_version": 2,
                "plan_id": plan.id,
                "plan_hash": plan.plan_hash,
                "session_id": plan.session_id,
                "session_root": session.root_path,
                "quarantine_dir": str(quarantine_dir),
                "created_at": time.time(),
                "items": expected_items,
            }
            expected_manifest_path, readme_path = write_quarantine_artifacts(
                self.safe_fs,
                quarantine_dir,
                expected_payload,
                readme_text,
                manifest_filename=manifest_name,
                write_readme=True,
            )
        elif not readme_path.exists():
            readme_path = self.safe_fs.write_new_copy(
                readme_path,
                readme_text.encode("utf-8"),
                allow_suffix=False,
            )

        manifest_items: list[dict[str, Any]] = []
        moved_files_total = 0
        paths_diverged = False

        for item in items:
            if item.status == "done":
                manifest_items.append(
                    self._build_manifest_item(
                        item,
                        dst_paths=item.dst_paths,
                        asset_meta=asset_meta_by_photo[item.photo_id],
                        compute_sha256=compute_sha256,
                    )
                )
                moved_files_total += len(item.dst_paths)
                continue

            # Re-verify protection immediately prior to moving this atomic group
            latest_photo = self.repo.get_photo(item.photo_id)
            if latest_photo is None or latest_photo.is_protected:
                raise StalePlanError(
                    f"Photo {item.photo_id} became protected/missing before move; aborting."
                )

            actual_dsts = list(item.dst_paths)
            try:
                for idx, (src_s, dst_s) in enumerate(
                    zip(item.src_paths, item.dst_paths, strict=True)
                ):
                    src_p = Path(src_s)
                    dst_p = Path(dst_s)
                    if not src_p.exists() and dst_p.exists():
                        # Crash-recovery: file already moved before interruption
                        final_dst = dst_p
                    else:
                        final_dst = self.safe_fs.rename_same_volume(
                            src_p, dst_p, allow_suffix=True
                        )
                    actual_dsts[idx] = str(final_dst)
                    if str(final_dst) != dst_s:
                        paths_diverged = True
            except Exception:
                # Persist any collision-suffixed dst paths already moved before raising
                if tuple(actual_dsts) != item.dst_paths:
                    self.repo.update_plan_item(
                        plan_id,
                        item.photo_id,
                        status="pending",
                        dst_paths=actual_dsts,
                    )
                raise

            # Short WAL transaction marking this atomic photo group done
            self.repo.update_plan_item(
                plan_id,
                item.photo_id,
                status="done",
                dst_paths=actual_dsts,
                done_at=time.time(),
            )
            moved_files_total += len(actual_dsts)
            manifest_items.append(
                self._build_manifest_item(
                    item,
                    dst_paths=actual_dsts,
                    asset_meta=asset_meta_by_photo[item.photo_id],
                    compute_sha256=compute_sha256,
                )
            )

        # Only emit a supplemental final manifest if collision suffixing changed a dst path
        # or full SHA-256 verification was explicitly requested. Normal runs & resumes reuse
        # the pre-written `manifest.json` without spawning `manifest__1.json` duplicates.
        manifest_path = expected_manifest_path
        if paths_diverged or compute_sha256:
            final_payload: dict[str, Any] = {
                "schema_version": 2,
                "plan_id": plan.id,
                "plan_hash": plan.plan_hash,
                "session_id": plan.session_id,
                "session_root": session.root_path,
                "quarantine_dir": str(quarantine_dir),
                "applied_at": time.time(),
                "items": manifest_items,
            }
            manifest_path, _ = write_quarantine_artifacts(
                self.safe_fs,
                quarantine_dir,
                final_payload,
                readme_text,
                manifest_filename="manifest_final.json",
                write_readme=False,
            )

        self.repo.update_plan_status(plan_id, "applied")

        return ApplyPlanResult(
            plan_id=plan.id,
            plan_hash=plan.plan_hash,
            status="applied",
            quarantine_dir=quarantine_dir,
            moved_photo_count=len(items),
            moved_file_count=moved_files_total,
            manifest_path=manifest_path,
            readme_path=readme_path,
        )

    def restore_plan(
        self,
        plan_id: str,
        *,
        photo_ids: Sequence[str] | None = None,
    ) -> RestorePlanResult:
        """Restore an entire batch (or selected `photo_ids`) from quarantine back to original paths."""
        plan = self.repo.get_plan(plan_id)
        if plan is None:
            raise PlanError(f"Unknown plan_id: {plan_id}")
        if plan.status not in {"applied", "applying"}:
            raise PlanStateError(
                f"Cannot restore plan {plan_id} in status {plan.status!r}."
            )

        items = self.repo.list_plan_items(plan_id)
        target_set = set(photo_ids) if photo_ids is not None else None

        restored_photos: list[str] = []
        restored_files = 0

        for item in items:
            if target_set is not None and item.photo_id not in target_set:
                continue
            if item.status == "reverted":
                continue

            for src_s, dst_s in zip(item.src_paths, item.dst_paths, strict=True):
                src_p = Path(src_s)
                dst_p = Path(dst_s)
                if dst_p.exists():
                    # Non-overwriting restore: exact original path when free, or collision-safe
                    # suffix if an identically named file already exists in the original folder.
                    self.safe_fs.rename_same_volume(dst_p, src_p, allow_suffix=True)
                    restored_files += 1
                elif src_p.exists():
                    # Already back in original folder (e.g. tail of interrupted plan)
                    continue
                else:
                    raise PlanError(
                        f"Missing file during restore for photo {item.photo_id}: "
                        f"neither {dst_p} nor {src_p} exists (it may have been removed in Finder)."
                    )

            self.repo.update_plan_item(
                plan_id,
                item.photo_id,
                status="reverted",
                done_at=time.time(),
            )
            restored_photos.append(item.photo_id)

        updated_items = self.repo.list_plan_items(plan_id)
        all_reverted = all(item.status == "reverted" for item in updated_items)
        new_status = "reverted" if all_reverted else plan.status
        if all_reverted:
            self.repo.update_plan_status(plan_id, "reverted")

        return RestorePlanResult(
            plan_id=plan_id,
            restored_photo_ids=tuple(restored_photos),
            restored_file_count=restored_files,
            plan_status=new_status,
        )


__all__ = [
    "ApplyPlanResult",
    "PlanError",
    "PlanExecutor",
    "PlanStateError",
    "RestorePlanResult",
    "StalePlanError",
    "compute_plan_hash",
]
