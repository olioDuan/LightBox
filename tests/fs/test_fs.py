"""T03 unit tests: SafeFS invariants, atomic quarantine/restore roundtrip, optimistic lock, and crash recovery."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from lightbox.fs import (
    MANIFEST_FILENAME,
    RESTORE_README_FILENAME,
    CrossVolumeError,
    DestinationExistsError,
    PlanExecutor,
    PlanStateError,
    ReadOnlyVolumeError,
    SafeFS,
    StalePlanError,
)
from lightbox.store import (
    AssetRecord,
    Database,
    LightboxRepository,
    PhotoRecord,
    RootRecord,
    SessionRecord,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_atomic_shoot(
    tmp_path: Path,
) -> tuple[Database, LightboxRepository, Path, Path, dict[str, str]]:
    """Create a realistic shoot with RAW+JPG+LivePhoto MOV+XMP across subdirs and register in DB."""
    shoot_dir = tmp_path / "my_shoot"
    home_dir = tmp_path / "fake_home"
    home_dir.mkdir()
    (shoot_dir / "day1").mkdir(parents=True)
    (shoot_dir / "day2").mkdir(parents=True)

    files_spec = {
        "p1": [
            ("day1/IMG_0001.ARW", "raw", b"RAW_DATA_0001" * 100),
            ("day1/IMG_0001.JPG", "jpg", b"JPG_DATA_0001" * 50),
            ("day1/IMG_0001.MOV", "live_mov", b"MOV_LIVE_0001" * 80),
            ("day1/IMG_0001.ARW.xmp", "sidecar", b"<x:xmpmeta>0001</x:xmpmeta>"),
        ],
        "p2": [
            ("day2/IMG_0002.CR3", "raw", b"RAW_DATA_0002" * 100),
            ("day2/IMG_0002.JPG", "jpg", b"JPG_DATA_0002" * 50),
            ("day2/IMG_0002.xmp", "sidecar", b"<x:xmpmeta>0002</x:xmpmeta>"),
        ],
    }

    hashes: dict[str, str] = {}
    db = Database(tmp_path / "lightbox.db")
    db.migrate()
    repo = LightboxRepository(db)

    dev_id = str(os.stat(shoot_dir).st_dev)
    repo.upsert_root(RootRecord(id="r1", path=str(shoot_dir), device_id=dev_id, kind="disk"))
    repo.upsert_session(
        SessionRecord(
            id="s1",
            root_id="r1",
            root_path=str(shoot_dir),
            name="Kyoto Shoot",
            sensitivity="conservative",
            engine_version="0.1.0",
            created_at=time.time(),
        )
    )

    for photo_id, entries in files_spec.items():
        first_rel = entries[0][0]
        rel_dir = str(Path(first_rel).parent)
        stem = Path(first_rel).stem
        repo.upsert_photo(
            PhotoRecord(
                id=photo_id,
                session_id="s1",
                stem=stem,
                rel_dir=rel_dir,
                primary_content_fp=f"fp_{photo_id}",
                cohort="camera",
            )
        )
        for rel_path, kind, content in entries:
            fpath = shoot_dir / rel_path
            fpath.write_bytes(content)
            hashes[str(fpath.resolve())] = _sha256(fpath)
            st = fpath.stat()
            repo.upsert_asset(
                AssetRecord(
                    photo_id=photo_id,
                    rel_path=rel_path,
                    abs_path=str(fpath.resolve()),
                    kind=kind,
                    size_bytes=st.st_size,
                    mtime=st.st_mtime,
                    content_fp=f"fp_{rel_path}",
                )
            )

    return db, repo, shoot_dir, home_dir, hashes


def test_safe_fs_exposes_only_four_primitives() -> None:
    public_methods = {
        name for name in dir(SafeFS) if not name.startswith("_") and callable(getattr(SafeFS, name))
    }
    assert public_methods == {"read", "mkdir_quarantine", "rename_same_volume", "write_new_copy"}


def test_safe_fs_prevents_overwrite_dcim_and_cross_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sfs = SafeFS()

    # 1. Collision-safe rename with suffix and strict no-overwrite mode
    src1 = tmp_path / "src1.jpg"
    src2 = tmp_path / "src2.jpg"
    dst = tmp_path / "out" / "photo.jpg"
    src1.write_bytes(b"first")
    src2.write_bytes(b"second")

    moved1 = sfs.rename_same_volume(src1, dst, allow_suffix=True)
    assert moved1 == dst.resolve()

    with pytest.raises(DestinationExistsError):
        sfs.rename_same_volume(src2, dst, allow_suffix=False)
    assert dst.read_bytes() == b"first"
    assert src2.exists()

    moved2 = sfs.rename_same_volume(src2, dst, allow_suffix=True)
    assert moved2.name == "photo__1.jpg"
    assert moved2.read_bytes() == b"second"
    assert dst.read_bytes() == b"first"

    # 2. write_new_copy never overwrites
    with pytest.raises(DestinationExistsError):
        sfs.write_new_copy(dst, b"clobber", allow_suffix=False)
    assert dst.read_bytes() == b"first"

    # 3. DCIM camera card protection
    dcim_file = tmp_path / "SDCARD" / "DCIM" / "100MSDCF" / "DSC00001.ARW"
    dcim_file.parent.mkdir(parents=True)
    dcim_file.write_bytes(b"raw")
    with pytest.raises(ReadOnlyVolumeError):
        sfs.rename_same_volume(dcim_file, tmp_path / "q" / "DSC00001.ARW")

    # 4. Cross-volume detection via st_dev mismatch
    a = tmp_path / "a.jpg"
    b_dir = tmp_path / "other_vol"
    b_dir.mkdir()
    a.write_bytes(b"img")

    real_stat = os.stat
    b_dir_str = str(b_dir.resolve())

    class _FakeStat:
        def __init__(self, real: os.stat_result, dev: int) -> None:
            self._real = real
            self.st_dev = dev

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real, name)

    def fake_stat(path: Any, *args: Any, **kwargs: Any) -> Any:
        res = real_stat(path, *args, **kwargs)
        if not isinstance(path, int) and os.fspath(path) == b_dir_str:
            return _FakeStat(res, res.st_dev + 9999)
        return res

    monkeypatch.setattr(os, "stat", fake_stat)
    with pytest.raises(CrossVolumeError):
        sfs.rename_same_volume(a, b_dir / "a.jpg")


def test_quarantine_and_restore_roundtrip_preserves_sha256(tmp_path: Path) -> None:
    db, repo, _shoot_dir, home_dir, original_hashes = _build_atomic_shoot(tmp_path)
    try:
        executor = PlanExecutor(repo)
        plan, items = executor.create_quarantine_plan(
            "s1", ["p1", "p2"], strategy="outside_library", user_home=home_dir
        )
        assert plan.status == "draft"
        assert len(items) == 2
        assert plan.quarantine_dir is not None
        # Quarantine directory must NOT be created until approved & applied
        assert not Path(plan.quarantine_dir).exists()

        executor.approve_plan(plan.id, approved_by="photographer")
        result = executor.apply_plan(plan.id)
        assert result.status == "applied"
        assert result.moved_photo_count == 2
        assert result.moved_file_count == 7
        assert (result.quarantine_dir / MANIFEST_FILENAME).is_file()
        assert (result.quarantine_dir / RESTORE_README_FILENAME).is_file()

        readme = (result.quarantine_dir / RESTORE_README_FILENAME).read_text(encoding="utf-8")
        assert "中文说明" in readme and "English Guide" in readme
        manifest = json.loads((result.quarantine_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert manifest["plan_hash"] == plan.plan_hash
        assert len(manifest["items"]) == 2
        # Default apply_plan uses lightweight content_fp + size_bytes (no full-file sha256)
        first_file_entry = manifest["items"][0]["files"][0]
        assert "content_fp" in first_file_entry and first_file_entry["content_fp"]
        assert "size_bytes" in first_file_entry and first_file_entry["size_bytes"] > 0
        assert "sha256" not in first_file_entry

        # All original paths are gone from shoot_dir and present in quarantine_dir
        for orig_path_str in original_hashes:
            assert not Path(orig_path_str).exists()

        # Single-photo restore: restore only p1 first
        partial = executor.restore_plan(plan.id, photo_ids=["p1"])
        assert partial.restored_photo_ids == ("p1",)
        assert partial.restored_file_count == 4
        assert partial.plan_status == "applied"

        # Restore remaining photo p2
        rest = executor.restore_plan(plan.id)
        assert rest.restored_photo_ids == ("p2",)
        assert rest.restored_file_count == 3
        assert rest.plan_status == "reverted"

        # Verify 100% SHA-256 identity for every single original file
        for orig_path_str, expected_hash in original_hashes.items():
            p = Path(orig_path_str)
            assert p.is_file()
            assert _sha256(p) == expected_hash
    finally:
        db.close()


def test_stale_plan_rejected_when_decision_version_changes(tmp_path: Path) -> None:
    db, repo, _shoot_dir, home_dir, original_hashes = _build_atomic_shoot(tmp_path)
    try:
        executor = PlanExecutor(repo)
        plan, _items = executor.create_quarantine_plan(
            "s1", ["p1", "p2"], strategy="outside_library", user_home=home_dir
        )
        executor.approve_plan(plan.id, approved_by="photographer")

        # Simulate concurrent user override bumping p1.decision_version (1 -> 2)
        repo.add_verdict(
            photo_id="p1",
            source="user",
            bucket="normal",
            confidence=1.0,
            reason_codes=["user_rescue"],
        )

        with pytest.raises(StalePlanError):
            executor.apply_plan(plan.id)

        updated = repo.get_plan(plan.id)
        assert updated is not None and updated.status == "stale"

        # Zero files moved; all original files intact
        for orig_path_str, expected_hash in original_hashes.items():
            assert _sha256(Path(orig_path_str)) == expected_hash
    finally:
        db.close()


class _CrashingSafeFS(SafeFS):
    """Inject a simulated crash after `fail_on_call` rename operations."""

    def __init__(self, fail_on_call: int) -> None:
        self.fail_on_call = fail_on_call
        self.calls = 0

    def rename_same_volume(
        self,
        src: Path,
        dst: Path,
        *,
        allow_suffix: bool = True,
    ) -> Path:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("Simulated power loss mid-move")
        return super().rename_same_volume(src, dst, allow_suffix=allow_suffix)


def test_crash_mid_apply_can_resume_to_completion(tmp_path: Path) -> None:
    db, repo, _shoot_dir, home_dir, original_hashes = _build_atomic_shoot(tmp_path)
    try:
        # Fail on 6th file (group 1 has 4 files -> done; group 2 moved 1 file then crashed)
        crashing_fs = _CrashingSafeFS(fail_on_call=6)
        executor = PlanExecutor(repo, safe_fs=crashing_fs)
        plan, _ = executor.create_quarantine_plan(
            "s1", ["p1", "p2"], strategy="outside_library", user_home=home_dir
        )
        executor.approve_plan(plan.id)

        with pytest.raises(RuntimeError, match="Simulated power loss"):
            executor.apply_plan(plan.id)

        mid_plan = repo.get_plan(plan.id)
        assert mid_plan is not None and mid_plan.status == "applying"
        assert mid_plan.quarantine_dir is not None
        q_dir = Path(mid_plan.quarantine_dir)
        # Review #6: expected manifest.json & RESTORE_README.txt must ALREADY exist before moving file #1
        assert (q_dir / MANIFEST_FILENAME).is_file()
        assert (q_dir / RESTORE_README_FILENAME).is_file()
        mid_items = {it.photo_id: it.status for it in repo.list_plan_items(plan.id)}
        assert mid_items == {"p1": "done", "p2": "pending"}

        # Restart with healthy SafeFS and resume!
        recovered_executor = PlanExecutor(repo, safe_fs=SafeFS())
        result = recovered_executor.apply_plan(plan.id, resume=True)
        assert result.status == "applied"
        assert result.moved_file_count == 7
        # Resume must NOT spawn duplicate manifest__1.json or RESTORE_README__1.txt
        assert not (q_dir / "manifest__1.json").exists()
        assert not (q_dir / "RESTORE_README__1.txt").exists()

        # Restore everything and confirm SHA-256 intact
        recovered_executor.restore_plan(plan.id)
        for orig_path_str, expected_hash in original_hashes.items():
            assert _sha256(Path(orig_path_str)) == expected_hash
    finally:
        db.close()


def test_crash_mid_apply_can_fully_rollback(tmp_path: Path) -> None:
    db, repo, _shoot_dir, home_dir, original_hashes = _build_atomic_shoot(tmp_path)
    try:
        # Fail mid-way through group 2
        crashing_fs = _CrashingSafeFS(fail_on_call=6)
        executor = PlanExecutor(repo, safe_fs=crashing_fs)
        plan, _ = executor.create_quarantine_plan(
            "s1", ["p1", "p2"], strategy="outside_library", user_home=home_dir
        )
        executor.approve_plan(plan.id)

        with pytest.raises(RuntimeError, match="Simulated power loss"):
            executor.apply_plan(plan.id)

        # Roll back the interrupted plan cleanly
        recovered_executor = PlanExecutor(repo, safe_fs=SafeFS())
        restore_res = recovered_executor.restore_plan(plan.id)
        assert restore_res.plan_status == "reverted"
        assert restore_res.restored_file_count == 5  # 4 from p1 + 1 from partial p2

        for orig_path_str, expected_hash in original_hashes.items():
            assert _sha256(Path(orig_path_str)) == expected_hash
    finally:
        db.close()


def test_reapproving_applied_plan_does_not_corrupt_to_stale_and_restore_still_works(
    tmp_path: Path,
) -> None:
    """Review #1 regression: duplicate approve_plan after verdict change must not block restore."""
    db, repo, _shoot_dir, home_dir, original_hashes = _build_atomic_shoot(tmp_path)
    try:
        executor = PlanExecutor(repo)
        plan, _ = executor.create_quarantine_plan(
            "s1", ["p1"], strategy="outside_library", user_home=home_dir
        )
        executor.approve_plan(plan.id)
        executor.apply_plan(plan.id)

        # User overrides verdict on quarantined photo -> bumps decision_version
        repo.add_verdict(
            photo_id="p1",
            source="user",
            bucket="normal",
            confidence=1.0,
            reason_codes=["user_rescue"],
        )

        # Duplicate approve_plan call must raise PlanStateError and keep status='applied'
        with pytest.raises(PlanStateError):
            executor.approve_plan(plan.id)

        current = repo.get_plan(plan.id)
        assert current is not None and current.status == "applied"

        # Restore must still succeed cleanly
        res = executor.restore_plan(plan.id)
        assert res.plan_status == "reverted"
        for orig_path_str, expected_hash in original_hashes.items():
            assert _sha256(Path(orig_path_str)) == expected_hash
    finally:
        db.close()


def test_photo_becoming_protected_after_approval_is_blocked_at_apply(tmp_path: Path) -> None:
    """Review #2 regression: photo marked protected after approval must NEVER be moved."""
    db, repo, _shoot_dir, home_dir, original_hashes = _build_atomic_shoot(tmp_path)
    try:
        executor = PlanExecutor(repo)
        plan, _ = executor.create_quarantine_plan(
            "s1", ["p1"], strategy="outside_library", user_home=home_dir
        )
        executor.approve_plan(plan.id)

        # Case A: direct SQL flip of is_protected (without bumping decision_version)
        with db.transaction() as cur:
            cur.execute("UPDATE photos SET is_protected = 1 WHERE id = 'p1'")

        with pytest.raises(StalePlanError, match="protected"):
            executor.apply_plan(plan.id)

        stale_plan = repo.get_plan(plan.id)
        assert stale_plan is not None and stale_plan.status == "stale"

        # Zero files moved
        for orig_path_str, expected_hash in original_hashes.items():
            assert _sha256(Path(orig_path_str)) == expected_hash
    finally:
        db.close()


def test_apply_plan_optional_compute_sha256_flag(tmp_path: Path) -> None:
    db, repo, _shoot_dir, home_dir, original_hashes = _build_atomic_shoot(tmp_path)
    try:
        executor = PlanExecutor(repo)
        plan, _ = executor.create_quarantine_plan(
            "s1", ["p2"], strategy="outside_library", user_home=home_dir
        )
        executor.approve_plan(plan.id)
        result = executor.apply_plan(plan.id, compute_sha256=True)

        assert result.manifest_path.name == "manifest_final.json"
        final_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        for file_entry in final_manifest["items"][0]["files"]:
            assert file_entry["sha256"] == original_hashes[file_entry["src_path"]]
    finally:
        db.close()
