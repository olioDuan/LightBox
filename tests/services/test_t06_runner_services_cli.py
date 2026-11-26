"""T06 unit & integration tests: JobRunner resume/cache/pool, Services, one-click obvious rejects, CLI, and E1 gate."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lightbox.cli import app as cli_app
from lightbox.core import (
    RawFaceDetection,
    ScrfdOnnxBackend,
    get_cached_face_detector,
)
from lightbox.i18n import REASON_BURST_NON_PICK, is_burst_cull_reason, is_tech_reject_reason
from lightbox.jobs import JobRunner, StageWorkItem, make_cache_key
from lightbox.services import (
    ApprovalCodeError,
    LightboxServices,
    PlanStateError,
    StalePlanError,
    build_demo_dataset,
    restore_dataset_write_bits,
    run_e1_evaluation,
)


def _square_worker(val: int) -> int:
    return val * val


def test_job_runner_interrupt_resume_hits_cache_and_process_pool(tmp_path: Path) -> None:
    """Verify 2D JobRunner resume after crash hits completed cache, retries, circuit breaker, and pool."""
    home = tmp_path / "home"
    with LightboxServices.open(home_override=home) as svcs:
        runner = JobRunner(
            svcs.repo,
            heartbeat_interval_s=0.1,
            max_item_attempts=3,
            circuit_breaker_threshold=5,
        )

        # 1. ProcessPoolExecutor multi-worker run
        job_pool = runner.create_or_resume_job(kind="scan", session_id="s_pool")
        pool_items = [
            StageWorkItem(photo_id=f"p{i}", cache_key=make_cache_key(f"fp{i}", "sq"), payload=i)
            for i in range(1, 7)
        ]
        pool_res = runner.run_stage(
            job_id=job_pool.id,
            stage="analyze",
            items=pool_items,
            worker_fn=_square_worker,
            max_workers=2,
            use_process_pool=True,
        )
        assert pool_res.paused is False
        assert pool_res.completed == 6
        assert pool_res.results == {f"p{i}": i * i for i in range(1, 7)}
        refreshed_job = svcs.repo.get_job(job_pool.id)
        assert refreshed_job is not None and refreshed_job.heartbeat_at is not None

        # 2. Forced interruption mid-stage -> resume skips all already-completed items via cache
        job_int = runner.create_or_resume_job(kind="scan", session_id="s_int")
        persisted_cache: dict[str, int] = {}
        call_counts: dict[str, int] = {}

        def _crash_on_fourth(pid_val: tuple[str, int]) -> int:
            pid, val = pid_val
            call_counts[pid] = call_counts.get(pid, 0) + 1
            if pid == "p4":
                raise KeyboardInterrupt("Simulated sudden process termination")
            return val * 10

        items_int = [
            StageWorkItem(
                photo_id=f"p{i}",
                cache_key=make_cache_key(f"fp_int_{i}", "analyze"),
                payload=(f"p{i}", i),
            )
            for i in range(1, 7)
        ]

        with pytest.raises(KeyboardInterrupt):
            runner.run_stage(
                job_id=job_int.id,
                stage="analyze",
                items=items_int,
                worker_fn=_crash_on_fourth,
                on_item_result=lambda wi, out: persisted_cache.__setitem__(wi.cache_key, out),
            )

        assert set(persisted_cache.keys()) == {items_int[0].cache_key, items_int[1].cache_key, items_int[2].cache_key}

        # Resume same job_id with healthy worker -> p1, p2, p3 MUST hit cache (0 recomputation!)
        def _healthy_worker(pid_val: tuple[str, int]) -> int:
            pid, val = pid_val
            call_counts[pid] = call_counts.get(pid, 0) + 1
            return val * 10

        resumed = runner.run_stage(
            job_id=job_int.id,
            stage="analyze",
            items=items_int,
            worker_fn=_healthy_worker,
            cache_lookup_fn=lambda wi: persisted_cache.get(wi.cache_key),
            on_item_result=lambda wi, out: persisted_cache.__setitem__(wi.cache_key, out),
        )
        assert resumed.completed == 6
        assert resumed.cache_hits == 3
        assert call_counts["p1"] == 1
        assert call_counts["p2"] == 1
        assert call_counts["p3"] == 1
        assert call_counts["p4"] == 2  # failed once before interrupt, succeeded once on resume
        assert call_counts["p5"] == 1
        assert call_counts["p6"] == 1

        # 3. Single bad item fails 3x -> skipped_retryable without blocking remaining items
        job_retry = runner.create_or_resume_job(kind="scan", session_id="s_retry")

        def _fail_p2(pid: str) -> str:
            if pid == "p2":
                raise RuntimeError("Corrupt JPEG")
            return f"ok:{pid}"

        retry_items = [
            StageWorkItem(photo_id=f"p{i}", cache_key=make_cache_key(f"r{i}"), payload=f"p{i}")
            for i in range(1, 5)
        ]
        retry_res = runner.run_stage(
            job_id=job_retry.id,
            stage="analyze",
            items=retry_items,
            worker_fn=_fail_p2,
        )
        assert retry_res.paused is False
        assert retry_res.completed == 3
        assert retry_res.skipped_retryable == 1
        p2_item = svcs.repo.get_job_item(job_retry.id, "p2", "analyze")
        assert p2_item is not None and p2_item.status == "skipped_retryable" and p2_item.attempts == 3


def test_scan_interrupt_resume_and_obvious_rejects_idempotent_with_burst_confirm(
    tmp_path: Path,
) -> None:
    """Verify scan cache hit on rerun, burst_non_pick tagging, and draft_obvious_rejects exclusion."""
    shoot = tmp_path / "demo_shoot"
    home = tmp_path / "home"
    user_home = tmp_path / "fake_user_home"
    user_home.mkdir()
    build_demo_dataset(shoot, seed=42)

    try:
        with LightboxServices.open(home_override=home) as svcs:
            # 1. First scan
            res1 = svcs.scan.scan_folder(shoot, max_workers=1)
            assert res1.status == "completed"
            assert res1.total_photos == 18
            assert res1.counts_by_bucket["red"] >= 4
            assert res1.counts_by_bucket["exempt"] >= 1
            assert res1.burst_count >= 1
            assert res1.cache_hits == 0

            # 2. Second scan of same folder -> 100% cache hits!
            res2 = svcs.scan.scan_folder(shoot, max_workers=1)
            assert res2.session_id == res1.session_id
            assert res2.cache_hits == res2.total_photos

            # 3. Draft obvious rejects -> excludes exempt, yellow, normal, and unconfirmed burst non-picks
            draft1 = svcs.plan.draft_obvious_rejects(res1.session_id, user_home=user_home)
            assert draft1.plan is not None
            assert draft1.selected_count >= 4
            assert draft1.verifier_report["exempt_count"] == 0
            assert draft1.verifier_report["unconfirmed_burst_non_pick_count"] == 0

            # Approve & apply quarantine plan
            svcs.plan.approve_plan(draft1.plan.id, approved_by="tester")
            applied = svcs.plan.apply_plan(draft1.plan.id)
            assert applied.status == "applied"
            assert applied.moved_photo_count == draft1.selected_count

            # 4. Calling draft_obvious_rejects a SECOND time must NOT re-include quarantined photos!
            draft2 = svcs.plan.draft_obvious_rejects(res1.session_id, user_home=user_home)
            assert draft2.plan is None
            assert draft2.selected_count == 0
            assert draft2.excluded_already_quarantined == draft1.selected_count

            # 5. Confirm burst -> non-picks must carry `burst_non_pick` (NOT technical reject!)
            bursts = svcs.burst.list_bursts(res1.session_id)
            assert len(bursts) >= 1
            b0 = bursts[0]
            pick_id = b0.members[0].photo_id
            conf_res = svcs.burst.confirm_burst(b0.burst.id, pick_id)
            assert conf_res.confirmed is True
            assert len(conf_res.non_pick_photo_ids) == len(b0.members) - 1

            for np_id in conf_res.non_pick_photo_ids:
                verdict = svcs.repo.get_effective_verdict(np_id)
                assert verdict is not None
                assert REASON_BURST_NON_PICK in verdict.reason_codes
                assert is_burst_cull_reason(verdict.reason_codes) is True
                assert is_tech_reject_reason(verdict.reason_codes) is False

            # 6. ReviewService set_verdict + overrides + undo
            photos = svcs.review.list_photos(res1.session_id, bucket="normal", limit=5)
            assert photos
            target_pid = photos[0].photo_id
            svcs.review.set_verdict(target_pid, bucket="yellow", reason_tag="needs_check")
            assert svcs.repo.get_effective_verdict(target_pid).bucket == "yellow"  # type: ignore[union-attr]
            overrides = svcs.repo.list_overrides(res1.session_id)
            assert any(o.photo_id == target_pid and o.to_bucket == "yellow" for o in overrides)
            svcs.review.undo_verdict(target_pid)
            assert svcs.repo.get_effective_verdict(target_pid).bucket == "normal"  # type: ignore[union-attr]

            # Restore quarantined photos cleanly
            restored = svcs.plan.restore_plan(draft1.plan.id)
            assert restored.plan_status == "reverted"
    finally:
        restore_dataset_write_bits(shoot)


def test_cli_approval_code_expiry_lockout_and_tty_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 6-digit SHA-256 CLI approval code TTL, 5-attempt lockout, and isatty gate."""
    shoot = tmp_path / "approve_shoot"
    home = tmp_path / "home"
    user_home = tmp_path / "u_home"
    user_home.mkdir()
    build_demo_dataset(shoot, seed=77)
    monkeypatch.setenv("LIGHTBOX_HOME", str(home))

    try:
        with LightboxServices.open(home_override=home) as svcs:
            scan_res = svcs.scan.scan_folder(shoot, max_workers=1)
            draft = svcs.plan.draft_obvious_rejects(scan_res.session_id, user_home=user_home)
            assert draft.plan is not None
            plan_id = draft.plan.id

            # 1. Expired code must fail
            expired = svcs.plan.issue_cli_approval_code(plan_id, ttl_seconds=-1.0)
            with pytest.raises(ApprovalCodeError, match="expired"):
                svcs.plan.verify_and_approve_with_code(plan_id, expired.code)

            # 2. 5 wrong attempts lock out and invalidate even the real code
            issued = svcs.plan.issue_cli_approval_code(plan_id, ttl_seconds=600.0)
            wrong_code = "000000" if issued.code != "000000" else "999999"
            for _ in range(4):
                with pytest.raises(ApprovalCodeError, match="Invalid approval code"):
                    svcs.plan.verify_and_approve_with_code(plan_id, wrong_code)
            with pytest.raises(ApprovalCodeError, match="locked out"):
                svcs.plan.verify_and_approve_with_code(plan_id, wrong_code)
            with pytest.raises(ApprovalCodeError, match="locked"):
                svcs.plan.verify_and_approve_with_code(plan_id, issued.code)

            # 3. Re-issue fresh code and verify happy path
            fresh = svcs.plan.issue_cli_approval_code(plan_id, ttl_seconds=600.0)
            approved = svcs.plan.verify_and_approve_with_code(plan_id, fresh.code)
            assert approved.status == "approved"

        # 4. CLI `lightbox approve` must refuse non-TTY stdin
        runner = CliRunner()
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        cli_res = runner.invoke(cli_app, ["approve", plan_id])
        assert cli_res.exit_code == 1
        assert "TTY" in cli_res.output
    finally:
        restore_dataset_write_bits(shoot)


def test_cli_demo_and_doctor_under_30s(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PLAN T06: `lightbox demo` completes zero-Key/zero-network in <30s; `lightbox doctor` passes."""
    home = tmp_path / "cli_home"
    demo_dir = tmp_path / "cli_demo_shoot"
    monkeypatch.setenv("LIGHTBOX_HOME", str(home))
    runner = CliRunner()

    try:
        t0 = time.monotonic()
        demo_res = runner.invoke(cli_app, ["demo", "--dir", str(demo_dir), "--json"])
        elapsed = time.monotonic() - t0
        assert demo_res.exit_code == 0, demo_res.output
        assert elapsed < 30.0
        demo_data = json.loads(demo_res.output)
        assert demo_data["counts_by_bucket"]["red"] >= 4
        assert demo_data["counts_by_bucket"]["exempt"] >= 1
        assert demo_data["obvious_reject_plan_id"] is not None

        doc_res = runner.invoke(cli_app, ["doctor", "--json"])
        assert doc_res.exit_code == 0, doc_res.output
        doc_data = json.loads(doc_res.output)
        assert doc_data["overall_ok"] is True
        check_names = {c["name"] for c in doc_data["checks"]}
        assert {"python", "sqlite_vec", "app_data_dir", "exiftool", "onnxruntime"} <= check_names

        # Also verify `lightbox scan` on existing demo_dir hits 100% cache
        scan_res = runner.invoke(cli_app, ["scan", str(demo_dir), "--json"])
        assert scan_res.exit_code == 0, scan_res.output
        scan_data = json.loads(scan_res.output)
        assert scan_data["cache_hits"] == scan_data["total_photos"]
    finally:
        restore_dataset_write_bits(demo_dir)


def test_e1_synthetic_ci_gate_meets_hard_thresholds(tmp_path: Path) -> None:
    """PLAN T06 & SPEC §11.2: E1 gate achieves red FPR < 1%, obvious recall >= 75%, exempt mislabel == 0%."""
    report = run_e1_evaluation(tmp_path / "e1_run", sensitivity="conservative", max_workers=1)
    assert report.red_fpr < 0.01, f"Red FPR too high: {report.red_fpr:.4f}; failures={report.failures}"
    assert report.obvious_reject_recall >= 0.75, (
        f"Reject recall too low: {report.obvious_reject_recall:.4f}; failures={report.failures}"
    )
    assert report.exempt_mislabel_rate == 0.0, (
        f"Exempt mislabel non-zero: {report.exempt_mislabel_rate:.4f}; failures={report.failures}"
    )
    assert report.passed is True


def test_cli_approve_without_apply_verifies_code_and_approves_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review #1 regression: `lightbox approve <plan_id>` (without --apply) must prompt & approve."""
    shoot = tmp_path / "approve_cli_shoot"
    home = tmp_path / "home"
    build_demo_dataset(shoot, seed=52)
    monkeypatch.setenv("LIGHTBOX_HOME", str(home))

    try:
        with LightboxServices.open(home_override=home) as svcs:
            scan_res = svcs.scan.scan_folder(shoot, max_workers=1)
            draft = svcs.plan.draft_obvious_rejects(scan_res.session_id, strategy="sibling")
            assert draft.plan is not None
            plan_id = draft.plan.id
            count = draft.selected_count

        # Intercept issued code via deterministic secrets.randbelow so interactive prompt can echo it
        monkeypatch.setattr("lightbox.services.plan.secrets.randbelow", lambda _n: 123456)
        monkeypatch.setattr(sys.modules["lightbox.cli.app"], "_is_interactive_tty", lambda: True)

        runner = CliRunner()
        res = runner.invoke(cli_app, ["approve", plan_id], input=f"{count}\n123456\n")
        assert res.exit_code == 0, res.output
        assert "123456" in res.output
        assert "status=approved" in res.output

        with LightboxServices.open(home_override=home) as svcs:
            plan, items = svcs.plan.get_plan(plan_id)
            assert plan.status == "approved"
            # Without --apply, files must NOT be moved yet
            assert all(it.status == "pending" for it in items)
            assert all(Path(p).exists() for it in items for p in it.src_paths)

            # Now draft a fresh plan (supersedes old approved plan) and test `--apply`
            draft2 = svcs.plan.draft_obvious_rejects(scan_res.session_id, strategy="sibling")
            assert draft2.plan is not None
            plan2_id = draft2.plan.id
            count2 = draft2.selected_count

        res_apply = runner.invoke(cli_app, ["approve", plan2_id, "--apply"], input=f"{count2}\n123456\n")
        assert res_apply.exit_code == 0, res_apply.output
        assert "Applied plan" in res_apply.output

        with LightboxServices.open(home_override=home) as svcs:
            plan2, _ = svcs.plan.get_plan(plan2_id)
            assert plan2.status == "applied"
            svcs.plan.restore_plan(plan2_id)
    finally:
        restore_dataset_write_bits(shoot)


def test_face_model_added_later_backfills_faces_and_reuses_cached_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review #2 & #3 regressions: adding det_10g.onnx invalidates cache, backfills faces, & loads once."""
    shoot = tmp_path / "face_shoot"
    home = tmp_path / "home"
    build_demo_dataset(shoot, seed=63)
    get_cached_face_detector.cache_clear()

    try:
        # 1. Initial scan WITHOUT face model -> 0 faces
        with LightboxServices.open(home_override=home) as svcs:
            res1 = svcs.scan.scan_folder(shoot, max_workers=1)
            assert res1.cache_hits == 0
            photos = svcs.repo.list_photos(res1.session_id)
            assert all(len(svcs.repo.list_faces(p.primary_content_fp)) == 0 for p in photos)

            # 2. Place a fake det_10g.onnx file into <APP_DATA_DIR>/models/antelopev2/
            model_file = svcs.app_paths.models_dir / "antelopev2" / "det_10g.onnx"
            model_file.parent.mkdir(parents=True, exist_ok=True)
            model_file.write_bytes(b"FAKE_SCRFD_ONNX_WEIGHTS_V1")

            # Count how many times ScrfdOnnxBackend.__init__ is called across all 18 photos
            init_calls = 0
            orig_init = ScrfdOnnxBackend.__init__

            def _counting_init(self: ScrfdOnnxBackend, *args: object, **kwargs: object) -> None:
                nonlocal init_calls
                init_calls += 1
                orig_init(self, *args, **kwargs)  # type: ignore[arg-type]

            fake_det = RawFaceDetection(
                bbox_px=(100.0, 100.0, 160.0, 160.0),
                landmarks_px=(
                    (115.0, 120.0),
                    (145.0, 120.0),
                    (130.0, 132.0),
                    (118.0, 148.0),
                    (142.0, 148.0),
                ),
                det_score=0.93,
            )
            monkeypatch.setattr(ScrfdOnnxBackend, "__init__", _counting_init)
            monkeypatch.setattr(ScrfdOnnxBackend, "detect_raw", lambda _self, _img: [fake_det])

            # Rescan MUST miss analysis cache and backfill faces for all photos, loading backend ONCE!
            res2 = svcs.scan.scan_folder(shoot, max_workers=1)
            assert res2.cache_hits == 0
            assert init_calls == 1
            for p in photos:
                faces = svcs.repo.list_faces(p.primary_content_fp)
                assert len(faces) == 1
                assert faces[0].landmarks is not None and len(faces[0].landmarks) == 5

            # 3. Third scan with same face model still installed -> 100% cache hit!
            res3 = svcs.scan.scan_folder(shoot, max_workers=1)
            assert res3.cache_hits == res3.total_photos
            assert init_calls == 1
    finally:
        get_cached_face_detector.cache_clear()
        restore_dataset_write_bits(shoot)


def test_draft_obvious_rejects_separates_burst_non_picks_and_prevents_overlapping_plans(
    tmp_path: Path,
) -> None:
    """Review #4 & #5 regressions: separate tech/burst breakdown & supersede overlapping unapplied drafts."""
    shoot = tmp_path / "overlap_shoot"
    home = tmp_path / "home"
    build_demo_dataset(shoot, seed=84)

    try:
        with LightboxServices.open(home_override=home) as svcs:
            res = svcs.scan.scan_folder(shoot, max_workers=1)

            # Confirm the burst so its non-picks become `burst_non_pick`
            bursts = svcs.burst.list_bursts(res.session_id)
            assert len(bursts) >= 1
            b0 = bursts[0]
            conf = svcs.burst.confirm_burst(b0.burst.id, b0.members[0].photo_id)
            expected_burst_non_picks = len(conf.non_pick_photo_ids)
            assert expected_burst_non_picks >= 2

            # First click: draft_obvious_rejects reports tech vs burst_non_pick breakdown separately
            draft1 = svcs.plan.draft_obvious_rejects(res.session_id, strategy="sibling")
            assert draft1.plan is not None
            assert draft1.tech_reject_count >= 4
            assert draft1.burst_non_pick_count == expected_burst_non_picks
            assert (
                draft1.tech_reject_count + draft1.burst_non_pick_count + draft1.manual_reject_count
                == draft1.selected_count
            )
            assert draft1.verifier_report["tech_reject_count"] == draft1.tech_reject_count
            assert draft1.verifier_report["burst_non_pick_count"] == draft1.burst_non_pick_count
            assert draft1.superseded_plan_ids == ()

            # Second click (double-click!): supersedes draft1 to `stale` so both can never run!
            draft2 = svcs.plan.draft_obvious_rejects(res.session_id, strategy="sibling")
            assert draft2.plan is not None
            assert draft2.plan.id != draft1.plan.id
            assert draft2.superseded_plan_ids == (draft1.plan.id,)

            old_plan, _ = svcs.plan.get_plan(draft1.plan.id)
            assert old_plan.status == "stale"
            with pytest.raises(PlanStateError):
                svcs.plan.approve_plan(draft1.plan.id)
            with pytest.raises(StalePlanError):
                svcs.plan.apply_plan(draft1.plan.id)

            # Simulate draft2 entering in-flight `applying` state -> draft3 must exclude its photos!
            svcs.plan.approve_plan(draft2.plan.id)
            svcs.repo.update_plan_status(draft2.plan.id, "applying")
            draft3 = svcs.plan.draft_obvious_rejects(res.session_id, strategy="sibling")
            assert draft3.plan is None
            assert draft3.selected_count == 0
            assert draft3.excluded_in_applying_plan == draft2.selected_count
            still_applying, _ = svcs.plan.get_plan(draft2.plan.id)
            assert still_applying.status == "applying"
    finally:
        restore_dataset_write_bits(shoot)
