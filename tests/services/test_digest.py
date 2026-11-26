"""T17 tests: Shooting Digest service, SQL GROUP BY slices, ToolRegistry agent-only tool, and REST API."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lightbox.api import create_app
from lightbox.core import ENGINE_VERSION
from lightbox.i18n import (
    REASON_BURST_NON_PICK,
    REASON_FACE_OUT_OF_FOCUS,
    REASON_MOTION_BLUR,
    REASON_OUT_OF_FOCUS,
    REASON_USER_REJECT,
)
from lightbox.paths import get_app_paths
from lightbox.services import DIGEST_DIMENSIONS, LightboxServices
from lightbox.store import (
    ExifRecord,
    MetricsRecord,
    PhotoRecord,
    RootRecord,
    SessionRecord,
)
from lightbox.tools import ToolContext, ToolRegistry


def _seed_session(
    svcs: LightboxServices,
    *,
    session_id: str,
    name: str,
    root_path: str,
) -> None:
    root_id = f"root_{session_id}"
    svcs.repo.upsert_root(
        RootRecord(id=root_id, path=root_path, device_id="dev1", kind="disk", readonly=False)
    )
    svcs.repo.upsert_session(
        SessionRecord(
            id=session_id,
            root_id=root_id,
            root_path=root_path,
            name=name,
            sensitivity="conservative",
            engine_version=ENGINE_VERSION,
            created_at=1_700_000_000.0,
            last_scanned_at=1_700_000_100.0,
        )
    )


def _seed_photo(
    svcs: LightboxServices,
    *,
    session_id: str,
    photo_id: str,
    capture_time: float | None = None,
    lens_model: str | None = None,
    is_protected: bool = False,
    bucket: str = "normal",
    source: str = "engine",
    reason_codes: tuple[str, ...] = (),
    exif: ExifRecord | None = None,
    scene_tag: str | None = None,
    motion_state: str | None = None,
) -> None:
    fp = f"fp_{photo_id}"
    svcs.repo.upsert_photo(
        PhotoRecord(
            id=photo_id,
            session_id=session_id,
            stem=photo_id,
            rel_dir="",
            primary_content_fp=fp,
            cohort="camera",
            capture_time=capture_time,
            camera_model="Sony A7 IV",
            lens_model=lens_model,
            is_protected=is_protected,
            exempt_reason="in_camera_protected" if is_protected else None,
        )
    )
    if exif is not None:
        svcs.repo.upsert_exif(exif)
    svcs.repo.upsert_metrics(
        MetricsRecord(
            content_fp=fp,
            engine_version=ENGINE_VERSION,
            sharp_p95=18.0,
            motion_state=motion_state,
            scene_tag=scene_tag,
        )
    )
    svcs.repo.add_verdict(
        photo_id=photo_id,
        source=source,
        bucket=bucket,
        confidence=0.95,
        reason_codes=reason_codes,
        increment_decision_version=False,
    )


def test_digest_strictly_separates_tech_reject_from_burst_cull(tmp_path: Path) -> None:
    """PLAN T17 acceptance: 10 sharp burst (1 pick) + 3 camera-shake blur -> tech=3/13, burst=9/13."""
    with LightboxServices.open(tmp_path / "home") as svcs:
        _seed_session(svcs, session_id="s1", name="Street Walk", root_path="/shoots/s1")
        ts = datetime(2026, 6, 1, 18, 0, tzinfo=UTC).timestamp()

        # 10 sharp burst frames
        svcs.repo.upsert_burst("b1", "s1", kind="burst", confirmed=False)
        for i in range(10):
            pid = f"burst_{i:02d}"
            _seed_photo(
                svcs,
                session_id="s1",
                photo_id=pid,
                capture_time=ts + i * 0.2,
                lens_model="FE 35mm F1.4 GM",
                bucket="normal",
                exif=ExifRecord(
                    photo_id=pid,
                    aperture=2.0,
                    shutter_s=1.0 / 500.0,
                    iso=200,
                    focal_eq=35.0,
                    light_bucket="bright",
                    flash_fired=False,
                    capture_local_hour=18,
                ),
                scene_tag="street",
                motion_state="moving",
            )
            svcs.repo.upsert_burst_member("b1", pid, rank=i + 1, is_pick=(i == 0))

        # Confirm burst picking burst_00 -> 9 qualified non-picks get `burst_non_pick`
        svcs.burst.confirm_burst("b1", "burst_00", actor="user")

        # 3 camera-shake blurry frames (tech_reject)
        for i in range(3):
            pid = f"blur_{i:02d}"
            _seed_photo(
                svcs,
                session_id="s1",
                photo_id=pid,
                capture_time=ts + 60.0 + i,
                lens_model="FE 35mm F1.4 GM",
                bucket="red",
                reason_codes=(REASON_MOTION_BLUR,),
                exif=ExifRecord(
                    photo_id=pid,
                    aperture=2.0,
                    shutter_s=1.0 / 15.0,
                    iso=1600,
                    focal_eq=35.0,
                    light_bucket="night",
                    flash_fired=False,
                    capture_local_hour=21,
                ),
                scene_tag="night_city",
                motion_state="moving",
            )

        report = svcs.digest.generate_digest(session_id="s1")
        ov = report.overview

        assert ov.total_shots == 13
        assert ov.tech_reject_count == 3
        assert ov.tech_reject_rate == round(3 / 13, 4)
        assert ov.burst_cull_count == 9
        assert ov.burst_cull_rate == round(9 / 13, 4)
        assert ov.kept_count == 1
        assert ov.non_burst_total == 4
        assert ov.kept_rate_excl_burst == round(1 / 4, 4)
        assert ov.kept_rate_total == round(1 / 13, 4)
        assert ov.uncertain_count == 0
        assert ov.subjective_reject_count == 0
        assert (
            ov.tech_reject_count
            + ov.burst_cull_count
            + ov.subjective_reject_count
            + ov.kept_count
            + ov.uncertain_count
            == ov.total_shots
        )

        assert len(report.top_tech_reasons) == 1
        assert report.top_tech_reasons[0].code == REASON_MOTION_BLUR
        assert report.top_tech_reasons[0].count == 3


def test_blurry_frame_inside_burst_still_counts_as_tech_reject(tmp_path: Path) -> None:
    """A genuinely blurry photo inside a confirmed burst must stay `tech_reject`, not `burst_cull`."""
    with LightboxServices.open(tmp_path / "home") as svcs:
        _seed_session(svcs, session_id="s_burst_blur", name="Burst Blur", root_path="/shoots/bb")
        svcs.repo.upsert_burst("b_mix", "s_burst_blur", kind="burst", confirmed=False)

        # 3 sharp burst frames + 1 blurry burst frame + 1 yellow + 1 exempt
        for i in range(3):
            pid = f"sharp_{i}"
            _seed_photo(svcs, session_id="s_burst_blur", photo_id=pid, bucket="normal")
            svcs.repo.upsert_burst_member("b_mix", pid, rank=i + 1)

        _seed_photo(
            svcs,
            session_id="s_burst_blur",
            photo_id="blur_in_burst",
            bucket="red",
            reason_codes=(REASON_MOTION_BLUR,),
        )
        svcs.repo.upsert_burst_member("b_mix", "blur_in_burst", rank=4)

        svcs.burst.confirm_burst("b_mix", "sharp_0", actor="user")

        # Also add a photo with both burst_non_pick and out_of_focus -> must count as tech_reject
        _seed_photo(
            svcs,
            session_id="s_burst_blur",
            photo_id="dual_code_blur",
            bucket="red",
            source="user",
            reason_codes=(REASON_BURST_NON_PICK, REASON_OUT_OF_FOCUS),
        )
        # Yellow uncertain & exempt protected
        _seed_photo(
            svcs,
            session_id="s_burst_blur",
            photo_id="yellow_face",
            bucket="yellow",
            reason_codes=(REASON_FACE_OUT_OF_FOCUS,),
        )
        _seed_photo(
            svcs,
            session_id="s_burst_blur",
            photo_id="locked_shot",
            bucket="exempt",
            is_protected=True,
        )

        ov = svcs.digest.generate_digest(session_id="s_burst_blur").overview
        assert ov.total_shots == 7
        assert ov.kept_count == 2  # sharp_0 + locked_shot
        assert ov.burst_cull_count == 2  # sharp_1 + sharp_2
        assert ov.tech_reject_count == 2  # blur_in_burst + dual_code_blur
        assert ov.uncertain_count == 1  # yellow_face
        assert ov.non_burst_total == 5
        assert ov.kept_rate_excl_burst == round(2 / 5, 4)


def test_digest_slices_min_samples_unknown_and_stable_setups(tmp_path: Path) -> None:
    """Verify all 9 SQL GROUP BY dimensions, `min_samples=5`, `unknown` fallback, and 2D focal x shutter."""
    with LightboxServices.open(tmp_path / "home") as svcs:
        _seed_session(svcs, session_id="s_rich", name="Rich Shoot", root_path="/shoots/rich")
        base_ts = datetime(2026, 7, 10, 14, 0, tzinfo=UTC).timestamp()

        # Group A: 6 kept daylight shots @ 35mm, 1/500s, ISO 100, static, portrait (high kept + stable setup)
        for i in range(6):
            pid = f"good_{i}"
            _seed_photo(
                svcs,
                session_id="s_rich",
                photo_id=pid,
                capture_time=base_ts + i,
                lens_model="FE 35mm F1.4 GM",
                bucket="normal",
                exif=ExifRecord(
                    photo_id=pid,
                    aperture=2.8,
                    shutter_s=1.0 / 500.0,
                    iso=100,
                    focal_eq=35.0,
                    light_bucket="bright",
                    flash_fired=False,
                    capture_local_hour=14,
                ),
                scene_tag="portrait",
                motion_state="static",
            )

        # Group B: 5 night telephoto shots @ 85mm, 1/15s (slower than safe), ISO 6400 -> 4 motion_blur + 1 kept
        for i in range(5):
            pid = f"night_{i}"
            is_blur = i < 4
            _seed_photo(
                svcs,
                session_id="s_rich",
                photo_id=pid,
                capture_time=base_ts + 3600 + i,
                lens_model="FE 85mm F1.8",
                bucket="red" if is_blur else "normal",
                reason_codes=(REASON_MOTION_BLUR,) if is_blur else (),
                exif=ExifRecord(
                    photo_id=pid,
                    aperture=1.8,
                    shutter_s=1.0 / 15.0,
                    iso=6400,
                    focal_eq=85.0,
                    light_bucket="night",
                    flash_fired=True,
                    capture_local_hour=21,
                ),
                scene_tag="night_city",
                motion_state="moving",
            )

        # Group C: 2 photos with missing EXIF/metrics (below min_samples=5 and mapped to `unknown`)
        for i in range(2):
            pid = f"sparse_{i}"
            _seed_photo(
                svcs,
                session_id="s_rich",
                photo_id=pid,
                capture_time=base_ts + 7200 + i,
                lens_model=None,
                bucket="normal",
                exif=None,
                scene_tag=None,
                motion_state=None,
            )

        report = svcs.digest.generate_digest(session_id="s_rich", min_samples=5, lang="zh-CN")
        assert tuple(d.key for d in report.dimensions) == DIGEST_DIMENSIONS
        dim_map = {d.key: d for d in report.dimensions}

        # Every single dimension sums to total_shots (13) and maps missing fields to `unknown`
        for dim_key, dim in dim_map.items():
            assert sum(b.total for b in dim.buckets) == 13, f"Dimension {dim_key} total mismatch"
            unknown_buckets = [b for b in dim.buckets if b.key == "unknown"]
            assert len(unknown_buckets) == 1
            assert unknown_buckets[0].total == 2
            assert unknown_buckets[0].meets_min_samples is False

        # High tech-reject conditions: night / 85mm / slower_over_1_stop / ISO 3201-6400 (4/5 = 80%, motion_blur)
        assert report.high_tech_reject_conditions
        top_bad = report.high_tech_reject_conditions[0]
        assert top_bad.tech_reject == 4
        assert top_bad.total == 5
        assert top_bad.tech_reject_rate == 0.8
        assert top_bad.top_reason_code == REASON_MOTION_BLUR
        assert all(c.bucket_key != "unknown" for c in report.high_tech_reject_conditions)

        # High kept conditions: bright / 35mm / static / ISO <= 200 (6/6 = 100%)
        assert report.high_kept_conditions
        top_good = report.high_kept_conditions[0]
        assert top_good.kept == 6
        assert top_good.kept_rate_excl_burst == 1.0
        assert all(c.bucket_key != "unknown" for c in report.high_kept_conditions)

        # Fixed 2D stable setups (focal_bucket x shutter_bucket):
        # 1st: wide_24_35 x fast_250_1000 (6/6 = 100%), 2nd: portrait_56_105 x slow_handheld (1/5 = 20%)
        assert len(report.stable_setups) == 2
        best_setup = report.stable_setups[0]
        assert best_setup.focal_bucket == "wide_24_35"
        assert best_setup.shutter_bucket == "fast_250_1000"
        assert best_setup.total == 6
        assert best_setup.kept_rate_excl_burst == 1.0
        assert best_setup.tech_reject_rate == 0.0


def test_cross_session_date_range_filtering_and_validation(tmp_path: Path) -> None:
    """Cross-session aggregation and ISO YYYY-MM-DD date filtering."""
    with LightboxServices.open(tmp_path / "home") as svcs:
        _seed_session(svcs, session_id="s_may", name="May Shoot", root_path="/shoots/may")
        _seed_session(svcs, session_id="s_jun", name="June Shoot", root_path="/shoots/jun")

        may_ts = datetime(2026, 5, 3, 10, 0, tzinfo=UTC).timestamp()
        jun_ts = datetime(2026, 6, 15, 16, 0, tzinfo=UTC).timestamp()

        for i in range(4):
            _seed_photo(svcs, session_id="s_may", photo_id=f"m_{i}", capture_time=may_ts + i)
        for i in range(3):
            _seed_photo(
                svcs,
                session_id="s_jun",
                photo_id=f"j_{i}",
                capture_time=jun_ts + i,
                bucket="red",
                reason_codes=(REASON_OUT_OF_FOCUS,),
            )

        all_rep = svcs.digest.generate_digest()
        assert all_rep.scope_mode == "cross_session"
        assert all_rep.session_count == 2
        assert all_rep.overview.total_shots == 7
        assert all_rep.overview.kept_count == 4
        assert all_rep.overview.tech_reject_count == 3

        jun_only = svcs.digest.generate_digest(date_from="2026-06-01", date_to="2026-06-30")
        assert jun_only.session_count == 1
        assert jun_only.overview.total_shots == 3
        assert jun_only.overview.tech_reject_count == 3

        may_only = svcs.digest.generate_digest(date_to="2026-05-31")
        assert may_only.session_count == 1
        assert may_only.overview.total_shots == 4
        assert may_only.overview.kept_count == 4

        with pytest.raises(ValueError, match="Unknown session_id"):
            svcs.digest.generate_digest(session_id="does_not_exist")
        with pytest.raises(ValueError, match="Invalid ISO date"):
            svcs.digest.generate_digest(date_from="2026/05/01")
        with pytest.raises(ValueError, match="on or before"):
            svcs.digest.generate_digest(date_from="2026-07-01", date_to="2026-06-01")


def test_shooting_digest_tool_registry_and_rest_api_zero_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify agent-only `shooting_digest` tool and zero-LLM `GET /api/insights/digest` endpoint."""
    for env_var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)

    paths = get_app_paths(tmp_path / "app_home")
    with LightboxServices(app_paths=paths, migrate=True) as svcs:
        _seed_session(svcs, session_id="s_api", name="API Shoot", root_path="/shoots/api")
        ts = datetime(2026, 8, 1, 17, 30, tzinfo=UTC).timestamp()
        for i in range(5):
            _seed_photo(
                svcs,
                session_id="s_api",
                photo_id=f"ok_{i}",
                capture_time=ts + i,
                bucket="normal",
                exif=ExifRecord(
                    photo_id=f"ok_{i}",
                    shutter_s=1.0 / 250.0,
                    focal_eq=50.0,
                    iso=200,
                    light_bucket="bright",
                    capture_local_hour=17,
                ),
            )
        _seed_photo(
            svcs,
            session_id="s_api",
            photo_id="bad_0",
            capture_time=ts + 10,
            bucket="red",
            reason_codes=(REASON_MOTION_BLUR,),
        )

        # 1. ToolRegistry: exposed to agent, excluded from MCP 14 tools
        reg = ToolRegistry()
        assert len(reg.mcp_specs()) == 14
        assert "shooting_digest" not in {s.name for s in reg.mcp_specs()}
        assert "shooting_digest" in {s["name"] for s in reg.agent_schemas()}
        assert reg.get("shooting_digest").risk_level == "READ"

        agent_res = reg.execute(
            "shooting_digest",
            {"session_id": "s_api"},
            ToolContext(services=svcs, surface="agent", lang="en-US"),
        )
        assert not agent_res.is_error
        assert agent_res.data["overview"]["total_shots"] == 6
        assert agent_res.data["overview"]["tech_reject_count"] == 1

        # Must be rejected if called on an MCP surface
        mcp_res = reg.execute(
            "shooting_digest",
            {"session_id": "s_api"},
            ToolContext(services=svcs, surface="mcp:test"),
        )
        assert mcp_res.is_error

    # 2. FastAPI GET /api/insights/digest (requires cookie auth, zero LLM)
    token = "digest-launch-tok"
    app = create_app(app_paths=paths, port=8765, bootstrap_token=token)
    client = TestClient(app, base_url="http://127.0.0.1:8765")

    assert client.get("/api/insights/digest").status_code == 401

    launch = client.get(f"/auth/launch?token={token}", follow_redirects=False)
    assert launch.status_code == 302

    resp = client.get("/api/insights/digest?session_id=s_api&lang=en-US")
    assert resp.status_code == 200
    body = resp.json()
    assert body["scope"]["mode"] == "session"
    assert body["scope"]["session_id"] == "s_api"
    assert body["overview"]["total_shots"] == 6
    assert body["overview"]["kept_count"] == 5
    assert body["overview"]["tech_reject_count"] == 1
    assert len(body["dimensions"]) == len(DIGEST_DIMENSIONS)
    assert len(body["stable_setups"]) == 1

    # Invalid session -> 404, invalid date -> 400
    assert client.get("/api/insights/digest?session_id=missing").status_code == 404
    assert client.get("/api/insights/digest?date_from=not-a-date").status_code == 400


def test_highlight_columns_compare_against_overall_baseline_and_never_contradict(
    tmp_path: Path,
) -> None:
    """Lens A (10% reject) vs Lens B (50% reject) with 30% overall must never appear in both columns."""
    with LightboxServices.open(tmp_path / "home") as svcs:
        _seed_session(svcs, session_id="s_cmp", name="Lens Compare", root_path="/shoots/cmp")
        ts = datetime(2026, 9, 1, 15, 0, tzinfo=UTC).timestamp()

        # Lens A: 10 shots, 1 motion_blur (10% tech reject, 90% kept)
        for i in range(10):
            is_blur = i == 0
            _seed_photo(
                svcs,
                session_id="s_cmp",
                photo_id=f"a_{i:02d}",
                capture_time=ts + i,
                lens_model="Lens A",
                bucket="red" if is_blur else "normal",
                reason_codes=(REASON_MOTION_BLUR,) if is_blur else (),
            )

        # Lens B: 10 shots, 5 motion_blur (50% tech reject, 50% kept)
        for i in range(10):
            is_blur = i < 5
            _seed_photo(
                svcs,
                session_id="s_cmp",
                photo_id=f"b_{i:02d}",
                capture_time=ts + 100 + i,
                lens_model="Lens B",
                bucket="red" if is_blur else "normal",
                reason_codes=(REASON_MOTION_BLUR,) if is_blur else (),
            )

        report = svcs.digest.generate_digest(session_id="s_cmp", min_samples=5)
        assert report.overview.tech_reject_rate == 0.3
        assert report.overview.kept_rate_excl_burst == 0.7

        bad_keys = {(c.dimension_key, c.bucket_key) for c in report.high_tech_reject_conditions}
        good_keys = {(c.dimension_key, c.bucket_key) for c in report.high_kept_conditions}

        # Strict mutual exclusivity between the two columns
        assert bad_keys.isdisjoint(good_keys)

        # Lens B (+20pp reject) is high-reject ONLY; Lens A (+20pp kept) is high-kept ONLY
        assert ("lens_model", "Lens B") in bad_keys
        assert ("lens_model", "Lens A") not in bad_keys
        assert ("lens_model", "Lens A") in good_keys
        assert ("lens_model", "Lens B") not in good_keys

        lens_b_hi = next(
            c for c in report.high_tech_reject_conditions if c.bucket_key == "Lens B"
        )
        lens_a_hi = next(c for c in report.high_kept_conditions if c.bucket_key == "Lens A")
        assert lens_b_hi.tech_reject_delta == 0.2
        assert lens_a_hi.kept_rate_delta == 0.2


def test_manual_subjective_reject_excluded_from_tech_reject_and_reasons(tmp_path: Path) -> None:
    """10 sharp photos with 3 manual `user_reject` -> tech_reject=0%, subjective_reject=30%, no `user_reject` in top_tech_reasons."""
    with LightboxServices.open(tmp_path / "home") as svcs:
        _seed_session(svcs, session_id="s_subj", name="Subjective Cull", root_path="/shoots/subj")
        ts = datetime(2026, 9, 2, 11, 0, tzinfo=UTC).timestamp()

        for i in range(10):
            _seed_photo(
                svcs,
                session_id="s_subj",
                photo_id=f"sharp_{i:02d}",
                capture_time=ts + i,
                lens_model="FE 50mm F1.2 GM",
                bucket="normal",
            )

        # User manually rejects 3 sharp photos (expression/framing) via ReviewService
        for i in range(3):
            svcs.review.set_verdict(f"sharp_{i:02d}", bucket="red", source="user")

        report = svcs.digest.generate_digest(session_id="s_subj", min_samples=5)
        ov = report.overview
        assert ov.total_shots == 10
        assert ov.tech_reject_count == 0
        assert ov.tech_reject_rate == 0.0
        assert ov.subjective_reject_count == 3
        assert ov.subjective_reject_rate == 0.3
        assert ov.kept_count == 7
        assert ov.kept_rate_excl_burst == 0.7
        assert report.top_tech_reasons == ()

        # Conversely, when user confirms a blurry engine-flagged photo, ReviewService preserves `motion_blur`
        # so it still counts as `tech_reject` with primary reason `motion_blur` (never `user_reject`).
        _seed_photo(
            svcs,
            session_id="s_subj",
            photo_id="blur_confirmed",
            capture_time=ts + 100,
            lens_model="FE 50mm F1.2 GM",
            bucket="red",
            reason_codes=(REASON_MOTION_BLUR,),
        )
        svcs.review.set_verdict("blur_confirmed", bucket="red", source="user")

        report2 = svcs.digest.generate_digest(session_id="s_subj", min_samples=5)
        assert report2.overview.tech_reject_count == 1
        assert report2.overview.subjective_reject_count == 3
        assert len(report2.top_tech_reasons) == 1
        assert report2.top_tech_reasons[0].code == REASON_MOTION_BLUR
        assert REASON_USER_REJECT not in {r.code for r in report2.top_tech_reasons}

