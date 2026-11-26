"""Unit & integration tests for TriageService, ToolRegistry triage, and POST /api/sessions/{id}/triage (PLAN T10)."""

from __future__ import annotations

import asyncio
import io
import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import BaseModel

from lightbox.api import CSRF_HEADER_NAME, SESSION_COOKIE_NAME, create_app
from lightbox.config import load_config
from lightbox.core import ENGINE_VERSION
from lightbox.models import (
    GenerationResult,
    Message,
    ModelCaps,
    ModelRouter,
    ToolSchema,
    Usage,
)
from lightbox.paths import get_app_paths
from lightbox.services import (
    BurstTriageOutput,
    LightboxServices,
    YellowTriageOutput,
)
from lightbox.store import (
    AssetRecord,
    ExifRecord,
    FaceRecord,
    MetricsRecord,
    PhotoRecord,
    RootRecord,
    SessionRecord,
)
from lightbox.tools import ToolContext, ToolRegistry


class _CapturingVLM:
    """Fake VLM that records every prompt & sanitized image bytes sent to it."""

    def __init__(
        self,
        *,
        yellow_decision: str = "unsure",
        cost_per_call: float = 0.02,
    ) -> None:
        self.id = "claude-haiku-5-5"
        self.provider = "anthropic"
        self.family = "anthropic"
        self.caps = ModelCaps(vision=True, tools=True, json_schema=True, prompt_cache=True)
        self.yellow_decision = yellow_decision
        self.cost_per_call = cost_per_call
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        response_schema: type[BaseModel] | None = None,
        max_tokens: int = 2048,
        effort: str | None = None,
    ) -> GenerationResult:
        del tools, max_tokens, effort
        images = [img for msg in messages for img in msg.images]
        texts = [msg.text for msg in messages if msg.text]
        self.calls.append({"texts": texts, "images": images, "schema": response_schema})

        if response_schema is YellowTriageOutput:
            out = YellowTriageOutput(
                decision=self.yellow_decision,  # type: ignore[arg-type]
                confidence=0.82,
                reason_code="vlm_unsure" if self.yellow_decision == "unsure" else "vlm_keep",
                rationale="Subject detail checked in 1:1 crop.",
            )
            txt = out.model_dump_json()
            return GenerationResult(
                text=txt,
                parsed=out,
                usage=Usage(in_tokens=1500, out_tokens=60),
                cost_usd=self.cost_per_call,
                stop_reason="end_turn",
                provider=self.provider,
                model=self.id,
            )

        out_b = BurstTriageOutput(
            pick_frame_no=2,
            confidence=0.91,
            eyes_open=True,
            rationale="Frame #2 has sharp eyes and natural smile.",
        )
        return GenerationResult(
            text=out_b.model_dump_json(),
            parsed=out_b,
            usage=Usage(in_tokens=1400, out_tokens=50),
            cost_usd=self.cost_per_call,
            stop_reason="end_turn",
            provider=self.provider,
            model=self.id,
        )


def _write_exif_jpeg(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (1200, 900)) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, color)
    exif = Image.Exif()
    exif[0x010F] = "Sony"
    exif[0x0110] = "ILCE-7M4"
    exif[0xA431] = "SECRET-SERIAL-123"
    exif[0x8825] = {1: "N", 2: (35.0, 40.0, 0.0)}
    im.save(path, format="JPEG", exif=exif.tobytes(), quality=92)
    return path.read_bytes()


def _seed_triage_session(svcs: LightboxServices, shoot_dir: Path) -> str:
    repo = svcs.repo
    repo.upsert_root(RootRecord(id="root_1", path=str(shoot_dir), device_id="1", kind="disk"))
    repo.upsert_session(
        SessionRecord(
            id="sess_triage",
            root_id="root_1",
            root_path=str(shoot_dir),
            name="portrait_shoot",
            sensitivity="conservative",
            engine_version=ENGINE_VERSION,
            created_at=1000.0,
            last_scanned_at=1001.0,
        )
    )

    # 2 yellow photos (one with 5-pt landmarks face, one without faces using sharpest_cell)
    specs = [
        ("ph_y1", "IMG_SECRET_NAME_01", (200, 120, 80), True),
        ("ph_y2", "IMG_SECRET_NAME_02", (80, 160, 210), False),
        ("ph_b1", "IMG_BURST_01", (90, 90, 200), True),
        ("ph_b2", "IMG_BURST_02", (95, 95, 205), True),
    ]
    for pid, stem, color, has_face in specs:
        p_path = shoot_dir / f"{stem}.jpg"
        raw = _write_exif_jpeg(p_path, color)
        fp = f"fp_{pid}"
        repo.upsert_photo(
            PhotoRecord(
                id=pid,
                session_id="sess_triage",
                stem=stem,
                rel_dir="",
                primary_content_fp=fp,
                cohort="camera",
                capture_time=1700000000.0,
                camera_model="Sony A7M4",
                preview_w=1200,
                preview_h=900,
            )
        )
        repo.upsert_asset(
            AssetRecord(
                photo_id=pid,
                rel_path=f"{stem}.jpg",
                abs_path=str(p_path),
                kind="jpg",
                size_bytes=len(raw),
                mtime=1700000000.0,
                content_fp=fp,
            )
        )
        repo.upsert_exif(
            ExifRecord(
                photo_id=pid,
                aperture=2.8,
                shutter_s=1 / 125,
                iso=400,
                focal_eq=50.0,
                light_bucket="indoor",
            )
        )
        repo.upsert_metrics(
            MetricsRecord(
                content_fp=fp,
                engine_version=ENGINE_VERSION,
                sharp_p95=120.0,
                motion_state="static",
                exposure={"sharpest_cell": [4, 10]},
                face_count=1 if has_face else 0,
                face_sharp_min=85.0 if has_face else None,
            )
        )
        if has_face:
            repo.replace_faces(
                fp,
                [
                    FaceRecord(
                        content_fp=fp,
                        idx=0,
                        bbox=(0.35, 0.25, 0.55, 0.55),
                        landmarks=(
                            (0.41, 0.35),
                            (0.49, 0.35),
                            (0.45, 0.42),
                            (0.42, 0.48),
                            (0.48, 0.48),
                        ),
                        det_score=0.98,
                        sharpness=85.0,
                        area_ratio=0.06,
                    )
                ],
            )
        bucket = "yellow" if pid.startswith("ph_y") else "normal"
        reasons = ["face_out_of_focus"] if pid == "ph_y1" else (["borderline_blur"] if pid == "ph_y2" else [])
        repo.add_verdict(
            photo_id=pid,
            source="engine",
            bucket=bucket,
            confidence=0.45 if bucket == "yellow" else 0.9,
            reason_codes=reasons,
        )

    repo.upsert_burst("burst_1", "sess_triage", kind="burst", confirmed=False)
    repo.upsert_burst_member("burst_1", "ph_b1", rank=1, is_pick=False)
    repo.upsert_burst_member("burst_1", "ph_b2", rank=2, is_pick=False)
    return "sess_triage"


def test_triage_exif_stripped_crops_and_zero_cost_repeat_run(tmp_path: Path) -> None:
    """PLAN T10 acceptance #1 & #2: empty getexif() on all uploads, and $0 repeat triage run."""
    home = tmp_path / "home"
    shoot = tmp_path / "shoot"
    fake = _CapturingVLM(yellow_decision="unsure", cost_per_call=0.03)

    with LightboxServices.open(home) as svcs:
        router = ModelRouter(svcs.repo, svcs.app_paths, svcs.config, adapters={"triage_vlm": fake})
        svcs.override_model_router(router)
        sid = _seed_triage_session(svcs, shoot)

        # Dry run estimate
        est = svcs.triage.estimate_triage(sid)
        assert est.dry_run is True
        assert est.yellow_count == 2
        assert est.burst_count == 1
        assert est.total_units == 3
        assert est.est_usd > 0.0

        # Run 1: pays for 2 yellow + 1 burst = 3 calls ($0.09)
        r1 = asyncio.run(svcs.triage.run_triage(sid))
        assert r1.status == "completed"
        assert r1.reviewed_yellow == 2
        assert r1.reviewed_bursts == 1
        assert r1.paid_calls == 3
        assert r1.cache_hits == 0
        assert r1.spent_usd == pytest.approx(0.09)
        assert len(fake.calls) == 3

        # Acceptance #1: EVERY uploaded image has empty EXIF, proper crop/context dimensions,
        # and prompts contain NO filenames or absolute shoot paths
        for call in fake.calls:
            joined_text = "\n".join(call["texts"])
            assert str(shoot) not in joined_text
            assert "IMG_SECRET_NAME" not in joined_text
            imgs = call["images"]
            assert len(imgs) == 2
            for img in imgs:
                with Image.open(io.BytesIO(img.data)) as im:
                    assert len(im.getexif()) == 0
            if call["schema"] is YellowTriageOutput:
                with Image.open(io.BytesIO(imgs[0].data)) as crop_im:
                    assert 512 <= crop_im.width <= 768
                    assert crop_im.width == crop_im.height
                with Image.open(io.BytesIO(imgs[1].data)) as ctx_im:
                    assert max(ctx_im.size) == 512

        # Verify VLM verdict & burst pick recommendation persisted without auto-confirming burst
        v_y1 = svcs.repo.get_effective_verdict("ph_y1")
        assert v_y1 is not None and v_y1.source == "vlm" and v_y1.llm_call_id is not None
        b_det = svcs.burst.get_burst_detail("burst_1")
        assert b_det.burst.confirmed is False
        picks = [m for m in b_det.members if m.is_pick]
        assert len(picks) == 1 and picks[0].photo_id == "ph_b2"
        assert picks[0].ai_note is not None and "eyes" in picks[0].ai_note.lower()

        cost_before_repeat = svcs.repo.sum_llm_cost(session_id=sid)

        # Acceptance #2: Run 2 on the same batch -> 100% cache hit, $0 additional cost!
        r2 = asyncio.run(svcs.triage.run_triage(sid))
        assert r2.status == "completed"
        assert r2.reviewed_yellow == 2
        assert r2.reviewed_bursts == 1
        assert r2.cache_hits == 3
        assert r2.paid_calls == 0
        assert r2.spent_usd == 0.0
        assert len(fake.calls) == 3  # Model was NOT called again
        assert svcs.repo.sum_llm_cost(session_id=sid) == pytest.approx(cost_before_repeat)


def test_triage_pauses_with_budget_exceeded(tmp_path: Path) -> None:
    """PLAN T10 acceptance #3: Exceeding session or monthly budget pauses with `budget_exceeded`."""
    home = tmp_path / "home"
    shoot = tmp_path / "shoot"
    fake = _CapturingVLM(yellow_decision="keep", cost_per_call=0.04)

    with LightboxServices.open(home) as svcs:
        router = ModelRouter(
            svcs.repo,
            svcs.app_paths,
            svcs.config,
            max_concurrency=1,
            adapters={"triage_vlm": fake},
        )
        svcs.override_model_router(router)
        sid = _seed_triage_session(svcs, shoot)

        # Budget only allows 1 call ($0.04) out of 3 units
        res = asyncio.run(svcs.triage.run_triage(sid, session_budget_usd=0.04))
        assert res.status == "paused"
        assert res.paused_reason == "budget_exceeded"
        assert res.paid_calls == 1
        job = svcs.repo.get_job(res.job_id)
        assert job is not None
        assert job.status == "paused"
        assert job.paused_reason == "budget_exceeded"

    # Also test monthly budget limit
    home_m = tmp_path / "home_monthly"
    shoot_m = tmp_path / "shoot_monthly"
    paths_m = get_app_paths(home_m)
    paths_m.ensure_dirs()
    paths_m.config_file.write_text(
        "[budget]\nsession_budget_usd = 5.0\nmonthly_budget_usd = 0.0001\n",
        encoding="utf-8",
    )
    cfg_m = load_config(paths_m)
    with LightboxServices(app_paths=paths_m, config=cfg_m) as svcs_m:
        router_m = ModelRouter(
            svcs_m.repo,
            svcs_m.app_paths,
            cfg_m,
            max_concurrency=1,
            adapters={"triage_vlm": fake},
        )
        svcs_m.override_model_router(router_m)
        sid_m = _seed_triage_session(svcs_m, shoot_m)
        res_m = asyncio.run(svcs_m.triage.run_triage(sid_m))
        assert res_m.status == "paused"
        assert res_m.paused_reason == "budget_exceeded"


def test_tool_registry_and_fastapi_triage_endpoints(tmp_path: Path) -> None:
    """Verify ToolRegistry `lightbox_triage_uncertain` and `POST /api/sessions/{id}/triage`."""
    home = tmp_path / "api_home"
    shoot = tmp_path / "api_shoot"
    fake = _CapturingVLM(yellow_decision="keep", cost_per_call=0.01)

    with LightboxServices.open(home) as svcs:
        router = ModelRouter(svcs.repo, svcs.app_paths, svcs.config, adapters={"triage_vlm": fake})
        svcs.override_model_router(router)
        sid = _seed_triage_session(svcs, shoot)

        reg = ToolRegistry()
        ctx = ToolContext(services=svcs, surface="mcp:test")

        dry = reg.execute("lightbox_triage_uncertain", {"session_id": sid, "dry_run": True}, ctx)
        assert dry.is_error is False
        assert dry.data["dry_run"] is True
        assert dry.data["count"] == 3

        live = reg.execute("lightbox_triage_uncertain", {"session_id": sid, "dry_run": False}, ctx)
        assert live.is_error is False
        assert live.data["status"] == "completed"
        assert live.data["reviewed_yellow"] == 2
        assert live.data["reviewed_bursts"] == 1

    # Re-seed a fresh session for FastAPI route test
    home2 = tmp_path / "web_home"
    shoot2 = tmp_path / "web_shoot"
    with LightboxServices.open(home2) as svcs2:
        sid2 = _seed_triage_session(svcs2, shoot2)

    fake2 = _CapturingVLM(yellow_decision="unsure", cost_per_call=0.02)
    app = create_app(
        app_paths=get_app_paths(home2),
        port=8765,
        bootstrap_token="tok-t10",
        model_router_factory=lambda s: ModelRouter(
            s.repo, s.app_paths, s.config, adapters={"triage_vlm": fake2}
        ),
    )
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    assert client.post(f"/api/sessions/{sid2}/triage", json={"dry_run": True}).status_code == 401

    launch = client.get("/auth/launch?token=tok-t10", follow_redirects=False)
    assert launch.status_code == 302 and SESSION_COOKIE_NAME in launch.headers.get("set-cookie", "")
    csrf = str(client.get("/api/auth/me").json()["csrf_token"])
    headers = {CSRF_HEADER_NAME: csrf, "Origin": "http://127.0.0.1:8765"}

    r_dry = client.post(f"/api/sessions/{sid2}/triage", json={"dry_run": True}, headers=headers)
    assert r_dry.status_code == 200
    assert r_dry.json()["total_units"] == 3
    assert r_dry.json()["skipped_bursts_due_to_limit"] == 0

    # wait=True runs off the FastAPI event loop via anyio.to_thread.run_sync
    r_live = client.post(
        f"/api/sessions/{sid2}/triage",
        json={"dry_run": False, "wait": True},
        headers=headers,
    )
    assert r_live.status_code == 200
    body = r_live.json()
    assert body["status"] == "completed"
    assert body["reviewed_yellow"] == 2
    assert body["reviewed_bursts"] == 1
    assert body["failed_count"] == 0
    assert json.loads(json.dumps(body))["paid_calls"] == 3

    # wait=False (default) returns immediately with queued job_id and completes in background thread
    r_bg = client.post(
        f"/api/sessions/{sid2}/triage",
        json={"dry_run": False},
        headers=headers,
    )
    assert r_bg.status_code == 200
    bg_body = r_bg.json()
    assert bg_body["status"] == "queued"
    bg_job_id = bg_body["job_id"]
    deadline = time.monotonic() + 5.0
    final_status = "queued"
    while time.monotonic() < deadline:
        jr = client.get(f"/api/jobs/{bg_job_id}")
        assert jr.status_code == 200
        final_status = jr.json()["status"]
        if final_status != "queued" and final_status != "running":
            break
        time.sleep(0.05)
    assert final_status == "completed"

    # Unknown session still returns 404
    r_404 = client.post(
        "/api/sessions/does_not_exist/triage",
        json={"dry_run": False, "wait": True},
        headers=headers,
    )
    assert r_404.status_code == 404


def test_persistence_error_partial_failure_and_burst_limit_notice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify post-call persistence errors never stick job in `running` or return 404, and partial errors report `completed_with_errors`."""
    home = tmp_path / "err_home"
    shoot = tmp_path / "err_shoot"
    fake = _CapturingVLM(yellow_decision="keep", cost_per_call=0.02)

    with LightboxServices.open(home) as svcs:
        router = ModelRouter(svcs.repo, svcs.app_paths, svcs.config, adapters={"triage_vlm": fake})
        svcs.override_model_router(router)
        sid = _seed_triage_session(svcs, shoot)

        # Dry-run with max_photos=2 fills with 2 yellow photos and explicitly reports 1 deferred burst
        est_limited = svcs.triage.estimate_triage(sid, max_photos=2)
        assert est_limited.yellow_count == 2
        assert est_limited.burst_count == 0
        assert est_limited.total_bursts_available == 1
        assert est_limited.skipped_bursts_due_to_limit == 1
        assert est_limited.note is not None and "deferred" in est_limited.note

        # Simulate burst deleted/mutated mid-run after paid VLM call returned
        monkeypatch.setattr(
            svcs.burst,
            "recommend_burst_pick",
            lambda *_a, **_kw: (_ for _ in ()).throw(ValueError("Unknown burst_id: burst_1")),
        )

        res = asyncio.run(svcs.triage.run_triage(sid))
        assert res.status == "completed_with_errors"
        assert res.reviewed_yellow == 2
        assert res.reviewed_bursts == 0
        assert res.failed_bursts == 1
        assert res.failed_count == 1
        assert res.paid_calls == 3
        assert res.spent_usd == pytest.approx(0.06)

        job = svcs.repo.get_job(res.job_id)
        assert job is not None
        assert job.status == "completed_with_errors"
        assert job.spent_usd == pytest.approx(0.06)
        burst_item = svcs.repo.get_job_item(res.job_id, "burst_1", "triage_burst")
        assert burst_item is not None and burst_item.status == "failed"
        assert "Unknown burst_id" in (burst_item.last_error or "")

    # Also verify via FastAPI wait=True that mid-run ValueError returns 200 completed_with_errors (NOT 404!)
    home_api = tmp_path / "err_api_home"
    shoot_api = tmp_path / "err_api_shoot"
    with LightboxServices.open(home_api) as svcs_api:
        sid_api = _seed_triage_session(svcs_api, shoot_api)

    def _router_with_broken_burst(s: LightboxServices) -> ModelRouter:
        s.burst.recommend_burst_pick = lambda *_a, **_kw: (_ for _ in ()).throw(  # type: ignore[method-assign]
            ValueError("Burst removed mid-run")
        )
        return ModelRouter(
            s.repo,
            s.app_paths,
            s.config,
            adapters={"triage_vlm": _CapturingVLM(yellow_decision="keep", cost_per_call=0.01)},
        )

    app = create_app(
        app_paths=get_app_paths(home_api),
        port=8765,
        bootstrap_token="tok-err",
        model_router_factory=_router_with_broken_burst,
    )
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    client.get("/auth/launch?token=tok-err", follow_redirects=False)
    csrf = str(client.get("/api/auth/me").json()["csrf_token"])
    resp = client.post(
        f"/api/sessions/{sid_api}/triage",
        json={"dry_run": False, "wait": True},
        headers={CSRF_HEADER_NAME: csrf, "Origin": "http://127.0.0.1:8765"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "completed_with_errors"
    assert payload["failed_count"] == 1
    assert payload["reviewed_yellow"] == 2

