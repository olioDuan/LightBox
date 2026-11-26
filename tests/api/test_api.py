"""T07 unit & integration tests: FastAPI local auth, DNS rebinding defense, CSRF, REST e2e, SSE, and server.lock."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lightbox.api import (
    CSRF_HEADER_NAME,
    LOCAL_CONFIRM_HEADER,
    SESSION_COOKIE_NAME,
    clear_server_lock,
    create_app,
    find_free_local_port,
    read_server_lock,
    write_server_lock,
)
from lightbox.paths import get_app_paths
from lightbox.services import build_demo_dataset, restore_dataset_write_bits

TEST_PORT = 8765
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"


def _bootstrap_authenticated_client(
    tmp_path: Path,
    *,
    token: str = "one-time-secret-token",
) -> tuple[TestClient, str]:
    paths = get_app_paths(tmp_path / "app_home")
    app = create_app(app_paths=paths, port=TEST_PORT, bootstrap_token=token)
    client = TestClient(app, base_url=BASE_URL)

    # One-time launch redirect sets HttpOnly + SameSite=Strict cookie
    launch = client.get(f"/auth/launch?token={token}", follow_redirects=False)
    assert launch.status_code == 302
    assert launch.headers["location"] == "/"
    set_cookie = launch.headers.get("set-cookie", "")
    assert SESSION_COOKIE_NAME in set_cookie
    assert "httponly" in set_cookie.lower()
    assert "samesite=strict" in set_cookie.lower()

    # Fetch CSRF token via authenticated /api/auth/me
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    csrf_token = str(me.json()["csrf_token"])
    assert len(csrf_token) >= 20
    return client, csrf_token


def test_security_gates_401_cookie_403_csrf_and_403_dns_rebinding(tmp_path: Path) -> None:
    """PLAN T07 acceptance: no cookie -> 401, missing CSRF -> 403, forged Host/Origin -> 403, no CORS."""
    token = "launch-tok-123"
    paths = get_app_paths(tmp_path / "sec_home")
    app = create_app(app_paths=paths, port=TEST_PORT, bootstrap_token=token)
    client = TestClient(app, base_url=BASE_URL)

    # 1. No cookie on image & read endpoints -> 401
    r_img = client.get("/api/photos/ph_any/image")
    assert r_img.status_code == 401
    assert "access-control-allow-origin" not in r_img.headers
    assert client.get("/api/sessions").status_code == 401
    assert client.get("/api/events?once=true").status_code == 401

    # 2. Forged Host header (DNS rebinding) -> 403
    assert client.get("/api/sessions", headers={"Host": "evil.com"}).status_code == 403
    assert client.get("/api/sessions", headers={"Host": f"localhost:{TEST_PORT}"}).status_code == 403
    assert client.get("/api/sessions", headers={"Host": "127.0.0.1:9999"}).status_code == 403

    # 3. Forged Origin header -> 403
    assert (
        client.get("/api/sessions", headers={"Origin": "https://evil.example"}).status_code == 403
    )
    assert (
        client.get("/api/sessions", headers={"Origin": f"http://localhost:{TEST_PORT}"}).status_code
        == 403
    )
    assert client.get("/api/sessions", headers={"Origin": "null"}).status_code == 403

    # 4. One-time launch token succeeds once and rejects replay with 401
    ok_launch = client.get(f"/auth/launch?token={token}", follow_redirects=False)
    assert ok_launch.status_code == 302
    replay_launch = client.get(f"/auth/launch?token={token}", follow_redirects=False)
    assert replay_launch.status_code == 401

    # 5. Authenticated Cookie WITHOUT X-Lightbox-CSRF on write endpoints -> 403
    assert (
        client.post("/api/verdicts", json={"photo_ids": ["p1"], "bucket": "normal"}).status_code
        == 403
    )
    assert client.put("/api/settings", json={"language": "en-US"}).status_code == 403
    assert (
        client.post(
            "/api/verdicts",
            json={"photo_ids": ["p1"], "bucket": "normal"},
            headers={CSRF_HEADER_NAME: "wrong-csrf"},
        ).status_code
        == 403
    )

    # 6. Local SameSite=Strict fallback /api/auth/session requires Origin + custom confirm header
    anon = TestClient(app, base_url=BASE_URL)
    assert anon.post("/api/auth/session").status_code == 403
    local_ok = anon.post(
        "/api/auth/session",
        headers={"Origin": BASE_URL, LOCAL_CONFIRM_HEADER: "1"},
    )
    assert local_ok.status_code == 200
    assert local_ok.json()["authenticated"] is True


def test_full_authenticated_workflow_scan_review_burst_plan_apply_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PLAN T07 acceptance: full e2e scan -> image -> verdict/undo -> burst -> draft -> apply -> restore."""
    shoot = tmp_path / "api_shoot"
    build_demo_dataset(shoot, seed=42)
    client, csrf = _bootstrap_authenticated_client(tmp_path)
    headers = {CSRF_HEADER_NAME: csrf, "Origin": BASE_URL}

    try:
        # 1. Scan folder
        scan_resp = client.post(
            "/api/sessions/scan",
            json={"folder_path": str(shoot), "sensitivity": "conservative", "wait": True, "max_workers": 1},
            headers=headers,
        )
        assert scan_resp.status_code == 200, scan_resp.text
        scan_data = scan_resp.json()
        session_id = scan_data["session_id"]
        job_id = scan_data["job_id"]
        assert scan_data["total_photos"] == 18
        assert scan_data["counts_by_bucket"]["red"] >= 4

        # 2. Sessions list, summary, job status, pause/resume
        assert any(s["id"] == session_id for s in client.get("/api/sessions").json()["items"])
        summary = client.get(f"/api/sessions/{session_id}/summary").json()
        assert summary["total_photos"] == 18
        assert summary["burst_count"] >= 1
        assert client.get(f"/api/jobs/{job_id}").json()["status"] == "completed"
        assert client.post(f"/api/jobs/{job_id}/pause", headers=headers).status_code == 200
        resume_resp = client.post(f"/api/jobs/{job_id}/resume", headers=headers)
        assert resume_resp.status_code == 200
        assert resume_resp.json()["cache_hits"] == 18

        # 3. Photos list & WebP image endpoint (grid, loupe, crop)
        photos_resp = client.get(f"/api/sessions/{session_id}/photos?limit=50")
        assert photos_resp.status_code == 200
        photos = photos_resp.json()["items"]
        assert len(photos) == 18
        first_pid = photos[0]["photo_id"]

        for query in ("size=grid", "size=loupe", "size=crop&region=sharpest"):
            img_r = client.get(f"/api/photos/{first_pid}/image?{query}")
            assert img_r.status_code == 200
            assert img_r.headers["content-type"] == "image/webp"
            assert img_r.content[:4] == b"RIFF" and img_r.content[8:12] == b"WEBP"

        # 4. Verdict override & undo
        normal_photo = next(p for p in photos if p["bucket"] == "normal")
        v_resp = client.post(
            "/api/verdicts",
            json={
                "photo_ids": [normal_photo["photo_id"]],
                "bucket": "yellow",
                "reason_tag": "check_later",
            },
            headers=headers,
        )
        assert v_resp.status_code == 200
        assert v_resp.json()["updated"][0]["bucket"] == "yellow"

        undo_resp = client.post(
            "/api/verdicts/undo",
            json={"photo_id": normal_photo["photo_id"]},
            headers=headers,
        )
        assert undo_resp.status_code == 200
        assert undo_resp.json()["undone"] is True
        assert undo_resp.json()["bucket"] == "normal"

        # 5. Bursts list & confirm
        bursts = client.get(f"/api/sessions/{session_id}/bursts").json()["items"]
        assert len(bursts) >= 1
        b0 = bursts[0]
        pick_id = b0["members"][0]["photo_id"]
        conf_resp = client.post(
            f"/api/bursts/{b0['id']}/confirm",
            json={"pick_photo_id": pick_id},
            headers=headers,
        )
        assert conf_resp.status_code == 200
        assert conf_resp.json()["confirmed"] is True

        # 6. Draft obvious rejects (one-click fast path) -> returns tech vs burst breakdown
        draft_resp = client.post(
            f"/api/sessions/{session_id}/plans/obvious-rejects",
            json={"strategy": "sibling"},
            headers=headers,
        )
        assert draft_resp.status_code == 200, draft_resp.text
        draft_data = draft_resp.json()
        plan = draft_data["plan"]
        assert plan is not None
        assert draft_data["tech_reject_count"] >= 4
        assert draft_data["burst_non_pick_count"] >= 2
        plan_id = plan["id"]
        plan_hash = plan["plan_hash"]

        # Verify plan retrieval
        assert client.get(f"/api/plans/{plan_id}").json()["plan_hash"] == plan_hash

        # 7. Wrong plan_hash -> 409 Conflict; valid plan_hash -> approve-and-apply moves files
        bad_apply = client.post(
            f"/api/plans/{plan_id}/approve-and-apply",
            json={"plan_hash": "deadbeef"},
            headers=headers,
        )
        assert bad_apply.status_code == 409

        apply_resp = client.post(
            f"/api/plans/{plan_id}/approve-and-apply",
            json={"plan_hash": plan_hash},
            headers=headers,
        )
        assert apply_resp.status_code == 200, apply_resp.text
        apply_data = apply_resp.json()
        assert apply_data["status"] == "applied"
        assert apply_data["moved_photo_count"] == draft_data["selected_count"]
        assert Path(apply_data["manifest_path"]).is_file()
        assert Path(apply_data["readme_path"]).is_file()

        # 8. Restore plan -> all original files returned intact
        restore_resp = client.post(
            f"/api/plans/{plan_id}/restore",
            json={},
            headers=headers,
        )
        assert restore_resp.status_code == 200
        assert restore_resp.json()["plan_status"] == "reverted"

        # 9. Settings GET/PUT (rejects unknown plaintext api_key!) + pick-folder + SSE stream
        assert client.get("/api/settings").json()["language"] == "zh-CN"
        put_ok = client.put("/api/settings", json={"language": "en-US"}, headers=headers)
        assert put_ok.status_code == 200 and put_ok.json()["language"] == "en-US"
        put_bad = client.put("/api/settings", json={"anthropic_api_key": "sk-secret"}, headers=headers)
        assert put_bad.status_code == 422

        monkeypatch.setattr("lightbox.services.system.SystemService.pick_folder", lambda _self: str(shoot))
        pick_r = client.post("/api/system/pick-folder", headers=headers)
        assert pick_r.status_code == 200 and pick_r.json()["path"] == str(shoot)

        sse_r = client.get(f"/api/events?session_id={session_id}&once=true")
        assert sse_r.status_code == 200
        assert "text/event-stream" in sse_r.headers["content-type"]
        assert "event: connected" in sse_r.text
    finally:
        restore_dataset_write_bits(shoot)


def test_demo_bootstrap_endpoint_and_server_lock_lifecycle(tmp_path: Path) -> None:
    client, csrf = _bootstrap_authenticated_client(tmp_path)
    headers = {CSRF_HEADER_NAME: csrf, "Origin": BASE_URL}
    demo_dir = tmp_path / "api_demo_shoot"

    try:
        resp = client.post(
            "/api/demo/bootstrap",
            json={"target_dir": str(demo_dir)},
            headers=headers,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["generated_photos"] == 18
        assert data["counts_by_bucket"]["red"] >= 4
    finally:
        restore_dataset_write_bits(demo_dir)

    # Server lock write / read / recycled-PID / unrelated-HTTP / live-healthz verification
    import http.server
    import json
    import threading

    lock_paths = get_app_paths(tmp_path / "lock_home")
    my_pid = os.getpid()

    # 1. Live Lightbox `/healthz` responder -> matches pid & instance_id; mismatched instance_id fails
    class _LightboxHealthHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/healthz":
                payload = json.dumps(
                    {"service": "lightbox", "pid": my_pid, "instance_id": "inst_live"}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, _fmt: str, *_args: object) -> None:
            pass

    lb_srv = http.server.HTTPServer(("127.0.0.1", 0), _LightboxHealthHandler)
    lb_port = int(lb_srv.server_address[1])
    t_lb = threading.Thread(target=lb_srv.serve_forever, daemon=True)
    t_lb.start()
    try:
        written = write_server_lock(
            lb_port, app_paths=lock_paths, pid=my_pid, instance_id="inst_live"
        )
        assert read_server_lock(app_paths=lock_paths, check_alive=True) == written

        # Mismatched instance_id (another instance took the port) -> treated as stale
        write_server_lock(lb_port, app_paths=lock_paths, pid=my_pid, instance_id="inst_old")
        assert read_server_lock(app_paths=lock_paths, check_alive=True) is None
    finally:
        lb_srv.shutdown()
        lb_srv.server_close()

    # 2. Alive (recycled) PID but nothing listening on the port -> stale lock
    write_server_lock(lb_port, app_paths=lock_paths, pid=my_pid, instance_id="inst_live")
    assert read_server_lock(app_paths=lock_paths, check_alive=True) is None

    # 3. Alive PID + unrelated HTTP service on the port -> still treated as stale lock
    class _OtherServerHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"service": "other_app"}')

        def log_message(self, _fmt: str, *_args: object) -> None:
            pass

    other_srv = http.server.HTTPServer(("127.0.0.1", 0), _OtherServerHandler)
    other_port = int(other_srv.server_address[1])
    t_other = threading.Thread(target=other_srv.serve_forever, daemon=True)
    t_other.start()
    try:
        write_server_lock(other_port, app_paths=lock_paths, pid=my_pid, instance_id="inst_x")
        assert read_server_lock(app_paths=lock_paths, check_alive=True) is None
    finally:
        other_srv.shutdown()
        other_srv.server_close()

    # 4. Dead PID -> stale lock; clear_server_lock stops without deleting file
    port = find_free_local_port()
    write_server_lock(port, app_paths=lock_paths, pid=999_999_999)
    assert read_server_lock(app_paths=lock_paths, check_alive=True) is None

    write_server_lock(port, app_paths=lock_paths, pid=my_pid)
    clear_server_lock(app_paths=lock_paths, only_if_pid=my_pid)
    assert read_server_lock(app_paths=lock_paths, check_alive=False) is None

    # 5. FastAPI `/healthz` returns expected identity and enforces Host check
    hz = client.get("/healthz")
    assert hz.status_code == 200
    assert hz.json()["service"] == "lightbox"
    assert hz.json()["pid"] == my_pid
    assert client.get("/healthz", headers={"Host": "evil.com"}).status_code == 403


def test_stale_plan_returns_409_and_event_bus_filters_by_session(tmp_path: Path) -> None:
    """Verify concurrent verdict override causes 409 Conflict on approve-and-apply, and SSE filters."""
    import asyncio

    from lightbox.api import EventBus

    shoot = tmp_path / "stale_shoot"
    build_demo_dataset(shoot, seed=91)
    client, csrf = _bootstrap_authenticated_client(tmp_path)
    headers = {CSRF_HEADER_NAME: csrf, "Origin": BASE_URL}

    try:
        scan_data = client.post(
            "/api/sessions/scan",
            json={"folder_path": str(shoot), "wait": True, "max_workers": 1},
            headers=headers,
        ).json()
        sid = scan_data["session_id"]

        draft = client.post(
            f"/api/sessions/{sid}/plans/obvious-rejects",
            json={"strategy": "sibling"},
            headers=headers,
        ).json()
        plan = draft["plan"]
        first_pid = plan["items"][0]["photo_id"]

        # Concurrent user rescue bumps decision_version -> plan is now stale!
        assert (
            client.post(
                "/api/verdicts",
                json={"photo_ids": [first_pid], "bucket": "normal"},
                headers=headers,
            ).status_code
            == 200
        )
        stale_resp = client.post(
            f"/api/plans/{plan['id']}/approve-and-apply",
            json={"plan_hash": plan["plan_hash"]},
            headers=headers,
        )
        assert stale_resp.status_code == 409
    finally:
        restore_dataset_write_bits(shoot)

    # Verify EventBus SSE session_id filtering
    async def _check_bus() -> list[str]:
        bus = EventBus()
        gen = bus.subscribe(session_id="sess_A", heartbeat_s=1.0)
        first = await anext(gen)
        bus.publish("plan.updated", {"plan_id": "other"}, session_id="sess_B")
        bus.publish("plan.updated", {"plan_id": "mine"}, session_id="sess_A")
        second = await anext(gen)
        await gen.aclose()
        return [first, second]

    chunks = asyncio.run(_check_bus())
    assert "event: connected" in chunks[0]
    assert "event: plan.updated" in chunks[1]
    assert '"plan_id": "mine"' in chunks[1]

    # Verify EventBus.close() terminates open SSE streams immediately (so Ctrl+C never hangs)
    async def _check_bus_close() -> list[str]:
        bus = EventBus()
        collected: list[str] = []

        async def _reader() -> None:
            async for chunk in bus.subscribe(heartbeat_s=60.0):
                collected.append(chunk)

        task = asyncio.create_task(_reader())
        await asyncio.sleep(0.05)
        bus.close()
        await asyncio.wait_for(task, timeout=1.0)
        return collected

    closed_chunks = asyncio.run(_check_bus_close())
    assert len(closed_chunks) == 1 and "event: connected" in closed_chunks[0]


def test_review_excludes_quarantined_and_tracks_progress_and_manual_burst_reject(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify quarantined photos vanish from active review, progress counts update, and manual red beats unconfirmed burst."""
    shoot = tmp_path / "ux_shoot"
    build_demo_dataset(shoot, seed=77)
    client, csrf = _bootstrap_authenticated_client(tmp_path)
    headers = {CSRF_HEADER_NAME: csrf, "Origin": BASE_URL}

    try:
        scan_data = client.post(
            "/api/sessions/scan",
            json={"folder_path": str(shoot), "wait": True, "max_workers": 1},
            headers=headers,
        ).json()
        sid = scan_data["session_id"]

        # 1. Initial summary before any user decisions or quarantine
        # Auto-flagged red photos also appear in `待选` (`unreviewed`) until user confirms or spares them
        s0 = client.get(f"/api/sessions/{sid}/summary").json()
        assert s0["total_photos"] == 18
        assert s0["active_total_photos"] == 18
        assert s0["quarantined_count"] == 0
        assert s0["reviewed_count"] == 0  # in-camera protected is in `exempt`, not user-reviewed progress
        initial_actionable = s0["actionable_reject_count"]
        assert initial_actionable >= 4
        assert (
            s0["reviewed_count"]
            + s0["unreviewed_count"]
            + s0["active_counts_by_bucket"]["yellow"]
            + s0["active_counts_by_bucket"]["exempt"]
            == s0["active_total_photos"]
        )

        # 2. Quarantine obvious rejects -> draft items include per-photo stem & reason_labels
        draft1 = client.post(
            f"/api/sessions/{sid}/plans/obvious-rejects?lang=zh-CN",
            json={"strategy": "sibling"},
            headers=headers,
        ).json()
        plan1 = draft1["plan"]
        assert plan1 is not None
        assert draft1["selected_count"] == initial_actionable
        assert all(item.get("stem") and item.get("reason_labels") for item in plan1["items"])

        apply1 = client.post(
            f"/api/plans/{plan1['id']}/approve-and-apply",
            json={"plan_hash": plan1["plan_hash"]},
            headers=headers,
        )
        assert apply1.status_code == 200, apply1.text
        moved_count = apply1.json()["moved_photo_count"]
        assert moved_count == initial_actionable

        active_photos = client.get(f"/api/sessions/{sid}/photos?limit=50&sort=name").json()["items"]
        all_photos = client.get(
            f"/api/sessions/{sid}/photos?limit=50&sort=name&include_quarantined=true"
        ).json()["items"]
        assert len(active_photos) == 18 - moved_count
        assert len(all_photos) == 18
        # Permanent seq_no stays 1..18 across full session even after some photos are moved out
        assert [p["seq_no"] for p in all_photos] == list(range(1, 19))
        all_seq_map = {p["photo_id"]: p["seq_no"] for p in all_photos}
        assert all(p["seq_no"] == all_seq_map[p["photo_id"]] for p in active_photos)
        quarantined_ids = {item["photo_id"] for item in plan1["items"]}
        assert quarantined_ids.isdisjoint({p["photo_id"] for p in active_photos})

        s1 = client.get(f"/api/sessions/{sid}/summary").json()
        assert s1["quarantined_count"] == moved_count
        assert s1["active_total_photos"] == 18 - moved_count
        assert s1["actionable_reject_count"] == 0
        assert (
            s1["reviewed_count"]
            + s1["unreviewed_count"]
            + s1["active_counts_by_bucket"]["yellow"]
            + s1["active_counts_by_bucket"]["exempt"]
            == 18 - moved_count
        )

        # 3. Manually mark a photo from an unconfirmed burst as reject (red) + mark another unreviewed photo as kept (normal)
        bursts = client.get(f"/api/sessions/{sid}/bursts").json()["items"]
        assert len(bursts) >= 1
        unconfirmed_burst = bursts[0]
        assert unconfirmed_burst["confirmed"] is False
        burst_member_id = unconfirmed_burst["members"][0]["photo_id"]
        other_active = next(
            p["photo_id"]
            for p in active_photos
            if p["photo_id"] != burst_member_id and not p["is_protected"] and p["bucket"] == "normal"
        )

        assert (
            client.post(
                "/api/verdicts",
                json={"photo_ids": [burst_member_id], "bucket": "red"},
                headers=headers,
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/verdicts",
                json={"photo_ids": [other_active], "bucket": "normal"},
                headers=headers,
            ).status_code
            == 200
        )

        s2 = client.get(f"/api/sessions/{sid}/summary").json()
        assert s2["reviewed_count"] == s1["reviewed_count"] + 1
        assert s2["unreviewed_count"] == s1["unreviewed_count"] - 2
        assert s2["actionable_reject_count"] == 1
        assert (
            s2["reviewed_count"]
            + s2["unreviewed_count"]
            + s2["active_counts_by_bucket"]["yellow"]
            + s2["active_counts_by_bucket"]["red"]
            + s2["active_counts_by_bucket"]["exempt"]
            == s2["active_total_photos"]
        )

        # Verify mutually-exclusive tab filters partition active_total_photos 1:1
        unrev_items = client.get(
            f"/api/sessions/{sid}/photos?limit=50&sort=name&bucket=unreviewed"
        ).json()["items"]
        kept_items = client.get(
            f"/api/sessions/{sid}/photos?limit=50&sort=name&bucket=kept"
        ).json()["items"]
        red_items = client.get(
            f"/api/sessions/{sid}/photos?limit=50&sort=name&bucket=red"
        ).json()["items"]
        yellow_items = client.get(
            f"/api/sessions/{sid}/photos?limit=50&sort=name&bucket=yellow"
        ).json()["items"]
        exempt_items = client.get(
            f"/api/sessions/{sid}/photos?limit=50&sort=name&bucket=exempt"
        ).json()["items"]
        assert len(unrev_items) == s2["unreviewed_count"]
        assert len(kept_items) == s2["reviewed_count"]
        assert (
            len(unrev_items)
            + len(kept_items)
            + len(red_items)
            + len(yellow_items)
            + len(exempt_items)
            == s2["active_total_photos"]
        )

        # Undoing a user-kept photo returns it to unreviewed
        undo_kept = client.post(
            "/api/verdicts/undo",
            json={"photo_id": other_active},
            headers=headers,
        )
        assert undo_kept.status_code == 200 and undo_kept.json()["undone"] is True
        s2_after_undo = client.get(f"/api/sessions/{sid}/summary").json()
        assert s2_after_undo["unreviewed_count"] == s2["unreviewed_count"] + 1
        # Re-mark kept so subsequent assertions remain identical
        assert (
            client.post(
                "/api/verdicts",
                json={"photo_ids": [other_active], "bucket": "normal"},
                headers=headers,
            ).status_code
            == 200
        )

        # Draft obvious rejects now picks up that 1 user-marked red photo even though its burst is unconfirmed
        draft2 = client.post(
            f"/api/sessions/{sid}/plans/obvious-rejects",
            json={"strategy": "sibling"},
            headers=headers,
        ).json()
        assert draft2["selected_count"] == 1
        assert draft2["manual_reject_count"] == 1
        assert draft2["plan"]["items"][0]["photo_id"] == burst_member_id

        # Open trash folder & open shoot folder endpoints
        opened_paths: list[Path] = []
        monkeypatch.setattr(
            "lightbox.services.system.SystemService.open_folder",
            lambda _self, folder: (opened_paths.append(folder) or True),
        )
        open_r = client.post(f"/api/plans/{plan1['id']}/open-folder", headers=headers)
        assert open_r.status_code == 200 and open_r.json()["opened"] is True
        open_sess_r = client.post(f"/api/sessions/{sid}/open-folder", headers=headers)
        assert open_sess_r.status_code == 200 and open_sess_r.json()["opened"] is True
        assert len(opened_paths) == 2 and opened_paths[1] == shoot.resolve()

        # Draft & apply second batch directly into `<shoot>/待删除废片/` via strategy='subfolder'
        draft_sub = client.post(
            f"/api/sessions/{sid}/plans/obvious-rejects",
            json={"strategy": "subfolder"},
            headers=headers,
        ).json()
        plan_sub = draft_sub["plan"]
        assert plan_sub is not None
        assert Path(plan_sub["quarantine_dir"]) == (shoot / "待删除废片").resolve()
        apply_sub = client.post(
            f"/api/plans/{plan_sub['id']}/approve-and-apply",
            json={"plan_hash": plan_sub["plan_hash"]},
            headers=headers,
        )
        assert apply_sub.status_code == 200, apply_sub.text
        assert (shoot / "待删除废片").is_dir()

        # Re-scanning the shoot folder must skip `<shoot>/待删除废片/`
        rescan = client.post(
            "/api/sessions/scan",
            json={"folder_path": str(shoot), "wait": True, "max_workers": 1},
            headers=headers,
        ).json()
        assert rescan["total_photos"] == 18 - moved_count - 1

        # Restore subfolder batch cleanly
        restore_sub = client.post(
            f"/api/plans/{plan_sub['id']}/restore",
            json={},
            headers=headers,
        )
        assert restore_sub.status_code == 200 and restore_sub.json()["plan_status"] == "reverted"

        # Plans list includes human-readable session_name
        plans_list = client.get("/api/plans").json()["items"]
        assert any(p["id"] == plan1["id"] and p["session_name"] == shoot.name for p in plans_list)

        # 4. Restore plan1 even when a same-named file already exists back in shoot folder
        # (reproduces re-running demo / copying original back: must NOT HTTP 500 and must NOT overwrite!)
        first_src = Path(plan1["items"][0]["src_paths"][0])
        first_src.write_bytes(b"existing-new-file-must-not-be-overwritten")

        restore1 = client.post(
            f"/api/plans/{plan1['id']}/restore",
            json={},
            headers=headers,
        )
        assert restore1.status_code == 200, restore1.text
        assert restore1.json()["plan_status"] == "reverted"
        # Existing file at src_p was NOT overwritten; restored file got safe collision suffix
        assert first_src.read_bytes() == b"existing-new-file-must-not-be-overwritten"
        assert first_src.with_name(f"{first_src.stem}__1{first_src.suffix}").is_file()

        # Restored photos return to active list as kept (not immediately re-queued as rejects!)
        s3 = client.get(f"/api/sessions/{sid}/summary").json()
        assert s3["quarantined_count"] == 0
        assert s3["active_total_photos"] == 18
        assert s3["actionable_reject_count"] == 0

        # Reset review returns all non-protected active photos back to `unreviewed` (`待选`)
        # and resets user-reviewed progress cleanly to 0 (protected stays in `exempt`)
        reset_r = client.post(f"/api/sessions/{sid}/reset-review", headers=headers)
        assert reset_r.status_code == 200 and reset_r.json()["reset_count"] >= 1
        s4 = client.get(f"/api/sessions/{sid}/summary").json()
        assert s4["unreviewed_count"] == 17  # 18 minus 1 in-camera protected photo
        assert s4["reviewed_count"] == 0
        assert s4["active_counts_by_bucket"]["exempt"] == 1
        assert s4["actionable_reject_count"] == 0

        # Auto-screen immediately re-flags obvious rejects (`red`) and borderline (`yellow`) after reset!
        screen_r = client.post(f"/api/sessions/{sid}/auto-screen", headers=headers)
        assert screen_r.status_code == 200, screen_r.text
        screen_data = screen_r.json()
        assert screen_data["flagged_red"] >= 4
        s5 = client.get(f"/api/sessions/{sid}/summary").json()
        assert s5["actionable_reject_count"] == screen_data["flagged_red"]
        assert s5["active_counts_by_bucket"]["yellow"] == screen_data["flagged_yellow"]

        # Subset selection in QuarantineModal: create & apply a plan for just 2 of the auto-flagged rejects
        draft_all = client.post(
            f"/api/sessions/{sid}/plans/obvious-rejects?lang=zh-CN",
            json={"strategy": "subfolder"},
            headers=headers,
        ).json()
        subset_ids = [it["photo_id"] for it in draft_all["plan"]["items"][:2]]
        sub_plan_resp = client.post(
            f"/api/sessions/{sid}/plans",
            json={"photo_ids": subset_ids, "strategy": "subfolder"},
            headers=headers,
        )
        assert sub_plan_resp.status_code == 200, sub_plan_resp.text
        sub_plan = sub_plan_resp.json()["plan"]
        assert sub_plan["item_count"] == 2
        apply_subset = client.post(
            f"/api/plans/{sub_plan['id']}/approve-and-apply",
            json={"plan_hash": sub_plan["plan_hash"]},
            headers=headers,
        )
        assert apply_subset.status_code == 200 and apply_subset.json()["moved_photo_count"] == 2

        # After moving 2 of the auto-flagged rejects, unchecked ones remain red AND stay visible in `待选`!
        unselected_ids = [it["photo_id"] for it in draft_all["plan"]["items"][2:]]
        if unselected_ids:
            unrev_with_red = {
                p["photo_id"]
                for p in client.get(
                    f"/api/sessions/{sid}/photos?limit=50&sort=name&bucket=unreviewed"
                ).json()["items"]
            }
            assert set(unselected_ids).issubset(unrev_with_red)
            s_mid = client.get(f"/api/sessions/{sid}/summary").json()
            assert s_mid["actionable_reject_count"] == len(unselected_ids)

            # Clicking explicit "放回待选" clears the red flag and keeps them cleanly in `待选`
            spare_r = client.post(
                "/api/verdicts",
                json={
                    "photo_ids": unselected_ids,
                    "bucket": "normal",
                    "reason_tag": "spared_from_trash",
                    "reason_codes": ["user_reset"],
                },
                headers=headers,
            )
            assert spare_r.status_code == 200
            s6 = client.get(f"/api/sessions/{sid}/summary").json()
            assert s6["actionable_reject_count"] == 0
            unrev_after = {
                p["photo_id"]
                for p in client.get(
                    f"/api/sessions/{sid}/photos?limit=50&sort=name&bucket=unreviewed"
                ).json()["items"]
            }
            assert set(unselected_ids).issubset(unrev_after)
    finally:
        restore_dataset_write_bits(shoot)


