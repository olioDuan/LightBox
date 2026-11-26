"""Tests for T18 post-cull people auto-classification, clustering, strict grouping, and SafeFS export."""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from PIL import Image

from lightbox.api import create_app
from lightbox.core import ENGINE_VERSION, compute_content_fp, find_face_model_path
from lightbox.core.faces import download_antelopev2_archive
from lightbox.models import (
    ARCFACE_INPUT_SIZE,
    FACE_EMBEDDER_MODEL_ID,
    FACE_EMBEDDING_DIM,
    align_face_arcface,
    find_face_embedder_model_path,
)
from lightbox.paths import get_app_paths
from lightbox.services import GROUP_KEY_GROUP_PHOTO, LightboxServices, PeopleError
from lightbox.services.people import _load_rgb_image_for_photo
from lightbox.store import (
    AssetRecord,
    Database,
    FaceRecord,
    PhotoRecord,
    RootRecord,
    SessionRecord,
)


def _face_vec(axis: int, jitter_axis: int = 0, jitter: float = 0.0) -> NDArray[np.float32]:
    v = np.zeros(FACE_EMBEDDING_DIM, dtype=np.float32)
    v[axis] = 1.0
    if jitter != 0.0:
        v[jitter_axis] += float(jitter)
    norm = float(np.linalg.norm(v))
    return (v / norm).astype(np.float32)


class FakeFaceEmbedder:
    """Deterministic 512-d face embedder keyed by crop RGB mean color (no real ONNX needed)."""

    model_id = FACE_EMBEDDER_MODEL_ID
    dim = FACE_EMBEDDING_DIM

    def __init__(self, palette: dict[tuple[int, int, int], NDArray[np.float32]]) -> None:
        self.palette = palette
        self.calls: int = 0

    def embed_aligned_faces(
        self,
        aligned_rgb_112: Sequence[NDArray[np.uint8]],
    ) -> NDArray[np.float32]:
        self.calls += 1
        rows: list[NDArray[np.float32]] = []
        for crop in aligned_rgb_112:
            assert crop.shape == (ARCFACE_INPUT_SIZE[1], ARCFACE_INPUT_SIZE[0], 3)
            # Match closest palette color by center pixel RGB
            center = crop[56, 56].astype(np.float32)
            best_vec = min(
                self.palette.items(),
                key=lambda kv: float(np.linalg.norm(center - np.array(kv[0], dtype=np.float32))),
            )[1]
            rows.append(best_vec)
        return np.stack(rows, axis=0) if rows else np.zeros((0, self.dim), dtype=np.float32)


def _write_color_blocks_jpg(
    path: Path,
    face_colors: Sequence[tuple[tuple[float, float, float, float], tuple[int, int, int]]],
) -> Path:
    """Create a 320x240 RGB JPEG painting distinct solid colors inside each face bounding box."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((240, 320, 3), 20, dtype=np.uint8)
    for (x1, y1, x2, y2), color in face_colors:
        px1, py1 = int(x1 * 320), int(y1 * 240)
        px2, py2 = int(x2 * 320), int(y2 * 240)
        arr[py1:py2, px1:px2] = color
    Image.fromarray(arr, mode="RGB").save(path, format="JPEG", quality=98)
    return path


def _landmarks_for_bbox(
    bbox: tuple[float, float, float, float],
) -> tuple[tuple[float, float], ...]:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return (
        (x1 + 0.34 * w, y1 + 0.45 * h),
        (x1 + 0.66 * w, y1 + 0.45 * h),
        (x1 + 0.50 * w, y1 + 0.64 * h),
        (x1 + 0.37 * w, y1 + 0.82 * h),
        (x1 + 0.63 * w, y1 + 0.82 * h),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_kept_photo(
    svcs: LightboxServices,
    *,
    session_id: str,
    session_name: str,
    photo_id: str,
    stem: str,
    img_path: Path,
    faces: Sequence[tuple[tuple[float, float, float, float], float]],
    bucket: str = "normal",
) -> str:
    if svcs.repo.get_session(session_id) is None:
        root_id = f"root_{session_id}"
        svcs.repo.upsert_root(
            RootRecord(id=root_id, path=str(img_path.parent), device_id="dev1", kind="disk")
        )
        svcs.repo.upsert_session(
            SessionRecord(
                id=session_id,
                root_id=root_id,
                root_path=str(img_path.parent),
                name=session_name,
                sensitivity="conservative",
                engine_version=ENGINE_VERSION,
                created_at=1_700_000_000.0,
                last_scanned_at=1_700_000_010.0,
            )
        )
    fp = f"fp_{photo_id}"
    svcs.repo.upsert_photo(
        PhotoRecord(
            id=photo_id,
            session_id=session_id,
            stem=stem,
            rel_dir=".",
            primary_content_fp=fp,
            cohort="camera",
            capture_time=1_700_000_000.0,
            preview_w=320,
            preview_h=240,
        )
    )
    svcs.repo.upsert_asset(
        AssetRecord(
            photo_id=photo_id,
            rel_path=img_path.name,
            abs_path=str(img_path.resolve()),
            kind="jpg",
            size_bytes=img_path.stat().st_size,
            mtime=img_path.stat().st_mtime,
            content_fp=fp,
        )
    )
    svcs.repo.add_verdict(
        photo_id=photo_id,
        source="user" if bucket == "normal" else "engine",
        bucket=bucket,
        confidence=0.95,
        reason_codes=[],
        increment_decision_version=False,
    )
    svcs.repo.replace_faces(
        fp,
        [
            FaceRecord(
                content_fp=fp,
                idx=idx,
                bbox=bbox,
                landmarks=_landmarks_for_bbox(bbox),
                det_score=0.95,
                sharpness=40.0,
                area_ratio=area_ratio,
            )
            for idx, (bbox, area_ratio) in enumerate(faces)
        ],
    )
    return fp


def test_migration_004_creates_people_tables_idempotently() -> None:
    """Migration 004 creates `people`, `face_embeddings`, `face_assignments` and is idempotent."""
    with Database(":memory:") as db:
        applied_first = db.migrate()
        assert 4 in applied_first
        applied_second = db.migrate()
        assert applied_second == []
        tables = {
            str(r["name"])
            for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {"people", "face_embeddings", "face_assignments"} <= tables


def test_align_face_arcface_uses_5pt_similarity_transform() -> None:
    """5-point landmark alignment warps a face cleanly onto 112x112 RGB uint8."""
    img = np.full((240, 320, 3), 120, dtype=np.uint8)
    bbox = (0.2, 0.2, 0.6, 0.7)
    aligned = align_face_arcface(img, bbox, _landmarks_for_bbox(bbox))
    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == np.uint8

    # Fallback without landmarks also returns 112x112 RGB uint8
    aligned_fallback = align_face_arcface(img, bbox, None)
    assert aligned_fallback.shape == (112, 112, 3)


def test_people_auto_classification_all_acceptance_criteria(tmp_path: Path) -> None:
    """Verify all 5 T18 acceptance criteria:
    1. Label 1-2 samples for Bob/Alice -> remaining solo shots auto-grouped (incl. cross-session).
    2. Bob + Alice photo -> strictly in `多人合影`, never mixed into solo groups.
    3. Marking bystander face in a 2-face photo as `ignored` -> photo moves into solo group.
    4. 50 solo photos for one person -> ALL 50 returned with zero truncation.
    5. Exporting copies via SafeFS -> originals SHA-256 100% unchanged, not in `plans`.
    """
    bob_color = (230, 40, 40)
    alice_color = (40, 220, 60)
    bystander_color = (40, 60, 230)
    stranger_color = (230, 210, 40)

    fake_embedder = FakeFaceEmbedder(
        {
            bob_color: _face_vec(10),
            alice_color: _face_vec(20),
            bystander_color: _face_vec(30),
            stranger_color: _face_vec(40),
        }
    )

    solo_box = (0.25, 0.20, 0.75, 0.80)
    left_box = (0.05, 0.20, 0.45, 0.80)
    right_box = (0.55, 0.20, 0.95, 0.80)

    shoot1 = tmp_path / "shoot1"
    shoot2 = tmp_path / "shoot2"
    paths = get_app_paths(tmp_path / "home")

    original_hashes: dict[Path, str] = {}

    with LightboxServices(app_paths=paths, face_embedder=fake_embedder) as svcs:
        # 50 solo photos of Bob across Session 1 (45) and Session 2 (5) to prove full 50 + cross-session
        bob_fps: list[str] = []
        for i in range(45):
            p = _write_color_blocks_jpg(shoot1 / f"BOB_{i:02d}.jpg", [(solo_box, bob_color)])
            original_hashes[p] = _sha256(p)
            fp = _seed_kept_photo(
                svcs,
                session_id="sess1",
                session_name="Trip Day 1",
                photo_id=f"bob_{i:02d}",
                stem=f"BOB_{i:02d}",
                img_path=p,
                faces=[(solo_box, 0.12)],
            )
            bob_fps.append(fp)

        for i in range(45, 50):
            p = _write_color_blocks_jpg(shoot2 / f"BOB_{i:02d}.jpg", [(solo_box, bob_color)])
            original_hashes[p] = _sha256(p)
            fp = _seed_kept_photo(
                svcs,
                session_id="sess2",
                session_name="Trip Day 2",
                photo_id=f"bob_{i:02d}",
                stem=f"BOB_{i:02d}",
                img_path=p,
                faces=[(solo_box, 0.12)],
            )
            bob_fps.append(fp)

        # 4 solo photos of Alice
        alice_fps: list[str] = []
        for i in range(4):
            p = _write_color_blocks_jpg(shoot1 / f"ALICE_{i:02d}.jpg", [(solo_box, alice_color)])
            original_hashes[p] = _sha256(p)
            fp = _seed_kept_photo(
                svcs,
                session_id="sess1",
                session_name="Trip Day 1",
                photo_id=f"alice_{i:02d}",
                stem=f"ALICE_{i:02d}",
                img_path=p,
                faces=[(solo_box, 0.12)],
            )
            alice_fps.append(fp)

        # 1 rejected (`red`) photo of Bob -> MUST be excluded from people classification
        p_rej = _write_color_blocks_jpg(shoot1 / "BOB_REJECT.jpg", [(solo_box, bob_color)])
        original_hashes[p_rej] = _sha256(p_rej)
        _seed_kept_photo(
            svcs,
            session_id="sess1",
            session_name="Trip Day 1",
            photo_id="bob_reject",
            stem="BOB_REJECT",
            img_path=p_rej,
            faces=[(solo_box, 0.12)],
            bucket="red",
        )

        # 1 group photo with Bob + Alice
        p_group = _write_color_blocks_jpg(
            shoot1 / "GROUP_BA.jpg",
            [(left_box, bob_color), (right_box, alice_color)],
        )
        original_hashes[p_group] = _sha256(p_group)
        _seed_kept_photo(
            svcs,
            session_id="sess1",
            session_name="Trip Day 1",
            photo_id="group_ba",
            stem="GROUP_BA",
            img_path=p_group,
            faces=[(left_box, 0.10), (right_box, 0.10)],
        )

        # 1 photo with Alice + background bystander (small 0.5% face >= 0.3%)
        p_bystander = _write_color_blocks_jpg(
            shoot1 / "ALICE_BYSTANDER.jpg",
            [(left_box, alice_color), (right_box, bystander_color)],
        )
        original_hashes[p_bystander] = _sha256(p_bystander)
        bystander_fp = _seed_kept_photo(
            svcs,
            session_id="sess1",
            session_name="Trip Day 1",
            photo_id="alice_bystander",
            stem="ALICE_BYSTANDER",
            img_path=p_bystander,
            faces=[(left_box, 0.10), (right_box, 0.005)],
        )

        # 1 unnamed solo stranger + 1 landscape with no face
        p_stranger = _write_color_blocks_jpg(
            shoot1 / "STRANGER.jpg", [(solo_box, stranger_color)]
        )
        original_hashes[p_stranger] = _sha256(p_stranger)
        _seed_kept_photo(
            svcs,
            session_id="sess1",
            session_name="Trip Day 1",
            photo_id="stranger_01",
            stem="STRANGER",
            img_path=p_stranger,
            faces=[(solo_box, 0.10)],
        )

        p_landscape = _write_color_blocks_jpg(shoot1 / "LANDSCAPE.jpg", [])
        original_hashes[p_landscape] = _sha256(p_landscape)
        _seed_kept_photo(
            svcs,
            session_id="sess1",
            session_name="Trip Day 1",
            photo_id="landscape_01",
            stem="LANDSCAPE",
            img_path=p_landscape,
            faces=[],
        )

        # Index both sessions
        idx1 = svcs.people.index_session("sess1")
        idx2 = svcs.people.index_session("sess2")
        assert idx1.kept_photo_count == 45 + 4 + 1 + 1 + 1 + 1  # bob_reject excluded!
        assert idx2.kept_photo_count == 5

        # Acceptance #1: Label ONLY 1-2 sample faces for Bob and Alice
        bob_person = svcs.people.label_name("Bob", faces=[(bob_fps[0], 0), (bob_fps[1], 0)])
        alice_person = svcs.people.label_name("Alice", faces=[(alice_fps[0], 0)])

        groups = svcs.people.get_people_groups()
        by_name = {g.name: g for g in groups.people}

        # Acceptance #1 & #4: ALL 50 solo photos of Bob (45 in sess1 + 5 in sess2) returned with zero truncation!
        assert by_name["Bob"].photo_count == 50
        assert len(by_name["Bob"].photos) == 50
        assert "bob_reject" not in {p.photo_id for p in by_name["Bob"].photos}

        # Alice has 4 solo photos initially (`alice_bystander` is still in group_photos)
        assert by_name["Alice"].photo_count == 4
        assert len(by_name["Alice"].photos) == 4

        # Acceptance #2: Bob + Alice photo is strictly in `多人合影`, NOT in Bob's or Alice's solo group
        group_ids = {p.photo_id for p in groups.group_photos}
        assert group_ids == {"group_ba", "alice_bystander"}
        assert "group_ba" not in {p.photo_id for p in by_name["Bob"].photos}
        assert "group_ba" not in {p.photo_id for p in by_name["Alice"].photos}

        # Group-photo single-person filter ("含 Bob")
        groups_with_bob = svcs.people.get_people_groups(group_person_id=bob_person.id)
        assert [p.photo_id for p in groups_with_bob.group_photos] == ["group_ba"]

        # Unnamed & No-face buckets
        assert [p.photo_id for p in groups.unnamed_photos] == ["stranger_01"]
        assert [p.photo_id for p in groups.no_face_photos] == ["landscape_01"]

        # Acceptance #3: Mark bystander face (face_idx=1) as `ignored` -> photo moves from `多人合影` to Alice!
        ignored_cnt = svcs.people.label_ignore(faces=[(bystander_fp, 1)])
        assert ignored_cnt == 1

        groups_after_ignore = svcs.people.get_people_groups()
        by_name_after = {g.name: g for g in groups_after_ignore.people}
        assert {p.photo_id for p in groups_after_ignore.group_photos} == {"group_ba"}
        assert by_name_after["Alice"].photo_count == 5
        assert "alice_bystander" in {p.photo_id for p in by_name_after["Alice"].photos}

        # Acceptance #5: Export copies via SafeFS.write_new_copy -> originals SHA-256 unchanged, zero `plans` rows
        export_dir = tmp_path / "exported_people"
        exp = svcs.people.export_groups(
            export_dir,
            group_keys=[
                f"person:{bob_person.id}",
                f"person:{alice_person.id}",
                GROUP_KEY_GROUP_PHOTO,
            ],
            lang="zh-CN",
        )
        assert exp.exported_photos == 50 + 5 + 1
        assert exp.exported_files == 50 + 5 + 1
        assert len(list((export_dir / "Bob").iterdir())) == 50
        assert len(list((export_dir / "Alice").iterdir())) == 5
        assert [p.name for p in (export_dir / "多人合影").iterdir()] == ["GROUP_BA.jpg"]
        assert svcs.plan.list_plans() == []  # Export must NOT enter `plans` table

        # Verify every single original file SHA-256 is 100% unchanged
        for orig_path, expected_hash in original_hashes.items():
            assert orig_path.is_file()
            assert _sha256(orig_path) == expected_hash


def test_people_api_merge_rename_and_csrf_export(tmp_path: Path) -> None:
    """Verify `/api/sessions/{id}/people/index`, `/api/people`, `/api/people/label` (name/merge/ignore), `/api/people/export` + CSRF."""
    c1 = (220, 50, 50)
    c2 = (50, 50, 220)
    fake_embedder = FakeFaceEmbedder({c1: _face_vec(1), c2: _face_vec(2)})
    box = (0.2, 0.2, 0.8, 0.8)

    shoot = tmp_path / "shoot"
    p1 = _write_color_blocks_jpg(shoot / "P1.jpg", [(box, c1)])
    p2 = _write_color_blocks_jpg(shoot / "P2.jpg", [(box, c2)])
    paths = get_app_paths(tmp_path / "home_api")

    with LightboxServices(app_paths=paths, face_embedder=fake_embedder) as svcs:
        _seed_kept_photo(
            svcs,
            session_id="s_api",
            session_name="API Shoot",
            photo_id="p1",
            stem="P1",
            img_path=p1,
            faces=[(box, 0.15)],
        )
        _seed_kept_photo(
            svcs,
            session_id="s_api",
            session_name="API Shoot",
            photo_id="p2",
            stem="P2",
            img_path=p2,
            faces=[(box, 0.15)],
        )

    app = create_app(app_paths=paths, port=8765, face_embedder_factory=lambda _s: fake_embedder)
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    auth = client.post(
        "/api/auth/session",
        headers={"X-Lightbox-Local-Confirm": "1", "Origin": "http://127.0.0.1:8765"},
    )
    assert auth.status_code == 200
    csrf = auth.json()["csrf_token"]
    headers = {"X-Lightbox-CSRF": csrf, "Origin": "http://127.0.0.1:8765"}

    # 1. Index session -> 2 distinct unnamed clusters
    r_idx = client.post("/api/sessions/s_api/people/index", headers=headers)
    assert r_idx.status_code == 200
    assert r_idx.json()["unnamed_cluster_count"] == 2

    # 2. Fetch groups + face crop endpoint
    r_list = client.get("/api/people")
    assert r_list.status_code == 200
    clusters = r_list.json()["unnamed_clusters"]
    assert len(clusters) == 2
    r_crop = client.get(
        f"/api/photos/{clusters[0]['cover_photo_id']}/face-crop",
        params={"face_idx": clusters[0]["cover_face_idx"]},
    )
    assert r_crop.status_code == 200
    assert r_crop.headers["content-type"] == "image/webp"

    # 3. Merge cluster[1] into cluster[0], then name merged cluster as "Me"
    r_merge = client.post(
        "/api/people/label",
        headers=headers,
        json={
            "action": "merge",
            "source_cluster_id": clusters[1]["cluster_id"],
            "target_cluster_id": clusters[0]["cluster_id"],
        },
    )
    assert r_merge.status_code == 200
    assert r_merge.json()["moved_faces"] == 1

    r_name = client.post(
        "/api/people/label",
        headers=headers,
        json={
            "action": "name",
            "name": "Me",
            "is_self": True,
            "cluster_id": clusters[0]["cluster_id"],
        },
    )
    assert r_name.status_code == 200
    r_after = client.get("/api/people").json()
    assert len(r_after["people"]) == 1
    assert r_after["people"][0]["name"] == "Me"
    assert r_after["people"][0]["photo_count"] == 2

    # 4. Export without CSRF MUST be rejected (403); with CSRF succeeds
    export_dir = tmp_path / "api_export"
    r_no_csrf = client.post(
        "/api/people/export",
        headers={"Origin": "http://127.0.0.1:8765"},
        json={"target_dir": str(export_dir)},
    )
    assert r_no_csrf.status_code == 403

    r_export = client.post(
        "/api/people/export",
        headers=headers,
        json={"target_dir": str(export_dir), "lang": "en-US"},
    )
    assert r_export.status_code == 200
    assert r_export.json()["exported_photos"] == 2
    assert len(list((export_dir / "Me").iterdir())) == 2


def test_find_face_model_recognizes_scrfd_10g_bnkps_and_atomic_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression #1 & #4: recognize `scrfd_10g_bnkps.onnx` from official antelopev2.zip & extract atomically."""
    paths = get_app_paths(tmp_path / "home_dl")
    paths.ensure_dirs()

    # Build a synthetic antelopev2.zip containing official filenames (no det_10g.onnx)
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("antelopev2/scrfd_10g_bnkps.onnx", b"scrfd-bytes")
        zf.writestr("antelopev2/glintr100.onnx", b"glintr-bytes")
    zip_bytes = zip_buf.getvalue()

    import urllib.request

    class _FakeResp(io.BytesIO):
        def __enter__(self) -> _FakeResp:
            return self

        def __exit__(self, *_args: Any) -> None:
            self.close()

    monkeypatch.setattr(urllib.request, "urlopen", lambda _req, timeout=120.0: _FakeResp(zip_bytes))

    download_antelopev2_archive(paths, download_url="https://example.invalid/antelopev2.zip")

    det_path = find_face_model_path(paths)
    rec_path = find_face_embedder_model_path(paths)
    assert det_path is not None and det_path.name == "scrfd_10g_bnkps.onnx"
    assert rec_path is not None and rec_path.name == "glintr100.onnx"
    # No leftover .part files
    assert not list((paths.models_dir / "antelopev2").glob("*.part"))


def test_merged_unnamed_clusters_persist_across_unrelated_naming_and_recluster(
    tmp_path: Path,
) -> None:
    """Regression #2: manually merged unnamed clusters stay merged when naming an unrelated cluster."""
    c_a = (220, 40, 40)
    c_b = (40, 220, 40)
    c_other = (40, 40, 220)
    # Orthogonal vectors so greedy clustering puts them in 3 separate clusters initially
    fake_embedder = FakeFaceEmbedder({c_a: _face_vec(1), c_b: _face_vec(2), c_other: _face_vec(3)})
    box = (0.2, 0.2, 0.8, 0.8)
    shoot = tmp_path / "shoot"
    paths = get_app_paths(tmp_path / "home_merge")

    with LightboxServices(app_paths=paths, face_embedder=fake_embedder) as svcs:
        for idx, (stem, color) in enumerate(
            [("A1", c_a), ("A2", c_a), ("B1", c_b), ("B2", c_b), ("O1", c_other)]
        ):
            img = _write_color_blocks_jpg(shoot / f"{stem}.jpg", [(box, color)])
            _seed_kept_photo(
                svcs,
                session_id="s_merge",
                session_name="Merge Shoot",
                photo_id=f"p_{idx}",
                stem=stem,
                img_path=img,
                faces=[(box, 0.15)],
            )

        svcs.people.index_session("s_merge")
        initial = svcs.people.get_people_groups()
        assert len(initial.unnamed_clusters) == 3

        by_stem = {
            c.photos[0].stem: c.cluster_id for c in initial.unnamed_clusters
        }
        clu_a = by_stem["A1"]
        clu_b = by_stem["B1"]
        clu_o = by_stem["O1"]

        # Merge B (2 faces) into A (2 faces) -> 4 faces
        svcs.people.label_merge(source_cluster_id=clu_b, target_cluster_id=clu_a)
        after_merge = svcs.people.get_people_groups()
        merged_clu = next(c for c in after_merge.unnamed_clusters if c.cluster_id == clu_a)
        assert merged_clu.face_count == 4

        # Name UNRELATED cluster O -> triggers recluster_and_match()
        svcs.people.label_name("Charlie", cluster_id=clu_o)

        # Merged cluster A+B MUST remain 1 cluster of 4 faces (NOT split back to 2+2!)
        after_unrelated = svcs.people.get_people_groups()
        assert len(after_unrelated.unnamed_clusters) == 1
        assert after_unrelated.unnamed_clusters[0].cluster_id == clu_a
        assert after_unrelated.unnamed_clusters[0].face_count == 4

        # Re-indexing also preserves the merged cluster
        svcs.people.index_session("s_merge")
        after_reindex = svcs.people.get_people_groups()
        assert len(after_reindex.unnamed_clusters) == 1
        assert after_reindex.unnamed_clusters[0].face_count == 4


def test_export_blocks_inside_scanned_dir_and_handles_collisions_and_repeat_idempotently(
    tmp_path: Path,
) -> None:
    """Regression #3 & #6: forbid exporting inside shoot dir, preserve RAW+JPG pairing, skip duplicate re-exports, avoid special folder collision."""
    c_p = (220, 60, 60)
    c_q = (60, 220, 60)
    fake_embedder = FakeFaceEmbedder({c_p: _face_vec(5), c_q: _face_vec(6)})
    solo_box = (0.2, 0.2, 0.8, 0.8)
    left_box = (0.05, 0.2, 0.45, 0.8)
    right_box = (0.55, 0.2, 0.95, 0.8)

    shoot = tmp_path / "shoot"
    paths = get_app_paths(tmp_path / "home_exp")

    with LightboxServices(app_paths=paths, face_embedder=fake_embedder) as svcs:
        # Two different subdirs both containing IMG_001.jpg + IMG_001.ARW (paired!)
        sub1_jpg = _write_color_blocks_jpg(shoot / "day1" / "IMG_001.jpg", [(solo_box, c_p)])
        sub1_raw = shoot / "day1" / "IMG_001.ARW"
        sub1_raw.write_bytes(b"RAW-DAY1-DATA")

        sub2_jpg = _write_color_blocks_jpg(shoot / "day2" / "IMG_001.jpg", [(solo_box, c_p)])
        # Ensure distinct content_fp by appending a harmless trailing comment byte
        sub2_jpg.write_bytes(sub2_jpg.read_bytes() + b"\x00")
        sub2_raw = shoot / "day2" / "IMG_001.ARW"
        sub2_raw.write_bytes(b"RAW-DAY2-DIFFERENT-DATA")

        # Group photo so special folder `多人合影` is populated
        grp_jpg = _write_color_blocks_jpg(
            shoot / "GROUP.jpg", [(left_box, c_p), (right_box, c_q)]
        )

        # Register session & photos with true content_fps
        svcs.repo.upsert_root(RootRecord(id="r1", path=str(shoot), device_id="d1", kind="disk"))
        svcs.repo.upsert_session(
            SessionRecord(
                id="s_exp",
                root_id="r1",
                root_path=str(shoot),
                name="Shoot",
                sensitivity="conservative",
                engine_version=ENGINE_VERSION,
                created_at=1_700_000_000.0,
            )
        )
        for pid, rel_dir, jpg, raw in [
            ("p_d1", "day1", sub1_jpg, sub1_raw),
            ("p_d2", "day2", sub2_jpg, sub2_raw),
        ]:
            fp_j = compute_content_fp(jpg)
            fp_r = compute_content_fp(raw)
            svcs.repo.upsert_photo(
                PhotoRecord(
                    id=pid,
                    session_id="s_exp",
                    stem="IMG_001",
                    rel_dir=rel_dir,
                    primary_content_fp=fp_j,
                    cohort="camera",
                )
            )
            svcs.repo.upsert_asset(
                AssetRecord(
                    photo_id=pid,
                    rel_path=f"{rel_dir}/{jpg.name}",
                    abs_path=str(jpg.resolve()),
                    kind="jpg",
                    size_bytes=jpg.stat().st_size,
                    mtime=jpg.stat().st_mtime,
                    content_fp=fp_j,
                )
            )
            svcs.repo.upsert_asset(
                AssetRecord(
                    photo_id=pid,
                    rel_path=f"{rel_dir}/{raw.name}",
                    abs_path=str(raw.resolve()),
                    kind="raw",
                    size_bytes=raw.stat().st_size,
                    mtime=raw.stat().st_mtime,
                    content_fp=fp_r,
                )
            )
            svcs.repo.replace_faces(
                fp_j,
                [
                    FaceRecord(
                        content_fp=fp_j,
                        idx=0,
                        bbox=solo_box,
                        landmarks=_landmarks_for_bbox(solo_box),
                        det_score=0.95,
                        sharpness=40.0,
                        area_ratio=0.15,
                    )
                ],
            )
            svcs.repo.add_verdict(
                photo_id=pid, source="user", bucket="normal", confidence=1.0, reason_codes=[]
            )

        fp_g = compute_content_fp(grp_jpg)
        svcs.repo.upsert_photo(
            PhotoRecord(
                id="p_grp",
                session_id="s_exp",
                stem="GROUP",
                rel_dir=".",
                primary_content_fp=fp_g,
                cohort="camera",
            )
        )
        svcs.repo.upsert_asset(
            AssetRecord(
                photo_id="p_grp",
                rel_path="GROUP.jpg",
                abs_path=str(grp_jpg.resolve()),
                kind="jpg",
                size_bytes=grp_jpg.stat().st_size,
                mtime=grp_jpg.stat().st_mtime,
                content_fp=fp_g,
            )
        )
        svcs.repo.replace_faces(
            fp_g,
            [
                FaceRecord(
                    content_fp=fp_g,
                    idx=0,
                    bbox=left_box,
                    landmarks=_landmarks_for_bbox(left_box),
                    det_score=0.95,
                    sharpness=40.0,
                    area_ratio=0.10,
                ),
                FaceRecord(
                    content_fp=fp_g,
                    idx=1,
                    bbox=right_box,
                    landmarks=_landmarks_for_bbox(right_box),
                    det_score=0.95,
                    sharpness=40.0,
                    area_ratio=0.10,
                ),
            ],
        )
        svcs.repo.add_verdict(
            photo_id="p_grp", source="user", bucket="normal", confidence=1.0, reason_codes=[]
        )

        svcs.people.index_session("s_exp")
        # Deliberately name the solo person `多人合影` to test special-folder collision disambiguation
        svcs.people.label_name("多人合影", faces=[(compute_content_fp(sub1_jpg), 0)])

        # 1. Exporting inside the scanned shoot directory MUST be rejected
        with pytest.raises(PeopleError, match="Refusing to export inside scanned photo folder"):
            svcs.people.export_groups(shoot / "按人物导出")

        # 2. Exporting outside shoot directory disambiguates folder names & keeps RAW+JPG suffixes paired
        out_dir = tmp_path / "clean_export"
        exp1 = svcs.people.export_groups(out_dir, lang="zh-CN")
        assert exp1.exported_photos == 3
        assert exp1.exported_files == 5  # 2*(jpg+arw) + 1 group jpg
        assert sorted(p.name for p in (out_dir / "多人合影").iterdir()) == ["GROUP.jpg"]
        assert sorted(p.name for p in (out_dir / "多人合影 (2)").iterdir()) == [
            "IMG_001.ARW",
            "IMG_001.jpg",
            "IMG_001__1.ARW",
            "IMG_001__1.jpg",
        ]

        # 3. Repeating the exact same export is idempotent (0 new duplicate files written)
        exp2 = svcs.people.export_groups(out_dir, lang="zh-CN")
        assert exp2.exported_photos == 0
        assert exp2.exported_files == 0
        assert len(list((out_dir / "多人合影 (2)").iterdir())) == 4


def test_load_rgb_fallback_applies_exif_orientation(tmp_path: Path) -> None:
    """Regression #7: when preview cache is absent, fallback raw JPEG read applies EXIF orientation."""
    shoot = tmp_path / "shoot"
    shoot.mkdir()
    # Create a 300x200 landscape JPEG tagged with EXIF Orientation=6 (90° CW -> should load as 200x300 portrait!)
    raw_jpg = shoot / "PORTRAIT_EXIF.jpg"
    img = Image.new("RGB", (300, 200), (100, 150, 200))
    exif = Image.Exif()
    exif[274] = 6  # Orientation = Rotate 90 CW
    img.save(raw_jpg, format="JPEG", exif=exif.tobytes())

    paths = get_app_paths(tmp_path / "home_exif")
    with LightboxServices(app_paths=paths) as svcs:
        _seed_kept_photo(
            svcs,
            session_id="s_ori",
            session_name="Ori",
            photo_id="p_ori",
            stem="PORTRAIT_EXIF",
            img_path=raw_jpg,
            faces=[],
        )
        photo = svcs.repo.get_photo("p_ori")
        assert photo is not None
        rgb = _load_rgb_image_for_photo(svcs.repo, paths, photo)
        assert rgb is not None
        assert rgb.shape == (300, 200, 3)  # H=300, W=200 after EXIF transpose!
