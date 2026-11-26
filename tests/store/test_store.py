"""T02 unit tests: SQLite WAL/migrations, effective verdicts, decision_version, and VectorIndex."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from lightbox.store import (
    DEFAULT_EMBEDDING_DIM,
    Database,
    ExifRecord,
    FaceRecord,
    LightboxRepository,
    MetricsRecord,
    NumpyVectorIndex,
    PhotoRecord,
    RootRecord,
    SessionRecord,
    SqliteVecIndex,
    create_vector_index,
)


def _seed_photo(repo: LightboxRepository, photo_id: str = "p1") -> None:
    repo.upsert_root(RootRecord(id="r1", path="/tmp/shoot", device_id="1", kind="disk"))
    repo.upsert_session(
        SessionRecord(
            id="s1",
            root_id="r1",
            root_path="/tmp/shoot",
            name="shoot",
            sensitivity="conservative",
            engine_version="0.1.0",
            created_at=time.time(),
        )
    )
    repo.upsert_photo(
        PhotoRecord(
            id=photo_id,
            session_id="s1",
            stem=f"IMG_{photo_id}",
            rel_dir=".",
            primary_content_fp=f"fp_{photo_id}",
            cohort="camera",
        )
    )


def test_db_pragmas_and_idempotent_migrations(tmp_path: Path) -> None:
    db_file = tmp_path / "lightbox.db"
    with Database(db_file) as db:
        assert db.sqlite_vec_available is True

        journal_mode = db.conn.execute("PRAGMA journal_mode;").fetchone()[0]
        busy_timeout = db.conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        foreign_keys = db.conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert str(journal_mode).lower() == "wal"
        assert int(busy_timeout) == 10000
        assert int(foreign_keys) == 1

        first = db.migrate()
        second = db.migrate()
        assert first == [1, 2, 3, 4]
        assert second == []

        # Verify migration 002 & 003 columns exist
        exif_cols = {
            str(r["name"]) for r in db.conn.execute("PRAGMA table_info(exif_data)").fetchall()
        }
        assert {"exposure_program", "af_mode", "flash_fired", "capture_local_hour"} <= exif_cols
        metrics_cols = {
            str(r["name"]) for r in db.conn.execute("PRAGMA table_info(metrics)").fetchall()
        }
        assert "motion_state" in metrics_cols
        faces_cols = {
            str(r["name"]) for r in db.conn.execute("PRAGMA table_info(faces)").fetchall()
        }
        assert {"landmarks_json", "det_score"} <= faces_cols
        llm_cols = {
            str(r["name"]) for r in db.conn.execute("PRAGMA table_info(llm_calls)").fetchall()
        }
        assert {"session_id", "cache_hit"} <= llm_cols

        tables = {
            str(row[0])
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        }
        expected = {
            "roots",
            "sessions",
            "photos",
            "assets",
            "exif_data",
            "metrics",
            "faces",
            "verdicts",
            "bursts",
            "burst_members",
            "vec_map",
            "vec_photos",
            "session_fts",
            "jobs",
            "job_items",
            "llm_calls",
            "agent_sessions",
            "agent_events",
            "overrides",
            "memory_rules_meta",
            "plans",
            "plan_items",
            "audit_logs",
            "people",
            "face_embeddings",
            "face_assignments",
            "effective_verdicts",
        }
        assert expected <= tables


def test_effective_verdicts_priority_and_atomic_decision_version() -> None:
    with Database(":memory:") as db:
        db.migrate()
        repo = LightboxRepository(db)
        _seed_photo(repo, "p1")

        # Initial version is 1; engine verdict does not bump decision_version
        repo.add_verdict(
            photo_id="p1",
            source="engine",
            bucket="red",
            confidence=0.9,
            reason_codes=["blur_global"],
        )
        photo = repo.get_photo("p1")
        assert photo is not None and photo.decision_version == 1
        eff = repo.get_effective_verdict("p1")
        assert eff is not None and eff.source == "engine" and eff.bucket == "red"

        # Agent/VLM/MCP overrides engine (tier 2 > tier 1) and bumps decision_version
        repo.add_verdict(
            photo_id="p1",
            source="vlm",
            bucket="yellow",
            confidence=0.75,
            reason_codes=["vlm_uncertain"],
        )
        photo = repo.get_photo("p1")
        assert photo is not None and photo.decision_version == 2
        eff = repo.get_effective_verdict("p1")
        assert eff is not None and eff.source == "vlm" and eff.bucket == "yellow"

        # Within tier 2, latest id wins (`mcp:claude` > `vlm`)
        repo.add_verdict(
            photo_id="p1",
            source="mcp:claude",
            bucket="normal",
            confidence=0.85,
            reason_codes=["mcp_keep"],
        )
        eff = repo.get_effective_verdict("p1")
        assert eff is not None and eff.source == "mcp:claude" and eff.bucket == "normal"

        # Rule (tier 3) overrides tier 2
        repo.add_verdict(
            photo_id="p1",
            source="rule",
            bucket="yellow",
            confidence=0.8,
            reason_codes=["compiled_rule_1"],
        )
        eff = repo.get_effective_verdict("p1")
        assert eff is not None and eff.source == "rule" and eff.bucket == "yellow"

        # User (tier 4) overrides everything
        repo.add_verdict(
            photo_id="p1",
            source="user",
            bucket="normal",
            confidence=1.0,
            reason_codes=["user_rescue"],
        )
        photo = repo.get_photo("p1")
        assert photo is not None and photo.decision_version == 5
        eff = repo.get_effective_verdict("p1")
        assert eff is not None and eff.source == "user" and eff.bucket == "normal"

        # Even if a later lower-priority engine/vlm/rule verdict is appended, user still wins!
        repo.add_verdict(
            photo_id="p1",
            source="rule",
            bucket="red",
            confidence=0.95,
            reason_codes=["later_rule"],
        )
        eff = repo.get_effective_verdict("p1")
        assert eff is not None and eff.source == "user" and eff.bucket == "normal"

        # A newer user verdict overrides the previous user verdict
        repo.add_verdict(
            photo_id="p1",
            source="user",
            bucket="red",
            confidence=1.0,
            reason_codes=["user_reject"],
        )
        eff = repo.get_effective_verdict("p1")
        assert eff is not None and eff.source == "user" and eff.bucket == "red"
        assert len(repo.list_verdict_history("p1")) == 7


def test_confirm_burst_bumps_member_decision_versions() -> None:
    with Database(":memory:") as db:
        db.migrate()
        repo = LightboxRepository(db)
        _seed_photo(repo, "p1")
        repo.upsert_photo(
            PhotoRecord(
                id="p2",
                session_id="s1",
                stem="IMG_p2",
                rel_dir=".",
                primary_content_fp="fp_p2",
                cohort="camera",
            )
        )
        repo.upsert_burst("b1", "s1", kind="burst", confirmed=False)
        repo.upsert_burst_member("b1", "p1", rank=1)
        repo.upsert_burst_member("b1", "p2", rank=2)

        repo.confirm_burst("b1", pick_photo_id="p1")
        versions = repo.get_photo_decision_versions(["p1", "p2"])
        assert versions == {"p1": 2, "p2": 2}


def test_session_fts_search() -> None:
    with Database(":memory:") as db:
        db.migrate()
        repo = LightboxRepository(db)
        repo.upsert_session_fts(
            photo_id="p1",
            session_name="Kyoto Autumn",
            rel_dir="day1",
            stem="DSC00123",
            scene_tag="temple sunset",
        )
        repo.upsert_session_fts(
            photo_id="p2",
            session_name="Tokyo Night",
            rel_dir="shibuya",
            stem="DSC00999",
            scene_tag="street neon",
        )
        assert repo.search_session_fts("sunset") == ["p1"]
        assert repo.search_session_fts("shibuya") == ["p2"]


def test_sqlite_vec_and_numpy_vector_index_parity() -> None:
    rng = np.random.default_rng(42)
    num_items = 60
    model_a = "siglip2-base"
    model_b = "other-model"

    vectors_a = rng.normal(size=(num_items, DEFAULT_EMBEDDING_DIM)).astype(np.float32)
    vectors_b = rng.normal(size=(10, DEFAULT_EMBEDDING_DIM)).astype(np.float32)
    query_vec = rng.normal(size=(DEFAULT_EMBEDDING_DIM,)).astype(np.float32)

    with Database(":memory:") as db:
        db.migrate()
        vec_idx = SqliteVecIndex(db)
        np_idx = NumpyVectorIndex()

        batch_a = [(f"fp_{i:03d}", model_a, vectors_a[i]) for i in range(num_items)]
        batch_b = [(f"fp_b_{i:03d}", model_b, vectors_b[i]) for i in range(10)]

        vec_idx.upsert_batch(batch_a)
        vec_idx.upsert_batch(batch_b)
        np_idx.upsert_batch(batch_a)
        np_idx.upsert_batch(batch_b)

        # Update an existing vector to confirm idempotent overwrite works in both backends
        replacement = rng.normal(size=(DEFAULT_EMBEDDING_DIM,)).astype(np.float32)
        vec_idx.upsert("fp_005", model_a, replacement)
        np_idx.upsert("fp_005", model_a, replacement)

        assert vec_idx.count(model_a) == num_items
        assert np_idx.count(model_a) == num_items

        top_k = 10
        res_vec = vec_idx.query(query_vec, model_a, top_k=top_k)
        res_np = np_idx.query(query_vec, model_a, top_k=top_k)

        assert [m.content_fp for m in res_vec] == [m.content_fp for m in res_np]
        for mv, mn in zip(res_vec, res_np, strict=True):
            assert mv.distance == pytest.approx(mn.distance, abs=1e-5)
            assert mv.score == pytest.approx(mn.score, abs=1e-5)

        # Subset filtering parity
        subset = {f"fp_{i:03d}" for i in range(0, 20, 2)}
        sub_vec = vec_idx.query(query_vec, model_a, top_k=5, filter_content_fps=subset)
        sub_np = np_idx.query(query_vec, model_a, top_k=5, filter_content_fps=subset)
        assert [m.content_fp for m in sub_vec] == [m.content_fp for m in sub_np]
        assert all(m.content_fp in subset for m in sub_vec)


def test_vector_index_factory_fallback() -> None:
    with Database(":memory:", load_vec=False) as db:
        db.migrate()
        idx = create_vector_index(db, prefer_sqlite_vec=True)
        assert idx.backend_name == "numpy"


def test_migration_002_exif_metrics_faces_and_protection_version_bump() -> None:
    with Database(":memory:") as db:
        db.migrate()
        repo = LightboxRepository(db)
        _seed_photo(repo, "p1")

        # Round-trip new T04.5 ExifRecord fields
        repo.upsert_exif(
            ExifRecord(
                photo_id="p1",
                aperture=2.8,
                shutter_s=1 / 500,
                iso=800,
                ev100=8.97,
                light_bucket="indoor",
                exposure_program="Av",
                af_mode="AF-C",
                flash_fired=True,
                capture_local_hour=21,
                field_confidence={"af_mode": "high"},
            )
        )
        ex = repo.get_exif("p1")
        assert ex is not None
        assert ex.exposure_program == "Av"
        assert ex.af_mode == "AF-C"
        assert ex.flash_fired is True
        assert ex.capture_local_hour == 21

        # Round-trip MetricsRecord including motion_state
        repo.upsert_metrics(
            MetricsRecord(
                content_fp="fp_p1",
                engine_version="0.1.0",
                sharp_p95=42.5,
                motion_aniso=0.82,
                motion_state="panning",
                face_count=1,
            )
        )
        m = repo.get_metrics("fp_p1", "0.1.0")
        assert m is not None
        assert m.motion_state == "panning"
        assert m.sharp_p95 == pytest.approx(42.5)

        # Round-trip FaceRecord including 5-point landmarks and det_score
        pts = ((10.0, 20.0), (30.0, 20.0), (20.0, 30.0), (12.0, 40.0), (28.0, 40.0))
        repo.replace_faces(
            "fp_p1",
            [
                FaceRecord(
                    content_fp="fp_p1",
                    idx=0,
                    bbox=(5.0, 10.0, 40.0, 50.0),
                    landmarks=pts,
                    det_score=0.97,
                    sharpness=31.0,
                    area_ratio=0.04,
                )
            ],
        )
        faces = repo.list_faces("fp_p1")
        assert len(faces) == 1
        assert faces[0].landmarks == pts
        assert faces[0].det_score == pytest.approx(0.97)

        # Flipping is_protected must bump decision_version automatically
        p_before = repo.get_photo("p1")
        assert p_before is not None and p_before.decision_version == 1
        repo.upsert_photo(
            PhotoRecord(
                id="p1",
                session_id="s1",
                stem="IMG_p1",
                rel_dir=".",
                primary_content_fp="fp_p1",
                cohort="camera",
                is_protected=True,
                exempt_reason="in_camera_lock",
            )
        )
        p_after = repo.get_photo("p1")
        assert p_after is not None and p_after.is_protected is True
        assert p_after.decision_version == 2
