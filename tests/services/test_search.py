"""Tests for T11 SigLIP 2 local vector indexing, zero-text-tower scene tagging, filtered search, and find_similar."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from PIL import Image

from lightbox.api import create_app
from lightbox.core import ENGINE_VERSION
from lightbox.models import (
    EMBEDDING_DIM,
    SCENE_VOCAB,
    SCENE_VOCAB_NPY_PATH,
    SIGLIP2_MODEL_ID,
    EmbedderError,
    Siglip2Embedder,
    classify_scene,
    clear_shared_siglip2_embedder_cache,
    load_scene_vocab,
)
from lightbox.paths import get_app_paths
from lightbox.services import LightboxServices
from lightbox.store import (
    AssetRecord,
    Database,
    ExifRecord,
    MetricsRecord,
    NumpyVectorIndex,
    PhotoRecord,
    RootRecord,
    SessionRecord,
    SqliteVecIndex,
)
from lightbox.tools import ToolContext, ToolRegistry


def _unit_vec(weights: dict[int, float]) -> NDArray[np.float32]:
    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for idx, val in weights.items():
        vec[idx] = float(val)
    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm > 0 else vec


class FakeEmbedder:
    """Deterministic test embedder mapping image filenames and text keywords onto known 768-d vectors."""

    model_id = SIGLIP2_MODEL_ID
    dim = EMBEDDING_DIM

    def __init__(
        self,
        *,
        image_vectors: dict[str, NDArray[np.float32]] | None = None,
        text_vectors: dict[str, NDArray[np.float32]] | None = None,
        forbid_text_tower: bool = False,
    ) -> None:
        self.image_vectors = dict(image_vectors or {})
        self.text_vectors = dict(text_vectors or {})
        self.forbid_text_tower = forbid_text_tower
        self.embed_images_calls: int = 0
        self.embed_texts_calls: int = 0

    def embed_images(self, images: Sequence[Image.Image | Path]) -> NDArray[np.float32]:
        self.embed_images_calls += 1
        rows: list[NDArray[np.float32]] = []
        for item in images:
            key = item.stem if isinstance(item, Path) else ""
            chosen: NDArray[np.float32] | None = None
            for needle, vec in self.image_vectors.items():
                if needle in str(item):
                    chosen = vec
                    break
            if chosen is None:
                # Deterministic fallback vector derived from key
                h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
                chosen = _unit_vec({h % EMBEDDING_DIM: 1.0})
            rows.append(chosen)
        return np.stack(rows, axis=0) if rows else np.zeros((0, self.dim), dtype=np.float32)

    def embed_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if self.forbid_text_tower:
            raise AssertionError("Text tower must NOT be called during vision-only tagging!")
        self.embed_texts_calls += 1
        rows: list[NDArray[np.float32]] = []
        for text in texts:
            lowered = text.strip().lower()
            chosen = self.text_vectors.get(lowered)
            if chosen is None:
                for needle, vec in self.text_vectors.items():
                    if needle in lowered:
                        chosen = vec
                        break
            if chosen is None:
                chosen = _unit_vec({0: 1.0})
            rows.append(chosen)
        return np.stack(rows, axis=0) if rows else np.zeros((0, self.dim), dtype=np.float32)


def _create_jpg(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (320, 240), color)
    img.save(path, format="JPEG", quality=92)
    return path


def _seed_photo_with_vector(
    svcs: LightboxServices,
    *,
    session_id: str,
    session_name: str,
    photo_id: str,
    stem: str,
    content_fp: str,
    camera_model: str,
    capture_iso: str,
    light_bucket: str,
    focal_eq: float,
    vec: NDArray[np.float32],
    img_path: Path,
    scene_tag: str | None = None,
) -> None:
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
    ts = datetime.strptime(capture_iso, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() + 43200.0
    svcs.repo.upsert_photo(
        PhotoRecord(
            id=photo_id,
            session_id=session_id,
            stem=stem,
            rel_dir=".",
            primary_content_fp=content_fp,
            cohort="camera",
            capture_time=ts,
            camera_model=camera_model,
            lens_model="35mm f/1.4",
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
            content_fp=content_fp,
        )
    )
    svcs.repo.upsert_exif(
        ExifRecord(
            photo_id=photo_id,
            aperture=2.0,
            shutter_s=1 / 500,
            iso=200,
            focal_eq=focal_eq,
            light_bucket=light_bucket,
        )
    )
    svcs.repo.upsert_metrics(
        MetricsRecord(
            content_fp=content_fp,
            engine_version=ENGINE_VERSION,
            sharp_p95=50.0,
            scene_tag=scene_tag,
        )
    )
    svcs.repo.add_verdict(
        photo_id=photo_id,
        source="engine",
        bucket="normal",
        confidence=0.95,
        reason_codes=[],
        increment_decision_version=False,
    )
    svcs.vector_index.upsert(content_fp, SIGLIP2_MODEL_ID, vec)


def test_bundled_scene_vocab_matrix_integrity() -> None:
    """Bundled `scene_vocab.npy` ships with ~40 normalized 768-d vectors matching SCENE_VOCAB."""
    assert SCENE_VOCAB_NPY_PATH.is_file()
    assert len(SCENE_VOCAB) >= 40
    entries, matrix = load_scene_vocab()
    assert matrix.shape == (len(entries), EMBEDDING_DIM)
    assert matrix.dtype == np.float32
    norms = np.linalg.norm(matrix, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)

    # Dot product of an entry's own vector picks that exact scene tag
    sunset_idx = next(i for i, e in enumerate(entries) if e.tag == "sunset_golden_hour")
    tag, score = classify_scene(matrix[sunset_idx], vocab_matrix=matrix)
    assert tag == "sunset_golden_hour"
    assert score > 0.99


def test_vector_index_get_and_list_content_fps_parity() -> None:
    """Both `SqliteVecIndex` and `NumpyVectorIndex` support `get()` and `list_content_fps()` identically."""
    v1 = _unit_vec({3: 0.8, 7: 0.6})
    v2 = _unit_vec({11: 1.0})

    with Database(":memory:") as db:
        db.migrate()
        np_idx = NumpyVectorIndex()
        np_idx.upsert_batch([("fp1", SIGLIP2_MODEL_ID, v1), ("fp2", SIGLIP2_MODEL_ID, v2)])

        assert np_idx.list_content_fps(SIGLIP2_MODEL_ID) == {"fp1", "fp2"}
        got_np = np_idx.get("fp1", SIGLIP2_MODEL_ID)
        assert got_np is not None
        assert np.allclose(got_np, v1, atol=1e-5)
        assert np_idx.get("missing", SIGLIP2_MODEL_ID) is None

        if db.sqlite_vec_available:
            sv_idx = SqliteVecIndex(db)
            sv_idx.upsert_batch([("fp1", SIGLIP2_MODEL_ID, v1), ("fp2", SIGLIP2_MODEL_ID, v2)])
            assert sv_idx.list_content_fps(SIGLIP2_MODEL_ID) == {"fp1", "fp2"}
            got_sv = sv_idx.get("fp1", SIGLIP2_MODEL_ID)
            assert got_sv is not None
            assert np.allclose(got_sv, v1, atol=1e-5)
            assert sv_idx.get("missing", SIGLIP2_MODEL_ID) is None


def test_filtered_search_cosine_ranking(tmp_path: Path) -> None:
    """Search filters by session/camera/date/light/focal FIRST, then ranks by descending cosine similarity."""
    q_sunset = _unit_vec({0: 1.0})
    fake = FakeEmbedder(text_vectors={"sunset": q_sunset, "日落": q_sunset})

    paths = get_app_paths(tmp_path / "home")
    with LightboxServices(app_paths=paths, embedder=fake) as svcs:
        # Session A: Sony camera, May 2026
        _seed_photo_with_vector(
            svcs,
            session_id="sess_a",
            session_name="Beach Trip",
            photo_id="p_high",
            stem="IMG_HIGH",
            content_fp="fp_high",
            camera_model="Sony A7IV",
            capture_iso="2026-05-10",
            light_bucket="daylight",
            focal_eq=35.0,
            vec=_unit_vec({0: 0.95, 1: 0.31}),  # cosine ~0.95 with q_sunset
            img_path=_create_jpg(tmp_path / "a" / "IMG_HIGH.jpg", (250, 140, 40)),
            scene_tag="sunset_golden_hour",
        )
        _seed_photo_with_vector(
            svcs,
            session_id="sess_a",
            session_name="Beach Trip",
            photo_id="p_mid",
            stem="IMG_MID",
            content_fp="fp_mid",
            camera_model="Sony A7IV",
            capture_iso="2026-05-12",
            light_bucket="daylight",
            focal_eq=85.0,
            vec=_unit_vec({0: 0.60, 1: 0.80}),  # cosine = 0.60
            img_path=_create_jpg(tmp_path / "a" / "IMG_MID.jpg", (180, 140, 120)),
            scene_tag="beach_ocean",
        )
        _seed_photo_with_vector(
            svcs,
            session_id="sess_a",
            session_name="Beach Trip",
            photo_id="p_low",
            stem="IMG_LOW",
            content_fp="fp_low",
            camera_model="Sony A7IV",
            capture_iso="2026-05-15",
            light_bucket="night",
            focal_eq=24.0,
            vec=_unit_vec({0: 0.10, 2: 0.995}),  # cosine ~0.10
            img_path=_create_jpg(tmp_path / "a" / "IMG_LOW.jpg", (15, 20, 40)),
            scene_tag="night_city",
        )
        # Session B: Fujifilm camera, June 2026, EVEN HIGHER similarity (0.99) to prove filters exclude it
        _seed_photo_with_vector(
            svcs,
            session_id="sess_b",
            session_name="Mountain Trip",
            photo_id="p_fuji",
            stem="IMG_FUJI",
            content_fp="fp_fuji",
            camera_model="Fujifilm X-T5",
            capture_iso="2026-06-20",
            light_bucket="daylight",
            focal_eq=50.0,
            vec=_unit_vec({0: 0.99, 3: 0.14}),  # cosine ~0.99
            img_path=_create_jpg(tmp_path / "b" / "IMG_FUJI.jpg", (255, 160, 50)),
            scene_tag="sunset_golden_hour",
        )

        # 1. Unfiltered Chinese query ranks all 4 strictly by cosine descending
        res_all = svcs.search.search_photos("日落")
        assert res_all.mode == "semantic"
        assert res_all.semantic_ready is True
        assert [m.photo_id for m in res_all.matches] == ["p_fuji", "p_high", "p_mid", "p_low"]
        assert res_all.matches[0].score > res_all.matches[1].score > res_all.matches[2].score

        # 2. Camera filter excludes p_fuji BEFORE ranking
        res_sony = svcs.search.search_photos("sunset", camera="Sony")
        assert [m.photo_id for m in res_sony.matches] == ["p_high", "p_mid", "p_low"]
        assert "Fujifilm X-T5" in res_sony.cameras and "Sony A7IV" in res_sony.cameras

        # 3. Date range filter (`2026-05-11`..`2026-05-31`) keeps only p_mid and p_low
        res_date = svcs.search.search_photos("sunset", date_from="2026-05-11", date_to="2026-05-31")
        assert [m.photo_id for m in res_date.matches] == ["p_mid", "p_low"]

        # 4. Session + light_bucket filter keeps only daylight Sony shots in sess_a
        res_light = svcs.search.search_photos("sunset", session_id="sess_a", light_bucket="daylight")
        assert [m.photo_id for m in res_light.matches] == ["p_high", "p_mid"]

        # 5. Focal length filter
        res_focal = svcs.search.search_photos("sunset", focal_min=40.0, focal_max=90.0)
        assert [m.photo_id for m in res_focal.matches] == ["p_fuji", "p_mid"]


def test_find_similar_excludes_anchor_and_ranks_by_cosine(tmp_path: Path) -> None:
    """`find_similar` never includes the anchor photo itself and ranks neighbors by cosine similarity."""
    fake = FakeEmbedder(forbid_text_tower=True)
    paths = get_app_paths(tmp_path / "home")
    with LightboxServices(app_paths=paths, embedder=fake) as svcs:
        anchor_vec = _unit_vec({4: 1.0})
        near_vec = _unit_vec({4: 0.96, 5: 0.28})
        mid_vec = _unit_vec({4: 0.50, 5: 0.866})
        far_vec = _unit_vec({6: 1.0})

        _seed_photo_with_vector(
            svcs,
            session_id="sess_1",
            session_name="Shoot 1",
            photo_id="p_anchor",
            stem="ANCHOR",
            content_fp="fp_anchor",
            camera_model="Sony A7IV",
            capture_iso="2026-05-10",
            light_bucket="daylight",
            focal_eq=50.0,
            vec=anchor_vec,
            img_path=_create_jpg(tmp_path / "s1" / "ANCHOR.jpg", (200, 100, 80)),
        )
        _seed_photo_with_vector(
            svcs,
            session_id="sess_1",
            session_name="Shoot 1",
            photo_id="p_near",
            stem="NEAR",
            content_fp="fp_near",
            camera_model="Sony A7IV",
            capture_iso="2026-05-10",
            light_bucket="daylight",
            focal_eq=50.0,
            vec=near_vec,
            img_path=_create_jpg(tmp_path / "s1" / "NEAR.jpg", (195, 105, 85)),
        )
        _seed_photo_with_vector(
            svcs,
            session_id="sess_2",
            session_name="Shoot 2",
            photo_id="p_mid",
            stem="MID",
            content_fp="fp_mid",
            camera_model="Canon R5",
            capture_iso="2026-05-11",
            light_bucket="daylight",
            focal_eq=50.0,
            vec=mid_vec,
            img_path=_create_jpg(tmp_path / "s2" / "MID.jpg", (150, 120, 100)),
        )
        _seed_photo_with_vector(
            svcs,
            session_id="sess_1",
            session_name="Shoot 1",
            photo_id="p_far",
            stem="FAR",
            content_fp="fp_far",
            camera_model="Sony A7IV",
            capture_iso="2026-05-10",
            light_bucket="night",
            focal_eq=24.0,
            vec=far_vec,
            img_path=_create_jpg(tmp_path / "s1" / "FAR.jpg", (10, 10, 30)),
        )

        # Cross-session find_similar: anchor excluded, sorted near -> mid -> far
        sim_all = svcs.search.find_similar("p_anchor", limit=10)
        assert sim_all.mode == "semantic"
        assert sim_all.semantic_ready is True
        assert sim_all.anchor_id == "p_anchor"
        returned_ids = [m.photo_id for m in sim_all.similar]
        assert "p_anchor" not in returned_ids
        assert returned_ids == ["p_near", "p_mid", "p_far"]
        assert fake.embed_texts_calls == 0  # find_similar never touches text tower

        # Session-scoped find_similar excludes other session and still excludes anchor
        sim_s1 = svcs.search.find_similar("p_anchor", limit=10, session_id="sess_1")
        assert [m.photo_id for m in sim_s1.similar] == ["p_near", "p_far"]


def test_tagging_and_scan_embed_never_load_text_tower(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scanning with `enable_embed=True` and `index_session` NEVER load or download the text tower."""
    shoot_dir = tmp_path / "shoot"
    _create_jpg(shoot_dir / "IMG_001.jpg", (240, 130, 40))
    _create_jpg(shoot_dir / "IMG_002.jpg", (20, 120, 40))

    entries, vocab_mat = load_scene_vocab()
    sunset_idx = next(i for i, e in enumerate(entries) if e.tag == "sunset_golden_hour")
    forest_idx = next(i for i, e in enumerate(entries) if e.tag == "forest_nature")

    # Part A: Real Siglip2Embedder with mocked ONNX vision session + forbidden text loader
    paths = get_app_paths(tmp_path / "home_real_mock")
    embedder = Siglip2Embedder(paths, allow_download=False)

    fake_vision_file = paths.models_dir / "siglip2-base" / "onnx" / "vision_model_int8.onnx"
    fake_vision_file.parent.mkdir(parents=True, exist_ok=True)
    fake_vision_file.write_bytes(b"fake-onnx")

    call_idx = {"n": 0}

    def _fake_ort_session(model_path: str, providers: list[str] | None = None) -> MagicMock:
        assert "vision_model" in model_path
        sess = MagicMock()
        inp = MagicMock()
        inp.name = "pixel_values"
        out = MagicMock()
        out.name = "pooler_output"
        sess.get_inputs.return_value = [inp]
        sess.get_outputs.return_value = [out]

        def _run(_out_names: list[str], feed: dict[str, Any]) -> list[NDArray[np.float32]]:
            batch_n = int(feed["pixel_values"].shape[0])
            rows = []
            for _ in range(batch_n):
                vec = vocab_mat[sunset_idx] if call_idx["n"] % 2 == 0 else vocab_mat[forest_idx]
                call_idx["n"] += 1
                rows.append(vec)
            return [np.stack(rows, axis=0)]

        sess.run.side_effect = _run
        return sess

    import onnxruntime as ort

    monkeypatch.setattr(ort, "InferenceSession", _fake_ort_session)
    text_resolver_spy = MagicMock(
        side_effect=AssertionError("Text tower path resolution must NOT run during tagging!")
    )
    monkeypatch.setattr(embedder, "_resolve_text_paths", text_resolver_spy)

    with LightboxServices(app_paths=paths, embedder=embedder) as svcs:
        scan_res = svcs.scan.scan_folder(shoot_dir, max_workers=1, enable_embed=True)
        assert scan_res.embedded_count == 2
        assert embedder.vision_loaded is True
        assert embedder.text_loaded is False
        text_resolver_spy.assert_not_called()

        # Both photos received deterministic scene_tags in SQLite metrics & 768-d vectors in VectorIndex
        photos = svcs.repo.list_photos(scan_res.session_id)
        tags = {
            (svcs.repo.get_metrics(p.primary_content_fp, ENGINE_VERSION) or MetricsRecord("", "")).scene_tag
            for p in photos
        }
        assert tags == {"sunset_golden_hour", "forest_nature"}
        assert svcs.vector_index.count(SIGLIP2_MODEL_ID) == 2

        # Re-running index_session skips already-indexed photos and still never touches text tower
        re_idx = svcs.search.index_session(scan_res.session_id)
        assert re_idx.indexed_count == 0
        assert re_idx.skipped_count == 2
        assert embedder.text_loaded is False
        text_resolver_spy.assert_not_called()


def test_api_and_tool_registry_semantic_search_and_similar(tmp_path: Path) -> None:
    """Verify `/api/search`, `/api/photos/{id}/similar`, `/api/sessions/{id}/embed`, and MCP ToolRegistry."""
    shoot_dir = tmp_path / "shoot"
    _create_jpg(shoot_dir / "SUNSET_01.jpg", (250, 140, 40))
    _create_jpg(shoot_dir / "SUNSET_02.jpg", (245, 135, 45))
    _create_jpg(shoot_dir / "FOREST_01.jpg", (20, 120, 40))

    entries, vocab_mat = load_scene_vocab()
    sunset_vec = vocab_mat[next(i for i, e in enumerate(entries) if e.tag == "sunset_golden_hour")]
    forest_vec = vocab_mat[next(i for i, e in enumerate(entries) if e.tag == "forest_nature")]

    # Map content_fp/preview files deterministically once scanned
    paths = get_app_paths(tmp_path / "home_api")
    with LightboxServices(app_paths=paths) as svcs:
        scan_res = svcs.scan.scan_folder(shoot_dir, max_workers=1, enable_embed=False)
        photos = svcs.repo.list_photos(scan_res.session_id)
        fp_by_stem = {p.stem: p.primary_content_fp for p in photos}
        pid_by_stem = {p.stem: p.id for p in photos}

    fake = FakeEmbedder(
        image_vectors={
            fp_by_stem["SUNSET_01"]: sunset_vec,
            fp_by_stem["SUNSET_02"]: _unit_vec({0: 0.05}) * 0.05 + sunset_vec * 0.99,
            fp_by_stem["FOREST_01"]: forest_vec,
        },
        text_vectors={
            "sunset": sunset_vec,
            "日落": sunset_vec,
            "forest": forest_vec,
        },
    )

    app = create_app(app_paths=paths, port=8765, embedder_factory=lambda _svcs: fake)
    client = TestClient(app, base_url="http://127.0.0.1:8765")
    auth_resp = client.post(
        "/api/auth/session",
        headers={"X-Lightbox-Local-Confirm": "1", "Origin": "http://127.0.0.1:8765"},
    )
    assert auth_resp.status_code == 200
    csrf = auth_resp.json()["csrf_token"]
    write_headers = {"X-Lightbox-CSRF": csrf, "Origin": "http://127.0.0.1:8765"}

    # 1. Index session via POST /api/sessions/{id}/embed
    embed_resp = client.post(f"/api/sessions/{scan_res.session_id}/embed", headers=write_headers)
    assert embed_resp.status_code == 200
    embed_body = embed_resp.json()
    assert embed_body["indexed_count"] == 3
    assert embed_body["tagged_count"] == 3
    assert fake.embed_texts_calls == 0

    # 2. Semantic search via GET /api/search
    search_resp = client.get("/api/search", params={"q": "日落", "lang": "zh-CN"})
    assert search_resp.status_code == 200
    s_body = search_resp.json()
    assert s_body["mode"] == "semantic"
    assert s_body["semantic_ready"] is True
    assert s_body["matches"][0]["stem"] in {"SUNSET_01", "SUNSET_02"}
    assert s_body["matches"][-1]["stem"] == "FOREST_01"
    assert s_body["matches"][0]["scene_tag"] == "sunset_golden_hour"

    # 3. Similar photos via GET /api/photos/{id}/similar
    anchor_id = pid_by_stem["SUNSET_01"]
    sim_resp = client.get(f"/api/photos/{anchor_id}/similar", params={"limit": 5})
    assert sim_resp.status_code == 200
    sim_body = sim_resp.json()
    assert sim_body["mode"] == "semantic"
    assert sim_body["anchor_id"] == anchor_id
    sim_stems = [m["stem"] for m in sim_body["similar"]]
    assert "SUNSET_01" not in sim_stems
    assert sim_stems[0] == "SUNSET_02"

    # 4. ToolRegistry (MCP surface) routes search_photos & find_similar through SearchService
    registry = ToolRegistry()
    with LightboxServices(app_paths=paths, embedder=fake, migrate=False) as svcs:
        ctx = ToolContext(services=svcs, surface="mcp:test")
        t_search = registry.execute("lightbox_search_photos", {"query": "forest"}, ctx)
        assert not t_search.is_error
        assert t_search.data["mode"] == "semantic"
        assert t_search.data["semantic_ready"] is True
        assert t_search.data["matches"][0]["stem"] == "FOREST_01"

        t_sim = registry.execute("lightbox_find_similar", {"photo_id": anchor_id, "limit": 5}, ctx)
        assert not t_sim.is_error
        assert t_sim.data["mode"] == "semantic"
        assert all(m["photo_id"] != anchor_id for m in t_sim.data["similar"])
        assert t_sim.data["similar"][0]["stem"] == "SUNSET_02"


def test_sqlite_vec_over_4096_vectors_filtered_and_similar(tmp_path: Path) -> None:
    """Regression: >4096 indexed vectors must never trigger sqlite-vec's k<=4096 OperationalError."""
    n_vectors = 4200
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((n_vectors, EMBEDDING_DIM)).astype(np.float32)
    # Force deterministic top-2 vectors aligned with axis 0
    raw[10] = _unit_vec({0: 0.99, 1: 0.14})
    raw[2500] = _unit_vec({0: 0.92, 1: 0.39})
    # Zero out axis 0 on all other random rows so top-2 are unambiguous
    mask = np.ones(n_vectors, dtype=bool)
    mask[[10, 2500]] = False
    raw[mask, 0] = 0.0
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    vecs = (raw / np.maximum(norms, np.float32(1e-12))).astype(np.float32)

    q = _unit_vec({0: 1.0})

    with Database(tmp_path / "big.db") as db:
        db.migrate()
        if not db.sqlite_vec_available:
            pytest.skip("sqlite-vec extension unavailable in this environment")
        idx = SqliteVecIndex(db)
        idx.upsert_batch([(f"fp_{i:05d}", SIGLIP2_MODEL_ID, vecs[i]) for i in range(n_vectors)])
        assert idx.count(SIGLIP2_MODEL_ID) == n_vectors

        # 1. Filtered query over ALL 4200 content_fps (previously crashed with k=4200 > 4096)
        all_fps = {f"fp_{i:05d}" for i in range(n_vectors)}
        res_all = idx.query(q, SIGLIP2_MODEL_ID, top_k=5, filter_content_fps=all_fps)
        assert [m.content_fp for m in res_all[:2]] == ["fp_00010", "fp_02500"]

        # 2. Filtered query over a subset excluding fp_00010 (like find_similar excluding anchor)
        without_anchor = all_fps - {"fp_00010"}
        res_sim = idx.query(
            vecs[10], SIGLIP2_MODEL_ID, top_k=5, filter_content_fps=without_anchor
        )
        assert res_sim[0].content_fp == "fp_02500"
        assert all(m.content_fp != "fp_00010" for m in res_sim)

        # 3. Filtered query over a small subset (<500)
        small_fps = {"fp_00001", "fp_02500", "fp_04199"}
        res_small = idx.query(q, SIGLIP2_MODEL_ID, top_k=3, filter_content_fps=small_fps)
        assert res_small[0].content_fp == "fp_02500"


def test_shared_embedder_loads_models_only_once_across_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: repeated API search/embed requests reuse one process-level Siglip2Embedder."""
    clear_shared_siglip2_embedder_cache()
    try:
        shoot_dir = tmp_path / "shoot"
        _create_jpg(shoot_dir / "IMG_01.jpg", (240, 130, 40))

        paths = get_app_paths(tmp_path / "home_shared")
        model_root = paths.models_dir / "siglip2-base"
        (model_root / "onnx").mkdir(parents=True, exist_ok=True)
        (model_root / "onnx" / "vision_model_int8.onnx").write_bytes(b"fake-vis")
        (model_root / "onnx" / "text_model_int8.onnx").write_bytes(b"fake-txt")
        (model_root / "tokenizer.json").write_bytes(b"{}")

        load_counts = {"vision": 0, "text": 0, "tokenizer": 0}
        unit0 = _unit_vec({0: 1.0})

        def _fake_ort(model_path: str, providers: list[str] | None = None) -> MagicMock:
            sess = MagicMock()
            inp = MagicMock()
            out = MagicMock()
            out.name = "pooler_output"
            if "vision_model" in model_path:
                load_counts["vision"] += 1
                inp.name = "pixel_values"
                sess.run.side_effect = lambda _o, feed: [
                    np.repeat(unit0[None, :], int(feed["pixel_values"].shape[0]), axis=0)
                ]
            else:
                load_counts["text"] += 1
                inp.name = "input_ids"
                sess.run.side_effect = lambda _o, feed: [
                    np.repeat(unit0[None, :], int(feed["input_ids"].shape[0]), axis=0)
                ]
            sess.get_inputs.return_value = [inp]
            sess.get_outputs.return_value = [out]
            return sess

        class _FakeTokenizer:
            def token_to_id(self, tok: str) -> int:
                return 0 if tok == "<pad>" else 1

            def enable_truncation(self, max_length: int) -> None:
                _ = max_length

            def enable_padding(self, pad_id: int, pad_token: str, length: int) -> None:
                _ = (pad_id, pad_token, length)

            def encode(self, _text: str) -> Any:
                enc = MagicMock()
                enc.ids = [10, 11, 1] + [0] * 61
                return enc

        def _fake_tok_from_file(_path: str) -> _FakeTokenizer:
            load_counts["tokenizer"] += 1
            return _FakeTokenizer()

        import onnxruntime as ort
        from tokenizers import Tokenizer

        monkeypatch.setattr(ort, "InferenceSession", _fake_ort)
        monkeypatch.setattr(Tokenizer, "from_file", staticmethod(_fake_tok_from_file))

        with LightboxServices(app_paths=paths) as svcs:
            scan_res = svcs.scan.scan_folder(shoot_dir, max_workers=1, enable_embed=False)

        app = create_app(app_paths=paths, port=8765)
        client = TestClient(app, base_url="http://127.0.0.1:8765")
        auth_resp = client.post(
            "/api/auth/session",
            headers={"X-Lightbox-Local-Confirm": "1", "Origin": "http://127.0.0.1:8765"},
        )
        csrf = auth_resp.json()["csrf_token"]
        headers = {"X-Lightbox-CSRF": csrf, "Origin": "http://127.0.0.1:8765"}

        # First embed loads vision tower once and never loads text tower
        r_embed = client.post(f"/api/sessions/{scan_res.session_id}/embed", headers=headers)
        assert r_embed.status_code == 200
        assert load_counts == {"vision": 1, "text": 0, "tokenizer": 0}

        # Three consecutive searches across fresh LightboxServices instances load text tower ONLY ONCE
        for query in ("sunset", "日落", "forest"):
            r_search = client.get("/api/search", params={"q": query})
            assert r_search.status_code == 200
            assert r_search.json()["mode"] == "semantic"

        assert load_counts == {"vision": 1, "text": 1, "tokenizer": 1}
    finally:
        clear_shared_siglip2_embedder_cache()


def test_download_failure_wrapped_as_embedder_error_and_api_503(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: HF download errors wrap into EmbedderError and return HTTP 503 (never 500)."""
    clear_shared_siglip2_embedder_cache()
    try:
        import huggingface_hub

        def _fail_download(*_args: Any, **_kwargs: Any) -> str:
            raise ConnectionError("offline or invalid HF_ENDPOINT")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fail_download)

        paths = get_app_paths(tmp_path / "home_offline")
        embedder = Siglip2Embedder(paths, allow_download=True)
        with pytest.raises(EmbedderError, match="Failed to download SigLIP 2"):
            embedder.embed_images([Image.new("RGB", (64, 64))])
        with pytest.raises(EmbedderError, match="Failed to download SigLIP 2"):
            embedder.embed_texts(["sunset"])

        # Seed a photo + vector so both /embed (vision) and /search (text) attempt HF download
        shoot_dir = tmp_path / "shoot"
        img_p = _create_jpg(shoot_dir / "IMG_OFFLINE.jpg", (120, 120, 120))
        with LightboxServices(app_paths=paths) as svcs:
            scan_res = svcs.scan.scan_folder(shoot_dir, max_workers=1, enable_embed=False)
            photo = svcs.repo.list_photos(scan_res.session_id)[0]
            svcs.vector_index.upsert(photo.primary_content_fp, SIGLIP2_MODEL_ID, _unit_vec({0: 1.0}))
        assert img_p.is_file()

        app = create_app(app_paths=paths, port=8765)
        client = TestClient(app, base_url="http://127.0.0.1:8765")
        auth_resp = client.post(
            "/api/auth/session",
            headers={"X-Lightbox-Local-Confirm": "1", "Origin": "http://127.0.0.1:8765"},
        )
        csrf = auth_resp.json()["csrf_token"]
        headers = {"X-Lightbox-CSRF": csrf, "Origin": "http://127.0.0.1:8765"}

        # Search triggers text tower download -> 503 (not 500)
        r_search = client.get("/api/search", params={"q": "sunset"})
        assert r_search.status_code == 503
        assert "HF_ENDPOINT" in r_search.json()["detail"]

        # Unindexed second photo triggers vision tower download -> 503 (not 500)
        _create_jpg(shoot_dir / "IMG_OFFLINE_2.jpg", (10, 10, 10))
        with LightboxServices(app_paths=paths) as svcs:
            svcs.scan.scan_folder(shoot_dir, max_workers=1, enable_embed=False)
        r_embed = client.post(f"/api/sessions/{scan_res.session_id}/embed", headers=headers)
        assert r_embed.status_code == 503
        assert "HF_ENDPOINT" in r_embed.json()["detail"]
    finally:
        clear_shared_siglip2_embedder_cache()
