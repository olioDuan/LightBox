"""SearchService: SigLIP 2 local semantic indexing, scene tagging, filtered search, and find_similar (PLAN T11)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lightbox.core import (
    ENGINE_VERSION,
    compute_dhash,
    load_analysis_gray,
    preview_cache_paths,
)
from lightbox.i18n import render_reasons
from lightbox.models import (
    Embedder,
    classify_scene,
    get_scene_label,
    load_scene_vocab,
)
from lightbox.paths import AppPaths, get_app_paths
from lightbox.store import (
    ExifRecord,
    LightboxRepository,
    MetricsRecord,
    PhotoRecord,
    SessionRecord,
    VectorIndex,
)

ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class EmbedSessionResult:
    """Summary of L2 SigLIP 2 image embedding and zero-text-tower scene tagging for a session."""

    session_id: str
    indexed_count: int
    skipped_count: int
    tagged_count: int
    total_photos: int
    model_id: str
    elapsed_s: float


@dataclass(frozen=True, slots=True)
class SearchMatch:
    """Single search or find-similar photo match with metadata and cosine score."""

    photo_id: str
    session_id: str
    session_name: str
    stem: str
    rel_dir: str
    bucket: str
    confidence: float
    reason_codes: tuple[str, ...]
    reason_labels: tuple[str, ...]
    camera_model: str | None
    lens_model: str | None
    capture_time: float | None
    light_bucket: str | None
    focal_eq: float | None
    scene_tag: str | None
    scene_label: str | None
    exif_brief: str
    score: float
    relation: str = "semantic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "photo_id": self.photo_id,
            "session_id": self.session_id,
            "session_name": self.session_name,
            "stem": self.stem,
            "rel_dir": self.rel_dir,
            "bucket": self.bucket,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "reason_labels": list(self.reason_labels),
            "camera_model": self.camera_model,
            "lens_model": self.lens_model,
            "capture_time": self.capture_time,
            "light_bucket": self.light_bucket,
            "focal_eq": self.focal_eq,
            "scene_tag": self.scene_tag,
            "scene_label": self.scene_label,
            "exif_brief": self.exif_brief,
            "score": self.score,
            "similarity": self.score,
            "relation": self.relation,
        }


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Structured result of `SearchService.search_photos`."""

    query: str
    mode: str  # 'semantic' | 'filter' | 'fts_exif'
    semantic_ready: bool
    indexed_count: int
    total_candidates: int
    cameras: tuple[str, ...]
    matches: tuple[SearchMatch, ...]


@dataclass(frozen=True, slots=True)
class SimilarResult:
    """Structured result of `SearchService.find_similar`."""

    anchor: SearchMatch
    anchor_id: str
    session_id: str
    mode: str  # 'semantic' | 'burst_dhash'
    semantic_ready: bool
    similar: tuple[SearchMatch, ...]


def _parse_iso_date_bounds(date_str: str, *, end_of_day: bool = False) -> float:
    """Parse `YYYY-MM-DD` into a UTC POSIX timestamp (start or end of day)."""
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date {date_str!r}; expected YYYY-MM-DD.") from exc
    ts = dt.timestamp()
    return ts + 86399.999 if end_of_day else ts


def _format_exif_brief(photo: PhotoRecord, exif: ExifRecord | None) -> str:
    parts: list[str] = []
    if photo.camera_model:
        parts.append(photo.camera_model)
    if exif is not None:
        if exif.focal_eq is not None:
            parts.append(f"{int(round(exif.focal_eq))}mm")
        if exif.aperture is not None:
            parts.append(f"f/{exif.aperture:g}")
        if exif.shutter_s is not None and exif.shutter_s > 0:
            if exif.shutter_s >= 1.0:
                parts.append(f"{exif.shutter_s:g}s")
            else:
                parts.append(f"1/{int(round(1.0 / exif.shutter_s))}s")
        if exif.iso is not None:
            parts.append(f"ISO {exif.iso}")
    return " · ".join(parts)


def _resolve_embedding_image_path(
    repo: LightboxRepository,
    app_paths: AppPaths,
    photo: PhotoRecord,
) -> Path | None:
    """Choose the fastest RGB image source for SigLIP 2 (256x256 input): cached grid -> loupe -> asset."""
    grid_p, loupe_p, _analysis_p = preview_cache_paths(
        app_paths.cache_dir, photo.primary_content_fp
    )
    if grid_p.is_file():
        return grid_p
    if loupe_p.is_file():
        return loupe_p
    assets = repo.list_assets_for_photo(photo.id)
    for kind in ("jpg", "heic", "raw"):
        for a in assets:
            if a.kind == kind and Path(a.abs_path).is_file():
                return Path(a.abs_path)
    return None


class SearchService:
    """Local SigLIP 2 vector indexing, scene tagging, filtered semantic search, and similarity lookup."""

    def __init__(
        self,
        repo: LightboxRepository,
        vector_index: VectorIndex,
        embedder: Embedder,
        *,
        app_paths: AppPaths | None = None,
        vocab_matrix: NDArray[np.float32] | None = None,
    ) -> None:
        self.repo = repo
        self.vector_index = vector_index
        self.embedder = embedder
        self.app_paths = app_paths or get_app_paths()
        self._vocab_matrix = vocab_matrix

    def _get_vocab_matrix(self) -> NDArray[np.float32]:
        if self._vocab_matrix is not None:
            return self._vocab_matrix
        return load_scene_vocab()[1]

    def index_session(
        self,
        session_id: str,
        *,
        on_progress: ProgressCallback | None = None,
        batch_size: int = 8,
    ) -> EmbedSessionResult:
        """Index all photos in `session_id` with vision tower only and write `metrics.scene_tag`.

        Never loads the SigLIP 2 text tower: scene tags are computed via dot product against the
        bundled `scene_vocab.npy` matrix.
        """
        started = time.monotonic()
        session = self.repo.get_session(session_id)
        if session is None:
            raise ValueError(f"Unknown session_id: {session_id}")

        photos = self.repo.list_photos(session_id)
        total = len(photos)
        if total == 0:
            return EmbedSessionResult(
                session_id=session_id,
                indexed_count=0,
                skipped_count=0,
                tagged_count=0,
                total_photos=0,
                model_id=self.embedder.model_id,
                elapsed_s=round(time.monotonic() - started, 3),
            )

        existing_fps = self.vector_index.list_content_fps(self.embedder.model_id)
        vocab_matrix = self._get_vocab_matrix()

        to_embed: list[tuple[PhotoRecord, Path]] = []
        skipped = 0
        tagged = 0

        # Deduplicate by content_fp within the same session while still ensuring every photo gets tagged
        pending_by_fp: dict[str, list[PhotoRecord]] = {}
        fp_image_path: dict[str, Path] = {}

        for photo in photos:
            fp = photo.primary_content_fp
            metrics = self.repo.get_metrics(fp, ENGINE_VERSION)
            has_vec = fp in existing_fps
            has_tag = metrics is not None and bool(metrics.scene_tag)

            if has_vec and has_tag:
                skipped += 1
                continue

            if has_vec and not has_tag:
                stored_vec = self.vector_index.get(fp, self.embedder.model_id)
                if stored_vec is not None:
                    tag, _score = classify_scene(stored_vec, vocab_matrix=vocab_matrix)
                    self._persist_scene_tag(session, photo, tag)
                    tagged += 1
                    skipped += 1
                    continue

            img_path = _resolve_embedding_image_path(self.repo, self.app_paths, photo)
            if img_path is None:
                continue
            if fp not in pending_by_fp:
                pending_by_fp[fp] = []
                fp_image_path[fp] = img_path
                to_embed.append((photo, img_path))
            pending_by_fp[fp].append(photo)

        done_count = skipped
        if on_progress is not None:
            on_progress("embed", done_count, total)

        indexed = 0
        step = max(1, batch_size)
        for start in range(0, len(to_embed), step):
            chunk = to_embed[start : start + step]
            paths = [img_p for _, img_p in chunk]
            vectors = self.embedder.embed_images(paths)

            upsert_items: list[tuple[str, str, NDArray[np.float32]]] = []
            for idx, (primary_photo, _img_p) in enumerate(chunk):
                fp = primary_photo.primary_content_fp
                vec = vectors[idx]
                upsert_items.append((fp, self.embedder.model_id, vec))

                tag, _score = classify_scene(vec, vocab_matrix=vocab_matrix)
                for p_same in pending_by_fp[fp]:
                    self._persist_scene_tag(session, p_same, tag)
                    tagged += 1
                    done_count += 1
                indexed += 1

            self.vector_index.upsert_batch(upsert_items)
            if on_progress is not None:
                on_progress("embed", min(done_count, total), total)

        return EmbedSessionResult(
            session_id=session_id,
            indexed_count=indexed,
            skipped_count=skipped,
            tagged_count=tagged,
            total_photos=total,
            model_id=self.embedder.model_id,
            elapsed_s=round(time.monotonic() - started, 3),
        )

    def _persist_scene_tag(
        self,
        session: SessionRecord,
        photo: PhotoRecord,
        scene_tag: str,
    ) -> None:
        existing = self.repo.get_metrics(photo.primary_content_fp, ENGINE_VERSION)
        if existing is not None:
            self.repo.upsert_metrics(
                MetricsRecord(
                    content_fp=existing.content_fp,
                    engine_version=existing.engine_version,
                    sharp_p95=existing.sharp_p95,
                    sharp_z=existing.sharp_z,
                    noise_sigma=existing.noise_sigma,
                    motion_aniso=existing.motion_aniso,
                    motion_state=existing.motion_state,
                    exposure=existing.exposure,
                    texture=existing.texture,
                    face_count=existing.face_count,
                    face_sharp_min=existing.face_sharp_min,
                    scene_tag=scene_tag,
                )
            )
        else:
            self.repo.upsert_metrics(
                MetricsRecord(
                    content_fp=photo.primary_content_fp,
                    engine_version=ENGINE_VERSION,
                    scene_tag=scene_tag,
                )
            )
        self.repo.upsert_session_fts(
            photo_id=photo.id,
            session_name=session.name,
            rel_dir=photo.rel_dir,
            stem=photo.stem,
            scene_tag=scene_tag,
        )

    def _build_match(
        self,
        photo: PhotoRecord,
        session_name: str,
        *,
        verdict: Any,
        exif: ExifRecord | None,
        metrics: MetricsRecord | None,
        score: float,
        lang: str,
        relation: str = "semantic",
    ) -> SearchMatch:
        bucket = verdict.bucket if verdict is not None else "normal"
        conf = float(verdict.confidence) if verdict is not None else 1.0
        reason_codes = tuple(verdict.reason_codes) if verdict is not None else ()
        scene_tag = metrics.scene_tag if metrics is not None else None
        return SearchMatch(
            photo_id=photo.id,
            session_id=photo.session_id,
            session_name=session_name,
            stem=photo.stem,
            rel_dir=photo.rel_dir,
            bucket=bucket,
            confidence=round(conf, 4),
            reason_codes=reason_codes,
            reason_labels=tuple(render_reasons(reason_codes, lang=lang)),
            camera_model=photo.camera_model,
            lens_model=photo.lens_model,
            capture_time=photo.capture_time,
            light_bucket=exif.light_bucket if exif is not None else None,
            focal_eq=exif.focal_eq if exif is not None else None,
            scene_tag=scene_tag,
            scene_label=get_scene_label(scene_tag, lang=lang),
            exif_brief=_format_exif_brief(photo, exif),
            score=round(float(score), 4),
            relation=relation,
        )

    def search_photos(
        self,
        query: str = "",
        *,
        session_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        camera: str | None = None,
        light_bucket: str | None = None,
        focal_min: float | None = None,
        focal_max: float | None = None,
        limit: int = 20,
        lang: str = "zh-CN",
    ) -> SearchResult:
        """Filter photos by EXIF/session/date first, then rank candidates by SigLIP 2 cosine similarity."""
        clamped_limit = max(1, min(limit, 100))
        q_clean = query.strip()
        cam_needle = camera.strip().lower() if camera else None
        light_needle = light_bucket.strip().lower() if light_bucket else None
        min_ts = _parse_iso_date_bounds(date_from, end_of_day=False) if date_from else None
        max_ts = _parse_iso_date_bounds(date_to, end_of_day=True) if date_to else None

        if session_id:
            sess = self.repo.get_session(session_id)
            if sess is None:
                raise ValueError(f"Unknown session_id: {session_id}")
            sessions = [sess]
        else:
            sessions = self.repo.list_sessions()

        all_cameras: set[str] = set()
        candidates: list[
            tuple[PhotoRecord, SessionRecord, Any, ExifRecord | None, MetricsRecord | None]
        ] = []

        for sess in sessions:
            quarantined = self.repo.list_quarantined_photo_ids(sess.id)
            verdicts = self.repo.list_effective_verdicts(sess.id)
            for p in self.repo.list_photos(sess.id):
                if p.id in quarantined:
                    continue
                if p.camera_model:
                    all_cameras.add(p.camera_model)
                if min_ts is not None and (p.capture_time is None or p.capture_time < min_ts):
                    continue
                if max_ts is not None and (p.capture_time is None or p.capture_time > max_ts):
                    continue
                if cam_needle and cam_needle not in (p.camera_model or "").lower():
                    continue
                exif = self.repo.get_exif(p.id)
                if light_needle and (exif is None or (exif.light_bucket or "").lower() != light_needle):
                    continue
                if focal_min is not None and (exif is None or exif.focal_eq is None or exif.focal_eq < focal_min):
                    continue
                if focal_max is not None and (exif is None or exif.focal_eq is None or exif.focal_eq > focal_max):
                    continue
                metrics = self.repo.get_metrics(p.primary_content_fp, ENGINE_VERSION)
                candidates.append((p, sess, verdicts.get(p.id), exif, metrics))

        sorted_cameras = tuple(sorted(all_cameras))
        indexed_fps = self.vector_index.list_content_fps(self.embedder.model_id)
        candidate_fps = {p.primary_content_fp for p, *_ in candidates}
        indexed_candidate_fps = candidate_fps & indexed_fps
        indexed_count = len(indexed_candidate_fps)

        if not candidates:
            return SearchResult(
                query=q_clean,
                mode="semantic" if (q_clean and indexed_fps) else ("filter" if not q_clean else "fts_exif"),
                semantic_ready=bool(indexed_fps),
                indexed_count=0,
                total_candidates=0,
                cameras=sorted_cameras,
                matches=(),
            )

        # Case 1: Empty query -> browse filtered candidates by capture time descending
        if not q_clean:
            ordered = sorted(
                candidates,
                key=lambda row: (-(row[0].capture_time or 0.0), row[0].rel_dir, row[0].stem, row[0].id),
            )
            matches = tuple(
                self._build_match(
                    p,
                    sess.name,
                    verdict=v,
                    exif=ex,
                    metrics=m,
                    score=1.0,
                    lang=lang,
                    relation="filter",
                )
                for p, sess, v, ex, m in ordered[:clamped_limit]
            )
            return SearchResult(
                query=q_clean,
                mode="filter",
                semantic_ready=indexed_count > 0,
                indexed_count=indexed_count,
                total_candidates=len(candidates),
                cameras=sorted_cameras,
                matches=matches,
            )

        # Case 2: Semantic query over indexed candidates (filter first -> cosine rank)
        if indexed_candidate_fps:
            q_vec = self.embedder.embed_texts([q_clean])[0]
            vec_matches = self.vector_index.query(
                q_vec,
                model_id=self.embedder.model_id,
                top_k=len(indexed_candidate_fps),
                filter_content_fps=indexed_candidate_fps,
            )
            score_by_fp = {vm.content_fp: vm.score for vm in vec_matches}

            ranked: list[tuple[float, SearchMatch]] = []
            for p, sess, v, ex, m in candidates:
                sim = score_by_fp.get(p.primary_content_fp)
                if sim is None:
                    continue
                ranked.append(
                    (
                        sim,
                        self._build_match(
                            p,
                            sess.name,
                            verdict=v,
                            exif=ex,
                            metrics=m,
                            score=sim,
                            lang=lang,
                            relation="semantic",
                        ),
                    )
                )
            ranked.sort(key=lambda pair: (-pair[0], pair[1].rel_dir, pair[1].stem, pair[1].photo_id))
            return SearchResult(
                query=q_clean,
                mode="semantic",
                semantic_ready=True,
                indexed_count=indexed_count,
                total_candidates=len(candidates),
                cameras=sorted_cameras,
                matches=tuple(match for _, match in ranked[:clamped_limit]),
            )

        # Case 3: Graceful FTS5 / EXIF metadata fallback when no vectors have been indexed yet
        fts_ranks: dict[str, int] = {}
        tokens = [tok.strip('"') for tok in q_clean.split() if tok.strip('"')]
        if tokens:
            fts_expr = " OR ".join(f'"{tok}"*' for tok in tokens)
            try:
                hits = self.repo.search_session_fts(fts_expr, limit=200)
                fts_ranks = {pid: idx for idx, pid in enumerate(hits)}
            except Exception:
                fts_ranks = {}

        q_lower = q_clean.lower()
        scored_fallback: list[tuple[float, SearchMatch]] = []
        for p, sess, v, ex, m in candidates:
            reasons = v.reason_codes if v is not None else ()
            haystack = " ".join(
                filter(
                    None,
                    [
                        sess.name,
                        p.rel_dir,
                        p.stem,
                        p.camera_model or "",
                        p.lens_model or "",
                        m.scene_tag if m and m.scene_tag else "",
                        get_scene_label(m.scene_tag, lang="zh-CN") if m and m.scene_tag else "",
                        get_scene_label(m.scene_tag, lang="en-US") if m and m.scene_tag else "",
                        m.motion_state if m and m.motion_state else "",
                        " ".join(reasons),
                    ],
                )
            ).lower()
            score = 0.0
            if p.id in fts_ranks:
                score += 2.0 - min(1.0, fts_ranks[p.id] * 0.01)
            if q_lower in haystack:
                score += 1.0
            else:
                tok_hits = sum(1 for tok in q_lower.split() if tok in haystack)
                if tok_hits > 0:
                    score += 0.5 * (tok_hits / max(1, len(q_lower.split())))
            if score <= 0.0:
                continue
            scored_fallback.append(
                (
                    score,
                    self._build_match(
                        p,
                        sess.name,
                        verdict=v,
                        exif=ex,
                        metrics=m,
                        score=score,
                        lang=lang,
                        relation="fts_exif",
                    ),
                )
            )
        scored_fallback.sort(
            key=lambda pair: (-pair[0], pair[1].rel_dir, pair[1].stem, pair[1].photo_id)
        )
        return SearchResult(
            query=q_clean,
            mode="fts_exif",
            semantic_ready=False,
            indexed_count=0,
            total_candidates=len(candidates),
            cameras=sorted_cameras,
            matches=tuple(match for _, match in scored_fallback[:clamped_limit]),
        )

    def find_similar(
        self,
        photo_id: str,
        *,
        limit: int = 12,
        session_id: str | None = None,
        lang: str = "zh-CN",
    ) -> SimilarResult:
        """Find visually similar photos by SigLIP 2 cosine similarity (always excluding anchor itself)."""
        anchor = self.repo.get_photo(photo_id)
        if anchor is None:
            raise ValueError(f"Unknown photo_id: {photo_id}")
        anchor_sess = self.repo.get_session(anchor.session_id)
        anchor_sess_name = anchor_sess.name if anchor_sess is not None else anchor.session_id
        anchor_match = self._build_match(
            anchor,
            anchor_sess_name,
            verdict=self.repo.get_effective_verdict(anchor.id),
            exif=self.repo.get_exif(anchor.id),
            metrics=self.repo.get_metrics(anchor.primary_content_fp, ENGINE_VERSION),
            score=1.0,
            lang=lang,
            relation="anchor",
        )

        clamped_limit = max(1, min(limit, 30))
        if session_id:
            target_sess = self.repo.get_session(session_id)
            if target_sess is None:
                raise ValueError(f"Unknown session_id: {session_id}")
            sessions = [target_sess]
        else:
            sessions = self.repo.list_sessions()

        # Gather all non-quarantined candidate photos strictly excluding anchor.id
        other_candidates: list[
            tuple[PhotoRecord, SessionRecord, Any, ExifRecord | None, MetricsRecord | None]
        ] = []
        for sess in sessions:
            quarantined = self.repo.list_quarantined_photo_ids(sess.id)
            verdicts = self.repo.list_effective_verdicts(sess.id)
            for p in self.repo.list_photos(sess.id):
                if p.id == anchor.id or p.id in quarantined:
                    continue
                other_candidates.append(
                    (
                        p,
                        sess,
                        verdicts.get(p.id),
                        self.repo.get_exif(p.id),
                        self.repo.get_metrics(p.primary_content_fp, ENGINE_VERSION),
                    )
                )

        anchor_vec = self.vector_index.get(anchor.primary_content_fp, self.embedder.model_id)
        indexed_fps = self.vector_index.list_content_fps(self.embedder.model_id)
        other_indexed_fps = {
            p.primary_content_fp
            for p, *_ in other_candidates
            if p.primary_content_fp in indexed_fps
        }

        if anchor_vec is not None and other_indexed_fps:
            vec_matches = self.vector_index.query(
                anchor_vec,
                model_id=self.embedder.model_id,
                top_k=len(other_indexed_fps),
                filter_content_fps=other_indexed_fps,
            )
            score_by_fp = {vm.content_fp: vm.score for vm in vec_matches}
            ranked: list[tuple[float, SearchMatch]] = []
            for p, sess, v, ex, m in other_candidates:
                if p.id == anchor.id:
                    continue
                sim = score_by_fp.get(p.primary_content_fp)
                if sim is None:
                    continue
                ranked.append(
                    (
                        sim,
                        self._build_match(
                            p,
                            sess.name,
                            verdict=v,
                            exif=ex,
                            metrics=m,
                            score=sim,
                            lang=lang,
                            relation="semantic",
                        ),
                    )
                )
            ranked.sort(key=lambda pair: (-pair[0], pair[1].rel_dir, pair[1].stem, pair[1].photo_id))
            return SimilarResult(
                anchor=anchor_match,
                anchor_id=anchor.id,
                session_id=anchor.session_id,
                mode="semantic",
                semantic_ready=True,
                similar=tuple(match for _, match in ranked[:clamped_limit]),
            )

        # Fallback when vectors are not indexed yet: burst siblings + dHash + nearby capture time
        burst_siblings: set[str] = set()
        for burst in self.repo.list_bursts(anchor.session_id):
            member_ids = {m.photo_id for m in self.repo.list_burst_members(burst.id)}
            if anchor.id in member_ids:
                burst_siblings.update(member_ids - {anchor.id})

        def _load_dhash(content_fp: str) -> int | None:
            _g, _l, analysis_p = preview_cache_paths(self.app_paths.cache_dir, content_fp)
            if not analysis_p.is_file():
                return None
            try:
                return compute_dhash(load_analysis_gray(analysis_p))
            except Exception:
                return None

        anchor_dhash = _load_dhash(anchor.primary_content_fp)
        fallback_ranked: list[tuple[float, SearchMatch]] = []
        for p, sess, v, ex, m in other_candidates:
            if p.session_id != anchor.session_id or p.id == anchor.id:
                continue
            is_sibling = p.id in burst_siblings
            dhash_sim = 0.0
            if anchor_dhash is not None:
                other_dhash = _load_dhash(p.primary_content_fp)
                if other_dhash is not None:
                    hamming = (anchor_dhash ^ other_dhash).bit_count()
                    dhash_sim = max(0.0, 1.0 - hamming / 64.0)

            time_sim = 0.0
            if anchor.capture_time is not None and p.capture_time is not None:
                dt = abs(anchor.capture_time - p.capture_time)
                if dt <= 60.0:
                    time_sim = max(0.0, 1.0 - dt / 60.0)

            if is_sibling:
                score = max(0.92, dhash_sim)
                relation = "burst_sibling"
            elif dhash_sim >= 0.72:
                score = dhash_sim
                relation = "visual_dhash"
            elif time_sim >= 0.50:
                score = 0.50 + 0.20 * time_sim
                relation = "nearby_time"
            else:
                continue

            fallback_ranked.append(
                (
                    score,
                    self._build_match(
                        p,
                        sess.name,
                        verdict=v,
                        exif=ex,
                        metrics=m,
                        score=score,
                        lang=lang,
                        relation=relation,
                    ),
                )
            )
        fallback_ranked.sort(
            key=lambda pair: (-pair[0], pair[1].rel_dir, pair[1].stem, pair[1].photo_id)
        )
        return SimilarResult(
            anchor=anchor_match,
            anchor_id=anchor.id,
            session_id=anchor.session_id,
            mode="burst_dhash",
            semantic_ready=False,
            similar=tuple(match for _, match in fallback_ranked[:clamped_limit]),
        )


__all__ = [
    "EmbedSessionResult",
    "SearchMatch",
    "SearchResult",
    "SearchService",
    "SimilarResult",
]
