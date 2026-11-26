"""ReviewService: session summaries, photo queries, human verdicts, overrides, and undo (SPEC §7.3, §9.2)."""

from __future__ import annotations

import io
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw

from lightbox.core import (
    ENGINE_VERSION,
    compute_dhash,
    generate_previews,
    load_analysis_gray,
    preview_cache_paths,
)
from lightbox.i18n import (
    REASON_BURST_NON_PICK,
    REASON_USER_REJECT,
    REASON_USER_RESCUE,
    REASON_USER_RESET,
    REASON_USER_UNSURE,
    is_tech_reject_reason,
)
from lightbox.paths import AppPaths, get_app_paths
from lightbox.store import (
    AssetRecord,
    ExifRecord,
    FaceRecord,
    LightboxRepository,
    MetricsRecord,
    PhotoRecord,
    SessionRecord,
    VerdictRecord,
)

BucketName = Literal["red", "yellow", "normal", "exempt"]
ImageSizeMode = Literal["grid", "loupe", "crop"]
McpImageMode = Literal["thumb", "face_crop", "sharpest_crop"]
CROP_EDGE_PX = 640
MCP_IMAGE_MAX_EDGE_PX = 768
DEFAULT_PREFERENCES_MD = """\
# Lightbox Preferences

No custom aesthetic or culling preference rules have been approved yet.
Approve rules in the Lightbox Insights view or edit this file directly.
"""


@dataclass(frozen=True, slots=True)
class SessionSummary:
    session: SessionRecord
    total_photos: int
    active_total_photos: int
    counts_by_bucket: dict[str, int]
    active_counts_by_bucket: dict[str, int]
    reviewed_count: int
    unreviewed_count: int
    actionable_reject_count: int
    top_reasons: tuple[tuple[str, int], ...]
    camera_distribution: dict[str, int]
    low_res_preview_count: int
    burst_count: int
    unconfirmed_burst_count: int
    quarantined_count: int


@dataclass(frozen=True, slots=True)
class PhotoListItem:
    photo_id: str
    stem: str
    rel_dir: str
    bucket: str
    source: str
    confidence: float
    reason_codes: tuple[str, ...]
    decision_version: int
    is_protected: bool
    is_quarantined: bool
    burst_id: str | None
    preview_w: int | None = None
    preview_h: int | None = None
    face_count: int = 0
    seq_no: int = 1


@dataclass(frozen=True, slots=True)
class PhotoDetail:
    photo: PhotoRecord
    assets: tuple[AssetRecord, ...]
    exif: ExifRecord | None
    metrics: MetricsRecord | None
    faces: tuple[FaceRecord, ...]
    effective_verdict: VerdictRecord | None
    verdict_history: tuple[VerdictRecord, ...]


@dataclass(frozen=True, slots=True)
class AutoScreenResult:
    session_id: str
    total_active: int
    flagged_red: int
    flagged_yellow: int
    unchanged_normal: int


def _is_unconfirmed_auto_red(verdict: VerdictRecord | None) -> bool:
    """Return True if a photo is auto-flagged `red` and not yet manually eliminated by the user."""
    if verdict is None or verdict.bucket != "red":
        return False
    return (
        REASON_USER_REJECT not in verdict.reason_codes
        and REASON_BURST_NON_PICK not in verdict.reason_codes
    )


class ReviewService:
    """Single source of truth for reviewing sessions, inspecting photos, and setting verdicts."""

    def __init__(
        self,
        repo: LightboxRepository,
        *,
        app_paths: AppPaths | None = None,
    ) -> None:
        self.repo = repo
        self.app_paths = app_paths or get_app_paths()

    def list_sessions(self) -> list[SessionRecord]:
        return self.repo.list_sessions()

    def get_photo_image_bytes(
        self,
        photo_id: str,
        *,
        size: ImageSizeMode = "grid",
        region: str | None = None,
    ) -> bytes:
        """Return oriented, sRGB, EXIF-free WebP bytes for `grid`, `loupe`, or 1:1 `crop`."""
        photo = self.repo.get_photo(photo_id)
        if photo is None:
            raise ValueError(f"Unknown photo_id: {photo_id}")

        grid_p, loupe_p, _analysis_p = preview_cache_paths(
            self.app_paths.cache_dir, photo.primary_content_fp
        )
        if not (grid_p.is_file() and loupe_p.is_file()):
            assets = self.repo.list_assets_for_photo(photo_id)
            stills = [a for a in assets if a.kind in {"jpg", "heic", "raw"}]
            if not stills:
                raise FileNotFoundError(f"No still asset found for photo {photo_id}")
            priority = {"jpg": 0, "heic": 1, "raw": 2}
            src = Path(min(stills, key=lambda a: (priority[a.kind], a.rel_path)).abs_path)
            generate_previews(src, photo.primary_content_fp, self.app_paths.cache_dir)

        if size == "grid":
            return grid_p.read_bytes()
        if size == "loupe":
            return loupe_p.read_bytes()

        # size == "crop": 1:1 EXIF-stripped WebP crop around face<N> or sharpest grid cell
        with Image.open(loupe_p) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            cx, cy = w * 0.5, h * 0.5
            crop_size = min(CROP_EDGE_PX, w, h)

            if region and region.startswith("face"):
                idx_str = region[4:]
                face_idx = int(idx_str) if idx_str.isdigit() else 0
                faces = self.repo.list_faces(photo.primary_content_fp)
                match = next((f for f in faces if f.idx == face_idx), faces[0] if faces else None)
                if match is not None:
                    x1, y1, x2, y2 = match.bbox
                    cx = ((x1 + x2) * 0.5) * w
                    cy = ((y1 + y2) * 0.5) * h
                    face_span = max((x2 - x1) * w, (y2 - y1) * h) * 1.8
                    crop_size = int(min(max(face_span, 256.0), float(CROP_EDGE_PX), float(w), float(h)))
            elif region == "sharpest":
                metrics = self.repo.get_metrics(photo.primary_content_fp, ENGINE_VERSION)
                cell = (metrics.exposure or {}).get("sharpest_cell") if metrics else None
                if isinstance(cell, list) and len(cell) == 2:
                    r, c = int(cell[0]), int(cell[1])
                    cx = ((c + 0.5) / 16.0) * w
                    cy = ((r + 0.5) / 16.0) * h

            half = crop_size * 0.5
            left = int(max(0, min(w - crop_size, round(cx - half))))
            top = int(max(0, min(h - crop_size, round(cy - half))))
            cropped = rgb.crop((left, top, left + crop_size, top + crop_size))
            buf = io.BytesIO()
            cropped.save(buf, format="WEBP", quality=86)
            return buf.getvalue()

    def get_photo_mcp_image_bytes(
        self,
        photo_id: str,
        *,
        mode: McpImageMode = "thumb",
    ) -> bytes:
        """Return EXIF-stripped WebP bytes bounded to <= 768px for MCP host vision models (SPEC §8.2)."""
        if mode == "face_crop":
            return self.get_photo_image_bytes(photo_id, size="crop", region="face0")
        if mode == "sharpest_crop":
            return self.get_photo_image_bytes(photo_id, size="crop", region="sharpest")

        loupe_bytes = self.get_photo_image_bytes(photo_id, size="loupe")
        with Image.open(io.BytesIO(loupe_bytes)) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            longest = max(w, h)
            if longest > MCP_IMAGE_MAX_EDGE_PX:
                scale = MCP_IMAGE_MAX_EDGE_PX / float(longest)
                rgb = rgb.resize(
                    (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                    Image.Resampling.LANCZOS,
                )
            buf = io.BytesIO()
            rgb.save(buf, format="WEBP", quality=84)
            return buf.getvalue()

    def build_burst_compare_images(
        self,
        photo_ids: Sequence[str],
        *,
        max_frames: int = 6,
    ) -> tuple[bytes, bytes | None]:
        """Build EXIF-stripped numbered contact sheet and optional 1:1 horizontal face crop strip."""
        selected = list(photo_ids[:max_frames])
        if not selected:
            raise ValueError("Cannot build burst comparison image for empty photo list.")

        tile_w, tile_h = 240, 180
        cols = min(3, len(selected))
        rows = (len(selected) + cols - 1) // cols
        pad = 8
        sheet_w = cols * tile_w + (cols + 1) * pad
        sheet_h = rows * tile_h + (rows + 1) * pad
        sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 27))
        draw = ImageDraw.Draw(sheet)

        face_tiles: list[tuple[int, Image.Image]] = []
        for idx, pid in enumerate(selected, start=1):
            thumb_bytes = self.get_photo_mcp_image_bytes(pid, mode="thumb")
            with Image.open(io.BytesIO(thumb_bytes)) as raw_im:
                im = raw_im.convert("RGB")
                im.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
                r = (idx - 1) // cols
                c = (idx - 1) % cols
                ox = pad + c * (tile_w + pad) + (tile_w - im.width) // 2
                oy = pad + r * (tile_h + pad) + (tile_h - im.height) // 2
                sheet.paste(im, (ox, oy))
                bx = pad + c * (tile_w + pad) + 4
                by = pad + r * (tile_h + pad) + 4
                draw.rectangle((bx, by, bx + 26, by + 20), fill=(0, 0, 0))
                draw.text((bx + 6, by + 4), f"#{idx}", fill=(255, 255, 255))

            photo = self.repo.get_photo(pid)
            if photo is not None and self.repo.list_faces(photo.primary_content_fp):
                crop_bytes = self.get_photo_image_bytes(pid, size="crop", region="face0")
                with Image.open(io.BytesIO(crop_bytes)) as face_im:
                    tile = face_im.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
                    face_tiles.append((idx, tile))

        sheet_buf = io.BytesIO()
        sheet.save(sheet_buf, format="WEBP", quality=82)
        sheet_bytes = sheet_buf.getvalue()

        if not face_tiles:
            return sheet_bytes, None

        f_pad = 6
        f_edge = 224
        strip_w = len(face_tiles) * f_edge + (len(face_tiles) + 1) * f_pad
        strip_h = f_edge + 2 * f_pad
        strip = Image.new("RGB", (strip_w, strip_h), (24, 24, 27))
        f_draw = ImageDraw.Draw(strip)
        for pos, (frame_idx, tile) in enumerate(face_tiles):
            fx = f_pad + pos * (f_edge + f_pad)
            strip.paste(tile, (fx, f_pad))
            f_draw.rectangle((fx + 4, f_pad + 4, fx + 34, f_pad + 24), fill=(0, 0, 0))
            f_draw.text((fx + 9, f_pad + 8), f"#{frame_idx}", fill=(255, 255, 255))

        strip_buf = io.BytesIO()
        strip.save(strip_buf, format="WEBP", quality=84)
        return sheet_bytes, strip_buf.getvalue()

    def format_exif_brief(self, photo_id: str) -> str:
        """Return a compact single-line EXIF brief suitable for handle-based listings."""
        exif = self.repo.get_exif(photo_id)
        if exif is None:
            return ""
        parts: list[str] = []
        if exif.focal_eq is not None:
            parts.append(f"{exif.focal_eq:.0f}mm")
        if exif.aperture is not None:
            parts.append(f"f/{exif.aperture:g}")
        if exif.shutter_s is not None and exif.shutter_s > 0:
            if exif.shutter_s < 1.0:
                denom = max(1, int(round(1.0 / exif.shutter_s)))
                parts.append(f"1/{denom}s")
            else:
                parts.append(f"{exif.shutter_s:g}s")
        if exif.iso is not None:
            parts.append(f"ISO {exif.iso}")
        if exif.ev_comp is not None and abs(exif.ev_comp) >= 0.05:
            parts.append(f"EV {exif.ev_comp:+.1f}")
        if exif.light_bucket:
            parts.append(exif.light_bucket)
        return " ".join(parts)

    def read_preferences_markdown(self) -> str:
        """Read `<APP_DATA_DIR>/preferences.md` or return default template if absent."""
        pref = self.app_paths.preferences_file
        if pref.is_file():
            return pref.read_text(encoding="utf-8")
        return DEFAULT_PREFERENCES_MD

    def list_photos_page(
        self,
        session_id: str,
        *,
        bucket: str | None = None,
        reason: str | None = None,
        sort: str = "uncertainty",
        limit: int = 30,
        cursor: str | None = None,
        include_quarantined: bool = False,
    ) -> tuple[list[PhotoListItem], str | None]:
        """Return a bounded page of photo handles plus opaque `next_cursor`."""
        offset = 0
        if cursor is not None and cursor.strip():
            try:
                offset = max(0, int(cursor.strip()))
            except ValueError as exc:
                raise ValueError(f"Invalid pagination cursor: {cursor!r}") from exc
        clamped_limit = max(1, min(limit, 2000))
        batch = self.list_photos(
            session_id,
            bucket=bucket,
            reason=reason,
            sort=sort,
            limit=clamped_limit + 1,
            offset=offset,
            include_quarantined=include_quarantined,
        )
        has_more = len(batch) > clamped_limit
        page = batch[:clamped_limit]
        next_cursor = str(offset + clamped_limit) if has_more else None
        return page, next_cursor

    def search_photos(
        self,
        query: str,
        *,
        session_id: str | None = None,
        date_from: str | None = None,
        camera: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search photos using FTS5 + EXIF/metadata filters (SigLIP 2 semantic layer activates in T11)."""
        clamped_limit = max(1, min(limit, 50))
        min_ts: float | None = None
        if date_from:
            try:
                dt = datetime.fromisoformat(date_from)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                min_ts = dt.timestamp()
            except ValueError as exc:
                raise ValueError(f"Invalid ISO date_from: {date_from!r}") from exc

        cam_needle = camera.strip().lower() if camera else None
        q_clean = query.strip()

        fts_ranks: dict[str, int] = {}
        if q_clean:
            tokens = [tok.strip('"') for tok in q_clean.split() if tok.strip('"')]
            if tokens:
                fts_expr = " OR ".join(f'"{tok}"*' for tok in tokens)
                try:
                    hits = self.repo.search_session_fts(fts_expr, limit=200)
                    fts_ranks = {pid: idx for idx, pid in enumerate(hits)}
                except Exception:
                    fts_ranks = {}

        sessions = (
            [s for s in [self.repo.get_session(session_id)] if s is not None]
            if session_id
            else self.repo.list_sessions()
        )
        if session_id and not sessions:
            raise ValueError(f"Unknown session_id: {session_id}")

        q_lower = q_clean.lower()
        scored: list[tuple[float, dict[str, Any]]] = []
        for sess in sessions:
            quarantined = self.repo.list_quarantined_photo_ids(sess.id)
            verdicts = self.repo.list_effective_verdicts(sess.id)
            for p in self.repo.list_photos(sess.id):
                if p.id in quarantined:
                    continue
                if min_ts is not None and (p.capture_time is None or p.capture_time < min_ts):
                    continue
                if cam_needle and cam_needle not in (p.camera_model or "").lower():
                    continue
                v = verdicts.get(p.id)
                reasons = v.reason_codes if v is not None else ()
                m = self.repo.get_metrics(p.primary_content_fp, ENGINE_VERSION)
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
                            m.motion_state if m and m.motion_state else "",
                            " ".join(reasons),
                        ],
                    )
                ).lower()

                score = 0.0
                if not q_lower:
                    score = 1.0
                else:
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
                scored.append(
                    (
                        score,
                        {
                            "photo_id": p.id,
                            "session_id": p.session_id,
                            "stem": p.stem,
                            "rel_dir": p.rel_dir,
                            "bucket": v.bucket if v is not None else "normal",
                            "confidence": v.confidence if v is not None else 1.0,
                            "reason_codes": list(reasons),
                            "camera_model": p.camera_model,
                            "exif_brief": self.format_exif_brief(p.id),
                            "score": round(score, 4),
                        },
                    )
                )

        scored.sort(key=lambda pair: (-pair[0], pair[1]["rel_dir"], pair[1]["stem"]))
        return [row for _, row in scored[:clamped_limit]]

    def find_similar_photos(
        self,
        photo_id: str,
        *,
        limit: int = 12,
    ) -> tuple[PhotoRecord, list[dict[str, Any]]]:
        """Find visually/temporally similar photos in the same session (burst + dHash fallback before T11)."""
        anchor = self.repo.get_photo(photo_id)
        if anchor is None:
            raise ValueError(f"Unknown photo_id: {photo_id}")
        clamped_limit = max(1, min(limit, 30))

        # Identify burst siblings
        burst_siblings: set[str] = set()
        for burst in self.repo.list_bursts(anchor.session_id):
            members = self.repo.list_burst_members(burst.id)
            member_ids = {m.photo_id for m in members}
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
        quarantined = self.repo.list_quarantined_photo_ids(anchor.session_id)
        verdicts = self.repo.list_effective_verdicts(anchor.session_id)

        candidates: list[tuple[float, dict[str, Any]]] = []
        for other in self.repo.list_photos(anchor.session_id):
            if other.id == anchor.id or other.id in quarantined:
                continue
            is_sibling = other.id in burst_siblings
            dhash_sim = 0.0
            if anchor_dhash is not None:
                other_dhash = _load_dhash(other.primary_content_fp)
                if other_dhash is not None:
                    hamming = (anchor_dhash ^ other_dhash).bit_count()
                    dhash_sim = max(0.0, 1.0 - hamming / 64.0)

            time_sim = 0.0
            if anchor.capture_time is not None and other.capture_time is not None:
                dt = abs(anchor.capture_time - other.capture_time)
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

            v = verdicts.get(other.id)
            candidates.append(
                (
                    score,
                    {
                        "photo_id": other.id,
                        "stem": other.stem,
                        "rel_dir": other.rel_dir,
                        "bucket": v.bucket if v is not None else "normal",
                        "similarity": round(score, 4),
                        "relation": relation,
                        "exif_brief": self.format_exif_brief(other.id),
                    },
                )
            )

        candidates.sort(key=lambda pair: (-pair[0], pair[1]["rel_dir"], pair[1]["stem"]))
        return anchor, [row for _, row in candidates[:clamped_limit]]

    def get_session_summary(self, session_id: str) -> SessionSummary:
        session = self.repo.get_session(session_id)
        if session is None:
            raise ValueError(f"Unknown session_id: {session_id}")

        photos = self.repo.list_photos(session_id)
        verdicts = self.repo.list_effective_verdicts(session_id)
        bursts = self.repo.list_bursts(session_id)
        quarantined = self.repo.list_quarantined_photo_ids(session_id)
        applying = self.repo.list_applying_plan_photo_ids(session_id)
        unconfirmed_non_picks = self.repo.list_unconfirmed_burst_non_pick_photo_ids(session_id)

        counts: dict[str, int] = {"red": 0, "yellow": 0, "normal": 0, "exempt": 0}
        active_counts: dict[str, int] = {"red": 0, "yellow": 0, "normal": 0, "exempt": 0}
        reason_counter: Counter[str] = Counter()
        cam_counter: Counter[str] = Counter()
        low_res_count = 0
        active_total = 0
        kept_count = 0
        unreviewed_count = 0
        actionable_rejects = 0

        for p in photos:
            v = verdicts.get(p.id)
            bucket = v.bucket if v is not None else "normal"
            source = v.source if v is not None else "engine"
            reasons = v.reason_codes if v is not None else ()
            counts[bucket] = counts.get(bucket, 0) + 1
            is_q = p.id in quarantined
            if not is_q:
                active_total += 1
                active_counts[bucket] = active_counts.get(bucket, 0) + 1
                if bucket == "red":
                    if (
                        not p.is_protected
                        and p.id not in applying
                        and bool(self.repo.list_assets_for_photo(p.id))
                        and (
                            p.id not in unconfirmed_non_picks
                            or source == "user"
                            or is_tech_reject_reason(reasons)
                        )
                    ):
                        actionable_rejects += 1
                    # Auto-flagged rejects not yet manually confirmed by user also appear in `待选`
                    if not p.is_protected and _is_unconfirmed_auto_red(v):
                        unreviewed_count += 1
                elif bucket == "yellow":
                    # Yellow photos live in the `存疑` tab (disjoint from `待选` and `已保留`).
                    pass
                elif p.is_protected or bucket == "exempt":
                    # In-camera protected photos live in `已保护` (not counted as user-reviewed progress).
                    pass
                elif source == "user" and REASON_USER_RESET not in reasons:
                    kept_count += 1
                else:
                    unreviewed_count += 1
            if v is not None:
                for code in v.reason_codes:
                    reason_counter[code] += 1
            cam_label = p.camera_model or "Unknown"
            cam_counter[cam_label] += 1
            if p.preview_w is not None and p.preview_h is not None:
                if max(p.preview_w, p.preview_h) < 2000:
                    low_res_count += 1

        active_bursts = [
            b
            for b in bursts
            if sum(1 for m in self.repo.list_burst_members(b.id) if m.photo_id not in quarantined)
            >= 2
        ]

        return SessionSummary(
            session=session,
            total_photos=len(photos),
            active_total_photos=active_total,
            counts_by_bucket=counts,
            active_counts_by_bucket=active_counts,
            reviewed_count=kept_count,
            unreviewed_count=unreviewed_count,
            actionable_reject_count=actionable_rejects,
            top_reasons=tuple(reason_counter.most_common(10)),
            camera_distribution=dict(cam_counter),
            low_res_preview_count=low_res_count,
            burst_count=len(active_bursts),
            unconfirmed_burst_count=sum(1 for b in active_bursts if not b.confirmed),
            quarantined_count=len(quarantined),
        )

    def list_photos(
        self,
        session_id: str,
        *,
        bucket: str | None = None,
        reason: str | None = None,
        sort: str = "uncertainty",
        limit: int = 30,
        offset: int = 0,
        include_quarantined: bool = False,
    ) -> list[PhotoListItem]:
        photos = self.repo.list_photos(session_id)
        verdicts = self.repo.list_effective_verdicts(session_id)
        quarantined = self.repo.list_quarantined_photo_ids(session_id)

        burst_by_photo: dict[str, str] = {}
        for burst in self.repo.list_bursts(session_id):
            for member in self.repo.list_burst_members(burst.id):
                burst_by_photo[member.photo_id] = burst.id

        # Stable permanent 1-based sequence number across the full session (never shifts on reject/quarantine)
        ordered_all = sorted(photos, key=lambda p: (p.rel_dir, p.stem, p.id))
        seq_by_photo: dict[str, int] = {p.id: idx + 1 for idx, p in enumerate(ordered_all)}

        items: list[PhotoListItem] = []
        for p in photos:
            is_q = p.id in quarantined
            if is_q and not include_quarantined:
                continue
            v = verdicts.get(p.id)
            p_bucket = v.bucket if v is not None else "normal"
            p_source = v.source if v is not None else "engine"
            p_conf = v.confidence if v is not None else 1.0
            p_reasons = v.reason_codes if v is not None else ()
            is_user_reviewed = p_source == "user" and REASON_USER_RESET not in p_reasons
            if bucket == "unreviewed":
                if p.is_protected or p_bucket in {"yellow", "exempt"}:
                    continue
                if p_bucket == "red":
                    if not _is_unconfirmed_auto_red(v):
                        continue
                elif is_user_reviewed:
                    continue
            elif bucket == "kept":
                if p_bucket != "normal" or p.is_protected or not is_user_reviewed:
                    continue
            elif bucket == "exempt":
                if not p.is_protected and p_bucket != "exempt":
                    continue
            elif bucket is not None and p_bucket != bucket:
                continue
            if reason is not None and reason not in p_reasons:
                continue
            m = self.repo.get_metrics(p.primary_content_fp, ENGINE_VERSION)
            items.append(
                PhotoListItem(
                    photo_id=p.id,
                    stem=p.stem,
                    rel_dir=p.rel_dir,
                    bucket=p_bucket,
                    source=p_source,
                    confidence=p_conf,
                    reason_codes=p_reasons,
                    decision_version=p.decision_version,
                    is_protected=p.is_protected,
                    is_quarantined=is_q,
                    burst_id=burst_by_photo.get(p.id),
                    preview_w=p.preview_w,
                    preview_h=p.preview_h,
                    face_count=int(m.face_count or 0) if m is not None else 0,
                    seq_no=seq_by_photo.get(p.id, 1),
                )
            )

        if sort == "uncertainty":
            items.sort(key=lambda it: (it.confidence, it.rel_dir, it.stem))
        elif sort == "name":
            items.sort(key=lambda it: (it.rel_dir, it.stem, it.photo_id))

        return items[offset : offset + max(1, limit)]

    def get_photo_reasons_map(self, session_id: str) -> dict[str, tuple[str, tuple[str, ...]]]:
        """Return `{photo_id: (stem, reason_codes)}` for enriching plan preview modals."""
        photos = self.repo.list_photos(session_id)
        verdicts = self.repo.list_effective_verdicts(session_id)
        out: dict[str, tuple[str, tuple[str, ...]]] = {}
        for p in photos:
            v = verdicts.get(p.id)
            out[p.id] = (p.stem, v.reason_codes if v is not None else ())
        return out

    def get_photo_detail(self, photo_id: str) -> PhotoDetail:
        photo = self.repo.get_photo(photo_id)
        if photo is None:
            raise ValueError(f"Unknown photo_id: {photo_id}")
        assets = tuple(self.repo.list_assets_for_photo(photo_id))
        exif = self.repo.get_exif(photo_id)
        metrics = self.repo.get_metrics(photo.primary_content_fp, ENGINE_VERSION)
        faces = tuple(self.repo.list_faces(photo.primary_content_fp))
        eff = self.repo.get_effective_verdict(photo_id)
        hist = tuple(self.repo.list_verdict_history(photo_id))
        return PhotoDetail(
            photo=photo,
            assets=assets,
            exif=exif,
            metrics=metrics,
            faces=faces,
            effective_verdict=eff,
            verdict_history=hist,
        )

    def _build_feature_snapshot(self, photo: PhotoRecord) -> dict[str, Any]:
        exif = self.repo.get_exif(photo.id)
        metrics = self.repo.get_metrics(photo.primary_content_fp, ENGINE_VERSION)
        return {
            "cohort": photo.cohort,
            "camera_model": photo.camera_model,
            "lens_model": photo.lens_model,
            "shutter_s": exif.shutter_s if exif else None,
            "aperture": exif.aperture if exif else None,
            "iso": exif.iso if exif else None,
            "focal_eq": exif.focal_eq if exif else None,
            "ev_comp": exif.ev_comp if exif else None,
            "light_bucket": exif.light_bucket if exif else None,
            "sharp_p95": metrics.sharp_p95 if metrics else None,
            "sharp_z": metrics.sharp_z if metrics else None,
            "noise_sigma": metrics.noise_sigma if metrics else None,
            "motion_aniso": metrics.motion_aniso if metrics else None,
            "motion_state": metrics.motion_state if metrics else None,
        }

    def set_verdict(
        self,
        photo_id: str,
        *,
        bucket: BucketName,
        source: str = "user",
        confidence: float = 1.0,
        reason_codes: Sequence[str] | None = None,
        reason_tag: str | None = None,
        actor: str = "user",
        detail: dict[str, Any] | None = None,
    ) -> VerdictRecord:
        """Append a verdict, bump `decision_version`, and record `overrides` on user changes."""
        photo = self.repo.get_photo(photo_id)
        if photo is None:
            raise ValueError(f"Unknown photo_id: {photo_id}")
        if photo.is_protected and bucket == "red":
            raise ValueError(f"Cannot mark exempt/protected photo {photo_id} as 'red'.")

        prev = self.repo.get_effective_verdict(photo_id)
        from_bucket = prev.bucket if prev is not None else "normal"

        if reason_codes is None:
            default_code = {
                "normal": REASON_USER_RESCUE,
                "red": REASON_USER_REJECT,
                "yellow": REASON_USER_UNSURE,
                "exempt": REASON_USER_RESCUE,
            }[bucket]
            if bucket == "red" and prev is not None and is_tech_reject_reason(prev.reason_codes):
                # Preserve technical reason codes alongside user confirmation so modal labels stay informative
                preserved = tuple(c for c in prev.reason_codes if c != REASON_USER_REJECT)
                final_reasons: tuple[str, ...] = (*preserved, REASON_USER_REJECT)
            else:
                final_reasons = (default_code,)
        else:
            final_reasons = tuple(reason_codes)

        verdict = self.repo.add_verdict(
            photo_id=photo_id,
            source=source,
            bucket=bucket,
            confidence=confidence,
            reason_codes=final_reasons,
            detail=detail,
            increment_decision_version=True,
        )

        if source == "user" and from_bucket != bucket:
            self.repo.add_override(
                photo_id=photo_id,
                session_id=photo.session_id,
                from_bucket=from_bucket,
                to_bucket=bucket,
                reason_tag=reason_tag,
                feature_snapshot=self._build_feature_snapshot(photo),
                actor=actor,
            )

        return verdict

    def undo_verdict(self, photo_id: str, *, actor: str = "user") -> VerdictRecord | None:
        """Append a compensating `user` verdict restoring the prior verdict state.

        When undoing back to an engine-only initial verdict, tags with `REASON_USER_RESET`
        so the photo returns to the `unreviewed` review queue while keeping `user` priority.
        """
        history = self.repo.list_verdict_history(photo_id)
        if len(history) < 2:
            return None
        previous = history[-2]
        restored_reasons: Sequence[str] = (
            (REASON_USER_RESET,)
            if previous.source == "engine"
            else (previous.reason_codes or (REASON_USER_RESET,))
        )
        return self.set_verdict(
            photo_id,
            bucket=previous.bucket,  # type: ignore[arg-type]
            source="user",
            confidence=previous.confidence,
            reason_codes=restored_reasons,
            reason_tag="undo",
            actor=actor,
        )

    def reset_session_review(self, session_id: str, *, actor: str = "user") -> int:
        """Reset all active (non-quarantined, non-protected) photos back to `unreviewed`."""
        session = self.repo.get_session(session_id)
        if session is None:
            raise ValueError(f"Unknown session_id: {session_id}")
        photos = self.repo.list_photos(session_id)
        quarantined = self.repo.list_quarantined_photo_ids(session_id)
        verdicts = self.repo.list_effective_verdicts(session_id)

        reset_count = 0
        for p in photos:
            if p.id in quarantined or p.is_protected:
                continue
            v = verdicts.get(p.id)
            already_unreviewed = (
                v is None
                or (
                    v.bucket == "normal"
                    and (v.source != "user" or REASON_USER_RESET in v.reason_codes)
                )
            )
            if already_unreviewed:
                continue
            self.repo.add_verdict(
                photo_id=p.id,
                source="user",
                bucket="normal",
                confidence=1.0,
                reason_codes=(REASON_USER_RESET,),
                detail={"actor": actor, "action": "reset_session_review"},
                increment_decision_version=True,
            )
            reset_count += 1
        return reset_count

    def apply_engine_screen(self, session_id: str, *, actor: str = "user") -> AutoScreenResult:
        """Surface all engine-detected obvious rejects (`red`) and borderline shots (`yellow`).

        Useful after `reset_session_review` or restoring photos from the trash folder so the user
        can re-run automatic reject screening in one click without losing already-kept sharp photos.
        """
        session = self.repo.get_session(session_id)
        if session is None:
            raise ValueError(f"Unknown session_id: {session_id}")
        photos = self.repo.list_photos(session_id)
        quarantined = self.repo.list_quarantined_photo_ids(session_id)

        total_active = 0
        flagged_red = 0
        flagged_yellow = 0
        unchanged_normal = 0

        for p in photos:
            if p.id in quarantined:
                continue
            if not self.repo.list_assets_for_photo(p.id):
                continue
            total_active += 1
            if p.is_protected:
                unchanged_normal += 1
                continue

            history = self.repo.list_verdict_history(p.id)
            engine_verdicts = [v for v in history if v.source == "engine"]
            if not engine_verdicts:
                unchanged_normal += 1
                continue
            latest_engine = engine_verdicts[-1]
            if latest_engine.bucket not in {"red", "yellow"}:
                unchanged_normal += 1
                continue

            current = self.repo.get_effective_verdict(p.id)
            if current is None or current.bucket != latest_engine.bucket:
                # Append with `source="user"` so it cleanly supersedes any earlier `user_reset` /
                # `user_restored` row while preserving exact engine technical reason codes.
                self.repo.add_verdict(
                    photo_id=p.id,
                    source="user",
                    bucket=latest_engine.bucket,
                    confidence=latest_engine.confidence,
                    reason_codes=latest_engine.reason_codes,
                    detail={"actor": actor, "action": "auto_screen", "engine_bucket": latest_engine.bucket},
                    increment_decision_version=True,
                )

            if latest_engine.bucket == "red":
                flagged_red += 1
            else:
                flagged_yellow += 1

        return AutoScreenResult(
            session_id=session_id,
            total_active=total_active,
            flagged_red=flagged_red,
            flagged_yellow=flagged_yellow,
            unchanged_normal=unchanged_normal,
        )


__all__ = [
    "AutoScreenResult",
    "BucketName",
    "CROP_EDGE_PX",
    "DEFAULT_PREFERENCES_MD",
    "ImageSizeMode",
    "MCP_IMAGE_MAX_EDGE_PX",
    "McpImageMode",
    "PhotoDetail",
    "PhotoListItem",
    "ReviewService",
    "SessionSummary",
]
