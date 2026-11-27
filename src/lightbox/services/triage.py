"""L3 concurrent VLM batch triage for uncertain (`yellow`) photos and unconfirmed bursts (SPEC §6.3, PLAN T10)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from lightbox.config import AppConfig, load_config
from lightbox.core import ENGINE_VERSION
from lightbox.i18n import (
    REASON_BORDERLINE_BLUR,
    REASON_EXTREME_EXPOSURE,
    REASON_FACE_OUT_OF_FOCUS,
    REASON_LOW_RELATIVE_SHARPNESS,
    REASON_MOTION_BLUR,
    REASON_OUT_OF_FOCUS,
    REASON_SEVERE_BLUR,
    REASON_VLM_KEEP,
    REASON_VLM_REJECT,
    REASON_VLM_UNSURE,
)
from lightbox.models import (
    BudgetExceededError,
    ImageInput,
    Message,
    ModelRouter,
)
from lightbox.models.secrets import has_api_key
from lightbox.paths import AppPaths, get_app_paths
from lightbox.services.burst import BurstDetail, BurstService
from lightbox.services.review import (
    BucketName,
    ReviewService,
    extract_effective_preferences_for_prompt,
)
from lightbox.store import (
    JobItemRecord,
    JobRecord,
    LightboxRepository,
    PhotoRecord,
)

YELLOW_CROP_MIN_PX = 512
YELLOW_CROP_TARGET_PX = 640
YELLOW_CROP_MAX_PX = 768
CONTEXT_MAX_EDGE_PX = 512
BURST_FACE_TILE_PX = 224

TriageStatus = Literal["completed", "completed_with_errors", "paused", "failed"]


class YellowTriageOutput(BaseModel):
    """Structured VLM judgment for a single uncertain (`yellow`) photo."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["keep", "reject", "unsure"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason_code: Literal[
        "motion_blur",
        "out_of_focus",
        "severe_blur",
        "extreme_exposure",
        "face_out_of_focus",
        "vlm_keep",
        "vlm_reject",
        "vlm_unsure",
    ] | None = None
    rationale: str = Field(min_length=1, max_length=300)


class BurstTriageOutput(BaseModel):
    """Structured VLM pick recommendation for an unconfirmed burst group."""

    model_config = ConfigDict(extra="forbid")

    pick_frame_no: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    eyes_open: bool | None = None
    rationale: str = Field(min_length=1, max_length=300)


@dataclass(frozen=True, slots=True)
class TriageEstimate:
    """Dry-run scope and cost estimate for L3 VLM batch triage."""

    session_id: str
    dry_run: bool
    yellow_count: int
    burst_count: int
    total_units: int
    est_usd: float
    profile: str
    model_alias: str
    model_id: str
    provider: str
    vlm_backend_ready: bool
    session_spent_usd: float
    session_budget_usd: float
    monthly_spent_usd: float
    monthly_budget_usd: float
    total_yellow_available: int = 0
    total_bursts_available: int = 0
    skipped_bursts_due_to_limit: int = 0
    note: str | None = None


@dataclass(frozen=True, slots=True)
class TriageResult:
    """Execution summary of an L3 VLM batch triage run."""

    job_id: str
    session_id: str
    status: TriageStatus
    paused_reason: str | None
    yellow_total: int
    burst_total: int
    reviewed_yellow: int
    reviewed_bursts: int
    cache_hits: int
    paid_calls: int
    fresh_calls: int
    spent_usd: float
    model_id: str
    provider: str
    failed_yellow: int = 0
    failed_bursts: int = 0
    failed_count: int = 0


def _month_start_ts() -> float:
    now = datetime.now(tz=UTC)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()


def _encode_lossless_png(im: Image.Image) -> bytes:
    """Encode an in-memory RGB PIL Image as lossless PNG (single JPEG pass happens in ModelRouter)."""
    rgb = im.convert("RGB") if im.mode != "RGB" else im
    clean = Image.new("RGB", rgb.size)
    clean.paste(rgb)
    buf = io.BytesIO()
    clean.save(buf, format="PNG")
    return buf.getvalue()


class TriageService:
    """Concurrent stateless L3 VLM batch triage service for `yellow` photos and `bursts`."""

    def __init__(
        self,
        repo: LightboxRepository,
        review: ReviewService,
        burst: BurstService,
        model_router: ModelRouter,
        *,
        app_paths: AppPaths | None = None,
        config: AppConfig | None = None,
    ) -> None:
        self.repo = repo
        self.review = review
        self.burst = burst
        self.model_router = model_router
        self.app_paths = app_paths or get_app_paths()
        self.config = config or load_config(self.app_paths)

    def _select_targets(
        self,
        session_id: str,
        *,
        max_photos: int,
    ) -> tuple[list[PhotoRecord], list[BurstDetail], int, int]:
        session = self.repo.get_session(session_id)
        if session is None:
            raise ValueError(f"Unknown session_id: {session_id}")

        quarantined = self.repo.list_quarantined_photo_ids(session_id)
        verdicts = self.repo.list_effective_verdicts(session_id)
        all_photos = {p.id: p for p in self.repo.list_photos(session_id)}

        yellow_candidates: list[tuple[float, str, PhotoRecord]] = []
        for pid, photo in all_photos.items():
            if pid in quarantined or photo.is_protected:
                continue
            v = verdicts.get(pid)
            if v is not None and v.bucket == "yellow":
                yellow_candidates.append((v.confidence, photo.stem, photo))
        yellow_candidates.sort(key=lambda item: (item[0], item[1], item[2].id))

        limit = max(1, max_photos)
        total_yellow_available = len(yellow_candidates)
        selected_yellow = [item[2] for item in yellow_candidates[:limit]]
        remaining_slots = max(0, limit - len(selected_yellow))

        unconfirmed_bursts = [
            b for b in self.burst.list_bursts(session_id) if not b.burst.confirmed
        ]
        total_bursts_available = len(unconfirmed_bursts)
        selected_bursts = unconfirmed_bursts[:remaining_slots] if remaining_slots > 0 else []
        return selected_yellow, selected_bursts, total_yellow_available, total_bursts_available

    def estimate_triage(
        self,
        session_id: str,
        *,
        max_photos: int = 50,
        profile: str | None = None,
        session_budget_usd: float | None = None,
    ) -> TriageEstimate:
        """Estimate scope and USD cost for L3 VLM triage without invoking any model."""
        (
            yellow_photos,
            bursts,
            total_yellow_available,
            total_bursts_available,
        ) = self._select_targets(session_id, max_photos=max_photos)
        prof_name = profile or self.config.model_profile
        alias, spec = self.model_router.resolve_spec("triage_vlm", profile=prof_name)

        # Conservative per-unit estimate: ~1,600 input tokens (2 stripped images + prompt) + ~120 output tokens
        per_unit_usd = (
            (1600 / 1_000_000.0) * spec.input_cost_per_mtok
            + (120 / 1_000_000.0) * spec.output_cost_per_mtok
        )
        total_units = len(yellow_photos) + len(bursts)
        est_usd = round(total_units * per_unit_usd, 6)

        skipped_bursts = max(0, total_bursts_available - len(bursts))
        note: str | None = None
        if skipped_bursts > 0:
            note = (
                f"max_photos={max_photos} filled by {len(yellow_photos)} yellow photo(s); "
                f"{skipped_bursts} unconfirmed burst group(s) deferred to a later run or higher max_photos."
            )

        session_spent = self.repo.sum_llm_cost(session_id=session_id)
        monthly_spent = self.repo.sum_llm_cost(since_ts=_month_start_ts())
        sess_limit = (
            session_budget_usd
            if session_budget_usd is not None
            else self.config.budget.session_budget_usd
        )
        return TriageEstimate(
            session_id=session_id,
            dry_run=True,
            yellow_count=len(yellow_photos),
            burst_count=len(bursts),
            total_units=total_units,
            est_usd=est_usd,
            profile=prof_name,
            model_alias=alias,
            model_id=spec.model_id,
            provider=spec.provider,
            vlm_backend_ready=has_api_key(spec.provider, app_paths=self.app_paths),
            session_spent_usd=round(session_spent, 6),
            session_budget_usd=sess_limit,
            monthly_spent_usd=round(monthly_spent, 6),
            monthly_budget_usd=self.config.budget.monthly_budget_usd,
            total_yellow_available=total_yellow_available,
            total_bursts_available=total_bursts_available,
            skipped_bursts_due_to_limit=skipped_bursts,
            note=note,
        )

    def create_queued_job(
        self,
        session_id: str,
        *,
        profile: str | None = None,
        session_budget_usd: float | None = None,
    ) -> str:
        """Validate session + model credentials and persist a `queued` job row for async execution."""
        if self.repo.get_session(session_id) is None:
            raise ValueError(f"Unknown session_id: {session_id}")
        prof_name = profile or self.config.model_profile
        self.model_router.resolve_spec("triage_vlm", profile=prof_name)
        self.model_router.get_model("triage_vlm", profile=prof_name)

        eff_budget = (
            session_budget_usd
            if session_budget_usd is not None
            else self.config.budget.session_budget_usd
        )
        job_id = f"triage_{uuid.uuid4().hex[:12]}"
        now = time.time()
        self.repo.create_job(
            JobRecord(
                id=job_id,
                kind="triage_vlm",
                session_id=session_id,
                status="queued",
                created_at=now,
                budget_usd=eff_budget,
                spent_usd=0.0,
                heartbeat_at=now,
            )
        )
        return job_id

    def _build_yellow_images(
        self,
        photo_id: str,
    ) -> tuple[ImageInput, ImageInput, str]:
        """Build lossless PNG 1:1 detail crop (512–768px) + 512px context image.

        Single lossy JPEG encoding and EXIF/max_pixels sanitization occur once inside ModelRouter.
        """
        loupe_bytes = self.review.get_photo_image_bytes(photo_id, size="loupe")
        with Image.open(io.BytesIO(loupe_bytes)) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            cx, cy = w * 0.5, h * 0.5
            crop_size = (
                min(YELLOW_CROP_TARGET_PX, w, h)
                if min(w, h) >= YELLOW_CROP_MIN_PX
                else min(w, h)
            )
            crop_region = "center"

            photo = self.repo.get_photo(photo_id)
            if photo is not None:
                faces = self.repo.list_faces(photo.primary_content_fp)
                if faces:
                    primary = max(
                        faces,
                        key=lambda f: (float(f.area_ratio or 0.0), float(f.det_score or 0.0)),
                    )
                    x1, y1, x2, y2 = primary.bbox
                    if primary.landmarks:
                        xs = [pt[0] for pt in primary.landmarks] + [x1, x2]
                        ys = [pt[1] for pt in primary.landmarks] + [y1, y2]
                        cx = ((min(xs) + max(xs)) * 0.5) * w
                        cy = ((min(ys) + max(ys)) * 0.5) * h
                    else:
                        cx = ((x1 + x2) * 0.5) * w
                        cy = ((y1 + y2) * 0.5) * h
                    span = max((x2 - x1) * w, (y2 - y1) * h) * 1.9
                    if min(w, h) >= YELLOW_CROP_MIN_PX:
                        crop_size = int(
                            min(
                                max(span, float(YELLOW_CROP_MIN_PX)),
                                float(YELLOW_CROP_MAX_PX),
                                float(w),
                                float(h),
                            )
                        )
                    else:
                        crop_size = min(w, h)
                    crop_region = f"face{primary.idx}"
                else:
                    metrics = self.repo.get_metrics(photo.primary_content_fp, ENGINE_VERSION)
                    cell = (metrics.exposure or {}).get("sharpest_cell") if metrics else None
                    if isinstance(cell, list) and len(cell) == 2:
                        r, c = int(cell[0]), int(cell[1])
                        cx = ((c + 0.5) / 16.0) * w
                        cy = ((r + 0.5) / 16.0) * h
                        crop_region = "sharpest"

            half = crop_size * 0.5
            left = int(max(0, min(w - crop_size, round(cx - half))))
            top = int(max(0, min(h - crop_size, round(cy - half))))
            cropped = rgb.crop((left, top, left + crop_size, top + crop_size))
            crop_png = _encode_lossless_png(cropped)

            longest = max(w, h)
            if longest > CONTEXT_MAX_EDGE_PX:
                scale = CONTEXT_MAX_EDGE_PX / float(longest)
                ctx_im = rgb.resize(
                    (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                    Image.Resampling.LANCZOS,
                )
            else:
                ctx_im = rgb
            ctx_png = _encode_lossless_png(ctx_im)

        return (
            ImageInput(data=crop_png, mime_type="image/png", label=f"{photo_id}:detail_crop"),
            ImageInput(data=ctx_png, mime_type="image/png", label=f"{photo_id}:context_512"),
            crop_region,
        )

    def _build_burst_images(
        self,
        burst_detail: BurstDetail,
    ) -> tuple[list[str], tuple[ImageInput, ...]]:
        """Build lossless PNG numbered contact sheet + 1:1 horizontal face strip (bbox + 5-pt landmarks).

        Single lossy JPEG encoding and EXIF/max_pixels sanitization occur once inside ModelRouter.
        """
        selected = list(burst_detail.members[:6])
        member_ids = [m.photo_id for m in selected]
        sheet_bytes, _legacy_strip = self.review.build_burst_compare_images(
            member_ids,
            max_frames=6,
        )

        face_tiles: list[tuple[int, Image.Image]] = []
        for idx, pid in enumerate(member_ids, start=1):
            photo = self.repo.get_photo(pid)
            if photo is None:
                continue
            faces = self.repo.list_faces(photo.primary_content_fp)
            if not faces:
                continue
            primary = max(
                faces,
                key=lambda f: (float(f.area_ratio or 0.0), float(f.det_score or 0.0)),
            )
            loupe_bytes = self.review.get_photo_image_bytes(pid, size="loupe")
            with Image.open(io.BytesIO(loupe_bytes)) as im:
                rgb = im.convert("RGB")
                w, h = rgb.size
                x1, y1, x2, y2 = primary.bbox
                if primary.landmarks:
                    xs = [pt[0] for pt in primary.landmarks] + [x1, x2]
                    ys = [pt[1] for pt in primary.landmarks] + [y1, y2]
                    cx = ((min(xs) + max(xs)) * 0.5) * w
                    cy = ((min(ys) + max(ys)) * 0.5) * h
                    span_x = (max(xs) - min(xs)) * w
                    span_y = (max(ys) - min(ys)) * h
                else:
                    cx = ((x1 + x2) * 0.5) * w
                    cy = ((y1 + y2) * 0.5) * h
                    span_x = (x2 - x1) * w
                    span_y = (y2 - y1) * h
                crop_edge = int(
                    min(
                        max(max(span_x, span_y) * 1.8, float(BURST_FACE_TILE_PX)),
                        float(min(w, h)),
                    )
                )
                half = crop_edge * 0.5
                left = int(max(0, min(w - crop_edge, round(cx - half))))
                top = int(max(0, min(h - crop_edge, round(cy - half))))
                tile = rgb.crop((left, top, left + crop_edge, top + crop_edge)).resize(
                    (BURST_FACE_TILE_PX, BURST_FACE_TILE_PX),
                    Image.Resampling.LANCZOS,
                )
                face_tiles.append((idx, tile))

        images: list[ImageInput] = [
            ImageInput(
                data=sheet_bytes,
                mime_type="image/webp",
                label=f"{burst_detail.burst.id}:contact_sheet",
            )
        ]

        if face_tiles:
            pad = 6
            strip_w = len(face_tiles) * BURST_FACE_TILE_PX + (len(face_tiles) + 1) * pad
            strip_h = BURST_FACE_TILE_PX + 2 * pad
            strip = Image.new("RGB", (strip_w, strip_h), (24, 24, 27))
            draw = ImageDraw.Draw(strip)
            for pos, (frame_idx, tile) in enumerate(face_tiles):
                fx = pad + pos * (BURST_FACE_TILE_PX + pad)
                strip.paste(tile, (fx, pad))
                draw.rectangle((fx + 4, pad + 4, fx + 34, pad + 24), fill=(0, 0, 0))
                draw.text((fx + 9, pad + 8), f"#{frame_idx}", fill=(255, 255, 255))
            images.append(
                ImageInput(
                    data=_encode_lossless_png(strip),
                    mime_type="image/png",
                    label=f"{burst_detail.burst.id}:face_strip",
                )
            )

        return member_ids, tuple(images)

    def _read_effective_preferences(self) -> str:
        """Read path-redacted active text preferences from `preferences.md` for VLM alignment (SPEC §7.6)."""
        session_roots = [s.root_path for s in self.repo.list_sessions()]
        return extract_effective_preferences_for_prompt(
            self.app_paths,
            max_chars=2000,
            extra_paths=session_roots,
        )

    def _build_yellow_messages(
        self,
        photo: PhotoRecord,
        crop_img: ImageInput,
        ctx_img: ImageInput,
    ) -> list[Message]:
        # Privacy: NEVER send filenames, paths, GPS, or API keys in prompts.
        # Use deterministic engine flags (not prior VLM verdicts) so repeat runs share the exact cache key.
        history = self.repo.list_verdict_history(photo.id)
        engine_verdict = next(
            (item for item in reversed(history) if item.source == "engine"),
            None,
        )
        m = self.repo.get_metrics(photo.primary_content_fp, ENGINE_VERSION)
        reasons = (
            ", ".join(engine_verdict.reason_codes)
            if engine_verdict and engine_verdict.reason_codes
            else "none"
        )
        exif_brief = self.review.format_exif_brief(photo.id) or "unknown"
        sharp_p95 = f"{m.sharp_p95:.2f}" if m and m.sharp_p95 is not None else "n/a"
        motion_state = m.motion_state if m and m.motion_state else "unknown"

        system_prompt = (
            "You are a photo culling visual reviewer. "
            "Conservative rule: never reject a potentially good or creatively intentional shot. "
            "If uncertain, return decision='unsure'. Return strict JSON matching the schema."
        )
        prefs = self._read_effective_preferences()
        if prefs:
            system_prompt = f"{system_prompt}\n\nActive photographer preferences:\n{prefs}"
        user_prompt = (
            f"Photo handle: {photo.id}\n"
            f"Engine flags: {reasons}\n"
            f"Metrics: sharp_p95={sharp_p95}, motion_state={motion_state}, exif={exif_brief}\n"
            "Image 1 is a 1:1 crop of the primary face or sharpest region. "
            "Image 2 is the 512px full-frame context.\n"
            "Classify as 'keep', 'reject' (only for clear technical failure), or 'unsure'."
        )
        return [
            Message(role="system", text=system_prompt),
            Message(role="user", text=user_prompt, images=(crop_img, ctx_img)),
        ]

    def _build_burst_messages(
        self,
        burst_detail: BurstDetail,
        member_ids: Sequence[str],
        images: Sequence[ImageInput],
    ) -> list[Message]:
        lines: list[str] = []
        for idx, pid in enumerate(member_ids, start=1):
            m = self.repo.get_photo(pid)
            metrics = (
                self.repo.get_metrics(m.primary_content_fp, ENGINE_VERSION)
                if m is not None
                else None
            )
            sharp = f"{metrics.sharp_p95:.2f}" if metrics and metrics.sharp_p95 is not None else "n/a"
            fsharp = (
                f"{metrics.face_sharp_min:.2f}"
                if metrics and metrics.face_sharp_min is not None
                else "n/a"
            )
            lines.append(f"Frame #{idx}: sharp_p95={sharp}, face_sharp_min={fsharp}")

        system_prompt = (
            "You are selecting the best frame from a burst sequence. "
            "Evaluate sharpness, eyes open, and natural expression. Return strict JSON."
        )
        prefs = self._read_effective_preferences()
        if prefs:
            system_prompt = f"{system_prompt}\n\nActive photographer preferences:\n{prefs}"
        user_prompt = (
            f"Burst group {burst_detail.burst.id} ({len(member_ids)} frames, numbered #1..#{len(member_ids)}).\n"
            + "\n".join(lines)
            + "\nImage 1 is the numbered contact sheet"
            + (" and Image 2 is the 1:1 horizontal face crop strip." if len(images) > 1 else ".")
            + f" Pick the single best frame number between 1 and {len(member_ids)}."
        )
        return [
            Message(role="system", text=system_prompt),
            Message(role="user", text=user_prompt, images=tuple(images)),
        ]

    async def run_triage(
        self,
        session_id: str,
        *,
        max_photos: int = 50,
        profile: str | None = None,
        session_budget_usd: float | None = None,
        bypass_monthly_budget: bool = False,
        job_id: str | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> TriageResult:
        """Execute concurrent L3 VLM triage on uncertain photos and unconfirmed bursts."""
        yellow_photos, bursts, _, _ = self._select_targets(session_id, max_photos=max_photos)
        prof_name = profile or self.config.model_profile
        _alias, spec = self.model_router.resolve_spec("triage_vlm", profile=prof_name)
        # Validate adapter/key availability upfront before mutating jobs table
        self.model_router.get_model("triage_vlm", profile=prof_name)

        eff_budget = (
            session_budget_usd
            if session_budget_usd is not None
            else self.config.budget.session_budget_usd
        )
        now = time.time()
        if job_id is not None and self.repo.get_job(job_id) is not None:
            resolved_job_id = job_id
            self.repo.update_job(
                resolved_job_id,
                status="running",
                clear_paused_reason=True,
                heartbeat_at=now,
            )
        else:
            resolved_job_id = job_id or f"triage_{uuid.uuid4().hex[:12]}"
            self.repo.create_job(
                JobRecord(
                    id=resolved_job_id,
                    kind="triage_vlm",
                    session_id=session_id,
                    status="running",
                    created_at=now,
                    budget_usd=eff_budget,
                    spent_usd=0.0,
                    heartbeat_at=now,
                )
            )

        for photo in yellow_photos:
            self.repo.upsert_job_item(
                JobItemRecord(
                    job_id=resolved_job_id,
                    photo_id=photo.id,
                    stage="triage_yellow",
                    status="pending",
                    updated_at=now,
                )
            )
        for bd in bursts:
            self.repo.upsert_job_item(
                JobItemRecord(
                    job_id=resolved_job_id,
                    photo_id=bd.burst.id,
                    stage="triage_burst",
                    status="pending",
                    updated_at=now,
                )
            )

        total_units = len(yellow_photos) + len(bursts)
        if total_units == 0:
            self.repo.update_job(
                resolved_job_id,
                status="completed",
                clear_paused_reason=True,
                spent_usd=0.0,
                heartbeat_at=time.time(),
            )
            return TriageResult(
                job_id=resolved_job_id,
                session_id=session_id,
                status="completed",
                paused_reason=None,
                yellow_total=0,
                burst_total=0,
                reviewed_yellow=0,
                reviewed_bursts=0,
                cache_hits=0,
                paid_calls=0,
                fresh_calls=0,
                spent_usd=0.0,
                model_id=spec.model_id,
                provider=spec.provider,
                failed_yellow=0,
                failed_bursts=0,
                failed_count=0,
            )

        budget_stop = asyncio.Event()
        counts = {
            "done": 0,
            "reviewed_yellow": 0,
            "reviewed_bursts": 0,
            "failed_yellow": 0,
            "failed_bursts": 0,
            "cache_hits": 0,
            "paid_calls": 0,
            "fresh_calls": 0,
            "spent_usd": 0.0,
        }
        first_error: Exception | None = None

        def _record_usage(cache_hit: bool, cost_usd: float) -> None:
            if cache_hit:
                counts["cache_hits"] += 1
            else:
                counts["fresh_calls"] += 1
                if cost_usd > 0.0:
                    counts["paid_calls"] += 1
                counts["spent_usd"] += cost_usd

        async def _process_yellow(photo: PhotoRecord) -> None:
            nonlocal first_error
            if budget_stop.is_set():
                return
            try:
                crop_img, ctx_img, crop_region = self._build_yellow_images(photo.id)
                msgs = self._build_yellow_messages(photo, crop_img, ctx_img)
                if budget_stop.is_set():
                    return
                res = await self.model_router.generate(
                    role="triage_vlm",
                    messages=msgs,
                    response_schema=YellowTriageOutput,
                    max_tokens=512,
                    effort=spec.default_effort or "low",
                    profile=prof_name,
                    session_id=session_id,
                    session_budget_usd=eff_budget,
                    bypass_monthly_budget=bypass_monthly_budget,
                )
                # Account spend immediately after paid call returns, even if persistence below fails
                _record_usage(res.cache_hit, res.cost_usd)

                if res.refused:
                    target_bucket: BucketName = "yellow"
                    reason_codes: tuple[str, ...] = (REASON_VLM_UNSURE,)
                    conf = 0.5
                    rationale = "VLM refused to classify; kept yellow for manual review."
                    decision = "unsure"
                else:
                    out = res.parsed
                    if not isinstance(out, YellowTriageOutput):
                        raise ValueError(f"Unexpected VLM output type for {photo.id}: {type(out)!r}")
                    decision = out.decision
                    conf = float(out.confidence)
                    rationale = out.rationale
                    if decision == "keep":
                        target_bucket = "normal"
                        reason_codes = (out.reason_code or REASON_VLM_KEEP,)
                    elif decision == "reject":
                        if photo.is_protected:
                            target_bucket = "exempt"
                            reason_codes = (REASON_VLM_UNSURE,)
                        else:
                            target_bucket = "red"
                            default_rej = out.reason_code or REASON_VLM_REJECT
                            if default_rej not in {
                                REASON_MOTION_BLUR,
                                REASON_OUT_OF_FOCUS,
                                REASON_SEVERE_BLUR,
                                REASON_EXTREME_EXPOSURE,
                                REASON_FACE_OUT_OF_FOCUS,
                                REASON_BORDERLINE_BLUR,
                                REASON_LOW_RELATIVE_SHARPNESS,
                            }:
                                default_rej = REASON_VLM_REJECT
                            reason_codes = (default_rej,)
                    else:
                        target_bucket = "yellow"
                        reason_codes = (out.reason_code or REASON_VLM_UNSURE,)

                self.repo.add_verdict(
                    photo_id=photo.id,
                    source="vlm",
                    bucket=target_bucket,
                    confidence=conf,
                    reason_codes=reason_codes,
                    detail={
                        "vlm_decision": decision,
                        "vlm_rationale": rationale,
                        "crop_region": crop_region,
                        "refused": res.refused,
                    },
                    llm_call_id=res.llm_call_id,
                )
                self.repo.upsert_job_item(
                    JobItemRecord(
                        job_id=resolved_job_id,
                        photo_id=photo.id,
                        stage="triage_yellow",
                        status="done",
                        attempts=1,
                        updated_at=time.time(),
                    )
                )
                counts["done"] += 1
                counts["reviewed_yellow"] += 1
                if on_progress is not None:
                    on_progress(
                        "triage_vlm",
                        int(counts["done"] + counts["failed_yellow"] + counts["failed_bursts"]),
                        total_units,
                    )
            except BudgetExceededError:
                budget_stop.set()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                counts["failed_yellow"] += 1
                self.repo.upsert_job_item(
                    JobItemRecord(
                        job_id=resolved_job_id,
                        photo_id=photo.id,
                        stage="triage_yellow",
                        status="failed",
                        attempts=1,
                        last_error=f"{type(exc).__name__}: {exc}",
                        updated_at=time.time(),
                    )
                )
                if on_progress is not None:
                    on_progress(
                        "triage_vlm",
                        int(counts["done"] + counts["failed_yellow"] + counts["failed_bursts"]),
                        total_units,
                    )

        async def _process_burst(burst_detail: BurstDetail) -> None:
            nonlocal first_error
            if budget_stop.is_set():
                return
            try:
                member_ids, images = self._build_burst_images(burst_detail)
                msgs = self._build_burst_messages(burst_detail, member_ids, images)
                if budget_stop.is_set():
                    return
                res = await self.model_router.generate(
                    role="triage_vlm",
                    messages=msgs,
                    response_schema=BurstTriageOutput,
                    max_tokens=512,
                    effort=spec.default_effort or "low",
                    profile=prof_name,
                    session_id=session_id,
                    session_budget_usd=eff_budget,
                    bypass_monthly_budget=bypass_monthly_budget,
                )
                # Account spend immediately after paid call returns, even if persistence below fails
                _record_usage(res.cache_hit, res.cost_usd)

                if not res.refused and isinstance(res.parsed, BurstTriageOutput):
                    out = res.parsed
                    idx = min(max(1, out.pick_frame_no), len(member_ids)) - 1
                    pick_pid = member_ids[idx]
                    note = out.rationale
                    if out.eyes_open is False:
                        note = f"[eyes_closed_warning] {note}"
                    self.burst.recommend_burst_pick(
                        burst_detail.burst.id,
                        pick_pid,
                        note=note[:300],
                    )

                self.repo.upsert_job_item(
                    JobItemRecord(
                        job_id=resolved_job_id,
                        photo_id=burst_detail.burst.id,
                        stage="triage_burst",
                        status="done",
                        attempts=1,
                        updated_at=time.time(),
                    )
                )
                counts["done"] += 1
                counts["reviewed_bursts"] += 1
                if on_progress is not None:
                    on_progress(
                        "triage_vlm",
                        int(counts["done"] + counts["failed_yellow"] + counts["failed_bursts"]),
                        total_units,
                    )
            except BudgetExceededError:
                budget_stop.set()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                counts["failed_bursts"] += 1
                self.repo.upsert_job_item(
                    JobItemRecord(
                        job_id=resolved_job_id,
                        photo_id=burst_detail.burst.id,
                        stage="triage_burst",
                        status="failed",
                        attempts=1,
                        last_error=f"{type(exc).__name__}: {exc}",
                        updated_at=time.time(),
                    )
                )
                if on_progress is not None:
                    on_progress(
                        "triage_vlm",
                        int(counts["done"] + counts["failed_yellow"] + counts["failed_bursts"]),
                        total_units,
                    )

        tasks = [_process_yellow(p) for p in yellow_photos] + [
            _process_burst(b) for b in bursts
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException as exc:
            spent = round(float(counts["spent_usd"]), 8)
            self.repo.update_job(
                resolved_job_id,
                status="failed",
                paused_reason=f"{type(exc).__name__}: {exc}",
                spent_usd=spent,
                heartbeat_at=time.time(),
            )
            raise

        spent = round(float(counts["spent_usd"]), 8)
        failed_yellow = int(counts["failed_yellow"])
        failed_bursts = int(counts["failed_bursts"])
        failed_count = failed_yellow + failed_bursts
        reviewed_total = int(counts["reviewed_yellow"]) + int(counts["reviewed_bursts"])

        if budget_stop.is_set():
            status: TriageStatus = "paused"
            paused_reason: str | None = "budget_exceeded"
            self.repo.update_job(
                resolved_job_id,
                status="paused",
                paused_reason="budget_exceeded",
                spent_usd=spent,
                heartbeat_at=time.time(),
            )
        elif failed_count > 0 and reviewed_total == 0:
            status = "failed"
            paused_reason = str(first_error) if first_error is not None else "all_items_failed"
            self.repo.update_job(
                resolved_job_id,
                status="failed",
                paused_reason=paused_reason,
                spent_usd=spent,
                heartbeat_at=time.time(),
            )
        elif failed_count > 0:
            status = "completed_with_errors"
            err_detail = str(first_error) if first_error is not None else "item_error"
            paused_reason = f"{failed_count} item(s) failed: {err_detail}"
            self.repo.update_job(
                resolved_job_id,
                status="completed_with_errors",
                paused_reason=paused_reason,
                spent_usd=spent,
                heartbeat_at=time.time(),
            )
        else:
            status = "completed"
            paused_reason = None
            self.repo.update_job(
                resolved_job_id,
                status="completed",
                clear_paused_reason=True,
                spent_usd=spent,
                heartbeat_at=time.time(),
            )

        return TriageResult(
            job_id=resolved_job_id,
            session_id=session_id,
            status=status,
            paused_reason=paused_reason,
            yellow_total=len(yellow_photos),
            burst_total=len(bursts),
            reviewed_yellow=int(counts["reviewed_yellow"]),
            reviewed_bursts=int(counts["reviewed_bursts"]),
            cache_hits=int(counts["cache_hits"]),
            paid_calls=int(counts["paid_calls"]),
            fresh_calls=int(counts["fresh_calls"]),
            spent_usd=spent,
            model_id=spec.model_id,
            provider=spec.provider,
            failed_yellow=failed_yellow,
            failed_bursts=failed_bursts,
            failed_count=failed_count,
        )

    def run_triage_sync(
        self,
        session_id: str,
        *,
        max_photos: int = 50,
        profile: str | None = None,
        session_budget_usd: float | None = None,
        bypass_monthly_budget: bool = False,
        job_id: str | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> TriageResult:
        """Synchronous wrapper around `run_triage` for CLI, worker threads, and sync ToolRegistry handlers."""
        coro = self.run_triage(
            session_id,
            max_photos=max_photos,
            profile=profile,
            session_budget_usd=session_budget_usd,
            bypass_monthly_budget=bypass_monthly_budget,
            job_id=job_id,
            on_progress=on_progress,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()


__all__ = [
    "BURST_FACE_TILE_PX",
    "BurstTriageOutput",
    "CONTEXT_MAX_EDGE_PX",
    "TriageEstimate",
    "TriageResult",
    "TriageService",
    "TriageStatus",
    "YELLOW_CROP_MAX_PX",
    "YELLOW_CROP_MIN_PX",
    "YELLOW_CROP_TARGET_PX",
    "YellowTriageOutput",
]
