"""DigestService: deterministic SQL shooting digest / daily report (SPEC §13(2), PLAN T17)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from lightbox.i18n import (
    BURST_CULL_REASON_CODES,
    TECH_REJECT_REASON_CODES,
    render_reason,
)
from lightbox.models import get_scene_label
from lightbox.store import (
    DigestBucketRow,
    DigestMatrixRow,
    LightboxRepository,
)

DEFAULT_MIN_SAMPLES = 5
MAX_HIGHLIGHT_ITEMS = 6
MAX_STABLE_SETUPS = 8
HIGHLIGHT_MIN_DELTA = 0.05
HIGHLIGHT_TECH_MIN_RATIO = 1.5
HIGHLIGHT_MIN_RATIO_DELTA = 0.02
HIGHLIGHT_KEPT_CEILING_BASE = 0.90

DIGEST_DIMENSIONS: tuple[str, ...] = (
    "light_bucket",
    "capture_local_hour",
    "flash_fired",
    "scene_tag",
    "motion_state",
    "lens_model",
    "focal_bucket",
    "shutter_vs_safe",
    "iso_bucket",
)

_DIMENSION_LABELS: dict[str, tuple[str, str]] = {
    "light_bucket": ("光线环境", "Lighting"),
    "capture_local_hour": ("拍摄时段", "Time of day"),
    "flash_fired": ("闪光灯", "Flash"),
    "scene_tag": ("画面场景", "Scene"),
    "motion_state": ("主体动静", "Subject motion"),
    "lens_model": ("镜头", "Lens"),
    "focal_bucket": ("等效焦段", "Focal length"),
    "shutter_vs_safe": ("快门 vs 安全快门", "Shutter vs safe speed"),
    "iso_bucket": ("ISO 档位", "ISO range"),
}

_BUCKET_LABELS: dict[str, dict[str, tuple[str, str]]] = {
    "light_bucket": {
        "bright": ("明亮日光", "Bright daylight"),
        "daylight": ("明亮日光", "Bright daylight"),
        "indoor": ("室内 / 阴天", "Indoor / overcast"),
        "overcast_indoor": ("室内 / 阴天", "Indoor / overcast"),
        "low_light": ("弱光", "Low light"),
        "night": ("暗光 / 夜景", "Low light / night"),
        "unknown": ("未知", "Unknown"),
    },
    "capture_local_hour": {
        "05_08_dawn": ("05:00–08:59 清晨", "05:00–08:59 Dawn"),
        "09_11_morning": ("09:00–11:59 上午", "09:00–11:59 Morning"),
        "12_16_afternoon": ("12:00–16:59 午后", "12:00–16:59 Afternoon"),
        "17_19_evening": ("17:00–19:59 傍晚 / 黄金时刻", "17:00–19:59 Evening / Golden hour"),
        "20_23_night": ("20:00–23:59 夜间", "20:00–23:59 Night"),
        "00_04_late_night": ("00:00–04:59 凌晨", "00:00–04:59 Late night"),
        "unknown": ("未知", "Unknown"),
    },
    "flash_fired": {
        "fired": ("开启闪光灯", "Flash fired"),
        "no_flash": ("未开闪光灯", "No flash"),
        "unknown": ("未知", "Unknown"),
    },
    "motion_state": {
        "static": ("静止画面", "Static"),
        "moving": ("运动主体", "Moving"),
        "panning": ("追焦摇拍", "Panning"),
        "unknown": ("未知", "Unknown"),
    },
    "focal_bucket": {
        "ultra_wide": ("<24mm 超广角", "<24mm Ultra-wide"),
        "wide_24_35": ("24–35mm 广角", "24–35mm Wide"),
        "standard_36_55": ("36–55mm 标准", "36–55mm Standard"),
        "portrait_56_105": ("56–105mm 中焦人像", "56–105mm Portrait"),
        "tele_106_200": ("106–200mm 长焦", "106–200mm Telephoto"),
        "super_tele_200p": (">200mm 超长焦", ">200mm Super-tele"),
        "unknown": ("未知", "Unknown"),
    },
    "shutter_bucket": {
        "fast_1000": ("≤1/1000s 极速", "≤1/1000s Ultra-fast"),
        "fast_250_1000": ("1/1000–1/250s 高速", "1/1000–1/250s Fast"),
        "mid_60_250": ("1/250–1/60s 常规手持", "1/250–1/60s Handheld"),
        "slow_handheld": ("1/60–1s 慢门", "1/60–1s Slow"),
        "long_1s_plus": ("≥1s 长曝光", "≥1s Long exposure"),
        "unknown": ("未知", "Unknown"),
    },
    "shutter_vs_safe": {
        "at_or_faster_safe": ("≥ 安全快门（1/焦距）", "≥ Safe shutter (1/focal)"),
        "slower_up_to_1_stop": ("慢于安全快门 ≤1 档", "Up to 1 stop slower"),
        "slower_over_1_stop": ("慢于安全快门 >1 档", ">1 stop slower"),
        "unknown": ("未知", "Unknown"),
    },
    "iso_bucket": {
        "iso_le_200": ("ISO ≤200", "ISO ≤200"),
        "iso_201_800": ("ISO 201–800", "ISO 201–800"),
        "iso_801_3200": ("ISO 801–3200", "ISO 801–3200"),
        "iso_3201_6400": ("ISO 3201–6400", "ISO 3201–6400"),
        "iso_gt_6400": ("ISO >6400", "ISO >6400"),
        "unknown": ("未知", "Unknown"),
    },
}

_ORDERED_BUCKET_KEYS: dict[str, tuple[str, ...]] = {
    "light_bucket": ("bright", "daylight", "indoor", "overcast_indoor", "low_light", "night"),
    "capture_local_hour": (
        "05_08_dawn",
        "09_11_morning",
        "12_16_afternoon",
        "17_19_evening",
        "20_23_night",
        "00_04_late_night",
    ),
    "flash_fired": ("fired", "no_flash"),
    "motion_state": ("static", "moving", "panning"),
    "focal_bucket": (
        "ultra_wide",
        "wide_24_35",
        "standard_36_55",
        "portrait_56_105",
        "tele_106_200",
        "super_tele_200p",
    ),
    "shutter_vs_safe": ("at_or_faster_safe", "slower_up_to_1_stop", "slower_over_1_stop"),
    "iso_bucket": (
        "iso_le_200",
        "iso_201_800",
        "iso_801_3200",
        "iso_3201_6400",
        "iso_gt_6400",
    ),
}


def _safe_rate(num: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return round(num / float(denom), 4)


def _is_zh(lang: str) -> bool:
    return not lang.lower().startswith("en")


def _dimension_label(dim_key: str, lang: str) -> str:
    pair = _DIMENSION_LABELS.get(dim_key)
    if pair is None:
        return dim_key
    return pair[0] if _is_zh(lang) else pair[1]


def _bucket_label(dim_key: str, bucket_key: str, lang: str) -> str:
    if bucket_key == "unknown":
        return "未知" if _is_zh(lang) else "Unknown"
    if dim_key == "scene_tag":
        return get_scene_label(bucket_key, lang=lang) or bucket_key
    if dim_key == "lens_model":
        return bucket_key
    dim_map = _BUCKET_LABELS.get(dim_key)
    if dim_map is not None and bucket_key in dim_map:
        pair = dim_map[bucket_key]
        return pair[0] if _is_zh(lang) else pair[1]
    return bucket_key


def _parse_iso_date_bounds(date_str: str, *, end_of_day: bool = False) -> float:
    """Parse `YYYY-MM-DD` into a UTC POSIX timestamp (start or end of day)."""
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date {date_str!r}; expected YYYY-MM-DD.") from exc
    ts = dt.timestamp()
    return ts + 86399.999 if end_of_day else ts


@dataclass(frozen=True, slots=True)
class DigestOverview:
    """Top-level mutually exclusive outcome counts and explicit rates (SPEC §13(2))."""

    total_shots: int
    tech_reject_count: int
    tech_reject_rate: float
    burst_cull_count: int
    burst_cull_rate: float
    subjective_reject_count: int
    subjective_reject_rate: float
    kept_count: int
    non_burst_total: int
    kept_rate_excl_burst: float
    kept_rate_total: float
    uncertain_count: int
    uncertain_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_shots": self.total_shots,
            "tech_reject_count": self.tech_reject_count,
            "tech_reject_rate": self.tech_reject_rate,
            "burst_cull_count": self.burst_cull_count,
            "burst_cull_rate": self.burst_cull_rate,
            "subjective_reject_count": self.subjective_reject_count,
            "subjective_reject_rate": self.subjective_reject_rate,
            "kept_count": self.kept_count,
            "non_burst_total": self.non_burst_total,
            "kept_rate_excl_burst": self.kept_rate_excl_burst,
            "kept_rate_total": self.kept_rate_total,
            "uncertain_count": self.uncertain_count,
            "uncertain_rate": self.uncertain_rate,
        }


@dataclass(frozen=True, slots=True)
class DigestReasonStat:
    """Technical failure reason breakdown."""

    code: str
    label: str
    count: int
    rate_of_tech_rejects: float
    rate_of_total: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "count": self.count,
            "rate_of_tech_rejects": self.rate_of_tech_rejects,
            "rate_of_total": self.rate_of_total,
        }


@dataclass(frozen=True, slots=True)
class DigestSliceBucket:
    """One bucket inside a single-dimension SQL GROUP BY slice."""

    key: str
    label: str
    total: int
    non_burst_total: int
    tech_reject: int
    tech_reject_rate: float
    burst_cull: int
    burst_cull_rate: float
    subjective_reject: int
    subjective_reject_rate: float
    kept: int
    kept_rate_excl_burst: float
    kept_rate_total: float
    uncertain: int
    uncertain_rate: float
    meets_min_samples: bool
    top_reason_code: str | None = None
    top_reason_label: str | None = None
    top_reason_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "total": self.total,
            "non_burst_total": self.non_burst_total,
            "tech_reject": self.tech_reject,
            "tech_reject_rate": self.tech_reject_rate,
            "burst_cull": self.burst_cull,
            "burst_cull_rate": self.burst_cull_rate,
            "subjective_reject": self.subjective_reject,
            "subjective_reject_rate": self.subjective_reject_rate,
            "kept": self.kept,
            "kept_rate_excl_burst": self.kept_rate_excl_burst,
            "kept_rate_total": self.kept_rate_total,
            "uncertain": self.uncertain,
            "uncertain_rate": self.uncertain_rate,
            "meets_min_samples": self.meets_min_samples,
            "top_reason_code": self.top_reason_code,
            "top_reason_label": self.top_reason_label,
            "top_reason_count": self.top_reason_count,
        }


@dataclass(frozen=True, slots=True)
class DigestDimension:
    """Single-dimension slice summary."""

    key: str
    label: str
    buckets: tuple[DigestSliceBucket, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "buckets": [b.to_dict() for b in self.buckets],
        }


@dataclass(frozen=True, slots=True)
class DigestConditionHighlight:
    """Highlighted high-tech-reject or high-kept condition meaningfully above overall baseline."""

    dimension_key: str
    dimension_label: str
    bucket_key: str
    bucket_label: str
    total: int
    non_burst_total: int
    tech_reject: int
    tech_reject_rate: float
    tech_reject_delta: float
    kept: int
    kept_rate_excl_burst: float
    kept_rate_delta: float
    kept_rate_total: float
    top_reason_code: str | None = None
    top_reason_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension_key": self.dimension_key,
            "dimension_label": self.dimension_label,
            "bucket_key": self.bucket_key,
            "bucket_label": self.bucket_label,
            "total": self.total,
            "non_burst_total": self.non_burst_total,
            "tech_reject": self.tech_reject,
            "tech_reject_rate": self.tech_reject_rate,
            "tech_reject_delta": self.tech_reject_delta,
            "kept": self.kept,
            "kept_rate_excl_burst": self.kept_rate_excl_burst,
            "kept_rate_delta": self.kept_rate_delta,
            "kept_rate_total": self.kept_rate_total,
            "top_reason_code": self.top_reason_code,
            "top_reason_label": self.top_reason_label,
        }


@dataclass(frozen=True, slots=True)
class DigestStableSetup:
    """Fixed 2D `focal_bucket x shutter_bucket` stable shooting setup summary."""

    focal_bucket: str
    focal_label: str
    shutter_bucket: str
    shutter_label: str
    total: int
    non_burst_total: int
    kept: int
    tech_reject: int
    burst_cull: int
    subjective_reject: int
    uncertain: int
    kept_rate_excl_burst: float
    kept_rate_total: float
    tech_reject_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "focal_bucket": self.focal_bucket,
            "focal_label": self.focal_label,
            "shutter_bucket": self.shutter_bucket,
            "shutter_label": self.shutter_label,
            "total": self.total,
            "non_burst_total": self.non_burst_total,
            "kept": self.kept,
            "tech_reject": self.tech_reject,
            "burst_cull": self.burst_cull,
            "subjective_reject": self.subjective_reject,
            "uncertain": self.uncertain,
            "kept_rate_excl_burst": self.kept_rate_excl_burst,
            "kept_rate_total": self.kept_rate_total,
            "tech_reject_rate": self.tech_reject_rate,
        }


@dataclass(frozen=True, slots=True)
class ShootingDigestReport:
    """Complete zero-LLM shooting digest report for a single session or cross-session range."""

    scope_mode: str  # 'session' | 'cross_session'
    session_id: str | None
    session_name: str | None
    session_count: int
    date_from: str | None
    date_to: str | None
    min_samples: int
    lang: str
    overview: DigestOverview
    top_tech_reasons: tuple[DigestReasonStat, ...]
    dimensions: tuple[DigestDimension, ...]
    high_tech_reject_conditions: tuple[DigestConditionHighlight, ...]
    high_kept_conditions: tuple[DigestConditionHighlight, ...]
    stable_setups: tuple[DigestStableSetup, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": {
                "mode": self.scope_mode,
                "session_id": self.session_id,
                "session_name": self.session_name,
                "session_count": self.session_count,
                "date_from": self.date_from,
                "date_to": self.date_to,
                "min_samples": self.min_samples,
            },
            "lang": self.lang,
            "overview": self.overview.to_dict(),
            "top_tech_reasons": [r.to_dict() for r in self.top_tech_reasons],
            "dimensions": [d.to_dict() for d in self.dimensions],
            "high_tech_reject_conditions": [
                c.to_dict() for c in self.high_tech_reject_conditions
            ],
            "high_kept_conditions": [c.to_dict() for c in self.high_kept_conditions],
            "stable_setups": [s.to_dict() for s in self.stable_setups],
        }


class DigestService:
    """SQL-backed shooting digest service strictly separating technical rejects from burst culls."""

    def __init__(self, repo: LightboxRepository) -> None:
        self.repo = repo

    def generate_digest(
        self,
        *,
        session_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        lang: str = "zh-CN",
    ) -> ShootingDigestReport:
        """Compute a single-session or cross-session shooting digest without any LLM calls."""
        clamped_min = max(1, min(int(min_samples), 1000))

        session_name: str | None = None
        if session_id is not None and session_id.strip():
            session_id = session_id.strip()
            sess = self.repo.get_session(session_id)
            if sess is None:
                raise ValueError(f"Unknown session_id: {session_id}")
            session_name = sess.name
            scope_mode = "session"
        else:
            session_id = None
            scope_mode = "cross_session"

        norm_date_from = date_from.strip() if date_from and date_from.strip() else None
        norm_date_to = date_to.strip() if date_to and date_to.strip() else None
        min_ts = (
            _parse_iso_date_bounds(norm_date_from, end_of_day=False) if norm_date_from else None
        )
        max_ts = _parse_iso_date_bounds(norm_date_to, end_of_day=True) if norm_date_to else None
        if min_ts is not None and max_ts is not None and min_ts > max_ts:
            raise ValueError("`date_from` must be on or before `date_to`.")

        tech_codes = tuple(sorted(TECH_REJECT_REASON_CODES))
        burst_codes = tuple(sorted(BURST_CULL_REASON_CODES))

        ov_row = self.repo.query_digest_overview(
            tech_reason_codes=tech_codes,
            burst_reason_codes=burst_codes,
            session_id=session_id,
            min_capture_ts=min_ts,
            max_capture_ts=max_ts,
        )
        non_burst_total = max(0, ov_row.total_shots - ov_row.burst_cull)
        overview = DigestOverview(
            total_shots=ov_row.total_shots,
            tech_reject_count=ov_row.tech_reject,
            tech_reject_rate=_safe_rate(ov_row.tech_reject, ov_row.total_shots),
            burst_cull_count=ov_row.burst_cull,
            burst_cull_rate=_safe_rate(ov_row.burst_cull, ov_row.total_shots),
            subjective_reject_count=ov_row.subjective_reject,
            subjective_reject_rate=_safe_rate(ov_row.subjective_reject, ov_row.total_shots),
            kept_count=ov_row.kept,
            non_burst_total=non_burst_total,
            kept_rate_excl_burst=_safe_rate(ov_row.kept, non_burst_total),
            kept_rate_total=_safe_rate(ov_row.kept, ov_row.total_shots),
            uncertain_count=ov_row.uncertain,
            uncertain_rate=_safe_rate(ov_row.uncertain, ov_row.total_shots),
        )

        reason_counts = self.repo.query_digest_tech_reasons(
            tech_reason_codes=tech_codes,
            burst_reason_codes=burst_codes,
            session_id=session_id,
            min_capture_ts=min_ts,
            max_capture_ts=max_ts,
        )
        top_tech_reasons = tuple(
            DigestReasonStat(
                code=code,
                label=render_reason(code, lang=lang),
                count=cnt,
                rate_of_tech_rejects=_safe_rate(cnt, ov_row.tech_reject),
                rate_of_total=_safe_rate(cnt, ov_row.total_shots),
            )
            for code, cnt in reason_counts
        )

        overall_tech_rate = overview.tech_reject_rate
        overall_kept_rate = overview.kept_rate_excl_burst

        dimensions: list[DigestDimension] = []
        high_reject_candidates: list[DigestConditionHighlight] = []
        high_kept_candidates: list[DigestConditionHighlight] = []

        for dim_key in DIGEST_DIMENSIONS:
            dim_label = _dimension_label(dim_key, lang)
            raw_buckets = self.repo.query_digest_dimension(
                dim_key,
                tech_reason_codes=tech_codes,
                burst_reason_codes=burst_codes,
                session_id=session_id,
                min_capture_ts=min_ts,
                max_capture_ts=max_ts,
            )
            buckets = self._build_dimension_buckets(
                dim_key,
                raw_buckets,
                min_samples=clamped_min,
                lang=lang,
            )
            dimensions.append(DigestDimension(key=dim_key, label=dim_label, buckets=buckets))

            for bucket in buckets:
                if bucket.key == "unknown" or not bucket.meets_min_samples:
                    continue
                tech_delta = round(bucket.tech_reject_rate - overall_tech_rate, 4)
                kept_delta = round(bucket.kept_rate_excl_burst - overall_kept_rate, 4)

                is_high_tech = (
                    bucket.tech_reject > 0
                    and tech_delta > 0.0
                    and (
                        tech_delta >= HIGHLIGHT_MIN_DELTA
                        or (
                            overall_tech_rate > 0.0
                            and bucket.tech_reject_rate
                            >= overall_tech_rate * HIGHLIGHT_TECH_MIN_RATIO
                            and tech_delta >= HIGHLIGHT_MIN_RATIO_DELTA
                        )
                    )
                )
                is_high_kept = (
                    not is_high_tech
                    and bucket.non_burst_total > 0
                    and bucket.kept > 0
                    and kept_delta > 0.0
                    and bucket.tech_reject_rate <= overall_tech_rate
                    and (
                        kept_delta >= HIGHLIGHT_MIN_DELTA
                        or (
                            overall_kept_rate >= HIGHLIGHT_KEPT_CEILING_BASE
                            and bucket.kept_rate_excl_burst == 1.0
                            and kept_delta >= HIGHLIGHT_MIN_RATIO_DELTA
                        )
                    )
                )
                if not (is_high_tech or is_high_kept):
                    continue

                highlight = DigestConditionHighlight(
                    dimension_key=dim_key,
                    dimension_label=dim_label,
                    bucket_key=bucket.key,
                    bucket_label=bucket.label,
                    total=bucket.total,
                    non_burst_total=bucket.non_burst_total,
                    tech_reject=bucket.tech_reject,
                    tech_reject_rate=bucket.tech_reject_rate,
                    tech_reject_delta=tech_delta,
                    kept=bucket.kept,
                    kept_rate_excl_burst=bucket.kept_rate_excl_burst,
                    kept_rate_delta=kept_delta,
                    kept_rate_total=bucket.kept_rate_total,
                    top_reason_code=bucket.top_reason_code,
                    top_reason_label=bucket.top_reason_label,
                )
                if is_high_tech:
                    high_reject_candidates.append(highlight)
                elif is_high_kept:
                    high_kept_candidates.append(highlight)

        high_reject_candidates.sort(
            key=lambda h: (
                -h.tech_reject_delta,
                -h.tech_reject_rate,
                -h.tech_reject,
                -h.total,
                h.dimension_key,
                h.bucket_key,
            )
        )
        high_kept_candidates.sort(
            key=lambda h: (
                -h.kept_rate_delta,
                -h.kept_rate_excl_burst,
                h.tech_reject_rate,
                -h.kept,
                -h.total,
                h.dimension_key,
                h.bucket_key,
            )
        )

        matrix_rows = self.repo.query_digest_focal_shutter_matrix(
            tech_reason_codes=tech_codes,
            burst_reason_codes=burst_codes,
            session_id=session_id,
            min_capture_ts=min_ts,
            max_capture_ts=max_ts,
        )
        stable_setups = self._build_stable_setups(
            matrix_rows,
            min_samples=clamped_min,
            lang=lang,
        )

        return ShootingDigestReport(
            scope_mode=scope_mode,
            session_id=session_id,
            session_name=session_name,
            session_count=ov_row.session_count,
            date_from=norm_date_from,
            date_to=norm_date_to,
            min_samples=clamped_min,
            lang=lang,
            overview=overview,
            top_tech_reasons=top_tech_reasons,
            dimensions=tuple(dimensions),
            high_tech_reject_conditions=tuple(high_reject_candidates[:MAX_HIGHLIGHT_ITEMS]),
            high_kept_conditions=tuple(high_kept_candidates[:MAX_HIGHLIGHT_ITEMS]),
            stable_setups=stable_setups,
        )

    def _build_dimension_buckets(
        self,
        dim_key: str,
        rows: list[DigestBucketRow],
        *,
        min_samples: int,
        lang: str,
    ) -> tuple[DigestSliceBucket, ...]:
        buckets: list[DigestSliceBucket] = []
        for r in rows:
            nb_total = max(0, r.total - r.burst_cull)
            reason_code = r.top_reason_code if r.tech_reject > 0 else None
            reason_label = render_reason(reason_code, lang=lang) if reason_code else None
            buckets.append(
                DigestSliceBucket(
                    key=r.bucket_key,
                    label=_bucket_label(dim_key, r.bucket_key, lang),
                    total=r.total,
                    non_burst_total=nb_total,
                    tech_reject=r.tech_reject,
                    tech_reject_rate=_safe_rate(r.tech_reject, r.total),
                    burst_cull=r.burst_cull,
                    burst_cull_rate=_safe_rate(r.burst_cull, r.total),
                    subjective_reject=r.subjective_reject,
                    subjective_reject_rate=_safe_rate(r.subjective_reject, r.total),
                    kept=r.kept,
                    kept_rate_excl_burst=_safe_rate(r.kept, nb_total),
                    kept_rate_total=_safe_rate(r.kept, r.total),
                    uncertain=r.uncertain,
                    uncertain_rate=_safe_rate(r.uncertain, r.total),
                    meets_min_samples=r.total >= min_samples,
                    top_reason_code=reason_code,
                    top_reason_label=reason_label,
                    top_reason_count=r.top_reason_count if reason_code else 0,
                )
            )

        order = _ORDERED_BUCKET_KEYS.get(dim_key)
        if order is not None:
            order_idx = {k: i for i, k in enumerate(order)}
            buckets.sort(
                key=lambda b: (
                    1 if b.key == "unknown" else 0,
                    order_idx.get(b.key, len(order)),
                    -b.total,
                    b.key,
                )
            )
        else:
            buckets.sort(key=lambda b: (1 if b.key == "unknown" else 0, -b.total, b.key))
        return tuple(buckets)

    def _build_stable_setups(
        self,
        rows: list[DigestMatrixRow],
        *,
        min_samples: int,
        lang: str,
    ) -> tuple[DigestStableSetup, ...]:
        setups: list[DigestStableSetup] = []
        for r in rows:
            if r.total < min_samples:
                continue
            nb_total = max(0, r.total - r.burst_cull)
            if nb_total <= 0:
                continue
            setups.append(
                DigestStableSetup(
                    focal_bucket=r.focal_bucket,
                    focal_label=_bucket_label("focal_bucket", r.focal_bucket, lang),
                    shutter_bucket=r.shutter_bucket,
                    shutter_label=_bucket_label("shutter_bucket", r.shutter_bucket, lang),
                    total=r.total,
                    non_burst_total=nb_total,
                    kept=r.kept,
                    tech_reject=r.tech_reject,
                    burst_cull=r.burst_cull,
                    subjective_reject=r.subjective_reject,
                    uncertain=r.uncertain,
                    kept_rate_excl_burst=_safe_rate(r.kept, nb_total),
                    kept_rate_total=_safe_rate(r.kept, r.total),
                    tech_reject_rate=_safe_rate(r.tech_reject, r.total),
                )
            )
        setups.sort(
            key=lambda s: (
                -s.kept_rate_excl_burst,
                s.tech_reject_rate,
                -s.total,
                s.focal_bucket,
                s.shutter_bucket,
            )
        )
        return tuple(setups[:MAX_STABLE_SETUPS])


__all__ = [
    "DEFAULT_MIN_SAMPLES",
    "DIGEST_DIMENSIONS",
    "DigestConditionHighlight",
    "DigestDimension",
    "DigestOverview",
    "DigestReasonStat",
    "DigestService",
    "DigestSliceBucket",
    "DigestStableSetup",
    "ShootingDigestReport",
]
