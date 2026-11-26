"""Session-relative cohort calibration, dual-threshold red gating, and verdict engine (SPEC §4.2–4.3)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from lightbox.core.detector import DetectedFace
from lightbox.core.exif import NormalizedExif
from lightbox.core.faces import evaluate_subject_faces_oof
from lightbox.core.guards import (
    SequenceGuardInput,
    check_creative_cap_reasons,
    check_photo_exemption,
    detect_aeb_sequence_photo_ids,
    detect_icm_sequence_photo_ids,
    is_slower_than_handheld_rule,
)
from lightbox.core.metrics import ImageMetrics, infer_motion_state
from lightbox.i18n import (
    REASON_BORDERLINE_BLUR,
    REASON_EXTREME_EXPOSURE,
    REASON_FACE_OUT_OF_FOCUS,
    REASON_LOW_RELATIVE_SHARPNESS,
    REASON_MOTION_BLUR,
    REASON_OUT_OF_FOCUS,
    REASON_PANNING_PROTECTED,
    REASON_SEVERE_BLUR,
    REASON_SLOW_SHUTTER_NOTE,
    REASON_SUSPECTED_ICM,
)

SensitivityMode = Literal["conservative", "standard"]


@dataclass(frozen=True, slots=True)
class SensitivityPreset:
    """Calibrated sharpness and relative-gating thresholds for a sensitivity mode."""

    name: SensitivityMode
    abs_red_sharp: float
    abs_red_solo_sharp: float  # Stricter absolute floor when cohort has < 4 peers
    abs_yellow_sharp: float
    rel_z_red: float
    rel_q75_red_ratio: float
    rel_z_yellow: float
    motion_aniso_blur_thresh: float


SENSITIVITY_PRESETS: dict[SensitivityMode, SensitivityPreset] = {
    "conservative": SensitivityPreset(
        name="conservative",
        abs_red_sharp=7.5,
        abs_red_solo_sharp=5.5,
        abs_yellow_sharp=12.5,
        rel_z_red=-1.25,
        rel_q75_red_ratio=0.40,
        rel_z_yellow=-2.0,
        motion_aniso_blur_thresh=0.45,
    ),
    "standard": SensitivityPreset(
        name="standard",
        abs_red_sharp=9.5,
        abs_red_solo_sharp=7.0,
        abs_yellow_sharp=14.5,
        rel_z_red=-1.0,
        rel_q75_red_ratio=0.48,
        rel_z_yellow=-1.75,
        motion_aniso_blur_thresh=0.42,
    ),
}


@dataclass(frozen=True, slots=True)
class PhotoEvaluationInput:
    """Per-photo inputs fed into session-level calibration and verdict evaluation."""

    photo_id: str
    content_fp: str
    metrics: ImageMetrics
    dhash: int
    exif: NormalizedExif | None = None
    faces: tuple[DetectedFace, ...] = ()
    file_path: Path | None = None
    burst_frame_motion: float | None = None
    scene_tag: str | None = None


@dataclass(frozen=True, slots=True)
class PhotoVerdictOutput:
    """Deterministic engine verdict and calibrated metrics for one photo."""

    photo_id: str
    bucket: Literal["red", "yellow", "normal", "exempt"]
    confidence: float
    reason_codes: tuple[str, ...]
    sharp_z: float | None
    motion_state: str
    is_protected: bool
    exempt_reason: str | None
    face_count: int
    face_sharp_min: float | None
    detail: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _CohortStats:
    size: int
    median: float
    robust_sigma: float
    q75: float


def _calibration_cohort_key(exif: NormalizedExif | None) -> tuple[str, str, str]:
    """Calibration cohort `(camera_model, lens_model, cohort)` strictly isolating phone vs camera."""
    if exif is None:
        return ("", "", "camera")
    return (exif.camera_model or "", exif.lens_model or "", exif.cohort)


def _camera_sequence_key(exif: NormalizedExif | None) -> str:
    if exif is None:
        return "unknown::camera"
    if exif.camera_serial:
        return f"serial::{exif.camera_serial}"
    return f"model::{exif.camera_model or 'unknown'}::{exif.cohort}"


def _compute_cohort_stats(values: Sequence[float]) -> _CohortStats:
    if not values:
        return _CohortStats(size=0, median=0.0, robust_sigma=1.0, q75=0.0)
    arr = np.asarray(values, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    robust_sigma = max(1.4826 * mad, 1.0)
    q75 = float(np.percentile(arr, 75))
    return _CohortStats(size=len(values), median=med, robust_sigma=robust_sigma, q75=q75)


def evaluate_session_verdicts(
    items: Sequence[PhotoEvaluationInput],
    *,
    sensitivity: SensitivityMode = "conservative",
) -> list[PhotoVerdictOutput]:
    """Run creative guards, cohort calibration, dual-threshold red gating, and verdict generation."""
    preset = SENSITIVITY_PRESETS[sensitivity]

    # Step 1: Detect session-level AEB bracket sequences
    seq_inputs_aeb = [
        SequenceGuardInput(
            photo_id=it.photo_id,
            capture_time=it.exif.capture_time if it.exif else None,
            camera_key=_camera_sequence_key(it.exif),
            dhash=it.dhash,
            shutter_s=it.exif.shutter_s if it.exif else None,
            ev_comp=it.exif.ev_comp if it.exif else None,
            ev100=it.exif.ev100 if it.exif else None,
        )
        for it in items
    ]
    aeb_ids = detect_aeb_sequence_photo_ids(seq_inputs_aeb)

    # Step 2: Single-photo hard exemptions
    exemptions = {
        it.photo_id: check_photo_exemption(
            it.exif,
            file_path=it.file_path,
            is_aeb_sequence_member=(it.photo_id in aeb_ids),
        )
        for it in items
    }

    # Step 3: Cohort-relative sharpness statistics (isolating phone vs camera)
    # Exclude extreme black/white mis-shots and exempt photos from cohort baseline
    cohort_samples: dict[tuple[str, str, str], list[float]] = {}
    for it in items:
        if exemptions[it.photo_id].is_exempt or it.metrics.is_extreme_exposure:
            continue
        key = _calibration_cohort_key(it.exif)
        cohort_samples.setdefault(key, []).append(it.metrics.sharp_p95)

    cohort_stats = {k: _compute_cohort_stats(v) for k, v in cohort_samples.items()}

    # Step 4: Preliminary per-photo softness classification (for ICM run detection)
    prelim_soft: dict[str, bool] = {}
    sharp_z_map: dict[str, float | None] = {}
    for it in items:
        key = _calibration_cohort_key(it.exif)
        stats = cohort_stats.get(key)
        if stats is not None and stats.size >= 2:
            z = (it.metrics.sharp_p95 - stats.median) / stats.robust_sigma
            sharp_z_map[it.photo_id] = round(z, 4)
        else:
            sharp_z_map[it.photo_id] = None

        is_soft = (
            not exemptions[it.photo_id].is_exempt
            and not it.metrics.panning_detected
            and not it.metrics.is_extreme_exposure
            and it.metrics.sharp_p95 < preset.abs_yellow_sharp
        )
        prelim_soft[it.photo_id] = is_soft

    # Step 5: Detect suspected ICM runs (>= 3 consecutive slow-shutter >= 1/15s soft shots)
    seq_inputs_icm = [
        SequenceGuardInput(
            photo_id=it.photo_id,
            capture_time=it.exif.capture_time if it.exif else None,
            camera_key=_camera_sequence_key(it.exif),
            dhash=it.dhash,
            shutter_s=it.exif.shutter_s if it.exif else None,
            ev_comp=it.exif.ev_comp if it.exif else None,
            ev100=it.exif.ev100 if it.exif else None,
            is_soft=prelim_soft[it.photo_id],
        )
        for it in items
    ]
    icm_ids = detect_icm_sequence_photo_ids(seq_inputs_icm)

    # Step 6: Final deterministic verdict per photo
    results: list[PhotoVerdictOutput] = []
    for it in items:
        ex_res = exemptions[it.photo_id]
        sharp_z = sharp_z_map[it.photo_id]
        key = _calibration_cohort_key(it.exif)
        stats = cohort_stats.get(key)

        face_oof, face_sharp_min = evaluate_subject_faces_oof(
            it.faces, sharp_p95=it.metrics.sharp_p95
        )
        motion_state = infer_motion_state(
            panning_detected=it.metrics.panning_detected,
            motion_aniso=it.metrics.motion_aniso,
            sharp_p95=it.metrics.sharp_p95,
            exif=it.exif,
            burst_frame_motion=it.burst_frame_motion,
            scene_tag=it.scene_tag,
        )
        slow_shutter = is_slower_than_handheld_rule(it.exif)

        detail: dict[str, Any] = {
            "sharp_p95": it.metrics.sharp_p95,
            "sharp_z": sharp_z,
            "noise_sigma": it.metrics.noise_sigma,
            "motion_aniso": it.metrics.motion_aniso,
            "motion_state": motion_state,
            "sharpest_cell": list(it.metrics.sharpest_cell),
            "cohort": key[2],
            "sensitivity": sensitivity,
            "slower_than_handheld_rule": slow_shutter,
        }

        # Hard rule 1: Exempt photos never participate in culling
        if ex_res.is_exempt:
            exempt_reason = ex_res.exempt_reason or "in_camera_protected"
            results.append(
                PhotoVerdictOutput(
                    photo_id=it.photo_id,
                    bucket="exempt",
                    confidence=1.0,
                    reason_codes=(exempt_reason,),
                    sharp_z=sharp_z,
                    motion_state=motion_state,
                    is_protected=True,
                    exempt_reason=exempt_reason,
                    face_count=len(it.faces),
                    face_sharp_min=face_sharp_min,
                    detail=detail,
                )
            )
            continue

        creative_caps = check_creative_cap_reasons(it.exif)
        is_icm = it.photo_id in icm_ids

        # Hard rule 2: Extreme black/white mis-shot -> red (unless capped by creative guard!)
        if it.metrics.is_extreme_exposure:
            reasons = [REASON_EXTREME_EXPOSURE]
            if creative_caps:
                reasons.extend(creative_caps)
                bucket: Literal["red", "yellow", "normal", "exempt"] = "yellow"
                confidence = 0.70
            else:
                bucket = "red"
                confidence = 0.98
            results.append(
                PhotoVerdictOutput(
                    photo_id=it.photo_id,
                    bucket=bucket,
                    confidence=confidence,
                    reason_codes=tuple(reasons),
                    sharp_z=sharp_z,
                    motion_state=motion_state,
                    is_protected=False,
                    exempt_reason=None,
                    face_count=len(it.faces),
                    face_sharp_min=face_sharp_min,
                    detail=detail,
                )
            )
            continue

        # Hard rule 3: Panning shot protection -> normal (unless face_oof -> yellow)
        if it.metrics.panning_detected:
            if face_oof:
                bucket = "yellow"
                reasons = [REASON_FACE_OUT_OF_FOCUS, REASON_PANNING_PROTECTED]
                confidence = 0.65
            else:
                bucket = "normal"
                reasons = [REASON_PANNING_PROTECTED]
                confidence = 0.90
            results.append(
                PhotoVerdictOutput(
                    photo_id=it.photo_id,
                    bucket=bucket,
                    confidence=confidence,
                    reason_codes=tuple(reasons),
                    sharp_z=sharp_z,
                    motion_state=motion_state,
                    is_protected=False,
                    exempt_reason=None,
                    face_count=len(it.faces),
                    face_sharp_min=face_sharp_min,
                    detail=detail,
                )
            )
            continue

        # Step 6b: Dual-threshold blur evaluation (absolute failure AND session-relative low)
        sharp = it.metrics.sharp_p95
        abs_fail_red = sharp < preset.abs_red_sharp
        abs_fail_yellow = sharp < preset.abs_yellow_sharp

        rel_low_red = False
        rel_low_yellow = False
        if stats is not None and stats.size >= 4:
            z_val = sharp_z if sharp_z is not None else 0.0
            ratio_low = (
                stats.q75 >= preset.abs_yellow_sharp
                and sharp <= preset.rel_q75_red_ratio * stats.q75
            )
            rel_low_red = (z_val <= preset.rel_z_red) or ratio_low
            rel_low_yellow = z_val <= preset.rel_z_yellow
        else:
            # Small cohort (< 4 peers): require stricter absolute solo threshold to protect FPR
            rel_low_red = sharp < preset.abs_red_solo_sharp

        reasons = []
        if abs_fail_red and rel_low_red:
            blur_reason = (
                REASON_MOTION_BLUR
                if it.metrics.motion_aniso >= preset.motion_aniso_blur_thresh
                else REASON_OUT_OF_FOCUS
            )
            reasons.append(blur_reason)
            if slow_shutter:
                reasons.append(REASON_SLOW_SHUTTER_NOTE)

            # Apply creative intent caps: long exposure (>= 1s), |EV| >= 1.5, or ICM run -> max yellow!
            if creative_caps or is_icm:
                reasons.extend(creative_caps)
                if is_icm:
                    reasons.append(REASON_SUSPECTED_ICM)
                bucket = "yellow"
                confidence = 0.68
            else:
                bucket = "red"
                # Confidence increases the further below threshold
                margin = max(0.0, (preset.abs_red_sharp - sharp) / max(preset.abs_red_sharp, 1.0))
                confidence = round(min(0.97, 0.80 + 0.17 * margin), 4)
        elif abs_fail_yellow or rel_low_yellow or face_oof:
            bucket = "yellow"
            if face_oof:
                reasons.append(REASON_FACE_OUT_OF_FOCUS)
            if abs_fail_yellow:
                if sharp < preset.abs_red_sharp:
                    reasons.append(REASON_SEVERE_BLUR)
                else:
                    reasons.append(REASON_BORDERLINE_BLUR)
            elif rel_low_yellow:
                reasons.append(REASON_LOW_RELATIVE_SHARPNESS)

            if is_icm:
                reasons.append(REASON_SUSPECTED_ICM)
            if creative_caps and abs_fail_yellow:
                reasons.extend(creative_caps)
            if slow_shutter and abs_fail_yellow:
                reasons.append(REASON_SLOW_SHUTTER_NOTE)
            confidence = 0.62
        else:
            bucket = "normal"
            # Note: handheld safe shutter NEVER triggers red or yellow on a sharp image!
            confidence = round(min(0.98, 0.82 + min(0.16, sharp / 250.0)), 4)

        # Deduplicate reasons while preserving order
        deduped_reasons = tuple(dict.fromkeys(reasons))
        results.append(
            PhotoVerdictOutput(
                photo_id=it.photo_id,
                bucket=bucket,
                confidence=confidence,
                reason_codes=deduped_reasons,
                sharp_z=sharp_z,
                motion_state=motion_state,
                is_protected=False,
                exempt_reason=None,
                face_count=len(it.faces),
                face_sharp_min=face_sharp_min,
                detail=detail,
            )
        )

    return results


__all__ = [
    "PhotoEvaluationInput",
    "PhotoVerdictOutput",
    "SENSITIVITY_PRESETS",
    "SensitivityMode",
    "SensitivityPreset",
    "evaluate_session_verdicts",
]
