"""Creative-intent protection matrix & exemption guards (SPEC §4.3)."""

from __future__ import annotations

import os
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from lightbox.core.bursts import dhash_hamming_distance
from lightbox.core.exif import NormalizedExif
from lightbox.i18n import (
    REASON_AEB_BRACKET,
    REASON_EXTREME_EV_COMP_PROTECTED,
    REASON_IN_CAMERA_PROTECTED,
    REASON_IN_CAMERA_RATED,
    REASON_LONG_EXPOSURE_PROTECTED,
)

LONG_EXPOSURE_MIN_SHUTTER_S = 1.0
EXTREME_EV_COMP_MIN_ABS = 1.5
ICM_MIN_SHUTTER_S = 1.0 / 15.0 - 1e-4  # >= 1/15s
ICM_MIN_RUN_LENGTH = 3
AEB_MIN_RUN_LENGTH = 3
AEB_MAX_STEP_DT_S = 5.0
AEB_MAX_DHASH_HAMMING = 14
AEB_MIN_EV_SPAN = 1.0

_INACTIVE_BRACKET_VALUES: frozenset[str] = frozenset({"", "0", "off", "none", "false", "no"})


@dataclass(frozen=True, slots=True)
class ExemptionResult:
    is_exempt: bool
    exempt_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SequenceGuardInput:
    """Minimal per-photo fields needed for session-wide AEB and ICM sequence detection."""

    photo_id: str
    capture_time: float | None
    camera_key: str
    dhash: int
    shutter_s: float | None
    ev_comp: float | None
    ev100: float | None
    is_soft: bool = False


def is_file_immutable_or_readonly(file_path: Path | None) -> bool:
    """Detect OS read-only permissions or macOS `chflags uchg` (`UF_IMMUTABLE`)."""
    if file_path is None:
        return False
    try:
        st = os.stat(file_path)
    except OSError:
        return False

    # User write bit cleared (read-only file)
    if not bool(st.st_mode & stat.S_IWUSR):
        return True

    # macOS / BSD user immutable flag (`chflags uchg`)
    uf_immutable = getattr(stat, "UF_IMMUTABLE", 0x00000002)
    st_flags = getattr(st, "st_flags", 0)
    if st_flags and bool(st_flags & uf_immutable):
        return True
    return False


def check_photo_exemption(
    exif: NormalizedExif | None,
    *,
    file_path: Path | None = None,
    is_aeb_sequence_member: bool = False,
) -> ExemptionResult:
    """Evaluate single-photo hard exemption rules (SPEC §4.3: always `exempt`)."""
    if exif is not None:
        if exif.protected is True:
            return ExemptionResult(is_exempt=True, exempt_reason=REASON_IN_CAMERA_PROTECTED)
        if exif.rating is not None and exif.rating > 0:
            return ExemptionResult(is_exempt=True, exempt_reason=REASON_IN_CAMERA_RATED)
        if exif.bracket_group is not None:
            normalized = exif.bracket_group.strip().lower()
            if normalized not in _INACTIVE_BRACKET_VALUES:
                return ExemptionResult(is_exempt=True, exempt_reason=REASON_AEB_BRACKET)

    if is_file_immutable_or_readonly(file_path):
        return ExemptionResult(is_exempt=True, exempt_reason=REASON_IN_CAMERA_PROTECTED)

    if is_aeb_sequence_member:
        return ExemptionResult(is_exempt=True, exempt_reason=REASON_AEB_BRACKET)

    return ExemptionResult(is_exempt=False, exempt_reason=None)


def check_creative_cap_reasons(exif: NormalizedExif | None) -> list[str]:
    """Return creative protection reasons that cap a photo at `yellow` (NEVER `red`)."""
    if exif is None:
        return []
    reasons: list[str] = []
    if exif.shutter_s is not None and exif.shutter_s >= LONG_EXPOSURE_MIN_SHUTTER_S:
        reasons.append(REASON_LONG_EXPOSURE_PROTECTED)
    if exif.ev_comp is not None and abs(exif.ev_comp) >= EXTREME_EV_COMP_MIN_ABS:
        reasons.append(REASON_EXTREME_EV_COMP_PROTECTED)
    return reasons


def is_slower_than_handheld_rule(exif: NormalizedExif | None) -> bool:
    """Check whether shutter is slower than `1 / focal_eq` (informational only; NEVER triggers `red`)."""
    if exif is None or exif.shutter_s is None or exif.focal_eq is None:
        return False
    if exif.focal_eq <= 0.0 or exif.shutter_s <= 0.0:
        return False
    return exif.shutter_s > (1.0 / exif.focal_eq)


def detect_aeb_sequence_photo_ids(items: Sequence[SequenceGuardInput]) -> set[str]:
    """Detect `>= 3` consecutive same-camera, same-composition frames with stepped exposure (AEB)."""
    by_cam: dict[str, list[SequenceGuardInput]] = {}
    for item in items:
        if item.capture_time is None:
            continue
        by_cam.setdefault(item.camera_key, []).append(item)

    aeb_ids: set[str] = set()
    for _cam, cam_items in by_cam.items():
        if len(cam_items) < AEB_MIN_RUN_LENGTH:
            continue
        ordered = sorted(cam_items, key=lambda x: (x.capture_time or 0.0, x.photo_id))

        # Partition into rapid similar-composition runs
        runs: list[list[SequenceGuardInput]] = [[ordered[0]]]
        for item in ordered[1:]:
            prev = runs[-1][-1]
            assert prev.capture_time is not None and item.capture_time is not None
            dt = item.capture_time - prev.capture_time
            hamming = dhash_hamming_distance(prev.dhash, item.dhash)
            if 0.0 <= dt <= AEB_MAX_STEP_DT_S and hamming <= AEB_MAX_DHASH_HAMMING:
                runs[-1].append(item)
            else:
                runs.append([item])

        for run in runs:
            if len(run) < AEB_MIN_RUN_LENGTH:
                continue
            # Prefer explicit ev_comp; fall back to computed ev100
            ev_comps = [round(x.ev_comp, 2) for x in run if x.ev_comp is not None]
            if len(ev_comps) >= AEB_MIN_RUN_LENGTH:
                unique_ev = set(ev_comps)
                if len(unique_ev) >= AEB_MIN_RUN_LENGTH and (
                    max(ev_comps) - min(ev_comps) >= AEB_MIN_EV_SPAN
                ):
                    aeb_ids.update(x.photo_id for x in run)
                    continue

            ev100s = [round(x.ev100, 2) for x in run if x.ev100 is not None]
            if len(ev100s) >= AEB_MIN_RUN_LENGTH:
                unique_ev100 = {round(v * 2.0) / 2.0 for v in ev100s}
                if len(unique_ev100) >= AEB_MIN_RUN_LENGTH and (
                    max(ev100s) - min(ev100s) >= AEB_MIN_EV_SPAN
                ):
                    aeb_ids.update(x.photo_id for x in run)

    return aeb_ids


def detect_icm_sequence_photo_ids(items: Sequence[SequenceGuardInput]) -> set[str]:
    """Detect `>= 3` consecutive slow-shutter (`>= 1/15s`) overall-soft frames (suspected ICM)."""
    by_cam: dict[str, list[SequenceGuardInput]] = {}
    for item in items:
        by_cam.setdefault(item.camera_key, []).append(item)

    icm_ids: set[str] = set()
    for _cam, cam_items in by_cam.items():
        if len(cam_items) < ICM_MIN_RUN_LENGTH:
            continue
        ordered = sorted(
            cam_items,
            key=lambda x: (x.capture_time if x.capture_time is not None else 0.0, x.photo_id),
        )

        current_run: list[SequenceGuardInput] = []
        for item in ordered:
            is_slow_and_soft = (
                item.is_soft
                and item.shutter_s is not None
                and item.shutter_s >= ICM_MIN_SHUTTER_S
            )
            if is_slow_and_soft:
                current_run.append(item)
            else:
                if len(current_run) >= ICM_MIN_RUN_LENGTH:
                    icm_ids.update(x.photo_id for x in current_run)
                current_run = []
        if len(current_run) >= ICM_MIN_RUN_LENGTH:
            icm_ids.update(x.photo_id for x in current_run)

    return icm_ids


__all__ = [
    "AEB_MAX_DHASH_HAMMING",
    "AEB_MAX_STEP_DT_S",
    "AEB_MIN_EV_SPAN",
    "AEB_MIN_RUN_LENGTH",
    "EXTREME_EV_COMP_MIN_ABS",
    "ExemptionResult",
    "ICM_MIN_RUN_LENGTH",
    "ICM_MIN_SHUTTER_S",
    "LONG_EXPOSURE_MIN_SHUTTER_S",
    "SequenceGuardInput",
    "check_creative_cap_reasons",
    "check_photo_exemption",
    "detect_aeb_sequence_photo_ids",
    "detect_icm_sequence_photo_ids",
    "is_file_immutable_or_readonly",
    "is_slower_than_handheld_rule",
]
