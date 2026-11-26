"""Cross-subdirectory burst clustering, dHash similarity, and inter-frame motion estimation (SPEC §4.2)."""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

BURST_MAX_DT_CAP_S = 2.5
BURST_DHASH_MAX_HAMMING = 12
BURST_EMBED_MIN_COSINE = 0.85
BURST_FALLBACK_ZERO_DT_S = 1.5
_MOTION_THUMB_EDGE = 128


@dataclass(frozen=True, slots=True)
class BurstCandidate:
    """Per-photo inputs required for cross-subdirectory burst clustering and ranking."""

    photo_id: str
    capture_time: float | None
    camera_serial: str | None
    camera_model: str | None
    cohort: str
    dhash: int
    sharp_p95: float
    noise_sigma: float = 0.0
    face_sharp_min: float | None = None
    is_extreme_exposure: bool = False
    is_exempt: bool = False
    motion_thumb: NDArray[np.float32] | None = None
    embedding: NDArray[np.float32] | None = None


@dataclass(frozen=True, slots=True)
class BurstMemberResult:
    photo_id: str
    rank: int
    is_pick: bool
    frame_motion: float


@dataclass(frozen=True, slots=True)
class BurstGroupResult:
    burst_id: str
    kind: str  # 'burst' | 'reshoot'
    members: tuple[BurstMemberResult, ...]
    mean_frame_motion: float


def compute_dhash(gray: NDArray[np.uint8]) -> int:
    """Compute a 64-bit difference hash (dHash) over a 9x8 downscaled grayscale image."""
    if gray.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image for dHash, got shape {gray.shape}")
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = diff.ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def dhash_hamming_distance(hash_a: int, hash_b: int) -> int:
    """Return Hamming distance (0..64) between two 64-bit dHash integers."""
    return (hash_a ^ hash_b).bit_count()


def make_motion_thumb(gray: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Create a lightweight 128px-long-edge float32 thumbnail for inter-frame motion estimation."""
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest <= _MOTION_THUMB_EDGE:
        return gray.astype(np.float32)
    scale = _MOTION_THUMB_EDGE / float(longest)
    new_w = max(8, int(round(w * scale)))
    new_h = max(8, int(round(h * scale)))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def estimate_interframe_motion(
    thumb_a: NDArray[np.float32] | None,
    thumb_b: NDArray[np.float32] | None,
) -> float:
    """Estimate lightweight normalized inter-frame shift / subject motion in `[0, 1]`."""
    if thumb_a is None or thumb_b is None:
        return 0.0
    if thumb_a.shape != thumb_b.shape:
        h = min(thumb_a.shape[0], thumb_b.shape[0])
        w = min(thumb_a.shape[1], thumb_b.shape[1])
        if h < 8 or w < 8:
            return 0.0
        a = cv2.resize(thumb_a, (w, h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(thumb_b, (w, h), interpolation=cv2.INTER_AREA)
    else:
        a = thumb_a
        b = thumb_b

    h, w = a.shape[:2]
    if h < 8 or w < 8:
        return 0.0

    # Global translation via phase correlation (normalized by diagonal)
    try:
        (dx, dy), response = cv2.phaseCorrelate(a, b)
        diag = float(np.hypot(w, h))
        shift_norm = float(np.hypot(dx, dy)) / max(diag, 1.0)
        if response < 0.1:
            shift_norm = 0.0
    except Exception:
        shift_norm = 0.0

    # Local subject motion proxy: mean normalized absolute pixel difference
    mad_norm = float(np.mean(np.abs(a - b))) / 255.0
    return round(min(1.0, max(shift_norm, mad_norm * 0.6)), 5)


def _cosine_similarity(vec_a: NDArray[np.float32], vec_b: NDArray[np.float32]) -> float:
    na = float(np.linalg.norm(vec_a))
    nb = float(np.linalg.norm(vec_b))
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (na * nb))


def _camera_grouping_key(candidate: BurstCandidate) -> tuple[str, str]:
    """Group bursts by camera serial across subdirectories; fallback to (model, cohort)."""
    if candidate.camera_serial:
        return ("serial", candidate.camera_serial)
    return ("model", f"{candidate.camera_model or 'unknown'}::{candidate.cohort}")


def _compute_adaptive_max_dt(sorted_candidates: Sequence[BurstCandidate]) -> float:
    """Compute adaptive burst gap cap `<= min(2.5s, 3 * median_dt)` (SPEC §4.2)."""
    close_dts: list[float] = []
    for prev, curr in zip(sorted_candidates, sorted_candidates[1:], strict=False):
        if prev.capture_time is None or curr.capture_time is None:
            continue
        dt = curr.capture_time - prev.capture_time
        if 0.0 <= dt <= BURST_MAX_DT_CAP_S:
            close_dts.append(dt)

    if not close_dts:
        return BURST_MAX_DT_CAP_S

    median_dt = float(statistics.median(close_dts))
    # Integer-second EXIF timestamps often yield median_dt == 0.0 for same-second bursts
    if median_dt < 0.05:
        return min(BURST_MAX_DT_CAP_S, BURST_FALLBACK_ZERO_DT_S)
    return min(BURST_MAX_DT_CAP_S, max(0.35, 3.0 * median_dt))


def _stable_burst_id(session_id: str, photo_ids: Sequence[str]) -> str:
    payload = f"{session_id}:" + ",".join(sorted(photo_ids))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"burst_{digest}"


def _member_quality_key(c: BurstCandidate) -> tuple[int, float, float, float, str]:
    """Higher tuple sorts better (non-extreme > sharp face > sharp_p95 > lower noise)."""
    not_extreme = 0 if c.is_extreme_exposure else 1
    face_score = c.face_sharp_min if c.face_sharp_min is not None else -1.0
    return (not_extreme, face_score, c.sharp_p95, -c.noise_sigma, c.photo_id)


def detect_bursts(
    session_id: str,
    candidates: Sequence[BurstCandidate],
    *,
    max_dhash_hamming: int = BURST_DHASH_MAX_HAMMING,
    min_embed_cosine: float = BURST_EMBED_MIN_COSINE,
) -> tuple[list[BurstGroupResult], dict[str, float]]:
    """Cluster same-camera rapid bursts across subdirectories and return inter-frame motion map.

    Returns `(burst_groups, frame_motion_by_photo_id)`.
    """
    by_camera: dict[tuple[str, str], list[BurstCandidate]] = {}
    for cand in candidates:
        if cand.capture_time is None:
            continue
        by_camera.setdefault(_camera_grouping_key(cand), []).append(cand)

    groups: list[BurstGroupResult] = []
    motion_by_photo: dict[str, float] = {}

    for _cam_key, cam_items in sorted(by_camera.items()):
        if len(cam_items) < 2:
            continue
        ordered = sorted(cam_items, key=lambda c: (c.capture_time or 0.0, c.photo_id))
        max_dt = _compute_adaptive_max_dt(ordered)

        raw_clusters: list[list[BurstCandidate]] = [[ordered[0]]]
        for cand in ordered[1:]:
            prev = raw_clusters[-1][-1]
            assert prev.capture_time is not None and cand.capture_time is not None
            dt = cand.capture_time - prev.capture_time
            hamming = dhash_hamming_distance(prev.dhash, cand.dhash)

            same_burst = 0.0 <= dt <= max_dt and hamming <= max_dhash_hamming
            if same_burst and prev.embedding is not None and cand.embedding is not None:
                if _cosine_similarity(prev.embedding, cand.embedding) < min_embed_cosine:
                    same_burst = False

            if same_burst:
                raw_clusters[-1].append(cand)
            else:
                raw_clusters.append([cand])

        for cluster in raw_clusters:
            if len(cluster) < 2:
                continue

            # Compute lightweight inter-frame motion across consecutive burst frames
            pair_motions: list[float] = []
            per_photo_motion: dict[str, list[float]] = {c.photo_id: [] for c in cluster}
            for prev, curr in zip(cluster, cluster[1:], strict=False):
                m = estimate_interframe_motion(prev.motion_thumb, curr.motion_thumb)
                pair_motions.append(m)
                per_photo_motion[prev.photo_id].append(m)
                per_photo_motion[curr.photo_id].append(m)

            mean_motion = round(float(np.mean(pair_motions)), 5) if pair_motions else 0.0

            ranked = sorted(cluster, key=_member_quality_key, reverse=True)
            member_results: list[BurstMemberResult] = []
            for rank_idx, member in enumerate(ranked, start=1):
                vals = per_photo_motion[member.photo_id]
                photo_motion = round(max(vals), 5) if vals else 0.0
                motion_by_photo[member.photo_id] = max(
                    motion_by_photo.get(member.photo_id, 0.0), photo_motion
                )
                member_results.append(
                    BurstMemberResult(
                        photo_id=member.photo_id,
                        rank=rank_idx,
                        is_pick=(rank_idx == 1),
                        frame_motion=photo_motion,
                    )
                )

            burst_id = _stable_burst_id(session_id, [c.photo_id for c in cluster])
            groups.append(
                BurstGroupResult(
                    burst_id=burst_id,
                    kind="burst",
                    members=tuple(member_results),
                    mean_frame_motion=mean_motion,
                )
            )

    groups.sort(key=lambda g: g.burst_id)
    return groups, motion_by_photo


def burst_lookup_by_photo(groups: Sequence[BurstGroupResult]) -> Mapping[str, BurstGroupResult]:
    """Map each `photo_id` belonging to a burst to its `BurstGroupResult`."""
    mapping: dict[str, BurstGroupResult] = {}
    for group in groups:
        for member in group.members:
            mapping[member.photo_id] = group
    return mapping


__all__ = [
    "BURST_DHASH_MAX_HAMMING",
    "BURST_EMBED_MIN_COSINE",
    "BURST_MAX_DT_CAP_S",
    "BurstCandidate",
    "BurstGroupResult",
    "BurstMemberResult",
    "burst_lookup_by_photo",
    "compute_dhash",
    "detect_bursts",
    "dhash_hamming_distance",
    "estimate_interframe_motion",
    "make_motion_thumb",
]
