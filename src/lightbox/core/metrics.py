"""Noise-normalized 16x16 sharpness, directional motion anisotropy, and extreme exposure detection (SPEC §4.2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from lightbox.core.detector import ENGINE_VERSION, PhotoContext
from lightbox.core.exif import NormalizedExif

GRID_ROWS = 16
GRID_COLS = 16

# Immerkær (1996) 3x3 fast noise estimation Laplacian kernel (L2 norm = 6.0)
_IMMERKAER_KERNEL: NDArray[np.float32] = np.array(
    [[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
    dtype=np.float32,
)
_IMMERKAER_MAD_SCALE = 0.67448975 * 6.0  # ~4.04694

# Extreme exposure thresholds (SPEC §4.2)
EXTREME_DARK_MAX_VAL = 8
EXTREME_BRIGHT_MIN_VAL = 248
EXTREME_FRAC_THRESHOLD = 0.985
EXTREME_ENTROPY_THRESHOLD = 1.0
EXTREME_GRAD_ENERGY_THRESHOLD = 25.0

# Grid texture & panning thresholds
FLAT_CELL_STD_THRESHOLD = 4.0
PANNING_SUBJECT_SHARP_MIN = 14.0
PANNING_BG_COHERENCY_MIN = 0.42
PANNING_BG_RATIO_MAX = 0.50
PANNING_SUBJ_COHERENCY_MAX = 0.60


@dataclass(frozen=True, slots=True)
class ImageMetrics:
    """Deterministic single-image quality and motion indicators."""

    sharp_p95: float
    noise_sigma: float
    motion_aniso: float
    motion_state: str  # 'static' | 'moving' | 'panning' | 'unknown'
    texture: float
    exposure: dict[str, Any]
    sharpest_cell: tuple[int, int]
    panning_detected: bool
    is_extreme_exposure: bool
    extreme_kind: str | None  # 'underexposed' | 'overexposed' | None
    cell_sharpness_p50: float


def compute_region_sharpness(
    gray_region: NDArray[np.uint8],
    *,
    noise_sigma: float = 0.0,
) -> float:
    """Compute noise-normalized Tenengrad + Laplacian sharpness for a grayscale ROI."""
    if gray_region.size < 16 or gray_region.shape[0] < 3 or gray_region.shape[1] < 3:
        return 0.0
    arr = gray_region.astype(np.float32, copy=False)
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    g2 = gx * gx + gy * gy
    lap = cv2.Laplacian(arr, cv2.CV_32F, ksize=1)

    tenengrad = float(np.mean(g2))
    lap_var = float(np.var(lap))
    return _normalize_cell_sharpness(tenengrad, lap_var, noise_sigma)


def _normalize_cell_sharpness(tenengrad: float, lap_var: float, noise_sigma: float) -> float:
    """Subtract expected Gaussian noise floor and apply SNR normalization."""
    s2 = noise_sigma * noise_sigma
    # Expected noise power: normalized Sobel (1/8) -> 24/64 = 0.375*sigma^2;
    # 4-neighbor Laplacian -> 20*sigma^2. Slight safety margin suppresses high-ISO blur noise.
    net_ten = max(0.0, tenengrad - 0.42 * s2)
    net_lap = max(0.0, lap_var - 22.0 * s2)
    snr_denom = math.sqrt(1.0 + (noise_sigma / 16.0) ** 2)
    return float((0.40 * math.sqrt(net_ten) + 0.60 * math.sqrt(net_lap)) / snr_denom)


def _shannon_entropy(gray: NDArray[np.uint8]) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = float(hist.sum())
    if total <= 0.0:
        return 0.0
    probs = hist[hist > 0.0] / total
    return float(-np.sum(probs * np.log2(probs)))


def _structure_tensor_coherency(jxx: float, jyy: float, jxy: float) -> float:
    denom = jxx + jyy
    if denom <= 1e-6:
        return 0.0
    diff = jxx - jyy
    return float(math.sqrt(diff * diff + 4.0 * jxy * jxy) / denom)


def infer_motion_state(
    *,
    panning_detected: bool,
    motion_aniso: float,
    sharp_p95: float,
    exif: NormalizedExif | None = None,
    burst_frame_motion: float | None = None,
    scene_tag: str | None = None,
) -> str:
    """Infer subject/camera motion state (`static` | `moving` | `panning` | `unknown`)."""
    if panning_detected:
        return "panning"

    af_mode = exif.af_mode if exif is not None else None
    shutter_s = exif.shutter_s if exif is not None else None

    if burst_frame_motion is not None and burst_frame_motion >= 0.025:
        return "moving"
    if af_mode == "AF-C":
        return "moving"
    if scene_tag and any(
        tok in scene_tag.lower() for tok in ("action", "sports", "running", "moving", "panning")
    ):
        return "moving"
    if motion_aniso >= 0.48 and sharp_p95 < 12.5:
        return "moving"
    if burst_frame_motion is not None and burst_frame_motion < 0.015 and motion_aniso < 0.35:
        return "static"
    if af_mode in {"AF-S", "MF"} and motion_aniso < 0.40:
        return "static"
    if shutter_s is not None and shutter_s >= 0.5 and motion_aniso < 0.35:
        return "static"
    if motion_aniso < 0.30:
        return "static"
    return "unknown"


def compute_image_metrics(
    gray: NDArray[np.uint8],
    *,
    exif: NormalizedExif | None = None,
) -> ImageMetrics:
    """Analyze an 8-bit grayscale image (`analysis` tier) for sharpness, noise, motion, and exposure."""
    if gray.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {gray.shape}")

    h, w = gray.shape
    if h < 8 or w < 8:
        raise ValueError(f"Image too small for 16x16 grid metrics: {w}x{h}")

    gray_f32 = gray.astype(np.float32)

    # Full-frame gradient (normalized by Sobel gain 8.0) and Laplacian maps
    gx_raw = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
    gy_raw = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
    gx = gx_raw / 8.0
    gy = gy_raw / 8.0
    g2 = gx * gx + gy * gy
    g2_raw = gx_raw * gx_raw + gy_raw * gy_raw
    lap = cv2.Laplacian(gray_f32, cv2.CV_32F, ksize=1)
    immerkaer_abs = np.abs(cv2.filter2D(gray_f32, cv2.CV_32F, _IMMERKAER_KERNEL))

    # Extreme exposure check (SPEC §4.2)
    total_px = float(gray.size)
    dark_frac = float(np.count_nonzero(gray <= EXTREME_DARK_MAX_VAL)) / total_px
    bright_frac = float(np.count_nonzero(gray >= EXTREME_BRIGHT_MIN_VAL)) / total_px
    entropy = _shannon_entropy(gray)
    mean_grad_energy = float(np.mean(g2_raw))

    is_under = dark_frac > EXTREME_FRAC_THRESHOLD
    is_over = bright_frac > EXTREME_FRAC_THRESHOLD
    is_extreme = bool(
        (is_under or is_over)
        and entropy < EXTREME_ENTROPY_THRESHOLD
        and mean_grad_energy < EXTREME_GRAD_ENERGY_THRESHOLD
    )
    extreme_kind: str | None = None
    if is_extreme:
        extreme_kind = "underexposed" if is_under else "overexposed"

    # Build 16x16 grid boundaries
    ys = np.linspace(0, h, GRID_ROWS + 1, dtype=int)
    xs = np.linspace(0, w, GRID_COLS + 1, dtype=int)

    cell_records: list[tuple[int, int, float, float, float, float, float, float, float, float]] = []
    # (r, c, mean, std, sigma_est, tenengrad, lap_var, jxx, jyy, jxy)
    for r in range(GRID_ROWS):
        y0, y1 = int(ys[r]), int(ys[r + 1])
        if y1 <= y0:
            continue
        for c in range(GRID_COLS):
            x0, x1 = int(xs[c]), int(xs[c + 1])
            if x1 <= x0:
                continue
            cell_px = gray_f32[y0:y1, x0:x1]
            c_mean = float(np.mean(cell_px))
            c_std = float(np.std(cell_px))
            c_mad = float(np.median(immerkaer_abs[y0:y1, x0:x1]))
            c_sigma = c_mad / _IMMERKAER_MAD_SCALE

            c_g2 = g2[y0:y1, x0:x1]
            c_ten = float(np.mean(c_g2))
            c_lap_var = float(np.var(lap[y0:y1, x0:x1]))

            c_gx = gx[y0:y1, x0:x1]
            c_gy = gy[y0:y1, x0:x1]
            c_jxx = float(np.mean(c_gx * c_gx))
            c_jyy = float(np.mean(c_gy * c_gy))
            c_jxy = float(np.mean(c_gx * c_gy))

            cell_records.append(
                (r, c, c_mean, c_std, c_sigma, c_ten, c_lap_var, c_jxx, c_jyy, c_jxy)
            )

    # Estimate image noise sigma from the flattest non-clipped cells (bottom 30% by std)
    non_clipped = [rec for rec in cell_records if 2.0 < rec[2] < 253.0]
    if non_clipped:
        non_clipped_sorted = sorted(non_clipped, key=lambda rec: rec[3])
        flat_count = max(1, int(math.ceil(len(non_clipped_sorted) * 0.30)))
        flat_sigmas = [rec[4] for rec in non_clipped_sorted[:flat_count]]
        noise_sigma = max(0.0, float(np.median(flat_sigmas)))
    else:
        noise_sigma = 0.0

    # Compute noise-normalized sharpness for every cell; skip very flat cells for P95
    all_cell_sharps: list[tuple[int, int, float, float, float, float, float]] = []
    # (r, c, sharp, std, jxx, jyy, jxy)
    valid_sharps: list[float] = []
    best_cell = (GRID_ROWS // 2, GRID_COLS // 2)
    best_sharp = -1.0

    for r, c, _c_mean, c_std, _c_sigma, c_ten, c_lap_var, c_jxx, c_jyy, c_jxy in cell_records:
        sharp = _normalize_cell_sharpness(c_ten, c_lap_var, noise_sigma)
        all_cell_sharps.append((r, c, sharp, c_std, c_jxx, c_jyy, c_jxy))
        if sharp > best_sharp:
            best_sharp = sharp
            best_cell = (r, c)
        if c_std >= FLAT_CELL_STD_THRESHOLD:
            valid_sharps.append(sharp)

    if valid_sharps:
        sharp_p95 = float(np.percentile(valid_sharps, 95))
        sharp_p50 = float(np.percentile(valid_sharps, 50))
    elif all_cell_sharps:
        raw_list = [row[2] for row in all_cell_sharps]
        sharp_p95 = float(np.percentile(raw_list, 95))
        sharp_p50 = float(np.percentile(raw_list, 50))
    else:
        sharp_p95 = 0.0
        sharp_p50 = 0.0

    texture = float(np.mean([rec[3] for rec in cell_records])) if cell_records else 0.0

    # High-gradient cell directional anisotropy (global structure tensor coherency)
    grad_energies = [row[4] + row[5] for row in all_cell_sharps]
    energy_thresh = max(0.6, float(np.percentile(grad_energies, 60))) if grad_energies else 0.6
    high_grad_cells = [row for row in all_cell_sharps if (row[4] + row[5]) >= energy_thresh]

    sum_jxx = 0.0
    sum_jyy = 0.0
    sum_jxy = 0.0
    for _r, _c, _sh, _std, jxx, jyy, jxy in high_grad_cells:
        norm = jxx + jyy + 1e-6
        sum_jxx += jxx / norm
        sum_jyy += jyy / norm
        sum_jxy += jxy / norm
    motion_aniso = _structure_tensor_coherency(sum_jxx, sum_jyy, sum_jxy)

    # Panning detection (SPEC §4.3): sharp multi-directional subject + directional blurred background
    panning_detected = False
    bg_coherency = 0.0
    if sharp_p95 >= PANNING_SUBJECT_SHARP_MIN and all_cell_sharps:
        subj_thresh = sharp_p95 * 0.65
        subj_cells = [row for row in all_cell_sharps if row[2] >= subj_thresh]
        bg_cells = [
            row
            for row in all_cell_sharps
            if row[2] < sharp_p95 * PANNING_BG_RATIO_MAX and row[3] >= FLAT_CELL_STD_THRESHOLD
        ]
        subj_frac = len(subj_cells) / float(len(all_cell_sharps))
        if 0.02 <= subj_frac <= 0.50 and len(bg_cells) >= 24:
            subj_jxx = 0.0
            subj_jyy = 0.0
            subj_jxy = 0.0
            for _r, _c, _sh, _std, jxx, jyy, jxy in subj_cells:
                norm = jxx + jyy + 1e-6
                subj_jxx += jxx / norm
                subj_jyy += jyy / norm
                subj_jxy += jxy / norm
            subj_coherency = _structure_tensor_coherency(subj_jxx, subj_jyy, subj_jxy)

            bg_jxx = 0.0
            bg_jyy = 0.0
            bg_jxy = 0.0
            for _r, _c, _sh, _std, jxx, jyy, jxy in bg_cells:
                norm = jxx + jyy + 1e-6
                bg_jxx += jxx / norm
                bg_jyy += jyy / norm
                bg_jxy += jxy / norm
            bg_coherency = _structure_tensor_coherency(bg_jxx, bg_jyy, bg_jxy)
            bg_med_sharp = float(np.median([row[2] for row in bg_cells]))
            if (
                subj_coherency <= PANNING_SUBJ_COHERENCY_MAX
                and bg_coherency >= PANNING_BG_COHERENCY_MIN
                and bg_med_sharp < sharp_p95 * PANNING_BG_RATIO_MAX
            ):
                panning_detected = True

    motion_state = infer_motion_state(
        panning_detected=panning_detected,
        motion_aniso=motion_aniso,
        sharp_p95=sharp_p95,
        exif=exif,
    )

    exposure_info: dict[str, Any] = {
        "dark_frac": round(dark_frac, 5),
        "bright_frac": round(bright_frac, 5),
        "entropy": round(entropy, 4),
        "mean_grad_energy": round(mean_grad_energy, 3),
        "is_extreme_exposure": is_extreme,
        "extreme_kind": extreme_kind,
        "sharpest_cell": [best_cell[0], best_cell[1]],
        "panning_detected": panning_detected,
        "bg_coherency": round(bg_coherency, 4),
    }

    return ImageMetrics(
        sharp_p95=round(sharp_p95, 4),
        noise_sigma=round(noise_sigma, 4),
        motion_aniso=round(motion_aniso, 4),
        motion_state=motion_state,
        texture=round(texture, 4),
        exposure=exposure_info,
        sharpest_cell=best_cell,
        panning_detected=panning_detected,
        is_extreme_exposure=is_extreme,
        extreme_kind=extreme_kind,
        cell_sharpness_p50=round(sharp_p50, 4),
    )


class MetricsDetector:
    """`Detector` implementation computing 16x16 sharpness, noise, motion, and exposure metrics."""

    name: str = "metrics"
    version: str = ENGINE_VERSION
    requires: set[str] = {"analysis_img"}

    def run(self, ctx: PhotoContext) -> dict[str, Any]:
        if ctx.analysis_img is None:
            raise ValueError(f"PhotoContext for {ctx.photo_id} missing `analysis_img`")
        result = compute_image_metrics(ctx.analysis_img, exif=ctx.exif)
        return {
            "sharp_p95": result.sharp_p95,
            "noise_sigma": result.noise_sigma,
            "motion_aniso": result.motion_aniso,
            "motion_state": result.motion_state,
            "texture": result.texture,
            "exposure": result.exposure,
            "sharpest_cell": result.sharpest_cell,
            "panning_detected": result.panning_detected,
            "is_extreme_exposure": result.is_extreme_exposure,
            "extreme_kind": result.extreme_kind,
            "cell_sharpness_p50": result.cell_sharpness_p50,
        }


__all__ = [
    "EXTREME_BRIGHT_MIN_VAL",
    "EXTREME_DARK_MAX_VAL",
    "EXTREME_ENTROPY_THRESHOLD",
    "EXTREME_FRAC_THRESHOLD",
    "EXTREME_GRAD_ENERGY_THRESHOLD",
    "GRID_COLS",
    "GRID_ROWS",
    "ImageMetrics",
    "MetricsDetector",
    "compute_image_metrics",
    "compute_region_sharpness",
    "infer_motion_state",
]
