"""Procedural CC0-equivalent synthetic dataset generator & E1 engine evaluation suite (SPEC §11.1–11.2)."""

from __future__ import annotations

import json
import math
import stat
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from lightbox.core import SensitivityMode
from lightbox.services.context import LightboxServices

GroundTruthLabel = Literal["keep", "protected_non_red", "exempt", "obvious_reject"]

E1_MAX_RED_FPR = 0.01  # < 1% false positive rate on good/protected photos
E1_MIN_REJECT_RECALL = 0.75  # >= 75% recall on obvious rejects
E1_MAX_EXEMPT_MISLABEL = 0.0  # == 0% exempt mis-label rate


@dataclass(frozen=True, slots=True)
class SynthSampleSpec:
    filename: str
    label: GroundTruthLabel
    category: str
    rel_path: str


@dataclass(frozen=True, slots=True)
class E1EvalReport:
    sensitivity: SensitivityMode
    total_samples: int
    keep_and_protected_count: int
    false_red_count: int
    red_fpr: float
    obvious_reject_count: int
    true_red_count: int
    obvious_reject_recall: float
    exempt_count: int
    exempt_mislabel_count: int
    exempt_mislabel_rate: float
    passed: bool
    elapsed_s: float
    failures: tuple[str, ...]
    category_breakdown: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": "E1",
            "sensitivity": self.sensitivity,
            "total_samples": self.total_samples,
            "keep_and_protected_count": self.keep_and_protected_count,
            "false_red_count": self.false_red_count,
            "red_fpr": round(self.red_fpr, 5),
            "obvious_reject_count": self.obvious_reject_count,
            "true_red_count": self.true_red_count,
            "obvious_reject_recall": round(self.obvious_reject_recall, 5),
            "exempt_count": self.exempt_count,
            "exempt_mislabel_count": self.exempt_mislabel_count,
            "exempt_mislabel_rate": round(self.exempt_mislabel_rate, 5),
            "gates": {
                "red_fpr_lt": E1_MAX_RED_FPR,
                "obvious_reject_recall_gte": E1_MIN_REJECT_RECALL,
                "exempt_mislabel_eq": E1_MAX_EXEMPT_MISLABEL,
            },
            "passed": self.passed,
            "elapsed_s": self.elapsed_s,
            "failures": list(self.failures),
            "category_breakdown": self.category_breakdown,
        }


@dataclass(frozen=True, slots=True)
class PersonalEvalReport:
    total_overrides: int
    rescued_from_red: int
    rejected_from_normal: int
    overrides_by_reason: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": "personal",
            "total_overrides": self.total_overrides,
            "rescued_from_red": self.rescued_from_red,
            "rejected_from_normal": self.rejected_from_normal,
            "overrides_by_reason": self.overrides_by_reason,
        }


def _make_procedural_rgb_scene(
    width: int = 640,
    height: int = 480,
    *,
    seed: int = 1,
    style: str = "architecture",
) -> NDArray[np.uint8]:
    """Generate a deterministic, crisp RGB photographic-like synthetic scene."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:height, :width].astype(np.float32)

    # Smooth sky-to-ground gradient backdrop
    sky_r = 90.0 + 70.0 * (1.0 - yy / float(height))
    sky_g = 110.0 + 60.0 * (1.0 - yy / float(height))
    sky_b = 140.0 + 75.0 * (1.0 - yy / float(height))
    img = np.stack([sky_r, sky_g, sky_b], axis=-1)

    if style == "architecture":
        # Crisp multi-orientation building silhouettes, windows, and diagonal roofs
        for _ in range(28):
            x0 = int(rng.integers(10, width - 80))
            y0 = int(rng.integers(20, height - 80))
            bw = int(rng.integers(28, 95))
            bh = int(rng.integers(35, 110))
            color = tuple(float(c) for c in rng.integers(30, 225, size=3))
            border = tuple(255.0 - c for c in color)
            cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), color, thickness=-1)
            cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), border, thickness=2)
        for _ in range(30):
            pt1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            pt2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
            color = tuple(float(c) for c in rng.integers(15, 240, size=3))
            cv2.line(img, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)
    elif style == "foliage":
        # Natural curved branches, leaves, and high-frequency organic detail
        for _ in range(90):
            cx = int(rng.integers(10, width - 10))
            cy = int(rng.integers(10, height - 10))
            rx = int(rng.integers(6, 26))
            ry = int(rng.integers(4, 18))
            angle = float(rng.uniform(0.0, 180.0))
            color = (
                float(rng.integers(35, 130)),
                float(rng.integers(80, 210)),
                float(rng.integers(30, 120)),
            )
            cv2.ellipse(img, (cx, cy), (rx, ry), angle, 0.0, 360.0, color, thickness=-1)
            cv2.ellipse(
                img,
                (cx, cy),
                (rx, ry),
                angle,
                0.0,
                360.0,
                (color[0] * 0.4, color[1] * 0.4, color[2] * 0.4),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
    else:
        # Soft harmonic landscape with crisp foreground landmarks
        wave = 28.0 * np.sin(xx / 28.0 + seed) * np.cos(yy / 24.0)
        img[:, :, 0] += wave
        img[:, :, 1] += wave * 0.8
        img[:, :, 2] -= wave * 0.5
        for _ in range(35):
            cx = int(rng.integers(20, width - 20))
            cy = int(rng.integers(20, height - 20))
            rad = int(rng.integers(10, 32))
            color = tuple(float(c) for c in rng.integers(25, 235, size=3))
            cv2.circle(img, (cx, cy), rad, color, thickness=2, lineType=cv2.LINE_AA)

    return np.asarray(np.clip(img, 0, 255), dtype=np.uint8)


def _disk_defocus_rgb(img: NDArray[np.uint8], radius: int) -> NDArray[np.uint8]:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    kernel /= float(kernel.sum())
    blurred = cv2.filter2D(img.astype(np.float32), -1, kernel)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def _linear_motion_blur_rgb(
    img: NDArray[np.uint8],
    length: int,
    angle_deg: float,
) -> NDArray[np.uint8]:
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    for t in np.linspace(-center, center, length * 4):
        x = int(round(center + t * dx))
        y = int(round(center + t * dy))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    kernel /= max(float(kernel.sum()), 1.0)
    blurred = cv2.filter2D(img.astype(np.float32), -1, kernel)
    # Slight optical softening mimics real handshake PSF
    blurred = cv2.GaussianBlur(blurred, (3, 3), 0.9)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def _make_shallow_dof_rgb(seed: int, width: int = 640, height: int = 480) -> NDArray[np.uint8]:
    bg = np.asarray(
        cv2.GaussianBlur(
            _make_procedural_rgb_scene(width, height, seed=seed, style="foliage"), (35, 35), 11.0
        ),
        dtype=np.uint8,
    )
    subj = _make_procedural_rgb_scene(width, height, seed=seed + 500, style="architecture")
    out = bg.copy()
    y0, y1 = int(height * 0.28), int(height * 0.72)
    x0, x1 = int(width * 0.30), int(width * 0.70)
    out[y0:y1, x0:x1] = subj[y0:y1, x0:x1]
    return out


def _make_panning_rgb(seed: int, width: int = 640, height: int = 480) -> NDArray[np.uint8]:
    bg = _linear_motion_blur_rgb(
        _make_procedural_rgb_scene(width, height, seed=seed, style="architecture"),
        length=29,
        angle_deg=0.0,
    )
    subj = _make_procedural_rgb_scene(width, height, seed=seed + 700, style="foliage")
    out = bg.copy()
    y0, y1 = int(height * 0.32), int(height * 0.68)
    x0, x1 = int(width * 0.34), int(width * 0.66)
    out[y0:y1, x0:x1] = subj[y0:y1, x0:x1]
    return out


def _save_jpeg_with_exif(
    path: Path,
    rgb: NDArray[np.uint8],
    *,
    capture_dt: datetime,
    make: str = "SONY",
    model: str = "ILCE-7M4",
    lens: str = "FE 35mm F1.4 GM",
    aperture: float = 2.8,
    shutter_s: float = 1 / 500,
    iso: int = 200,
    focal_eq: int = 35,
    ev_comp: float = 0.0,
    exposure_program: int = 3,  # Av
    flash: int = 0,
    readonly: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # If regenerating an existing read-only file, temporarily restore write permission
    if path.exists():
        mode = path.stat().st_mode
        if not bool(mode & stat.S_IWUSR):
            path.chmod(mode | stat.S_IWUSR)

    img = Image.fromarray(rgb, mode="RGB")
    exif = Image.Exif()
    dt_str = capture_dt.strftime("%Y:%m:%d %H:%M:%S")
    exif[0x010F] = make
    exif[0x0110] = model
    exif[0x0112] = 1
    exif[0x0132] = dt_str
    exif[0x9003] = dt_str
    exif[0x829D] = aperture
    exif[0x829A] = shutter_s
    exif[0x8822] = exposure_program
    exif[0x8827] = iso
    exif[0x9204] = ev_comp
    exif[0x9209] = flash
    exif[0xA405] = focal_eq
    exif[0xA434] = lens
    img.save(path, format="JPEG", quality=92, exif=exif.tobytes())

    if readonly:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def build_demo_dataset(target_dir: Path, *, seed: int = 42) -> list[SynthSampleSpec]:
    """Build a lightweight (~20 JPG, <2 MB) deterministic demo shoot for `lightbox demo`."""
    root = Path(target_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    base_dt = datetime(2026, 10, 15, 16, 0, 0, tzinfo=UTC)
    specs: list[SynthSampleSpec] = []
    idx = 0

    def _next_dt(step_s: float = 15.0) -> datetime:
        nonlocal base_dt
        base_dt = base_dt + timedelta(seconds=step_s)
        return base_dt

    # 1. 7 sharp keepers (varied styles + shallow DoF + panning)
    for i in range(5):
        idx += 1
        fname = f"DEMO_{idx:03d}_sharp.jpg"
        style = ("architecture", "foliage", "landscape")[i % 3]
        rgb = _make_procedural_rgb_scene(seed=seed + i, style=style)
        _save_jpeg_with_exif(root / fname, rgb, capture_dt=_next_dt())
        specs.append(SynthSampleSpec(fname, "keep", "sharp", fname))

    idx += 1
    fname = f"DEMO_{idx:03d}_shallow_dof.jpg"
    _save_jpeg_with_exif(
        root / fname,
        _make_shallow_dof_rgb(seed=seed + 20),
        capture_dt=_next_dt(),
        aperture=1.4,
    )
    specs.append(SynthSampleSpec(fname, "keep", "shallow_dof", fname))

    idx += 1
    fname = f"DEMO_{idx:03d}_panning.jpg"
    _save_jpeg_with_exif(
        root / fname,
        _make_panning_rgb(seed=seed + 30),
        capture_dt=_next_dt(),
        shutter_s=1 / 40,
    )
    specs.append(SynthSampleSpec(fname, "keep", "panning", fname))

    # 2. 4-frame rapid burst (same scene with tiny shift, 0.3s apart -> triggers burst clustering)
    burst_base = _make_procedural_rgb_scene(seed=seed + 40, style="foliage")
    for b_idx, shift_px in enumerate((0, 4, 8, 12)):
        idx += 1
        fname = f"DEMO_{idx:03d}_burst_{b_idx + 1}.jpg"
        frame = np.roll(burst_base, shift=shift_px, axis=1)
        _save_jpeg_with_exif(root / fname, frame, capture_dt=_next_dt(step_s=0.3 if b_idx > 0 else 20.0))
        specs.append(SynthSampleSpec(fname, "keep", "burst", fname))

    # 3. 1 exempt (OS read-only locked file) + 1 long exposure creative protection
    idx += 1
    fname = f"DEMO_{idx:03d}_locked_exempt.jpg"
    _save_jpeg_with_exif(
        root / fname,
        _disk_defocus_rgb(_make_procedural_rgb_scene(seed=seed + 50), radius=10),
        capture_dt=_next_dt(step_s=20.0),
        readonly=True,
    )
    specs.append(SynthSampleSpec(fname, "exempt", "locked_readonly", fname))

    idx += 1
    fname = f"DEMO_{idx:03d}_long_exposure.jpg"
    _save_jpeg_with_exif(
        root / fname,
        _disk_defocus_rgb(_make_procedural_rgb_scene(seed=seed + 51), radius=9),
        capture_dt=_next_dt(),
        shutter_s=2.0,
    )
    specs.append(SynthSampleSpec(fname, "protected_non_red", "long_exposure", fname))

    # 4. 5 obvious rejects (defocus, motion blur, high-ISO noisy blur, extreme dark, extreme white)
    idx += 1
    fname = f"DEMO_{idx:03d}_defocus_reject.jpg"
    _save_jpeg_with_exif(
        root / fname,
        _disk_defocus_rgb(_make_procedural_rgb_scene(seed=seed + 60), radius=10),
        capture_dt=_next_dt(),
    )
    specs.append(SynthSampleSpec(fname, "obvious_reject", "defocus_blur", fname))

    idx += 1
    fname = f"DEMO_{idx:03d}_motion_reject.jpg"
    _save_jpeg_with_exif(
        root / fname,
        _linear_motion_blur_rgb(_make_procedural_rgb_scene(seed=seed + 61, style="foliage"), length=31, angle_deg=33.0),
        capture_dt=_next_dt(),
    )
    specs.append(SynthSampleSpec(fname, "obvious_reject", "motion_blur", fname))

    idx += 1
    fname = f"DEMO_{idx:03d}_noisy_blur_reject.jpg"
    rng = np.random.default_rng(seed + 62)
    nb_base = _disk_defocus_rgb(_make_procedural_rgb_scene(seed=seed + 62), radius=10).astype(np.float32)
    nb = np.clip(nb_base + rng.normal(0.0, 14.0, nb_base.shape), 0, 255).astype(np.uint8)
    _save_jpeg_with_exif(root / fname, nb, capture_dt=_next_dt(), iso=6400)
    specs.append(SynthSampleSpec(fname, "obvious_reject", "noisy_blur", fname))

    idx += 1
    fname = f"DEMO_{idx:03d}_lenscap_dark.jpg"
    dark = np.full((480, 640, 3), 2, dtype=np.uint8)
    _save_jpeg_with_exif(root / fname, dark, capture_dt=_next_dt())
    specs.append(SynthSampleSpec(fname, "obvious_reject", "extreme_dark", fname))

    idx += 1
    fname = f"DEMO_{idx:03d}_blown_white.jpg"
    white = np.full((480, 640, 3), 253, dtype=np.uint8)
    _save_jpeg_with_exif(root / fname, white, capture_dt=_next_dt())
    specs.append(SynthSampleSpec(fname, "obvious_reject", "extreme_white", fname))

    return specs


def build_e1_dataset(target_dir: Path, *, seed: int = 101) -> list[SynthSampleSpec]:
    """Generate the deterministic E1 benchmark suite (65 labeled photos across all categories)."""
    root = Path(target_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    base_dt = datetime(2026, 11, 1, 9, 0, 0, tzinfo=UTC)
    specs: list[SynthSampleSpec] = []
    idx = 0

    def _next_dt(step_s: float = 12.0) -> datetime:
        nonlocal base_dt
        base_dt = base_dt + timedelta(seconds=step_s)
        return base_dt

    # 1. 14 sharp normal scenes
    for i in range(14):
        idx += 1
        fname = f"E1_{idx:03d}_sharp.jpg"
        style = ("architecture", "foliage", "landscape")[i % 3]
        rgb = _make_procedural_rgb_scene(seed=seed + i, style=style)
        _save_jpeg_with_exif(root / fname, rgb, capture_dt=_next_dt())
        specs.append(SynthSampleSpec(fname, "keep", "sharp_normal", fname))

    # 2. 6 shallow DoF sharp-subject counter-examples
    for i in range(6):
        idx += 1
        fname = f"E1_{idx:03d}_shallow.jpg"
        _save_jpeg_with_exif(
            root / fname,
            _make_shallow_dof_rgb(seed=seed + 100 + i),
            capture_dt=_next_dt(),
            aperture=1.4,
        )
        specs.append(SynthSampleSpec(fname, "keep", "shallow_dof", fname))

    # 3. 4 panning shot counter-examples
    for i in range(4):
        idx += 1
        fname = f"E1_{idx:03d}_panning.jpg"
        _save_jpeg_with_exif(
            root / fname,
            _make_panning_rgb(seed=seed + 200 + i),
            capture_dt=_next_dt(),
            shutter_s=1 / 30,
        )
        specs.append(SynthSampleSpec(fname, "keep", "panning", fname))

    # 4. 4 high-ISO noisy sharp scenes
    for i in range(4):
        idx += 1
        fname = f"E1_{idx:03d}_noisy_sharp.jpg"
        base = _make_procedural_rgb_scene(seed=seed + 300 + i, style="architecture").astype(np.float32)
        noisy = np.clip(base + rng.normal(0.0, 11.0, base.shape), 0, 255).astype(np.uint8)
        _save_jpeg_with_exif(root / fname, noisy, capture_dt=_next_dt(), iso=6400)
        specs.append(SynthSampleSpec(fname, "keep", "noisy_sharp", fname))

    # 5. 7 creative protected scenes (must NEVER enter red)
    for i in range(2):
        idx += 1
        fname = f"E1_{idx:03d}_long_exp.jpg"
        rgb = _disk_defocus_rgb(_make_procedural_rgb_scene(seed=seed + 400 + i), radius=10)
        _save_jpeg_with_exif(root / fname, rgb, capture_dt=_next_dt(), shutter_s=2.5)
        specs.append(SynthSampleSpec(fname, "protected_non_red", "long_exposure", fname))

    for i, ev_val in enumerate((-2.0, 2.0)):
        idx += 1
        fname = f"E1_{idx:03d}_ev_comp.jpg"
        rgb = _disk_defocus_rgb(_make_procedural_rgb_scene(seed=seed + 410 + i), radius=10)
        _save_jpeg_with_exif(root / fname, rgb, capture_dt=_next_dt(), ev_comp=ev_val)
        specs.append(SynthSampleSpec(fname, "protected_non_red", "extreme_ev_comp", fname))

    # 3 consecutive slow-shutter soft frames (suspected ICM)
    for i in range(3):
        idx += 1
        fname = f"E1_{idx:03d}_icm.jpg"
        rgb = _linear_motion_blur_rgb(
            _make_procedural_rgb_scene(seed=seed + 420 + i, style="foliage"),
            length=31,
            angle_deg=35.0,
        )
        _save_jpeg_with_exif(
            root / fname,
            rgb,
            capture_dt=_next_dt(step_s=1.0 if i > 0 else 20.0),
            shutter_s=1 / 8,
        )
        specs.append(SynthSampleSpec(fname, "protected_non_red", "icm_sequence", fname))

    # 6. 5 exempt scenes (2 OS read-only locked + 3 stepped-EV AEB bracket)
    for i in range(2):
        idx += 1
        fname = f"E1_{idx:03d}_exempt_locked.jpg"
        rgb = _disk_defocus_rgb(_make_procedural_rgb_scene(seed=seed + 500 + i), radius=10)
        _save_jpeg_with_exif(root / fname, rgb, capture_dt=_next_dt(step_s=20.0), readonly=True)
        specs.append(SynthSampleSpec(fname, "exempt", "exempt_locked", fname))

    aeb_scene = _make_procedural_rgb_scene(seed=seed + 510, style="landscape")
    for i, ev_val in enumerate((-1.5, 0.0, 1.5)):
        idx += 1
        fname = f"E1_{idx:03d}_exempt_aeb.jpg"
        shifted_brightness = np.clip(aeb_scene.astype(np.float32) * (1.0 + 0.15 * ev_val), 0, 255).astype(np.uint8)
        _save_jpeg_with_exif(
            root / fname,
            shifted_brightness,
            capture_dt=_next_dt(step_s=0.5 if i > 0 else 20.0),
            ev_comp=ev_val,
        )
        specs.append(SynthSampleSpec(fname, "exempt", "exempt_aeb", fname))

    # 7. 25 obvious rejects (must achieve >= 75% red recall)
    # 7a. 8 severe disk defocus
    for i in range(8):
        idx += 1
        fname = f"E1_{idx:03d}_reject_defocus.jpg"
        style = ("architecture", "foliage", "landscape")[i % 3]
        rgb = _disk_defocus_rgb(
            _make_procedural_rgb_scene(seed=seed + 600 + i, style=style),
            radius=9 + (i % 3),
        )
        _save_jpeg_with_exif(root / fname, rgb, capture_dt=_next_dt(step_s=20.0))
        specs.append(SynthSampleSpec(fname, "obvious_reject", "defocus_blur", fname))

    # 7b. 7 strong diagonal linear motion blur
    for i in range(7):
        idx += 1
        fname = f"E1_{idx:03d}_reject_motion.jpg"
        style = ("foliage", "landscape")[i % 2]
        rgb = _linear_motion_blur_rgb(
            _make_procedural_rgb_scene(seed=seed + 700 + i, style=style),
            length=29 + (i % 3) * 2,
            angle_deg=25.0 + i * 11.0,
        )
        _save_jpeg_with_exif(root / fname, rgb, capture_dt=_next_dt(step_s=20.0))
        specs.append(SynthSampleSpec(fname, "obvious_reject", "motion_blur", fname))

    # 7c. 5 high-ISO noise + severe defocus blur
    for i in range(5):
        idx += 1
        fname = f"E1_{idx:03d}_reject_noisy_blur.jpg"
        blurred = _disk_defocus_rgb(
            _make_procedural_rgb_scene(seed=seed + 800 + i, style="foliage"),
            radius=10,
        ).astype(np.float32)
        noisy_blur = np.clip(blurred + rng.normal(0.0, 14.0, blurred.shape), 0, 255).astype(np.uint8)
        _save_jpeg_with_exif(root / fname, noisy_blur, capture_dt=_next_dt(step_s=20.0), iso=12800)
        specs.append(SynthSampleSpec(fname, "obvious_reject", "noisy_blur", fname))

    # 7d. 3 extreme black + 2 extreme white mis-shots
    for i in range(3):
        idx += 1
        fname = f"E1_{idx:03d}_reject_dark.jpg"
        dark = np.full((480, 640, 3), i + 1, dtype=np.uint8)
        _save_jpeg_with_exif(root / fname, dark, capture_dt=_next_dt(step_s=20.0))
        specs.append(SynthSampleSpec(fname, "obvious_reject", "extreme_dark", fname))

    for i in range(2):
        idx += 1
        fname = f"E1_{idx:03d}_reject_white.jpg"
        white = np.full((480, 640, 3), 254 - i, dtype=np.uint8)
        _save_jpeg_with_exif(root / fname, white, capture_dt=_next_dt(step_s=20.0))
        specs.append(SynthSampleSpec(fname, "obvious_reject", "extreme_white", fname))

    manifest_path = root / "ground_truth.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "filename": s.filename,
                    "label": s.label,
                    "category": s.category,
                    "rel_path": s.rel_path,
                }
                for s in specs
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return specs


def restore_dataset_write_bits(target_dir: Path) -> None:
    """Restore user-write bits on synthetic read-only files so temp dirs clean up smoothly."""
    for path in Path(target_dir).rglob("*"):
        if path.is_file():
            try:
                mode = path.stat().st_mode
                if not bool(mode & stat.S_IWUSR):
                    path.chmod(mode | stat.S_IWUSR)
            except OSError:
                pass


def run_e1_evaluation(
    work_dir: Path,
    *,
    sensitivity: SensitivityMode = "conservative",
    seed: int = 101,
    max_workers: int = 1,
) -> E1EvalReport:
    """Run the zero-network E1 synthetic engine benchmark and assert CI hard gates (SPEC §11.2)."""
    started = time.monotonic()
    root = Path(work_dir).expanduser().resolve()
    dataset_dir = root / "e1_dataset"
    home_dir = root / "e1_home"
    specs = build_e1_dataset(dataset_dir, seed=seed)

    try:
        with LightboxServices.open(home_override=home_dir) as svcs:
            scan_res = svcs.scan.scan_folder(
                dataset_dir,
                sensitivity=sensitivity,
                max_workers=max_workers,
            )
            photos = svcs.repo.list_photos(scan_res.session_id)
            verdicts = svcs.repo.list_effective_verdicts(scan_res.session_id)
            bucket_by_stem = {
                p.stem: (verdicts[p.id].bucket if p.id in verdicts else "normal")
                for p in photos
            }
    finally:
        restore_dataset_write_bits(dataset_dir)

    keep_prot_total = 0
    false_red = 0
    reject_total = 0
    true_red = 0
    exempt_total = 0
    exempt_mislabel = 0
    failures: list[str] = []
    breakdown: dict[str, dict[str, int]] = {}

    for spec in specs:
        stem = Path(spec.filename).stem
        bucket = bucket_by_stem.get(stem, "missing")
        cat_counts = breakdown.setdefault(spec.category, {})
        cat_counts[bucket] = cat_counts.get(bucket, 0) + 1

        if spec.label in {"keep", "protected_non_red"}:
            keep_prot_total += 1
            if bucket == "red":
                false_red += 1
                failures.append(f"FALSE_RED: {spec.filename} ({spec.category}) -> {bucket}")
        elif spec.label == "obvious_reject":
            reject_total += 1
            if bucket == "red":
                true_red += 1
            else:
                failures.append(f"MISSED_REJECT: {spec.filename} ({spec.category}) -> {bucket}")
        elif spec.label == "exempt":
            exempt_total += 1
            if bucket != "exempt":
                exempt_mislabel += 1
                failures.append(f"EXEMPT_MISLABEL: {spec.filename} ({spec.category}) -> {bucket}")

    red_fpr = (false_red / float(keep_prot_total)) if keep_prot_total > 0 else 0.0
    reject_recall = (true_red / float(reject_total)) if reject_total > 0 else 1.0
    exempt_rate = (exempt_mislabel / float(exempt_total)) if exempt_total > 0 else 0.0

    passed = (
        red_fpr < E1_MAX_RED_FPR
        and reject_recall >= E1_MIN_REJECT_RECALL
        and exempt_rate == E1_MAX_EXEMPT_MISLABEL
    )
    return E1EvalReport(
        sensitivity=sensitivity,
        total_samples=len(specs),
        keep_and_protected_count=keep_prot_total,
        false_red_count=false_red,
        red_fpr=red_fpr,
        obvious_reject_count=reject_total,
        true_red_count=true_red,
        obvious_reject_recall=reject_recall,
        exempt_count=exempt_total,
        exempt_mislabel_count=exempt_mislabel,
        exempt_mislabel_rate=exempt_rate,
        passed=passed,
        elapsed_s=round(time.monotonic() - started, 3),
        failures=tuple(failures),
        category_breakdown=breakdown,
    )


def run_personal_eval(svcs: LightboxServices) -> PersonalEvalReport:
    """Summarize offline personal override history (`lightbox eval --personal`)."""
    overrides = svcs.repo.list_overrides()
    rescued_from_red = sum(1 for o in overrides if o.from_bucket == "red" and o.to_bucket == "normal")
    rejected_from_normal = sum(1 for o in overrides if o.from_bucket == "normal" and o.to_bucket == "red")
    by_reason: dict[str, int] = {}
    for ov in overrides:
        tag = ov.reason_tag or "untagged"
        by_reason[tag] = by_reason.get(tag, 0) + 1
    return PersonalEvalReport(
        total_overrides=len(overrides),
        rescued_from_red=rescued_from_red,
        rejected_from_normal=rejected_from_normal,
        overrides_by_reason=by_reason,
    )


__all__ = [
    "E1EvalReport",
    "E1_MAX_EXEMPT_MISLABEL",
    "E1_MAX_RED_FPR",
    "E1_MIN_REJECT_RECALL",
    "GroundTruthLabel",
    "PersonalEvalReport",
    "SynthSampleSpec",
    "build_demo_dataset",
    "build_e1_dataset",
    "restore_dataset_write_bits",
    "run_e1_evaluation",
    "run_personal_eval",
]
