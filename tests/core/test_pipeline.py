"""T05 unit tests: deterministic detection pipeline, InsightFace decoder/thresholds, bursts, guards, and i18n."""

from __future__ import annotations

import math
import stat
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from lightbox.core import (
    ENGINE_VERSION,
    BurstCandidate,
    Detector,
    FaceDetector,
    MetricsDetector,
    NormalizedExif,
    PhotoContext,
    PhotoEvaluationInput,
    RawFaceDetection,
    compute_dhash,
    compute_image_metrics,
    decode_scrfd_outputs,
    detect_bursts,
    evaluate_session_verdicts,
    make_motion_thumb,
)
from lightbox.i18n import (
    REASON_AEB_BRACKET,
    REASON_BURST_NON_PICK,
    REASON_EXTREME_EV_COMP_PROTECTED,
    REASON_EXTREME_EXPOSURE,
    REASON_FACE_OUT_OF_FOCUS,
    REASON_IN_CAMERA_PROTECTED,
    REASON_IN_CAMERA_RATED,
    REASON_LONG_EXPOSURE_PROTECTED,
    REASON_MOTION_BLUR,
    REASON_OUT_OF_FOCUS,
    REASON_PANNING_PROTECTED,
    REASON_SUSPECTED_ICM,
    is_burst_cull_reason,
    is_tech_reject_reason,
    render_reason,
)
from lightbox.store import Database, FaceRecord, LightboxRepository


def _make_sharp_scene(width: int = 512, height: int = 512, seed: int = 1) -> NDArray[np.uint8]:
    """Generate a crisp multi-orientation textured grayscale scene."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width), 120.0, dtype=np.float32)

    # Multi-angle crisp geometric buildings, windows, and foliage strokes
    for _ in range(45):
        x0 = int(rng.integers(10, width - 60))
        y0 = int(rng.integers(10, height - 60))
        w = int(rng.integers(20, 70))
        h = int(rng.integers(20, 70))
        val = float(rng.integers(20, 240))
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), val, thickness=-1)
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), 255.0 - val, thickness=2)

    for _ in range(40):
        pt1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        pt2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        val = float(rng.integers(10, 245))
        cv2.line(img, pt1, pt2, val, thickness=2, lineType=cv2.LINE_8)

    return np.clip(img, 0, 255).astype(np.uint8)


def _apply_disk_defocus(img: NDArray[np.uint8], radius: int) -> NDArray[np.uint8]:
    """Convolve with a uniform circular disk PSF simulating severe lens defocus."""
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    kernel /= float(kernel.sum())
    blurred = cv2.filter2D(img.astype(np.float32), -1, kernel)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def _apply_linear_motion_blur(img: NDArray[np.uint8], length: int, angle_deg: float = 0.0) -> NDArray[np.uint8]:
    """Convolve with a 1D linear motion kernel simulating camera shake."""
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    rad = math.radians(angle_deg)
    dx = math.cos(rad)
    dy = math.sin(rad)
    for t in np.linspace(-center, center, length * 3):
        x = int(round(center + t * dx))
        y = int(round(center + t * dy))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    kernel /= max(float(kernel.sum()), 1.0)
    blurred = cv2.filter2D(img.astype(np.float32), -1, kernel)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def _make_shallow_dof_scene(width: int = 512, height: int = 512) -> NDArray[np.uint8]:
    """85% smooth creamy bokeh background + 15% crisp centered subject."""
    base = _make_sharp_scene(width, height, seed=7)
    bg = cv2.GaussianBlur(base, (31, 31), 10.0)
    subject_patch = _make_sharp_scene(width, height, seed=8)
    out = bg.copy()
    y0, y1 = int(height * 0.30), int(height * 0.70)
    x0, x1 = int(width * 0.32), int(width * 0.68)
    out[y0:y1, x0:x1] = subject_patch[y0:y1, x0:x1]
    return out


def _make_panning_scene(width: int = 512, height: int = 512) -> NDArray[np.uint8]:
    """Horizontally motion-blurred background + sharp centered moving subject."""
    bg = _apply_linear_motion_blur(_make_sharp_scene(width, height, seed=11), length=27, angle_deg=0.0)
    subj = _make_sharp_scene(width, height, seed=12)
    out = bg.copy()
    y0, y1 = int(height * 0.35), int(height * 0.65)
    x0, x1 = int(width * 0.35), int(width * 0.65)
    out[y0:y1, x0:x1] = subj[y0:y1, x0:x1]
    return out


class _StubFaceBackend:
    def __init__(self, detections: list[RawFaceDetection]) -> None:
        self._detections = detections

    @property
    def available(self) -> bool:
        return True

    def detect_raw(self, _image_rgb: NDArray[np.uint8]) -> list[RawFaceDetection]:
        return list(self._detections)


def _eval_with_sharp_peers(
    target_input: PhotoEvaluationInput,
    *,
    num_peers: int = 6,
    sensitivity: str = "conservative",
) -> dict[str, object]:
    """Helper embedding `target_input` alongside sharp peers of the same cohort."""
    peer_exif = NormalizedExif(
        camera_model=target_input.exif.camera_model if target_input.exif else "SONY A7M4",
        lens_model=target_input.exif.lens_model if target_input.exif else "FE 35mm F1.4",
        cohort=target_input.exif.cohort if target_input.exif else "camera",
        shutter_s=1 / 500,
        aperture=2.8,
        iso=200,
        focal_eq=35.0,
    )
    items: list[PhotoEvaluationInput] = []
    for i in range(num_peers):
        img = _make_sharp_scene(seed=100 + i)
        items.append(
            PhotoEvaluationInput(
                photo_id=f"peer_{i}",
                content_fp=f"fp_peer_{i}",
                metrics=compute_image_metrics(img, exif=peer_exif),
                dhash=compute_dhash(img),
                exif=peer_exif,
            )
        )
    items.append(target_input)
    verdicts = evaluate_session_verdicts(items, sensitivity=sensitivity)  # type: ignore[arg-type]
    by_id = {v.photo_id: v for v in verdicts}
    return {"target": by_id[target_input.photo_id], "all": by_id}


def test_detector_protocol_and_metrics_noise_normalization() -> None:
    sharp = _make_sharp_scene(seed=1)
    defocused = _apply_disk_defocus(sharp, radius=9)
    motion = _apply_linear_motion_blur(sharp, length=27, angle_deg=32.0)

    # High-ISO noise added to defocused image must NOT create fake sharpness
    rng = np.random.default_rng(42)
    noisy_blur = np.clip(
        defocused.astype(np.float32) + rng.normal(0.0, 14.0, defocused.shape), 0, 255
    ).astype(np.uint8)

    md = MetricsDetector()
    assert isinstance(md, Detector)
    assert md.version == ENGINE_VERSION

    ctx = PhotoContext(photo_id="p1", content_fp="fp1", analysis_img=sharp)
    out = md.run(ctx)
    assert out["sharp_p95"] > 40.0

    m_sharp = compute_image_metrics(sharp)
    m_defocus = compute_image_metrics(defocused)
    m_motion = compute_image_metrics(motion)
    m_noisy_blur = compute_image_metrics(noisy_blur)

    assert m_sharp.sharp_p95 > 40.0
    assert m_defocus.sharp_p95 < 6.0
    assert m_motion.sharp_p95 < 16.0
    assert m_motion.motion_aniso > m_defocus.motion_aniso
    assert m_noisy_blur.noise_sigma > 9.0
    assert m_noisy_blur.sharp_p95 < 6.0

    # Shallow DoF sharp subject must stay sharp & normal
    shallow = _make_shallow_dof_scene()
    m_shallow = compute_image_metrics(shallow)
    assert m_shallow.sharp_p95 > 35.0

    # Panning shot must be detected as panning and kept normal
    pan = _make_panning_scene()
    m_pan = compute_image_metrics(pan)
    assert m_pan.panning_detected is True
    assert m_pan.motion_state == "panning"

    pan_eval = _eval_with_sharp_peers(
        PhotoEvaluationInput(
            photo_id="pan_1",
            content_fp="fp_pan",
            metrics=m_pan,
            dhash=compute_dhash(pan),
            exif=NormalizedExif(camera_model="SONY A7M4", lens_model="FE 35mm F1.4", shutter_s=1 / 30),
        )
    )
    pan_v = pan_eval["target"]
    assert pan_v.bucket == "normal"  # type: ignore[attr-defined]
    assert REASON_PANNING_PROTECTED in pan_v.reason_codes  # type: ignore[attr-defined]


def test_extreme_exposure_misshots_hit_red_while_textured_scenes_safe() -> None:
    black = np.full((256, 256), 2, dtype=np.uint8)
    white = np.full((256, 256), 253, dtype=np.uint8)

    # Dark night sky with a small textured bright building (entropy / gradient above threshold)
    night_scene = np.full((256, 256), 4, dtype=np.uint8)
    night_scene[90:170, 90:170] = _make_sharp_scene(80, 80, seed=5)

    m_black = compute_image_metrics(black)
    m_white = compute_image_metrics(white)
    m_night = compute_image_metrics(night_scene)

    assert m_black.is_extreme_exposure is True and m_black.extreme_kind == "underexposed"
    assert m_white.is_extreme_exposure is True and m_white.extreme_kind == "overexposed"
    assert m_night.is_extreme_exposure is False

    res_black = _eval_with_sharp_peers(
        PhotoEvaluationInput(
            photo_id="blk",
            content_fp="fp_blk",
            metrics=m_black,
            dhash=compute_dhash(black),
            exif=NormalizedExif(camera_model="SONY A7M4", lens_model="FE 35mm F1.4", shutter_s=1 / 250),
        )
    )
    v_blk = res_black["target"]
    assert v_blk.bucket == "red"  # type: ignore[attr-defined]
    assert REASON_EXTREME_EXPOSURE in v_blk.reason_codes  # type: ignore[attr-defined]


def test_scrfd_decoder_decodes_bbox_and_5point_landmarks() -> None:
    """Verify SCRFD multi-stride decoder decodes synthetic score/bbox/kps tensors + NMS."""
    input_size = (64, 64)
    strides = (8, 16, 32)
    outputs: list[NDArray[np.float32]] = []
    scores_list: list[NDArray[np.float32]] = []
    bboxes_list: list[NDArray[np.float32]] = []
    kps_list: list[NDArray[np.float32]] = []

    for stride in strides:
        n = (input_size[0] // stride) * (input_size[1] // stride) * 2
        sc = np.zeros((n, 1), dtype=np.float32)
        bb = np.zeros((n, 4), dtype=np.float32)
        kp = np.zeros((n, 10), dtype=np.float32)
        if stride == 16:
            # Place a high-confidence face centered at grid (2, 2) -> pixel (32, 32)
            # feat_w = 4, cell (y=2, x=2) -> index (2*4 + 2)*2 = 20
            idx = 20
            sc[idx, 0] = 0.92
            bb[idx] = [1.0, 1.0, 1.0, 1.0]  # *16 -> [16, 16, 48, 48]
            kp[idx] = [-0.4, -0.3, 0.4, -0.3, 0.0, 0.0, -0.3, 0.4, 0.3, 0.4]
            # Duplicate slightly weaker anchor at same cell -> should be suppressed by NMS
            sc[idx + 1, 0] = 0.75
            bb[idx + 1] = [1.0, 1.0, 1.0, 1.0]
            kp[idx + 1] = kp[idx]
        scores_list.append(sc)
        bboxes_list.append(bb)
        kps_list.append(kp)

    outputs.extend(scores_list)
    outputs.extend(bboxes_list)
    outputs.extend(kps_list)

    dets = decode_scrfd_outputs(outputs, input_size=input_size, det_scale=0.5)
    assert len(dets) == 1
    det = dets[0]
    assert det.det_score > 0.9
    # Scaled by 1 / 0.5 = 2x -> [32, 32, 96, 96]
    assert det.bbox_px == (32.0, 32.0, 96.0, 96.0)
    assert det.landmarks_px is not None and len(det.landmarks_px) == 5


def test_group_photo_small_face_stored_with_landmarks_and_no_false_oof(tmp_path: Path) -> None:
    """PLAN T05: 0.5% area small face in group photo stored with 5-pt landmarks & no OOF warning."""
    img = _make_sharp_scene(512, 512, seed=21)
    # Soften a tiny 36x36 region (area_ratio = 1296 / 262144 ~= 0.00494 ~= 0.5%)
    img[100:136, 100:136] = 128

    small_face = RawFaceDetection(
        bbox_px=(100.0, 100.0, 136.0, 136.0),
        landmarks_px=(
            (110.0, 112.0),
            (126.0, 112.0),
            (118.0, 120.0),
            (112.0, 128.0),
            (124.0, 128.0),
        ),
        det_score=0.88,
    )
    fd = FaceDetector(backend=_StubFaceBackend([small_face]))
    assert isinstance(fd, Detector)

    m = compute_image_metrics(img)
    ctx = PhotoContext(
        photo_id="grp1",
        content_fp="fp_grp1",
        analysis_img=img,
        metrics={"sharp_p95": m.sharp_p95, "noise_sigma": m.noise_sigma},
    )
    face_out = fd.run(ctx)
    assert face_out["face_count"] == 1
    assert face_out["face_out_of_focus"] is False  # 0.5% < 1.5% subject threshold!
    stored_face = ctx.faces[0]
    assert 0.004 <= stored_face.area_ratio <= 0.006
    assert stored_face.landmarks is not None and len(stored_face.landmarks) == 5
    assert stored_face.det_score == 0.88

    # Verify DB persistence with 5-point landmarks_json and det_score
    db = Database(tmp_path / "faces.db")
    db.migrate()
    repo = LightboxRepository(db)
    try:
        repo.replace_faces(
            "fp_grp1",
            [
                FaceRecord(
                    content_fp="fp_grp1",
                    idx=stored_face.idx,
                    bbox=stored_face.bbox,
                    landmarks=stored_face.landmarks,
                    det_score=stored_face.det_score,
                    sharpness=stored_face.sharpness,
                    area_ratio=stored_face.area_ratio,
                )
            ],
        )
        loaded = repo.list_faces("fp_grp1")
        assert len(loaded) == 1
        assert loaded[0].landmarks is not None and len(loaded[0].landmarks) == 5
        assert loaded[0].det_score == 0.88
    finally:
        db.close()

    # Verdict must stay normal (small blurred face does NOT trigger OOF)
    ev = _eval_with_sharp_peers(
        PhotoEvaluationInput(
            photo_id="grp1",
            content_fp="fp_grp1",
            metrics=m,
            dhash=compute_dhash(img),
            faces=ctx.faces,
            exif=NormalizedExif(camera_model="SONY A7M4", lens_model="FE 35mm F1.4"),
        )
    )
    assert ev["target"].bucket == "normal"  # type: ignore[attr-defined]


def test_subject_face_out_of_focus_caps_at_yellow_never_red() -> None:
    """Hard rule: subject face (>= 1.5% area) out-of-focus -> yellow, NEVER red."""
    img = _make_sharp_scene(512, 512, seed=25)
    # Blur a 120x120 subject face area (area_ratio ~= 5.5%)
    img[180:300, 180:300] = _apply_disk_defocus(img[180:300, 180:300], radius=12)

    subj_face = RawFaceDetection(
        bbox_px=(180.0, 180.0, 300.0, 300.0),
        landmarks_px=(
            (215.0, 220.0),
            (265.0, 220.0),
            (240.0, 245.0),
            (220.0, 275.0),
            (260.0, 275.0),
        ),
        det_score=0.95,
    )
    fd = FaceDetector(backend=_StubFaceBackend([subj_face]))
    m = compute_image_metrics(img)
    ctx = PhotoContext(
        photo_id="portrait_oof",
        content_fp="fp_poof",
        analysis_img=img,
        metrics={"sharp_p95": m.sharp_p95, "noise_sigma": m.noise_sigma},
    )
    face_out = fd.run(ctx)
    assert face_out["face_out_of_focus"] is True

    ev = _eval_with_sharp_peers(
        PhotoEvaluationInput(
            photo_id="portrait_oof",
            content_fp="fp_poof",
            metrics=m,
            dhash=compute_dhash(img),
            faces=ctx.faces,
            exif=NormalizedExif(camera_model="SONY A7M4", lens_model="FE 35mm F1.4"),
        )
    )
    v = ev["target"]
    assert v.bucket == "yellow"  # type: ignore[attr-defined]
    assert REASON_FACE_OUT_OF_FOCUS in v.reason_codes  # type: ignore[attr-defined]


def test_creative_protection_hard_rules(tmp_path: Path) -> None:
    """Verify every creative-intent protection hard rule in SPEC §4.3 & PLAN T05."""
    sharp = _make_sharp_scene(512, 512, seed=31)
    severely_blurred = _apply_disk_defocus(sharp, radius=11)
    m_blur = compute_image_metrics(severely_blurred)
    m_sharp = compute_image_metrics(sharp)
    dhash_blur = compute_dhash(severely_blurred)
    dhash_sharp = compute_dhash(sharp)

    # 1. Control: unprotected severely blurred photo MUST enter red
    ctrl = _eval_with_sharp_peers(
        PhotoEvaluationInput(
            photo_id="ctrl_blur",
            content_fp="fp_ctrl",
            metrics=m_blur,
            dhash=dhash_blur,
            exif=NormalizedExif(
                camera_model="SONY A7M4", lens_model="FE 35mm F1.4", shutter_s=1 / 250, ev_comp=0.0
            ),
        )
    )
    assert ctrl["target"].bucket == "red"  # type: ignore[attr-defined]
    assert REASON_OUT_OF_FOCUS in ctrl["target"].reason_codes  # type: ignore[attr-defined]

    # 2. Long exposure (shutter_s >= 1.0) MUST NOT enter red (max yellow)
    long_exp = _eval_with_sharp_peers(
        PhotoEvaluationInput(
            photo_id="long_exp",
            content_fp="fp_le",
            metrics=m_blur,
            dhash=dhash_blur,
            exif=NormalizedExif(
                camera_model="SONY A7M4", lens_model="FE 35mm F1.4", shutter_s=2.5, ev_comp=0.0
            ),
        )
    )
    assert long_exp["target"].bucket == "yellow"  # type: ignore[attr-defined]
    assert REASON_LONG_EXPOSURE_PROTECTED in long_exp["target"].reason_codes  # type: ignore[attr-defined]

    # 3. Extreme EV compensation (|ev_comp| >= 1.5) MUST NOT enter red (max yellow)
    ev_comp_res = _eval_with_sharp_peers(
        PhotoEvaluationInput(
            photo_id="ev_comp",
            content_fp="fp_evc",
            metrics=m_blur,
            dhash=dhash_blur,
            exif=NormalizedExif(
                camera_model="SONY A7M4", lens_model="FE 35mm F1.4", shutter_s=1 / 250, ev_comp=-2.0
            ),
        )
    )
    assert ev_comp_res["target"].bucket == "yellow"  # type: ignore[attr-defined]
    assert REASON_EXTREME_EV_COMP_PROTECTED in ev_comp_res["target"].reason_codes  # type: ignore[attr-defined]

    # 4. In-camera protected, rated, AEB tag, and OS read-only file -> MUST be exempt
    ro_file = tmp_path / "readonly.jpg"
    ro_file.write_bytes(b"readonly-photo")
    ro_file.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    for pid, exif_obj, fpath, expected_reason in (
        (
            "ex_lock",
            NormalizedExif(camera_model="SONY A7M4", lens_model="FE 35mm F1.4", protected=True),
            None,
            REASON_IN_CAMERA_PROTECTED,
        ),
        (
            "ex_rate",
            NormalizedExif(camera_model="SONY A7M4", lens_model="FE 35mm F1.4", rating=3),
            None,
            REASON_IN_CAMERA_RATED,
        ),
        (
            "ex_aeb",
            NormalizedExif(camera_model="SONY A7M4", lens_model="FE 35mm F1.4", bracket_group="AEB 3"),
            None,
            REASON_AEB_BRACKET,
        ),
        (
            "ex_ro",
            NormalizedExif(camera_model="SONY A7M4", lens_model="FE 35mm F1.4"),
            ro_file,
            REASON_IN_CAMERA_PROTECTED,
        ),
    ):
        res = _eval_with_sharp_peers(
            PhotoEvaluationInput(
                photo_id=pid,
                content_fp=f"fp_{pid}",
                metrics=m_blur,
                dhash=dhash_blur,
                exif=exif_obj,
                file_path=fpath,
            )
        )
        v = res["target"]
        assert v.bucket == "exempt"  # type: ignore[attr-defined]
        assert v.is_protected is True  # type: ignore[attr-defined]
        assert expected_reason in v.reason_codes  # type: ignore[attr-defined]

    # Restore write bit on temp file for clean pytest tmp_path teardown
    ro_file.chmod(stat.S_IRUSR | stat.S_IWUSR)

    # 5. Handheld safe shutter (shutter_s > 1/focal_eq) MUST NOT trigger red (or yellow) on sharp photo
    handheld = _eval_with_sharp_peers(
        PhotoEvaluationInput(
            photo_id="slow_but_sharp",
            content_fp="fp_hs",
            metrics=m_sharp,
            dhash=dhash_sharp,
            exif=NormalizedExif(
                camera_model="SONY A7M4",
                lens_model="FE 35mm F1.4",
                focal_eq=200.0,
                shutter_s=1 / 30,  # Much slower than 1/200s, yet shot is sharp (IBIS/stabilized)
            ),
        )
    )
    assert handheld["target"].bucket == "normal"  # type: ignore[attr-defined]
    assert handheld["target"].detail["slower_than_handheld_rule"] is True  # type: ignore[attr-defined]


def test_icm_sequence_and_aeb_sequence_protection() -> None:
    """Consecutive >=3 slow-shutter soft frames -> yellow + suspected_icm; stepped exposure -> exempt."""
    sharp = _make_sharp_scene(512, 512, seed=41)
    # Realistic ICM: strong diagonal sweep + slight lens softness
    soft = cv2.GaussianBlur(_apply_linear_motion_blur(sharp, length=29, angle_deg=32.0), (5, 5), 1.5)
    m_sharp = compute_image_metrics(sharp)
    m_soft = compute_image_metrics(soft)
    dh_sharp = compute_dhash(sharp)
    dh_soft = compute_dhash(soft)

    items: list[PhotoEvaluationInput] = []
    # 6 sharp peers
    for i in range(6):
        items.append(
            PhotoEvaluationInput(
                photo_id=f"sharp_{i}",
                content_fp=f"fp_s_{i}",
                metrics=m_sharp,
                dhash=dh_sharp,
                exif=NormalizedExif(
                    camera_model="FUJI X-T5",
                    lens_model="XF 23mm",
                    camera_serial="SN001",
                    capture_time=1000.0 + i * 10.0,
                    shutter_s=1 / 500,
                ),
            )
        )

    # 3 consecutive slow-shutter (>= 1/15s) soft shots -> suspected_icm -> yellow
    for j in range(3):
        items.append(
            PhotoEvaluationInput(
                photo_id=f"icm_{j}",
                content_fp=f"fp_icm_{j}",
                metrics=m_soft,
                dhash=dh_soft,
                exif=NormalizedExif(
                    camera_model="FUJI X-T5",
                    lens_model="XF 23mm",
                    camera_serial="SN001",
                    capture_time=2000.0 + j * 1.0,
                    shutter_s=1 / 8,  # 0.125s >= 1/15s
                ),
            )
        )

    # 3 consecutive stepped-EV frames (AEB sequence without bracket tag) -> exempt
    for k, ev_val in enumerate((-1.0, 0.0, 1.0)):
        items.append(
            PhotoEvaluationInput(
                photo_id=f"aeb_{k}",
                content_fp=f"fp_aeb_{k}",
                metrics=m_soft,
                dhash=dh_sharp,
                exif=NormalizedExif(
                    camera_model="FUJI X-T5",
                    lens_model="XF 23mm",
                    camera_serial="SN001",
                    capture_time=3000.0 + k * 0.5,
                    shutter_s=1 / 125,
                    ev_comp=ev_val,
                ),
            )
        )

    verdicts = {v.photo_id: v for v in evaluate_session_verdicts(items)}
    for j in range(3):
        v = verdicts[f"icm_{j}"]
        assert v.bucket == "yellow"
        assert REASON_SUSPECTED_ICM in v.reason_codes
        assert REASON_MOTION_BLUR in v.reason_codes

    for k in range(3):
        v = verdicts[f"aeb_{k}"]
        assert v.bucket == "exempt"
        assert REASON_AEB_BRACKET in v.reason_codes


def test_phone_cohort_isolated_and_all_soft_session_never_red() -> None:
    """Phone cohort never mixes with camera cohort; an all-soft misty session produces 0 red."""
    sharp = _make_sharp_scene(512, 512, seed=55)
    soft_misty = _apply_disk_defocus(sharp, radius=8)
    m_sharp = compute_image_metrics(sharp)
    m_soft = compute_image_metrics(soft_misty)

    # Camera has 6 sharp shots; phone has 5 uniformly soft misty shots -> phone cohort isolated -> 0 red!
    items: list[PhotoEvaluationInput] = []
    for i in range(6):
        items.append(
            PhotoEvaluationInput(
                photo_id=f"cam_{i}",
                content_fp=f"fp_cam_{i}",
                metrics=m_sharp,
                dhash=compute_dhash(sharp),
                exif=NormalizedExif(camera_model="SONY A7M4", lens_model="FE 50mm", cohort="camera"),
            )
        )
    for j in range(5):
        items.append(
            PhotoEvaluationInput(
                photo_id=f"phone_{j}",
                content_fp=f"fp_ph_{j}",
                metrics=m_soft,
                dhash=compute_dhash(soft_misty),
                exif=NormalizedExif(camera_model="iPhone 16 Pro", lens_model="Main", cohort="phone"),
            )
        )

    verdicts = {v.photo_id: v for v in evaluate_session_verdicts(items, sensitivity="conservative")}
    for j in range(5):
        assert verdicts[f"phone_{j}"].bucket != "red"


def test_cross_subdir_burst_clustering_and_motion_estimation() -> None:
    base = _make_sharp_scene(512, 512, seed=61)
    # Shift image slightly between burst frames to simulate moving subject / camera follow
    shifted1 = np.roll(base, shift=12, axis=1)
    shifted2 = np.roll(base, shift=24, axis=1)
    different = _make_sharp_scene(512, 512, seed=99)

    cands = [
        BurstCandidate(
            photo_id="b1",
            capture_time=100.0,
            camera_serial="SN_A",
            camera_model="A7M4",
            cohort="camera",
            dhash=compute_dhash(base),
            sharp_p95=70.0,
            motion_thumb=make_motion_thumb(base),
        ),
        BurstCandidate(
            photo_id="b2",
            capture_time=100.3,
            camera_serial="SN_A",
            camera_model="A7M4",
            cohort="camera",
            dhash=compute_dhash(shifted1),
            sharp_p95=95.0,  # sharpest -> should rank #1
            motion_thumb=make_motion_thumb(shifted1),
        ),
        BurstCandidate(
            photo_id="b3",
            capture_time=100.6,
            camera_serial="SN_A",
            camera_model="A7M4",
            cohort="camera",
            dhash=compute_dhash(shifted2),
            sharp_p95=60.0,
            motion_thumb=make_motion_thumb(shifted2),
        ),
        # Different camera serial at the exact same timestamp -> must NOT merge
        BurstCandidate(
            photo_id="other_cam",
            capture_time=100.2,
            camera_serial="SN_B",
            camera_model="A7M4",
            cohort="camera",
            dhash=compute_dhash(base),
            sharp_p95=80.0,
            motion_thumb=make_motion_thumb(base),
        ),
        # Visually totally different scene 0.2s later -> must NOT merge
        BurstCandidate(
            photo_id="diff_scene",
            capture_time=100.8,
            camera_serial="SN_A",
            camera_model="A7M4",
            cohort="camera",
            dhash=compute_dhash(different),
            sharp_p95=80.0,
            motion_thumb=make_motion_thumb(different),
        ),
    ]

    groups, motion_map = detect_bursts("sess_1", cands)
    assert len(groups) == 1
    grp = groups[0]
    assert [m.photo_id for m in grp.members] == ["b2", "b1", "b3"]
    assert grp.members[0].rank == 1 and grp.members[0].is_pick is True
    assert grp.members[1].is_pick is False
    assert motion_map["b2"] > 0.01


def test_i18n_bilingual_rendering_and_tech_vs_burst_separation() -> None:
    zh = render_reason(REASON_MOTION_BLUR, "zh-CN")
    en = render_reason(REASON_MOTION_BLUR, "en-US")
    assert "拖影" in zh
    assert "motion blur" in en.lower()

    assert is_tech_reject_reason([REASON_OUT_OF_FOCUS]) is True
    assert is_tech_reject_reason([REASON_BURST_NON_PICK]) is False
    assert is_burst_cull_reason([REASON_BURST_NON_PICK]) is True
    assert is_burst_cull_reason([REASON_BURST_NON_PICK, REASON_MOTION_BLUR]) is False
