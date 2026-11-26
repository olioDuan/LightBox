"""T04 unit tests: ingest scanning, atomic pairing, content_fp invariance, ExifTool fallback, and preview cache."""

from __future__ import annotations

import io
import os
import shutil
import stat
import struct
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from lightbox.core import (
    ExifExtractor,
    ExifToolPool,
    compute_content_fp,
    compute_ev100,
    enforce_cache_limit,
    generate_previews,
    load_analysis_gray,
    scan_directory,
)
from lightbox.core.exif import _parse_exiftool_record
from lightbox.paths import get_app_paths


def _make_jpeg_with_exif(
    path: Path,
    *,
    size: tuple[int, int] = (800, 600),
    color: tuple[int, int, int] = (120, 160, 200),
    orientation: int = 1,
    make: str = "SONY",
    model: str = "ILCE-7M4",
    datetime_original: str = "2026:10:01 14:30:00",
    fnumber: tuple[int, int] = (28, 10),
    exposure_time: tuple[int, int] = (1, 250),
    iso: int = 400,
    content_id: str | None = None,
) -> None:
    img = Image.new("RGB", size, color=color)
    exif = Image.Exif()
    exif[0x010F] = make  # Make
    exif[0x0110] = model  # Model
    exif[0x0112] = orientation  # Orientation
    exif[0x0132] = datetime_original  # DateTime
    exif[0x9003] = datetime_original  # DateTimeOriginal (root fallback)
    exif[0x829D] = fnumber[0] / fnumber[1]  # FNumber
    exif[0x829A] = exposure_time[0] / exposure_time[1]  # ExposureTime
    exif[0x8827] = iso  # ISOSpeedRatings
    if content_id is not None:
        exif[0xA420] = content_id  # ImageUniqueID (fallback tag for ContentIdentifier)
    img.save(path, format="JPEG", exif=exif.tobytes())


def _make_synthetic_mov(
    path: Path,
    *,
    duration_s: float,
    creation_unix: float | None = None,
    content_identifier: str | None = None,
) -> None:
    """Build a minimal valid QuickTime MOV (`ftyp` + `moov` -> `mvhd` + optional Apple ContentIdentifier)."""
    timescale = 600
    dur_units = int(round(duration_s * timescale))
    mac_epoch_offset = 2082844800
    mac_created = int(creation_unix + mac_epoch_offset) if creation_unix is not None else 0

    # mvhd v0 atom (100 bytes body)
    mvhd_body = struct.pack(">BBHIIII", 0, 0, 0, mac_created, mac_created, timescale, dur_units)
    mvhd_body += b"\x00" * (100 - len(mvhd_body))
    mvhd_atom = struct.pack(">I4s", 8 + len(mvhd_body), b"mvhd") + mvhd_body

    extra = b""
    if content_identifier is not None:
        meta_body = b"com.apple.quicktime.content.identifier\x00" + content_identifier.encode("ascii")
        extra = struct.pack(">I4s", 8 + len(meta_body), b"udta") + meta_body

    moov_body = mvhd_atom + extra
    moov_atom = struct.pack(">I4s", 8 + len(moov_body), b"moov") + moov_body
    ftyp_body = b"qt  \x00\x00\x02\x00qt  "
    ftyp_atom = struct.pack(">I4s", 8 + len(ftyp_body), b"ftyp") + ftyp_body
    path.write_bytes(ftyp_atom + moov_atom)


def test_content_fp_ignores_mtime_and_detects_head_tail_changes(tmp_path: Path) -> None:
    # 1. Small file (< 128KB)
    small = tmp_path / "small.jpg"
    small.write_bytes(b"A" * 4096)
    fp_before = compute_content_fp(small)
    os.utime(small, (1_600_000_000, 1_600_000_000))
    assert compute_content_fp(small) == fp_before

    # 2. Large file (> 128KB): head 64KB + tail 64KB + size
    large = tmp_path / "large.raw"
    payload = bytearray(b"\x01" * (200 * 1024))
    large.write_bytes(payload)
    fp_large = compute_content_fp(large)
    os.utime(large, (1_500_000_000, 1_500_000_000))
    assert compute_content_fp(large) == fp_large

    # Flip 1 byte in tail -> fingerprint must change
    payload[-1] = 0xFF
    large.write_bytes(payload)
    assert compute_content_fp(large) != fp_large


def test_scan_directory_atomic_pairing_and_filtering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force fallback mode (no ExifTool) to prove graceful degradation
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)

    shoot = tmp_path / "mixed_shoot"
    shoot.mkdir()

    # 1. RAW + JPG + XMP atomic group
    (shoot / "IMG_0001.ARW").write_bytes(b"FAKE_ARW_HEADER" * 100)
    _make_jpeg_with_exif(shoot / "IMG_0001.JPG", datetime_original="2026:10:01 10:00:00")
    (shoot / "IMG_0001.ARW.xmp").write_text("<xmp>sidecar</xmp>", encoding="utf-8")

    # 2. Live Photo paired by Apple ContentIdentifier (even with different stem!)
    live_uuid = "12345678-1234-5678-1234-567812345678"
    _make_jpeg_with_exif(
        shoot / "IMG_0002.JPG",
        make="Apple",
        model="iPhone 16 Pro",
        datetime_original="2026:10:01 10:05:00",
        content_id=live_uuid,
    )
    _make_synthetic_mov(
        shoot / "IMG_0002_LIVE.MOV",
        duration_s=2.8,
        content_identifier=live_uuid,
    )

    # 3. Live Photo fallback pairing: same stem + |Δt| <= 2s + duration <= 5s (no ContentIdentifier)
    _make_jpeg_with_exif(
        shoot / "IMG_0003.JPG",
        datetime_original="2026:10:01 10:10:00",
    )
    # 2026-10-01 10:10:01 UTC -> within 2s tolerance
    from datetime import UTC, datetime

    ts_0003 = datetime(2026, 10, 1, 10, 10, 1, tzinfo=UTC).timestamp()
    _make_synthetic_mov(shoot / "IMG_0003.MOV", duration_s=3.0, creation_unix=ts_0003)

    # 4. Standalone long MOV (> 5s) with same stem -> MOV must be skipped
    _make_jpeg_with_exif(shoot / "IMG_0004.JPG", datetime_original="2026:10:01 10:20:00")
    _make_synthetic_mov(shoot / "IMG_0004.MOV", duration_s=12.5)

    # 5. Ignored items: standalone MP4, hidden macOS files, .DS_Store, Lightbox_Quarantine/
    (shoot / "CLIP_0001.MP4").write_bytes(b"mp4video")
    (shoot / "._IMG_0001.JPG").write_bytes(b"appledouble")
    (shoot / ".DS_Store").write_bytes(b"dsstore")
    q_dir = shoot / "Lightbox_Quarantine" / "old_session"
    q_dir.mkdir(parents=True)
    _make_jpeg_with_exif(q_dir / "IMG_9999.JPG")

    app_paths = get_app_paths(tmp_path / "app_home")
    with ExifExtractor(app_paths=app_paths) as extractor:
        assert extractor.using_exiftool is False
        result = scan_directory(shoot, exif_extractor=extractor)

    by_stem = {g.stem: g for g in result.groups}
    assert set(by_stem.keys()) == {"IMG_0001", "IMG_0002", "IMG_0003", "IMG_0004"}

    # Group 1: RAW + JPG + XMP
    g1_kinds = sorted(a.kind for a in by_stem["IMG_0001"].assets)
    assert g1_kinds == ["jpg", "raw", "sidecar"]
    assert by_stem["IMG_0001"].primary_asset.kind == "raw"

    # Group 2: Live Photo via ContentIdentifier + phone cohort
    g2 = by_stem["IMG_0002"]
    assert sorted(a.kind for a in g2.assets) == ["jpg", "live_mov"]
    assert g2.exif.cohort == "phone"
    assert g2.exif.content_identifier == live_uuid

    # Group 3: Live Photo via stem + time + duration <= 5s fallback
    g3 = by_stem["IMG_0003"]
    assert sorted(a.kind for a in g3.assets) == ["jpg", "live_mov"]

    # Group 4: Standalone long MOV skipped
    g4 = by_stem["IMG_0004"]
    assert [a.kind for a in g4.assets] == ["jpg"]

    # Verify skipped paths include Quarantine, MP4, hidden files, and unpaired MOV
    skipped_names = {p.name for p in result.skipped_paths}
    assert "Lightbox_Quarantine" in skipped_names
    assert "CLIP_0001.MP4" in skipped_names
    assert "._IMG_0001.JPG" in skipped_names
    assert ".DS_Store" in skipped_names
    assert "IMG_0004.MOV" in skipped_names


def test_exif_fallback_normalizes_ev100_and_marks_private_fields_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    jpg = tmp_path / "sunny.jpg"
    _make_jpeg_with_exif(
        jpg,
        fnumber=(80, 10),  # f/8
        exposure_time=(1, 250),  # 1/250s
        iso=100,
    )

    with ExifExtractor(allow_exiftool=False) as extractor:
        ex = extractor.extract(jpg)

    assert ex.backend == "fallback"
    assert ex.aperture == pytest.approx(8.0)
    assert ex.shutter_s == pytest.approx(1 / 250)
    assert ex.iso == 100
    assert ex.ev100 == pytest.approx(14.0, abs=0.05)
    assert ex.light_bucket == "bright"
    assert ex.field_confidence["protected"] == "low"
    assert ex.field_confidence["bracket_group"] == "low"

    # Direct EV100 bucket sanity checks
    _, night_bucket = compute_ev100(2.8, 1 / 15, 3200)
    _, indoor_bucket = compute_ev100(2.8, 1 / 60, 400)
    assert night_bucket == "night"
    assert indoor_bucket == "indoor"


def test_preview_generation_orientation_srgb_and_cache_lru(tmp_path: Path) -> None:
    # Create a 1600x900 JPEG with EXIF Orientation=6 (90 deg CW -> transposed to 900x1600)
    src = tmp_path / "portrait.jpg"
    _make_jpeg_with_exif(src, size=(1600, 900), orientation=6)
    fp = compute_content_fp(src)
    cache_dir = tmp_path / "cache"

    res1 = generate_previews(src, fp, cache_dir)
    assert res1.cache_hit is False
    assert (res1.preview_w, res1.preview_h) == (900, 1600)
    assert res1.is_low_res is True  # max(900, 1600) < 2000

    with Image.open(res1.grid_path) as grid_img:
        assert max(grid_img.size) == 320
        assert grid_img.height > grid_img.width

    with Image.open(res1.analysis_path) as analysis_img:
        assert max(analysis_img.size) == 1024
        # WebP stores grayscale losslessly with identical R==G==B channels
        import numpy as np

        rgb_arr = np.asarray(analysis_img.convert("RGB"))
        assert np.array_equal(rgb_arr[:, :, 0], rgb_arr[:, :, 1])
        assert np.array_equal(rgb_arr[:, :, 1], rgb_arr[:, :, 2])

    gray_arr = load_analysis_gray(res1.analysis_path)
    assert gray_arr.ndim == 2
    assert max(gray_arr.shape) == 1024

    # Second call must hit cache
    res2 = generate_previews(src, fp, cache_dir)
    assert res2.cache_hit is True
    assert (res2.preview_w, res2.preview_h) == (900, 1600)

    # LRU eviction test: create two dated files and enforce tight limit
    lru_dir = tmp_path / "lru_cache"
    lru_dir.mkdir()
    old_file = lru_dir / "old.webp"
    new_file = lru_dir / "new.webp"
    old_file.write_bytes(b"X" * 2000)
    new_file.write_bytes(b"Y" * 2000)
    os.utime(old_file, (1_000_000, 1_000_000))
    os.utime(new_file, (2_000_000, 2_000_000))

    evicted = enforce_cache_limit(lru_dir, max_bytes=2500)
    assert evicted == [old_file.resolve()]
    assert not old_file.exists()
    assert new_file.exists()


def test_exif_t045_digest_fields_and_local_hour_without_tz_shift(tmp_path: Path) -> None:
    # 1. Fallback JPEG with ExposureProgram=3 (Av), Flash=1 (Fired), local hour 23
    jpg = tmp_path / "night_flash.jpg"
    img = Image.new("RGB", (64, 64), color=(10, 20, 30))
    exif = Image.Exif()
    exif[0x9003] = "2026:10:01 23:45:00"
    exif[0x8822] = 3  # ExposureProgram = Aperture Priority (Av)
    exif[0x9209] = 1  # Flash = Fired (bit 0 = 1)
    img.save(jpg, format="JPEG", exif=exif.tobytes())

    with ExifExtractor(allow_exiftool=False) as extractor:
        ex = extractor.extract(jpg)

    assert ex.exposure_program == "Av"
    assert ex.flash_fired is True
    assert ex.capture_local_hour == 23
    assert ex.field_confidence["af_mode"] == "low"

    # 2. ExifTool JSON record with timezone offset (-05:00) must preserve local hour 23 (not UTC 04)
    parsed = _parse_exiftool_record(
        {
            "DateTimeOriginal": "2026:10:01 23:15:00-05:00",
            "ExposureProgram": "Manual",
            "FocusMode": "AF-C",
            "Flash": "Off, Did not fire",
            "FNumber": 2.8,
            "ExposureTime": 0.005,
            "ISO": 1600,
        }
    )
    assert parsed.capture_local_hour == 23
    assert parsed.exposure_program == "M"
    assert parsed.af_mode == "AF-C"
    assert parsed.flash_fired is False


def test_live_photo_fallback_pairs_across_timezone_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review #5 regression: naive local JPEG DateTimeOriginal vs UTC MOV mvhd pairs cleanly."""
    monkeypatch.setattr(shutil, "which", lambda _cmd: None)
    shoot = tmp_path / "iphone_tz"
    shoot.mkdir()

    # Local time 10:10:00 (UTC-4 -> actual UTC is 14:10:00)
    _make_jpeg_with_exif(shoot / "IMG_1001.JPG", datetime_original="2026:10:01 10:10:00")
    ts_utc_plus_4h = datetime(2026, 10, 1, 14, 10, 1, tzinfo=UTC).timestamp()
    _make_synthetic_mov(shoot / "IMG_1001.MOV", duration_s=2.5, creation_unix=ts_utc_plus_4h)

    # Non-timezone offset (5m30s off) must NOT pair
    _make_jpeg_with_exif(shoot / "IMG_1002.JPG", datetime_original="2026:10:01 10:10:00")
    ts_bad_offset = datetime(2026, 10, 1, 10, 15, 30, tzinfo=UTC).timestamp()
    _make_synthetic_mov(shoot / "IMG_1002.MOV", duration_s=2.5, creation_unix=ts_bad_offset)

    with ExifExtractor(allow_exiftool=False) as extractor:
        res = scan_directory(shoot, exif_extractor=extractor)

    by_stem = {g.stem: g for g in res.groups}
    assert sorted(a.kind for a in by_stem["IMG_1001"].assets) == ["jpg", "live_mov"]
    assert [a.kind for a in by_stem["IMG_1002"].assets] == ["jpg"]


def test_raw_embedded_preview_without_orientation_rotates_via_fallback_orientation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review #3 regression: landscape RAW embedded JPEG lacking Orientation tag rotates upright."""
    raw_file = tmp_path / "portrait.ARW"
    raw_file.write_bytes(b"FAKE_RAW")

    # Build a landscape 1600x900 JPEG with NO Orientation tag (simulates PreviewImage / extract_thumb)
    buf = io.BytesIO()
    Image.new("RGB", (1600, 900), color=(40, 80, 120)).save(buf, format="JPEG")
    embedded_jpeg = buf.getvalue()

    class _FakeExifPool:
        def extract_binary_tag(self, _path: Path, tag: str) -> bytes | None:
            return embedded_jpeg if tag == "PreviewImage" else None

        def read_metadata(self, _paths: list[Path]) -> list[dict[str, Any]]:
            return [{"Orientation": 6}]

    cache_dir = tmp_path / "cache"
    fp = compute_content_fp(raw_file)

    # Case A: explicit exif_orientation=6
    res = generate_previews(
        raw_file,
        fp,
        cache_dir,
        exif_pool=_FakeExifPool(),  # type: ignore[arg-type]
        exif_orientation=6,
    )
    assert (res.preview_w, res.preview_h) == (900, 1600)
    gray = load_analysis_gray(res.analysis_path)
    assert gray.shape[0] > gray.shape[1]  # Upright portrait analysis image for face detection

    # Case B: omitted exif_orientation -> auto-probes RAW orientation via exif_pool
    cache_dir2 = tmp_path / "cache2"
    res2 = generate_previews(
        raw_file,
        fp,
        cache_dir2,
        exif_pool=_FakeExifPool(),  # type: ignore[arg-type]
    )
    assert (res2.preview_w, res2.preview_h) == (900, 1600)


def test_exiftool_pool_timeout_and_batch_chunking(tmp_path: Path) -> None:
    """Review #4 & T04.5 regression: hung ExifTool times out cleanly; extract_batch chunks calls."""
    # 1. Fake hanging exiftool script that ignores stdin and sleeps
    hanging_script = tmp_path / "fake_hang_exiftool"
    hanging_script.write_text(
        f"#!{sys.executable}\nimport time\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    hanging_script.chmod(hanging_script.stat().st_mode | stat.S_IXUSR)

    with ExifToolPool(hanging_script, timeout_s=0.25) as pool:
        with pytest.raises(TimeoutError):
            pool.execute(["-ver"])

    # 2. Verify ExifExtractor.extract_batch splits 250 files into chunks of 150 + 100
    chunk_sizes: list[int] = []

    class _CountingPool:
        def read_metadata(self, paths: list[Path]) -> list[dict[str, Any]]:
            chunk_sizes.append(len(paths))
            return [
                {"SourceFile": str(p.resolve()), "FNumber": 2.8, "ISO": 100}
                for p in paths
            ]

        def close(self) -> None:
            pass

    extractor = ExifExtractor(allow_exiftool=False)
    extractor._pool = _CountingPool()  # type: ignore[assignment]
    dummy_paths = [tmp_path / f"img_{i:04d}.jpg" for i in range(250)]
    out = extractor.extract_batch(dummy_paths, batch_size=150)

    assert chunk_sizes == [150, 100]
    assert len(out) == 250
    assert all(v.backend == "exiftool" for v in out.values())
