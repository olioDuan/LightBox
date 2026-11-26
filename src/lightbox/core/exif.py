"""ExifTool persistent process pool, graceful fallback, EXIF normalization & EV100 (SPEC §4.1–4.2)."""

from __future__ import annotations

import io
import json
import math
import os
import re
import select
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import exifread
from PIL import ExifTags, Image

from lightbox.paths import AppPaths, get_app_paths

_READY_SENTINEL = "{ready}"
DEFAULT_EXIF_BATCH_SIZE = 150

# Cross-device smartphone identification tokens (no single-brand hardcoding; unified lookup table)
_PHONE_MODEL_TOKENS = (
    "iphone",
    "pixel",
    "galaxy",
    "sm-s",
    "sm-g",
    "sm-a",
    "sm-f",
    "xiaomi",
    "redmi",
    "poco",
    "oneplus",
    "oppo",
    "vivo",
    "huawei",
    "honor",
    "realme",
    "nothing phone",
    "moto g",
    "moto edge",
)
_PHONE_LENS_TOKENS = ("back camera", "front camera", "triple camera", "dual camera", "wide camera")
_EXIF_DATE_FORMATS = (
    "%Y:%m:%d %H:%M:%S%z",
    "%Y:%m:%d %H:%M:%S.%f%z",
    "%Y:%m:%d %H:%M:%S",
    "%Y:%m:%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
)
_LOCAL_HOUR_RE = re.compile(r"^\d{4}[:\-]\d{2}[:\-]\d{2}[ T](\d{2}):\d{2}:\d{2}")

_EXPOSURE_PROGRAM_CODES: dict[int, str] = {
    1: "M",
    2: "P",
    3: "Av",
    4: "Tv",
    5: "Creative",
    6: "Sports",
    7: "Portrait",
    8: "Landscape",
    9: "Bulb",
}

_VENDOR_PRIVATE_FIELDS = (
    "protected",
    "rating",
    "drive_mode",
    "bracket_group",
    "camera_serial",
    "af_mode",
)


@dataclass(frozen=True, slots=True)
class NormalizedExif:
    """Normalized camera/image metadata shared across all camera and phone brands."""

    aperture: float | None = None
    shutter_s: float | None = None
    iso: int | None = None
    focal_eq: float | None = None
    ev_comp: float | None = None
    ev100: float | None = None
    light_bucket: str | None = None  # 'bright' | 'indoor' | 'night'
    exposure_program: str | None = None  # 'M' | 'Av' | 'Tv' | 'P' | 'Sports' | ...
    af_mode: str | None = None  # 'AF-C' | 'AF-S' | 'AF-A' | 'MF' | ...
    flash_fired: bool | None = None
    capture_local_hour: int | None = None  # 0..23 from raw camera local timestamp
    capture_time: float | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    camera_serial: str | None = None
    lens_model: str | None = None
    orientation: int = 1
    drive_mode: str | None = None
    bracket_group: str | None = None
    rating: int | None = None
    protected: bool | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    content_identifier: str | None = None
    duration_s: float | None = None
    cohort: str = "camera"  # 'camera' | 'phone'
    backend: str = "fallback"  # 'exiftool' | 'fallback'
    field_confidence: dict[str, str] = field(default_factory=dict)


def compute_ev100(
    aperture: float | None,
    shutter_s: float | None,
    iso: int | None,
) -> tuple[float | None, str | None]:
    """Compute EV100 = log2(N^2 / t) - log2(ISO / 100) and map to light bucket (SPEC §4.2)."""
    if (
        aperture is None
        or shutter_s is None
        or iso is None
        or aperture <= 0.0
        or shutter_s <= 0.0
        or iso <= 0
    ):
        return None, None
    ev100 = math.log2((aperture * aperture) / shutter_s) - math.log2(iso / 100.0)
    if ev100 >= 10.0:
        bucket = "bright"
    elif ev100 >= 5.0:
        bucket = "indoor"
    else:
        bucket = "night"
    return round(ev100, 3), bucket


def infer_cohort(
    camera_make: str | None,
    camera_model: str | None,
    lens_model: str | None,
) -> str:
    """Classify capture device into `'phone'` or `'camera'` via unified metadata table."""
    combined_model = f"{camera_make or ''} {camera_model or ''}".lower()
    if any(token in combined_model for token in _PHONE_MODEL_TOKENS):
        return "phone"
    lens_lower = (lens_model or "").lower()
    if any(token in lens_lower for token in _PHONE_LENS_TOKENS):
        return "phone"
    return "camera"


def find_exiftool(app_paths: AppPaths | None = None) -> Path | None:
    """Locate ExifTool in system `PATH` first, then `<APP_DATA_DIR>/bin/` (SPEC §4.1(1))."""
    on_path = shutil.which("exiftool")
    if on_path:
        return Path(on_path)

    paths = app_paths or get_app_paths()
    candidates = ("exiftool.exe", "exiftool") if sys.platform == "win32" else ("exiftool",)
    for name in candidates:
        candidate = paths.bin_dir / name
        if candidate.is_file():
            return candidate
    return None


class ExifToolPool:
    """Persistent `exiftool -stay_open True` process wrapper with stderr isolation & deadline timeout."""

    def __init__(self, executable: Path | str, *, timeout_s: float = 30.0) -> None:
        self.executable = str(executable)
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None

    def _ensure_started(self) -> subprocess.Popen[bytes]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        self._proc = subprocess.Popen(
            [
                self.executable,
                "-stay_open",
                "True",
                "-@",
                "-",
                "-common_args",
                "-charset",
                "filename=UTF8",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Discard stderr so unread error output can never fill the OS pipe buffer & deadlock.
            stderr=subprocess.DEVNULL,
        )
        return self._proc

    def _kill_and_reset_unlocked(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2.0)
        except Exception:
            pass
        finally:
            for stream in (proc.stdin, proc.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass

    def _read_until_sentinel_unlocked(self, proc: subprocess.Popen[bytes]) -> bytes:
        assert proc.stdout is not None
        sentinel = _READY_SENTINEL.encode("ascii")
        deadline = time.monotonic() + self.timeout_s
        fd = proc.stdout.fileno()
        buf = bytearray()

        while True:
            # Check whether a complete sentinel line is already in buffer
            lines = buf.splitlines(keepends=True)
            for idx, line in enumerate(lines):
                if line.strip() == sentinel:
                    return b"".join(lines[:idx])

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill_and_reset_unlocked()
                raise TimeoutError(
                    f"ExifTool did not respond within {self.timeout_s:.1f}s timeout."
                )

            if sys.platform != "win32":
                ready, _, _ = select.select([fd], [], [], remaining)
                if not ready:
                    self._kill_and_reset_unlocked()
                    raise TimeoutError(
                        f"ExifTool did not respond within {self.timeout_s:.1f}s timeout."
                    )
                chunk = os.read(fd, 65536)
            else:
                chunk = proc.stdout.readline()

            if not chunk:
                self._kill_and_reset_unlocked()
                raise RuntimeError("ExifTool process terminated unexpectedly (EOF).")
            buf.extend(chunk)

    def execute(self, args: Sequence[str]) -> bytes:
        """Send one batch of arguments to the stay_open process and read until `{ready}`."""
        with self._lock:
            proc = self._ensure_started()
            assert proc.stdin is not None
            payload = "\n".join([*args, "-execute"]) + "\n"
            try:
                proc.stdin.write(payload.encode("utf-8"))
                proc.stdin.flush()
            except OSError as exc:
                self._kill_and_reset_unlocked()
                raise RuntimeError(f"Failed writing to ExifTool stdin: {exc}") from exc
            return self._read_until_sentinel_unlocked(proc)

    def read_metadata(self, paths: Sequence[Path]) -> list[dict[str, Any]]:
        """Read normalized numeric JSON metadata for one or more media files."""
        if not paths:
            return []
        raw = self.execute(["-json", "-n", "-s", *(str(p) for p in paths)])
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return []
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []

    def extract_binary_tag(self, path: Path, tag: str) -> bytes | None:
        """Extract binary embedded image payload (e.g. `JpgFromRaw` / `PreviewImage`)."""
        data = self.execute(["-b", f"-{tag}", str(path)])
        return data if data else None

    def close(self) -> None:
        with self._lock:
            if self._proc is None:
                return
            proc = self._proc
            self._proc = None
            try:
                if proc.poll() is None and proc.stdin is not None:
                    proc.stdin.write(b"-stay_open\nFalse\n")
                    proc.stdin.flush()
                    proc.wait(timeout=3.0)
            except Exception:
                proc.kill()
            finally:
                for stream in (proc.stdin, proc.stdout):
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass

    def __enter__(self) -> ExifToolPool:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _parse_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "num") and hasattr(value, "den"):
        den = float(value.den)
        return float(value.num) / den if den != 0.0 else None
    text = str(value).strip()
    if not text:
        return None
    if "/" in text:
        parts = text.split("/", 1)
        try:
            num = float(parts[0].strip())
            den = float(parts[1].strip())
            return num / den if den != 0.0 else None
        except ValueError:
            return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def _parse_int(value: Any) -> int | None:
    flt = _parse_float(value)
    return int(round(flt)) if flt is not None else None


def _extract_local_hour(value: Any) -> int | None:
    """Extract camera local hour (0..23) directly from raw EXIF timestamp string without TZ shift."""
    if value is None or isinstance(value, (int, float, bool)):
        return None
    text = str(value).strip()
    if not text or text.startswith("0000:00:00"):
        return None
    match = _LOCAL_HOUR_RE.match(text)
    if not match:
        return None
    hour = int(match.group(1))
    return hour if 0 <= hour <= 23 else None


def _normalize_exposure_program(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return _EXPOSURE_PROGRAM_CODES.get(value)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return _EXPOSURE_PROGRAM_CODES.get(int(text))
    lowered = text.lower()
    if lowered in {"not defined", "unknown", "0"}:
        return None
    if "aperture" in lowered or lowered in {"av", "a"}:
        return "Av"
    if "shutter" in lowered or lowered in {"tv", "s"}:
        return "Tv"
    if "manual" in lowered or lowered == "m":
        return "M"
    if "program" in lowered or lowered == "p":
        return "P"
    if "sport" in lowered or "action" in lowered:
        return "Sports"
    if "portrait" in lowered:
        return "Portrait"
    if "landscape" in lowered:
        return "Landscape"
    if "creative" in lowered:
        return "Creative"
    if "bulb" in lowered:
        return "Bulb"
    return text


def _normalize_af_mode(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if any(tok in lowered for tok in ("af-c", "ai servo", "continuous", "af_c")):
        return "AF-C"
    if any(tok in lowered for tok in ("af-s", "one-shot", "single", "af_s")):
        return "AF-S"
    if any(tok in lowered for tok in ("af-a", "ai focus", "auto", "af_a")):
        return "AF-A"
    if lowered == "mf" or "manual" in lowered:
        return "MF"
    return text


def _normalize_flash_fired(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value & 0x1)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return bool(int(text) & 0x1)
    lowered = text.lower()
    if "did not fire" in lowered or "no flash" in lowered or lowered in {"off", "false"}:
        return False
    if "fired" in lowered or lowered in {"on", "true"}:
        return True
    return None


def _parse_datetime(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.startswith("0000:00:00"):
        return None
    for fmt in _EXIF_DATE_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def _parse_exiftool_record(raw: dict[str, Any]) -> NormalizedExif:
    aperture = _parse_float(raw.get("FNumber") or raw.get("Aperture"))
    shutter_s = _parse_float(raw.get("ExposureTime") or raw.get("ShutterSpeed"))
    iso = _parse_int(raw.get("ISO") or raw.get("ISOSpeedRatings"))
    focal_eq = _parse_float(raw.get("FocalLengthIn35mmFormat") or raw.get("FocalLength"))
    ev_comp = _parse_float(raw.get("ExposureCompensation"))
    ev100, light_bucket = compute_ev100(aperture, shutter_s, iso)

    raw_dt = raw.get("DateTimeOriginal") or raw.get("CreateDate") or raw.get("MediaCreateDate")
    capture_time = _parse_datetime(raw_dt)
    capture_local_hour = _extract_local_hour(raw_dt)

    exposure_program = _normalize_exposure_program(raw.get("ExposureProgram"))
    af_mode = _normalize_af_mode(
        raw.get("FocusMode") or raw.get("AFMode") or raw.get("FocusSetting")
    )
    flash_fired = _normalize_flash_fired(raw.get("Flash"))

    make = str(raw["Make"]).strip() if raw.get("Make") else None
    model = str(raw["Model"]).strip() if raw.get("Model") else None
    serial = (
        str(raw.get("SerialNumber") or raw.get("InternalSerialNumber") or "").strip() or None
    )
    lens = str(raw.get("LensModel") or raw.get("LensID") or raw.get("Lens") or "").strip() or None
    orientation = _parse_int(raw.get("Orientation")) or 1
    drive_mode = str(raw["DriveMode"]).strip() if raw.get("DriveMode") is not None else None
    bracket_group = (
        str(raw.get("AutoBracketing") or raw.get("BracketMode") or raw.get("SequenceNumber") or "")
        .strip()
        or None
    )
    rating = _parse_int(raw.get("Rating"))
    protected_raw = raw.get("Protect") or raw.get("Locked")
    protected = bool(protected_raw) if protected_raw is not None else None
    gps_lat = _parse_float(raw.get("GPSLatitude"))
    gps_lon = _parse_float(raw.get("GPSLongitude"))
    content_id = (
        str(raw.get("ContentIdentifier") or raw.get("MediaGroupUUID") or "").strip() or None
    )
    duration_s = _parse_float(raw.get("Duration") or raw.get("TrackDuration"))

    confidence = {f: "high" for f in _VENDOR_PRIVATE_FIELDS}
    return NormalizedExif(
        aperture=aperture,
        shutter_s=shutter_s,
        iso=iso,
        focal_eq=focal_eq,
        ev_comp=ev_comp,
        ev100=ev100,
        light_bucket=light_bucket,
        exposure_program=exposure_program,
        af_mode=af_mode,
        flash_fired=flash_fired,
        capture_local_hour=capture_local_hour,
        capture_time=capture_time,
        camera_make=make,
        camera_model=model,
        camera_serial=serial,
        lens_model=lens,
        orientation=orientation,
        drive_mode=drive_mode,
        bracket_group=bracket_group,
        rating=rating,
        protected=protected,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        content_identifier=content_id,
        duration_s=duration_s,
        cohort=infer_cohort(make, model, lens),
        backend="exiftool",
        field_confidence=confidence,
    )


def _extract_mov_fallback(path: Path) -> NormalizedExif:
    """Pure-Python QuickTime/MOV parser for duration, creation time, and Apple ContentIdentifier."""
    duration_s: float | None = None
    capture_time: float | None = None
    content_id: str | None = None

    try:
        data = path.read_bytes()
    except OSError:
        data = b""

    def _walk_atoms(buf: bytes, start: int, end: int) -> None:
        nonlocal duration_s, capture_time
        pos = start
        while pos + 8 <= end:
            size, atom_type = struct.unpack(">I4s", buf[pos : pos + 8])
            header_size = 8
            if size == 1 and pos + 16 <= end:
                size = struct.unpack(">Q", buf[pos + 8 : pos + 16])[0]
                header_size = 16
            elif size == 0:
                size = end - pos
            if size < header_size or pos + size > end:
                break
            body_start = pos + header_size
            body_end = pos + size

            if atom_type == b"moov":
                _walk_atoms(buf, body_start, body_end)
            elif atom_type == b"mvhd" and body_end - body_start >= 20:
                version = buf[body_start]
                if version == 0 and body_end - body_start >= 20:
                    creation_mac, _mod, timescale, dur = struct.unpack(
                        ">IIII", buf[body_start + 4 : body_start + 20]
                    )
                    if timescale > 0:
                        duration_s = dur / float(timescale)
                    if creation_mac > 2082844800:
                        capture_time = float(creation_mac - 2082844800)
                elif version == 1 and body_end - body_start >= 32:
                    creation_mac, _mod, timescale, dur = struct.unpack(
                        ">QQIQ", buf[body_start + 4 : body_start + 32]
                    )
                    if timescale > 0:
                        duration_s = dur / float(timescale)
                    if creation_mac > 2082844800:
                        capture_time = float(creation_mac - 2082844800)
            pos += size

    _walk_atoms(data, 0, len(data))

    marker = b"com.apple.quicktime.content.identifier"
    idx = data.find(marker)
    if idx != -1:
        window = data[idx : idx + 2048]
        uuid_match = re.search(
            rb"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}",
            window,
        )
        if uuid_match:
            content_id = uuid_match.group(0).decode("ascii")

    confidence = {f: "low" for f in _VENDOR_PRIVATE_FIELDS}
    return NormalizedExif(
        capture_time=capture_time,
        content_identifier=content_id,
        duration_s=duration_s,
        cohort="phone" if content_id else "camera",
        backend="fallback",
        field_confidence=confidence,
    )


def _extract_image_bytes_exif(raw_bytes: bytes) -> dict[str, Any]:
    """Read standard EXIF tags from image bytes using Pillow + exifread."""
    parsed: dict[str, Any] = {}

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            exif_obj = img.getexif()
            for tag_id, value in exif_obj.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                parsed[tag_name] = value
            ifd = exif_obj.get_ifd(ExifTags.IFD.Exif)
            for tag_id, value in ifd.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                parsed[tag_name] = value
    except Exception:
        pass

    try:
        tags = exifread.process_file(io.BytesIO(raw_bytes), details=False)
        mapping = {
            "EXIF FNumber": "FNumber",
            "EXIF ExposureTime": "ExposureTime",
            "EXIF ISOSpeedRatings": "ISOSpeedRatings",
            "EXIF FocalLengthIn35mmFilm": "FocalLengthIn35mmFilm",
            "EXIF FocalLength": "FocalLength",
            "EXIF ExposureBiasValue": "ExposureBiasValue",
            "EXIF ExposureProgram": "ExposureProgram",
            "EXIF Flash": "Flash",
            "EXIF DateTimeOriginal": "DateTimeOriginal",
            "Image DateTime": "DateTime",
            "Image Make": "Make",
            "Image Model": "Model",
            "EXIF LensModel": "LensModel",
            "Image Orientation": "Orientation",
        }
        for er_key, out_key in mapping.items():
            if out_key not in parsed and er_key in tags:
                val = tags[er_key].values
                parsed[out_key] = val[0] if isinstance(val, list) and val else str(tags[er_key])
    except Exception:
        pass

    return parsed


def _extract_fallback(path: Path) -> NormalizedExif:
    """Fallback metadata extraction via `rawpy.extract_thumb()` + `Pillow`/`exifread` (SPEC §4.1(1))."""
    ext = path.suffix.lower()
    if ext == ".mov":
        return _extract_mov_fallback(path)

    parsed: dict[str, Any] = {}
    if ext in {
        ".arw",
        ".cr3",
        ".cr2",
        ".nef",
        ".raf",
        ".rw2",
        ".orf",
        ".dng",
        ".pef",
        ".srw",
        ".iiq",
        ".3fr",
    }:
        try:
            import rawpy

            with rawpy.imread(str(path)) as raw:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG and isinstance(
                    thumb.data, (bytes, bytearray)
                ):
                    parsed.update(_extract_image_bytes_exif(bytes(thumb.data)))
        except Exception:
            pass
        try:
            with path.open("rb") as fh:
                header_bytes = fh.read(512 * 1024)
            header_parsed = _extract_image_bytes_exif(header_bytes)
            for k, v in header_parsed.items():
                parsed.setdefault(k, v)
        except Exception:
            pass
    else:
        try:
            parsed.update(_extract_image_bytes_exif(path.read_bytes()))
        except Exception:
            pass

    aperture = _parse_float(parsed.get("FNumber"))
    shutter_s = _parse_float(parsed.get("ExposureTime"))
    iso = _parse_int(parsed.get("ISOSpeedRatings") or parsed.get("PhotographicSensitivity"))
    focal_eq = _parse_float(parsed.get("FocalLengthIn35mmFilm") or parsed.get("FocalLength"))
    ev_comp = _parse_float(parsed.get("ExposureBiasValue"))
    ev100, light_bucket = compute_ev100(aperture, shutter_s, iso)

    raw_dt = parsed.get("DateTimeOriginal") or parsed.get("DateTime")
    capture_time = _parse_datetime(raw_dt)
    capture_local_hour = _extract_local_hour(raw_dt)

    exposure_program = _normalize_exposure_program(parsed.get("ExposureProgram"))
    flash_fired = _normalize_flash_fired(parsed.get("Flash"))

    make = str(parsed["Make"]).strip() if parsed.get("Make") else None
    model = str(parsed["Model"]).strip() if parsed.get("Model") else None
    lens = str(parsed["LensModel"]).strip() if parsed.get("LensModel") else None
    orientation = _parse_int(parsed.get("Orientation")) or 1

    content_id: str | None = None
    for key in ("ContentIdentifier", "ImageUniqueID"):
        if parsed.get(key):
            content_id = str(parsed[key]).strip()
            break

    # Vendor-private fields cannot be reliably read without ExifTool -> mark confidence="low"
    confidence = {f: "low" for f in _VENDOR_PRIVATE_FIELDS}
    return NormalizedExif(
        aperture=aperture,
        shutter_s=shutter_s,
        iso=iso,
        focal_eq=focal_eq,
        ev_comp=ev_comp,
        ev100=ev100,
        light_bucket=light_bucket,
        exposure_program=exposure_program,
        af_mode=None,
        flash_fired=flash_fired,
        capture_local_hour=capture_local_hour,
        capture_time=capture_time,
        camera_make=make,
        camera_model=model,
        camera_serial=None,
        lens_model=lens,
        orientation=orientation,
        drive_mode=None,
        bracket_group=None,
        rating=None,
        protected=None,
        gps_lat=None,
        gps_lon=None,
        content_identifier=content_id,
        duration_s=None,
        cohort=infer_cohort(make, model, lens),
        backend="fallback",
        field_confidence=confidence,
    )


class ExifExtractor:
    """Unified EXIF extractor using chunked `ExifToolPool` when available, with automatic fallback."""

    def __init__(
        self,
        *,
        app_paths: AppPaths | None = None,
        exiftool_path: Path | str | None = None,
        allow_exiftool: bool = True,
        timeout_s: float = 30.0,
    ) -> None:
        self._pool: ExifToolPool | None = None
        if allow_exiftool:
            resolved_bin = (
                Path(exiftool_path)
                if exiftool_path is not None
                else find_exiftool(app_paths=app_paths)
            )
            if resolved_bin is not None and resolved_bin.is_file():
                self._pool = ExifToolPool(resolved_bin, timeout_s=timeout_s)

    @property
    def using_exiftool(self) -> bool:
        return self._pool is not None

    @property
    def pool(self) -> ExifToolPool | None:
        return self._pool

    def extract(self, path: Path) -> NormalizedExif:
        if self._pool is not None:
            try:
                records = self._pool.read_metadata([path])
                if records:
                    return _parse_exiftool_record(records[0])
            except Exception:
                pass
        return _extract_fallback(path)

    def extract_batch(
        self,
        paths: Sequence[Path],
        *,
        batch_size: int = DEFAULT_EXIF_BATCH_SIZE,
    ) -> dict[Path, NormalizedExif]:
        """Extract EXIF metadata in chunks (`batch_size`, default 150) to scale to 1000+ photos."""
        results: dict[Path, NormalizedExif] = {}
        if not paths:
            return results
        chunk_size = max(1, batch_size)

        if self._pool is not None:
            for start in range(0, len(paths), chunk_size):
                chunk = paths[start : start + chunk_size]
                try:
                    records = self._pool.read_metadata(chunk)
                    by_source: dict[Path, dict[str, Any]] = {}
                    for rec in records:
                        src = rec.get("SourceFile")
                        if src:
                            by_source[Path(str(src)).resolve()] = rec
                    for p in chunk:
                        resolved = p.resolve()
                        if resolved in by_source:
                            results[p] = _parse_exiftool_record(by_source[resolved])
                except Exception:
                    # Only this chunk falls back cleanly; remaining chunks continue normally
                    continue

        for p in paths:
            if p not in results:
                results[p] = _extract_fallback(p)
        return results

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()

    def __enter__(self) -> ExifExtractor:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "DEFAULT_EXIF_BATCH_SIZE",
    "ExifExtractor",
    "ExifToolPool",
    "NormalizedExif",
    "compute_ev100",
    "find_exiftool",
    "infer_cohort",
]
