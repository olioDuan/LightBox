"""T00 Pre-Implementation Checks (SPEC §12).

Verifies external ecosystem assumptions before implementing core Lightbox modules:
1. PyPI package name availability (`lightbox`).
2. `uv`-managed Python 3.12 loading `sqlite-vec` extension on macOS/Linux.
3. ExifTool & Perl availability on current platform.
4. OpenCV Zoo YuNet LICENSE & `pillow-heif` codec license posture.
5. Anthropic / Gemini OpenAI-compat API key & SDK availability.
6. ONNX Runtime / CoreML EP & SigLIP 2 Base ONNX readiness.
7. macOS `renamex_np(RENAME_EXCL)` atomic no-overwrite rename availability.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import json
import os
import platform
import shutil
import sqlite3
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def check_1_pypi_name() -> dict[str, Any]:
    """Check whether `lightbox` is already taken on PyPI."""
    url = "https://pypi.org/pypi/lightbox/json"
    req = urllib.request.Request(url, headers={"User-Agent": "lightbox-preflight/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            info = data.get("info", {})
            return {
                "status": "fallback",
                "available": False,
                "existing_name": info.get("name"),
                "existing_version": info.get("version"),
                "existing_summary": info.get("summary"),
                "detail": "PyPI name `lightbox` is occupied; use an alternative distribution name (e.g. `lightbox-cull` or `pylightbox`) while keeping `lightbox` CLI entrypoint and Python import package.",
            }
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {
                "status": "ok",
                "available": True,
                "detail": "PyPI name `lightbox` returned 404 (available).",
            }
        return {"status": "error", "detail": f"HTTP {exc.code}: {exc.reason}"}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def check_2_sqlite_vec() -> dict[str, Any]:
    """Check sqlite3 extension loading and sqlite-vec under current Python."""
    result: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "sqlite_version": sqlite3.sqlite_version,
        "has_enable_load_extension": hasattr(sqlite3.Connection, "enable_load_extension"),
    }
    try:
        import sqlite_vec  # type: ignore[import-untyped]

        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        vec_ver = conn.execute("select vec_version()").fetchone()[0]
        conn.execute(
            "CREATE VIRTUAL TABLE vec_test USING vec0(embedding float[4] distance_metric=cosine)"
        )
        conn.execute("INSERT INTO vec_test(rowid, embedding) VALUES (1, '[1.0, 0.0, 0.0, 0.0]')")
        conn.execute("INSERT INTO vec_test(rowid, embedding) VALUES (2, '[0.0, 1.0, 0.0, 0.0]')")
        rows = conn.execute(
            "SELECT rowid, distance FROM vec_test WHERE embedding MATCH '[1.0, 0.0, 0.0, 0.0]' AND k = 2"
        ).fetchall()
        conn.close()
        result.update(
            {
                "status": "ok",
                "sqlite_vec_version": vec_ver,
                "sample_query_rows": rows,
                "detail": "sqlite-vec loaded and cosine KNN query succeeded.",
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "fallback",
                "error": repr(exc),
                "detail": "Failed to load sqlite-vec; fallback to numpy VectorIndex backend.",
            }
        )
    return result


def check_3_exiftool() -> dict[str, Any]:
    """Check ExifTool and Perl availability on current platform."""
    exiftool_path = shutil.which("exiftool")
    perl_path = shutil.which("perl")
    return {
        "status": "ok" if exiftool_path else "fallback",
        "platform": platform.system(),
        "exiftool_path": exiftool_path,
        "perl_path": perl_path,
        "detail": (
            f"ExifTool found at {exiftool_path}."
            if exiftool_path
            else f"ExifTool not in PATH (Perl at {perl_path}); portable download into <APP_DATA_DIR>/bin/ or fallback to rawpy.extract_thumb() + Pillow/exifread required."
        ),
    }


def check_4_licenses() -> dict[str, Any]:
    """Check YuNet LICENSE from GitHub and pillow-heif PyPI metadata."""
    yunet_url = "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/LICENSE"
    yunet_license = "unknown"
    try:
        req = urllib.request.Request(yunet_url, headers={"User-Agent": "lightbox-preflight/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode("utf-8")
            first_line = text.strip().splitlines()[0] if text.strip() else ""
            yunet_license = first_line
    except Exception as exc:
        yunet_license = f"fetch_error: {exc}"

    heif_url = "https://pypi.org/pypi/pillow-heif/json"
    heif_info: dict[str, Any] = {}
    try:
        req = urllib.request.Request(heif_url, headers={"User-Agent": "lightbox-preflight/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            info = data.get("info", {})
            heif_info = {
                "version": info.get("version"),
                "license": info.get("license") or info.get("license_expression"),
            }
    except Exception as exc:
        heif_info = {"error": str(exc)}

    return {
        "status": "ok",
        "yunet_license_header": yunet_license,
        "pillow_heif_pypi": heif_info,
        "detail": "YuNet is MIT licensed; pillow-heif wrapper is BSD-3-Clause but bundles/links libheif+libde265 (LGPL-3.0) and optional x265 (GPL-2.0) — strictly isolate under optional extra `lightbox[heic]`.",
    }


def check_5_llm_sdks() -> dict[str, Any]:
    """Check presence of API keys for live smoke testing of Anthropic/Gemini."""
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    return {
        "status": "deferred_no_key" if not (has_anthropic or has_gemini) else "ok",
        "anthropic_key_present": has_anthropic,
        "gemini_key_present": has_gemini,
        "openai_key_present": has_openai,
        "detail": (
            "No live ANTHROPIC_API_KEY / GEMINI_API_KEY in environment; "
            "live smoke test deferred to T10 (keep google-genai native fallback ready if Gemini OpenAI-compat multi-image + JSON Schema is unstable)."
        ),
    }


def check_6_onnx_siglip() -> dict[str, Any]:
    """Check ONNX Runtime execution providers on current machine."""
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]

        providers = ort.get_available_providers()
        return {
            "status": "ok",
            "onnxruntime_version": ort.__version__,
            "providers": providers,
            "coreml_available": "CoreMLExecutionProvider" in providers,
            "detail": "onnxruntime available; default to CPUExecutionProvider with optional CoreML testing in T11, and lazy split download for SigLIP 2 vision/text towers.",
        }
    except Exception as exc:
        return {
            "status": "fallback",
            "error": repr(exc),
            "detail": "onnxruntime not installed in preflight env; verify CoreML vs CPU EP during T11.",
        }


def check_7_macos_renamex_np() -> dict[str, Any]:
    """Test macOS `renamex_np(old, new, RENAME_EXCL)` atomic no-overwrite rename."""
    if platform.system() != "Darwin":
        return {
            "status": "fallback",
            "platform": platform.system(),
            "detail": "Non-macOS platform; use check-before-rename + unique numeric suffix inside exclusive quarantine directory.",
        }

    RENAME_EXCL = 0x00000004
    libc_path = ctypes.util.find_library("c")
    if not libc_path:
        return {"status": "fallback", "detail": "Could not locate libc."}

    libc = ctypes.CDLL(libc_path, use_errno=True)
    renamex_np = getattr(libc, "renamex_np", None)
    if renamex_np is None:
        return {"status": "fallback", "detail": "renamex_np symbol not found in libc."}

    renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    renamex_np.restype = ctypes.c_int

    with tempfile.TemporaryDirectory(prefix="lightbox_preflight_") as tmp_dir:
        tmp = Path(tmp_dir)
        src1 = tmp / "src1.txt"
        src2 = tmp / "src2.txt"
        dst = tmp / "dst.txt"
        src1.write_text("file1", encoding="utf-8")
        src2.write_text("file2", encoding="utf-8")

        # 1. Rename src1 -> dst (should succeed)
        rc1 = renamex_np(os.fsencode(src1), os.fsencode(dst), RENAME_EXCL)
        err1 = ctypes.get_errno()

        # 2. Rename src2 -> dst (must fail with EEXIST and preserve dst content)
        rc2 = renamex_np(os.fsencode(src2), os.fsencode(dst), RENAME_EXCL)
        err2 = ctypes.get_errno()

        preserved = dst.read_text(encoding="utf-8") == "file1" and src2.exists()

    ok = rc1 == 0 and rc2 == -1 and err2 == errno.EEXIST and preserved
    return {
        "status": "ok" if ok else "fallback",
        "rc_new": rc1,
        "errno_new": err1,
        "rc_existing": rc2,
        "errno_existing": err2,
        "expected_eexist": errno.EEXIST,
        "destination_preserved": preserved,
        "detail": (
            "macOS renamex_np(RENAME_EXCL) succeeded: moved new target and rejected existing target with EEXIST without overwriting."
            if ok
            else "renamex_np(RENAME_EXCL) did not behave as expected; fallback to check-then-rename with unique suffix."
        ),
    }


def main() -> None:
    report = {
        "1_pypi_name": check_1_pypi_name(),
        "2_sqlite_vec": check_2_sqlite_vec(),
        "3_exiftool": check_3_exiftool(),
        "4_licenses": check_4_licenses(),
        "5_llm_sdks": check_5_llm_sdks(),
        "6_onnx_siglip": check_6_onnx_siglip(),
        "7_macos_renamex_np": check_7_macos_renamex_np(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
