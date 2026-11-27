"""Outbound image privacy sanitization and `max_pixels` scaling (SPEC §6.2–6.3, §10.2(5))."""

from __future__ import annotations

import io
import math
import os
import re
from collections.abc import Sequence
from typing import Literal

from PIL import Image, ImageOps

from lightbox.models.secrets import PROVIDER_ENV_VARS
from lightbox.models.types import ModelCallRequest

_KNOWN_ABS_ROOT_RE = re.compile(
    r"^(?:/(?:Users|home|Volumes|mnt|media|tmp|private|var|opt|root|data|srv|sessions)/.+|[A-Za-z]:\\.+)"
)
_ABS_PATH_START_RE = re.compile(
    r"""(?<![A-Za-z0-9_.:/-])(?:/[^\s"'`)\]},;，。！？、：<>]|[A-Za-z]:\\[^\s"'`)\]},;，。！？、：<>])"""
)
_PATH_HARD_STOP_RE = re.compile(r"""[\r\n"'`)\]},;<>，。！？、：\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]""")
_API_KEY_PREFIX_RE = re.compile(
    r"\b(?:sk-ant-[A-Za-z0-9_\-]{12,}|sk-proj-[A-Za-z0-9_\-]{12,}|sk-[A-Za-z0-9]{20,}|AIzaSy[A-Za-z0-9_\-]{16,})\b"
)


def _safe_path_exists(candidate: str) -> bool:
    if not candidate or len(candidate) > 512 or "\x00" in candidate:
        return False
    try:
        return os.path.exists(candidate)
    except OSError:
        return False


def _safe_root_dir_exists(posix_path: str) -> bool:
    if not posix_path.startswith("/") or "/" not in posix_path[1:]:
        return False
    first_seg = posix_path[1:].split("/", 1)[0]
    if not first_seg or first_seg in {".", ".."} or "\x00" in first_seg:
        return False
    try:
        return os.path.isdir(f"/{first_seg}")
    except OSError:
        return False


def find_absolute_path_spans(text: str) -> list[tuple[int, int]]:
    """Find character spans `(start, end)` of local absolute filesystem paths in `text`.

    Handles:
    1. Existing local paths under ANY root prefix (e.g. `/sessions/...`, `/data/...`)
       including folder names containing spaces (longest whitespace-delimited prefix wins).
    2. Quoted paths with spaces under known OS roots or existing host root directories.
    3. Non-existent unquoted single-token paths under known OS roots, Windows drive roots,
       or existing host root directories.
    """
    if not text or ("/" not in text and ":\\" not in text):
        return []

    spans: list[tuple[int, int]] = []
    pos = 0
    n = len(text)

    while pos < n:
        match = _ABS_PATH_START_RE.search(text, pos)
        if match is None:
            break
        start = match.start()

        stop_match = _PATH_HARD_STOP_RE.search(text, start)
        raw_end = stop_match.start() if stop_match is not None else n
        raw_slice = text[start:raw_end]
        trimmed = raw_slice.rstrip(" \t.,:;!?")
        if not trimmed:
            pos = start + 1
            continue

        matched_end: int | None = None

        # 1. Existing local path (longest whitespace-delimited prefix first -> supports spaces & any root)
        ws_ends: list[int] = [len(trimmed)]
        for ws in re.finditer(r"[ \t]+", trimmed):
            if len(ws_ends) >= 16:
                break
            prefix_end = ws.start()
            if prefix_end > 3:
                ws_ends.append(prefix_end)
        ws_ends = sorted(set(ws_ends), reverse=True)
        for rel_end in ws_ends:
            cand = trimmed[:rel_end].rstrip(" \t.,:;!?")
            if len(cand) <= 3:
                continue
            has_subpath = (
                (cand.startswith("/") and "/" in cand[1:])
                or (len(cand) > 3 and cand[1:3] == ":\\")
            )
            if has_subpath and _safe_path_exists(cand):
                matched_end = start + len(cand)
                break

        # 2. Quoted path with spaces under known root or existing host root directory
        if (
            matched_end is None
            and start > 0
            and text[start - 1] in {'"', "'", "`"}
            and raw_end < n
            and text[raw_end] == text[start - 1]
        ):
            if _KNOWN_ABS_ROOT_RE.match(trimmed) or _safe_root_dir_exists(trimmed):
                matched_end = start + len(trimmed)

        # 3. Fallback: non-existent unquoted single-token path
        if matched_end is None:
            first_token = trimmed.split(None, 1)[0].rstrip(".,:;!?")
            if first_token and (
                _KNOWN_ABS_ROOT_RE.match(first_token) or _safe_root_dir_exists(first_token)
            ):
                matched_end = start + len(first_token)

        if matched_end is not None and matched_end > start:
            spans.append((start, matched_end))
            pos = matched_end
        else:
            pos = start + 1

    return spans


def redact_absolute_paths_in_text(
    text: str,
    *,
    replacement: str = "<redacted_path>",
    extra_paths: Sequence[str] = (),
) -> str:
    """Replace any known `extra_paths` and detected local absolute paths with `replacement`."""
    if not text:
        return text

    out = text
    # First scrub any explicit known roots (e.g. unmounted session root paths) longest-first
    cleaned_extra = sorted(
        {p.rstrip("/") for p in extra_paths if p and len(p) > 3},
        key=len,
        reverse=True,
    )
    for root in cleaned_extra:
        if root in out:
            out = out.replace(root, replacement)

    spans = find_absolute_path_spans(out)
    if not spans:
        return out

    chunks: list[str] = []
    cursor = 0
    for start, end in spans:
        chunks.append(out[cursor:start])
        chunks.append(replacement)
        cursor = end
    chunks.append(out[cursor:])
    return "".join(chunks)


def strip_metadata_and_encode(
    image_bytes: bytes,
    *,
    max_edge: int | None = None,
    max_pixels: int | None = None,
    fmt: Literal["JPEG", "WEBP", "PNG"] = "JPEG",
    quality: int = 85,
) -> tuple[bytes, str]:
    """Re-encode an image via Pillow to strip all EXIF/XMP/GPS/MakerNotes and bound dimensions."""
    with Image.open(io.BytesIO(image_bytes)) as src:
        oriented = ImageOps.exif_transpose(src)
        rgb = oriented.convert("RGB") if oriented.mode != "RGB" else oriented.copy()

    w, h = rgb.size
    if max_edge is not None and max_edge > 0 and max(w, h) > max_edge:
        scale = max_edge / float(max(w, h))
        w = max(1, int(round(w * scale)))
        h = max(1, int(round(h * scale)))
        rgb = rgb.resize((w, h), Image.Resampling.LANCZOS)

    if max_pixels is not None and max_pixels > 0 and (w * h) > max_pixels:
        scale = math.sqrt(max_pixels / float(w * h))
        new_w = max(1, int(math.floor(w * scale)))
        new_h = max(1, int(math.floor(h * scale)))
        while (new_w * new_h) > max_pixels and max(new_w, new_h) > 1:
            if new_w >= new_h:
                new_w -= 1
            else:
                new_h -= 1
        rgb = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Copy raw pixel data into a fresh Image instance to guarantee zero residual metadata info
    clean = Image.new("RGB", rgb.size)
    clean.paste(rgb)

    buf = io.BytesIO()
    if fmt == "PNG":
        clean.save(buf, format="PNG")
        mime = "image/png"
    elif fmt == "WEBP":
        clean.save(buf, format="WEBP", quality=quality)
        mime = "image/webp"
    else:
        clean.save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    return buf.getvalue(), mime


def check_outbound_privacy(
    request: ModelCallRequest,
) -> tuple[Literal["ALLOW", "DENY", "ASK"], str | None]:
    """Non-bypassable outbound privacy check: EXIF-free images, no API keys, no absolute paths."""
    env_secrets: list[str] = []
    for var_names in PROVIDER_ENV_VARS.values():
        for var in var_names:
            val = os.environ.get(var, "").strip()
            if len(val) >= 8:
                env_secrets.append(val)

    for msg in request.messages:
        for img in msg.images:
            try:
                with Image.open(io.BytesIO(img.data)) as im:
                    exif = im.getexif()
                    has_exif = bool(exif and len(exif) > 0)
                    has_raw_meta = any(
                        k in im.info for k in ("exif", "xmp", "XML:com.adobe.xmp")
                    )
                    if has_exif or has_raw_meta:
                        return (
                            "DENY",
                            f"Outbound image {img.label or '<unnamed>'} still contains EXIF/XMP metadata.",
                        )
            except Exception as exc:
                return (
                    "DENY",
                    f"Outbound image {img.label or '<unnamed>'} could not be verified: {exc}",
                )

        texts = [msg.text, *(tr.content for tr in msg.tool_results)]
        for text in texts:
            if not text:
                continue
            if _API_KEY_PREFIX_RE.search(text) or any(sec in text for sec in env_secrets):
                return ("DENY", "Outbound model request contains a potential API key or secret.")
            for root_path in request.session_root_paths:
                if root_path and len(root_path) > 3 and root_path in text:
                    return (
                        "DENY",
                        "Outbound model request contains an absolute session directory path.",
                    )
            if find_absolute_path_spans(text):
                return ("DENY", "Outbound model request contains an absolute filesystem path.")

    return ("ALLOW", None)


__all__ = [
    "check_outbound_privacy",
    "find_absolute_path_spans",
    "redact_absolute_paths_in_text",
    "strip_metadata_and_encode",
]
