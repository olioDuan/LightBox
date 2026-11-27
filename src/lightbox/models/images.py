"""Outbound image privacy sanitization and `max_pixels` scaling (SPEC §6.2–6.3, §10.2(5))."""

from __future__ import annotations

import io
import math
import os
import re
from typing import Literal

from PIL import Image, ImageOps

from lightbox.models.secrets import PROVIDER_ENV_VARS
from lightbox.models.types import ModelCallRequest

_ABS_PATH_RE = re.compile(
    r"""(?:^|[\s"'`(\[:=])(/(?:Users|home|Volumes|mnt|media|tmp|private|var|opt|root)/[^\s"'`)\]]+|[A-Za-z]:\\[^\s"'`)\]]+)"""
)
_API_KEY_PREFIX_RE = re.compile(
    r"\b(?:sk-ant-[A-Za-z0-9_\-]{12,}|sk-proj-[A-Za-z0-9_\-]{12,}|sk-[A-Za-z0-9]{20,}|AIzaSy[A-Za-z0-9_\-]{16,})\b"
)


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
            if _ABS_PATH_RE.search(text):
                return ("DENY", "Outbound model request contains an absolute filesystem path.")

    return ("ALLOW", None)


__all__ = ["check_outbound_privacy", "strip_metadata_and_encode"]
