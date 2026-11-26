"""Outbound image privacy sanitization and `max_pixels` scaling (SPEC §6.2–6.3, §10.2(5))."""

from __future__ import annotations

import io
import math
from typing import Literal

from PIL import Image, ImageOps


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


__all__ = ["strip_metadata_and_encode"]
