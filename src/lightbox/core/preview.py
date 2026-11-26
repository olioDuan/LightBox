"""RAW/still preview extraction, EXIF orientation + sRGB normalization, and 3-tier WebP cache (SPEC §4.1(2))."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import exifread
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageCms, ImageOps

from lightbox.core.cache_lru import enforce_cache_limit
from lightbox.core.exif import ExifToolPool
from lightbox.core.ingest import HEIC_EXTS, RAW_EXTS

GRID_MAX_EDGE = 320
ANALYSIS_MAX_EDGE = 1024
LOW_RES_THRESHOLD = 2000
RAW_EMBEDDED_PREVIEW_TAGS = ("JpgFromRaw", "PreviewImage", "OtherImage")
_EXIF_ORIENTATION_TAG = 0x0112

_ORIENTATION_TRANSFORMS: dict[int, Image.Transpose] = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


@dataclass(frozen=True, slots=True)
class PreviewResult:
    """Paths and resolution metadata for the 3-tier content-addressed preview cache."""

    content_fp: str
    grid_path: Path
    loupe_path: Path
    analysis_path: Path
    preview_w: int
    preview_h: int
    is_low_res: bool
    cache_hit: bool


def _to_srgb(image: Image.Image) -> Image.Image:
    """Convert image to sRGB RGB colorspace, applying embedded ICC profile when present."""
    icc_bytes = image.info.get("icc_profile")
    rgb = image.convert("RGB") if image.mode not in ("RGB", "RGBA") else image
    if rgb.mode == "RGBA":
        rgb = rgb.convert("RGB")
    if icc_bytes:
        try:
            src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
            dst_profile = ImageCms.createProfile("sRGB")
            converted = ImageCms.profileToProfile(
                rgb, src_profile, dst_profile, outputMode="RGB"
            )
            if converted is not None:
                return converted
        except Exception:
            pass
    return rgb


def _probe_raw_orientation(source_path: Path, exif_pool: ExifToolPool | None) -> int | None:
    """Read outer RAW EXIF Orientation tag when an embedded preview lacks its own tag."""
    if exif_pool is not None:
        try:
            records = exif_pool.read_metadata([source_path])
            if records:
                orient = records[0].get("Orientation")
                if isinstance(orient, int) and 1 <= orient <= 8:
                    return orient
        except Exception:
            pass
    try:
        with source_path.open("rb") as fh:
            tags = exifread.process_file(fh, details=False, stop_tag="Image Orientation")
        if "Image Orientation" in tags:
            vals = tags["Image Orientation"].values
            if isinstance(vals, list) and vals and isinstance(vals[0], int) and 1 <= vals[0] <= 8:
                return int(vals[0])
    except Exception:
        pass
    return None


def _apply_orientation(
    image: Image.Image,
    fallback_orientation: int | None,
) -> Image.Image:
    """Apply embedded EXIF orientation if present, else fall back to outer RAW EXIF orientation."""
    embedded = image.getexif().get(_EXIF_ORIENTATION_TAG)
    if isinstance(embedded, int) and 2 <= embedded <= 8:
        return ImageOps.exif_transpose(image)
    if fallback_orientation is not None and fallback_orientation in _ORIENTATION_TRANSFORMS:
        return image.transpose(_ORIENTATION_TRANSFORMS[fallback_orientation])
    return ImageOps.exif_transpose(image)


def _open_raw_best_preview(
    source_path: Path,
    exif_pool: ExifToolPool | None,
) -> tuple[Image.Image, bool]:
    """Extract largest embedded RAW preview, or fallback to `rawpy` half-size demosaic.

    Returns `(image, already_oriented)` where `already_oriented=True` only for `rawpy.postprocess()`
    (which applies LibRaw user_flip automatically).
    """
    # 1. Probe ExifTool embedded preview tags and select the largest pixel area
    if exif_pool is not None:
        best_img: Image.Image | None = None
        best_area = 0
        for tag in RAW_EMBEDDED_PREVIEW_TAGS:
            try:
                blob = exif_pool.extract_binary_tag(source_path, tag)
                if not blob:
                    continue
                candidate = Image.open(io.BytesIO(blob))
                candidate.load()
                area = candidate.width * candidate.height
                if area > best_area:
                    if best_img is not None:
                        best_img.close()
                    best_img = candidate
                    best_area = area
                else:
                    candidate.close()
            except Exception:
                continue
        if best_img is not None:
            return best_img, False

    # 2. Try `rawpy.extract_thumb()`
    import rawpy

    with rawpy.imread(str(source_path)) as raw:
        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG and isinstance(
                thumb.data, (bytes, bytearray)
            ):
                img = Image.open(io.BytesIO(bytes(thumb.data)))
                img.load()
                return img, False
            if thumb.format == rawpy.ThumbFormat.BITMAP and isinstance(thumb.data, np.ndarray):
                return Image.fromarray(thumb.data), False
        except Exception:
            pass

        # 3. Fallback: half-size fast demosaic via rawpy (already rotated by LibRaw)
        rgb_array = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
        return Image.fromarray(rgb_array), True


def _open_source_image(
    source_path: Path,
    exif_pool: ExifToolPool | None,
) -> tuple[Image.Image, bool]:
    ext = source_path.suffix.lower()
    if ext in RAW_EXTS:
        return _open_raw_best_preview(source_path, exif_pool)

    if ext in HEIC_EXTS:
        try:
            import pillow_heif  # type: ignore[import-not-found]

            pillow_heif.register_heif_opener()
        except ImportError:
            pass

    img = Image.open(source_path)
    img.load()
    return img, False


def _resize_long_edge(image: Image.Image, max_edge: int) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= max_edge:
        return image.copy()
    scale = max_edge / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def preview_cache_paths(cache_dir: Path, content_fp: str) -> tuple[Path, Path, Path]:
    """Return `(grid_path, loupe_path, analysis_path)` under `<cache_dir>/previews/<fp[:2]>/`."""
    shard = content_fp[:2] if len(content_fp) >= 2 else "00"
    base_dir = Path(cache_dir) / "previews" / shard
    return (
        base_dir / f"{content_fp}_grid.webp",
        base_dir / f"{content_fp}_loupe.webp",
        base_dir / f"{content_fp}_analysis.webp",
    )


def generate_previews(
    source_path: Path,
    content_fp: str,
    cache_dir: Path,
    *,
    exif_pool: ExifToolPool | None = None,
    exif_orientation: int | None = None,
    max_cache_bytes: int | None = None,
) -> PreviewResult:
    """Generate (or reuse) 3-tier content-addressed WebP previews and optionally run cache LRU.

    If a RAW embedded preview (`JpgFromRaw`/`PreviewImage`/thumb) lacks its own EXIF
    Orientation tag, falls back to `exif_orientation` (or probes the outer RAW file's
    EXIF Orientation) so vertical RAW frames are always rotated upright.
    """
    grid_path, loupe_path, analysis_path = preview_cache_paths(cache_dir, content_fp)

    if grid_path.is_file() and loupe_path.is_file() and analysis_path.is_file():
        with Image.open(loupe_path) as existing:
            preview_w, preview_h = existing.size
        return PreviewResult(
            content_fp=content_fp,
            grid_path=grid_path,
            loupe_path=loupe_path,
            analysis_path=analysis_path,
            preview_w=preview_w,
            preview_h=preview_h,
            is_low_res=max(preview_w, preview_h) < LOW_RES_THRESHOLD,
            cache_hit=True,
        )

    raw_img, already_oriented = _open_source_image(source_path, exif_pool)
    try:
        if already_oriented:
            oriented = raw_img
        else:
            fallback_orient = exif_orientation
            if (
                fallback_orient is None
                and source_path.suffix.lower() in RAW_EXTS
                and raw_img.getexif().get(_EXIF_ORIENTATION_TAG) is None
            ):
                fallback_orient = _probe_raw_orientation(source_path, exif_pool)
            oriented = _apply_orientation(raw_img, fallback_orient)
        srgb = _to_srgb(oriented)
    finally:
        raw_img.close()

    preview_w, preview_h = srgb.size
    loupe_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Loupe (original preview resolution, sRGB WebP)
    srgb.save(loupe_path, format="WEBP", quality=88)

    # 2. Grid (320px long edge, sRGB WebP)
    grid_img = _resize_long_edge(srgb, GRID_MAX_EDGE)
    grid_img.save(grid_path, format="WEBP", quality=82)

    # 3. Analysis (1024px long edge, 8-bit grayscale WebP lossless for deterministic metrics)
    analysis_img = _resize_long_edge(srgb, ANALYSIS_MAX_EDGE).convert("L")
    analysis_img.save(analysis_path, format="WEBP", lossless=True)

    if max_cache_bytes is not None:
        enforce_cache_limit(cache_dir, max_cache_bytes)

    return PreviewResult(
        content_fp=content_fp,
        grid_path=grid_path,
        loupe_path=loupe_path,
        analysis_path=analysis_path,
        preview_w=preview_w,
        preview_h=preview_h,
        is_low_res=max(preview_w, preview_h) < LOW_RES_THRESHOLD,
        cache_hit=False,
    )


def load_analysis_gray(analysis_path: Path) -> NDArray[np.uint8]:
    """Load the cached `analysis` image as a 2-D `uint8` grayscale NumPy array."""
    with Image.open(analysis_path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


__all__ = [
    "ANALYSIS_MAX_EDGE",
    "GRID_MAX_EDGE",
    "LOW_RES_THRESHOLD",
    "PreviewResult",
    "generate_previews",
    "load_analysis_gray",
    "preview_cache_paths",
]
