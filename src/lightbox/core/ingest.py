"""Universal media directory scanner, pure content fingerprint, and atomic file pairing (SPEC §4.1)."""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path

from lightbox.core.exif import DEFAULT_EXIF_BATCH_SIZE, ExifExtractor, NormalizedExif

RAW_EXTS: frozenset[str] = frozenset(
    {
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
    }
)
JPG_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".webp"})
HEIC_EXTS: frozenset[str] = frozenset({".heic", ".heif"})
SIDECAR_EXTS: frozenset[str] = frozenset({".xmp", ".dop", ".pp3"})
LIVE_MOV_EXTS: frozenset[str] = frozenset({".mov"})

SKIP_DIR_NAMES: frozenset[str] = frozenset({"Lightbox_Quarantine", "待删除废片"})
SKIP_FILE_NAMES: frozenset[str] = frozenset({".DS_Store", "Thumbs.db"})

PAIR_TIME_TOLERANCE_S = 2.0
LIVE_PHOTO_MAX_DURATION_S = 5.0
_MAX_TZ_OFFSET_S = 14 * 3600
_TZ_INCREMENT_S = 900.0  # 15 minutes (covers all real-world UTC offsets)
_FP_CHUNK_SIZE = 64 * 1024


@dataclass(frozen=True, slots=True)
class DiscoveredAsset:
    """Single physical file belonging to a logical photo group."""

    rel_path: str
    abs_path: Path
    kind: str  # 'raw' | 'jpg' | 'heic' | 'live_mov' | 'sidecar'
    size_bytes: int
    mtime: float
    content_fp: str
    exif: NormalizedExif | None = None


@dataclass(frozen=True, slots=True)
class PhotoGroup:
    """Atomic logical photo grouping RAW + JPG/HEIC + Live Photo .MOV + sidecars."""

    rel_dir: str
    stem: str
    primary_content_fp: str
    primary_asset: DiscoveredAsset
    assets: tuple[DiscoveredAsset, ...]
    exif: NormalizedExif


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Result of scanning a root folder."""

    root_path: Path
    groups: tuple[PhotoGroup, ...]
    skipped_paths: tuple[Path, ...]


def compute_content_fp(path: Path) -> str:
    """Compute pure content fingerprint: `sha256(file_size + head_64KB + tail_64KB)` (NO mtime!)."""
    file_path = Path(path)
    size = file_path.stat().st_size
    hasher = hashlib.sha256()
    hasher.update(struct.pack(">Q", size))

    with file_path.open("rb") as fh:
        if size <= _FP_CHUNK_SIZE * 2:
            hasher.update(fh.read())
        else:
            hasher.update(fh.read(_FP_CHUNK_SIZE))
            fh.seek(size - _FP_CHUNK_SIZE)
            hasher.update(fh.read(_FP_CHUNK_SIZE))

    return hasher.hexdigest()


def _classify_extension(ext_lower: str) -> str | None:
    if ext_lower in RAW_EXTS:
        return "raw"
    if ext_lower in JPG_EXTS:
        return "jpg"
    if ext_lower in HEIC_EXTS:
        return "heic"
    if ext_lower in SIDECAR_EXTS:
        return "sidecar"
    if ext_lower in LIVE_MOV_EXTS:
        return "live_mov"
    return None


def _sidecar_base_stem(filename: str) -> str:
    """Support both `IMG_0001.xmp` and `IMG_0001.ARW.xmp` sidecar naming conventions."""
    p = Path(filename)
    stem = p.stem
    inner = Path(stem)
    if inner.suffix.lower() in (RAW_EXTS | JPG_EXTS | HEIC_EXTS):
        return inner.stem
    return stem


def _is_hidden_or_system_file(name: str) -> bool:
    return name in SKIP_FILE_NAMES or name.startswith("._") or name.startswith(".")


def _should_skip_dir(name: str) -> bool:
    return name in SKIP_DIR_NAMES or name.startswith(".")


def _live_photo_times_compatible(
    still_time: float,
    mov_time: float,
    *,
    tolerance_s: float = PAIR_TIME_TOLERANCE_S,
) -> bool:
    """Check if still and Live Photo MOV capture times match within tolerance.

    Also tolerates timezone offset differences (up to ±14h in 15-minute increments)
    when fallback EXIF parses naive local `DateTimeOriginal` against UTC `mvhd` time.
    """
    diff = abs(still_time - mov_time)
    if diff <= tolerance_s:
        return True
    if diff <= _MAX_TZ_OFFSET_S + tolerance_s:
        nearest_tz = round(diff / _TZ_INCREMENT_S) * _TZ_INCREMENT_S
        if abs(diff - nearest_tz) <= tolerance_s:
            return True
    return False


def _stills_compatible(a: DiscoveredAsset, b: DiscoveredAsset) -> bool:
    """Check whether two same-(rel_dir, stem) still images belong to the same atomic shot."""
    ea = a.exif
    eb = b.exif
    if ea is not None and eb is not None:
        if (
            ea.camera_serial is not None
            and eb.camera_serial is not None
            and ea.camera_serial != eb.camera_serial
        ):
            return False
        if (
            ea.capture_time is not None
            and eb.capture_time is not None
            and abs(ea.capture_time - eb.capture_time) > PAIR_TIME_TOLERANCE_S
        ):
            return False
    return True


def _choose_primary_still(stills: list[DiscoveredAsset]) -> DiscoveredAsset:
    """Prefer RAW > JPG > HEIC as the primary asset for stable content identity."""
    priority = {"raw": 0, "jpg": 1, "heic": 2}
    return min(stills, key=lambda a: (priority.get(a.kind, 9), a.rel_path))


def _merge_group_exif(stills: list[DiscoveredAsset]) -> NormalizedExif:
    """Choose the richest EXIF metadata among paired stills."""
    candidates = [a.exif for a in stills if a.exif is not None]
    if not candidates:
        return NormalizedExif()

    def _richness(ex: NormalizedExif) -> tuple[int, int]:
        filled = sum(
            1
            for v in (
                ex.aperture,
                ex.shutter_s,
                ex.iso,
                ex.focal_eq,
                ex.capture_time,
                ex.camera_model,
                ex.lens_model,
                ex.content_identifier,
            )
            if v is not None
        )
        return (1 if ex.backend == "exiftool" else 0, filled)

    return max(candidates, key=_richness)


def scan_directory(
    root_dir: Path,
    *,
    exif_extractor: ExifExtractor | None = None,
    exif_batch_size: int = DEFAULT_EXIF_BATCH_SIZE,
) -> IngestResult:
    """Recursively scan `root_dir`, filter ignored paths, and pair atomic photo groups (SPEC §4.1)."""
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Scan target is not a directory: {root}")

    own_extractor = exif_extractor is None
    extractor = exif_extractor or ExifExtractor()

    skipped: list[Path] = []
    # Bucket discovered files by relative directory
    dir_stills: dict[str, list[tuple[Path, str, str]]] = {}
    dir_sidecars: dict[str, list[tuple[Path, str]]] = {}
    dir_movs: dict[str, list[tuple[Path, str]]] = {}

    try:
        for current_root, dirnames, filenames in os.walk(root):
            # Prune skipped directories in place (`Lightbox_Quarantine`, dot-dirs)
            kept_dirs: list[str] = []
            for dname in sorted(dirnames):
                if _should_skip_dir(dname):
                    skipped.append(Path(current_root) / dname)
                else:
                    kept_dirs.append(dname)
            dirnames[:] = kept_dirs

            cur_path = Path(current_root)
            rel_dir_path = cur_path.relative_to(root)
            rel_dir = "." if str(rel_dir_path) == "." else rel_dir_path.as_posix()

            for fname in sorted(filenames):
                fpath = cur_path / fname
                if _is_hidden_or_system_file(fname):
                    skipped.append(fpath)
                    continue
                if fpath.is_symlink() or not fpath.is_file():
                    skipped.append(fpath)
                    continue

                kind = _classify_extension(fpath.suffix.lower())
                if kind is None:
                    skipped.append(fpath)
                    continue

                rel_path = fpath.relative_to(root).as_posix()
                if kind in {"raw", "jpg", "heic"}:
                    dir_stills.setdefault(rel_dir, []).append((fpath, rel_path, kind))
                elif kind == "sidecar":
                    dir_sidecars.setdefault(rel_dir, []).append((fpath, rel_path))
                elif kind == "live_mov":
                    dir_movs.setdefault(rel_dir, []).append((fpath, rel_path))

        # Batch extract EXIF for all still images and candidate MOVs
        media_paths: list[Path] = []
        for still_entries in dir_stills.values():
            media_paths.extend(p for p, _rel, _kind in still_entries)
        for mov_entries in dir_movs.values():
            media_paths.extend(p for p, _rel in mov_entries)

        exif_map = extractor.extract_batch(media_paths, batch_size=exif_batch_size)
    finally:
        if own_extractor:
            extractor.close()

    groups: list[PhotoGroup] = []
    all_dirs = sorted(set(dir_stills.keys()) | set(dir_movs.keys()) | set(dir_sidecars.keys()))

    for rel_dir in all_dirs:
        # 1. Build still asset objects and cluster by (rel_dir, stem_lower) + time/serial check
        stills_by_stem: dict[str, list[DiscoveredAsset]] = {}
        for fpath, rel_path, kind in dir_stills.get(rel_dir, []):
            st = fpath.stat()
            asset = DiscoveredAsset(
                rel_path=rel_path,
                abs_path=fpath,
                kind=kind,
                size_bytes=st.st_size,
                mtime=st.st_mtime,
                content_fp=compute_content_fp(fpath),
                exif=exif_map.get(fpath),
            )
            stills_by_stem.setdefault(fpath.stem.lower(), []).append(asset)

        dir_clusters: list[tuple[str, list[DiscoveredAsset]]] = []
        for _stem_key, stem_assets in sorted(stills_by_stem.items()):
            clusters: list[list[DiscoveredAsset]] = []
            for asset in stem_assets:
                placed = False
                for cluster in clusters:
                    if all(_stills_compatible(asset, member) for member in cluster):
                        cluster.append(asset)
                        placed = True
                        break
                if not placed:
                    clusters.append([asset])
            for cluster in clusters:
                canonical_stem = _choose_primary_still(cluster).abs_path.stem
                dir_clusters.append((canonical_stem, cluster))

        # 2. Pair Live Photo .MOV files
        #    Priority 1: exact Apple ContentIdentifier match with a still (HEIC/JPG)
        #    Priority 2: same (rel_dir, stem) + |Δcapture_time| <= 2s + duration <= 5s
        cluster_extras: list[list[DiscoveredAsset]] = [[] for _ in dir_clusters]

        # Map ContentIdentifier -> cluster index (only for HEIC/JPG stills)
        cid_to_cluster: dict[str, int] = {}
        for idx, (_stem, cluster) in enumerate(dir_clusters):
            for member in cluster:
                if member.kind in {"heic", "jpg"} and member.exif and member.exif.content_identifier:
                    cid_to_cluster.setdefault(member.exif.content_identifier, idx)

        for mov_path, mov_rel in dir_movs.get(rel_dir, []):
            mov_exif = exif_map.get(mov_path, NormalizedExif())
            matched_idx: int | None = None

            # Priority 1: ContentIdentifier
            if mov_exif.content_identifier and mov_exif.content_identifier in cid_to_cluster:
                matched_idx = cid_to_cluster[mov_exif.content_identifier]
            else:
                # Priority 2: same stem + duration <= 5s + |Δcapture_time| <= 2s
                mov_stem_lower = mov_path.stem.lower()
                if (
                    mov_exif.duration_s is not None
                    and mov_exif.duration_s <= LIVE_PHOTO_MAX_DURATION_S
                ):
                    for idx, (c_stem, cluster) in enumerate(dir_clusters):
                        if c_stem.lower() != mov_stem_lower:
                            continue
                        still_times = [
                            m.exif.capture_time
                            for m in cluster
                            if m.exif is not None and m.exif.capture_time is not None
                        ]
                        if mov_exif.capture_time is not None and still_times:
                            if any(
                                _live_photo_times_compatible(st_t, mov_exif.capture_time)
                                for st_t in still_times
                            ):
                                matched_idx = idx
                                break
                        else:
                            matched_idx = idx
                            break

            if matched_idx is not None:
                st = mov_path.stat()
                cluster_extras[matched_idx].append(
                    DiscoveredAsset(
                        rel_path=mov_rel,
                        abs_path=mov_path,
                        kind="live_mov",
                        size_bytes=st.st_size,
                        mtime=st.st_mtime,
                        content_fp=compute_content_fp(mov_path),
                        exif=mov_exif,
                    )
                )
            else:
                # Unpaired .MOV is a standalone video -> skip
                skipped.append(mov_path)

        # 3. Pair sidecars (.xmp / .dop / .pp3) by matching stem in same directory
        stem_to_cluster_indices: dict[str, list[int]] = {}
        for idx, (c_stem, _cluster) in enumerate(dir_clusters):
            stem_to_cluster_indices.setdefault(c_stem.lower(), []).append(idx)

        for sc_path, sc_rel in dir_sidecars.get(rel_dir, []):
            sc_stem_lower = _sidecar_base_stem(sc_path.name).lower()
            target_indices = stem_to_cluster_indices.get(sc_stem_lower)
            if target_indices:
                st = sc_path.stat()
                cluster_extras[target_indices[0]].append(
                    DiscoveredAsset(
                        rel_path=sc_rel,
                        abs_path=sc_path,
                        kind="sidecar",
                        size_bytes=st.st_size,
                        mtime=st.st_mtime,
                        content_fp=compute_content_fp(sc_path),
                        exif=None,
                    )
                )
            else:
                skipped.append(sc_path)

        # 4. Finalize PhotoGroups for this directory
        for idx, (c_stem, still_cluster) in enumerate(dir_clusters):
            primary = _choose_primary_still(still_cluster)
            all_assets = tuple(
                sorted([*still_cluster, *cluster_extras[idx]], key=lambda a: a.rel_path)
            )
            groups.append(
                PhotoGroup(
                    rel_dir=rel_dir,
                    stem=c_stem,
                    primary_content_fp=primary.content_fp,
                    primary_asset=primary,
                    assets=all_assets,
                    exif=_merge_group_exif(still_cluster),
                )
            )

    return IngestResult(
        root_path=root,
        groups=tuple(groups),
        skipped_paths=tuple(sorted(skipped)),
    )


__all__ = [
    "DiscoveredAsset",
    "HEIC_EXTS",
    "IngestResult",
    "JPG_EXTS",
    "LIVE_MOV_EXTS",
    "LIVE_PHOTO_MAX_DURATION_S",
    "PAIR_TIME_TOLERANCE_S",
    "PhotoGroup",
    "RAW_EXTS",
    "SIDECAR_EXTS",
    "SKIP_DIR_NAMES",
    "compute_content_fp",
    "scan_directory",
]
