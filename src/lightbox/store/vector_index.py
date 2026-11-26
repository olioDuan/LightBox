"""VectorIndex abstraction with `sqlite-vec` primary and pure `numpy` fallback (SPEC §3.3)."""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from lightbox.store.db import Database

DEFAULT_EMBEDDING_DIM = 768


@dataclass(frozen=True, slots=True)
class VectorMatch:
    """Single nearest-neighbor cosine retrieval result."""

    content_fp: str
    distance: float  # Cosine distance in [0.0, 2.0] (1.0 - cosine_similarity)
    score: float  # Cosine similarity in [-1.0, 1.0]


def _normalize_vector(
    vec: NDArray[np.float32] | Sequence[float],
    dim: int,
) -> NDArray[np.float32]:
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != dim:
        raise ValueError(f"Expected 1-D vector of length {dim}, got shape {arr.shape}.")
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        arr = (arr / norm).astype(np.float32, copy=False)
    return arr


def _pack_float32(vec: NDArray[np.float32]) -> bytes:
    return struct.pack(f"<{vec.shape[0]}f", *vec.tolist())


@runtime_checkable
class VectorIndex(Protocol):
    """Unified vector index interface backed by `sqlite-vec` or pure `numpy`."""

    dim: int
    backend_name: str

    def upsert(
        self,
        content_fp: str,
        model_id: str,
        embedding: NDArray[np.float32] | Sequence[float],
    ) -> None: ...

    def upsert_batch(
        self,
        items: Sequence[tuple[str, str, NDArray[np.float32] | Sequence[float]]],
    ) -> None: ...

    def query(
        self,
        query_embedding: NDArray[np.float32] | Sequence[float],
        model_id: str,
        top_k: int = 10,
        filter_content_fps: set[str] | None = None,
    ) -> list[VectorMatch]: ...

    def count(self, model_id: str | None = None) -> int: ...


class SqliteVecIndex:
    """`sqlite-vec` backend mapping `(content_fp, model_id)` through integer `vec_map.rowid`."""

    backend_name = "sqlite-vec"

    def __init__(self, db: Database, *, dim: int = DEFAULT_EMBEDDING_DIM) -> None:
        if not db.sqlite_vec_available:
            raise RuntimeError("sqlite-vec extension is not available on this SQLite connection.")
        if dim != DEFAULT_EMBEDDING_DIM:
            raise ValueError(
                f"vec_photos table is fixed to {DEFAULT_EMBEDDING_DIM} dimensions, got {dim}."
            )
        self.db = db
        self.dim = dim

    def upsert(
        self,
        content_fp: str,
        model_id: str,
        embedding: NDArray[np.float32] | Sequence[float],
    ) -> None:
        self.upsert_batch([(content_fp, model_id, embedding)])

    def upsert_batch(
        self,
        items: Sequence[tuple[str, str, NDArray[np.float32] | Sequence[float]]],
    ) -> None:
        if not items:
            return
        prepared = [
            (content_fp, model_id, _pack_float32(_normalize_vector(emb, self.dim)))
            for content_fp, model_id, emb in items
        ]
        with self.db.transaction() as cur:
            for content_fp, model_id, blob in prepared:
                existing = cur.execute(
                    "SELECT rowid FROM vec_map WHERE content_fp = ? AND model_id = ?",
                    (content_fp, model_id),
                ).fetchone()
                if existing is not None:
                    rowid = int(existing["rowid"])
                    cur.execute("DELETE FROM vec_photos WHERE rowid = ?", (rowid,))
                else:
                    cur.execute(
                        "INSERT INTO vec_map(content_fp, model_id) VALUES (?, ?)",
                        (content_fp, model_id),
                    )
                    rowid = int(cur.lastrowid or 0)
                cur.execute(
                    "INSERT INTO vec_photos(rowid, embedding) VALUES (?, ?)",
                    (rowid, blob),
                )

    def query(
        self,
        query_embedding: NDArray[np.float32] | Sequence[float],
        model_id: str,
        top_k: int = 10,
        filter_content_fps: set[str] | None = None,
    ) -> list[VectorMatch]:
        if top_k <= 0:
            return []
        total_row = self.db.conn.execute("SELECT COUNT(*) AS cnt FROM vec_map").fetchone()
        total = int(total_row["cnt"]) if total_row is not None else 0
        if total == 0:
            return []

        model_count_row = self.db.conn.execute(
            "SELECT COUNT(*) AS cnt FROM vec_map WHERE model_id = ?", (model_id,)
        ).fetchone()
        model_count = int(model_count_row["cnt"]) if model_count_row is not None else 0
        if model_count == 0:
            return []

        q_blob = _pack_float32(_normalize_vector(query_embedding, self.dim))
        # When all stored vectors belong to the requested model_id and no subset filter is set,
        # sqlite-vec's native k=top_k is exact; otherwise fetch across stored vectors and filter.
        needs_post_filter = (model_count < total) or (filter_content_fps is not None)
        k_limit = total if needs_post_filter else min(top_k, total)

        rows = self.db.conn.execute(
            """
            SELECT m.content_fp, m.model_id, v.distance
            FROM vec_photos v
            JOIN vec_map m ON m.rowid = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance ASC, m.content_fp ASC
            """,
            (q_blob, k_limit),
        ).fetchall()

        matches: list[VectorMatch] = []
        for row in rows:
            if str(row["model_id"]) != model_id:
                continue
            fp = str(row["content_fp"])
            if filter_content_fps is not None and fp not in filter_content_fps:
                continue
            dist = float(row["distance"])
            matches.append(VectorMatch(content_fp=fp, distance=dist, score=1.0 - dist))
            if len(matches) >= top_k:
                break
        return matches

    def count(self, model_id: str | None = None) -> int:
        if model_id is None:
            row = self.db.conn.execute("SELECT COUNT(*) AS cnt FROM vec_map").fetchone()
        else:
            row = self.db.conn.execute(
                "SELECT COUNT(*) AS cnt FROM vec_map WHERE model_id = ?", (model_id,)
            ).fetchone()
        return int(row["cnt"]) if row is not None else 0


class NumpyVectorIndex:
    """Pure-`numpy` brute-force cosine similarity fallback backend (SPEC §3.3)."""

    backend_name = "numpy"

    def __init__(self, *, dim: int = DEFAULT_EMBEDDING_DIM) -> None:
        self.dim = dim
        self._vectors: dict[str, dict[str, NDArray[np.float32]]] = {}

    def upsert(
        self,
        content_fp: str,
        model_id: str,
        embedding: NDArray[np.float32] | Sequence[float],
    ) -> None:
        bucket = self._vectors.setdefault(model_id, {})
        bucket[content_fp] = _normalize_vector(embedding, self.dim)

    def upsert_batch(
        self,
        items: Sequence[tuple[str, str, NDArray[np.float32] | Sequence[float]]],
    ) -> None:
        for content_fp, model_id, embedding in items:
            self.upsert(content_fp, model_id, embedding)

    def query(
        self,
        query_embedding: NDArray[np.float32] | Sequence[float],
        model_id: str,
        top_k: int = 10,
        filter_content_fps: set[str] | None = None,
    ) -> list[VectorMatch]:
        if top_k <= 0:
            return []
        bucket = self._vectors.get(model_id)
        if not bucket:
            return []

        if filter_content_fps is not None:
            keys = [fp for fp in bucket if fp in filter_content_fps]
        else:
            keys = list(bucket.keys())
        if not keys:
            return []

        q = _normalize_vector(query_embedding, self.dim)
        matrix = np.stack([bucket[fp] for fp in keys], axis=0)
        sims = matrix @ q
        dists = 1.0 - sims

        ranked = sorted(
            ((float(dists[i]), keys[i], float(sims[i])) for i in range(len(keys))),
            key=lambda item: (item[0], item[1]),
        )
        return [
            VectorMatch(content_fp=fp, distance=dist, score=sim)
            for dist, fp, sim in ranked[:top_k]
        ]

    def count(self, model_id: str | None = None) -> int:
        if model_id is None:
            return sum(len(bucket) for bucket in self._vectors.values())
        return len(self._vectors.get(model_id, {}))


def create_vector_index(
    db: Database,
    *,
    prefer_sqlite_vec: bool = True,
    dim: int = DEFAULT_EMBEDDING_DIM,
) -> VectorIndex:
    """Create `SqliteVecIndex` when available, else transparently fall back to `NumpyVectorIndex`."""
    if prefer_sqlite_vec and db.sqlite_vec_available and dim == DEFAULT_EMBEDDING_DIM:
        return SqliteVecIndex(db, dim=dim)
    return NumpyVectorIndex(dim=dim)


__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "NumpyVectorIndex",
    "SqliteVecIndex",
    "VectorIndex",
    "VectorMatch",
    "create_vector_index",
]
