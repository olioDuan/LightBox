"""Lightbox SQLite storage, repositories, and vector index layer (SPEC §3)."""

from __future__ import annotations

from lightbox.store.db import Database, configure_connection
from lightbox.store.repos import (
    AssetRecord,
    BurstMemberRecord,
    BurstRecord,
    ExifRecord,
    FaceRecord,
    JobItemRecord,
    JobRecord,
    LightboxRepository,
    MetricsRecord,
    OverrideRecord,
    PhotoRecord,
    PlanItemRecord,
    PlanRecord,
    RootRecord,
    SessionRecord,
    VerdictRecord,
)
from lightbox.store.vector_index import (
    DEFAULT_EMBEDDING_DIM,
    NumpyVectorIndex,
    SqliteVecIndex,
    VectorIndex,
    VectorMatch,
    create_vector_index,
)

__all__ = [
    "AssetRecord",
    "BurstMemberRecord",
    "BurstRecord",
    "DEFAULT_EMBEDDING_DIM",
    "Database",
    "ExifRecord",
    "FaceRecord",
    "JobItemRecord",
    "JobRecord",
    "LightboxRepository",
    "MetricsRecord",
    "NumpyVectorIndex",
    "OverrideRecord",
    "PhotoRecord",
    "PlanItemRecord",
    "PlanRecord",
    "RootRecord",
    "SessionRecord",
    "SqliteVecIndex",
    "VectorIndex",
    "VectorMatch",
    "VerdictRecord",
    "configure_connection",
    "create_vector_index",
]
