"""SQLite database connection management and idempotent schema migrations (SPEC §3.2)."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_MIGRATIONS_DDL = """\
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at REAL NOT NULL
);
"""

MIGRATION_001_SQL = """\
-- 1. Roots and sessions
CREATE TABLE IF NOT EXISTS roots (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    device_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    readonly INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    root_id TEXT NOT NULL REFERENCES roots(id),
    root_path TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    sensitivity TEXT DEFAULT 'conservative',
    engine_version TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_scanned_at REAL
);

-- 2. Logical photos and physical assets
CREATE TABLE IF NOT EXISTS photos (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    stem TEXT NOT NULL,
    rel_dir TEXT NOT NULL,
    primary_content_fp TEXT NOT NULL,
    capture_time REAL,
    camera_serial TEXT,
    camera_model TEXT,
    lens_model TEXT,
    cohort TEXT NOT NULL,
    preview_w INTEGER,
    preview_h INTEGER,
    is_protected INTEGER DEFAULT 0,
    exempt_reason TEXT,
    is_private INTEGER DEFAULT 0,
    decision_version INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id TEXT NOT NULL REFERENCES photos(id),
    rel_path TEXT NOT NULL,
    abs_path TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    mtime REAL NOT NULL,
    content_fp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exif_data (
    photo_id TEXT PRIMARY KEY REFERENCES photos(id),
    aperture REAL,
    shutter_s REAL,
    iso INTEGER,
    focal_eq REAL,
    ev_comp REAL,
    ev100 REAL,
    light_bucket TEXT,
    drive_mode TEXT,
    bracket_group TEXT,
    rating INTEGER,
    protected INTEGER,
    gps_lat REAL,
    gps_lon REAL,
    field_confidence_json TEXT
);

-- 3. Detection metrics, faces, append-only verdicts, and burst groups
CREATE TABLE IF NOT EXISTS metrics (
    content_fp TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    sharp_p95 REAL,
    sharp_z REAL,
    noise_sigma REAL,
    motion_aniso REAL,
    exposure_json TEXT,
    texture REAL,
    face_count INTEGER,
    face_sharp_min REAL,
    scene_tag TEXT,
    PRIMARY KEY (content_fp, engine_version)
);

CREATE TABLE IF NOT EXISTS faces (
    content_fp TEXT NOT NULL,
    idx INTEGER NOT NULL,
    bbox_json TEXT NOT NULL,
    sharpness REAL,
    area_ratio REAL,
    PRIMARY KEY (content_fp, idx)
);

CREATE TABLE IF NOT EXISTS verdicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id TEXT NOT NULL REFERENCES photos(id),
    source TEXT NOT NULL,
    bucket TEXT NOT NULL,
    confidence REAL NOT NULL,
    reason_codes_json TEXT NOT NULL,
    detail_json TEXT,
    llm_call_id TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS bursts (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    kind TEXT NOT NULL,
    confirmed INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS burst_members (
    burst_id TEXT NOT NULL REFERENCES bursts(id),
    photo_id TEXT NOT NULL REFERENCES photos(id),
    rank INTEGER NOT NULL,
    is_pick INTEGER DEFAULT 0,
    ai_note TEXT,
    PRIMARY KEY (burst_id, photo_id)
);

-- 4. Vector rowid mapping and FTS5 full-text search
CREATE TABLE IF NOT EXISTS vec_map (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    content_fp TEXT NOT NULL,
    model_id TEXT NOT NULL,
    UNIQUE(content_fp, model_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS session_fts USING fts5(
    photo_id UNINDEXED,
    session_name,
    rel_dir,
    stem,
    scene_tag
);

-- 5. Jobs, LLM call cache, and Agent sessions
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    session_id TEXT NOT NULL,
    status TEXT NOT NULL,
    paused_reason TEXT,
    budget_usd REAL DEFAULT 0.50,
    spent_usd REAL DEFAULT 0.0,
    heartbeat_at REAL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS job_items (
    job_id TEXT NOT NULL REFERENCES jobs(id),
    photo_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER DEFAULT 0,
    cache_key TEXT,
    last_error TEXT,
    updated_at REAL,
    PRIMARY KEY (job_id, photo_id, stage)
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id TEXT PRIMARY KEY,
    cache_key TEXT UNIQUE,
    role TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    in_tokens INTEGER,
    out_tokens INTEGER,
    cached_tokens INTEGER,
    cost_usd REAL,
    latency_ms INTEGER,
    response_json TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_sessions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    constraints_json TEXT,
    plan_json TEXT,
    stage_summary_json TEXT,
    step INTEGER DEFAULT 0,
    spent_usd REAL DEFAULT 0.0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_events (
    agent_session_id TEXT NOT NULL REFERENCES agent_sessions(id),
    seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (agent_session_id, seq)
);

-- 6. Memory, immutable execution plans, and audit logs
CREATE TABLE IF NOT EXISTS overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id TEXT NOT NULL REFERENCES photos(id),
    session_id TEXT NOT NULL,
    from_bucket TEXT NOT NULL,
    to_bucket TEXT NOT NULL,
    reason_tag TEXT,
    feature_snapshot_json TEXT NOT NULL,
    actor TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_rules_meta (
    rule_id TEXT PRIMARY KEY,
    condition_dsl_json TEXT,
    evidence_ids_json TEXT,
    hits INTEGER DEFAULT 0,
    conflicts INTEGER DEFAULT 0,
    last_matched_at REAL
);

CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    plan_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    verifier_report_json TEXT,
    approval_code_hash TEXT,
    approval_attempts INTEGER DEFAULT 0,
    approval_expires_at REAL,
    approved_by TEXT,
    approved_at REAL,
    quarantine_dir TEXT
);

CREATE TABLE IF NOT EXISTS plan_items (
    plan_id TEXT NOT NULL REFERENCES plans(id),
    photo_id TEXT NOT NULL,
    decision_version INTEGER NOT NULL,
    src_paths_json TEXT NOT NULL,
    dst_paths_json TEXT NOT NULL,
    status TEXT NOT NULL,
    done_at REAL,
    PRIMARY KEY (plan_id, photo_id)
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    surface TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    args_redacted_json TEXT,
    hook_decision TEXT NOT NULL,
    duration_ms INTEGER,
    summary TEXT
);

-- Effective verdict view: user (4) > rule (3) > agent/vlm/mcp (2) > engine (1), latest id wins
CREATE VIEW IF NOT EXISTS effective_verdicts AS
WITH ranked AS (
    SELECT
        v.id,
        v.photo_id,
        v.source,
        v.bucket,
        v.confidence,
        v.reason_codes_json,
        v.detail_json,
        v.llm_call_id,
        v.created_at,
        ROW_NUMBER() OVER (
            PARTITION BY v.photo_id
            ORDER BY
                CASE
                    WHEN v.source = 'user' THEN 4
                    WHEN v.source = 'rule' THEN 3
                    WHEN v.source IN ('agent', 'vlm') OR v.source LIKE 'mcp:%' THEN 2
                    ELSE 1
                END DESC,
                v.id DESC
        ) AS rn
    FROM verdicts v
)
SELECT
    id,
    photo_id,
    source,
    bucket,
    confidence,
    reason_codes_json,
    detail_json,
    llm_call_id,
    created_at
FROM ranked
WHERE rn = 1;

-- Supporting indexes
CREATE INDEX IF NOT EXISTS idx_photos_session ON photos(session_id);
CREATE INDEX IF NOT EXISTS idx_photos_fp ON photos(primary_content_fp);
CREATE INDEX IF NOT EXISTS idx_assets_photo ON assets(photo_id);
CREATE INDEX IF NOT EXISTS idx_verdicts_photo_id ON verdicts(photo_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_bursts_session ON bursts(session_id);
CREATE INDEX IF NOT EXISTS idx_plan_items_plan ON plan_items(plan_id);
CREATE INDEX IF NOT EXISTS idx_overrides_photo ON overrides(photo_id);
"""

MIGRATION_002_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("exif_data", "exposure_program", "TEXT"),
    ("exif_data", "af_mode", "TEXT"),
    ("exif_data", "flash_fired", "INTEGER"),
    ("exif_data", "capture_local_hour", "INTEGER"),
    ("metrics", "motion_state", "TEXT"),
    ("faces", "landmarks_json", "TEXT"),
    ("faces", "det_score", "REAL"),
)

MIGRATION_003_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("llm_calls", "session_id", "TEXT"),
    ("llm_calls", "cache_hit", "INTEGER DEFAULT 0"),
)

MIGRATION_003_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_llm_calls_session_created ON llm_calls(session_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_llm_calls_created ON llm_calls(created_at)",
)

MIGRATION_004_SQL = """\
-- 7. People auto-classification tables (SPEC §3.2, PLAN T18)
CREATE TABLE IF NOT EXISTS people (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    is_self INTEGER DEFAULT 0,
    cover_content_fp TEXT,
    cover_face_idx INTEGER,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    content_fp TEXT NOT NULL,
    face_idx INTEGER NOT NULL,
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    embedding_blob BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (content_fp, face_idx, model_id)
);

CREATE TABLE IF NOT EXISTS face_assignments (
    content_fp TEXT NOT NULL,
    face_idx INTEGER NOT NULL,
    person_id TEXT REFERENCES people(id) ON DELETE SET NULL,
    status TEXT NOT NULL,
    source TEXT NOT NULL,
    confidence REAL,
    cluster_id TEXT,
    updated_at REAL NOT NULL,
    PRIMARY KEY (content_fp, face_idx)
);

CREATE INDEX IF NOT EXISTS idx_face_assignments_person ON face_assignments(person_id);
CREATE INDEX IF NOT EXISTS idx_face_assignments_cluster ON face_assignments(cluster_id);
CREATE INDEX IF NOT EXISTS idx_face_assignments_status ON face_assignments(status);
"""

VEC_PHOTOS_DDL = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS vec_photos "
    "USING vec0(embedding float[768] distance_metric=cosine)"
)

MIGRATIONS: tuple[tuple[int, str, str | None], ...] = (
    (1, "001_initial_schema", MIGRATION_001_SQL),
    (2, "002_t045_digest_and_face_fields", None),
    (3, "003_t10_llm_calls_budget_and_cache", None),
    (4, "004_t18_people_classification", MIGRATION_004_SQL),
)


def _try_load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    """Attempt to load the `sqlite-vec` extension on `conn`."""
    if not hasattr(conn, "enable_load_extension"):
        return False
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
        return True
    except Exception:
        return False


def configure_connection(conn: sqlite3.Connection, *, load_vec: bool = True) -> bool:
    """Apply mandatory Lightbox SQLite PRAGMAs and optionally load `sqlite-vec`."""
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 10000;")
    conn.execute("PRAGMA foreign_keys = ON;")
    if load_vec:
        return _try_load_sqlite_vec(conn)
    return False


class Database:
    """SQLite database handle enforcing WAL mode, foreign keys, and idempotent migrations."""

    def __init__(self, db_path: Path | str, *, load_vec: bool = True) -> None:
        self.db_path = str(db_path)
        self._load_vec = load_vec
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.db_path,
            timeout=10.0,
            isolation_level=None,  # Explicit transaction control
            check_same_thread=False,  # Per-request connections hop anyio threadpool enter/run/exit
        )
        self.sqlite_vec_available = configure_connection(self._conn, load_vec=load_vec)

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def migrate(self) -> list[int]:
        """Run all unapplied schema migrations idempotently; returns newly applied versions."""
        self._conn.execute(SCHEMA_MIGRATIONS_DDL)
        applied_rows = self._conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        applied = {int(row["version"]) for row in applied_rows}

        newly_applied: list[int] = []
        for version, name, ddl in MIGRATIONS:
            if version in applied:
                continue
            with self.transaction() as cur:
                if ddl is not None:
                    for statement in _split_sql_statements(ddl):
                        cur.execute(statement)
                elif version == 2:
                    _apply_migration_002(cur)
                elif version == 3:
                    _apply_migration_003(cur)
                cur.execute(
                    "INSERT INTO schema_migrations(version, name, applied_at) VALUES (?, ?, ?)",
                    (version, name, time.time()),
                )
            newly_applied.append(version)

        # Ensure vec_photos virtual table exists whenever sqlite-vec is loaded
        if self.sqlite_vec_available:
            self._conn.execute(VEC_PHOTOS_DDL)

        return newly_applied

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """Execute a short atomic SQLite write transaction (BEGIN IMMEDIATE)."""
        if self._conn.in_transaction:
            cur = self._conn.cursor()
            try:
                yield cur
            finally:
                cur.close()
            return

        cur = self._conn.cursor()
        cur.execute("BEGIN IMMEDIATE")
        try:
            yield cur
            cur.execute("COMMIT")
        except BaseException:
            if self._conn.in_transaction:
                cur.execute("ROLLBACK")
            raise
        finally:
            cur.close()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _apply_migration_002(cur: sqlite3.Cursor) -> None:
    """Idempotently add T04.5 columns to `exif_data`, `metrics`, and `faces`."""
    _apply_column_additions(cur, MIGRATION_002_COLUMNS)


def _apply_migration_003(cur: sqlite3.Cursor) -> None:
    """Idempotently add T10 columns and indexes on `llm_calls`."""
    _apply_column_additions(cur, MIGRATION_003_COLUMNS)
    for idx_sql in MIGRATION_003_INDEXES:
        cur.execute(idx_sql)


def _apply_column_additions(
    cur: sqlite3.Cursor,
    columns: tuple[tuple[str, str, str], ...],
) -> None:
    existing_by_table: dict[str, set[str]] = {}
    for table, col_name, col_type in columns:
        if table not in existing_by_table:
            rows = cur.execute(f"PRAGMA table_info({table})").fetchall()
            existing_by_table[table] = {str(r["name"]) for r in rows}
        if col_name not in existing_by_table[table]:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
            existing_by_table[table].add(col_name)


def _split_sql_statements(script: str) -> list[str]:
    """Split a DDL script into individual executable SQL statements cleanly."""
    statements: list[str] = []
    buffer: list[str] = []
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buffer.append(line)
        joined = "\n".join(buffer)
        if sqlite3.complete_statement(joined):
            statements.append(joined.strip())
            buffer.clear()
    if buffer:
        remainder = "\n".join(buffer).strip()
        if remainder:
            statements.append(remainder)
    return statements


__all__ = ["Database", "MIGRATIONS", "configure_connection"]
