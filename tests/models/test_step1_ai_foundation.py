"""Step 1 unit tests: atomic config_io, DB migration 006 upgrade & job resume, bound secrets, Router guardrails, and `[ai]` routing."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import sqlite3
import stat
import threading
import time
import tomllib
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

import lightbox.config_io as config_io_mod
from lightbox.agent import IndependentVerifier
from lightbox.config import ActiveAIConfig, AppConfig, load_config
from lightbox.config_io import atomic_update_text, atomic_write_text
from lightbox.jobs import JobRunner, StageWorkItem
from lightbox.models import (
    CallLimitExceededError,
    EndpointBindingError,
    GenerationResult,
    Message,
    ModelCaps,
    ModelError,
    ModelRouter,
    ToolSchema,
    UnpricedTriageConfirmationRequiredError,
    Usage,
    bound_keyring_account,
    delete_provider_secret,
    inspect_provider_secret,
    resolve_api_key,
    resolve_api_key_with_source,
    save_api_key_bound,
)
from lightbox.paths import get_app_paths
from lightbox.services import LightboxServices
from lightbox.services.system import _dump_config_toml
from lightbox.store import Database, LightboxRepository
from lightbox.store.db import (
    MIGRATION_001_SQL,
    MIGRATION_004_SQL,
    SCHEMA_MIGRATIONS_DDL,
    _apply_migration_002,
    _apply_migration_003,
    _apply_migration_005,
    _split_sql_statements,
)


class _SimpleOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool


class _FailThenSucceedAdapter:
    def __init__(self, *, fail_exc: BaseException, model_id: str = "claude-haiku-5-5") -> None:
        self.id = model_id
        self.provider = "anthropic"
        self.family = "anthropic"
        self.caps = ModelCaps(vision=True, tools=True, json_schema=True, prompt_cache=True)
        self.fail_exc = fail_exc
        self.calls = 0

    async def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        response_schema: type[BaseModel] | None = None,
        max_tokens: int = 2048,
        effort: str | None = None,
    ) -> GenerationResult:
        del messages, tools, max_tokens, effort
        self.calls += 1
        if self.calls == 1:
            raise self.fail_exc
        text = '{"ok": true}'
        parsed = response_schema.model_validate_json(text) if response_schema else None
        return GenerationResult(
            text=text,
            parsed=parsed,
            usage=Usage(in_tokens=100, out_tokens=10),
            cost_usd=0.0,
            stop_reason="end_turn",
            provider=self.provider,
            model=self.id,
        )


def _patch_fake_keyring(
    monkeypatch: pytest.MonkeyPatch,
    store: dict[str, str],
    *,
    fail_writes: bool = False,
    corrupt_readback: bool = False,
) -> list[str]:
    deleted_accounts: list[str] = []

    def _read(acct: str) -> str | None:
        if corrupt_readback and "::" in acct and not acct.endswith("__bound_endpoint"):
            return "corrupted-value"
        return store.get(acct)

    def _write(acct: str, val: str) -> None:
        if fail_writes:
            raise RuntimeError("keyring unavailable")
        store[acct] = val

    def _delete(acct: str) -> None:
        deleted_accounts.append(acct)
        store.pop(acct, None)

    monkeypatch.setattr("lightbox.models.secrets._read_from_keyring", _read)
    monkeypatch.setattr("lightbox.models.secrets._write_to_keyring", _write)
    monkeypatch.setattr("lightbox.models.secrets._delete_from_keyring", _delete)
    return deleted_accounts


# ---------------------------------------------------------------------------
# 1. Unified DB Migration 006 (v5 -> v6 upgrade + existing job resume)
# ---------------------------------------------------------------------------


def test_db_migration_006_upgrades_v5_database_and_resumes_existing_job(tmp_path: Path) -> None:
    """Create a genuine v5 DB with existing job & llm_call rows, upgrade to v6, and resume the job."""
    db_file = tmp_path / "legacy_v5.db"
    conn = sqlite3.connect(str(db_file), isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(SCHEMA_MIGRATIONS_DDL)
        cur = conn.cursor()
        for stmt in _split_sql_statements(MIGRATION_001_SQL):
            cur.execute(stmt)
        _apply_migration_002(cur)
        _apply_migration_003(cur)
        for stmt in _split_sql_statements(MIGRATION_004_SQL):
            cur.execute(stmt)
        _apply_migration_005(cur)
        for ver, name in (
            (1, "001_initial_schema"),
            (2, "002_t045_digest_and_face_fields"),
            (3, "003_t10_llm_calls_budget_and_cache"),
            (4, "004_t18_people_classification"),
            (5, "005_t14_memory_rules_columns"),
        ):
            cur.execute(
                "INSERT INTO schema_migrations(version, name, applied_at) VALUES (?, ?, ?)",
                (ver, name, 1000.0),
            )
        # Seed legacy v5 job + done item + pending item + legacy llm_call
        cur.execute(
            """
            INSERT INTO jobs(id, kind, session_id, status, paused_reason, budget_usd, spent_usd, heartbeat_at, created_at)
            VALUES ('job_legacy', 'scan', 'sess_legacy', 'paused', 'interrupted', 0.5, 0.0, 1001.0, 1000.0)
            """
        )
        cur.execute(
            """
            INSERT INTO job_items(job_id, photo_id, stage, status, attempts, cache_key, last_error, updated_at)
            VALUES ('job_legacy', 'p1', 'analyze', 'done', 1, 'ck_1', NULL, 1001.0)
            """
        )
        cur.execute(
            """
            INSERT INTO job_items(job_id, photo_id, stage, status, attempts, cache_key, last_error, updated_at)
            VALUES ('job_legacy', 'p2', 'analyze', 'pending', 0, 'ck_2', NULL, 1001.0)
            """
        )
        cur.execute(
            """
            INSERT INTO llm_calls(
                id, cache_key, role, provider, model, in_tokens, out_tokens,
                cached_tokens, cost_usd, latency_ms, response_json, created_at, session_id, cache_hit
            )
            VALUES ('llm_old', 'ck_llm', 'triage_vlm', 'anthropic', 'claude-haiku-5-5', 100, 20, 0, 0.01, 50, '{}', 1002.0, 'sess_legacy', 0)
            """
        )
    finally:
        conn.close()

    # Open with Database -> migrates cleanly from v5 to v6
    with Database(db_file) as db:
        newly = db.migrate()
        assert newly == [6]
        assert db.migrate() == []

        repo = LightboxRepository(db)
        old_call = repo.get_llm_call_by_cache_key("ck_llm")
        assert old_call is not None
        assert old_call.status == "ok"
        assert old_call.error_class is None
        assert old_call.connection_fingerprint is None

        old_job = repo.get_job("job_legacy")
        assert old_job is not None
        assert old_job.status == "paused"
        assert old_job.owner_pid is None
        assert old_job.owner_pid_started_at is None
        assert old_job.owner_instance_id is None

        # Existing legacy job resumes cleanly without re-executing already-done p1!
        runner = JobRunner(repo)
        resumed = runner.create_or_resume_job(
            kind="scan",
            session_id="sess_legacy",
            job_id="job_legacy",
        )
        assert resumed.status == "running"

        executed: list[str] = []
        stage_res = runner.run_stage(
            job_id="job_legacy",
            stage="analyze",
            items=[
                StageWorkItem(photo_id="p1", cache_key="ck_1", payload="p1"),
                StageWorkItem(photo_id="p2", cache_key="ck_2", payload="p2"),
            ],
            worker_fn=lambda pid: (executed.append(pid) or f"done:{pid}"),
            max_workers=1,
        )
        assert executed == ["p2"]
        assert stage_res.cache_hits == 1
        assert stage_res.completed == 2
        runner.mark_completed("job_legacy")

        final_job = repo.get_job("job_legacy")
        assert final_job is not None and final_job.status == "completed"


# ---------------------------------------------------------------------------
# 2. Atomic config_io: lock, 0o600 .env, PermissionError retry, boundary, rollback
# ---------------------------------------------------------------------------


def test_config_io_atomic_lock_permissions_retry_and_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = get_app_paths(tmp_path / "home")
    paths.ensure_dirs()

    # 1. Rejects writes outside <APP_DATA_DIR>
    outside = tmp_path / "photo_dir" / "IMG_001.jpg"
    outside.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="outside <APP_DATA_DIR>"):
        atomic_write_text(paths, outside, "nope")

    # 2. `.env` temp file and final file BOTH preserve 0o600 permissions
    dotenv_path = paths.root / ".env"
    seen_tmp_modes: list[int] = []
    real_replace = config_io_mod.os.replace

    def _inspect_tmp_mode_replace(src: Any, dst: Any) -> None:
        if os.name == "posix" and Path(dst).name == ".env":
            seen_tmp_modes.append(stat.S_IMODE(Path(src).stat().st_mode))
        real_replace(src, dst)

    monkeypatch.setattr(config_io_mod.os, "replace", _inspect_tmp_mode_replace)
    atomic_write_text(paths, dotenv_path, 'OPENAI_API_KEY="sk-1"\n', mode=0o600)
    monkeypatch.setattr(config_io_mod.os, "replace", real_replace)
    if os.name == "posix":
        assert seen_tmp_modes == [0o600]
        assert stat.S_IMODE(dotenv_path.stat().st_mode) == 0o600

    # 3. Failure in transform leaves original file completely untouched
    cfg_path = paths.config_file
    atomic_write_text(paths, cfg_path, 'language = "zh-CN"\n')

    def _boom(_existing: str | None) -> str:
        raise RuntimeError("transform exploded")

    with pytest.raises(RuntimeError, match="transform exploded"):
        atomic_update_text(paths, cfg_path, _boom)
    assert cfg_path.read_text(encoding="utf-8") == 'language = "zh-CN"\n'

    # 3b. Temp-file write failure leaves original file completely untouched
    real_write_tmp = config_io_mod._write_temp_file

    def _fail_tmp_write(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("simulated disk full during tmp write")

    monkeypatch.setattr(config_io_mod, "_write_temp_file", _fail_tmp_write)
    with pytest.raises(OSError, match="simulated disk full"):
        atomic_write_text(paths, cfg_path, 'language = "en-US"\n')
    monkeypatch.setattr(config_io_mod, "_write_temp_file", real_write_tmp)
    assert cfg_path.read_text(encoding="utf-8") == 'language = "zh-CN"\n'

    # 3c. Exhausted PermissionError retries on os.replace raises and leaves original file untouched
    def _always_denied(_src: Any, _dst: Any) -> None:
        raise PermissionError("persistently locked")

    monkeypatch.setattr(config_io_mod.os, "replace", _always_denied)
    with pytest.raises(PermissionError, match="persistently locked"):
        atomic_write_text(paths, cfg_path, 'language = "en-US"\n', max_replace_retries=3, retry_delay_s=0.001)
    monkeypatch.setattr(config_io_mod.os, "replace", real_replace)
    assert cfg_path.read_text(encoding="utf-8") == 'language = "zh-CN"\n'

    # 4. Windows-style transient PermissionError on os.replace is retried and succeeds
    attempts = {"n": 0}

    def _flaky_replace(src: Any, dst: Any) -> None:
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise PermissionError("file briefly locked by AV")
        real_replace(src, dst)

    monkeypatch.setattr(config_io_mod.os, "replace", _flaky_replace)
    atomic_write_text(paths, cfg_path, 'language = "en-US"\n', max_replace_retries=5, retry_delay_s=0.001)
    assert attempts["n"] == 3
    assert cfg_path.read_text(encoding="utf-8") == 'language = "en-US"\n'
    monkeypatch.setattr(config_io_mod.os, "replace", real_replace)

    # 5. Full read->modify->replace lock prevents lost updates across concurrent threads
    counter_file = paths.root / "counter.txt"
    atomic_write_text(paths, counter_file, "0")

    def _inc() -> None:
        for _ in range(15):
            atomic_update_text(
                paths,
                counter_file,
                lambda cur: str(int((cur or "0").strip()) + 1),
            )

    threads = [threading.Thread(target=_inc) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert int(counter_file.read_text(encoding="utf-8").strip()) == 60


def test_dump_config_toml_roundtrip_with_and_without_ai(tmp_path: Path) -> None:
    base_cfg = AppConfig()
    dumped_base = _dump_config_toml(base_cfg)
    assert "[ai]" not in dumped_base
    assert AppConfig.model_validate(tomllib.loads(dumped_base)) == base_cfg

    ai_cfg = AppConfig(
        language="en-US",
        ai=ActiveAIConfig(
            provider="custom",
            agent_model="my-agent-vlm",
            triage_model="my-triage-vlm",
            base_url="https://proxy.example.com/v1",
            reviewer="none",
            is_local_free=False,
            input_cost_per_mtok=1.25,
            output_cost_per_mtok=5.0,
            max_session_calls=15,
            max_triage_photos_unpriced=12,
        ),
    )
    dumped_ai = _dump_config_toml(ai_cfg)
    assert "[ai]" in dumped_ai
    assert "api_key" not in dumped_ai
    assert AppConfig.model_validate(tomllib.loads(dumped_ai)) == ai_cfg

    # Ensure config.toml rejects plaintext api_key inside [ai]
    paths = get_app_paths(tmp_path / "bad_cfg")
    paths.ensure_dirs()
    paths.config_file.write_text(dumped_ai + 'api_key = "sk-bad"\n', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(paths)


# ---------------------------------------------------------------------------
# 3. Endpoint-bound secrets, keyring readback, fallback, and rebinding cleanup
# ---------------------------------------------------------------------------


def test_endpoint_bound_secrets_rebinding_cleanup_and_env_restriction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = get_app_paths(tmp_path / "sec_home")
    paths.ensure_dirs()
    for v in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "LIGHTBOX_CUSTOM_API_KEY"):
        monkeypatch.delenv(v, raising=False)

    fake_kr: dict[str, str] = {}
    deleted = _patch_fake_keyring(monkeypatch, fake_kr)

    # 1. Unbound env var is allowed for official OpenAI endpoint, BLOCKED for custom base_url!
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-official-1111")
    res_official = resolve_api_key_with_source("openai", app_paths=paths, base_url=None)
    assert res_official.source == "env" and res_official.api_key == "sk-env-official-1111"

    with pytest.raises(EndpointBindingError):
        resolve_api_key("openai", app_paths=paths, base_url="https://relay-a.example.com/v1")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # 2. Save bound key for relay A -> stored in keyring with endpoint encoded
    saved_a = save_api_key_bound(
        "openai",
        "sk-relay-a-2222",
        paths,
        base_url="https://relay-a.example.com/v1/",
    )
    assert saved_a.storage == "keyring"
    assert saved_a.bound_endpoint == "https://relay-a.example.com/v1"
    acct_a = bound_keyring_account("OPENAI_API_KEY", "https://relay-a.example.com/v1")
    assert fake_kr[acct_a] == "sk-relay-a-2222"
    assert resolve_api_key("openai", app_paths=paths, base_url="https://relay-a.example.com/v1") == "sk-relay-a-2222"

    # Changing base_url to relay B without re-entering key MUST fail with EndpointBindingError
    with pytest.raises(EndpointBindingError):
        resolve_api_key("openai", app_paths=paths, base_url="https://relay-b.example.com/v1")

    # 3. Rebinding to relay B cleans up old keyring entry for relay A!
    saved_b = save_api_key_bound(
        "openai",
        "sk-relay-b-3333",
        paths,
        base_url="https://relay-b.example.com/v1",
        previous_base_url="https://relay-a.example.com/v1",
    )
    assert saved_b.storage == "keyring"
    assert acct_a in deleted
    assert acct_a not in fake_kr
    assert resolve_api_key("openai", app_paths=paths, base_url="https://relay-b.example.com/v1") == "sk-relay-b-3333"

    # 4. Keyring readback mismatch -> smooth fallback to 0o600 .env WITH endpoint binding line
    _patch_fake_keyring(monkeypatch, fake_kr, corrupt_readback=True)
    saved_dotenv = save_api_key_bound(
        "custom",
        "sk-custom-4444",
        paths,
        base_url="https://custom-1.example.com/v1",
    )
    assert saved_dotenv.storage == "dotenv"
    dotenv_file = paths.root / ".env"
    assert dotenv_file.is_file()
    if os.name == "posix":
        assert stat.S_IMODE(dotenv_file.stat().st_mode) == 0o600
    assert 'LIGHTBOX_CUSTOM_API_KEY_ENDPOINT="https://custom-1.example.com/v1"' in dotenv_file.read_text(encoding="utf-8")

    # Restore clean keyring read for resolution check
    _patch_fake_keyring(monkeypatch, fake_kr)
    assert resolve_api_key("custom", app_paths=paths, base_url="https://custom-1.example.com/v1") == "sk-custom-4444"
    with pytest.raises(EndpointBindingError):
        resolve_api_key("custom", app_paths=paths, base_url="https://custom-2.example.com/v1")

    # 5. Switching custom endpoint back to keyring cleans up `.env` binding line too!
    saved_kr2 = save_api_key_bound(
        "custom",
        "sk-custom-5555",
        paths,
        base_url="https://custom-2.example.com/v1",
        previous_base_url="https://custom-1.example.com/v1",
    )
    assert saved_kr2.storage == "keyring"
    dotenv_after = dotenv_file.read_text(encoding="utf-8")
    assert "LIGHTBOX_CUSTOM_API_KEY" not in dotenv_after
    assert "LIGHTBOX_CUSTOM_API_KEY_ENDPOINT" not in dotenv_after

    # 6. Deleting secret does not remove OS env var and re-inspection reflects env takeover
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-backup-9999")
    assert delete_provider_secret("openai", paths, base_url="https://relay-b.example.com/v1") is True
    rem = inspect_provider_secret("openai", app_paths=paths, base_url=None)
    assert rem.source == "env" and rem.key_suffix == "9999"


# ---------------------------------------------------------------------------
# 4. ModelRouter error ledger (cache_key=None) + retry-then-cache-hit
# ---------------------------------------------------------------------------


def test_router_records_failed_call_without_blocking_retry_cache(tmp_path: Path) -> None:
    paths = get_app_paths(tmp_path / "router_err")
    paths.ensure_dirs()
    db = Database(paths.db_file)
    db.migrate()
    repo = LightboxRepository(db)
    cfg = load_config(paths)
    adapter = _FailThenSucceedAdapter(
        fail_exc=ModelError("Unauthorized", status_code=401, provider_code="authentication_error")
    )
    router = ModelRouter(repo, paths, cfg, adapters={"triage_vlm": adapter})
    msgs = [Message(role="user", text="Retry after transient/auth error")]

    try:
        # 1. First call fails -> records error row with cache_key=None, cost_usd=0.0, cache_hit=False
        with pytest.raises(ModelError):
            asyncio.run(
                router.generate(
                    role="triage_vlm",
                    messages=msgs,
                    response_schema=_SimpleOutput,
                    session_id="sess_retry",
                )
            )

        err_rows = repo.list_llm_calls(session_id="sess_retry", limit=10)
        assert len(err_rows) == 1
        assert err_rows[0].status == "error"
        assert err_rows[0].error_class == "auth_failed"
        assert err_rows[0].cache_key is None
        assert err_rows[0].cost_usd == 0.0
        assert err_rows[0].cache_hit is False
        assert err_rows[0].response_json is None
        assert err_rows[0].connection_fingerprint is not None

        # 2. Same request retried -> succeeds and writes cache cleanly (no UNIQUE constraint clash)
        r2 = asyncio.run(
            router.generate(
                role="triage_vlm",
                messages=msgs,
                response_schema=_SimpleOutput,
                session_id="sess_retry",
            )
        )
        assert r2.cache_hit is False
        assert isinstance(r2.parsed, _SimpleOutput) and r2.parsed.ok is True

        # 3. Third identical call -> hits SQLite cache ($0, adapter not called again)
        r3 = asyncio.run(
            router.generate(
                role="triage_vlm",
                messages=msgs,
                response_schema=_SimpleOutput,
                session_id="sess_retry",
            )
        )
        assert r3.cache_hit is True
        assert adapter.calls == 2
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 5. Router cost classification (`known_free` vs `price_unknown` localhost) & guardrails
# ---------------------------------------------------------------------------


def test_router_cost_classification_and_price_unknown_guardrails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = get_app_paths(tmp_path / "cost_guard")
    paths.ensure_dirs()
    _patch_fake_keyring(monkeypatch, {})
    db = Database(paths.db_file)
    db.migrate()
    repo = LightboxRepository(db)

    try:
        # A. Built-in local `mlx-vlm` -> known_free; built-in `claude-haiku-5-5` -> known_paid
        base_cfg = AppConfig()
        base_router = ModelRouter(repo, paths, base_cfg)
        a_mlx, s_mlx = base_router.resolve_spec("triage_vlm", profile="local")
        assert base_router.classify_cost(a_mlx, s_mlx) == "known_free"
        a_haiku, s_haiku = base_router.resolve_spec("triage_vlm", profile="balanced")
        assert base_router.classify_cost(a_haiku, s_haiku) == "known_paid"

        # B. Localhost custom proxy (e.g. LiteLLM on 127.0.0.1:4000) defaults to `price_unknown`!
        proxy_cfg = AppConfig(
            ai=ActiveAIConfig(
                provider="custom",
                agent_model="proxy-vlm",
                triage_model="proxy-vlm",
                base_url="http://127.0.0.1:4000/v1",
                is_local_free=False,
                max_session_calls=2,
            )
        )
        fake_proxy = _FailThenSucceedAdapter(fail_exc=RuntimeError("unused"), model_id="proxy-vlm")
        fake_proxy.calls = 1  # skip initial failure
        proxy_router = ModelRouter(repo, paths, proxy_cfg, adapters={"agent": fake_proxy})
        alias, spec = proxy_router.resolve_spec("agent")
        assert spec.provider == "custom"
        assert proxy_router.classify_cost(alias, spec) == "price_unknown"

        # Enforce single-session call limit on `price_unknown`
        for idx in range(2):
            asyncio.run(
                proxy_router.generate(
                    role="agent",
                    messages=[Message(role="user", text=f"Prompt {idx}")],
                    response_schema=_SimpleOutput,
                    session_id="sess_proxy",
                )
            )
        with pytest.raises(CallLimitExceededError):
            asyncio.run(
                proxy_router.generate(
                    role="agent",
                    messages=[Message(role="user", text="Prompt 3 exceeds cap")],
                    response_schema=_SimpleOutput,
                    session_id="sess_proxy",
                )
            )

        # C. Explicitly checked `is_local_free=True` -> `known_free`
        free_cfg = AppConfig(
            ai=ActiveAIConfig(
                provider="custom",
                agent_model="local-vlm",
                triage_model="local-vlm",
                base_url="http://127.0.0.1:8080/v1",
                is_local_free=True,
            )
        )
        free_router = ModelRouter(repo, paths, free_cfg)
        fa, fs = free_router.resolve_spec("agent")
        assert free_router.classify_cost(fa, fs) == "known_free"

        # D. User-entered estimated pricing on custom endpoint -> `known_paid`
        priced_cfg = AppConfig(
            ai=ActiveAIConfig(
                provider="custom",
                agent_model="proxy-vlm",
                triage_model="proxy-vlm",
                base_url="http://127.0.0.1:4000/v1",
                input_cost_per_mtok=1.0,
                output_cost_per_mtok=3.0,
            )
        )
        priced_router = ModelRouter(repo, paths, priced_cfg)
        pa, ps = priced_router.resolve_spec("agent")
        assert priced_router.classify_cost(pa, ps) == "known_paid"
        assert ps.input_cost_per_mtok == 1.0 and ps.output_cost_per_mtok == 3.0
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 6. Global `[ai]` routing priority in IndependentVerifier & TriageService
# ---------------------------------------------------------------------------


def test_global_ai_routing_priority_in_verifier_and_triage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_keyring(monkeypatch, {})
    for v in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "LIGHTBOX_CUSTOM_API_KEY"):
        monkeypatch.delenv(v, raising=False)

    paths = get_app_paths(tmp_path / "ai_routing")
    paths.ensure_dirs()
    cfg = AppConfig(
        model_profile="balanced",
        ai=ActiveAIConfig(
            provider="custom",
            agent_model="my-agent",
            triage_model="my-triage",
            base_url="http://127.0.0.1:4000/v1",
            reviewer="none",
            max_triage_photos_unpriced=5,
        ),
    )

    with LightboxServices(app_paths=paths, config=cfg) as svcs:
        # 1. Verifier respects `[ai]` default reviewer="none" when profile is None
        verifier = IndependentVerifier(svcs)
        choice_ai = verifier.resolve_reviewer_model()
        assert choice_ai.alias is None
        assert choice_ai.skipped_reason == "disabled_in_profile"

        # Explicit profile="balanced" still overrides `[ai]`!
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-explicit-profile")
        choice_explicit = verifier.resolve_reviewer_model(profile="balanced")
        assert choice_explicit.alias == "gemini-2-5-flash"

        # 2. TriageService respects `[ai]` when profile is None & requires unpriced confirmation
        from lightbox.core import ENGINE_VERSION
        from lightbox.store import PhotoRecord, RootRecord, SessionRecord

        svcs.repo.upsert_root(RootRecord(id="r1", path=str(tmp_path), device_id="1", kind="disk"))
        svcs.repo.upsert_session(
            SessionRecord(
                id="s_ai",
                root_id="r1",
                root_path=str(tmp_path),
                name="ai_test",
                sensitivity="conservative",
                engine_version=ENGINE_VERSION,
                created_at=time.time(),
            )
        )
        for i in range(8):
            pid = f"p_{i}"
            svcs.repo.upsert_photo(
                PhotoRecord(
                    id=pid,
                    session_id="s_ai",
                    stem=f"IMG_{i:02d}",
                    rel_dir="",
                    primary_content_fp=f"fp_{i}",
                    cohort="camera",
                )
            )
            svcs.repo.add_verdict(
                photo_id=pid,
                source="engine",
                bucket="yellow",
                confidence=0.4,
                reason_codes=["borderline_blur"],
            )

        est = svcs.triage.estimate_triage("s_ai", max_photos=50)
        assert est.model_id == "my-triage"
        assert est.provider == "custom"
        assert est.cost_classification == "price_unknown"
        assert est.requires_unpriced_confirmation is True
        assert est.unpriced_max_photos == 5
        assert est.yellow_count == 5  # capped by max_triage_photos_unpriced

        # Running without confirm_unpriced=True raises UnpricedTriageConfirmationRequiredError
        with pytest.raises(UnpricedTriageConfirmationRequiredError):
            svcs.triage.create_queued_job("s_ai", confirm_unpriced=False)

        # Explicit profile="balanced" overrides `[ai]` and uses claude-haiku-5-5
        est_bal = svcs.triage.estimate_triage("s_ai", max_photos=50, profile="balanced")
        assert est_bal.model_id == "claude-haiku-5-5"
        assert est_bal.cost_classification == "known_paid"
        assert est_bal.requires_unpriced_confirmation is False


def test_save_ai_connection_write_order_and_failure_safety(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `save_ai_connection` writes bound key first, then atomically writes config.toml."""
    paths = get_app_paths(tmp_path / "write_order")
    paths.ensure_dirs()
    fake_kr: dict[str, str] = {}
    _patch_fake_keyring(monkeypatch, fake_kr)
    call_order: list[str] = []

    from lightbox.models import secrets as secrets_mod
    from lightbox.services import ai_connection as ai_conn_mod

    orig_save_key = secrets_mod.save_api_key_bound
    orig_update_text = ai_conn_mod.atomic_update_text

    def _tracked_save_key(*args: Any, **kwargs: Any) -> Any:
        call_order.append("save_key")
        return orig_save_key(*args, **kwargs)

    def _tracked_update_text(app_paths: Any, target: Path, transform: Any, **kwargs: Any) -> Path:
        call_order.append(f"write:{target.name}")
        return orig_update_text(app_paths, target, transform, **kwargs)

    monkeypatch.setattr(ai_conn_mod, "save_api_key_bound", _tracked_save_key)
    monkeypatch.setattr(ai_conn_mod, "atomic_update_text", _tracked_update_text)

    with LightboxServices(app_paths=paths) as svcs:
        svcs.system.save_ai_connection(
            provider="openai",
            api_key="sk-order-test-1234",
        )
        assert call_order == ["save_key", "write:config.toml"]

        # If writing config.toml fails, original config.toml remains untouched
        original_toml = paths.config_file.read_text(encoding="utf-8")

        def _failing_update(*_args: Any, **_kwargs: Any) -> Path:
            raise OSError("disk full")

        monkeypatch.setattr(ai_conn_mod, "atomic_update_text", _failing_update)
        with pytest.raises(OSError, match="disk full"):
            svcs.system.save_ai_connection(
                provider="anthropic",
                api_key="sk-ant-fail-5678",
            )
        assert paths.config_file.read_text(encoding="utf-8") == original_toml


def _mp_increment_worker(home_dir: str, counter_path: str, iterations: int) -> None:
    """Top-level multiprocessing worker verifying OS-level file locking across processes."""
    worker_paths = get_app_paths(Path(home_dir))
    target = Path(counter_path)
    for _ in range(iterations):
        atomic_update_text(
            worker_paths,
            target,
            lambda cur: str(int((cur or "0").strip()) + 1),
        )


def test_config_io_cross_process_file_lock(tmp_path: Path) -> None:
    """Verify OS-level file locking (flock/msvcrt) prevents lost updates across separate processes."""
    paths = get_app_paths(tmp_path / "mp_home")
    paths.ensure_dirs()
    counter_file = paths.root / "mp_counter.txt"
    atomic_write_text(paths, counter_file, "0")

    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(
            target=_mp_increment_worker,
            args=(str(paths.root), str(counter_file), 15),
        )
        for _ in range(4)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0
    assert int(counter_file.read_text(encoding="utf-8").strip()) == 60


def test_concurrent_update_settings_and_save_ai_connection_never_lose_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deterministic interleaving of update_settings + save_ai_connection preserves BOTH changes."""
    paths = get_app_paths(tmp_path / "interleave_home")
    paths.ensure_dirs()
    _patch_fake_keyring(monkeypatch, {})
    load_config(paths, write_default=True)

    entered_first = threading.Event()
    release_first = threading.Event()
    orig_update = config_io_mod.atomic_update_text

    def _pausing_update(app_paths: Any, target: Path, transform: Any, **kwargs: Any) -> Path:
        if target.name == "config.toml" and not entered_first.is_set():
            def _wrapped(existing: str | None) -> str:
                entered_first.set()
                release_first.wait(timeout=5.0)
                return str(transform(existing))
            return orig_update(app_paths, target, _wrapped, **kwargs)
        return orig_update(app_paths, target, transform, **kwargs)

    import lightbox.services.ai_connection as ai_conn_mod
    import lightbox.services.system as system_mod

    monkeypatch.setattr(system_mod, "atomic_update_text", _pausing_update)
    monkeypatch.setattr(ai_conn_mod, "atomic_update_text", _pausing_update)

    errors: list[BaseException] = []

    def _run_settings() -> None:
        try:
            with LightboxServices(app_paths=paths, migrate=False) as s1:
                s1.system.update_settings({"language": "en-US"})
        except BaseException as exc:
            errors.append(exc)

    def _run_save_ai() -> None:
        try:
            entered_first.wait(timeout=5.0)
            # Release thread 1 shortly after thread 2 queues behind the OS/thread lock
            threading.Timer(0.05, release_first.set).start()
            with LightboxServices(app_paths=paths, migrate=False) as s2:
                s2.system.save_ai_connection(provider="openai", api_key="sk-concurrent-1234")
        except BaseException as exc:
            errors.append(exc)

    with LightboxServices(app_paths=paths, migrate=True):
        pass

    t1 = threading.Thread(target=_run_settings)
    t2 = threading.Thread(target=_run_save_ai)
    t1.start()
    t2.start()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)

    assert not errors, errors
    final_cfg = load_config(paths)
    assert final_cfg.language == "en-US"
    assert final_cfg.ai is not None and final_cfg.ai.provider == "openai"


def test_no_ai_baseline_preserves_custom_zero_cost_models_toml_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `[ai]` is NOT configured, user-defined 0-cost models in models.toml remain uncapped (100% baseline)."""
    paths = get_app_paths(tmp_path / "no_ai_baseline")
    paths.ensure_dirs()
    _patch_fake_keyring(monkeypatch, {})

    # Custom models.toml with local LM Studio and unpriced 0-cost model
    paths.models_file.write_text(
        """\
[profiles.balanced]
embedder = "siglip2-base"
triage_vlm = "lmstudio-vlm"
agent = "unpriced-gpt"
reviewer = "none"
consolidate = "unpriced-gpt"

[models.siglip2-base]
provider = "local"
protocol = "local_onnx"
family = "siglip2"
model_id = "google/siglip2-base-patch16-256"

[models.lmstudio-vlm]
provider = "local"
protocol = "openai_compat"
family = "local"
model_id = "qwen2.5-vl-lmstudio"
base_url = "http://127.0.0.1:1234/v1"
input_cost_per_mtok = 0.0
output_cost_per_mtok = 0.0

[models.unpriced-gpt]
provider = "openai"
protocol = "openai_compat"
family = "openai"
model_id = "gpt-4o"
input_cost_per_mtok = 0.0
output_cost_per_mtok = 0.0
""",
        encoding="utf-8",
    )

    with LightboxServices(app_paths=paths) as svcs:
        assert svcs.config.ai is None
        fake = _FailThenSucceedAdapter(fail_exc=RuntimeError("unused"), model_id="gpt-4o")
        fake.calls = 1
        router = ModelRouter(svcs.repo, paths, svcs.config, adapters={"agent": fake, "triage_vlm": fake})

        a_lms, s_lms = router.resolve_spec("triage_vlm")
        a_gpt, s_gpt = router.resolve_spec("agent")
        assert router.classify_cost(a_lms, s_lms) == "known_free"
        assert router.classify_cost(a_gpt, s_gpt) == "known_free"

        # Exceeding 20 calls in one session MUST NOT trigger CallLimitExceededError when [ai] is absent
        for i in range(22):
            asyncio.run(
                router.generate(
                    role="agent",
                    messages=[Message(role="user", text=f"Legacy uncapped prompt {i}")],
                    response_schema=_SimpleOutput,
                    session_id="sess_legacy_uncapped",
                )
            )
        assert fake.calls == 23


def test_router_caches_connection_fingerprint_without_repeated_keyring_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ModelRouter caches fingerprint per instance so batch runs do not hammer keyring."""
    paths = get_app_paths(tmp_path / "fp_cache")
    paths.ensure_dirs()
    kr_reads = {"count": 0}

    def _counting_read(acct: str) -> str | None:
        kr_reads["count"] += 1
        if acct.endswith("::https://api.openai.com/v1"):
            return "sk-cached-fp-9999"
        return None

    monkeypatch.setattr("lightbox.models.secrets._read_from_keyring", _counting_read)
    cfg = AppConfig(
        ai=ActiveAIConfig(
            provider="openai",
            agent_model="gpt-4o",
            triage_model="gpt-4o-mini",
        )
    )
    db = Database(paths.db_file)
    db.migrate()
    repo = LightboxRepository(db)
    fake = _FailThenSucceedAdapter(fail_exc=RuntimeError("unused"), model_id="gpt-4o-mini")
    fake.calls = 1
    router = ModelRouter(repo, paths, cfg, adapters={"triage_vlm": fake})

    try:
        for i in range(6):
            asyncio.run(
                router.generate(
                    role="triage_vlm",
                    messages=[Message(role="user", text=f"Batch item {i}")],
                    response_schema=_SimpleOutput,
                    session_id="sess_batch_fp",
                )
            )
        # Only the first generate() inspected keyring for the fingerprint; calls 2..6 used Router cache!
        assert kr_reads["count"] == 1
    finally:
        db.close()


def test_paired_pricing_validation_separate_triage_pricing_and_meta_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Partial pricing (input without output, or output without input) must fail fast
    with pytest.raises(ValidationError, match="input_cost_per_mtok and output_cost_per_mtok"):
        ActiveAIConfig(
            provider="custom",
            agent_model="a",
            triage_model="t",
            base_url="https://proxy.example.com/v1",
            input_cost_per_mtok=1.0,
        )
    with pytest.raises(ValidationError, match="triage_input_cost_per_mtok and triage_output_cost_per_mtok"):
        ActiveAIConfig(
            provider="custom",
            agent_model="a",
            triage_model="t",
            base_url="https://proxy.example.com/v1",
            triage_output_cost_per_mtok=2.0,
        )

    # 2. Separate Agent & Triage rates resolve independently in ModelRouter
    paths = get_app_paths(tmp_path / "split_pricing")
    paths.ensure_dirs()
    cfg = AppConfig(
        ai=ActiveAIConfig(
            provider="custom",
            agent_model="big-agent",
            triage_model="cheap-triage",
            base_url="https://proxy.example.com/v1",
            input_cost_per_mtok=5.0,
            output_cost_per_mtok=15.0,
            triage_input_cost_per_mtok=0.2,
            triage_output_cost_per_mtok=0.8,
        )
    )
    assert AppConfig.model_validate(tomllib.loads(_dump_config_toml(cfg))) == cfg
    with Database(paths.db_file) as db:
        db.migrate()
        router = ModelRouter(LightboxRepository(db), paths, cfg)
        _, a_spec = router.resolve_spec("agent")
        _, t_spec = router.resolve_spec("triage_vlm")
        assert a_spec.input_cost_per_mtok == 5.0 and a_spec.output_cost_per_mtok == 15.0
        assert t_spec.input_cost_per_mtok == 0.2 and t_spec.output_cost_per_mtok == 0.8

    # 3. Even if writing keyring `__bound_endpoint` meta fails, local backup index cleans up old endpoint on rebind!
    fake_kr: dict[str, str] = {}
    deleted_accounts: list[str] = []

    def _write_fail_meta(acct: str, val: str) -> None:
        if acct.endswith("__bound_endpoint"):
            raise RuntimeError("keyring meta write rejected")
        fake_kr[acct] = val

    monkeypatch.setattr("lightbox.models.secrets._read_from_keyring", lambda a: fake_kr.get(a))
    monkeypatch.setattr("lightbox.models.secrets._write_to_keyring", _write_fail_meta)
    monkeypatch.setattr(
        "lightbox.models.secrets._delete_from_keyring",
        lambda a: (deleted_accounts.append(a), fake_kr.pop(a, None)),
    )

    save_api_key_bound("openai", "sk-first-1111", paths, base_url="https://ep-one.example.com/v1")
    acct_one = bound_keyring_account("OPENAI_API_KEY", "https://ep-one.example.com/v1")
    assert acct_one in fake_kr
    # Rebind WITHOUT passing previous_base_url -> still finds & deletes acct_one via `.endpoint_bindings.json`!
    save_api_key_bound("openai", "sk-second-2222", paths, base_url="https://ep-two.example.com/v1")
    assert acct_one in deleted_accounts
    assert acct_one not in fake_kr


def test_unpriced_triage_via_agent_ask_and_mcp_elicitation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `price_unknown` triage works cleanly through Agent ASK confirmation and MCP Elicitation."""
    from PIL import Image

    from lightbox.agent import AgentRunner, attach_default_hooks
    from lightbox.core import ENGINE_VERSION
    from lightbox.models import ToolCall
    from lightbox.services import YellowTriageOutput
    from lightbox.store import AssetRecord, PhotoRecord, RootRecord, SessionRecord
    from lightbox.tools import ToolContext, ToolRegistry

    _patch_fake_keyring(monkeypatch, {})
    paths = get_app_paths(tmp_path / "tool_unpriced")
    paths.ensure_dirs()
    shoot = tmp_path / "shoot"
    shoot.mkdir(parents=True, exist_ok=True)
    img_path = shoot / "IMG_01.jpg"
    Image.new("RGB", (600, 400), (120, 140, 160)).save(img_path, format="JPEG")

    cfg = AppConfig(
        ai=ActiveAIConfig(
            provider="custom",
            agent_model="proxy-agent",
            triage_model="proxy-triage",
            base_url="http://127.0.0.1:4000/v1",
            max_triage_photos_unpriced=10,
        )
    )

    class _FakeAgentAndTriage:
        def __init__(self) -> None:
            self.id = "proxy-triage"
            self.provider = "custom"
            self.family = "custom"
            self.caps = ModelCaps(vision=True, tools=True, json_schema=True, prompt_cache=False)
            self.agent_turns = 0

        async def generate(
            self,
            messages: list[Message],
            *,
            tools: list[ToolSchema] | None = None,
            response_schema: type[BaseModel] | None = None,
            max_tokens: int = 2048,
            effort: str | None = None,
        ) -> GenerationResult:
            del messages, max_tokens, effort
            if response_schema is YellowTriageOutput:
                out = YellowTriageOutput(
                    decision="keep",
                    confidence=0.9,
                    reason_code="vlm_keep",
                    rationale="Sharp subject.",
                )
                return GenerationResult(
                    text=out.model_dump_json(),
                    parsed=out,
                    usage=Usage(in_tokens=100, out_tokens=20),
                    provider="custom",
                    model="proxy-triage",
                )
            if tools:
                self.agent_turns += 1
                if self.agent_turns == 1:
                    return GenerationResult(
                        text="",
                        tool_calls=(
                            ToolCall(
                                id="call_triage_1",
                                name="run_triage",
                                arguments={"session_id": "s_unpriced", "dry_run": False, "max_photos": 20},
                            ),
                        ),
                        provider="custom",
                        model="proxy-agent",
                    )
                return GenerationResult(
                    text="Triage completed after your confirmation.",
                    provider="custom",
                    model="proxy-agent",
                )
            return GenerationResult(text="done", provider="custom", model="proxy-agent")

    with LightboxServices(app_paths=paths, config=cfg) as svcs:
        fake_model = _FakeAgentAndTriage()
        router = ModelRouter(
            svcs.repo,
            paths,
            cfg,
            adapters={"agent": fake_model, "triage_vlm": fake_model},
        )
        svcs.override_model_router(router)
        svcs.repo.upsert_root(RootRecord(id="r1", path=str(shoot), device_id="1", kind="disk"))
        svcs.repo.upsert_session(
            SessionRecord(
                id="s_unpriced",
                root_id="r1",
                root_path=str(shoot),
                name="unpriced_shoot",
                sensitivity="conservative",
                engine_version=ENGINE_VERSION,
                created_at=time.time(),
            )
        )
        svcs.repo.upsert_photo(
            PhotoRecord(
                id="p_u1",
                session_id="s_unpriced",
                stem="IMG_01",
                rel_dir="",
                primary_content_fp="fp_u1",
                cohort="camera",
                preview_w=600,
                preview_h=400,
            )
        )
        svcs.repo.upsert_asset(
            AssetRecord(
                photo_id="p_u1",
                rel_path="IMG_01.jpg",
                abs_path=str(img_path),
                kind="jpg",
                size_bytes=img_path.stat().st_size,
                mtime=time.time(),
                content_fp="fp_u1",
            )
        )
        svcs.repo.add_verdict(
            photo_id="p_u1",
            source="engine",
            bucket="yellow",
            confidence=0.45,
            reason_codes=["borderline_blur"],
        )

        # 1. MCP surface: Elicitation declined -> returns error; Elicitation accepted -> succeeds!
        reg = ToolRegistry()
        attach_default_hooks(registry=reg, router=router)

        ctx_decline = ToolContext(
            services=svcs,
            surface="mcp:claude-desktop",
            triage_elicitation_approver=lambda _sid, _cap, _mid: False,
        )
        res_dec = reg.execute(
            "lightbox_triage_uncertain",
            {"session_id": "s_unpriced", "dry_run": False, "max_photos": 20},
            ctx_decline,
        )
        assert res_dec.is_error is True
        assert "declined" in (res_dec.error_message or "").lower()

        elicit_calls: list[tuple[str, int, str]] = []
        ctx_accept = ToolContext(
            services=svcs,
            surface="mcp:claude-desktop",
            triage_elicitation_approver=lambda sid, cap, mid: (
                elicit_calls.append((sid, cap, mid)) or True
            ),
        )
        res_acc = reg.execute(
            "lightbox_triage_uncertain",
            {"session_id": "s_unpriced", "dry_run": False, "max_photos": 20},
            ctx_accept,
        )
        assert res_acc.is_error is False, res_acc.error_message
        assert elicit_calls == [("s_unpriced", 10, "proxy-triage")]
        assert res_acc.data["status"] == "completed"
        assert res_acc.data["reviewed_yellow"] == 1

        # Re-mark photo yellow so Agent ASK test runs a fresh triage item
        svcs.repo.add_verdict(
            photo_id="p_u1",
            source="engine",
            bucket="yellow",
            confidence=0.4,
            reason_codes=["borderline_blur"],
        )

        # 2. Agent surface: triggers BudgetGuardHook ASK -> continue(approve_ask=True) executes triage!
        agent_reg = ToolRegistry()
        attach_default_hooks(registry=agent_reg, router=router)
        runner = AgentRunner(svcs, registry=agent_reg)
        start = runner.start_session_sync("s_unpriced", mode="chat", message="Run triage on yellow photos")
        assert start.status == "awaiting_user"
        assert start.pending_ask is not None
        assert start.pending_ask["ask_kind"] == "unpriced_triage"
        assert start.pending_ask["unpriced_max_photos"] == 10

        cont = runner.continue_session_sync(start.agent_session_id, "confirm", approve_ask=True)
        assert cont.status == "completed"
        assert "confirmation" in cont.answer.lower()
