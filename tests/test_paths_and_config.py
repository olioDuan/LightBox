"""Unit tests for `<APP_DATA_DIR>` resolution, config/models loading, logging, and sqlite-vec."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import platformdirs
import pytest
from pydantic import ValidationError

from lightbox.config import load_config, load_models_config
from lightbox.logging import REDACTED, configure_logging, get_logger
from lightbox.paths import HOME_ENV_VAR, get_app_paths, resolve_app_data_dir


def test_app_data_dir_respects_lightbox_home_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_home = tmp_path / "custom_lightbox_home"
    monkeypatch.setenv(HOME_ENV_VAR, str(custom_home))

    paths = get_app_paths()
    assert paths.root == custom_home.resolve()
    assert paths.config_file == custom_home.resolve() / "config.toml"
    assert paths.models_file == custom_home.resolve() / "models.toml"
    assert paths.preferences_file == custom_home.resolve() / "preferences.md"
    assert paths.db_file == custom_home.resolve() / "lightbox.db"
    assert paths.server_lock_file == custom_home.resolve() / "server.lock"
    assert paths.previews_cache_dir == custom_home.resolve() / "cache" / "previews"
    assert paths.crops_cache_dir == custom_home.resolve() / "cache" / "crops"
    assert paths.log_file == custom_home.resolve() / "logs" / "lightbox.log"

    paths.ensure_dirs()
    assert paths.bin_dir.is_dir()
    assert paths.models_dir.is_dir()
    assert paths.previews_cache_dir.is_dir()
    assert paths.crops_cache_dir.is_dir()
    assert paths.logs_dir.is_dir()


def test_app_data_dir_defaults_to_platformdirs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(HOME_ENV_VAR, raising=False)
    expected = Path(platformdirs.user_data_dir("Lightbox")).resolve()
    resolved = resolve_app_data_dir()
    assert resolved == expected
    assert resolved != (Path.home() / ".lightbox").resolve()


def test_load_config_defaults_write_and_override(tmp_path: Path) -> None:
    paths = get_app_paths(tmp_path)
    cfg = load_config(paths, write_default=True)
    assert paths.config_file.is_file()
    assert cfg.language == "zh-CN"
    assert cfg.sensitivity == "conservative"
    assert cfg.model_profile == "balanced"
    assert cfg.budget.session_budget_usd == 0.50
    assert cfg.budget.monthly_budget_usd == 5.00
    assert cfg.quarantine.strategy == "outside_library"

    paths.config_file.write_text(
        'language = "en-US"\nsensitivity = "standard"\nmodel_profile = "local"\n',
        encoding="utf-8",
    )
    updated = load_config(paths)
    assert updated.language == "en-US"
    assert updated.sensitivity == "standard"
    assert updated.model_profile == "local"


def test_config_rejects_plaintext_api_key(tmp_path: Path) -> None:
    paths = get_app_paths(tmp_path)
    paths.ensure_dirs()
    paths.config_file.write_text('api_key = "sk-should-fail"\n', encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(paths)


def test_load_models_config_defaults_and_write(tmp_path: Path) -> None:
    paths = get_app_paths(tmp_path)
    models_cfg = load_models_config(paths, write_default=True)
    assert paths.models_file.is_file()
    assert {"balanced", "best", "local"} <= set(models_cfg.profiles.keys())
    balanced = models_cfg.profiles["balanced"]
    assert balanced.embedder in models_cfg.models
    assert balanced.triage_vlm in models_cfg.models
    assert balanced.agent in models_cfg.models
    assert balanced.reviewer in models_cfg.models


def test_structlog_writes_json_and_redacts_secrets(tmp_path: Path) -> None:
    paths = get_app_paths(tmp_path)
    configure_logging(paths, level="INFO", log_to_file=True, log_to_stderr=False)

    logger = get_logger("lightbox.test")
    logger.info(
        "test_event",
        api_key="sk-secret-value",
        nested={"authorization": "Bearer xyz", "safe": "visible"},
        session_id="sess_123",
    )

    lines = paths.log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    payload = json.loads(lines[-1])
    assert payload["event"] == "test_event"
    assert payload["session_id"] == "sess_123"
    assert payload["api_key"] == REDACTED
    assert payload["nested"]["authorization"] == REDACTED
    assert payload["nested"]["safe"] == "visible"


def test_sqlite_vec_extension_loads_and_queries() -> None:
    import sqlite_vec  # type: ignore[import-untyped]

    conn = sqlite3.connect(":memory:")
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        conn.execute(
            "CREATE VIRTUAL TABLE vec_test USING vec0(embedding float[4] distance_metric=cosine)"
        )
        conn.execute("INSERT INTO vec_test(rowid, embedding) VALUES (1, '[1.0, 0.0, 0.0, 0.0]')")
        conn.execute("INSERT INTO vec_test(rowid, embedding) VALUES (2, '[0.0, 1.0, 0.0, 0.0]')")
        rows = conn.execute(
            "SELECT rowid, distance FROM vec_test WHERE embedding MATCH '[1.0, 0.0, 0.0, 0.0]' AND k = 2"
        ).fetchall()
        assert rows[0][0] == 1
        assert rows[0][1] == pytest.approx(0.0)
        assert rows[1][0] == 2
        assert rows[1][1] == pytest.approx(1.0)
    finally:
        conn.close()
