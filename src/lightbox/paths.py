"""Application data directory resolution (SPEC §3.1).

Uses `platformdirs.user_data_dir("Lightbox")` with `LIGHTBOX_HOME` environment
variable override. Never hardcodes `~/.lightbox/`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import platformdirs

APP_NAME = "Lightbox"
HOME_ENV_VAR = "LIGHTBOX_HOME"


def resolve_app_data_dir(override: Path | str | None = None) -> Path:
    """Resolve `<APP_DATA_DIR>` from explicit override, `LIGHTBOX_HOME`, or platformdirs."""
    if override is not None:
        return Path(override).expanduser().resolve()

    env_home = os.environ.get(HOME_ENV_VAR, "").strip()
    if env_home:
        return Path(env_home).expanduser().resolve()

    return Path(platformdirs.user_data_dir(APP_NAME)).resolve()


@dataclass(frozen=True, slots=True)
class AppPaths:
    """Canonical file and directory locations under `<APP_DATA_DIR>` (SPEC §3.1)."""

    root: Path

    @property
    def config_file(self) -> Path:
        return self.root / "config.toml"

    @property
    def models_file(self) -> Path:
        return self.root / "models.toml"

    @property
    def preferences_file(self) -> Path:
        return self.root / "preferences.md"

    @property
    def db_file(self) -> Path:
        return self.root / "lightbox.db"

    @property
    def server_lock_file(self) -> Path:
        return self.root / "server.lock"

    @property
    def bin_dir(self) -> Path:
        return self.root / "bin"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def previews_cache_dir(self) -> Path:
        return self.cache_dir / "previews"

    @property
    def crops_cache_dir(self) -> Path:
        return self.cache_dir / "crops"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / "lightbox.log"

    def ensure_dirs(self) -> None:
        """Create required application data subdirectories if missing."""
        for directory in (
            self.root,
            self.bin_dir,
            self.models_dir,
            self.previews_cache_dir,
            self.crops_cache_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def get_app_paths(override: Path | str | None = None) -> AppPaths:
    """Return an `AppPaths` instance for the resolved `<APP_DATA_DIR>`."""
    return AppPaths(root=resolve_app_data_dir(override))


__all__ = ["APP_NAME", "HOME_ENV_VAR", "AppPaths", "get_app_paths", "resolve_app_data_dir"]
