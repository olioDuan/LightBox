"""Service container wiring Database, Repository, SafeFS, and core Lightbox services."""

from __future__ import annotations

from pathlib import Path

from lightbox.config import AppConfig, load_config
from lightbox.fs import SafeFS
from lightbox.jobs import JobRunner
from lightbox.paths import AppPaths, get_app_paths
from lightbox.services.burst import BurstService
from lightbox.services.plan import PlanService
from lightbox.services.review import ReviewService
from lightbox.services.scan import ScanService
from lightbox.services.system import SystemService
from lightbox.store import Database, LightboxRepository


class LightboxServices:
    """Lifecycle-managed container exposing all Lightbox services as the single source of truth."""

    def __init__(
        self,
        *,
        app_paths: AppPaths | None = None,
        config: AppConfig | None = None,
        db: Database | None = None,
        safe_fs: SafeFS | None = None,
        migrate: bool = True,
    ) -> None:
        self.app_paths = app_paths or get_app_paths()
        self.app_paths.ensure_dirs()
        self.config = config or load_config(self.app_paths)
        self._owns_db = db is None
        self.db = db or Database(self.app_paths.db_file)
        if migrate:
            self.db.migrate()
        self.repo = LightboxRepository(self.db)
        self.safe_fs = safe_fs or SafeFS()
        self.job_runner = JobRunner(self.repo)

        self.scan = ScanService(
            self.repo,
            self.app_paths,
            self.config,
            job_runner=self.job_runner,
        )
        self.review = ReviewService(self.repo, app_paths=self.app_paths)
        self.burst = BurstService(self.repo)
        self.plan = PlanService(self.repo, self.config, safe_fs=self.safe_fs)
        self.system = SystemService(self.app_paths, self.db)

    @classmethod
    def open(cls, home_override: Path | str | None = None, *, migrate: bool = True) -> LightboxServices:
        paths = get_app_paths(home_override)
        return cls(app_paths=paths, migrate=migrate)

    def close(self) -> None:
        if self._owns_db:
            self.db.close()

    def __enter__(self) -> LightboxServices:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = ["LightboxServices"]
