"""Service container wiring Database, Repository, SafeFS, and core Lightbox services."""

from __future__ import annotations

from pathlib import Path

from lightbox.config import AppConfig, load_config
from lightbox.fs import SafeFS
from lightbox.jobs import JobRunner
from lightbox.models import (
    Embedder,
    FaceEmbedder,
    ModelRouter,
    get_shared_face_embedder,
    get_shared_siglip2_embedder,
)
from lightbox.paths import AppPaths, get_app_paths
from lightbox.services.burst import BurstService
from lightbox.services.digest import DigestService
from lightbox.services.people import PeopleService
from lightbox.services.plan import PlanService
from lightbox.services.review import ReviewService
from lightbox.services.scan import ScanService
from lightbox.services.search import SearchService
from lightbox.services.system import SystemService
from lightbox.services.triage import TriageService
from lightbox.store import Database, LightboxRepository, VectorIndex, create_vector_index


class LightboxServices:
    """Lifecycle-managed container exposing all Lightbox services as the single source of truth."""

    def __init__(
        self,
        *,
        app_paths: AppPaths | None = None,
        config: AppConfig | None = None,
        db: Database | None = None,
        safe_fs: SafeFS | None = None,
        model_router: ModelRouter | None = None,
        vector_index: VectorIndex | None = None,
        embedder: Embedder | None = None,
        face_embedder: FaceEmbedder | None = None,
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
        self.model_router = model_router or ModelRouter(
            self.repo,
            self.app_paths,
            self.config,
        )
        self.vector_index: VectorIndex = vector_index or create_vector_index(self.db)
        self.embedder: Embedder = embedder or get_shared_siglip2_embedder(self.app_paths)
        self.face_embedder: FaceEmbedder = face_embedder or get_shared_face_embedder(self.app_paths)

        self.scan = ScanService(
            self.repo,
            self.app_paths,
            self.config,
            job_runner=self.job_runner,
            vector_index=self.vector_index,
            embedder=self.embedder,
        )
        self.review = ReviewService(self.repo, app_paths=self.app_paths)
        self.search = SearchService(
            self.repo,
            self.vector_index,
            self.embedder,
            app_paths=self.app_paths,
        )
        self.people = PeopleService(
            self.repo,
            self.face_embedder,
            safe_fs=self.safe_fs,
            app_paths=self.app_paths,
        )
        self.burst = BurstService(self.repo)
        self.digest = DigestService(self.repo)
        self.plan = PlanService(self.repo, self.config, safe_fs=self.safe_fs)
        self.system = SystemService(self.app_paths, self.db)
        self.triage = TriageService(
            self.repo,
            self.review,
            self.burst,
            self.model_router,
            app_paths=self.app_paths,
            config=self.config,
        )

    def override_model_router(self, router: ModelRouter) -> None:
        """Replace the active ModelRouter instance (e.g. for testing with fake adapters)."""
        self.model_router = router
        self.triage.model_router = router

    def override_embedder(self, embedder: Embedder) -> None:
        """Replace the active Embedder instance (e.g. for testing with fake deterministic vectors)."""
        self.embedder = embedder
        self.scan.embedder = embedder
        self.search.embedder = embedder

    def override_face_embedder(self, face_embedder: FaceEmbedder) -> None:
        """Replace the active FaceEmbedder instance (e.g. for testing with fake 512-d vectors)."""
        self.face_embedder = face_embedder
        self.people.face_embedder = face_embedder

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
