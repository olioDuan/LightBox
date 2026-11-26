"""Resumable 2D idempotent JobRunner with heartbeat, ProcessPoolExecutor, retries, and circuit breaker (SPEC §5)."""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field

from lightbox.store import JobItemRecord, JobRecord, LightboxRepository

HEARTBEAT_INTERVAL_S = 5.0
DEFAULT_MAX_ITEM_ATTEMPTS = 3
DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5

ProgressCallback = Callable[[str, int, int], None]


def make_cache_key(*parts: str) -> str:
    """Build a deterministic SHA-256 content-addressed cache key."""
    joined = "|".join(parts).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


@dataclass(frozen=True, slots=True)
class StageWorkItem[TIn]:
    """Single `(photo_id, cache_key, payload)` unit for one stage of a resumable job."""

    photo_id: str
    cache_key: str
    payload: TIn


@dataclass(frozen=True, slots=True)
class StageRunResult[TOut]:
    """Execution summary for one stage of a job."""

    job_id: str
    stage: str
    total: int
    completed: int
    cache_hits: int
    skipped_retryable: int
    failed: int
    paused: bool
    paused_reason: str | None
    results: dict[str, TOut] = field(default_factory=dict)


class JobRunner:
    """2D `(job_id, photo_id, stage)` idempotent state machine with heartbeat and crash resume."""

    def __init__(
        self,
        repo: LightboxRepository,
        *,
        heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S,
        max_item_attempts: int = DEFAULT_MAX_ITEM_ATTEMPTS,
        circuit_breaker_threshold: int = DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
    ) -> None:
        self.repo = repo
        self.heartbeat_interval_s = heartbeat_interval_s
        self.max_item_attempts = max_item_attempts
        self.circuit_breaker_threshold = circuit_breaker_threshold

    def create_or_resume_job(
        self,
        *,
        kind: str,
        session_id: str,
        job_id: str | None = None,
        budget_usd: float = 0.50,
    ) -> JobRecord:
        """Create a new job or resume an existing non-completed job."""
        now = time.time()
        if job_id is not None:
            existing = self.repo.get_job(job_id)
            if existing is not None:
                self.repo.update_job(
                    existing.id,
                    status="running",
                    clear_paused_reason=True,
                    heartbeat_at=now,
                )
                refreshed = self.repo.get_job(existing.id)
                assert refreshed is not None
                return refreshed

        new_id = job_id or f"job_{uuid.uuid4().hex[:12]}"
        job = JobRecord(
            id=new_id,
            kind=kind,
            session_id=session_id,
            status="running",
            created_at=now,
            budget_usd=budget_usd,
            spent_usd=0.0,
            heartbeat_at=now,
        )
        self.repo.create_job(job)
        return job

    def heartbeat(self, job_id: str) -> None:
        self.repo.update_job(job_id, heartbeat_at=time.time())

    def mark_completed(self, job_id: str) -> None:
        self.repo.update_job(
            job_id,
            status="completed",
            clear_paused_reason=True,
            heartbeat_at=time.time(),
        )

    def mark_paused(self, job_id: str, reason: str) -> None:
        self.repo.update_job(
            job_id,
            status="paused",
            paused_reason=reason,
            heartbeat_at=time.time(),
        )

    def mark_failed(self, job_id: str, reason: str) -> None:
        self.repo.update_job(
            job_id,
            status="failed",
            paused_reason=reason,
            heartbeat_at=time.time(),
        )

    def run_stage[TIn, TOut](
        self,
        *,
        job_id: str,
        stage: str,
        items: Sequence[StageWorkItem[TIn]],
        worker_fn: Callable[[TIn], TOut],
        cache_lookup_fn: Callable[[StageWorkItem[TIn]], TOut | None] | None = None,
        on_item_result: Callable[[StageWorkItem[TIn], TOut], None] | None = None,
        max_workers: int = 1,
        use_process_pool: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> StageRunResult[TOut]:
        """Execute a stage idempotently with cache hits, resume skipping, retries, and heartbeat."""
        total = len(items)
        existing_items = {
            ji.photo_id: ji for ji in self.repo.list_job_items(job_id, stage=stage)
        }

        results: dict[str, TOut] = {}
        pending_work: list[StageWorkItem[TIn]] = []
        cache_hits = 0
        completed = 0
        skipped_retryable = 0
        failed = 0
        consecutive_failures = 0
        last_hb = time.monotonic()

        def _maybe_heartbeat(force: bool = False) -> None:
            nonlocal last_hb
            now_mono = time.monotonic()
            if force or (now_mono - last_hb) >= self.heartbeat_interval_s:
                self.heartbeat(job_id)
                last_hb = now_mono

        _maybe_heartbeat(force=True)

        # Phase 1: Register job items, skip already-done items on resume, and serve cache hits
        for item in items:
            prev = existing_items.get(item.photo_id)
            if prev is not None and prev.status == "done" and prev.cache_key == item.cache_key:
                # Already finished in a previous run before interruption
                if cache_lookup_fn is not None:
                    cached = cache_lookup_fn(item)
                    if cached is not None:
                        results[item.photo_id] = cached
                cache_hits += 1
                completed += 1
                if on_progress is not None:
                    on_progress(stage, completed, total)
                continue

            if cache_lookup_fn is not None:
                cached = cache_lookup_fn(item)
                if cached is not None:
                    results[item.photo_id] = cached
                    self.repo.upsert_job_item(
                        JobItemRecord(
                            job_id=job_id,
                            photo_id=item.photo_id,
                            stage=stage,
                            status="done",
                            attempts=prev.attempts if prev is not None else 0,
                            cache_key=item.cache_key,
                            last_error=None,
                            updated_at=time.time(),
                        )
                    )
                    cache_hits += 1
                    completed += 1
                    if on_progress is not None:
                        on_progress(stage, completed, total)
                    continue

            self.repo.upsert_job_item(
                JobItemRecord(
                    job_id=job_id,
                    photo_id=item.photo_id,
                    stage=stage,
                    status="pending",
                    attempts=prev.attempts if prev is not None else 0,
                    cache_key=item.cache_key,
                    last_error=prev.last_error if prev is not None else None,
                    updated_at=time.time(),
                )
            )
            pending_work.append(item)

        if not pending_work:
            _maybe_heartbeat(force=True)
            return StageRunResult(
                job_id=job_id,
                stage=stage,
                total=total,
                completed=completed,
                cache_hits=cache_hits,
                skipped_retryable=skipped_retryable,
                failed=failed,
                paused=False,
                paused_reason=None,
                results=results,
            )

        def _handle_success(work_item: StageWorkItem[TIn], output: TOut, attempts: int) -> None:
            nonlocal completed, consecutive_failures
            if on_item_result is not None:
                on_item_result(work_item, output)
            results[work_item.photo_id] = output
            self.repo.upsert_job_item(
                JobItemRecord(
                    job_id=job_id,
                    photo_id=work_item.photo_id,
                    stage=stage,
                    status="done",
                    attempts=attempts,
                    cache_key=work_item.cache_key,
                    last_error=None,
                    updated_at=time.time(),
                )
            )
            completed += 1
            consecutive_failures = 0
            if on_progress is not None:
                on_progress(stage, completed, total)

        def _handle_failure(
            work_item: StageWorkItem[TIn],
            exc: BaseException,
            attempts: int,
        ) -> bool:
            """Record an item failure; return True if circuit breaker tripped."""
            nonlocal skipped_retryable, failed, consecutive_failures
            consecutive_failures += 1
            if attempts >= self.max_item_attempts:
                status = "skipped_retryable"
                skipped_retryable += 1
            else:
                status = "failed"
                failed += 1

            self.repo.upsert_job_item(
                JobItemRecord(
                    job_id=job_id,
                    photo_id=work_item.photo_id,
                    stage=stage,
                    status=status,
                    attempts=attempts,
                    cache_key=work_item.cache_key,
                    last_error=f"{type(exc).__name__}: {exc}",
                    updated_at=time.time(),
                )
            )
            return consecutive_failures >= self.circuit_breaker_threshold

        # Phase 2: Execute pending items (ProcessPoolExecutor for multi-worker CPU stages)
        if use_process_pool and max_workers > 1 and len(pending_work) >= 2:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                in_flight: dict[Future[TOut], tuple[StageWorkItem[TIn], int]] = {}
                queue: list[tuple[StageWorkItem[TIn], int]] = []
                for wi in pending_work:
                    prev = existing_items.get(wi.photo_id)
                    queue.append((wi, prev.attempts if prev is not None else 0))

                paused_reason: str | None = None
                while queue or in_flight:
                    while queue and len(in_flight) < max_workers * 2 and paused_reason is None:
                        wi, prev_attempts = queue.pop(0)
                        fut = pool.submit(worker_fn, wi.payload)
                        in_flight[fut] = (wi, prev_attempts + 1)

                    if not in_flight:
                        break

                    done_futs, _ = wait(
                        set(in_flight.keys()),
                        timeout=self.heartbeat_interval_s,
                        return_when=FIRST_COMPLETED,
                    )
                    _maybe_heartbeat()

                    for fut in done_futs:
                        wi, attempts = in_flight.pop(fut)
                        try:
                            output = fut.result()
                        except Exception as exc:
                            if attempts < self.max_item_attempts and paused_reason is None:
                                queue.append((wi, attempts))
                            if _handle_failure(wi, exc, attempts):
                                paused_reason = f"circuit_breaker_tripped: {exc}"
                                self.mark_paused(job_id, paused_reason)
                                # Cancel queued futures cleanly
                                for pending_fut in list(in_flight.keys()):
                                    pending_fut.cancel()
                                queue.clear()
                                break
                        else:
                            _handle_success(wi, output, attempts)

                    if paused_reason is not None:
                        break

                if paused_reason is not None:
                    return StageRunResult(
                        job_id=job_id,
                        stage=stage,
                        total=total,
                        completed=completed,
                        cache_hits=cache_hits,
                        skipped_retryable=skipped_retryable,
                        failed=failed,
                        paused=True,
                        paused_reason=paused_reason,
                        results=results,
                    )
        else:
            for wi in pending_work:
                cur_job = self.repo.get_job(job_id)
                if cur_job is not None and cur_job.status == "paused":
                    return StageRunResult(
                        job_id=job_id,
                        stage=stage,
                        total=total,
                        completed=completed,
                        cache_hits=cache_hits,
                        skipped_retryable=skipped_retryable,
                        failed=failed,
                        paused=True,
                        paused_reason=cur_job.paused_reason or "user_paused",
                        results=results,
                    )
                prev = existing_items.get(wi.photo_id)
                attempts = prev.attempts if prev is not None else 0
                while attempts < self.max_item_attempts:
                    attempts += 1
                    try:
                        output = worker_fn(wi.payload)
                    except Exception as exc:
                        tripped = _handle_failure(wi, exc, attempts)
                        _maybe_heartbeat()
                        if tripped:
                            reason = f"circuit_breaker_tripped: {exc}"
                            self.mark_paused(job_id, reason)
                            return StageRunResult(
                                job_id=job_id,
                                stage=stage,
                                total=total,
                                completed=completed,
                                cache_hits=cache_hits,
                                skipped_retryable=skipped_retryable,
                                failed=failed,
                                paused=True,
                                paused_reason=reason,
                                results=results,
                            )
                        if attempts >= self.max_item_attempts:
                            break
                    else:
                        _handle_success(wi, output, attempts)
                        _maybe_heartbeat()
                        break

        _maybe_heartbeat(force=True)
        return StageRunResult(
            job_id=job_id,
            stage=stage,
            total=total,
            completed=completed,
            cache_hits=cache_hits,
            skipped_retryable=skipped_retryable,
            failed=failed,
            paused=False,
            paused_reason=None,
            results=results,
        )


__all__ = [
    "DEFAULT_CIRCUIT_BREAKER_THRESHOLD",
    "DEFAULT_MAX_ITEM_ATTEMPTS",
    "HEARTBEAT_INTERVAL_S",
    "JobRunner",
    "ProgressCallback",
    "StageRunResult",
    "StageWorkItem",
    "make_cache_key",
]
