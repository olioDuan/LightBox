"""PlanService: one-click obvious rejects fast path, deterministic pre-checks, CLI approval codes, and SafeFS execution (SPEC §4.3, §10.2)."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lightbox.config import AppConfig
from lightbox.fs import (
    ApplyPlanResult,
    DestinationExistsError,
    PlanError,
    PlanExecutor,
    PlanStateError,
    RestorePlanResult,
    SafeFS,
    SafeFSError,
    StalePlanError,
)
from lightbox.i18n import is_burst_cull_reason, is_tech_reject_reason
from lightbox.store import LightboxRepository, PlanItemRecord, PlanRecord, VerdictRecord

CLI_APPROVAL_TTL_SECONDS = 600.0
CLI_APPROVAL_MAX_ATTEMPTS = 5
HIGH_REJECT_RATIO_WARNING = 0.40


class ApprovalCodeError(PlanError):
    """Raised when a CLI 6-digit approval code is missing, invalid, expired, or locked out."""


@dataclass(frozen=True, slots=True)
class DraftPlanResult:
    """Result of drafting a quarantine plan (including deterministic verification & reject breakdown)."""

    plan: PlanRecord | None
    items: tuple[PlanItemRecord, ...]
    total_session_photos: int
    selected_count: int
    tech_reject_count: int
    burst_non_pick_count: int
    manual_reject_count: int
    reject_ratio: float
    high_ratio_warning: bool
    excluded_already_quarantined: int
    excluded_unconfirmed_burst: int
    excluded_in_applying_plan: int
    superseded_plan_ids: tuple[str, ...]
    verifier_report: dict[str, Any]


@dataclass(frozen=True, slots=True)
class IssuedApprovalCode:
    plan_id: str
    plan_hash: str
    code: str
    expires_at: float
    photo_count: int
    quarantine_dir: str | None


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.strip().encode("utf-8")).hexdigest()


class PlanService:
    """Business logic facade for drafting, verifying, approving, applying, and restoring plans."""

    def __init__(
        self,
        repo: LightboxRepository,
        config: AppConfig,
        *,
        safe_fs: SafeFS | None = None,
    ) -> None:
        self.repo = repo
        self.config = config
        self.safe_fs = safe_fs or SafeFS()
        self.executor = PlanExecutor(repo, safe_fs=self.safe_fs)

    @staticmethod
    def _classify_selected_breakdown(
        photo_ids: Sequence[str],
        verdicts: dict[str, VerdictRecord],
    ) -> tuple[int, int, int]:
        """Split selected reject photos into `(tech_reject_count, burst_non_pick_count, manual_reject_count)`."""
        tech_count = 0
        burst_count = 0
        manual_count = 0
        for pid in photo_ids:
            v = verdicts.get(pid)
            codes = v.reason_codes if v is not None else ()
            if is_burst_cull_reason(codes):
                burst_count += 1
            elif is_tech_reject_reason(codes):
                tech_count += 1
            else:
                manual_count += 1
        return tech_count, burst_count, manual_count

    def _run_deterministic_plan_checks(
        self,
        session_id: str,
        items: Sequence[PlanItemRecord],
        *,
        total_session_photos: int,
        tech_reject_count: int = 0,
        burst_non_pick_count: int = 0,
        manual_reject_count: int = 0,
    ) -> dict[str, Any]:
        """Run deterministic Layer 1 safety checks (SPEC §4.3, §7.5)."""
        unconfirmed_non_picks = self.repo.list_unconfirmed_burst_non_pick_photo_ids(session_id)
        exempt_hits: list[str] = []
        unconfirmed_hits: list[str] = []
        incomplete_pairs: list[str] = []

        for item in items:
            photo = self.repo.get_photo(item.photo_id)
            if photo is None or photo.is_protected:
                exempt_hits.append(item.photo_id)
            if item.photo_id in unconfirmed_non_picks:
                eff = self.repo.get_effective_verdict(item.photo_id)
                if eff is None or (
                    eff.source != "user" and not is_tech_reject_reason(eff.reason_codes)
                ):
                    unconfirmed_hits.append(item.photo_id)
            assets = self.repo.list_assets_for_photo(item.photo_id)
            expected_src = {str(Path(a.abs_path).expanduser().resolve()) for a in assets}
            actual_src = set(item.src_paths)
            if not expected_src or expected_src != actual_src:
                incomplete_pairs.append(item.photo_id)

        ratio = (len(items) / float(total_session_photos)) if total_session_photos > 0 else 0.0
        warnings: list[str] = []
        if ratio > HIGH_REJECT_RATIO_WARNING:
            warnings.append(
                f"high_reject_ratio: {len(items)}/{total_session_photos} ({ratio:.1%}) exceeds 40%"
            )

        passed = not exempt_hits and not unconfirmed_hits and not incomplete_pairs
        return {
            "layer": "deterministic",
            "passed": passed,
            "exempt_count": len(exempt_hits),
            "unconfirmed_burst_non_pick_count": len(unconfirmed_hits),
            "incomplete_atomic_pair_count": len(incomplete_pairs),
            "selected_count": len(items),
            "tech_reject_count": tech_reject_count,
            "burst_non_pick_count": burst_non_pick_count,
            "manual_reject_count": manual_reject_count,
            "total_session_photos": total_session_photos,
            "reject_ratio": round(ratio, 4),
            "high_ratio_warning": ratio > HIGH_REJECT_RATIO_WARNING,
            "warnings": warnings,
        }

    def draft_obvious_rejects(
        self,
        session_id: str,
        *,
        strategy: str | None = None,
        custom_dir: Path | None = None,
        user_home: Path | None = None,
        include_confirmed_burst_non_picks: bool = True,
    ) -> DraftPlanResult:
        """One-click obvious rejects fast path (SPEC §4.3, PLAN T06).

        Selects ONLY effective `red` photos while strictly excluding:
        - `exempt` / `is_protected` photos
        - non-pick members of unconfirmed bursts (`bursts.confirmed == 0`)
        - photos already moved (`plan_items.status == 'done'`) or in an in-flight `applying` plan

        Automatically supersedes (marks `stale`) any older overlapping unapplied
        `draft`/`reviewed`/`approved` quarantine plans for the same session so double-clicks
        never leave two overlapping executable plans.
        """
        session = self.repo.get_session(session_id)
        if session is None:
            raise PlanError(f"Unknown session_id: {session_id}")

        photos = self.repo.list_photos(session_id)
        verdicts = self.repo.list_effective_verdicts(session_id)
        quarantined_ids = self.repo.list_quarantined_photo_ids(session_id)
        applying_ids = self.repo.list_applying_plan_photo_ids(session_id)
        unconfirmed_non_picks = self.repo.list_unconfirmed_burst_non_pick_photo_ids(session_id)

        selected_ids: list[str] = []
        excluded_quarantined = 0
        excluded_unconfirmed = 0
        excluded_applying = 0

        for photo in photos:
            if photo.is_protected:
                continue
            verdict = verdicts.get(photo.id)
            if verdict is None or verdict.bucket != "red":
                continue
            if photo.id in quarantined_ids:
                excluded_quarantined += 1
                continue
            if photo.id in applying_ids:
                excluded_applying += 1
                continue
            if not self.repo.list_assets_for_photo(photo.id):
                continue
            if (
                not include_confirmed_burst_non_picks
                and is_burst_cull_reason(verdict.reason_codes)
            ):
                excluded_unconfirmed += 1
                continue
            if (
                photo.id in unconfirmed_non_picks
                and verdict.source != "user"
                and not is_tech_reject_reason(verdict.reason_codes)
            ):
                excluded_unconfirmed += 1
                continue
            selected_ids.append(photo.id)

        if not selected_ids:
            empty_report = {
                "layer": "deterministic",
                "passed": True,
                "exempt_count": 0,
                "unconfirmed_burst_non_pick_count": 0,
                "incomplete_atomic_pair_count": 0,
                "selected_count": 0,
                "tech_reject_count": 0,
                "burst_non_pick_count": 0,
                "manual_reject_count": 0,
                "total_session_photos": len(photos),
                "reject_ratio": 0.0,
                "high_ratio_warning": False,
                "warnings": [],
            }
            return DraftPlanResult(
                plan=None,
                items=(),
                total_session_photos=len(photos),
                selected_count=0,
                tech_reject_count=0,
                burst_non_pick_count=0,
                manual_reject_count=0,
                reject_ratio=0.0,
                high_ratio_warning=False,
                excluded_already_quarantined=excluded_quarantined,
                excluded_unconfirmed_burst=excluded_unconfirmed,
                excluded_in_applying_plan=excluded_applying,
                superseded_plan_ids=(),
                verifier_report=empty_report,
            )

        tech_count, burst_count, manual_count = self._classify_selected_breakdown(
            selected_ids, verdicts
        )

        strat = strategy or self.config.quarantine.strategy
        c_dir = (
            custom_dir
            if custom_dir is not None
            else (
                Path(self.config.quarantine.custom_dir)
                if self.config.quarantine.custom_dir
                else None
            )
        )
        plan, items = self.executor.create_quarantine_plan(
            session_id,
            selected_ids,
            strategy=strat,
            custom_dir=c_dir,
            user_home=user_home,
        )
        superseded = self.repo.supersede_unapplied_overlapping_plans(
            session_id,
            selected_ids,
            exclude_plan_id=plan.id,
        )
        report = self._run_deterministic_plan_checks(
            session_id,
            items,
            total_session_photos=len(photos),
            tech_reject_count=tech_count,
            burst_non_pick_count=burst_count,
            manual_reject_count=manual_count,
        )
        self.repo.update_plan_verifier_report(
            plan.id, json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
        refreshed = self.repo.get_plan(plan.id)
        assert refreshed is not None
        return DraftPlanResult(
            plan=refreshed,
            items=tuple(items),
            total_session_photos=len(photos),
            selected_count=len(items),
            tech_reject_count=tech_count,
            burst_non_pick_count=burst_count,
            manual_reject_count=manual_count,
            reject_ratio=float(report["reject_ratio"]),
            high_ratio_warning=bool(report["high_ratio_warning"]),
            excluded_already_quarantined=excluded_quarantined,
            excluded_unconfirmed_burst=excluded_unconfirmed,
            excluded_in_applying_plan=excluded_applying,
            superseded_plan_ids=tuple(superseded),
            verifier_report=report,
        )

    def create_plan(
        self,
        session_id: str,
        photo_ids: Sequence[str],
        *,
        strategy: str | None = None,
        custom_dir: Path | None = None,
        user_home: Path | None = None,
    ) -> DraftPlanResult:
        """Draft a custom quarantine plan for explicit `photo_ids` with deterministic checks."""
        session = self.repo.get_session(session_id)
        if session is None:
            raise PlanError(f"Unknown session_id: {session_id}")

        photos = self.repo.list_photos(session_id)
        verdicts = self.repo.list_effective_verdicts(session_id)
        quarantined_ids = self.repo.list_quarantined_photo_ids(session_id)
        applying_ids = self.repo.list_applying_plan_photo_ids(session_id)
        unconfirmed_non_picks = self.repo.list_unconfirmed_burst_non_pick_photo_ids(session_id)

        filtered: list[str] = []
        excluded_quarantined = 0
        excluded_unconfirmed = 0
        excluded_applying = 0
        for pid in photo_ids:
            if pid in quarantined_ids:
                excluded_quarantined += 1
                continue
            if pid in applying_ids:
                excluded_applying += 1
                continue
            if pid in unconfirmed_non_picks:
                v = verdicts.get(pid)
                if v is None or (
                    v.source != "user" and not is_tech_reject_reason(v.reason_codes)
                ):
                    excluded_unconfirmed += 1
                    continue
            filtered.append(pid)

        tech_count, burst_count, manual_count = self._classify_selected_breakdown(
            filtered, verdicts
        )

        strat = strategy or self.config.quarantine.strategy
        c_dir = (
            custom_dir
            if custom_dir is not None
            else (
                Path(self.config.quarantine.custom_dir)
                if self.config.quarantine.custom_dir
                else None
            )
        )
        plan, items = self.executor.create_quarantine_plan(
            session_id,
            filtered,
            strategy=strat,
            custom_dir=c_dir,
            user_home=user_home,
        )
        superseded = self.repo.supersede_unapplied_overlapping_plans(
            session_id,
            filtered,
            exclude_plan_id=plan.id,
        )
        report = self._run_deterministic_plan_checks(
            session_id,
            items,
            total_session_photos=len(photos),
            tech_reject_count=tech_count,
            burst_non_pick_count=burst_count,
            manual_reject_count=manual_count,
        )
        self.repo.update_plan_verifier_report(
            plan.id, json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
        refreshed = self.repo.get_plan(plan.id)
        assert refreshed is not None
        return DraftPlanResult(
            plan=refreshed,
            items=tuple(items),
            total_session_photos=len(photos),
            selected_count=len(items),
            tech_reject_count=tech_count,
            burst_non_pick_count=burst_count,
            manual_reject_count=manual_count,
            reject_ratio=float(report["reject_ratio"]),
            high_ratio_warning=bool(report["high_ratio_warning"]),
            excluded_already_quarantined=excluded_quarantined,
            excluded_unconfirmed_burst=excluded_unconfirmed,
            excluded_in_applying_plan=excluded_applying,
            superseded_plan_ids=tuple(superseded),
            verifier_report=report,
        )

    def get_plan(self, plan_id: str) -> tuple[PlanRecord, list[PlanItemRecord]]:
        plan = self.repo.get_plan(plan_id)
        if plan is None:
            raise PlanError(f"Unknown plan_id: {plan_id}")
        items = self.repo.list_plan_items(plan_id)
        return plan, items

    def list_plans(self, session_id: str | None = None) -> list[PlanRecord]:
        return self.repo.list_plans(session_id)

    def issue_cli_approval_code(
        self,
        plan_id: str,
        *,
        ttl_seconds: float = CLI_APPROVAL_TTL_SECONDS,
    ) -> IssuedApprovalCode:
        """Generate a 6-digit SHA-256-hashed one-time approval code valid for `ttl_seconds`."""
        plan, items = self.executor.verify_plan_freshness(plan_id, check_hash=True)
        if plan.status not in {"draft", "reviewed"}:
            raise PlanStateError(
                f"Cannot issue approval code for plan {plan_id} in status {plan.status!r}."
            )

        code = f"{secrets.randbelow(1_000_000):06d}"
        expires_at = time.time() + ttl_seconds
        self.repo.set_plan_approval_code(
            plan_id,
            code_hash=_hash_code(code),
            expires_at=expires_at,
        )
        return IssuedApprovalCode(
            plan_id=plan.id,
            plan_hash=plan.plan_hash,
            code=code,
            expires_at=expires_at,
            photo_count=len(items),
            quarantine_dir=plan.quarantine_dir,
        )

    def verify_and_approve_with_code(
        self,
        plan_id: str,
        code: str,
        *,
        max_attempts: int = CLI_APPROVAL_MAX_ATTEMPTS,
        approved_by: str = "cli:user",
    ) -> PlanRecord:
        """Verify a one-time 6-digit CLI approval code with expiry & attempt-limit lockout."""
        plan = self.repo.get_plan(plan_id)
        if plan is None:
            raise PlanError(f"Unknown plan_id: {plan_id}")
        if plan.approval_code_hash is None:
            if plan.approval_attempts >= max_attempts:
                raise ApprovalCodeError(
                    f"Plan {plan_id} approval code is locked after {plan.approval_attempts} failed attempts."
                )
            raise ApprovalCodeError(f"No active approval code issued for plan {plan_id}.")
        if plan.approval_expires_at is None or time.time() > plan.approval_expires_at:
            self.repo.clear_plan_approval_code(plan_id)
            raise ApprovalCodeError(f"Approval code for plan {plan_id} has expired.")
        if plan.approval_attempts >= max_attempts:
            self.repo.clear_plan_approval_code(plan_id)
            raise ApprovalCodeError(
                f"Plan {plan_id} approval code is locked after {max_attempts} failed attempts."
            )

        candidate_hash = _hash_code(code)
        if not hmac.compare_digest(candidate_hash, plan.approval_code_hash):
            attempts, locked = self.repo.record_plan_approval_failure(
                plan_id, max_attempts=max_attempts
            )
            if locked:
                raise ApprovalCodeError(
                    f"Invalid approval code for plan {plan_id}; locked out after {attempts} attempts."
                )
            raise ApprovalCodeError(
                f"Invalid approval code for plan {plan_id} ({attempts}/{max_attempts} attempts)."
            )

        approved = self.executor.approve_plan(plan_id, approved_by=approved_by)
        self.repo.clear_plan_approval_code(plan_id)
        refreshed = self.repo.get_plan(plan_id)
        assert refreshed is not None
        return refreshed or approved

    def approve_plan(self, plan_id: str, *, approved_by: str = "user") -> PlanRecord:
        return self.executor.approve_plan(plan_id, approved_by=approved_by)

    def apply_plan(
        self,
        plan_id: str,
        *,
        resume: bool = False,
        compute_sha256: bool = False,
    ) -> ApplyPlanResult:
        return self.executor.apply_plan(plan_id, resume=resume, compute_sha256=compute_sha256)

    def restore_plan(
        self,
        plan_id: str,
        *,
        photo_ids: Sequence[str] | None = None,
    ) -> RestorePlanResult:
        result = self.executor.restore_plan(plan_id, photo_ids=photo_ids)
        # Restoring a photo from the trash folder means the user wants to keep/re-review it,
        # not have it immediately swept back into the next trash batch.
        for pid in result.restored_photo_ids:
            self.repo.add_verdict(
                photo_id=pid,
                source="user",
                bucket="normal",
                confidence=1.0,
                reason_codes=["user_restored"],
            )
        return result


__all__ = [
    "ApplyPlanResult",
    "ApprovalCodeError",
    "CLI_APPROVAL_MAX_ATTEMPTS",
    "CLI_APPROVAL_TTL_SECONDS",
    "DestinationExistsError",
    "DraftPlanResult",
    "HIGH_REJECT_RATIO_WARNING",
    "IssuedApprovalCode",
    "PlanError",
    "PlanService",
    "PlanStateError",
    "RestorePlanResult",
    "SafeFSError",
    "StalePlanError",
]
