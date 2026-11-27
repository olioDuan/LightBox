"""Code-level HookRegistry and 6 built-in non-bypassable governance hooks (SPEC §7.4, PLAN T12)."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from lightbox.fs import PlanError, StalePlanError
from lightbox.models import (
    ModelCallRequest,
    ModelRouter,
    check_outbound_privacy,
)
from lightbox.tools.registry import (
    ToolContext,
    ToolExecutionResult,
    ToolSpec,
    _truncate_untrusted,
    redact_tool_args,
)

if TYPE_CHECKING:
    from lightbox.tools.registry import ToolRegistry


class HookDecision(StrEnum):
    """Decision returned by code-level pre-execution hooks (SPEC §7.4, PLAN T12)."""

    ALLOW = "ALLOW"
    DENY = "DENY"
    ASK = "ASK"


@dataclass(frozen=True, slots=True)
class HookResult:
    """Structured outcome from evaluating a hook."""

    decision: HookDecision
    reason: str | None = None
    hook_name: str = ""
    payload: dict[str, Any] | None = None

    @classmethod
    def allow(cls, hook_name: str = "") -> HookResult:
        return cls(decision=HookDecision.ALLOW, reason=None, hook_name=hook_name)

    @classmethod
    def deny(
        cls,
        reason: str,
        *,
        hook_name: str = "",
        payload: dict[str, Any] | None = None,
    ) -> HookResult:
        return cls(decision=HookDecision.DENY, reason=reason, hook_name=hook_name, payload=payload)

    @classmethod
    def ask(
        cls,
        reason: str,
        *,
        hook_name: str = "",
        payload: dict[str, Any] | None = None,
    ) -> HookResult:
        return cls(decision=HookDecision.ASK, reason=reason, hook_name=hook_name, payload=payload)


class PreToolHook(Protocol):
    name: str

    def check(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> HookResult: ...


class PostToolHook(Protocol):
    name: str

    def on_result(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        result: ToolExecutionResult,
        ctx: ToolContext,
        *,
        duration_ms: int = 0,
    ) -> None: ...


class PreModelHook(Protocol):
    name: str

    def check(self, request: ModelCallRequest) -> HookResult: ...


def wrap_untrusted_metadata(value: str | None, *, max_len: int = 160) -> str:
    """Truncate and wrap untrusted photo filename/EXIF metadata in `<untrusted_metadata>` tags."""
    truncated = _truncate_untrusted(value, max_len=max_len) or ""
    # Escape all angle brackets so nested or crafted payloads can never break out of the wrapper
    safe = truncated.replace("<", "&lt;").replace(">", "&gt;")
    return f"<untrusted_metadata>{safe}</untrusted_metadata>"


def _month_start_ts() -> float:
    now = datetime.now(tz=UTC)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _has_dcim_segment(path_str: str) -> bool:
    return any(part.lower() == "dcim" for part in Path(path_str).parts)


# ---------------------------------------------------------------------------
# 1. NoHardDeleteHook (pre_tool_use)
# ---------------------------------------------------------------------------

_FORBIDDEN_DELETE_NAMES: frozenset[str] = frozenset(
    {"delete", "remove", "unlink", "rmdir", "rmtree", "rm", "erase", "purge", "shred"}
)
_DELETE_PRIMITIVE_PATTERNS: tuple[str, ...] = (
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "shutil.rmtree",
    ".unlink(",
    ".rmdir(",
    "rm -rf",
    "rm -f",
)


class NoHardDeleteHook:
    """Blocks any deletion tool name or hard-delete primitive in tool arguments."""

    name = "NoHardDeleteHook"

    def check(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> HookResult:
        _ = ctx
        for candidate in (spec.name, spec.mcp_name or ""):
            tokens = set(re.split(r"[^a-z0-9]+", candidate.lower()))
            if tokens & _FORBIDDEN_DELETE_NAMES:
                return HookResult.deny(
                    f"Hard-delete tool {candidate!r} is forbidden (SPEC §10.2(1)).",
                    hook_name=self.name,
                )
        serialized = json.dumps(arguments, ensure_ascii=False, default=str).lower()
        for prim in _DELETE_PRIMITIVE_PATTERNS:
            if prim in serialized:
                return HookResult.deny(
                    f"Hard-delete primitive {prim!r} detected in tool arguments.",
                    hook_name=self.name,
                )
        return HookResult.allow(self.name)


# ---------------------------------------------------------------------------
# 2. PathBoundaryHook (pre_tool_use)
# ---------------------------------------------------------------------------

_PATH_ARG_KEYS: tuple[str, ...] = (
    "folder_path",
    "target_dir",
    "custom_dir",
    "export_dir",
    "path",
)
_MUTATING_TOOL_NAMES: frozenset[str] = frozenset(
    {"create_plan", "apply_plan", "export_clean_copies"}
)


class PathBoundaryHook:
    """Normalizes paths via `realpath`, blocks `../` traversal, enforces roots, and locks `DCIM/` read-only."""

    name = "PathBoundaryHook"

    def __init__(self, *, authorized_dirs: Sequence[Path | str] = ()) -> None:
        self._authorized_dirs: tuple[Path, ...] = tuple(
            Path(os.path.realpath(os.path.expanduser(str(p)))) for p in authorized_dirs
        )

    def _allowed_roots(self, ctx: ToolContext) -> list[Path]:
        roots: list[Path] = list(self._authorized_dirs)
        roots.append(Path(os.path.realpath(str(ctx.services.app_paths.root))))
        for sess in ctx.services.review.list_sessions():
            roots.append(Path(os.path.realpath(sess.root_path)))
        for plan in ctx.services.plan.list_plans():
            if plan.quarantine_dir:
                roots.append(Path(os.path.realpath(plan.quarantine_dir)))
        return roots

    def check(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> HookResult:
        is_mutating = spec.name in _MUTATING_TOOL_NAMES or spec.risk_level in {"MOVE", "WRITE_NEW"}

        for key in _PATH_ARG_KEYS:
            raw = arguments.get(key)
            if not isinstance(raw, str) or not raw.strip():
                continue
            raw_parts = Path(raw).parts
            if ".." in raw_parts:
                return HookResult.deny(
                    f"Path traversal ('..') is forbidden in {key!r}.",
                    hook_name=self.name,
                )
            real = Path(os.path.realpath(os.path.expanduser(raw)))
            if is_mutating and _has_dcim_segment(str(real)):
                return HookResult.deny(
                    f"Camera card directory (DCIM/) is read-only: {real}",
                    hook_name=self.name,
                )
            # `scan_folder` registers new roots; other tools must stay within registered/authorized dirs
            if spec.name != "scan_folder":
                allowed = self._allowed_roots(ctx)
                if not any(_is_within(real, root) for root in allowed):
                    return HookResult.deny(
                        f"Path {real} is outside registered session or authorized directories.",
                        hook_name=self.name,
                    )

        # Also lock DCIM/ read-only when creating or applying plans against a DCIM session
        if is_mutating:
            session_id = arguments.get("session_id")
            if not session_id and isinstance(arguments.get("plan_id"), str):
                try:
                    plan, _ = ctx.services.plan.get_plan(arguments["plan_id"])
                    session_id = plan.session_id
                except PlanError:
                    session_id = None
            if isinstance(session_id, str) and session_id:
                try:
                    summary = ctx.services.review.get_session_summary(session_id)
                    if _has_dcim_segment(summary.session.root_path):
                        return HookResult.deny(
                            "Camera card session (DCIM/) is strictly read-only; cannot move files.",
                            hook_name=self.name,
                        )
                except ValueError:
                    pass

        return HookResult.allow(self.name)


# ---------------------------------------------------------------------------
# 3. BudgetGuardHook (pre_tool_use)
# ---------------------------------------------------------------------------


class BudgetGuardHook:
    """Enforces session & monthly USD budgets on `READ_COST` / `COMPUTE_COST` tools (returns `ASK`)."""

    name = "BudgetGuardHook"

    def check(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> HookResult:
        if ctx.budget_ask_approved:
            return HookResult.allow(self.name)
        if spec.risk_level == "READ_COST":
            # MCP image tools send bytes to the host AI (zero Lightbox LLM spend);
            # only enforce Lightbox's LLM budget on built-in Agent visual calls.
            if not ctx.surface.startswith("agent"):
                return HookResult.allow(self.name)
            if spec.name == "get_photo" and arguments.get("image", "none") == "none":
                return HookResult.allow(self.name)
        elif spec.risk_level == "COMPUTE_COST":
            if spec.name == "run_triage" and arguments.get("dry_run", True) is True:
                return HookResult.allow(self.name)
        else:
            return HookResult.allow(self.name)

        cfg_budget = ctx.services.config.budget
        session_limit = (
            ctx.session_budget_override_usd
            if ctx.session_budget_override_usd is not None
            else cfg_budget.session_budget_usd
        )
        session_id = arguments.get("session_id") or ctx.session_id
        est_usd = 0.0
        if spec.name == "run_triage" and isinstance(session_id, str):
            try:
                est = ctx.services.triage.estimate_triage(
                    session_id,
                    max_photos=int(arguments.get("max_photos", 50)),
                )
                est_usd = est.est_usd
            except Exception:
                est_usd = 0.0

        if isinstance(session_id, str) and session_id:
            session_spent = ctx.services.repo.sum_llm_cost(session_id=session_id)
            if (
                session_spent >= session_limit
                or session_spent + est_usd > session_limit
            ):
                suggested = round(session_spent + est_usd + 0.50, 2)
                return HookResult.ask(
                    f"Session budget exceeded (${session_spent:.4f} spent + ${est_usd:.4f} est > "
                    f"${session_limit:.2f} limit). Approve to temporarily raise this session's limit to ${suggested:.2f}.",
                    hook_name=self.name,
                    payload={
                        "scope": "session",
                        "spent_usd": round(session_spent, 6),
                        "est_usd": round(est_usd, 6),
                        "budget_usd": session_limit,
                        "suggested_budget_usd": suggested,
                    },
                )

        if not ctx.bypass_monthly_budget:
            monthly_spent = ctx.services.repo.sum_llm_cost(since_ts=_month_start_ts())
            if (
                monthly_spent >= cfg_budget.monthly_budget_usd
                or monthly_spent + est_usd > cfg_budget.monthly_budget_usd
            ):
                suggested = round(
                    (
                        ctx.services.repo.sum_llm_cost(session_id=session_id)
                        if isinstance(session_id, str) and session_id
                        else 0.0
                    )
                    + est_usd
                    + 0.50,
                    2,
                )
                return HookResult.ask(
                    f"Monthly budget exceeded (${monthly_spent:.4f} spent + ${est_usd:.4f} est > "
                    f"${cfg_budget.monthly_budget_usd:.2f} limit). Approve to allow this run with a one-time override (session cap ${suggested:.2f}).",
                    hook_name=self.name,
                    payload={
                        "scope": "monthly",
                        "spent_usd": round(monthly_spent, 6),
                        "est_usd": round(est_usd, 6),
                        "budget_usd": cfg_budget.monthly_budget_usd,
                        "suggested_budget_usd": suggested,
                    },
                )

        return HookResult.allow(self.name)


# ---------------------------------------------------------------------------
# 4. ApprovalGateHook (pre_tool_use)
# ---------------------------------------------------------------------------


class ApprovalGateHook:
    """Verifies human approval record and `plan_hash` freshness before `apply_plan` / export."""

    name = "ApprovalGateHook"

    def check(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> HookResult:
        if spec.name not in {"apply_plan", "export_clean_copies"} and spec.risk_level != "MOVE":
            return HookResult.allow(self.name)

        plan_id = arguments.get("plan_id")
        if not isinstance(plan_id, str) or not plan_id.strip():
            return HookResult.deny("Missing required `plan_id` for approval gate.", hook_name=self.name)

        try:
            plan, _items = ctx.services.plan.verify_plan_freshness(plan_id, check_hash=True)
        except StalePlanError as exc:
            return HookResult.deny(
                f"Plan {plan_id} is stale or plan_hash changed: {exc}",
                hook_name=self.name,
            )
        except PlanError as exc:
            return HookResult.deny(str(exc), hook_name=self.name)

        if plan.status == "approved" and plan.approved_by and plan.approved_at is not None:
            return HookResult.allow(self.name)

        # Allow through when caller presents an active CLI 6-digit code or MCP Elicitation channel
        code = arguments.get("confirmation_code")
        if isinstance(code, str) and code.strip():
            if plan.approval_code_hash is None:
                return HookResult.deny(
                    f"No active CLI approval code for plan {plan_id}.",
                    hook_name=self.name,
                )
            if plan.approval_expires_at is None or time.time() > plan.approval_expires_at:
                return HookResult.deny(
                    f"CLI approval code for plan {plan_id} has expired.",
                    hook_name=self.name,
                )
            return HookResult.allow(self.name)

        if ctx.elicitation_approver is not None:
            return HookResult.allow(self.name)

        approve_url = ctx.approve_url(plan.session_id, plan.id)
        return HookResult.deny(
            f"Human approval is required before applying plan {plan.id}. "
            f"Approve in Web UI ({approve_url}) or run `lightbox approve {plan.id}`.",
            hook_name=self.name,
            payload={"plan_id": plan.id, "plan_hash": plan.plan_hash, "approve_url": approve_url},
        )


# ---------------------------------------------------------------------------
# 5. OutboundPrivacyHook (pre_model_call)
# ---------------------------------------------------------------------------


class OutboundPrivacyHook:
    """Validates outbound LLM payloads: EXIF-free images, no absolute paths, no API keys."""

    name = "OutboundPrivacyHook"

    def check(self, request: ModelCallRequest) -> HookResult:
        decision, reason = check_outbound_privacy(request)
        if decision == "ALLOW":
            return HookResult.allow(self.name)
        if decision == "ASK":
            return HookResult.ask(reason or "Outbound privacy confirmation required.", hook_name=self.name)
        return HookResult.deny(reason or "Blocked by OutboundPrivacyHook.", hook_name=self.name)


# ---------------------------------------------------------------------------
# 6. AuditLogHook (post_tool_use)
# ---------------------------------------------------------------------------


class AuditLogHook:
    """Writes redacted audit entries to SQLite `audit_logs` via `SystemService`."""

    name = "AuditLogHook"

    def on_result(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        result: ToolExecutionResult,
        ctx: ToolContext,
        *,
        duration_ms: int = 0,
    ) -> None:
        tool_log_name = spec.mcp_name or spec.name
        redacted = redact_tool_args(arguments)
        decision = "deny" if result.is_error else "allow"
        summary_text = (
            result.error_message
            if result.is_error
            else json.dumps(
                {k: v for k, v in result.data.items() if k not in ("items", "frames", "matches")},
                ensure_ascii=False,
            )[:240]
        )
        ctx.services.system.record_audit_log(
            surface=ctx.surface,
            tool_name=tool_log_name,
            hook_decision=decision,
            args_redacted=redacted,
            duration_ms=duration_ms,
            summary=summary_text,
        )


# ---------------------------------------------------------------------------
# HookRegistry & Attachment Helpers
# ---------------------------------------------------------------------------


class HookRegistry:
    """Central registry orchestrating `pre_tool_use`, `post_tool_use`, and `pre_model_call` hooks."""

    def __init__(
        self,
        *,
        pre_tool_hooks: Sequence[PreToolHook] = (),
        post_tool_hooks: Sequence[PostToolHook] = (),
        pre_model_hooks: Sequence[PreModelHook] = (),
    ) -> None:
        self.pre_tool_hooks: list[PreToolHook] = list(pre_tool_hooks)
        self.post_tool_hooks: list[PostToolHook] = list(post_tool_hooks)
        self.pre_model_hooks: list[PreModelHook] = list(pre_model_hooks)

    def run_pre_tool_use(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        ctx: ToolContext,
    ) -> HookResult:
        for hook in self.pre_tool_hooks:
            res = hook.check(spec, arguments, ctx)
            if res.decision != HookDecision.ALLOW:
                return res
        return HookResult.allow()

    def run_post_tool_use(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        result: ToolExecutionResult,
        ctx: ToolContext,
        *,
        duration_ms: int = 0,
    ) -> None:
        for hook in self.post_tool_hooks:
            hook.on_result(spec, arguments, result, ctx, duration_ms=duration_ms)

    def run_pre_model_call(self, request: ModelCallRequest) -> HookResult:
        for hook in self.pre_model_hooks:
            res = hook.check(request)
            if res.decision != HookDecision.ALLOW:
                return res
        return HookResult.allow()

    def attach_to_tool_registry(self, registry: ToolRegistry) -> None:
        """Mount all pre/post tool hooks onto a `ToolRegistry` instance."""

        def _pre_adapter(
            spec: ToolSpec,
            arguments: dict[str, Any],
            ctx: ToolContext,
        ) -> tuple[Literal["allow", "deny", "ask"], str | None]:
            res = self.run_pre_tool_use(spec, arguments, ctx)
            mapped: Literal["allow", "deny", "ask"] = (
                "allow"
                if res.decision == HookDecision.ALLOW
                else ("ask" if res.decision == HookDecision.ASK else "deny")
            )
            return mapped, res.reason

        def _post_adapter(
            spec: ToolSpec,
            arguments: dict[str, Any],
            result: ToolExecutionResult,
            ctx: ToolContext,
        ) -> None:
            self.run_post_tool_use(spec, arguments, result, ctx)

        registry.add_pre_hook(_pre_adapter)
        registry.add_post_hook(_post_adapter)
        if any(isinstance(h, AuditLogHook) for h in self.post_tool_hooks):
            registry.mark_audit_hook_attached()

    def attach_to_model_router(self, router: ModelRouter) -> None:
        """Mount all `pre_model_call` hooks onto a `ModelRouter` instance."""
        hook_key = ",".join(h.name for h in self.pre_model_hooks) or f"hooks:{id(self)}"

        def _model_adapter(
            request: ModelCallRequest,
        ) -> tuple[Literal["ALLOW", "DENY", "ASK"], str | None]:
            res = self.run_pre_model_call(request)
            return res.decision.value, res.reason

        router.add_pre_model_call_hook(_model_adapter, name=hook_key)


def create_default_hook_registry(
    *,
    authorized_dirs: Sequence[Path | str] = (),
) -> HookRegistry:
    """Create the canonical Lightbox `HookRegistry` with all 6 built-in governance hooks."""
    return HookRegistry(
        pre_tool_hooks=(
            NoHardDeleteHook(),
            PathBoundaryHook(authorized_dirs=authorized_dirs),
            BudgetGuardHook(),
            ApprovalGateHook(),
        ),
        post_tool_hooks=(AuditLogHook(),),
        pre_model_hooks=(OutboundPrivacyHook(),),
    )


def attach_default_hooks(
    *,
    registry: ToolRegistry | None = None,
    router: ModelRouter | None = None,
    authorized_dirs: Sequence[Path | str] = (),
) -> HookRegistry:
    """Create and attach the 6 built-in hooks to the given `ToolRegistry` and/or `ModelRouter`."""
    hooks = create_default_hook_registry(authorized_dirs=authorized_dirs)
    if registry is not None:
        hooks.attach_to_tool_registry(registry)
    if router is not None:
        hooks.attach_to_model_router(router)
    return hooks


__all__ = [
    "ApprovalGateHook",
    "AuditLogHook",
    "BudgetGuardHook",
    "HookDecision",
    "HookRegistry",
    "HookResult",
    "NoHardDeleteHook",
    "OutboundPrivacyHook",
    "PathBoundaryHook",
    "PostToolHook",
    "PreModelHook",
    "PreToolHook",
    "attach_default_hooks",
    "create_default_hook_registry",
    "wrap_untrusted_metadata",
]
