"""Self-built single-main Agent loop supporting `chat` and `supervisor` modes (SPEC §7.1–7.2, PLAN T13)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from lightbox.agent.context import (
    STAGE_REBUILD_TOKEN_LIMIT,
    SUPERVISOR_GOAL_PROMPT,
    AgentStage,
    alias_paths_in_text,
    build_agent_tool_schemas,
    build_stage_anchor,
    build_stage_messages,
    estimate_messages_tokens,
    next_stage,
    redact_path_aliases_in_text,
    resolve_path_aliases_in_args,
)
from lightbox.agent.hooks import attach_default_hooks
from lightbox.agent.memory import extract_effective_preferences_for_prompt
from lightbox.models import ImageInput, ModelError, ToolCall
from lightbox.services import LightboxServices
from lightbox.store import AgentEventRecord, AgentSessionRecord
from lightbox.tools import ToolContext, ToolExecutionResult, ToolRegistry

AgentMode = Literal["chat", "supervisor"]
AgentStatus = Literal["running", "awaiting_user", "completed", "failed"]
AgentEventCallback = Callable[[str, dict[str, Any]], None]

DEFAULT_MAX_STEPS = 12
MAX_TOOL_RESULT_CHARS = 8000
ASK_BUDGET_BUMP_USD = 0.50

# In-process active-session lock prevents concurrent double-runs of the same agent_session_id,
# while allowing orphaned DB `status="running"` sessions after a process crash to auto-recover.
_ACTIVE_SESSIONS_LOCK = threading.Lock()
_ACTIVE_AGENT_SESSIONS: set[str] = set()


class AgentSessionBusyError(RuntimeError):
    """Raised when a caller tries to send a message to an Agent session that is actively running."""


@contextmanager
def _acquire_active_session(agent_session_id: str) -> Iterator[None]:
    with _ACTIVE_SESSIONS_LOCK:
        if agent_session_id in _ACTIVE_AGENT_SESSIONS:
            raise AgentSessionBusyError(
                f"Agent session {agent_session_id} is currently running; wait for the current turn to finish."
            )
        _ACTIVE_AGENT_SESSIONS.add(agent_session_id)
    try:
        yield
    finally:
        with _ACTIVE_SESSIONS_LOCK:
            _ACTIVE_AGENT_SESSIONS.discard(agent_session_id)


def _month_start_ts() -> float:
    now = datetime.now(tz=UTC)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Snapshot returned after starting or continuing an Agent session."""

    agent_session_id: str
    session_id: str
    mode: str
    status: str
    stage: str
    step: int
    answer: str
    todos: tuple[dict[str, Any], ...]
    constraints: tuple[str, ...]
    spent_usd: float
    viewed_photos: int
    context_tokens_est: int
    stage_rebuilds: int
    pending_ask: dict[str, Any] | None
    latest_plan: dict[str, Any] | None
    verifier_report: dict[str, Any] | None
    error: str | None = None
    events: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_session_id": self.agent_session_id,
            "session_id": self.session_id,
            "mode": self.mode,
            "status": self.status,
            "stage": self.stage,
            "step": self.step,
            "answer": self.answer,
            "todos": list(self.todos),
            "constraints": list(self.constraints),
            "spent_usd": round(self.spent_usd, 6),
            "viewed_photos": self.viewed_photos,
            "context_tokens_est": self.context_tokens_est,
            "stage_rebuilds": self.stage_rebuilds,
            "pending_ask": self.pending_ask,
            "latest_plan": self.latest_plan,
            "verifier_report": self.verifier_report,
            "error": self.error,
            "events": list(self.events),
        }


def _serialize_tool_result(res: ToolExecutionResult, path_aliases: dict[str, str]) -> str:
    payload = dict(res.data)
    raw = json.dumps(payload, ensure_ascii=False)
    safe = redact_path_aliases_in_text(raw, path_aliases)
    if len(safe) <= MAX_TOOL_RESULT_CHARS:
        return safe
    return safe[: MAX_TOOL_RESULT_CHARS - 16] + "...(truncated)"


def _serialize_events(events: Sequence[AgentEventRecord]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "seq": e.seq,
            "event_type": e.event_type,
            "payload": e.payload,
            "created_at": e.created_at,
        }
        for e in events
    )


def _is_affirmative_reply(text: str) -> bool:
    cleaned = text.strip().lower()
    return cleaned in {
        "y",
        "yes",
        "ok",
        "okay",
        "approve",
        "approved",
        "continue",
        "proceed",
        "确认",
        "同意",
        "继续",
        "好",
        "好的",
        "是",
    }


class AgentRunner:
    """Single-main Agent orchestrator with path aliasing, frozen stage anchors, concurrency lock, and ASK resume."""

    def __init__(
        self,
        services: LightboxServices,
        *,
        registry: ToolRegistry | None = None,
        base_url_resolver: Callable[[], str] | None = None,
        lang: str = "zh-CN",
    ) -> None:
        self.services = services
        self.registry = registry or ToolRegistry()
        self.base_url_resolver = base_url_resolver
        self.lang = lang
        attach_default_hooks(router=services.model_router)

    def _read_preferences(self) -> str:
        return extract_effective_preferences_for_prompt(self.services.app_paths)

    def _emit(
        self,
        agent_session_id: str,
        event_type: str,
        payload: dict[str, Any],
        on_event: AgentEventCallback | None,
    ) -> AgentEventRecord:
        rec = self.services.repo.append_agent_event(agent_session_id, event_type, payload)
        if on_event is not None:
            on_event(
                event_type,
                {
                    "agent_session_id": agent_session_id,
                    "seq": rec.seq,
                    "event_type": event_type,
                    **payload,
                },
            )
        return rec

    def _freeze_stage_anchor(
        self,
        agent_session_id: str,
        stage: AgentStage,
        stage_summary: dict[str, Any],
        *,
        latest_user_message: str,
        milestone_summary: str | None = None,
        on_event: AgentEventCallback | None = None,
    ) -> int:
        """Build and persist an immutable `stage_anchor` event once on stage entry; returns its seq."""
        sess = self.services.repo.get_agent_session(agent_session_id)
        assert sess is not None
        # Ensure anchor reflects the latest stage_summary state
        sess_snapshot = AgentSessionRecord(
            id=sess.id,
            session_id=sess.session_id,
            mode=sess.mode,
            status=sess.status,
            constraints=sess.constraints,
            plan_todos=sess.plan_todos,
            stage_summary=stage_summary,
            step=sess.step,
            spent_usd=sess.spent_usd,
            created_at=sess.created_at,
        )
        anchor_text = build_stage_anchor(
            self.services,
            sess_snapshot,
            stage=stage,
            latest_user_message=latest_user_message,
            milestone_summary=milestone_summary,
        )
        evt = self._emit(
            agent_session_id,
            "stage_anchor",
            {"stage": stage, "text": anchor_text},
            on_event,
        )
        stage_summary["stage"] = stage
        stage_summary["stage_start_seq"] = evt.seq
        return evt.seq

    def _build_snapshot(
        self,
        agent_session_id: str,
        *,
        answer: str | None = None,
        error: str | None = None,
    ) -> AgentRunResult:
        sess = self.services.repo.get_agent_session(agent_session_id)
        if sess is None:
            raise ValueError(f"Unknown agent_session_id: {agent_session_id}")
        events = self.services.repo.list_agent_events(agent_session_id)
        ss = sess.stage_summary
        resolved_answer = (
            answer if answer is not None else str(ss.get("last_answer", "") or "")
        )
        return AgentRunResult(
            agent_session_id=sess.id,
            session_id=sess.session_id,
            mode=sess.mode,
            status=sess.status,
            stage=str(ss.get("stage", "explore")),
            step=sess.step,
            answer=resolved_answer,
            todos=sess.plan_todos,
            constraints=sess.constraints,
            spent_usd=sess.spent_usd,
            viewed_photos=int(ss.get("viewed_photos", 0) or 0),
            context_tokens_est=int(ss.get("context_tokens_est", 0) or 0),
            stage_rebuilds=int(ss.get("stage_rebuilds", 0) or 0),
            pending_ask=ss.get("pending_ask"),
            latest_plan=ss.get("latest_plan"),
            verifier_report=ss.get("verifier_report"),
            error=error or ss.get("error"),
            events=_serialize_events(events),
        )

    def get_session(self, agent_session_id: str) -> AgentRunResult:
        return self._build_snapshot(agent_session_id)

    def _compute_budget_ask_details(
        self,
        sess: AgentSessionRecord,
        call: ToolCall,
        stage_summary: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute temporary budget bump details for a BudgetGuardHook ASK payload."""
        cfg_budget = self.services.config.budget
        cur_limit = float(
            stage_summary.get("session_budget_override_usd") or cfg_budget.session_budget_usd
        )
        est_usd = 0.0
        if call.name == "run_triage":
            sid = str(call.arguments.get("session_id") or sess.session_id)
            try:
                est = self.services.triage.estimate_triage(
                    sid,
                    max_photos=int(call.arguments.get("max_photos", 50)),
                )
                est_usd = est.est_usd
                if est.requires_unpriced_confirmation:
                    session_spent = self.services.repo.sum_llm_cost(session_id=sess.session_id)
                    monthly_spent = self.services.repo.sum_llm_cost(since_ts=_month_start_ts())
                    return {
                        "scope": "unpriced_triage",
                        "ask_kind": "unpriced_triage",
                        "model_id": est.model_id,
                        "unpriced_max_photos": est.unpriced_max_photos or 20,
                        "total_units": est.total_units,
                        "session_spent_usd": round(session_spent, 6),
                        "monthly_spent_usd": round(monthly_spent, 6),
                        "est_usd": 0.0,
                        "suggested_budget_usd": None,
                        "needs_monthly_bypass": False,
                    }
            except Exception:
                est_usd = 0.0

        session_spent = self.services.repo.sum_llm_cost(session_id=sess.session_id)
        monthly_spent = self.services.repo.sum_llm_cost(since_ts=_month_start_ts())
        exceeded_monthly = (
            monthly_spent >= cfg_budget.monthly_budget_usd
            or monthly_spent + est_usd > cfg_budget.monthly_budget_usd
        )
        suggested_budget = round(max(cur_limit, session_spent + est_usd) + ASK_BUDGET_BUMP_USD, 2)
        return {
            "scope": "monthly" if exceeded_monthly else "session",
            "session_spent_usd": round(session_spent, 6),
            "monthly_spent_usd": round(monthly_spent, 6),
            "est_usd": round(est_usd, 6),
            "suggested_budget_usd": suggested_budget,
            "needs_monthly_bypass": exceeded_monthly,
        }

    def _execute_single_tool(
        self,
        call: ToolCall,
        *,
        allowed_tools: set[str],
        tctx: ToolContext,
        agent_session_id: str,
        stage_summary: dict[str, Any],
        on_event: AgentEventCallback | None,
    ) -> tuple[ToolExecutionResult, tuple[ImageInput, ...]]:
        """Execute a single tool call through whitelist, local path alias resolution, and HookRegistry."""
        if call.name not in allowed_tools:
            reason = (
                f"Tool {call.name!r} is forbidden by Agent tool whitelist (SPEC §7.1/§7.3)."
            )
            self.services.system.record_audit_log(
                surface=tctx.surface,
                tool_name=call.name,
                hook_decision="deny",
                args_redacted={"blocked_by": "agent_whitelist"},
                duration_ms=0,
                summary=reason,
            )
            res = ToolExecutionResult.error(reason, hook_decision="DENY")
        else:
            raw_aliases = stage_summary.get("path_aliases")
            path_aliases = dict(raw_aliases) if isinstance(raw_aliases, dict) else {}
            resolved_args = resolve_path_aliases_in_args(call.arguments, path_aliases)
            res = self.registry.execute(call.name, resolved_args, tctx)

        images = tuple(
            ImageInput(data=img.data, mime_type=img.mime_type, label=img.label, sanitized=True)
            for img in res.images
        )

        # Capture Verifier report & draft plan handle whenever `create_plan` succeeds
        if call.name == "create_plan" and not res.is_error:
            vrep = res.data.get("verifier_report")
            if isinstance(vrep, dict):
                stage_summary["verifier_report"] = vrep
                self._emit(
                    agent_session_id,
                    "verifier_report",
                    {"plan_id": res.data.get("plan_id"), "verifier_report": vrep},
                    on_event,
                )
            if res.data.get("plan_id"):
                stage_summary["latest_plan"] = {
                    "plan_id": res.data.get("plan_id"),
                    "plan_hash": res.data.get("plan_hash"),
                    "count": res.data.get("count", 0),
                    "status": res.data.get("status"),
                    "approve_url": res.data.get("approve_url"),
                }

        stage_summary["viewed_photos"] = tctx.viewed_photo_count
        if tctx.saved_sets:
            stage_summary["saved_sets"] = dict(tctx.saved_sets)
        return res, images

    async def start_session(
        self,
        session_id: str,
        *,
        mode: AgentMode = "chat",
        message: str | None = None,
        constraints: Sequence[str] = (),
        max_steps: int = DEFAULT_MAX_STEPS,
        stage_token_limit: int = STAGE_REBUILD_TOKEN_LIMIT,
        profile: str | None = None,
        on_event: AgentEventCallback | None = None,
    ) -> AgentRunResult:
        """Create and run a new `chat` or `supervisor` Agent session."""
        self.services.review.get_session_summary(session_id)

        raw_msg = (
            (message or "").strip()
            if mode == "chat"
            else (message.strip() if message and message.strip() else SUPERVISOR_GOAL_PROMPT)
        )
        if not raw_msg:
            raise ValueError("Initial message is required for chat mode.")

        known_sessions = self.services.review.list_sessions()
        masked_msg, path_aliases = alias_paths_in_text(
            raw_msg,
            {},
            known_sessions=known_sessions,
        )
        masked_constraints: list[str] = []
        for c in constraints:
            cleaned = c.strip()
            if cleaned:
                mc, path_aliases = alias_paths_in_text(
                    cleaned,
                    path_aliases,
                    known_sessions=known_sessions,
                )
                masked_constraints.append(mc)

        agent_session_id = f"agent_{uuid.uuid4().hex[:12]}"
        stage_summary: dict[str, Any] = {
            "stage": "explore",
            "stage_start_seq": 0,
            "stage_rebuilds": 0,
            "latest_user_message": masked_msg,
            "path_aliases": path_aliases,
            "viewed_photos": 0,
            "saved_sets": {},
            "context_tokens_est": 0,
            "pending_ask": None,
            "latest_plan": None,
            "verifier_report": None,
            "session_budget_override_usd": None,
            "bypass_monthly_budget": False,
            "last_answer": "",
            "error": None,
        }
        self.services.repo.create_agent_session(
            AgentSessionRecord(
                id=agent_session_id,
                session_id=session_id,
                mode=mode,
                status="running",
                constraints=tuple(masked_constraints),
                plan_todos=(),
                stage_summary=stage_summary,
                step=0,
                spent_usd=0.0,
                created_at=time.time(),
            )
        )
        with _acquire_active_session(agent_session_id):
            self._emit(
                agent_session_id,
                "session_started",
                {
                    "session_id": session_id,
                    "mode": mode,
                    "stage": "explore",
                    "message": raw_msg,
                    "constraints": masked_constraints,
                },
                on_event,
            )
            # Freeze the initial `explore` stage anchor once (append-only for the rest of the stage)
            self._freeze_stage_anchor(
                agent_session_id,
                "explore",
                stage_summary,
                latest_user_message=masked_msg,
                on_event=on_event,
            )
            self.services.repo.update_agent_session(
                agent_session_id,
                stage_summary=stage_summary,
            )
            return await self._run_loop(
                agent_session_id,
                max_steps=max_steps,
                stage_token_limit=stage_token_limit,
                profile=profile,
                on_event=on_event,
            )

    def _recover_orphaned_running_session(
        self,
        sess: AgentSessionRecord,
        stage_summary: dict[str, Any],
        on_event: AgentEventCallback | None,
    ) -> None:
        """Auto-recover a session orphaned in DB `running` state after a prior process crash."""
        stage_start_seq = int(stage_summary.get("stage_start_seq", 0) or 0)
        events = self.services.repo.list_agent_events(sess.id)
        turn_events = [
            e
            for e in events
            if e.seq >= stage_start_seq
            and e.event_type in {"assistant_turn", "tool_results_turn"}
        ]
        if turn_events and turn_events[-1].event_type == "assistant_turn":
            dangling_calls = turn_events[-1].payload.get("tool_calls", [])
            if isinstance(dangling_calls, list) and dangling_calls:
                paired = [
                    {
                        "tool_call_id": str(tc["id"]),
                        "name": str(tc.get("name", "tool")),
                        "content": json.dumps(
                            {"error": "Previous run was interrupted before this tool completed."},
                            ensure_ascii=False,
                        ),
                        "is_error": True,
                    }
                    for tc in dangling_calls
                    if isinstance(tc, dict) and tc.get("id")
                ]
                if paired:
                    self._emit(
                        sess.id,
                        "tool_results_turn",
                        {"tool_results": paired, "text": "", "recovered": True},
                        on_event,
                    )
        self._emit(
            sess.id,
            "session_recovered",
            {"previous_status": sess.status},
            on_event,
        )

    async def continue_session(
        self,
        agent_session_id: str,
        user_message: str,
        *,
        approve_ask: bool | None = None,
        constraints: Sequence[str] = (),
        max_steps: int = DEFAULT_MAX_STEPS,
        stage_token_limit: int = STAGE_REBUILD_TOKEN_LIMIT,
        profile: str | None = None,
        on_event: AgentEventCallback | None = None,
    ) -> AgentRunResult:
        """Resume an existing session with in-process concurrency guard and crash auto-recovery."""
        with _acquire_active_session(agent_session_id):
            sess = self.services.repo.get_agent_session(agent_session_id)
            if sess is None:
                raise ValueError(f"Unknown agent_session_id: {agent_session_id}")

            stage_summary = dict(sess.stage_summary)
            # If DB says `running` but we acquired the in-process slot, a previous process crashed -> auto-recover!
            if sess.status == "running":
                self._recover_orphaned_running_session(sess, stage_summary, on_event)

            known_sessions = self.services.review.list_sessions()
            raw_aliases = stage_summary.get("path_aliases")
            path_aliases: dict[str, str] = dict(raw_aliases) if isinstance(raw_aliases, dict) else {}

            raw_reply = user_message.strip()
            masked_reply, path_aliases = alias_paths_in_text(
                raw_reply,
                path_aliases,
                known_sessions=known_sessions,
            )

            merged_constraints = list(sess.constraints)
            for c in constraints:
                cleaned = c.strip()
                if cleaned:
                    mc, path_aliases = alias_paths_in_text(
                        cleaned,
                        path_aliases,
                        known_sessions=known_sessions,
                    )
                    if mc not in merged_constraints:
                        merged_constraints.append(mc)

            stage_summary["path_aliases"] = path_aliases
            stage_summary["latest_user_message"] = masked_reply or stage_summary.get(
                "latest_user_message", ""
            )
            stage_summary["error"] = None
            transient_images: dict[int, tuple[ImageInput, ...]] = {}

            pending_ask = stage_summary.get("pending_ask")
            if isinstance(pending_ask, dict):
                approved = approve_ask if approve_ask is not None else _is_affirmative_reply(raw_reply)
                if approved:
                    # Honor user's explicit ASK confirmation by raising this session's effective budget
                    suggested = pending_ask.get("suggested_budget_usd")
                    if isinstance(suggested, (int, float)) and suggested > 0:
                        prev_override = stage_summary.get("session_budget_override_usd")
                        stage_summary["session_budget_override_usd"] = max(
                            float(prev_override or 0.0),
                            float(suggested),
                        )
                    if pending_ask.get("needs_monthly_bypass"):
                        stage_summary["bypass_monthly_budget"] = True

                tool_schemas = build_agent_tool_schemas(self.registry)
                allowed_tools = {t.name for t in tool_schemas}
                tctx = self._make_tool_context(
                    sess,
                    stage_summary,
                    on_event,
                    budget_ask_approved=approved,
                )

                combined_results: list[dict[str, Any]] = [
                    dict(r) for r in pending_ask.get("completed_results", []) if isinstance(r, dict)
                ]
                collected_imgs: list[ImageInput] = []

                ask_call_raw = pending_ask.get("tool_call", {})
                ask_call = ToolCall(
                    id=str(ask_call_raw.get("id", "")),
                    name=str(ask_call_raw.get("name", "")),
                    arguments=dict(ask_call_raw.get("arguments", {})),
                )
                trailing_raw = pending_ask.get("remaining_calls", [])
                trailing_calls = [
                    ToolCall(
                        id=str(rc["id"]),
                        name=str(rc["name"]),
                        arguments=dict(rc.get("arguments", {})),
                    )
                    for rc in trailing_raw
                    if isinstance(rc, dict)
                ]

                if approved:
                    for call in [ask_call, *trailing_calls]:
                        res, imgs = self._execute_single_tool(
                            call,
                            allowed_tools=allowed_tools,
                            tctx=tctx,
                            agent_session_id=agent_session_id,
                            stage_summary=stage_summary,
                            on_event=on_event,
                        )
                        collected_imgs.extend(imgs)
                        combined_results.append(
                            {
                                "tool_call_id": call.id,
                                "name": call.name,
                                "content": _serialize_tool_result(res, path_aliases),
                                "is_error": res.is_error,
                            }
                        )
                else:
                    combined_results.append(
                        {
                            "tool_call_id": ask_call.id,
                            "name": ask_call.name,
                            "content": json.dumps(
                                {
                                    "error": "User declined the requested action.",
                                    "user_reply": masked_reply,
                                },
                                ensure_ascii=False,
                            ),
                            "is_error": True,
                        }
                    )
                    for call in trailing_calls:
                        combined_results.append(
                            {
                                "tool_call_id": call.id,
                                "name": call.name,
                                "content": json.dumps(
                                    {"error": "Skipped because user declined prior confirmation."},
                                    ensure_ascii=False,
                                ),
                                "is_error": True,
                            }
                        )

                evt = self._emit(
                    agent_session_id,
                    "tool_results_turn",
                    {
                        "tool_results": combined_results,
                        "text": masked_reply,
                        "ask_approved": approved,
                    },
                    on_event,
                )
                if collected_imgs:
                    transient_images[evt.seq] = tuple(collected_imgs)
                stage_summary["pending_ask"] = None
            else:
                if not masked_reply:
                    raise ValueError("Follow-up message must not be empty.")
                self._emit(
                    agent_session_id,
                    "user_message",
                    {"text": masked_reply, "display_text": raw_reply},
                    on_event,
                )

            self.services.repo.update_agent_session(
                agent_session_id,
                status="running",
                constraints=merged_constraints,
                stage_summary=stage_summary,
            )
            return await self._run_loop(
                agent_session_id,
                max_steps=max_steps,
                stage_token_limit=stage_token_limit,
                profile=profile,
                on_event=on_event,
                transient_images=transient_images,
            )

    def _make_tool_context(
        self,
        sess: AgentSessionRecord,
        stage_summary: dict[str, Any],
        on_event: AgentEventCallback | None,
        *,
        budget_ask_approved: bool = False,
    ) -> ToolContext:
        saved_sets_raw = stage_summary.get("saved_sets")
        saved_sets: dict[str, list[str]] = (
            {str(k): list(v) for k, v in saved_sets_raw.items() if isinstance(v, list)}
            if isinstance(saved_sets_raw, dict)
            else {}
        )

        def _on_todos(todos: list[dict[str, Any]]) -> None:
            self._emit(sess.id, "plan_todos", {"todos": todos}, on_event)

        raw_override = stage_summary.get("session_budget_override_usd")
        budget_override = float(raw_override) if isinstance(raw_override, (int, float)) else None

        return ToolContext(
            services=self.services,
            surface=f"agent:{sess.mode}",
            base_url_resolver=self.base_url_resolver,
            lang=self.lang,
            session_id=sess.session_id,
            agent_session_id=sess.id,
            viewed_photo_count=int(stage_summary.get("viewed_photos", 0) or 0),
            max_viewed_photos=40,
            saved_sets=saved_sets,
            on_update_plan=_on_todos,
            budget_ask_approved=budget_ask_approved,
            session_budget_override_usd=budget_override,
            bypass_monthly_budget=bool(stage_summary.get("bypass_monthly_budget", False)),
        )

    async def _run_loop(
        self,
        agent_session_id: str,
        *,
        max_steps: int,
        stage_token_limit: int,
        profile: str | None,
        on_event: AgentEventCallback | None,
        transient_images: dict[int, tuple[ImageInput, ...]] | None = None,
    ) -> AgentRunResult:
        tool_schemas = build_agent_tool_schemas(self.registry)
        allowed_tools = {t.name for t in tool_schemas}
        preferences_text = self._read_preferences()
        img_cache: dict[int, tuple[ImageInput, ...]] = dict(transient_images or {})

        steps_taken = 0
        while steps_taken < max_steps:
            sess = self.services.repo.get_agent_session(agent_session_id)
            assert sess is not None
            stage_summary = dict(sess.stage_summary)
            raw_aliases = stage_summary.get("path_aliases")
            path_aliases: dict[str, str] = dict(raw_aliases) if isinstance(raw_aliases, dict) else {}
            events = self.services.repo.list_agent_events(agent_session_id)

            stage_str = str(stage_summary.get("stage", "explore"))
            cur_stage: AgentStage = (
                "propose"
                if stage_str == "propose"
                else ("review" if stage_str == "review" else "explore")
            )
            stage_start_seq = int(stage_summary.get("stage_start_seq", 0) or 0)
            latest_user_msg = str(stage_summary.get("latest_user_message", ""))

            messages = build_stage_messages(
                self.services,
                sess,
                events,
                stage=cur_stage,
                stage_start_seq=stage_start_seq,
                latest_user_message=latest_user_msg,
                preferences_text=preferences_text,
                transient_images=img_cache,
            )
            est_tokens = estimate_messages_tokens(messages)

            # Forced deterministic stage-boundary rebuild when stage context exceeds token threshold
            has_stage_turns = any(
                e.seq > stage_start_seq and e.event_type in {"assistant_turn", "tool_results_turn"}
                for e in events
            )
            if est_tokens > stage_token_limit and has_stage_turns:
                new_stage = next_stage(cur_stage)
                rebuilds = int(stage_summary.get("stage_rebuilds", 0) or 0) + 1
                stage_summary["stage_rebuilds"] = rebuilds
                self._emit(
                    agent_session_id,
                    "stage_rebuild",
                    {
                        "from_stage": cur_stage,
                        "to_stage": new_stage,
                        "reason": "token_threshold",
                        "prev_tokens_est": est_tokens,
                        "stage_rebuilds": rebuilds,
                    },
                    on_event,
                )
                stage_start_seq = self._freeze_stage_anchor(
                    agent_session_id,
                    new_stage,
                    stage_summary,
                    latest_user_message=latest_user_msg,
                    on_event=on_event,
                )
                cur_stage = new_stage
                self.services.repo.update_agent_session(
                    agent_session_id,
                    stage_summary=stage_summary,
                )
                sess = self.services.repo.get_agent_session(agent_session_id)
                assert sess is not None
                events = self.services.repo.list_agent_events(agent_session_id)
                messages = build_stage_messages(
                    self.services,
                    sess,
                    events,
                    stage=cur_stage,
                    stage_start_seq=stage_start_seq,
                    latest_user_message=latest_user_msg,
                    preferences_text=preferences_text,
                    transient_images=img_cache,
                )
                est_tokens = estimate_messages_tokens(messages)

            stage_summary["context_tokens_est"] = est_tokens
            self.services.repo.update_agent_session(agent_session_id, stage_summary=stage_summary)

            raw_override = stage_summary.get("session_budget_override_usd")
            session_budget_override = (
                float(raw_override) if isinstance(raw_override, (int, float)) else None
            )
            try:
                resp = await self.services.model_router.generate(
                    role="agent",
                    messages=messages,
                    tools=tool_schemas,
                    max_tokens=2048,
                    profile=profile,
                    session_id=sess.session_id,
                    session_budget_usd=session_budget_override,
                    bypass_monthly_budget=bool(stage_summary.get("bypass_monthly_budget", False)),
                )
            except ModelError as exc:
                err_msg = str(exc)
                stage_summary["error"] = err_msg
                self.services.repo.update_agent_session(
                    agent_session_id,
                    status="failed",
                    stage_summary=stage_summary,
                )
                self._emit(agent_session_id, "agent_error", {"error": err_msg}, on_event)
                return self._build_snapshot(agent_session_id, error=err_msg)

            new_spent = sess.spent_usd + resp.cost_usd
            new_step = sess.step + 1
            steps_taken += 1

            self._emit(
                agent_session_id,
                "assistant_turn",
                {
                    "text": resp.text,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in resp.tool_calls
                    ],
                    "model": resp.model,
                    "provider": resp.provider,
                    "cost_usd": resp.cost_usd,
                    "cache_hit": resp.cache_hit,
                    "in_tokens": resp.usage.in_tokens,
                    "out_tokens": resp.usage.out_tokens,
                    "cached_tokens": resp.usage.cached_tokens,
                    "stage": cur_stage,
                },
                on_event,
            )

            if not resp.tool_calls:
                pending_trans = stage_summary.pop("pending_stage_transition", None)
                if isinstance(pending_trans, dict) and isinstance(pending_trans.get("to_stage"), str):
                    final_stage = str(pending_trans["to_stage"])
                    if final_stage != cur_stage:
                        stage_summary["stage"] = final_stage
                        self._emit(
                            agent_session_id,
                            "stage_changed",
                            {"from_stage": cur_stage, "to_stage": final_stage},
                            on_event,
                        )
                        cur_stage = (
                            "propose"
                            if final_stage == "propose"
                            else ("review" if final_stage == "review" else "explore")
                        )
                stage_summary["last_answer"] = resp.text
                self.services.repo.update_agent_session(
                    agent_session_id,
                    status="completed",
                    step=new_step,
                    spent_usd=new_spent,
                    stage_summary=stage_summary,
                )
                self._emit(
                    agent_session_id,
                    "completed",
                    {"answer": resp.text, "stage": cur_stage, "spent_usd": new_spent},
                    on_event,
                )
                return self._build_snapshot(agent_session_id, answer=resp.text)

            # Execute tool calls sequentially; pause cleanly on any `ASK` decision
            tctx = self._make_tool_context(sess, stage_summary, on_event)
            completed_results: list[dict[str, Any]] = []
            turn_images: list[ImageInput] = []
            milestone_stage: AgentStage | None = None
            milestone_notes: list[str] = []

            for idx, call in enumerate(resp.tool_calls):
                res, imgs = self._execute_single_tool(
                    call,
                    allowed_tools=allowed_tools,
                    tctx=tctx,
                    agent_session_id=agent_session_id,
                    stage_summary=stage_summary,
                    on_event=on_event,
                )

                if res.hook_decision == "ASK":
                    remaining = [
                        {"id": rc.id, "name": rc.name, "arguments": rc.arguments}
                        for rc in resp.tool_calls[idx + 1 :]
                    ]
                    budget_details = self._compute_budget_ask_details(sess, call, stage_summary)
                    ask_payload = {
                        "tool_call": {
                            "id": call.id,
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                        "reason": res.error_message or "Confirmation required.",
                        "completed_results": completed_results,
                        "remaining_calls": remaining,
                        **budget_details,
                    }
                    stage_summary["pending_ask"] = ask_payload
                    self.services.repo.update_agent_session(
                        agent_session_id,
                        status="awaiting_user",
                        step=new_step,
                        spent_usd=new_spent,
                        stage_summary=stage_summary,
                    )
                    self._emit(
                        agent_session_id,
                        "ask",
                        {
                            "tool_name": call.name,
                            "reason": ask_payload["reason"],
                            "tool_call_id": call.id,
                            "suggested_budget_usd": budget_details["suggested_budget_usd"],
                        },
                        on_event,
                    )
                    return self._build_snapshot(agent_session_id)

                turn_images.extend(imgs)
                completed_results.append(
                    {
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": _serialize_tool_result(res, path_aliases),
                        "is_error": res.is_error,
                        "hook_decision": res.hook_decision,
                    }
                )

                # Track semantic milestone completions (never drop fresh images mid-turn!)
                if not res.is_error:
                    if call.name == "create_plan":
                        milestone_stage = "propose"
                        milestone_notes.append(
                            f"create_plan -> plan_id={res.data.get('plan_id')}, "
                            f"count={res.data.get('count', 0)}, "
                            f"verifier_passed={res.data.get('verifier_report', {}).get('passed')}, "
                            f"flagged={len(res.data.get('verifier_report', {}).get('flagged_photos', []))}"
                        )
                    elif (
                        call.name == "run_triage"
                        and call.arguments.get("dry_run") is False
                        and cur_stage == "explore"
                    ):
                        milestone_stage = "review"
                        milestone_notes.append(
                            f"run_triage -> status={res.data.get('status')}, "
                            f"reviewed_yellow={res.data.get('reviewed_yellow', 0)}, "
                            f"reviewed_bursts={res.data.get('reviewed_bursts', 0)}"
                        )

            tr_evt = self._emit(
                agent_session_id,
                "tool_results_turn",
                {"tool_results": completed_results, "text": ""},
                on_event,
            )
            if turn_images:
                img_cache[tr_evt.seq] = tuple(turn_images)

            # Persist step & spend before optional milestone stage-boundary rebuild
            self.services.repo.update_agent_session(
                agent_session_id,
                step=new_step,
                spent_usd=new_spent,
                stage_summary=stage_summary,
            )

            # Semantic milestone rebuild (`explore -> review` on run_triage, `-> propose` on create_plan),
            # deferring the rebuild if this turn returned fresh visual images (`view_photos` / `compare_burst`).
            if milestone_stage is not None and milestone_stage != cur_stage:
                if not turn_images:
                    stage_summary.pop("pending_stage_transition", None)
                    rebuilds = int(stage_summary.get("stage_rebuilds", 0) or 0) + 1
                    stage_summary["stage_rebuilds"] = rebuilds
                    self._emit(
                        agent_session_id,
                        "stage_rebuild",
                        {
                            "from_stage": cur_stage,
                            "to_stage": milestone_stage,
                            "reason": "milestone_transition",
                            "prev_tokens_est": est_tokens,
                            "stage_rebuilds": rebuilds,
                        },
                        on_event,
                    )
                    self._freeze_stage_anchor(
                        agent_session_id,
                        milestone_stage,
                        stage_summary,
                        latest_user_message=latest_user_msg,
                        milestone_summary="; ".join(milestone_notes),
                        on_event=on_event,
                    )
                else:
                    stage_summary["pending_stage_transition"] = {
                        "to_stage": milestone_stage,
                        "milestone_summary": "; ".join(milestone_notes),
                    }
                self.services.repo.update_agent_session(
                    agent_session_id,
                    stage_summary=stage_summary,
                )

        # Max steps safety cap reached
        sess = self.services.repo.get_agent_session(agent_session_id)
        assert sess is not None
        stage_summary = dict(sess.stage_summary)
        fallback_answer = (
            stage_summary.get("last_answer")
            or f"Reached maximum step limit ({max_steps}). Review progress above or send another message to continue."
        )
        stage_summary["last_answer"] = fallback_answer
        self.services.repo.update_agent_session(
            agent_session_id,
            status="completed",
            stage_summary=stage_summary,
        )
        return self._build_snapshot(agent_session_id, answer=str(fallback_answer))

    # ------------------------------------------------------------------
    # Synchronous convenience wrappers (safe inside or outside an event loop)
    # ------------------------------------------------------------------

    def start_session_sync(
        self,
        session_id: str,
        *,
        mode: AgentMode = "chat",
        message: str | None = None,
        constraints: Sequence[str] = (),
        max_steps: int = DEFAULT_MAX_STEPS,
        stage_token_limit: int = STAGE_REBUILD_TOKEN_LIMIT,
        profile: str | None = None,
        on_event: AgentEventCallback | None = None,
    ) -> AgentRunResult:
        coro = self.start_session(
            session_id,
            mode=mode,
            message=message,
            constraints=constraints,
            max_steps=max_steps,
            stage_token_limit=stage_token_limit,
            profile=profile,
            on_event=on_event,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    def continue_session_sync(
        self,
        agent_session_id: str,
        user_message: str,
        *,
        approve_ask: bool | None = None,
        constraints: Sequence[str] = (),
        max_steps: int = DEFAULT_MAX_STEPS,
        stage_token_limit: int = STAGE_REBUILD_TOKEN_LIMIT,
        profile: str | None = None,
        on_event: AgentEventCallback | None = None,
    ) -> AgentRunResult:
        coro = self.continue_session(
            agent_session_id,
            user_message,
            approve_ask=approve_ask,
            constraints=constraints,
            max_steps=max_steps,
            stage_token_limit=stage_token_limit,
            profile=profile,
            on_event=on_event,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()


__all__ = [
    "ASK_BUDGET_BUMP_USD",
    "AgentEventCallback",
    "AgentMode",
    "AgentRunResult",
    "AgentRunner",
    "AgentSessionBusyError",
    "AgentStatus",
    "DEFAULT_MAX_STEPS",
]
