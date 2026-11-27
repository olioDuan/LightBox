import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  type AgentEventItem,
  type AgentSessionDetail,
  type AgentTodoItem,
  type DraftObviousResponse,
  type VerifierReport,
} from "../api";
import { useI18n } from "../i18n";
import { QuarantineModal } from "./QuarantineModal";

interface AgentDrawerProps {
  open: boolean;
  activeSessionId?: string | null;
  onClose: () => void;
  onPlanApplied?: () => void;
}

const STAGES = ["explore", "review", "propose"] as const;

function renderVerifierSeverity(report: VerifierReport): "green" | "yellow" | "red" {
  if (report.passed === false) return "red";
  if (
    (report.flagged_photos?.length ?? 0) > 0 ||
    report.high_ratio_warning ||
    report.layer2?.status === "flagged"
  ) {
    return "yellow";
  }
  return "green";
}

export function AgentDrawer({
  open,
  activeSessionId,
  onClose,
  onPlanApplied,
}: AgentDrawerProps) {
  const { lang, t } = useI18n();
  const queryClient = useQueryClient();

  const [selectedSessionId, setSelectedSessionId] = useState<string>(activeSessionId ?? "");
  const [agentSession, setAgentSession] = useState<AgentSessionDetail | null>(null);
  const [liveEvents, setLiveEvents] = useState<AgentEventItem[]>([]);
  const [liveTodos, setLiveTodos] = useState<AgentTodoItem[]>([]);
  const [liveStage, setLiveStage] = useState<string>("explore");
  const [inputText, setInputText] = useState("");
  const [constraintInput, setConstraintInput] = useState("");
  const [constraints, setConstraints] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openingPlan, setOpeningPlan] = useState(false);
  const [approvalDraft, setApprovalDraft] = useState<DraftObviousResponse | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const sessionsQ = useQuery({
    queryKey: ["sessions"],
    queryFn: () => api.listSessions(),
    enabled: open,
  });
  const sessions = sessionsQ.data?.items ?? [];

  useEffect(() => {
    if (activeSessionId) {
      setSelectedSessionId(activeSessionId);
    } else if (!selectedSessionId && sessions.length > 0) {
      setSelectedSessionId(sessions[0].id);
    }
  }, [activeSessionId, selectedSessionId, sessions]);

  const recentRunsQ = useQuery({
    queryKey: ["agent-sessions", selectedSessionId],
    queryFn: () => api.listAgentSessions(selectedSessionId || undefined),
    enabled: open && Boolean(selectedSessionId),
  });
  const recentRuns = recentRunsQ.data?.items ?? [];

  // Subscribe to SSE `agent.event` stream while drawer is open
  useEffect(() => {
    if (!open || !selectedSessionId) return;
    const es = new EventSource(
      `/api/events?session_id=${encodeURIComponent(selectedSessionId)}`,
    );
    const onAgentEvt = (evt: MessageEvent<string>) => {
      try {
        const data = JSON.parse(evt.data) as Record<string, unknown>;
        const evType = String(data.event_type ?? "");
        const seq = Number(data.seq ?? Date.now());
        if (evType === "plan_todos" && Array.isArray(data.todos)) {
          setLiveTodos(data.todos as AgentTodoItem[]);
        }
        if (evType === "stage_changed" || evType === "stage_rebuild") {
          if (typeof data.to_stage === "string") {
            setLiveStage(data.to_stage);
          }
        }
        setLiveEvents((prev) => {
          if (prev.some((item) => item.seq === seq && item.event_type === evType)) {
            return prev;
          }
          return [
            ...prev,
            {
              seq,
              event_type: evType,
              payload: data,
              created_at: Date.now() / 1000,
            },
          ];
        });
      } catch {
        // ignore malformed SSE event
      }
    };
    es.addEventListener("agent.event", onAgentEvt as EventListener);
    return () => {
      es.close();
    };
  }, [open, selectedSessionId]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [agentSession?.events.length, liveEvents.length, running]);

  const displayedEvents = useMemo(() => {
    if (running && liveEvents.length > 0) {
      return liveEvents;
    }
    return agentSession?.events ?? liveEvents;
  }, [agentSession?.events, liveEvents, running]);

  const displayedTodos = running && liveTodos.length > 0
    ? liveTodos
    : (agentSession?.todos ?? liveTodos);

  const displayedStage = running ? liveStage : (agentSession?.stage ?? liveStage);
  const displayedConstraints = agentSession?.constraints?.length
    ? agentSession.constraints
    : constraints;

  if (!open) return null;

  function handleAddConstraint() {
    const trimmed = constraintInput.trim();
    if (!trimmed) return;
    if (!constraints.includes(trimmed)) {
      setConstraints((prev) => [...prev, trimmed]);
    }
    setConstraintInput("");
  }

  function handleResetConversation() {
    setAgentSession(null);
    setLiveEvents([]);
    setLiveTodos([]);
    setLiveStage("explore");
    setError(null);
  }

  async function handleSelectRecent(agentSessionId: string) {
    setError(null);
    try {
      const detail = await api.getAgentSession(agentSessionId);
      setAgentSession(detail);
      setLiveEvents(detail.events);
      setLiveTodos(detail.todos);
      setLiveStage(detail.stage);
      setConstraints(detail.constraints);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleRunSupervisor() {
    if (!selectedSessionId || running) return;
    setRunning(true);
    setError(null);
    setLiveEvents([]);
    setLiveTodos([]);
    setLiveStage("explore");
    try {
      const res = await api.createAgentSession({
        session_id: selectedSessionId,
        mode: "supervisor",
        constraints,
        lang,
      });
      setAgentSession(res);
      setLiveEvents(res.events);
      setLiveTodos(res.todos);
      setLiveStage(res.stage);
      void queryClient.invalidateQueries({ queryKey: ["agent-sessions"] });
      void queryClient.invalidateQueries({ queryKey: ["llm-traces"] });
      void queryClient.invalidateQueries({ queryKey: ["session-summary"] });
      void queryClient.invalidateQueries({ queryKey: ["session-photos"] });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }

  async function handleSendMessage(override?: { message?: string; approveAsk?: boolean }) {
    if (!selectedSessionId || running) return;
    const msg = (override?.message ?? inputText).trim();
    if (!msg && override?.approveAsk === undefined) return;
    setRunning(true);
    setError(null);
    if (!override) setInputText("");
    try {
      let res: AgentSessionDetail;
      if (!agentSession) {
        setLiveEvents([]);
        setLiveTodos([]);
        setLiveStage("explore");
        res = await api.createAgentSession({
          session_id: selectedSessionId,
          mode: "chat",
          message: msg,
          constraints,
          lang,
        });
      } else {
        res = await api.sendAgentMessage(agentSession.agent_session_id, {
          message: msg,
          approve_ask: override?.approveAsk,
          constraints,
          lang,
        });
      }
      setAgentSession(res);
      setLiveEvents(res.events);
      setLiveTodos(res.todos);
      setLiveStage(res.stage);
      void queryClient.invalidateQueries({ queryKey: ["agent-sessions"] });
      void queryClient.invalidateQueries({ queryKey: ["llm-traces"] });
      void queryClient.invalidateQueries({ queryKey: ["session-summary"] });
      void queryClient.invalidateQueries({ queryKey: ["session-photos"] });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }

  async function handleOpenApprovePlan(planId: string) {
    setOpeningPlan(true);
    setError(null);
    try {
      const plan = await api.getPlan(planId, lang);
      const rep = (plan.verifier_report ?? agentSession?.verifier_report ?? {}) as VerifierReport;
      setApprovalDraft({
        plan,
        selected_count: plan.item_count,
        tech_reject_count: Number(rep.tech_reject_count ?? 0),
        burst_non_pick_count: Number(rep.burst_non_pick_count ?? 0),
        manual_reject_count: Number(rep.manual_reject_count ?? 0),
        high_ratio_warning: Boolean(rep.high_ratio_warning),
        excluded_already_quarantined: 0,
        excluded_unconfirmed_burst: 0,
        excluded_in_applying_plan: 0,
        superseded_plan_ids: [],
        verifier_report: rep,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setOpeningPlan(false);
    }
  }

  const verifierReport = agentSession?.verifier_report ?? null;
  const verifierSeverity = verifierReport ? renderVerifierSeverity(verifierReport) : null;
  const ctxK = ((agentSession?.context_tokens_est ?? 0) / 1000).toFixed(1);

  return (
    <>
      <div
        className="fixed inset-0 z-40 flex justify-end bg-black/45"
        onClick={onClose}
        role="presentation"
      >
        <aside
          className="flex h-full w-full max-w-[460px] flex-col border-l border-[#33363B] bg-[#1B1C1E] text-[#F3F4F6] shadow-2xl"
          onClick={(e) => e.stopPropagation()}
          aria-label={t.agentDrawerTitle}
        >
          {/* Header */}
          <div className="shrink-0 space-y-2.5 border-b border-[#33363B] bg-[#242629] px-4 py-3">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="inline-block h-2 w-2 rounded-full bg-[#5B8DEF]" />
                <h2 className="text-sm font-semibold text-[#F3F4F6]">{t.agentDrawerTitle}</h2>
                <kbd className="rounded border border-[#33363B] bg-[#1B1C1E] px-1.5 py-0.5 font-mono text-[10px] text-[#9CA3AF]">
                  ⌘K
                </kbd>
              </div>
              <div className="flex items-center gap-1.5">
                {agentSession && (
                  <button
                    type="button"
                    onClick={handleResetConversation}
                    className="rounded border border-[#33363B] px-2 py-1 text-[11px] text-[#D1D5DB] hover:bg-[#2C2F34]"
                  >
                    {t.agentNewChatBtn}
                  </button>
                )}
                <button
                  type="button"
                  onClick={onClose}
                  className="rounded border border-[#33363B] px-2 py-1 text-[11px] text-[#9CA3AF] hover:bg-[#2C2F34] hover:text-[#F3F4F6]"
                >
                  {t.agentClose}
                </button>
              </div>
            </div>

            {/* Session selector + Manual Supervisor Trigger */}
            <div className="flex items-center gap-2">
              <select
                value={selectedSessionId}
                onChange={(e) => {
                  setSelectedSessionId(e.target.value);
                  handleResetConversation();
                }}
                aria-label={t.agentSelectSession}
                className="min-w-0 flex-1 rounded border border-[#33363B] bg-[#1B1C1E] px-2.5 py-1.5 text-xs text-[#F3F4F6] focus:border-[#5B8DEF] focus:outline-none"
              >
                {sessions.length === 0 ? (
                  <option value="">{t.agentNoSessions}</option>
                ) : (
                  sessions.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.name}
                    </option>
                  ))
                )}
              </select>
              <button
                type="button"
                disabled={!selectedSessionId || running}
                onClick={() => void handleRunSupervisor()}
                className="shrink-0 rounded border border-[#5B8DEF]/50 bg-[#5B8DEF]/15 px-2.5 py-1.5 text-xs font-medium text-[#93C5FD] hover:bg-[#5B8DEF]/25 disabled:opacity-45"
              >
                {t.agentSupervisorBtn}
              </button>
            </div>

            {/* Technical Architecture Telemetry Bar: Stages + Context + Budget + Whitelist */}
            <div className="space-y-1.5 rounded border border-[#33363B] bg-[#1B1C1E] px-2.5 py-2">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-1">
                  {STAGES.map((st, idx) => {
                    const active = displayedStage === st;
                    return (
                      <span key={st} className="flex items-center gap-1">
                        <span
                          className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                            active
                              ? "bg-[#5B8DEF]/25 text-[#93C5FD]"
                              : "bg-[#242629] text-[#6B7280]"
                          }`}
                        >
                          {t.agentStageLabels[st] ?? st}
                        </span>
                        {idx < STAGES.length - 1 && (
                          <span className="text-[10px] text-[#4B5563]">→</span>
                        )}
                      </span>
                    );
                  })}
                </div>
                <span className="rounded bg-[#3D9A6B]/15 px-1.5 py-0.5 text-[10px] text-[#86EFAC]">
                  {t.agentWhitelistBadge}
                </span>
              </div>

              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px] text-[#9CA3AF] tabular-nums">
                <span>{t.agentContextTokens(ctxK)}</span>
                {(agentSession?.stage_rebuilds ?? 0) > 0 && (
                  <span className="text-[#FCD34D]">
                    {t.agentStageRebuilds(agentSession?.stage_rebuilds ?? 0)}
                  </span>
                )}
                <span>{t.agentViewedPhotos(agentSession?.viewed_photos ?? 0)}</span>
                <span>${(agentSession?.spent_usd ?? 0).toFixed(4)}</span>
              </div>
            </div>

            {/* Pinned Hard Constraints */}
            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5">
                <input
                  type="text"
                  value={constraintInput}
                  onChange={(e) => setConstraintInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      handleAddConstraint();
                    }
                  }}
                  placeholder={t.agentConstraintPlaceholder}
                  className="min-w-0 flex-1 rounded border border-[#33363B] bg-[#1B1C1E] px-2 py-1 text-[11px] text-[#F3F4F6] placeholder-[#6B7280] focus:border-[#5B8DEF] focus:outline-none"
                />
                <button
                  type="button"
                  onClick={handleAddConstraint}
                  className="rounded border border-[#33363B] bg-[#1B1C1E] px-2 py-1 text-[11px] text-[#D1D5DB] hover:bg-[#2C2F34]"
                >
                  {t.agentConstraintAdd}
                </button>
              </div>
              {displayedConstraints.length > 0 && (
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[10px] text-[#9CA3AF]">{t.agentConstraintsLabel}:</span>
                  {displayedConstraints.map((c) => (
                    <span
                      key={c}
                      className="inline-flex items-center gap-1 rounded bg-[#D99B26]/15 px-1.5 py-0.5 text-[10px] text-[#FCD34D]"
                    >
                      <span>{c}</span>
                      {!agentSession && (
                        <button
                          type="button"
                          onClick={() =>
                            setConstraints((prev) => prev.filter((item) => item !== c))
                          }
                          className="text-[#F5D07A] hover:text-white"
                        >
                          ×
                        </button>
                      )}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Scrollable Timeline Body */}
          <div ref={scrollRef} className="min-h-0 flex-1 space-y-3 overflow-y-auto px-4 py-3">
            {/* Live `update_plan` Todos */}
            {displayedTodos.length > 0 && (
              <div className="rounded-md border border-[#33363B] bg-[#242629] p-3">
                <div className="text-[11px] font-medium text-[#D1D5DB]">{t.agentTodosTitle}</div>
                <ul className="mt-2 space-y-1.5">
                  {displayedTodos.map((todo) => {
                    const dot =
                      todo.status === "done"
                        ? "bg-[#3D9A6B]"
                        : todo.status === "in_progress"
                          ? "bg-[#5B8DEF] animate-pulse"
                          : "bg-[#4B5563]";
                    return (
                      <li
                        key={todo.id}
                        className="flex items-center gap-2 text-xs text-[#E5E7EB]"
                      >
                        <span className={`inline-block h-2 w-2 shrink-0 rounded-full ${dot}`} />
                        <span
                          className={
                            todo.status === "done" ? "text-[#9CA3AF] line-through" : ""
                          }
                        >
                          {todo.title}
                        </span>
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}

            {/* Empty State + Recent Sessions */}
            {displayedEvents.length === 0 && !running && (
              <div className="space-y-3 py-4">
                <p className="text-xs leading-relaxed text-[#9CA3AF]">{t.agentEmptyHint}</p>
                <p className="rounded border border-[#33363B] bg-[#242629] px-3 py-2 text-[11px] text-[#9CA3AF]">
                  {t.agentZeroKeyNote}
                </p>
                {recentRuns.length > 0 && (
                  <div className="space-y-1.5 pt-2">
                    <div className="text-[11px] font-medium text-[#9CA3AF]">{t.agentRecentRuns}</div>
                    {recentRuns.slice(0, 4).map((run) => (
                      <button
                        key={run.agent_session_id}
                        type="button"
                        onClick={() => void handleSelectRecent(run.agent_session_id)}
                        className="flex w-full items-center justify-between gap-2 rounded border border-[#33363B] bg-[#242629] px-2.5 py-2 text-left text-xs hover:bg-[#2C2F34]"
                      >
                        <span className="truncate text-[#E5E7EB]">
                          {run.latest_user_message || run.mode}
                        </span>
                        <span className="shrink-0 font-mono text-[10px] text-[#9CA3AF]">
                          {run.status}
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Event Timeline: User messages, Tool calls, Stage rebuilds, Assistant replies */}
            {displayedEvents.map((ev) => {
              const p = ev.payload;
              if (ev.event_type === "session_started" || ev.event_type === "user_message") {
                const text = String(p.display_text ?? p.message ?? p.text ?? "").trim();
                if (!text) return null;
                return (
                  <div
                    key={`${ev.seq}-${ev.event_type}`}
                    className="ml-6 rounded-lg border border-[#5B8DEF]/30 bg-[#5B8DEF]/10 px-3 py-2 text-xs text-[#E5E7EB]"
                  >
                    {text}
                  </div>
                );
              }

              if (ev.event_type === "stage_rebuild") {
                return (
                  <div
                    key={`${ev.seq}-${ev.event_type}`}
                    className="rounded border border-[#D99B26]/35 bg-[#D99B26]/10 px-2.5 py-1.5 font-mono text-[10px] text-[#FCD34D]"
                  >
                    Stage rebuild: {String(p.from_stage ?? "")} → {String(p.to_stage ?? "")} (
                    {String(p.prev_tokens_est ?? "")} tok → clean SQLite anchor)
                  </div>
                );
              }

              if (ev.event_type === "assistant_turn") {
                const text = String(p.text ?? "").trim();
                const calls = Array.isArray(p.tool_calls)
                  ? (p.tool_calls as Array<{ id: string; name: string; arguments?: unknown }>)
                  : [];
                return (
                  <div key={`${ev.seq}-${ev.event_type}`} className="space-y-1.5">
                    {calls.length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {calls.map((c) => (
                          <span
                            key={c.id}
                            className="inline-flex items-center gap-1 rounded border border-[#33363B] bg-[#242629] px-2 py-0.5 font-mono text-[10px] text-[#93C5FD]"
                          >
                            <span>⚡ {c.name}</span>
                          </span>
                        ))}
                      </div>
                    )}
                    {text && (
                      <div className="whitespace-pre-wrap rounded-lg border border-[#33363B] bg-[#242629] px-3 py-2.5 text-xs leading-relaxed text-[#F3F4F6]">
                        {text}
                      </div>
                    )}
                  </div>
                );
              }

              if (ev.event_type === "tool_results_turn") {
                const results = Array.isArray(p.tool_results)
                  ? (p.tool_results as Array<{
                      tool_call_id: string;
                      name?: string;
                      is_error?: boolean;
                      hook_decision?: string;
                    }>)
                  : [];
                if (results.length === 0) return null;
                return (
                  <div key={`${ev.seq}-${ev.event_type}`} className="flex flex-wrap gap-1.5">
                    {results.map((r) => (
                      <span
                        key={r.tool_call_id}
                        className={`rounded px-2 py-0.5 font-mono text-[10px] ${
                          r.is_error
                            ? "bg-[#DC4C46]/15 text-[#FCA5A5]"
                            : "bg-[#3D9A6B]/15 text-[#86EFAC]"
                        }`}
                      >
                        {r.name ?? "tool"}:{" "}
                        {r.hook_decision === "DENY"
                          ? "DENY"
                          : r.is_error
                            ? "ERR"
                            : "OK"}
                      </span>
                    ))}
                  </div>
                );
              }

              return null;
            })}

            {/* Interactive ASK Pause Card */}
            {agentSession?.status === "awaiting_user" && agentSession.pending_ask && (
              <div className="space-y-2.5 rounded-md border border-[#D99B26]/50 bg-[#D99B26]/10 p-3">
                <div className="text-xs font-semibold text-[#FCD34D]">{t.agentAskTitle}</div>
                <div className="font-mono text-[11px] text-[#FDE68A]">
                  {agentSession.pending_ask.tool_call.name}
                </div>
                <p className="text-xs text-[#F5D07A]">{agentSession.pending_ask.reason}</p>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    disabled={running}
                    onClick={() =>
                      void handleSendMessage({ message: "Approved", approveAsk: true })
                    }
                    className="rounded bg-[#D99B26] px-3 py-1.5 text-xs font-medium text-[#1B1C1E] hover:bg-[#E5A935] disabled:opacity-50"
                  >
                    {agentSession.pending_ask.suggested_budget_usd
                      ? `${t.agentAskApprove} ($${agentSession.pending_ask.suggested_budget_usd.toFixed(2)})`
                      : t.agentAskApprove}
                  </button>
                  <button
                    type="button"
                    disabled={running}
                    onClick={() =>
                      void handleSendMessage({ message: "Declined", approveAsk: false })
                    }
                    className="rounded border border-[#33363B] bg-[#1B1C1E] px-3 py-1.5 text-xs text-[#D1D5DB] hover:bg-[#2C2F34] disabled:opacity-50"
                  >
                    {t.agentAskDecline}
                  </button>
                </div>
              </div>
            )}

            {/* Verifier Green / Yellow / Red Checklist Card + "去批准" Button */}
            {verifierReport && verifierSeverity && (
              <div
                className={`space-y-2 rounded-md border p-3 ${
                  verifierSeverity === "red"
                    ? "border-[#DC4C46]/50 bg-[#DC4C46]/10"
                    : verifierSeverity === "yellow"
                      ? "border-[#D99B26]/50 bg-[#D99B26]/10"
                      : "border-[#3D9A6B]/50 bg-[#3D9A6B]/10"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span
                      className={`inline-block h-2 w-2 rounded-full ${
                        verifierSeverity === "red"
                          ? "bg-[#DC4C46]"
                          : verifierSeverity === "yellow"
                            ? "bg-[#D99B26]"
                            : "bg-[#3D9A6B]"
                      }`}
                    />
                    <span className="text-xs font-medium text-[#F3F4F6]">
                      {t.agentVerifierCardTitle}
                    </span>
                  </div>
                  {verifierReport.layer2?.model_id && (
                    <span className="rounded border border-[#33363B] bg-[#1B1C1E] px-1.5 py-0.5 font-mono text-[10px] text-[#D1D5DB]">
                      {verifierReport.layer2.model_id}
                    </span>
                  )}
                </div>

                <div className="text-[11px] text-[#D1D5DB] tabular-nums">
                  {t.verifierLayer1Checks(
                    Number(verifierReport.exempt_count ?? 0),
                    Number(verifierReport.unconfirmed_burst_non_pick_count ?? 0),
                    Number(verifierReport.incomplete_atomic_pair_count ?? 0),
                  )}
                </div>

                {(verifierReport.flagged_photos?.length ?? 0) > 0 && (
                  <div className="text-[11px] text-[#FCD34D]">
                    {t.verifierFlaggedHeader(verifierReport.flagged_photos?.length ?? 0)}
                  </div>
                )}

                {agentSession?.latest_plan?.plan_id && (
                  <div className="flex items-center justify-between gap-2 pt-1">
                    <span className="font-mono text-[11px] text-[#D1D5DB] tabular-nums">
                      {t.photoCount(agentSession.latest_plan.count)}
                    </span>
                    <button
                      type="button"
                      disabled={openingPlan}
                      onClick={() =>
                        void handleOpenApprovePlan(agentSession.latest_plan!.plan_id)
                      }
                      className="rounded bg-[#DC4C46] px-3 py-1.5 text-xs font-medium text-white hover:bg-[#C63E38] disabled:opacity-50"
                    >
                      {openingPlan ? t.agentOpeningPlan : t.agentGoApprove}
                    </button>
                  </div>
                )}
              </div>
            )}

            {running && (
              <div className="flex items-center gap-2 px-1 py-1 text-xs text-[#93C5FD]">
                <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[#5B8DEF]" />
                <span>{t.agentThinking}</span>
              </div>
            )}

            {(error || agentSession?.error) && (
              <div className="rounded border border-[#DC4C46]/50 bg-[#DC4C46]/10 px-3 py-2 text-xs text-[#FCA5A5]">
                {error || agentSession?.error}
              </div>
            )}
          </div>

          {/* Input Bar */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void handleSendMessage();
            }}
            className="flex shrink-0 items-center gap-2 border-t border-[#33363B] bg-[#242629] px-4 py-3"
          >
            <input
              type="text"
              value={inputText}
              disabled={!selectedSessionId || running}
              onChange={(e) => setInputText(e.target.value)}
              placeholder={t.agentInputPlaceholder}
              className="min-w-0 flex-1 rounded-md border border-[#33363B] bg-[#1B1C1E] px-3 py-2 text-xs text-[#F3F4F6] placeholder-[#6B7280] focus:border-[#5B8DEF] focus:outline-none disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={!selectedSessionId || running || !inputText.trim()}
              className="rounded-md bg-[#F3F4F6] px-3.5 py-2 text-xs font-medium text-[#1B1C1E] hover:bg-white disabled:opacity-45"
            >
              {t.agentSend}
            </button>
          </form>
        </aside>
      </div>

      {approvalDraft && (
        <QuarantineModal
          draft={approvalDraft}
          onClose={() => setApprovalDraft(null)}
          onApplied={() => {
            setApprovalDraft(null);
            void queryClient.invalidateQueries({ queryKey: ["sessions"] });
            void queryClient.invalidateQueries({ queryKey: ["session-summary"] });
            void queryClient.invalidateQueries({ queryKey: ["session-photos"] });
            void queryClient.invalidateQueries({ queryKey: ["plans"] });
            onPlanApplied?.();
          }}
        />
      )}
    </>
  );
}
