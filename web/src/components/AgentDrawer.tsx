import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  type AgentEventItem,
  type AgentSessionDetail,
  type AgentTodoItem,
  type AiConnectionState,
  type DraftObviousResponse,
  type VerifierReport,
} from "../api";
import { useI18n } from "../i18n";
import { AiConnectionBar } from "./AiConnectionBar";
import { QuarantineModal } from "./QuarantineModal";
import { Button, Icon, IconButton, Kbd, ProgressLine } from "./ui";
import { Check, Lightning, Plus, Send, Sparkles, X } from "reicon-react";

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
  const [aiConn, setAiConn] = useState<AiConnectionState | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  const handleConnectionLoaded = useCallback((state: AiConnectionState) => {
    setAiConn(state);
  }, []);

  const agentRoleDisabled = Boolean(
    aiConn?.configured &&
      !aiConn.agent_ready &&
      (aiConn.agent_probes !== null || aiConn.agent_error !== null),
  );

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
      void queryClient.invalidateQueries({ queryKey: ["ai-connection"] });
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
      void queryClient.invalidateQueries({ queryKey: ["ai-connection"] });
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

  const stageIdx = STAGES.indexOf(displayedStage as (typeof STAGES)[number]);

  return (
    <>
      <div
        className="fade-enter fixed inset-0 z-40 flex justify-end bg-[#181818]/20"
        onClick={onClose}
        role="presentation"
      >
        <aside
          className="sheet-enter flex h-full w-full max-w-[480px] flex-col bg-paper sm:my-2 sm:mr-2 sm:h-[calc(100%-16px)] sm:rounded-dialog"
          onClick={(e) => e.stopPropagation()}
          aria-label={t.agentDrawerTitle}
        >
          {/* Header */}
          <div className="shrink-0 space-y-4 px-6 pb-4 pt-6">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Icon as={Sparkles} />
                <h2 className="text-lg font-semibold text-ink">{t.agentDrawerTitle}</h2>
                <Kbd>⌘K</Kbd>
              </div>
              <div className="flex items-center gap-1">
                {agentSession && (
                  <Button variant="ghost" size="sm" icon={Plus} onClick={handleResetConversation}>
                    {t.agentNewChatBtn}
                  </Button>
                )}
                <IconButton icon={X} label={t.agentClose} onClick={onClose} />
              </div>
            </div>

            <AiConnectionBar open={open} onConnectionLoaded={handleConnectionLoaded} />

            <div className="flex items-center gap-2">
              <select
                value={selectedSessionId}
                onChange={(e) => {
                  setSelectedSessionId(e.target.value);
                  handleResetConversation();
                }}
                aria-label={t.agentSelectSession}
                className="field min-w-0 flex-1"
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
              <Button
                variant="secondary"
                disabled={!selectedSessionId || running || agentRoleDisabled}
                onClick={() => void handleRunSupervisor()}
              >
                {t.agentSupervisorBtn}
              </Button>
            </div>

            {/* Stages are a real sequence: explore → review → propose */}
            <div className="flex flex-col gap-2">
              <ol className="flex items-center gap-2 text-sm">
                {STAGES.map((st, idx) => {
                  const active = displayedStage === st;
                  const passed = stageIdx > idx;
                  return (
                    <li key={st} className="flex flex-1 flex-col gap-2">
                      <span
                        className={`h-1 rounded-full transition-colors duration-300 ${
                          active && running
                            ? "progress-indeterminate bg-table-2"
                            : active
                              ? "bg-ink"
                              : passed
                                ? "bg-ink-3"
                                : "bg-table-2"
                        }`}
                      />
                      <span className={active ? "font-medium text-ink" : "text-ink-3"}>
                        {t.agentStageLabels[st] ?? st}
                      </span>
                    </li>
                  );
                })}
              </ol>
              <div
                className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-ink-3 tabular-nums"
                title={`${t.agentWhitelistBadge}\n${t.agentZeroKeyNote}`}
              >
                <span>{t.agentContextTokens(ctxK)}</span>
                {(agentSession?.stage_rebuilds ?? 0) > 0 && (
                  <span className="text-ink-2">{t.agentStageRebuilds(agentSession?.stage_rebuilds ?? 0)}</span>
                )}
                <span>{t.agentViewedPhotos(agentSession?.viewed_photos ?? 0)}</span>
                <span>${(agentSession?.spent_usd ?? 0).toFixed(4)}</span>
              </div>
            </div>

            {/* Pinned hard constraints */}
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-2">
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
                  aria-label={t.agentConstraintsLabel}
                  className="field h-8 min-w-0 flex-1 rounded-chip px-3"
                />
                {constraintInput.trim() && (
                  <Button variant="secondary" size="sm" className="fade-enter" onClick={handleAddConstraint}>
                    {t.agentConstraintAdd}
                  </Button>
                )}
              </div>
              {displayedConstraints.length > 0 && (
                <div className="flex flex-wrap items-center gap-1" aria-label={t.agentConstraintsLabel}>
                  {displayedConstraints.map((c) => (
                    <span
                      key={c}
                      className="inline-flex h-8 items-center gap-1 rounded-chip bg-table pl-3 pr-1 text-sm text-ink"
                    >
                      <span>{c}</span>
                      {!agentSession ? (
                        <button
                          type="button"
                          aria-label={`${t.uiDismiss} ${c}`}
                          onClick={() => setConstraints((prev) => prev.filter((item) => item !== c))}
                          className="flex h-6 w-6 items-center justify-center rounded-[6px] text-ink-3 hover:bg-table-2 hover:text-ink"
                        >
                          <X size={14} strokeWidth={1.5} />
                        </button>
                      ) : (
                        <span className="w-2" />
                      )}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Timeline */}
          <div ref={scrollRef} className="min-h-0 flex-1 space-y-6 overflow-y-auto px-6 py-4">
            {displayedTodos.length > 0 && (
              <section className="flex flex-col gap-2">
                <h3 className="text-sm font-medium text-ink">{t.agentTodosTitle}</h3>
                <ul className="flex flex-col gap-1">
                  {displayedTodos.map((todo) => (
                    <li key={todo.id} className="flex items-center gap-2 text-sm">
                      <span className="flex h-6 w-6 shrink-0 items-center justify-center">
                        {todo.status === "done" ? (
                          <Check size={16} strokeWidth={1.5} className="text-ink-3" />
                        ) : todo.status === "in_progress" ? (
                          <span className="h-2 w-2 animate-pulse rounded-full bg-ink" />
                        ) : (
                          <span className="h-2 w-2 rounded-full shadow-[inset_0_0_0_1.5px_var(--color-ink-3)]" />
                        )}
                      </span>
                      <span
                        className={
                          todo.status === "done"
                            ? "text-ink-3 line-through"
                            : todo.status === "in_progress"
                              ? "font-medium text-ink"
                              : "text-ink-2"
                        }
                      >
                        {todo.title}
                      </span>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {displayedEvents.length === 0 && !running && recentRuns.length > 0 && (
              <section className="flex flex-col gap-2" title={t.agentEmptyHint}>
                <h3 className="text-sm font-medium text-ink">{t.agentRecentRuns}</h3>
                <div className="-mx-2 flex flex-col">
                  {recentRuns.slice(0, 4).map((run) => (
                    <button
                      key={run.agent_session_id}
                      type="button"
                      onClick={() => void handleSelectRecent(run.agent_session_id)}
                      className="flex h-12 items-center justify-between gap-4 rounded-btn px-2 text-left text-sm hover:bg-table"
                    >
                      <span className="truncate text-ink">{run.latest_user_message || run.mode}</span>
                      <span className="shrink-0 text-ink-3">{run.status}</span>
                    </button>
                  ))}
                </div>
              </section>
            )}

            {displayedEvents.map((ev) => {
              const p = ev.payload;
              if (ev.event_type === "session_started" || ev.event_type === "user_message") {
                const text = String(p.display_text ?? p.message ?? p.text ?? "").trim();
                if (!text) return null;
                return (
                  <div key={`${ev.seq}-${ev.event_type}`} className="flex justify-end">
                    <div className="max-w-[85%] rounded-[20px] rounded-br-[8px] bg-table px-4 py-2 text-sm leading-6 text-ink">
                      {text}
                    </div>
                  </div>
                );
              }

              if (ev.event_type === "stage_rebuild") {
                return (
                  <p key={`${ev.seq}-${ev.event_type}`} className="text-sm text-ink-3 tabular-nums">
                    {String(p.from_stage ?? "")} → {String(p.to_stage ?? "")} ·{" "}
                    {String(p.prev_tokens_est ?? "")} tok
                  </p>
                );
              }

              if (ev.event_type === "assistant_turn") {
                const text = String(p.text ?? "").trim();
                const calls = Array.isArray(p.tool_calls)
                  ? (p.tool_calls as Array<{ id: string; name: string; arguments?: unknown }>)
                  : [];
                return (
                  <div key={`${ev.seq}-${ev.event_type}`} className="flex flex-col gap-2">
                    {calls.length > 0 && (
                      <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-ink-3">
                        {calls.map((c) => (
                          <span key={c.id} className="inline-flex items-center gap-1">
                            <Lightning size={14} strokeWidth={1.5} />
                            {c.name}
                          </span>
                        ))}
                      </div>
                    )}
                    {text && <div className="whitespace-pre-wrap text-sm leading-6 text-ink">{text}</div>}
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
                  <div key={`${ev.seq}-${ev.event_type}`} className="flex flex-wrap gap-x-4 gap-y-1 text-sm">
                    {results.map((r) => {
                      const bad = r.is_error || r.hook_decision === "DENY";
                      return (
                        <span key={r.tool_call_id} className={bad ? "text-mark" : "text-ink-3"}>
                          {r.name ?? "tool"}{" "}
                          {r.hook_decision === "DENY" ? "DENY" : r.is_error ? "ERR" : "OK"}
                        </span>
                      );
                    })}
                  </div>
                );
              }

              return null;
            })}

            {agentSession?.status === "awaiting_user" && agentSession.pending_ask && (
              <section className="fade-enter flex flex-col gap-2 rounded-panel bg-table p-4">
                <div className="flex items-center gap-2 text-sm font-medium text-ink">
                  <span className="h-2 w-2 rounded-full bg-unsure" />
                  {t.agentAskTitle}
                </div>
                <div className="text-sm text-ink-2">{agentSession.pending_ask.tool_call.name}</div>
                <p className="text-sm leading-6 text-ink">{agentSession.pending_ask.reason}</p>
                <div className="flex items-center gap-2 pt-2">
                  <Button
                    variant="primary"
                    size="sm"
                    disabled={running}
                    onClick={() => void handleSendMessage({ message: "Approved", approveAsk: true })}
                  >
                    {agentSession.pending_ask.ask_kind === "unpriced_triage"
                      ? t.agentAskUnpricedApprove(
                          agentSession.pending_ask.unpriced_max_photos ?? 20,
                        )
                      : agentSession.pending_ask.suggested_budget_usd
                        ? `${t.agentAskApprove} ($${agentSession.pending_ask.suggested_budget_usd.toFixed(2)})`
                        : t.agentAskApprove}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={running}
                    onClick={() => void handleSendMessage({ message: "Declined", approveAsk: false })}
                  >
                    {t.agentAskDecline}
                  </Button>
                </div>
              </section>
            )}

            {verifierReport && verifierSeverity && (
              <section className="fade-enter flex flex-col gap-2 rounded-panel bg-table p-4">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 text-sm font-medium text-ink">
                    <span
                      className={`h-2 w-2 rounded-full ${
                        verifierSeverity === "green"
                          ? "bg-keep"
                          : verifierSeverity === "yellow"
                            ? "bg-unsure"
                            : "bg-reject"
                      }`}
                    />
                    {t.agentVerifierCardTitle}
                  </div>
                  {verifierReport.layer2?.model_id && (
                    <span className="truncate text-sm text-ink-3">{verifierReport.layer2.model_id}</span>
                  )}
                </div>
                <div className="text-sm text-ink-2 tabular-nums">
                  {t.verifierLayer1Checks(
                    Number(verifierReport.exempt_count ?? 0),
                    Number(verifierReport.unconfirmed_burst_non_pick_count ?? 0),
                    Number(verifierReport.incomplete_atomic_pair_count ?? 0),
                  )}
                </div>
                {(verifierReport.flagged_photos?.length ?? 0) > 0 && (
                  <div className="text-sm text-mark">
                    {t.verifierFlaggedHeader(verifierReport.flagged_photos?.length ?? 0)}
                  </div>
                )}
                {agentSession?.latest_plan?.plan_id && (
                  <div className="flex items-center justify-between gap-2 pt-2">
                    <span className="text-sm font-medium text-ink tabular-nums">
                      {t.photoCount(agentSession.latest_plan.count)}
                    </span>
                    <Button
                      variant="mark"
                      size="sm"
                      disabled={openingPlan}
                      onClick={() => void handleOpenApprovePlan(agentSession.latest_plan!.plan_id)}
                    >
                      {openingPlan ? t.agentOpeningPlan : t.agentGoApprove}
                    </Button>
                  </div>
                )}
              </section>
            )}

            {running && (
              <div className="fade-enter flex items-center gap-4 text-sm text-ink-2">
                <ProgressLine indeterminate className="w-16" />
                <span>{t.agentThinking}</span>
              </div>
            )}

            {(error || agentSession?.error) && (
              <p className="text-sm text-mark">{error || agentSession?.error}</p>
            )}
          </div>

          {/* Composer */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void handleSendMessage();
            }}
            className="relative shrink-0 p-4"
          >
            <input
              type="text"
              value={inputText}
              disabled={!selectedSessionId || running || agentRoleDisabled}
              onChange={(e) => setInputText(e.target.value)}
              placeholder={t.agentInputPlaceholder}
              aria-label={t.agentInputPlaceholder}
              className="field h-12 w-full rounded-[24px] pl-5 pr-14"
            />
            <IconButton
              type="submit"
              icon={Send}
              label={t.agentSend}
              variant="primary"
              size="sm"
              disabled={!selectedSessionId || running || agentRoleDisabled || !inputText.trim()}
              className="absolute right-6 top-1/2 -translate-y-1/2 rounded-full"
            />
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
