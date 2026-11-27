import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, photoImageUrl, type DigestDimension, type PlanDetail } from "../api";
import {
  Button,
  EmptyState,
  Icon,
  Loading,
  Notice,
  NoticeDock,
  ProgressLine,
  Tabs,
  Thumb,
} from "../components/ui";
import { useI18n } from "../i18n";
import { ChevronDown, Folder, Refresh, RotateLeft, Sparkles } from "reicon-react";

/** Same verdict colors as the review screen; the two non-verdict buckets stay gray. */
const SEG = {
  kept: "var(--color-keep)",
  burst: "#a3a3a3",
  subjective: "#6a6a6a",
  uncertain: "var(--color-unsure)",
  tech: "var(--color-reject)",
} as const;

function Swatch({ color }: { color: string }) {
  return (
    <span
      aria-hidden="true"
      className="inline-block h-2 w-2 shrink-0 rounded-full"
      style={{ background: color }}
    />
  );
}

function StackedBar({
  parts,
  height = "h-2",
}: {
  parts: Array<{ value: number; color: string; title?: string }>;
  height?: string;
}) {
  return (
    <div className={`flex w-full gap-[2px] overflow-hidden rounded-full bg-table ${height}`}>
      {parts
        .filter((p) => p.value > 0)
        .map((p, i) => (
          <div
            key={i}
            title={p.title}
            className="h-full transition-[width] duration-500 ease-out first:rounded-l-full last:rounded-r-full"
            style={{ width: clampBarPct(p.value), background: p.color }}
          />
        ))}
    </div>
  );
}

function SectionTitle({ children, aside }: { children: React.ReactNode; aside?: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-4">
      <h2 className="text-lg font-semibold text-ink">{children}</h2>
      {aside}
    </div>
  );
}

interface InsightsViewProps {
  activeTab: "digest" | "trash" | "traces" | "memory";
  onChangeTab: (tab: "digest" | "trash" | "traces" | "memory") => void;
}

function fmtPct(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`;
}

function clampBarPct(rate: number): string {
  const pct = Math.max(0, Math.min(100, rate * 100));
  return `${pct.toFixed(1)}%`;
}

export function InsightsView({ activeTab, onChangeTab }: InsightsViewProps) {
  const { lang, t } = useI18n();
  const queryClient = useQueryClient();

  // Digest scope state
  const [sessionId, setSessionId] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [selectedDim, setSelectedDim] = useState("light_bucket");

  // Trash / restore state
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Memory state (T14)
  const [prefsDraft, setPrefsDraft] = useState("");
  const [prefsDirty, setPrefsDirty] = useState(false);
  const [ruleEdits, setRuleEdits] = useState<Record<string, string>>({});
  const [memoryBusy, setMemoryBusy] = useState<string | null>(null);
  const [memoryNotice, setMemoryNotice] = useState<string | null>(null);
  const [memoryError, setMemoryError] = useState<string | null>(null);

  const sessionsQ = useQuery({
    queryKey: ["sessions"],
    queryFn: () => api.listSessions(),
  });
  const sessions = sessionsQ.data?.items ?? [];

  const digestQ = useQuery({
    queryKey: ["digest", sessionId, dateFrom, dateTo, lang],
    queryFn: () =>
      api.getShootingDigest({
        sessionId: sessionId || undefined,
        dateFrom: dateFrom || undefined,
        dateTo: dateTo || undefined,
        lang,
      }),
    enabled: activeTab === "digest",
  });

  const plansQ = useQuery({
    queryKey: ["plans"],
    queryFn: () => api.listPlans(),
  });

  const tracesQ = useQuery({
    queryKey: ["llm-traces", sessionId],
    queryFn: () => api.getLlmTraces({ sessionId: sessionId || undefined, limit: 150 }),
    enabled: activeTab === "traces",
  });

  const memoryQ = useQuery({
    queryKey: ["memory-preferences"],
    queryFn: () => api.getMemoryPreferences(),
  });

  useEffect(() => {
    if (memoryQ.data && !prefsDirty) {
      setPrefsDraft(memoryQ.data.preferences_md);
    }
  }, [memoryQ.data, prefsDirty]);

  async function handleSavePreferences() {
    setMemoryBusy("save");
    setMemoryError(null);
    setMemoryNotice(null);
    try {
      const updated = await api.saveMemoryPreferences(prefsDraft);
      setPrefsDraft(updated.preferences_md);
      setPrefsDirty(false);
      setMemoryNotice(t.memorySavedToast);
      void queryClient.invalidateQueries({ queryKey: ["memory-preferences"] });
    } catch (err) {
      setMemoryError(err instanceof Error ? err.message : String(err));
    } finally {
      setMemoryBusy(null);
    }
  }

  async function handleConsolidate(force: boolean) {
    setMemoryBusy("consolidate");
    setMemoryError(null);
    setMemoryNotice(null);
    try {
      const res = await api.consolidateMemory(force);
      void queryClient.invalidateQueries({ queryKey: ["memory-preferences"] });
      void queryClient.invalidateQueries({ queryKey: ["llm-traces"] });
      if (res.result.status === "proposed") {
        setMemoryNotice(t.memoryConsolidateProposed(res.result.proposed_rules.length));
      } else if (res.result.status === "skipped_insufficient_overrides") {
        setMemoryNotice(
          t.memoryConsolidateSkipped(
            res.result.new_override_count,
            res.state.min_overrides_threshold,
          ),
        );
      } else {
        setMemoryNotice(t.memoryConsolidateNoNew);
      }
    } catch (err) {
      setMemoryError(err instanceof Error ? err.message : String(err));
    } finally {
      setMemoryBusy(null);
    }
  }

  async function handleDecideRule(ruleId: string, decision: "accept" | "reject", fallbackText: string) {
    if (prefsDirty) {
      const proceed = window.confirm(t.memoryUnsavedBeforeDecideConfirm);
      if (!proceed) return;
    }
    setMemoryBusy(`rule:${ruleId}`);
    setMemoryError(null);
    setMemoryNotice(null);
    try {
      if (prefsDirty) {
        await api.saveMemoryPreferences(prefsDraft);
      }
      const editedText = ruleEdits[ruleId] ?? fallbackText;
      const res = await api.decideMemoryRule(ruleId, {
        decision,
        ...(decision === "accept" ? { rule_text: editedText } : {}),
      });
      setPrefsDraft(res.state.preferences_md);
      setPrefsDirty(false);
      setMemoryNotice(
        decision === "accept" ? t.memoryRuleAcceptedToast : t.memoryRuleRejectedToast,
      );
      void queryClient.invalidateQueries({ queryKey: ["memory-preferences"] });
    } catch (err) {
      setMemoryError(err instanceof Error ? err.message : String(err));
    } finally {
      setMemoryBusy(null);
    }
  }

  const plans = (plansQ.data?.items ?? []).filter(
    (p) => p.status === "applied" || p.status === "reverted" || p.status === "applying",
  );
  const activeTrashBatches = plans.filter(
    (p) => p.status === "applied" || p.status === "applying",
  ).length;

  async function handleRestoreBatch(plan: PlanDetail) {
    setBusyKey(plan.id);
    setError(null);
    setNotice(null);
    try {
      const res = await api.restorePlan(plan.id);
      setNotice(t.restoreSuccessBatch(res.restored_photo_ids.length));
      await queryClient.invalidateQueries({ queryKey: ["plans"] });
      await queryClient.invalidateQueries({ queryKey: ["digest"] });
      await queryClient.invalidateQueries({ queryKey: ["session-summary"] });
      await queryClient.invalidateQueries({ queryKey: ["session-photos"] });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyKey(null);
    }
  }

  async function handleRestorePhoto(planId: string, photoId: string) {
    setBusyKey(`${planId}:${photoId}`);
    setError(null);
    setNotice(null);
    try {
      await api.restorePlan(planId, [photoId]);
      setNotice(t.restoreSuccessSingle);
      await queryClient.invalidateQueries({ queryKey: ["plans"] });
      await queryClient.invalidateQueries({ queryKey: ["digest"] });
      await queryClient.invalidateQueries({ queryKey: ["session-summary"] });
      await queryClient.invalidateQueries({ queryKey: ["session-photos"] });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyKey(null);
    }
  }

  async function handleOpenFolder(planId: string) {
    setError(null);
    try {
      await api.openPlanFolder(planId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  const digest = digestQ.data;
  const activeDim: DigestDimension | undefined =
    digest?.dimensions.find((d) => d.key === selectedDim) ?? digest?.dimensions[0];

  const hasScopeFilters = Boolean(sessionId || dateFrom || dateTo);

  const memoryAttention =
    (memoryQ.data?.proposed_count ?? 0) > 0 || Boolean(memoryQ.data?.ready_to_consolidate);
  const memoryBadge =
    (memoryQ.data?.proposed_count ?? 0) > 0
      ? memoryQ.data?.proposed_count
      : memoryQ.data?.ready_to_consolidate
        ? memoryQ.data?.new_override_count
        : undefined;

  const sessionSelect = (
    <select
      value={sessionId}
      onChange={(e) => setSessionId(e.target.value)}
      aria-label={t.digestAllSessions}
      className={`field ${sessionId ? "font-medium" : "text-ink-2"}`}
    >
      <option value="">{t.digestAllSessions}</option>
      {sessions.map((s) => (
        <option key={s.id} value={s.id}>
          {s.name}
        </option>
      ))}
    </select>
  );

  return (
    <div className="mx-auto max-w-5xl px-6 pb-32 pt-12 sm:px-12">
      <Tabs
        value={activeTab}
        onChange={onChangeTab}
        items={[
          { id: "digest", label: t.insightsTabDigest },
          {
            id: "trash",
            label: t.insightsTabTrash,
            count: activeTrashBatches > 0 ? activeTrashBatches : undefined,
            attention: activeTrashBatches > 0,
          },
          { id: "traces", label: t.insightsTabTraces },
          {
            id: "memory",
            label: t.insightsTabMemory,
            count: memoryAttention ? memoryBadge : undefined,
            attention: memoryAttention,
          },
        ]}
      />

      {activeTab === "digest" ? (
        <div className="mt-12 flex flex-col gap-16">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-baseline gap-4">
              <h1 className="text-lg font-semibold text-ink" title={t.digestSubtitle}>
                {t.digestTitle}
              </h1>
              {digest && (
                <span
                  className="text-sm text-ink-3 tabular-nums"
                  title={t.digestMinSamplesHint(digest.scope.min_samples)}
                >
                  {t.digestTotalShots(digest.overview.total_shots)}
                  <span className="ml-4">{t.digestScopeSessions(digest.scope.session_count)}</span>
                </span>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {sessionSelect}
              <input
                type="date"
                value={dateFrom}
                onChange={(e) => setDateFrom(e.target.value)}
                aria-label={t.searchDateFrom}
                title={t.searchDateFrom}
                className={`field ${dateFrom ? "" : "text-ink-2"}`}
              />
              <span className="text-ink-3" aria-hidden="true">
                –
              </span>
              <input
                type="date"
                value={dateTo}
                onChange={(e) => setDateTo(e.target.value)}
                aria-label={t.searchDateTo}
                title={t.searchDateTo}
                className={`field ${dateTo ? "" : "text-ink-2"}`}
              />
              {hasScopeFilters && (
                <Button
                  variant="ghost"
                  icon={Refresh}
                  onClick={() => {
                    setSessionId("");
                    setDateFrom("");
                    setDateTo("");
                  }}
                >
                  {t.digestResetScope}
                </Button>
              )}
            </div>
          </div>

          {digestQ.isLoading ? (
            <Loading label={t.loading} />
          ) : digestQ.isError ? (
            <p className="text-sm text-mark">
              {t.errorPrefix}
              {digestQ.error instanceof Error ? digestQ.error.message : String(digestQ.error)}
            </p>
          ) : !digest || digest.overview.total_shots === 0 ? (
            <EmptyState title={t.digestEmpty}>
              {hasScopeFilters && (
                <Button
                  variant="secondary"
                  onClick={() => {
                    setSessionId("");
                    setDateFrom("");
                    setDateTo("");
                  }}
                >
                  {t.digestResetScope}
                </Button>
              )}
            </EmptyState>
          ) : (
            <>
              {/* Overview: one bar, five numbers */}
              <section className="flex flex-col gap-8">
                <StackedBar
                  height="h-3"
                  parts={[
                    { value: digest.overview.kept_rate_total, color: SEG.kept, title: `${t.digestCardKept}: ${digest.overview.kept_count}` },
                    { value: digest.overview.uncertain_rate, color: SEG.uncertain, title: `${t.digestCardUncertain}: ${digest.overview.uncertain_count}` },
                    { value: digest.overview.burst_cull_rate, color: SEG.burst, title: `${t.digestCardBurstCull}: ${digest.overview.burst_cull_count}` },
                    { value: digest.overview.subjective_reject_rate, color: SEG.subjective, title: `${t.digestCardSubjectiveReject}: ${digest.overview.subjective_reject_count}` },
                    { value: digest.overview.tech_reject_rate, color: SEG.tech, title: `${t.digestCardTechReject}: ${digest.overview.tech_reject_count}` },
                  ]}
                />
                <dl className="grid grid-cols-2 gap-x-8 gap-y-8 sm:grid-cols-3 lg:grid-cols-5">
                  {(
                    [
                      {
                        color: SEG.kept,
                        label: t.digestCardKept,
                        value: fmtPct(digest.overview.kept_rate_excl_burst),
                        sub: `${digest.overview.kept_count} / ${digest.overview.non_burst_total}`,
                        hint: `${t.digestCardKeptExclBurst}；${t.digestCardKeptTotal} ${digest.overview.kept_count} / ${digest.overview.total_shots} (${fmtPct(digest.overview.kept_rate_total)})`,
                        accent: false,
                      },
                      {
                        color: SEG.tech,
                        label: t.digestCardTechReject,
                        value: fmtPct(digest.overview.tech_reject_rate),
                        sub: `${digest.overview.tech_reject_count} / ${digest.overview.total_shots}`,
                        hint: t.digestCardTechRejectHint,
                        accent: true,
                      },
                      {
                        color: SEG.burst,
                        label: t.digestCardBurstCull,
                        value: fmtPct(digest.overview.burst_cull_rate),
                        sub: `${digest.overview.burst_cull_count} / ${digest.overview.total_shots}`,
                        hint: t.digestCardBurstCullHint,
                        accent: false,
                      },
                      {
                        color: SEG.subjective,
                        label: t.digestCardSubjectiveReject,
                        value: fmtPct(digest.overview.subjective_reject_rate),
                        sub: `${digest.overview.subjective_reject_count} / ${digest.overview.total_shots}`,
                        hint: t.digestCardSubjectiveRejectHint,
                        accent: false,
                      },
                      {
                        color: SEG.uncertain,
                        label: t.digestCardUncertain,
                        value: fmtPct(digest.overview.uncertain_rate),
                        sub: `${digest.overview.uncertain_count} / ${digest.overview.total_shots}`,
                        hint: t.digestCardUncertainHint,
                        accent: false,
                      },
                    ] as const
                  ).map((m) => (
                    <div key={m.label} title={m.hint} className="flex flex-col gap-1">
                      <dt className="flex items-center gap-2 text-sm text-ink-2">
                        <Swatch color={m.color} />
                        {m.label}
                      </dt>
                      <dd
                        className={`text-[32px] font-semibold leading-10 tabular-nums ${
                          m.accent ? "text-reject" : m.color === SEG.kept ? "text-keep-ink" : "text-ink"
                        }`}
                      >
                        {m.value}
                      </dd>
                      <dd className="text-sm text-ink-3 tabular-nums">{m.sub}</dd>
                    </div>
                  ))}
                </dl>
              </section>

              {digest.top_tech_reasons.length > 0 && (
                <section className="flex flex-col gap-6">
                  <SectionTitle>{t.digestTopReasonsTitle}</SectionTitle>
                  <ul className="flex flex-col gap-4">
                    {digest.top_tech_reasons.map((r) => (
                      <li key={r.code} className="flex flex-col gap-2">
                        <div className="flex items-baseline justify-between gap-4 text-sm">
                          <span className="truncate text-ink">{r.label}</span>
                          <span className="shrink-0 text-ink-2 tabular-nums">
                            {r.count} / {digest.overview.tech_reject_count}
                            <span className="ml-2 text-ink-3">{fmtPct(r.rate_of_tech_rejects)}</span>
                          </span>
                        </div>
                        <ProgressLine value={r.rate_of_tech_rejects * 100} tone="mark" />
                      </li>
                    ))}
                  </ul>
                </section>
              )}

              <div className="grid grid-cols-1 gap-16 lg:grid-cols-2 lg:gap-12">
                <section className="flex flex-col gap-6">
                  <SectionTitle>{t.digestHighRejectTitle}</SectionTitle>
                  {digest.high_tech_reject_conditions.length === 0 ? (
                    <p className="text-sm text-ink-3">{t.digestHighRejectEmpty(digest.scope.min_samples)}</p>
                  ) : (
                    <ul className="flex flex-col gap-6">
                      {digest.high_tech_reject_conditions.map((c) => (
                        <li key={`${c.dimension_key}:${c.bucket_key}`} className="flex flex-col gap-2 text-sm">
                          <div className="flex items-baseline justify-between gap-4">
                            <span className="min-w-0 truncate">
                              <span className="text-ink-3">{c.dimension_label}</span>
                              <span className="ml-2 font-medium text-ink">{c.bucket_label}</span>
                            </span>
                            <span className="shrink-0 font-medium text-mark tabular-nums">
                              {fmtPct(c.tech_reject_rate)}
                            </span>
                          </div>
                          <ProgressLine value={c.tech_reject_rate * 100} tone="mark" />
                          <div className="flex flex-wrap items-center justify-between gap-x-4 text-ink-3 tabular-nums">
                            <span>
                              {c.tech_reject} / {c.total}
                              <span className="ml-2">
                                {t.digestDeltaAboveOverall((c.tech_reject_delta * 100).toFixed(1))}
                              </span>
                            </span>
                            {c.top_reason_label && (
                              <span className="truncate">
                                {t.digestMainReasonPrefix}
                                {c.top_reason_label}
                              </span>
                            )}
                          </div>
                        </li>
                      ))}
                    </ul>
                  )}
                </section>

                <section className="flex flex-col gap-6">
                  <SectionTitle>{t.digestHighKeptTitle}</SectionTitle>
                  {digest.high_kept_conditions.length === 0 ? (
                    <p className="text-sm text-ink-3">{t.digestHighKeptEmpty(digest.scope.min_samples)}</p>
                  ) : (
                    <ul className="flex flex-col gap-6">
                      {digest.high_kept_conditions.map((c) => (
                        <li key={`${c.dimension_key}:${c.bucket_key}`} className="flex flex-col gap-2 text-sm">
                          <div className="flex items-baseline justify-between gap-4">
                            <span className="min-w-0 truncate">
                              <span className="text-ink-3">{c.dimension_label}</span>
                              <span className="ml-2 font-medium text-ink">{c.bucket_label}</span>
                            </span>
                            <span className="shrink-0 font-medium text-keep-ink tabular-nums">
                              {fmtPct(c.kept_rate_excl_burst)}
                            </span>
                          </div>
                          <ProgressLine value={c.kept_rate_excl_burst * 100} tone="keep" />
                          <div className="flex flex-wrap items-center justify-between gap-x-4 text-ink-3 tabular-nums">
                            <span>
                              {c.kept} / {c.non_burst_total}
                              <span className="ml-2">
                                {t.digestDeltaAboveOverall((c.kept_rate_delta * 100).toFixed(1))}
                              </span>
                            </span>
                            <span>
                              {t.digestColTechRate} {c.tech_reject} / {c.total} ({fmtPct(c.tech_reject_rate)})
                            </span>
                          </div>
                        </li>
                      ))}
                    </ul>
                  )}
                </section>
              </div>

              <section className="flex flex-col gap-6">
                <SectionTitle>{t.digestStableSetupsTitle}</SectionTitle>
                {digest.stable_setups.length === 0 ? (
                  <p className="text-sm text-ink-3">{t.digestStableSetupsEmpty(digest.scope.min_samples)}</p>
                ) : (
                  <ul className="grid grid-cols-1 gap-x-12 gap-y-6 md:grid-cols-2">
                    {digest.stable_setups.map((st) => (
                      <li key={`${st.focal_bucket}:${st.shutter_bucket}`} className="flex flex-col gap-2 text-sm">
                        <div className="flex items-baseline justify-between gap-4">
                          <span className="truncate font-medium text-ink">
                            {st.focal_label} × {st.shutter_label}
                          </span>
                          <span className="shrink-0 font-medium text-ink tabular-nums">
                            {fmtPct(st.kept_rate_excl_burst)}
                          </span>
                        </div>
                        <StackedBar
                          parts={[
                            { value: st.kept_rate_total, color: SEG.kept },
                            { value: st.total > 0 ? st.burst_cull / st.total : 0, color: SEG.burst },
                            { value: st.tech_reject_rate, color: SEG.tech },
                          ]}
                        />
                        <div className="flex flex-wrap items-center justify-between gap-x-4 text-ink-3 tabular-nums">
                          <span>
                            {st.kept} / {st.non_burst_total}
                            <span className="ml-2">{t.digestTotalShots(st.total)}</span>
                          </span>
                          <span>
                            {t.digestColTechRate} {st.tech_reject} / {st.total} ({fmtPct(st.tech_reject_rate)})
                          </span>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </section>

              <section className="flex flex-col gap-6">
                <SectionTitle>{t.digestDimensionsTitle}</SectionTitle>
                <Tabs
                  size="sm"
                  value={activeDim?.key ?? ""}
                  onChange={setSelectedDim}
                  items={digest.dimensions.map((dim) => ({ id: dim.key, label: dim.label }))}
                />
                {activeDim && (
                  <ul className="flex flex-col gap-8 pt-2">
                    {activeDim.buckets.map((b) => (
                      <li key={b.key} className="flex flex-col gap-2 text-sm">
                        <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
                          <span className="flex items-baseline gap-2">
                            <span className="font-medium text-ink">{b.label}</span>
                            <span className="text-ink-3 tabular-nums">{t.photoCount(b.total)}</span>
                            {!b.meets_min_samples && (
                              <span className="text-ink-3">{t.digestLowSampleBadge(digest.scope.min_samples)}</span>
                            )}
                          </span>
                          <span className="flex flex-wrap items-center gap-x-4 gap-y-1 text-ink-2 tabular-nums">
                            <span className="inline-flex items-center gap-2">
                              <Swatch color={SEG.kept} />
                              {t.digestColKeptRate} {b.kept}/{b.non_burst_total} ({fmtPct(b.kept_rate_excl_burst)})
                            </span>
                            <span className="inline-flex items-center gap-2">
                              <Swatch color={SEG.tech} />
                              {t.digestColTechRate} {b.tech_reject}/{b.total} ({fmtPct(b.tech_reject_rate)})
                            </span>
                            {b.burst_cull > 0 && (
                              <span className="inline-flex items-center gap-2">
                                <Swatch color={SEG.burst} />
                                {t.digestColBurstRate} {b.burst_cull}/{b.total} ({fmtPct(b.burst_cull_rate)})
                              </span>
                            )}
                            {b.subjective_reject > 0 && (
                              <span className="inline-flex items-center gap-2">
                                <Swatch color={SEG.subjective} />
                                {t.digestColSubjectiveRate} {b.subjective_reject}/{b.total} (
                                {fmtPct(b.subjective_reject_rate)})
                              </span>
                            )}
                            {b.uncertain > 0 && (
                              <span className="inline-flex items-center gap-2">
                                <Swatch color={SEG.uncertain} />
                                {t.digestColUncertain} {b.uncertain}/{b.total} ({fmtPct(b.uncertain_rate)})
                              </span>
                            )}
                          </span>
                        </div>
                        <StackedBar
                          parts={[
                            { value: b.kept_rate_total, color: SEG.kept },
                            { value: b.uncertain_rate, color: SEG.uncertain },
                            { value: b.burst_cull_rate, color: SEG.burst },
                            { value: b.subjective_reject_rate, color: SEG.subjective },
                            { value: b.tech_reject_rate, color: SEG.tech },
                          ]}
                        />
                        {b.top_reason_label && (
                          <span className="text-ink-3">
                            {t.digestMainReasonPrefix}
                            {b.top_reason_label} ({b.top_reason_count})
                          </span>
                        )}
                      </li>
                    ))}
                  </ul>
                )}
              </section>
            </>
          )}
        </div>
      ) : activeTab === "memory" ? (
        <div className="mt-12 flex flex-col gap-16">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex flex-col gap-1">
              <h1
                className="text-lg font-semibold text-ink"
                title={`${t.memorySubtitle}\n${t.memoryScopeInjected}\n${t.memoryScopeZeroLlm}`}
              >
                {t.memoryTitle}
              </h1>
              <p className="flex flex-wrap items-center gap-x-4 text-sm text-ink-3">
                {memoryQ.data && (
                  <span
                    className={`tabular-nums ${memoryQ.data.ready_to_consolidate ? "font-medium text-mark" : ""}`}
                    title={memoryQ.data.ready_to_consolidate ? t.memoryReadyHint(memoryQ.data.new_override_count) : undefined}
                  >
                    {t.memoryNewOverridesRatio(
                      memoryQ.data.new_override_count,
                      memoryQ.data.min_overrides_threshold,
                    )}
                  </span>
                )}
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {memoryQ.data &&
                !memoryQ.data.ready_to_consolidate &&
                memoryQ.data.total_override_count > 0 && (
                  <Button
                    variant="ghost"
                    disabled={memoryBusy !== null}
                    onClick={() => void handleConsolidate(true)}
                  >
                    {t.memoryConsolidateForceBtn}
                  </Button>
                )}
              <Button
                variant={memoryQ.data?.ready_to_consolidate ? "primary" : "secondary"}
                icon={Sparkles}
                disabled={memoryBusy !== null}
                onClick={() => void handleConsolidate(false)}
              >
                {memoryBusy === "consolidate" ? t.memoryConsolidating : t.memoryConsolidateBtn}
              </Button>
            </div>
          </div>

          {(() => {
            const allRules = memoryQ.data?.rules ?? [];
            const proposed = allRules.filter((r) => r.status === "proposed");
            const decided = allRules.filter((r) => r.status !== "proposed");
            return (
              <>
                <section className="flex flex-col gap-6">
                  <SectionTitle>{t.memoryProposedTitle(proposed.length)}</SectionTitle>
                  {proposed.length === 0 ? (
                    <p className="text-sm text-ink-3">{t.memoryProposedEmpty}</p>
                  ) : (
                    <ul className="flex flex-col gap-8">
                      {proposed.map((rule) => {
                        const edited = ruleEdits[rule.rule_id] ?? rule.rule_text;
                        const isBusy = memoryBusy === `rule:${rule.rule_id}`;
                        return (
                          <li key={rule.rule_id} className="flex flex-col gap-2">
                            <div className="flex items-center gap-4 text-sm text-ink-3 tabular-nums">
                              <span>{rule.rule_id}</span>
                              <span>{t.memoryEvidenceCount(rule.evidence_ids.length)}</span>
                            </div>
                            <input
                              type="text"
                              value={edited}
                              onChange={(e) =>
                                setRuleEdits((prev) => ({ ...prev, [rule.rule_id]: e.target.value }))
                              }
                              aria-label={rule.rule_id}
                              className="field h-12 w-full text-base"
                            />
                            {rule.rationale && <p className="text-sm text-ink-2">{rule.rationale}</p>}
                            <div className="flex items-center justify-end gap-2 pt-2">
                              <Button
                                variant="ghost"
                                disabled={isBusy}
                                onClick={() => void handleDecideRule(rule.rule_id, "reject", rule.rule_text)}
                              >
                                {t.memoryRejectBtn}
                              </Button>
                              <Button
                                variant="primary"
                                disabled={isBusy || !edited.trim()}
                                onClick={() => void handleDecideRule(rule.rule_id, "accept", rule.rule_text)}
                              >
                                {t.memoryAcceptBtn}
                              </Button>
                            </div>
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </section>

                <section className="flex flex-col gap-6">
                  <SectionTitle
                    aside={
                      <div className="flex items-center gap-4">
                        {prefsDirty && (
                          <span className="fade-enter flex items-center gap-2 text-sm text-ink-2">
                            <span className="h-2 w-2 rounded-full bg-mark" />
                            {t.memoryUnsavedBadge}
                          </span>
                        )}
                        <Button
                          variant="primary"
                          disabled={memoryBusy === "save" || !prefsDirty}
                          onClick={() => void handleSavePreferences()}
                        >
                          {memoryBusy === "save" ? t.memorySaving : t.memorySaveBtn}
                        </Button>
                      </div>
                    }
                  >
                    <span title={t.memoryEditorHint}>{t.memoryEditorTitle}</span>
                  </SectionTitle>
                  <textarea
                    value={prefsDraft}
                    onChange={(e) => {
                      setPrefsDraft(e.target.value);
                      setPrefsDirty(true);
                    }}
                    rows={12}
                    aria-label={t.memoryEditorTitle}
                    className="field w-full rounded-panel p-6 text-sm leading-6"
                  />
                </section>

                {decided.length > 0 && (
                  <section className="flex flex-col gap-6">
                    <SectionTitle>{t.memoryHistoryTitle}</SectionTitle>
                    <ul className="flex flex-col gap-1 text-sm">
                      {decided.map((r) => (
                        <li key={r.rule_id} className="flex items-center justify-between gap-4 py-2">
                          <span className="flex min-w-0 items-center gap-4">
                            <span className="shrink-0 text-ink-3 tabular-nums">{r.rule_id}</span>
                            <span className={`truncate ${r.status === "accepted" ? "text-ink" : "text-ink-3 line-through"}`}>
                              {r.rule_text}
                            </span>
                          </span>
                          <span className="shrink-0 text-ink-2">{t.memoryStatusLabels[r.status] ?? r.status}</span>
                        </li>
                      ))}
                    </ul>
                  </section>
                )}
              </>
            );
          })()}
        </div>
      ) : activeTab === "traces" ? (
        <div className="mt-12 flex flex-col gap-16">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <h1 className="text-lg font-semibold text-ink" title={t.tracesSubtitle}>
              {t.tracesTitle}
            </h1>
            {sessionSelect}
          </div>

          {tracesQ.isLoading ? (
            <Loading label={t.loading} />
          ) : tracesQ.isError ? (
            <p className="text-sm text-mark">
              {t.errorPrefix}
              {tracesQ.error instanceof Error ? tracesQ.error.message : String(tracesQ.error)}
            </p>
          ) : tracesQ.data ? (
            <>
              <dl className="grid grid-cols-2 gap-x-8 gap-y-8 lg:grid-cols-4">
                <div className="flex flex-col gap-1">
                  <dt className="text-sm text-ink-2">{t.tracesMonthlySpend}</dt>
                  <dd className="text-[32px] font-semibold leading-10 text-ink tabular-nums">
                    ${tracesQ.data.monthly_spent_usd.toFixed(4)}
                  </dd>
                  <dd className="flex flex-col gap-2 pt-1">
                    <ProgressLine
                      tone={
                        tracesQ.data.monthly_budget_usd > 0 &&
                        tracesQ.data.monthly_spent_usd / tracesQ.data.monthly_budget_usd > 0.8
                          ? "mark"
                          : "ink"
                      }
                      value={
                        tracesQ.data.monthly_budget_usd > 0
                          ? (tracesQ.data.monthly_spent_usd / tracesQ.data.monthly_budget_usd) * 100
                          : 0
                      }
                    />
                    <span className="text-sm text-ink-3 tabular-nums">
                      {t.tracesBudgetOf(tracesQ.data.monthly_budget_usd.toFixed(2))}
                      <span className="ml-2">{tracesQ.data.model_profile}</span>
                    </span>
                  </dd>
                </div>
                <div className="flex flex-col gap-1">
                  <dt className="text-sm text-ink-2">{t.tracesTotalCalls}</dt>
                  <dd className="text-[32px] font-semibold leading-10 text-ink tabular-nums">
                    {tracesQ.data.total_calls}
                  </dd>
                  <dd className="text-sm text-ink-3 tabular-nums">${tracesQ.data.total_cost_usd.toFixed(4)}</dd>
                </div>
                <div className="flex flex-col gap-1">
                  <dt className="text-sm text-ink-2">{t.tracesCacheHitRate}</dt>
                  <dd className="text-[32px] font-semibold leading-10 text-ink tabular-nums">
                    {fmtPct(tracesQ.data.cache_hit_rate)}
                  </dd>
                  <dd className="text-sm text-ink-3 tabular-nums">{t.tracesCacheHitHint(tracesQ.data.cache_hits)}</dd>
                </div>
                <div className="flex flex-col gap-1">
                  <dt className="text-sm text-ink-2">{t.tracesTokensSummary}</dt>
                  <dd className="text-[32px] font-semibold leading-10 text-ink tabular-nums">
                    {(tracesQ.data.total_in_tokens + tracesQ.data.total_out_tokens).toLocaleString()}
                  </dd>
                  <dd className="text-sm text-ink-3 tabular-nums">
                    {t.tracesCachedTokensHint(tracesQ.data.total_cached_tokens)}
                  </dd>
                </div>
              </dl>

              {tracesQ.data.items.length === 0 ? (
                <EmptyState title={t.tracesEmpty} className="py-16" />
              ) : (
                <div className="-mx-4 overflow-x-auto px-4">
                  <table className="w-full text-left text-sm tabular-nums">
                    <thead className="text-ink-3">
                      <tr>
                        <th className="h-10 pr-6 font-normal">{t.tracesColTime}</th>
                        <th className="h-10 pr-6 font-normal">{t.tracesColRole}</th>
                        <th className="h-10 pr-6 font-normal">{t.tracesColModel}</th>
                        <th className="h-10 pr-6 text-right font-normal">{t.tracesColTokens}</th>
                        <th className="h-10 pr-6 text-right font-normal">{t.tracesColLatency}</th>
                        <th className="h-10 pr-6 text-right font-normal">{t.tracesColCost}</th>
                        <th className="h-10 font-normal">{t.tracesColCache}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {tracesQ.data.items.map((row) => (
                        <tr key={row.id} className="hover:bg-table">
                          <td className="h-12 whitespace-nowrap pr-6 text-ink-2">
                            {new Date(row.created_at * 1000).toLocaleString()}
                          </td>
                          <td className="h-12 whitespace-nowrap pr-6 font-medium text-ink">
                            {t.tracesRoleLabels[row.role] ?? row.role}
                          </td>
                          <td className="h-12 whitespace-nowrap pr-6 text-ink-2">{row.model}</td>
                          <td className="h-12 whitespace-nowrap pr-6 text-right text-ink-2">
                            {row.in_tokens.toLocaleString()} / {row.cached_tokens.toLocaleString()} /{" "}
                            {row.out_tokens.toLocaleString()}
                          </td>
                          <td className="h-12 whitespace-nowrap pr-6 text-right text-ink-2">{row.latency_ms} ms</td>
                          <td className="h-12 whitespace-nowrap pr-6 text-right text-ink">${row.cost_usd.toFixed(5)}</td>
                          <td className="h-12 whitespace-nowrap">
                            <span className={row.cache_hit ? "font-medium text-ink" : "text-ink-3"}>
                              {row.cache_hit ? t.tracesCacheHitBadge : t.tracesCacheMissBadge}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          ) : null}
        </div>
      ) : (
        <div className="mt-12 flex flex-col gap-12">
          <h1 className="text-lg font-semibold text-ink" title={t.insightsSubtitle}>
            {t.insightsTitle}
          </h1>

          <div className="flex flex-col gap-12">
            {plansQ.isLoading ? (
              <Loading label={t.loading} />
            ) : plans.length === 0 ? (
              <EmptyState title={t.insightsEmpty} />
            ) : (
              plans.map((plan) => {
                const expanded = expandedId === plan.id;
                const canRestore = plan.status === "applied" || plan.status === "applying";
                const activeInBatch = plan.items.filter((it) => it.status !== "reverted").length;
                const verifierLabel = (() => {
                  if (!plan.verifier_report) return null;
                  const l2 = plan.verifier_report.layer2;
                  const flaggedLen = plan.verifier_report.flagged_photos?.length ?? 0;
                  if (flaggedLen > 0) return { text: t.verifierTrashBadgeFlagged(flaggedLen), title: l2?.model_id ?? undefined, accent: true };
                  if (l2?.status === "passed") return { text: t.verifierTrashBadgeTwoLayer, title: l2.model_id ?? undefined, accent: false };
                  return { text: t.verifierTrashBadgeRules, title: undefined, accent: false };
                })();
                return (
                  <article key={plan.id} className="intent flex flex-col gap-6">
                    <div className="flex flex-wrap items-start justify-between gap-4">
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
                          <h2 className="text-lg font-semibold text-ink">{plan.session_name ?? plan.id}</h2>
                          <span className="text-sm text-ink-2 tabular-nums">
                            {t.photoCount(canRestore ? activeInBatch : plan.item_count)}
                          </span>
                          <span className={`text-sm ${canRestore ? "font-medium text-ink" : "text-ink-3"}`}>
                            {t.planStatus[plan.status] ?? plan.status}
                          </span>
                          {verifierLabel && (
                            <span
                              title={verifierLabel.title}
                              className={`inline-flex items-center gap-1 text-sm ${verifierLabel.accent ? "text-mark" : "text-ink-3"}`}
                            >
                              <Icon as={Sparkles} />
                              {verifierLabel.text}
                            </span>
                          )}
                        </div>
                        {plan.quarantine_dir && (
                          <p className="mt-1 truncate text-sm text-ink-3" title={plan.quarantine_dir}>
                            {plan.quarantine_dir}
                          </p>
                        )}
                      </div>

                      <div className="flex shrink-0 flex-wrap items-center gap-2">
                        {plan.quarantine_dir && canRestore && (
                          <Button variant="ghost" icon={Folder} className="reveal" onClick={() => void handleOpenFolder(plan.id)}>
                            {t.openTrashFolder}
                          </Button>
                        )}
                        <Button
                          variant="ghost"
                          iconRight={ChevronDown}
                          aria-expanded={expanded}
                          className={expanded ? "[&_svg]:rotate-180" : ""}
                          onClick={() => setExpandedId(expanded ? null : plan.id)}
                        >
                          {expanded ? t.hideDetails : t.showDetails}
                        </Button>
                        {canRestore && (
                          <Button
                            variant="primary"
                            icon={RotateLeft}
                            disabled={busyKey === plan.id}
                            onClick={() => void handleRestoreBatch(plan)}
                          >
                            {t.restoreAll}
                          </Button>
                        )}
                      </div>
                    </div>

                    {/* Contact strip of what's in the batch */}
                    {!expanded && plan.items.length > 0 && (
                      <button
                        type="button"
                        tabIndex={-1}
                        aria-hidden="true"
                        onClick={() => setExpandedId(plan.id)}
                        className="grid w-full grid-cols-[repeat(auto-fill,minmax(88px,1fr))] grid-rows-[64px] auto-rows-[0px] gap-x-2 gap-y-0 overflow-hidden"
                      >
                        {plan.items.map((item) => (
                          <Thumb
                            key={item.photo_id}
                            src={photoImageUrl(item.photo_id, "grid")}
                            verdict={item.status === "reverted" ? "keep" : "reject"}
                            veiled={item.status !== "reverted"}
                            className="h-16"
                          />
                        ))}
                      </button>
                    )}

                    {expanded && (
                      <ul className="fade-enter grid grid-cols-1 gap-x-8 gap-y-4 sm:grid-cols-2">
                        {plan.items.map((item) => {
                          const isReverted = item.status === "reverted";
                          return (
                            <li key={item.photo_id} className="flex items-center justify-between gap-4">
                              <div className="flex min-w-0 items-center gap-4">
                                <Thumb
                                  src={photoImageUrl(item.photo_id, "grid")}
                                  verdict={isReverted ? "keep" : "reject"}
                                  veiled={!isReverted}
                                  className="h-12 w-16 shrink-0"
                                />
                                <div className="min-w-0 text-sm">
                                  <div className="truncate font-medium text-ink">
                                    {item.src_paths[0]?.split("/").pop() ?? item.photo_id}
                                  </div>
                                  <div className="text-ink-3 tabular-nums">{t.fileCount(item.src_paths.length)}</div>
                                </div>
                              </div>
                              {isReverted ? (
                                <span className="shrink-0 text-sm text-ink-2">{t.restoredBadge}</span>
                              ) : (
                                <Button
                                  variant="secondary"
                                  size="sm"
                                  disabled={busyKey === `${plan.id}:${item.photo_id}`}
                                  onClick={() => void handleRestorePhoto(plan.id, item.photo_id)}
                                >
                                  {t.restoreSingle}
                                </Button>
                              )}
                            </li>
                          );
                        })}
                      </ul>
                    )}
                  </article>
                );
              })
            )}
          </div>
        </div>
      )}

      <NoticeDock>
        {activeTab === "trash" && notice && (
          <Notice onDismiss={() => setNotice(null)} dismissLabel={t.uiDismiss}>
            {notice}
          </Notice>
        )}
        {activeTab === "trash" && error && (
          <Notice tone="error" onDismiss={() => setError(null)} dismissLabel={t.uiDismiss}>
            {t.errorPrefix}
            {error}
          </Notice>
        )}
        {activeTab === "memory" && memoryNotice && (
          <Notice onDismiss={() => setMemoryNotice(null)} dismissLabel={t.uiDismiss}>
            {memoryNotice}
          </Notice>
        )}
        {activeTab === "memory" && memoryError && (
          <Notice tone="error" onDismiss={() => setMemoryError(null)} dismissLabel={t.uiDismiss}>
            {t.errorPrefix}
            {memoryError}
          </Notice>
        )}
      </NoticeDock>
    </div>
  );
}
