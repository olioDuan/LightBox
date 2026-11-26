import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, photoImageUrl, type DigestDimension, type PlanDetail } from "../api";
import { useI18n } from "../i18n";

interface InsightsViewProps {
  activeTab: "digest" | "trash";
  onChangeTab: (tab: "digest" | "trash") => void;
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

  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      {/* Section Tabs: Shooting Digest | Trash Folder */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#33363B] pb-4">
        <div className="inline-flex rounded-lg border border-[#33363B] bg-[#242629] p-1">
          <button
            type="button"
            onClick={() => onChangeTab("digest")}
            className={`rounded-md px-3.5 py-1.5 text-sm font-medium transition ${
              activeTab === "digest"
                ? "bg-[#33363B] text-[#F3F4F6]"
                : "text-[#9CA3AF] hover:text-[#F3F4F6]"
            }`}
          >
            {t.insightsTabDigest}
          </button>
          <button
            type="button"
            onClick={() => onChangeTab("trash")}
            className={`flex items-center gap-2 rounded-md px-3.5 py-1.5 text-sm font-medium transition ${
              activeTab === "trash"
                ? "bg-[#33363B] text-[#F3F4F6]"
                : "text-[#9CA3AF] hover:text-[#F3F4F6]"
            }`}
          >
            <span>{t.insightsTabTrash}</span>
            {activeTrashBatches > 0 && (
              <span className="rounded bg-[#DC4C46]/25 px-1.5 py-0.5 text-[11px] text-[#FCA5A5] tabular-nums">
                {activeTrashBatches}
              </span>
            )}
          </button>
        </div>

        {activeTab === "digest" && digest && (
          <div className="flex flex-wrap items-center gap-3 text-xs text-[#9CA3AF] tabular-nums">
            <span>{t.digestTotalShots(digest.overview.total_shots)}</span>
            <span>·</span>
            <span>{t.digestScopeSessions(digest.scope.session_count)}</span>
            <span>·</span>
            <span>{t.digestMinSamplesHint(digest.scope.min_samples)}</span>
          </div>
        )}
      </div>

      {activeTab === "digest" ? (
        <div className="mt-6 space-y-6">
          {/* Header & Scope Controls */}
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div className="space-y-1">
              <h1 className="text-2xl font-medium tracking-tight text-[#F3F4F6]">
                {t.digestTitle}
              </h1>
              <p className="text-sm text-[#9CA3AF]">{t.digestSubtitle}</p>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <select
                value={sessionId}
                onChange={(e) => setSessionId(e.target.value)}
                className="rounded-md border border-[#33363B] bg-[#242629] px-3 py-1.5 text-xs text-[#F3F4F6] focus:border-[#5B8DEF] focus:outline-none"
              >
                <option value="">{t.digestAllSessions}</option>
                {sessions.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>

              <input
                type="date"
                value={dateFrom}
                onChange={(e) => setDateFrom(e.target.value)}
                aria-label={t.searchDateFrom}
                className="rounded-md border border-[#33363B] bg-[#242629] px-2.5 py-1.5 text-xs text-[#F3F4F6] focus:border-[#5B8DEF] focus:outline-none"
              />
              <span className="text-xs text-[#6B7280]">–</span>
              <input
                type="date"
                value={dateTo}
                onChange={(e) => setDateTo(e.target.value)}
                aria-label={t.searchDateTo}
                className="rounded-md border border-[#33363B] bg-[#242629] px-2.5 py-1.5 text-xs text-[#F3F4F6] focus:border-[#5B8DEF] focus:outline-none"
              />

              {hasScopeFilters && (
                <button
                  type="button"
                  onClick={() => {
                    setSessionId("");
                    setDateFrom("");
                    setDateTo("");
                  }}
                  className="rounded-md border border-[#33363B] bg-[#242629] px-2.5 py-1.5 text-xs text-[#D1D5DB] hover:bg-[#2C2F34]"
                >
                  {t.digestResetScope}
                </button>
              )}
            </div>
          </div>

          {digestQ.isLoading ? (
            <p className="py-12 text-sm text-[#9CA3AF]">{t.loading}</p>
          ) : digestQ.isError ? (
            <div className="rounded-md border border-[#DC4C46]/50 bg-[#DC4C46]/10 px-4 py-3 text-sm text-[#FCA5A5]">
              {t.errorPrefix}
              {digestQ.error instanceof Error ? digestQ.error.message : String(digestQ.error)}
            </div>
          ) : !digest || digest.overview.total_shots === 0 ? (
            <div className="rounded-lg border border-[#33363B] bg-[#242629] py-14 text-center">
              <p className="text-base text-[#F3F4F6]">{t.digestEmpty}</p>
              <p className="mt-1 text-sm text-[#9CA3AF]">{t.digestEmptyHint}</p>
            </div>
          ) : (
            <>
              {/* 4-Way Mutually Exclusive Stacked Bar + Metric Cards */}
              <div className="space-y-4 rounded-lg border border-[#33363B] bg-[#242629] p-5">
                <div className="flex h-3 w-full overflow-hidden rounded-full bg-[#1B1C1E]">
                  {digest.overview.kept_count > 0 && (
                    <div
                      className="h-full bg-[#3D9A6B]"
                      style={{ width: clampBarPct(digest.overview.kept_rate_total) }}
                      title={`${t.digestCardKept}: ${digest.overview.kept_count}`}
                    />
                  )}
                  {digest.overview.uncertain_count > 0 && (
                    <div
                      className="h-full bg-[#D97706]"
                      style={{ width: clampBarPct(digest.overview.uncertain_rate) }}
                      title={`${t.digestCardUncertain}: ${digest.overview.uncertain_count}`}
                    />
                  )}
                  {digest.overview.burst_cull_count > 0 && (
                    <div
                      className="h-full bg-[#5B8DEF]"
                      style={{ width: clampBarPct(digest.overview.burst_cull_rate) }}
                      title={`${t.digestCardBurstCull}: ${digest.overview.burst_cull_count}`}
                    />
                  )}
                  {digest.overview.subjective_reject_count > 0 && (
                    <div
                      className="h-full bg-[#8B5CF6]"
                      style={{ width: clampBarPct(digest.overview.subjective_reject_rate) }}
                      title={`${t.digestCardSubjectiveReject}: ${digest.overview.subjective_reject_count}`}
                    />
                  )}
                  {digest.overview.tech_reject_count > 0 && (
                    <div
                      className="h-full bg-[#DC4C46]"
                      style={{ width: clampBarPct(digest.overview.tech_reject_rate) }}
                      title={`${t.digestCardTechReject}: ${digest.overview.tech_reject_count}`}
                    />
                  )}
                </div>

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
                  {/* Kept */}
                  <div className="rounded-md border border-[#33363B] bg-[#1B1C1E] p-3.5">
                    <div className="flex items-center gap-2 text-xs text-[#9CA3AF]">
                      <span className="h-2 w-2 rounded-full bg-[#3D9A6B]" />
                      <span>{t.digestCardKept}</span>
                    </div>
                    <div className="mt-2 flex items-baseline gap-2 tabular-nums">
                      <span className="text-2xl font-semibold text-[#F3F4F6]">
                        {fmtPct(digest.overview.kept_rate_excl_burst)}
                      </span>
                      <span className="text-xs text-[#86EFAC]">
                        {digest.overview.kept_count} / {digest.overview.non_burst_total}
                      </span>
                    </div>
                    <div className="mt-1 space-y-0.5 text-[11px] text-[#9CA3AF] tabular-nums">
                      <div>{t.digestCardKeptExclBurst}</div>
                      <div>
                        {t.digestCardKeptTotal}：{digest.overview.kept_count} /{" "}
                        {digest.overview.total_shots} ({fmtPct(digest.overview.kept_rate_total)})
                      </div>
                    </div>
                  </div>

                  {/* Tech Reject */}
                  <div className="rounded-md border border-[#33363B] bg-[#1B1C1E] p-3.5">
                    <div className="flex items-center gap-2 text-xs text-[#9CA3AF]">
                      <span className="h-2 w-2 rounded-full bg-[#DC4C46]" />
                      <span>{t.digestCardTechReject}</span>
                    </div>
                    <div className="mt-2 flex items-baseline gap-2 tabular-nums">
                      <span className="text-2xl font-semibold text-[#FCA5A5]">
                        {fmtPct(digest.overview.tech_reject_rate)}
                      </span>
                      <span className="text-xs text-[#9CA3AF]">
                        {digest.overview.tech_reject_count} / {digest.overview.total_shots}
                      </span>
                    </div>
                    <div className="mt-1 text-[11px] text-[#9CA3AF]">
                      {t.digestCardTechRejectHint}
                    </div>
                  </div>

                  {/* Burst Cull */}
                  <div className="rounded-md border border-[#33363B] bg-[#1B1C1E] p-3.5">
                    <div className="flex items-center gap-2 text-xs text-[#9CA3AF]">
                      <span className="h-2 w-2 rounded-full bg-[#5B8DEF]" />
                      <span>{t.digestCardBurstCull}</span>
                    </div>
                    <div className="mt-2 flex items-baseline gap-2 tabular-nums">
                      <span className="text-2xl font-semibold text-[#93C5FD]">
                        {fmtPct(digest.overview.burst_cull_rate)}
                      </span>
                      <span className="text-xs text-[#9CA3AF]">
                        {digest.overview.burst_cull_count} / {digest.overview.total_shots}
                      </span>
                    </div>
                    <div className="mt-1 text-[11px] text-[#9CA3AF]">
                      {t.digestCardBurstCullHint}
                    </div>
                  </div>

                  {/* Subjective Reject */}
                  <div className="rounded-md border border-[#33363B] bg-[#1B1C1E] p-3.5">
                    <div className="flex items-center gap-2 text-xs text-[#9CA3AF]">
                      <span className="h-2 w-2 rounded-full bg-[#8B5CF6]" />
                      <span>{t.digestCardSubjectiveReject}</span>
                    </div>
                    <div className="mt-2 flex items-baseline gap-2 tabular-nums">
                      <span className="text-2xl font-semibold text-[#C4B5FD]">
                        {fmtPct(digest.overview.subjective_reject_rate)}
                      </span>
                      <span className="text-xs text-[#9CA3AF]">
                        {digest.overview.subjective_reject_count} / {digest.overview.total_shots}
                      </span>
                    </div>
                    <div className="mt-1 text-[11px] text-[#9CA3AF]">
                      {t.digestCardSubjectiveRejectHint}
                    </div>
                  </div>

                  {/* Uncertain */}
                  <div className="rounded-md border border-[#33363B] bg-[#1B1C1E] p-3.5">
                    <div className="flex items-center gap-2 text-xs text-[#9CA3AF]">
                      <span className="h-2 w-2 rounded-full bg-[#D97706]" />
                      <span>{t.digestCardUncertain}</span>
                    </div>
                    <div className="mt-2 flex items-baseline gap-2 tabular-nums">
                      <span className="text-2xl font-semibold text-[#FCD34D]">
                        {fmtPct(digest.overview.uncertain_rate)}
                      </span>
                      <span className="text-xs text-[#9CA3AF]">
                        {digest.overview.uncertain_count} / {digest.overview.total_shots}
                      </span>
                    </div>
                    <div className="mt-1 text-[11px] text-[#9CA3AF]">
                      {t.digestCardUncertainHint}
                    </div>
                  </div>
                </div>
              </div>

              {/* Top Technical Reject Reasons */}
              {digest.top_tech_reasons.length > 0 && (
                <div className="rounded-lg border border-[#33363B] bg-[#242629] p-5">
                  <h2 className="text-sm font-medium text-[#F3F4F6]">{t.digestTopReasonsTitle}</h2>
                  <div className="mt-3 space-y-2.5">
                    {digest.top_tech_reasons.map((r) => (
                      <div key={r.code} className="space-y-1">
                        <div className="flex items-center justify-between gap-2 text-xs">
                          <span className="truncate text-[#E5E7EB]">{r.label}</span>
                          <span className="shrink-0 text-[#9CA3AF] tabular-nums">
                            {r.count} / {digest.overview.tech_reject_count} (
                            {fmtPct(r.rate_of_tech_rejects)})
                          </span>
                        </div>
                        <div className="h-2 w-full overflow-hidden rounded-full bg-[#1B1C1E]">
                          <div
                            className="h-full rounded-full bg-[#DC4C46]/85"
                            style={{ width: clampBarPct(r.rate_of_tech_rejects) }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* High Tech-Reject Conditions vs High Kept-Rate Conditions */}
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                <div className="rounded-lg border border-[#33363B] bg-[#242629] p-5">
                  <h2 className="text-sm font-medium text-[#F3F4F6]">{t.digestHighRejectTitle}</h2>
                  {digest.high_tech_reject_conditions.length === 0 ? (
                    <p className="mt-3 text-xs text-[#9CA3AF]">
                      {t.digestHighRejectEmpty(digest.scope.min_samples)}
                    </p>
                  ) : (
                    <div className="mt-3 space-y-3">
                      {digest.high_tech_reject_conditions.map((c) => (
                        <div
                          key={`${c.dimension_key}:${c.bucket_key}`}
                          className="rounded-md border border-[#33363B] bg-[#1B1C1E] p-3"
                        >
                          <div className="flex items-baseline justify-between gap-2">
                            <div className="min-w-0">
                              <span className="text-[11px] text-[#9CA3AF]">
                                {c.dimension_label}
                              </span>
                              <span className="mx-1.5 text-[#4B5563]">·</span>
                              <span className="text-xs font-medium text-[#F3F4F6]">
                                {c.bucket_label}
                              </span>
                            </div>
                            <span className="shrink-0 text-xs font-medium text-[#FCA5A5] tabular-nums">
                              {c.tech_reject} / {c.total} ({fmtPct(c.tech_reject_rate)})
                            </span>
                          </div>
                          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-[#242629]">
                            <div
                              className="h-full rounded-full bg-[#DC4C46]"
                              style={{ width: clampBarPct(c.tech_reject_rate) }}
                            />
                          </div>
                          <div className="mt-1.5 flex flex-wrap items-center justify-between gap-2 text-[11px] text-[#9CA3AF] tabular-nums">
                            <span className="text-[#FCA5A5]">
                              {t.digestDeltaAboveOverall((c.tech_reject_delta * 100).toFixed(1))}
                            </span>
                            {c.top_reason_label && (
                              <span className="truncate">
                                {t.digestMainReasonPrefix}
                                {c.top_reason_label}
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="rounded-lg border border-[#33363B] bg-[#242629] p-5">
                  <h2 className="text-sm font-medium text-[#F3F4F6]">{t.digestHighKeptTitle}</h2>
                  {digest.high_kept_conditions.length === 0 ? (
                    <p className="mt-3 text-xs text-[#9CA3AF]">
                      {t.digestHighKeptEmpty(digest.scope.min_samples)}
                    </p>
                  ) : (
                    <div className="mt-3 space-y-3">
                      {digest.high_kept_conditions.map((c) => (
                        <div
                          key={`${c.dimension_key}:${c.bucket_key}`}
                          className="rounded-md border border-[#33363B] bg-[#1B1C1E] p-3"
                        >
                          <div className="flex items-baseline justify-between gap-2">
                            <div className="min-w-0">
                              <span className="text-[11px] text-[#9CA3AF]">
                                {c.dimension_label}
                              </span>
                              <span className="mx-1.5 text-[#4B5563]">·</span>
                              <span className="text-xs font-medium text-[#F3F4F6]">
                                {c.bucket_label}
                              </span>
                            </div>
                            <span className="shrink-0 text-xs font-medium text-[#86EFAC] tabular-nums">
                              {c.kept} / {c.non_burst_total} ({fmtPct(c.kept_rate_excl_burst)})
                            </span>
                          </div>
                          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-[#242629]">
                            <div
                              className="h-full rounded-full bg-[#3D9A6B]"
                              style={{ width: clampBarPct(c.kept_rate_excl_burst) }}
                            />
                          </div>
                          <div className="mt-1.5 flex flex-wrap items-center justify-between gap-2 text-[11px] text-[#9CA3AF] tabular-nums">
                            <span className="text-[#86EFAC]">
                              {t.digestDeltaAboveOverall((c.kept_rate_delta * 100).toFixed(1))}
                            </span>
                            <span>
                              {t.digestTotalShots(c.total)} · {t.digestColTechRate}{" "}
                              {c.tech_reject} / {c.total} ({fmtPct(c.tech_reject_rate)})
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Fixed 2D Stable Setups: Focal Bucket x Shutter Bucket */}
              <div className="rounded-lg border border-[#33363B] bg-[#242629] p-5">
                <h2 className="text-sm font-medium text-[#F3F4F6]">{t.digestStableSetupsTitle}</h2>
                {digest.stable_setups.length === 0 ? (
                  <p className="mt-3 text-xs text-[#9CA3AF]">
                    {t.digestStableSetupsEmpty(digest.scope.min_samples)}
                  </p>
                ) : (
                  <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">
                    {digest.stable_setups.map((s) => (
                      <div
                        key={`${s.focal_bucket}:${s.shutter_bucket}`}
                        className="rounded-md border border-[#33363B] bg-[#1B1C1E] p-3"
                      >
                        <div className="flex items-baseline justify-between gap-2">
                          <span className="text-xs font-medium text-[#F3F4F6]">
                            {s.focal_label} × {s.shutter_label}
                          </span>
                          <span className="shrink-0 text-xs font-medium text-[#86EFAC] tabular-nums">
                            {s.kept} / {s.non_burst_total} ({fmtPct(s.kept_rate_excl_burst)})
                          </span>
                        </div>
                        <div className="mt-2 flex h-1.5 w-full overflow-hidden rounded-full bg-[#242629]">
                          <div
                            className="h-full bg-[#3D9A6B]"
                            style={{ width: clampBarPct(s.kept_rate_total) }}
                          />
                          <div
                            className="h-full bg-[#5B8DEF]"
                            style={{
                              width: clampBarPct(s.total > 0 ? s.burst_cull / s.total : 0),
                            }}
                          />
                          <div
                            className="h-full bg-[#DC4C46]"
                            style={{ width: clampBarPct(s.tech_reject_rate) }}
                          />
                        </div>
                        <div className="mt-1.5 flex flex-wrap items-center justify-between gap-2 text-[11px] text-[#9CA3AF] tabular-nums">
                          <span>{t.digestTotalShots(s.total)}</span>
                          <span>
                            {t.digestColTechRate} {s.tech_reject} / {s.total} (
                            {fmtPct(s.tech_reject_rate)})
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Single-Dimension Breakdown Explorer (9 Dimensions) */}
              <div className="rounded-lg border border-[#33363B] bg-[#242629] p-5">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <h2 className="text-sm font-medium text-[#F3F4F6]">{t.digestDimensionsTitle}</h2>
                  <div className="flex flex-wrap gap-1.5">
                    {digest.dimensions.map((dim) => (
                      <button
                        key={dim.key}
                        type="button"
                        onClick={() => setSelectedDim(dim.key)}
                        className={`rounded-md px-2.5 py-1 text-xs transition ${
                          activeDim?.key === dim.key
                            ? "bg-[#5B8DEF] font-medium text-white"
                            : "border border-[#33363B] bg-[#1B1C1E] text-[#9CA3AF] hover:text-[#F3F4F6]"
                        }`}
                      >
                        {dim.label}
                      </button>
                    ))}
                  </div>
                </div>

                {activeDim && (
                  <div className="mt-4 space-y-2.5">
                    {activeDim.buckets.map((b) => (
                      <div
                        key={b.key}
                        className="rounded-md border border-[#33363B] bg-[#1B1C1E] p-3"
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-medium text-[#F3F4F6]">{b.label}</span>
                            <span className="text-[11px] text-[#9CA3AF] tabular-nums">
                              {t.photoCount(b.total)}
                            </span>
                            {!b.meets_min_samples && (
                              <span className="rounded bg-[#2C2F34] px-1.5 py-0.5 text-[10px] text-[#9CA3AF]">
                                {t.digestLowSampleBadge(digest.scope.min_samples)}
                              </span>
                            )}
                          </div>

                          <div className="flex flex-wrap items-center gap-3 text-[11px] tabular-nums">
                            <span className="text-[#86EFAC]">
                              {t.digestColKeptRate}: {b.kept}/{b.non_burst_total} (
                              {fmtPct(b.kept_rate_excl_burst)})
                            </span>
                            <span className="text-[#FCA5A5]">
                              {t.digestColTechRate}: {b.tech_reject}/{b.total} (
                              {fmtPct(b.tech_reject_rate)})
                            </span>
                            {b.burst_cull > 0 && (
                              <span className="text-[#93C5FD]">
                                {t.digestColBurstRate}: {b.burst_cull}/{b.total} (
                                {fmtPct(b.burst_cull_rate)})
                              </span>
                            )}
                            {b.subjective_reject > 0 && (
                              <span className="text-[#C4B5FD]">
                                {t.digestColSubjectiveRate}: {b.subjective_reject}/{b.total} (
                                {fmtPct(b.subjective_reject_rate)})
                              </span>
                            )}
                            {b.uncertain > 0 && (
                              <span className="text-[#FCD34D]">
                                {t.digestColUncertain}: {b.uncertain}/{b.total} (
                                {fmtPct(b.uncertain_rate)})
                              </span>
                            )}
                          </div>
                        </div>

                        <div className="mt-2 flex h-2 w-full overflow-hidden rounded-full bg-[#242629]">
                          {b.kept > 0 && (
                            <div
                              className="h-full bg-[#3D9A6B]"
                              style={{ width: clampBarPct(b.kept_rate_total) }}
                            />
                          )}
                          {b.uncertain > 0 && (
                            <div
                              className="h-full bg-[#D97706]"
                              style={{ width: clampBarPct(b.uncertain_rate) }}
                            />
                          )}
                          {b.burst_cull > 0 && (
                            <div
                              className="h-full bg-[#5B8DEF]"
                              style={{ width: clampBarPct(b.burst_cull_rate) }}
                            />
                          )}
                          {b.subjective_reject > 0 && (
                            <div
                              className="h-full bg-[#8B5CF6]"
                              style={{ width: clampBarPct(b.subjective_reject_rate) }}
                            />
                          )}
                          {b.tech_reject > 0 && (
                            <div
                              className="h-full bg-[#DC4C46]"
                              style={{ width: clampBarPct(b.tech_reject_rate) }}
                            />
                          )}
                        </div>

                        {b.top_reason_label && (
                          <div className="mt-1.5 text-[11px] text-[#9CA3AF]">
                            {t.digestMainReasonPrefix}
                            {b.top_reason_label} ({b.top_reason_count})
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      ) : (
        <div className="mt-6">
          <div className="space-y-1">
            <h1 className="text-2xl font-medium tracking-tight text-[#F3F4F6]">
              {t.insightsTitle}
            </h1>
            <p className="text-sm text-[#9CA3AF]">{t.insightsSubtitle}</p>
          </div>

          {notice && (
            <div className="mt-4 rounded-md border border-[#3D9A6B]/50 bg-[#3D9A6B]/10 px-4 py-2.5 text-sm text-[#A7F3D0]">
              {notice}
            </div>
          )}

          {error && (
            <div className="mt-4 rounded-md border border-[#DC4C46]/50 bg-[#DC4C46]/10 px-4 py-2.5 text-sm text-[#FCA5A5]">
              {t.errorPrefix}
              {error}
            </div>
          )}

          <div className="mt-6 border-t border-[#33363B]">
            {plansQ.isLoading ? (
              <p className="py-10 text-sm text-[#9CA3AF]">{t.loading}</p>
            ) : plans.length === 0 ? (
              <div className="py-12 text-center">
                <p className="text-base text-[#F3F4F6]">{t.insightsEmpty}</p>
                <p className="mt-1 text-sm text-[#9CA3AF]">{t.insightsEmptyHint}</p>
              </div>
            ) : (
              plans.map((plan) => {
                const expanded = expandedId === plan.id;
                const canRestore = plan.status === "applied" || plan.status === "applying";
                const activeInBatch = plan.items.filter((it) => it.status !== "reverted").length;
                return (
                  <div key={plan.id} className="border-b border-[#33363B] py-4">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                      <div className="min-w-0 space-y-1">
                        <div className="flex flex-wrap items-center gap-2.5">
                          <span className="text-sm font-medium text-[#F3F4F6]">
                            {plan.session_name ?? plan.id}
                          </span>
                          <span
                            className={`rounded px-2 py-0.5 text-xs ${
                              canRestore
                                ? "bg-[#DC4C46]/20 text-[#FCA5A5]"
                                : "bg-[#3D9A6B]/20 text-[#86EFAC]"
                            }`}
                          >
                            {t.planStatus[plan.status] ?? plan.status}
                          </span>
                          <span className="text-xs text-[#9CA3AF] tabular-nums">
                            {t.photoCount(canRestore ? activeInBatch : plan.item_count)}
                          </span>
                        </div>
                        {plan.quarantine_dir && (
                          <div className="truncate font-mono text-xs text-[#9CA3AF]">
                            {plan.quarantine_dir}
                          </div>
                        )}
                      </div>

                      <div className="flex shrink-0 flex-wrap items-center gap-2">
                        {plan.quarantine_dir && canRestore && (
                          <button
                            type="button"
                            onClick={() => void handleOpenFolder(plan.id)}
                            className="rounded-md border border-[#33363B] bg-[#242629] px-3 py-1.5 text-xs text-[#F3F4F6] hover:bg-[#2C2F34]"
                          >
                            {t.openTrashFolder}
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={() => setExpandedId(expanded ? null : plan.id)}
                          className="rounded-md border border-[#33363B] bg-[#242629] px-3 py-1.5 text-xs text-[#D1D5DB] hover:bg-[#2C2F34]"
                        >
                          {expanded ? t.hideDetails : t.showDetails}
                        </button>
                        {canRestore && (
                          <button
                            type="button"
                            disabled={busyKey === plan.id}
                            onClick={() => void handleRestoreBatch(plan)}
                            className="rounded-md bg-[#3D9A6B] px-3.5 py-1.5 text-xs font-medium text-white hover:bg-[#32825A] disabled:opacity-50"
                          >
                            {t.restoreAll}
                          </button>
                        )}
                      </div>
                    </div>

                    {expanded && (
                      <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
                        {plan.items.map((item) => {
                          const isReverted = item.status === "reverted";
                          return (
                            <div
                              key={item.photo_id}
                              className="flex items-center justify-between gap-3 rounded-md border border-[#33363B] bg-[#242629] p-2"
                            >
                              <div className="flex min-w-0 items-center gap-3">
                                <img
                                  src={photoImageUrl(item.photo_id, "grid")}
                                  alt=""
                                  className="h-12 w-16 shrink-0 rounded bg-[#1B1C1E] object-cover"
                                />
                                <div className="min-w-0">
                                  <div className="truncate text-xs text-[#F3F4F6]">
                                    {item.src_paths[0]?.split("/").pop() ?? item.photo_id}
                                  </div>
                                  <div className="text-[11px] text-[#9CA3AF] tabular-nums">
                                    {t.fileCount(item.src_paths.length)}
                                  </div>
                                </div>
                              </div>

                              {isReverted ? (
                                <span className="shrink-0 text-xs text-[#3D9A6B]">
                                  {t.restoredBadge}
                                </span>
                              ) : (
                                <button
                                  type="button"
                                  disabled={busyKey === `${plan.id}:${item.photo_id}`}
                                  onClick={() => void handleRestorePhoto(plan.id, item.photo_id)}
                                  className="shrink-0 rounded border border-[#33363B] px-2.5 py-1 text-xs text-[#F3F4F6] hover:bg-[#2C2F34] disabled:opacity-50"
                                >
                                  {t.restoreSingle}
                                </button>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}
