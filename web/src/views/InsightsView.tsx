import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, photoImageUrl, type PlanDetail } from "../api";
import { useI18n } from "../i18n";

export function InsightsView() {
  const { t } = useI18n();
  const queryClient = useQueryClient();
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const plansQ = useQuery({
    queryKey: ["plans"],
    queryFn: () => api.listPlans(),
  });

  const plans = (plansQ.data?.items ?? []).filter(
    (p) => p.status === "applied" || p.status === "reverted" || p.status === "applying",
  );

  async function handleRestoreBatch(plan: PlanDetail) {
    setBusyKey(plan.id);
    setError(null);
    setNotice(null);
    try {
      const res = await api.restorePlan(plan.id);
      setNotice(t.restoreSuccessBatch(res.restored_photo_ids.length));
      await queryClient.invalidateQueries({ queryKey: ["plans"] });
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

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <div className="space-y-1">
        <h1 className="text-2xl font-medium tracking-tight text-[#F3F4F6]">{t.insightsTitle}</h1>
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
  );
}
