import { useEffect, useState } from "react";
import {
  api,
  photoImageUrl,
  type DraftObviousResponse,
  type VerifierReport,
} from "../api";
import { useI18n, type Messages } from "../i18n";

interface QuarantineModalProps {
  draft: DraftObviousResponse | null;
  verifying?: boolean;
  onClose: () => void;
  onApplied: (movedCount: number, planId: string) => void;
  onReturned?: (returnedCount: number) => void;
}

function formatSkippedReason(reason: string | null | undefined, t: Messages): string {
  if (reason === "no_api_key") return t.verifierSkippedNoKey;
  if (reason === "all_user_confirmed") return t.verifierSkippedUserOnly;
  if (reason === "disabled_in_profile") return t.verifierSkippedProfile;
  if (reason === "budget_exceeded") return t.verifierSkippedBudget;
  return t.verifierSkippedGeneric;
}

function VerifierBanner({ report, t }: { report: VerifierReport; t: Messages }) {
  const l2 = report.layer2;
  const flagged = report.flagged_photos ?? [];
  const exemptCount = Number(report.exempt_count ?? 0);
  const unconfirmedCount = Number(report.unconfirmed_burst_non_pick_count ?? 0);
  const incompleteCount = Number(report.incomplete_atomic_pair_count ?? 0);

  const isFlagged = flagged.length > 0 || l2?.status === "flagged";
  const isTwoLayerPassed = !isFlagged && l2?.status === "passed";

  const statusPillClass = isFlagged
    ? "bg-[#D99B26]/20 text-[#FCD34D]"
    : isTwoLayerPassed
      ? "bg-[#3D9A6B]/20 text-[#86EFAC]"
      : "bg-[#2C2F34] text-[#D1D5DB]";

  const statusText = isFlagged
    ? t.verifierFlaggedBadge(flagged.length)
    : isTwoLayerPassed
      ? t.verifierPassedTwoLayer(Boolean(l2?.cross_family))
      : t.verifierPassedDeterministic;

  return (
    <div className="space-y-2.5 rounded-md border border-[#33363B] bg-[#1B1C1E] px-3.5 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <span
            aria-hidden="true"
            className={`inline-block h-2 w-2 rounded-full ${
              isFlagged ? "bg-[#D99B26]" : "bg-[#3D9A6B]"
            }`}
          />
          <span className="text-xs font-medium text-[#E5E7EB]">{t.verifierLabel}</span>
          <span className={`rounded px-2 py-0.5 text-[11px] font-medium ${statusPillClass}`}>
            {statusText}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-[11px] text-[#9CA3AF] tabular-nums">
          {l2 && l2.status !== "skipped" && l2.model_id ? (
            <>
              <span className="rounded border border-[#33363B] bg-[#242629] px-1.5 py-0.5 font-mono text-[10px] text-[#D1D5DB]">
                {l2.model_id}
              </span>
              {l2.checked_count > 0 && <span>{t.verifierCheckedCount(l2.checked_count)}</span>}
            </>
          ) : (
            <span>{formatSkippedReason(l2?.skipped_reason, t)}</span>
          )}
        </div>
      </div>

      <div className="text-[11px] text-[#9CA3AF] tabular-nums">
        {t.verifierLayer1Checks(exemptCount, unconfirmedCount, incompleteCount)}
        {l2?.summary ? ` · ${l2.summary}` : ""}
      </div>

      {flagged.length > 0 && (
        <div className="rounded border border-[#D99B26]/40 bg-[#D99B26]/10 px-3 py-2 text-xs text-[#F5D07A]">
          <div className="font-medium">{t.verifierFlaggedHeader(flagged.length)}</div>
          <ul className="mt-1.5 space-y-1 text-[11px] text-[#FDE68A]">
            {flagged.map((item) => (
              <li key={item.photo_id} className="flex flex-wrap gap-x-2">
                <span className="font-mono font-medium">{item.stem}</span>
                <span className="text-[#F5D07A]/90">— {item.reason}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export function QuarantineModal({
  draft,
  verifying = false,
  onClose,
  onApplied,
  onReturned,
}: QuarantineModalProps) {
  const { t } = useI18n();
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const plan = draft?.plan ?? null;
  const items = plan ? plan.items : [];
  const report = draft?.verifier_report ?? plan?.verifier_report ?? null;
  const flaggedCount = report?.flagged_photos?.length ?? 0;

  const [selectedIds, setSelectedIds] = useState<Set<string>>(
    () => new Set(items.map((it) => it.photo_id)),
  );

  useEffect(() => {
    setSelectedIds(new Set(items.map((it) => it.photo_id)));
  }, [plan?.id]);

  const ratio = Number(report?.reject_ratio ?? 0);
  const ratioPct = `${(ratio * 100).toFixed(1)}%`;
  const allSelected = items.length > 0 && selectedIds.size === items.length;

  function toggleOne(photoId: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(photoId)) {
        next.delete(photoId);
      } else {
        next.add(photoId);
      }
      return next;
    });
  }

  function handleToggleAll() {
    if (allSelected) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(items.map((it) => it.photo_id)));
    }
  }

  async function handleConfirm() {
    if (!plan || selectedIds.size === 0) return;
    setApplying(true);
    setError(null);
    try {
      if (selectedIds.size === items.length) {
        const res = await api.approveAndApplyPlan(plan.id, plan.plan_hash);
        onApplied(res.moved_photo_count, res.plan_id);
      } else {
        // Move ONLY the checked photos; skip duplicate LLM audit since this subset was already verified
        const sub = await api.createPlan(
          plan.session_id,
          Array.from(selectedIds),
          "subfolder",
          true,
        );
        const res = await api.approveAndApplyPlan(sub.plan.id, sub.plan.plan_hash);
        onApplied(res.moved_photo_count, res.plan_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setApplying(false);
    }
  }

  async function handleReturnToReview() {
    if (!plan || selectedIds.size === 0) return;
    setApplying(true);
    setError(null);
    try {
      const ids = Array.from(selectedIds);
      await api.setVerdict(ids, "normal", "spared_from_trash", ["user_reset"]);
      if (onReturned) {
        onReturned(ids.length);
      } else {
        onClose();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setApplying(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="quarantine-modal-title"
    >
      <div className="flex max-h-[88vh] w-full max-w-3xl flex-col rounded-lg border border-[#33363B] bg-[#242629] text-[#F3F4F6]">
        {/* Header */}
        <div className="shrink-0 border-b border-[#33363B] px-6 py-4">
          <h2 id="quarantine-modal-title" className="text-lg font-medium">
            {t.modalTitle}
          </h2>
          {!verifying && plan && items.length > 0 && (
            <p className="mt-1 text-sm text-[#9CA3AF] tabular-nums">
              {t.modalSummary(selectedIds.size)}
            </p>
          )}
        </div>

        {verifying || !draft ? (
          <div className="space-y-4 px-6 py-8">
            <div className="flex items-center gap-3">
              <span className="inline-block h-2.5 w-2.5 animate-pulse rounded-full bg-[#5B8DEF]" />
              <span className="text-sm font-medium text-[#F3F4F6]">
                {t.verifierAuditingTitle}
              </span>
            </div>
            <div className="space-y-1.5 rounded-md border border-[#33363B] bg-[#1B1C1E] px-4 py-3 text-xs text-[#9CA3AF]">
              <div>{t.verifierAuditingLayer1}</div>
              <div>{t.verifierAuditingLayer2}</div>
            </div>
          </div>
        ) : !plan || draft.selected_count === 0 ? (
          <div className="space-y-4 p-6">
            {report && <VerifierBanner report={report} t={t} />}
            <p className="text-sm text-[#9CA3AF]">
              {flaggedCount > 0 ? t.verifierAllFlaggedEmpty : t.modalEmpty}
            </p>
            <div className="flex justify-end">
              <button
                type="button"
                onClick={onClose}
                className="rounded-md border border-[#33363B] px-4 py-2 text-sm text-[#F3F4F6] hover:bg-[#2C2F34]"
              >
                {t.modalCancel}
              </button>
            </div>
          </div>
        ) : (
          <>
            {/* Scrollable Body */}
            <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-6 py-4">
              {report && <VerifierBanner report={report} t={t} />}

              <ul className="flex flex-wrap gap-x-6 gap-y-1 rounded-md border border-[#33363B] bg-[#1B1C1E] px-3.5 py-2.5 text-xs text-[#9CA3AF] tabular-nums">
                {draft.tech_reject_count > 0 && (
                  <li>{t.modalTechCount(draft.tech_reject_count)}</li>
                )}
                {draft.burst_non_pick_count > 0 && (
                  <li>{t.modalBurstCount(draft.burst_non_pick_count)}</li>
                )}
                {draft.manual_reject_count > 0 && (
                  <li>{t.modalManualCount(draft.manual_reject_count)}</li>
                )}
              </ul>

              {draft.high_ratio_warning && (
                <div className="rounded-md border border-[#D99B26]/50 bg-[#D99B26]/10 px-3.5 py-2 text-xs text-[#F5D07A]">
                  {t.modalHighRatioWarn(ratioPct)}
                </div>
              )}

              {/* Full scrollable selectable photo grid with per-photo reason labels */}
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
                {items.map((it) => {
                  const checked = selectedIds.has(it.photo_id);
                  const filename =
                    it.stem ?? it.src_paths[0]?.split("/").pop() ?? it.photo_id;
                  const reasonText =
                    it.reason_labels && it.reason_labels.length > 0
                      ? it.reason_labels.join("；")
                      : t.modalReasonFallback;
                  return (
                    <button
                      key={it.photo_id}
                      type="button"
                      onClick={() => toggleOne(it.photo_id)}
                      className={`group flex flex-col overflow-hidden rounded-md border text-left transition ${
                        checked
                          ? "border-[#DC4C46] bg-[#1B1C1E] ring-1 ring-[#DC4C46]/50"
                          : "border-[#33363B] bg-[#1B1C1E]/60 opacity-55 hover:opacity-90"
                      }`}
                    >
                      <div className="relative aspect-[4/3] w-full overflow-hidden bg-[#141517]">
                        <img
                          src={photoImageUrl(it.photo_id, "grid")}
                          alt={filename}
                          loading="lazy"
                          className="h-full w-full object-cover"
                        />
                        <span
                          className={`absolute top-1.5 left-1.5 flex h-5 w-5 items-center justify-center rounded border text-xs font-bold ${
                            checked
                              ? "border-[#DC4C46] bg-[#DC4C46] text-white"
                              : "border-white/50 bg-black/55 text-transparent"
                          }`}
                        >
                          ✓
                        </span>
                      </div>
                      <div className="space-y-0.5 px-2 py-1.5">
                        <div className="truncate text-[11px] font-medium text-[#F3F4F6]">
                          {filename}
                        </div>
                        <div className="line-clamp-2 text-[11px] leading-snug text-[#FCA5A5]">
                          {reasonText}
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>

              <div className="space-y-1 text-xs text-[#9CA3AF]">
                <div>{t.modalTargetDir}</div>
                <div className="truncate rounded border border-[#33363B] bg-[#1B1C1E] px-2.5 py-1.5 font-mono text-[#D1D5DB]">
                  {plan.quarantine_dir}
                </div>
              </div>

              <p className="text-xs leading-relaxed text-[#9CA3AF]">{t.modalSafetyPromise}</p>

              {error && (
                <div className="rounded border border-[#DC4C46]/50 bg-[#DC4C46]/10 px-3 py-2 text-sm text-[#FCA5A5]">
                  {t.errorPrefix}
                  {error}
                </div>
              )}
            </div>

            {/* Footer with Select All + Cancel + Move */}
            <div className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-t border-[#33363B] px-6 py-3.5">
              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  disabled={applying}
                  onClick={handleToggleAll}
                  className="rounded-md border border-[#33363B] bg-[#1B1C1E] px-3 py-1.5 text-xs font-medium text-[#F3F4F6] hover:bg-[#2C2F34] disabled:opacity-50"
                >
                  {allSelected ? t.modalDeselectAll : t.modalSelectAll}
                </button>
                <span className="text-xs text-[#9CA3AF] tabular-nums">
                  {t.modalSelectedCount(selectedIds.size, items.length)}
                </span>
                {items.length - selectedIds.size > 0 && selectedIds.size > 0 && (
                  <span className="text-xs text-[#9CA3AF] tabular-nums">
                    {t.modalUncheckedStayHint(items.length - selectedIds.size)}
                  </span>
                )}
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  disabled={applying}
                  onClick={onClose}
                  className="rounded-md border border-[#33363B] px-3.5 py-2 text-sm text-[#D1D5DB] hover:bg-[#2C2F34] disabled:opacity-50"
                >
                  {t.modalCancel}
                </button>
                <button
                  type="button"
                  disabled={applying || selectedIds.size === 0}
                  onClick={() => void handleReturnToReview()}
                  className="rounded-md border border-[#5B8DEF]/50 bg-[#5B8DEF]/15 px-3.5 py-2 text-sm font-medium text-[#93C5FD] hover:bg-[#5B8DEF]/25 disabled:cursor-not-allowed disabled:opacity-45 tabular-nums"
                >
                  {t.modalReturnToReview(selectedIds.size)}
                </button>
                <button
                  type="button"
                  disabled={applying || selectedIds.size === 0}
                  onClick={() => void handleConfirm()}
                  className="rounded-md bg-[#DC4C46] px-4 py-2 text-sm font-medium text-white hover:bg-[#C63E38] disabled:cursor-not-allowed disabled:opacity-45 tabular-nums"
                >
                  {applying ? t.modalApplying : t.modalConfirmWithCount(selectedIds.size)}
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
