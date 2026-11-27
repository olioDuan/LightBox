import { useEffect, useState } from "react";
import {
  api,
  photoImageUrl,
  type DraftObviousResponse,
  type VerifierReport,
} from "../api";
import { useI18n, type Messages } from "../i18n";
import { Button, Dialog, Icon, ProgressLine } from "./ui";
import { AlertTriangle, Check, Folder, Sparkles } from "reicon-react";

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

  const statusText = isFlagged
    ? t.verifierFlaggedBadge(flagged.length)
    : isTwoLayerPassed
      ? t.verifierPassedTwoLayer(Boolean(l2?.cross_family))
      : t.verifierPassedDeterministic;

  const detail = [
    t.verifierLayer1Checks(exemptCount, unconfirmedCount, incompleteCount),
    l2?.summary ?? null,
    l2 && l2.status !== "skipped" && l2.model_id
      ? `${l2.model_id}${l2.checked_count > 0 ? ` ${t.verifierCheckedCount(l2.checked_count)}` : ""}`
      : formatSkippedReason(l2?.skipped_reason, t),
  ]
    .filter(Boolean)
    .join("；");

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 text-sm" title={detail}>
        <Icon as={isFlagged ? AlertTriangle : Sparkles} className={isFlagged ? "text-unsure-ink" : "text-ink-2"} />
        <span className="font-medium text-ink">{t.verifierLabel}</span>
        <span className={isFlagged ? "text-unsure-ink" : "text-ink-2"}>{statusText}</span>
      </div>

      {flagged.length > 0 && (
        <ul className="space-y-1 pl-8 text-sm">
          {flagged.map((item) => (
            <li key={item.photo_id} className="flex flex-wrap gap-x-2">
              <span className="font-medium text-ink">{item.stem}</span>
              <span className="text-ink-2">{item.reason}</span>
            </li>
          ))}
        </ul>
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

  const counts = [
    draft && draft.tech_reject_count > 0 ? t.modalTechCount(draft.tech_reject_count) : null,
    draft && draft.burst_non_pick_count > 0 ? t.modalBurstCount(draft.burst_non_pick_count) : null,
    draft && draft.manual_reject_count > 0 ? t.modalManualCount(draft.manual_reject_count) : null,
  ].filter((c): c is string => Boolean(c));

  return (
    <Dialog labelledBy="quarantine-modal-title" width={880} onDismiss={applying ? undefined : onClose}>
      {/* Header */}
      <div className="shrink-0 px-8 pb-6 pt-8 sm:px-12 sm:pt-12">
        <h2 id="quarantine-modal-title" className="text-lg font-semibold text-ink">
          {t.modalTitle}
        </h2>
        {!verifying && plan && items.length > 0 && (
          <p className="mt-1 text-sm text-ink-2 tabular-nums">
            {t.modalSummary(selectedIds.size)}
            {counts.map((c) => (
              <span key={c} className="ml-4 text-ink-3">
                {c}
              </span>
            ))}
          </p>
        )}
      </div>

      {verifying || !draft ? (
        <div className="space-y-6 px-8 pb-12 sm:px-12">
          <div className="flex items-center gap-4">
            <ProgressLine indeterminate className="w-32" />
            <span className="text-sm font-medium text-ink">{t.verifierAuditingTitle}</span>
          </div>
          <ul className="space-y-1 text-sm text-ink-2">
            <li>{t.verifierAuditingLayer1}</li>
            <li>{t.verifierAuditingLayer2}</li>
          </ul>
        </div>
      ) : !plan || draft.selected_count === 0 ? (
        <div className="space-y-8 px-8 pb-8 sm:px-12 sm:pb-12">
          {report && <VerifierBanner report={report} t={t} />}
          <p className="text-sm text-ink-2">
            {flaggedCount > 0 ? t.verifierAllFlaggedEmpty : t.modalEmpty}
          </p>
          <div className="flex justify-end">
            <Button variant="secondary" onClick={onClose}>
              {t.modalCancel}
            </Button>
          </div>
        </div>
      ) : (
        <>
          {/* Scrollable body */}
          <div className="min-h-0 flex-1 space-y-8 overflow-y-auto px-8 pb-8 sm:px-12">
            {report && <VerifierBanner report={report} t={t} />}

            {draft.high_ratio_warning && (
              <p className="flex items-center gap-2 text-sm font-medium text-unsure-ink">
                <Icon as={AlertTriangle} />
                {t.modalHighRatioWarn(ratioPct)}
              </p>
            )}

            {/* Every photo in the plan; click to keep one back */}
            <div className="grid grid-cols-2 gap-x-4 gap-y-6 sm:grid-cols-3 md:grid-cols-4">
              {items.map((it) => {
                const checked = selectedIds.has(it.photo_id);
                const filename = it.stem ?? it.src_paths[0]?.split("/").pop() ?? it.photo_id;
                const reasonText =
                  it.reason_labels && it.reason_labels.length > 0
                    ? it.reason_labels.join("；")
                    : t.modalReasonFallback;
                return (
                  <button
                    key={it.photo_id}
                    type="button"
                    role="checkbox"
                    aria-checked={checked}
                    onClick={() => toggleOne(it.photo_id)}
                    className="group flex flex-col gap-2 text-left"
                  >
                    <div
                      className={`relative aspect-[4/3] w-full overflow-hidden rounded-chip border-2 bg-table bg-clip-padding transition-[border-color] duration-200 ${
                        checked ? "border-reject" : "border-transparent group-hover:border-line"
                      }`}
                    >
                      <img
                        src={photoImageUrl(it.photo_id, "grid")}
                        alt={filename}
                        loading="lazy"
                        className={`h-full w-full rounded-[5px] object-cover transition-[filter,opacity] duration-300 ${
                          checked ? "opacity-70 grayscale" : ""
                        }`}
                      />
                      <span
                        aria-hidden="true"
                        className={`absolute left-2 top-2 flex h-6 w-6 items-center justify-center rounded-full transition-colors duration-150 ${
                          checked
                            ? "bg-reject text-white"
                            : "bg-paper/80 shadow-[inset_0_0_0_1.5px_var(--color-ink-3)]"
                        }`}
                      >
                        {checked && <Check size={16} strokeWidth={2} />}
                      </span>
                    </div>
                    <div className="min-w-0 px-1">
                      <div className="truncate text-sm font-medium text-ink">{filename}</div>
                      <div className={`line-clamp-2 text-sm ${checked ? "text-ink-2" : "text-ink-3"}`}>
                        {reasonText}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="flex min-w-0 items-center gap-2 text-sm text-ink-2" title={t.modalSafetyPromise}>
              <Icon as={Folder} />
              <span className="shrink-0">{t.modalTargetDir}</span>
              <span className="min-w-0 truncate text-ink">{plan.quarantine_dir}</span>
            </div>

            {error && (
              <p className="text-sm text-mark">
                {t.errorPrefix}
                {error}
              </p>
            )}
          </div>

          {/* Footer */}
          <div className="flex shrink-0 flex-wrap items-center justify-between gap-4 px-8 py-6 sm:px-12">
            <div className="flex flex-wrap items-center gap-4 text-sm text-ink-2 tabular-nums">
              <Button variant="ghost" size="sm" disabled={applying} onClick={handleToggleAll}>
                {allSelected ? t.modalDeselectAll : t.modalSelectAll}
              </Button>
              <span>{t.modalSelectedCount(selectedIds.size, items.length)}</span>
              {items.length - selectedIds.size > 0 && selectedIds.size > 0 && (
                <span className="hidden md:inline">
                  {t.modalUncheckedStayHint(items.length - selectedIds.size)}
                </span>
              )}
            </div>

            <div className="ml-auto flex flex-wrap items-center gap-2">
              <Button variant="ghost" disabled={applying} onClick={onClose}>
                {t.modalCancel}
              </Button>
              <Button
                variant="secondary"
                disabled={applying || selectedIds.size === 0}
                onClick={() => void handleReturnToReview()}
              >
                {t.modalReturnToReview(selectedIds.size)}
              </Button>
              <Button
                variant="mark"
                disabled={applying || selectedIds.size === 0}
                onClick={() => void handleConfirm()}
              >
                {applying ? t.modalApplying : t.modalConfirmWithCount(selectedIds.size)}
              </Button>
            </div>
          </div>
        </>
      )}
    </Dialog>
  );
}
