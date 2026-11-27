import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  type DraftObviousResponse,
  type SessionInfo,
  type SessionSummary,
} from "../api";
import { QuarantineModal } from "../components/QuarantineModal";
import { useI18n } from "../i18n";

interface SessionsViewProps {
  onOpenReview: (sessionId: string) => void;
}

interface LiveProgress {
  jobId?: string;
  stage?: string;
  done?: number;
  total?: number;
  status?: string;
}

function SessionRow({
  session,
  screening,
  onOpenReview,
  onOpenFolder,
  onAutoScreen,
  onResetReview,
  onQuickQuarantine,
}: {
  session: SessionInfo;
  screening: boolean;
  onOpenReview: (id: string) => void;
  onOpenFolder: (id: string) => void;
  onAutoScreen: (session: SessionInfo) => void;
  onResetReview: (session: SessionInfo) => void;
  onQuickQuarantine: (sessionId: string) => void;
}) {
  const { lang, t } = useI18n();
  const summaryQ = useQuery<SessionSummary>({
    queryKey: ["session-summary", session.id, lang],
    queryFn: () => api.getSessionSummary(session.id, lang),
  });

  const s = summaryQ.data;
  const activeCounts = s?.active_counts_by_bucket ?? s?.counts_by_bucket ?? {};
  const yellowCount = activeCounts.yellow ?? 0;
  const exemptCount = activeCounts.exempt ?? 0;
  const quarantined = s?.quarantined_count ?? 0;
  const actionableRed = s?.actionable_reject_count ?? activeCounts.red ?? 0;
  const reviewed = s?.reviewed_count ?? 0;
  const unreviewed = s?.unreviewed_count ?? 0;
  const keptCount = reviewed;
  const reviewTotal = reviewed + yellowCount + unreviewed;
  const isAllDone = s !== undefined && unreviewed === 0 && actionableRed === 0 && yellowCount === 0;
  const isReadyTrash = s !== undefined && unreviewed === 0 && actionableRed > 0;
  const isOnlyUnsure = s !== undefined && unreviewed === 0 && actionableRed === 0 && yellowCount > 0;
  const progressPct =
    reviewTotal > 0 ? Math.min(100, Math.round((reviewed / reviewTotal) * 100)) : 100;

  return (
    <div className="flex flex-col gap-3 border-b border-[#33363B] py-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0 flex-1 space-y-1.5">
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={() => onOpenReview(session.id)}
            className="truncate text-left text-base font-medium text-[#F3F4F6] hover:underline"
          >
            {session.name}
          </button>
          {s && (
            <span
              className={`rounded px-2 py-0.5 text-xs font-medium tabular-nums ${
                isAllDone
                  ? "bg-[#3D9A6B]/20 text-[#86EFAC]"
                  : isReadyTrash
                    ? "bg-[#DC4C46]/20 text-[#FCA5A5]"
                    : isOnlyUnsure
                      ? "bg-[#D99B26]/20 text-[#FCD34D]"
                      : "bg-[#2C2F34] text-[#D1D5DB]"
              }`}
            >
              {isAllDone
                ? t.sessionStatusDone
                : isReadyTrash
                  ? t.sessionStatusReadyTrash(actionableRed)
                  : isOnlyUnsure
                    ? t.sessionStatusUnsure(yellowCount)
                    : t.sessionStatusInProgress(unreviewed)}
            </span>
          )}
        </div>
        <div className="truncate text-xs text-[#9CA3AF]">{session.root_path}</div>

        {s && unreviewed > 0 && reviewTotal > 0 && (
          <div className="flex items-center gap-2.5 pt-0.5">
            <div className="h-1.5 w-36 overflow-hidden rounded-full bg-[#242629]">
              <div
                className="h-full bg-[#3D9A6B] transition-all"
                style={{ width: `${progressPct}%` }}
              />
            </div>
            <span className="text-xs text-[#9CA3AF] tabular-nums">
              {t.sessionProgressRatio(reviewed, reviewTotal)}
            </span>
          </div>
        )}

        {s && (
          <div className="flex flex-wrap items-center gap-4 pt-0.5 text-xs tabular-nums">
            {(keptCount > 0 || isAllDone) && (
              <span className="text-[#3D9A6B]">{t.sessionKeptCount(keptCount)}</span>
            )}
            {exemptCount > 0 && (
              <span className="text-[#93C5FD]">
                {t.bucketExempt} {exemptCount}
              </span>
            )}
            {yellowCount > 0 && (
              <span className="text-[#D99B26]">
                {t.bucketYellow} {yellowCount}
              </span>
            )}
            {actionableRed > 0 && (
              <span className="text-[#DC4C46]">
                {t.bucketRed} {actionableRed}
              </span>
            )}
            {s.burst_count > 0 && (
              <span className="text-[#9CA3AF]">
                {t.bucketBursts} {s.burst_count}
              </span>
            )}
            {quarantined > 0 && (
              <span className="text-[#9CA3AF]">
                {t.quarantinedBadge} {quarantined}
              </span>
            )}
          </div>
        )}
      </div>

      <div className="flex shrink-0 flex-wrap items-center gap-2">
        {actionableRed > 0 && (
          <button
            type="button"
            onClick={() => onQuickQuarantine(session.id)}
            className="rounded-md bg-[#DC4C46] px-3.5 py-2 text-sm font-medium text-white hover:bg-[#C63E38] tabular-nums"
          >
            {t.oneClickWithCount(actionableRed)}
          </button>
        )}
        <button
          type="button"
          disabled={screening}
          onClick={() => onAutoScreen(session)}
          className="rounded-md border border-[#5B8DEF]/50 bg-[#5B8DEF]/15 px-3 py-2 text-sm font-medium text-[#93C5FD] hover:bg-[#5B8DEF]/25 disabled:opacity-50"
        >
          {screening ? t.autoScreening : t.autoScreen}
        </button>
        {s && (reviewed > 0 || yellowCount > 0 || actionableRed > 0) && (
          <button
            type="button"
            onClick={() => onResetReview(session)}
            className="rounded-md border border-[#33363B] bg-[#1B1C1E] px-3 py-2 text-sm text-[#9CA3AF] hover:bg-[#2C2F34] hover:text-[#F3F4F6]"
          >
            {t.resetReview}
          </button>
        )}
        <button
          type="button"
          title={t.openShootFolder}
          onClick={() => onOpenFolder(session.id)}
          className="inline-flex items-center gap-1.5 rounded-md border border-[#33363B] bg-[#242629] px-3 py-2 text-sm text-[#D1D5DB] hover:bg-[#2C2F34] hover:text-[#F3F4F6]"
        >
          <svg
            aria-hidden="true"
            viewBox="0 0 20 20"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            className="h-4 w-4 shrink-0"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M2.5 5.5A1.5 1.5 0 0 1 4 4h3.5l1.75 2H16a1.5 1.5 0 0 1 1.5 1.5v7A1.5 1.5 0 0 1 16 16H4a1.5 1.5 0 0 1-1.5-1.5v-9Z"
            />
          </svg>
          <span>{t.openShootFolder}</span>
        </button>
        <button
          type="button"
          onClick={() => onOpenReview(session.id)}
          className="rounded-md border border-[#33363B] bg-[#242629] px-3.5 py-2 text-sm font-medium text-[#F3F4F6] hover:bg-[#2C2F34]"
        >
          {t.openReview}
        </button>
      </div>
    </div>
  );
}

export function SessionsView({ onOpenReview }: SessionsViewProps) {
  const { lang, t } = useI18n();
  const queryClient = useQueryClient();

  const [folderInput, setFolderInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [screeningSessionId, setScreeningSessionId] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [lastAppliedPlanId, setLastAppliedPlanId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [liveProgress, setLiveProgress] = useState<LiveProgress | null>(null);
  const [activeDraft, setActiveDraft] = useState<DraftObviousResponse | null>(null);
  const [verifyingPlan, setVerifyingPlan] = useState(false);

  const sessionsQ = useQuery({
    queryKey: ["sessions"],
    queryFn: () => api.listSessions(),
  });

  // Subscribe to SSE job.progress updates
  useEffect(() => {
    const es = new EventSource("/api/events");
    const onProgress = (evt: MessageEvent<string>) => {
      try {
        const data = JSON.parse(evt.data) as {
          job_id?: string;
          stage?: string;
          done?: number;
          total?: number;
          status?: string;
        };
        setLiveProgress({
          jobId: data.job_id,
          stage: data.stage,
          done: data.done,
          total: data.total,
          status: data.status,
        });
        if (data.status === "completed") {
          void queryClient.invalidateQueries({ queryKey: ["sessions"] });
          void queryClient.invalidateQueries({ queryKey: ["session-summary"] });
        }
      } catch {
        // ignore malformed SSE event
      }
    };
    es.addEventListener("job.progress", onProgress as EventListener);
    return () => {
      es.close();
    };
  }, [queryClient]);

  async function runScan(path: string) {
    const trimmed = path.trim();
    if (!trimmed) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const res = await api.scanFolder(trimmed, true);
      setFolderInput("");
      await queryClient.invalidateQueries({ queryKey: ["sessions"] });
      await queryClient.invalidateQueries({ queryKey: ["session-summary"] });
      setLiveProgress({
        jobId: res.job_id,
        stage: "done",
        done: res.total_photos,
        total: res.total_photos,
        status: res.status,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function handlePickFolder() {
    setError(null);
    try {
      const res = await api.pickFolder();
      if (res.path) {
        setFolderInput(res.path);
        await runScan(res.path);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleRunDemo() {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const res = await api.runDemoBootstrap();
      await queryClient.invalidateQueries({ queryKey: ["sessions"] });
      await queryClient.invalidateQueries({ queryKey: ["session-summary"] });
      setLiveProgress({
        jobId: res.job_id,
        stage: "done",
        done: res.total_photos,
        total: res.total_photos,
        status: res.status,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleQuickQuarantine(sessionId: string) {
    setError(null);
    setNotice(null);
    setActiveDraft(null);
    setVerifyingPlan(true);
    try {
      const draft = await api.draftObviousRejects(sessionId, "subfolder", lang);
      setActiveDraft(draft);
      if ((draft.verifier_report?.flagged_photos?.length ?? 0) > 0) {
        void queryClient.invalidateQueries({ queryKey: ["session-summary", sessionId] });
        void queryClient.invalidateQueries({ queryKey: ["session-photos", sessionId] });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setVerifyingPlan(false);
    }
  }

  async function handleOpenShootFolder(sessionId: string) {
    setError(null);
    try {
      await api.openSessionFolder(sessionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleAutoScreen(session: SessionInfo) {
    setScreeningSessionId(session.id);
    setError(null);
    setNotice(null);
    setLastAppliedPlanId(null);
    try {
      const res = await api.autoScreenSession(session.id);
      await queryClient.invalidateQueries({ queryKey: ["session-summary", session.id] });
      await queryClient.invalidateQueries({ queryKey: ["session-photos", session.id] });
      setNotice(
        t.autoScreenResult(session.name, res.flagged_red, res.flagged_yellow, res.total_active),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setScreeningSessionId(null);
    }
  }

  async function handleResetReview(session: SessionInfo) {
    if (!window.confirm(t.resetReviewConfirm)) return;
    setError(null);
    setNotice(null);
    setLastAppliedPlanId(null);
    try {
      await api.resetSessionReview(session.id);
      await queryClient.invalidateQueries({ queryKey: ["session-summary", session.id] });
      await queryClient.invalidateQueries({ queryKey: ["session-photos", session.id] });
      setNotice(t.resetReviewDone(session.name));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handlePauseResume() {
    if (!liveProgress?.jobId) return;
    try {
      if (liveProgress.status === "paused") {
        const res = await api.resumeJob(liveProgress.jobId);
        setLiveProgress({
          jobId: res.job_id,
          stage: "done",
          done: res.total_photos,
          total: res.total_photos,
          status: res.status,
        });
      } else {
        const res = await api.pauseJob(liveProgress.jobId);
        setLiveProgress((prev) => (prev ? { ...prev, status: res.status } : null));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  const sessions = sessionsQ.data?.items ?? [];

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <div className="space-y-1">
        <h1 className="text-2xl font-medium tracking-tight text-[#F3F4F6]">{t.sessionsTitle}</h1>
        <p className="text-sm text-[#9CA3AF]">{t.sessionsSubtitle}</p>
      </div>

      {/* Primary actions */}
      <div className="mt-6 flex flex-wrap items-center gap-3">
        <button
          type="button"
          disabled={busy}
          onClick={() => void handlePickFolder()}
          className="rounded-md bg-[#F3F4F6] px-4 py-2.5 text-sm font-medium text-[#1B1C1E] hover:bg-white disabled:opacity-50"
        >
          {t.pickFolder}
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={() => void handleRunDemo()}
          className="rounded-md border border-[#33363B] bg-[#242629] px-4 py-2.5 text-sm text-[#F3F4F6] hover:bg-[#2C2F34] disabled:opacity-50"
        >
          {t.runDemo}
        </button>
      </div>

      {/* Direct path input fallback */}
      <form
        className="mt-4 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          void runScan(folderInput);
        }}
      >
        <input
          type="text"
          value={folderInput}
          onChange={(e) => setFolderInput(e.target.value)}
          placeholder={t.scanPathPlaceholder}
          className="flex-1 rounded-md border border-[#33363B] bg-[#242629] px-3.5 py-2 text-sm text-[#F3F4F6] placeholder-[#6B7280]"
        />
        <button
          type="submit"
          disabled={busy || !folderInput.trim()}
          className="rounded-md border border-[#33363B] bg-[#242629] px-4 py-2 text-sm text-[#F3F4F6] hover:bg-[#2C2F34] disabled:opacity-40"
        >
          {t.startScan}
        </button>
      </form>

      {/* Live SSE scan progress */}
      {(busy || (liveProgress && liveProgress.status !== "completed")) && (
        <div className="mt-4 flex items-center justify-between gap-4 rounded-md border border-[#33363B] bg-[#242629] px-4 py-3 text-sm">
          <div className="space-y-1">
            <div className="text-[#F3F4F6]">
              {t.scanning}
              {liveProgress?.stage ? ` (${liveProgress.stage})` : ""}
              {liveProgress?.total
                ? ` · ${liveProgress.done ?? 0} / ${liveProgress.total}`
                : ""}
            </div>
            {liveProgress?.total ? (
              <div className="h-1.5 w-64 overflow-hidden rounded-full bg-[#1B1C1E]">
                <div
                  className="h-full bg-[#5B8DEF]"
                  style={{
                    width: `${Math.min(100, Math.round(((liveProgress.done ?? 0) / liveProgress.total) * 100))}%`,
                  }}
                />
              </div>
            ) : null}
          </div>
          {liveProgress?.jobId && (
            <button
              type="button"
              onClick={() => void handlePauseResume()}
              className="rounded border border-[#33363B] px-3 py-1.5 text-xs text-[#F3F4F6] hover:bg-[#2C2F34]"
            >
              {liveProgress.status === "paused" ? t.resumeScan : t.pauseScan}
            </button>
          )}
        </div>
      )}

      {notice && (
        <div className="mt-4 flex flex-wrap items-center justify-between gap-2 rounded-md border border-[#3D9A6B]/50 bg-[#3D9A6B]/10 px-4 py-2.5 text-sm text-[#A7F3D0]">
          <span>{notice}</span>
          {lastAppliedPlanId && (
            <button
              type="button"
              onClick={() => void api.openPlanFolder(lastAppliedPlanId)}
              className="rounded border border-[#A7F3D0]/40 px-2.5 py-1 text-xs text-[#ECFDF5] hover:bg-[#3D9A6B]/30"
            >
              {t.openTrashFolder}
            </button>
          )}
        </div>
      )}

      {error && (
        <div className="mt-4 rounded-md border border-[#DC4C46]/50 bg-[#DC4C46]/10 px-4 py-2.5 text-sm text-[#FCA5A5]">
          {t.errorPrefix}
          {error}
        </div>
      )}

      {/* Session list */}
      <div className="mt-8 border-t border-[#33363B]">
        {sessionsQ.isLoading ? (
          <p className="py-10 text-sm text-[#9CA3AF]">{t.loading}</p>
        ) : sessions.length === 0 ? (
          <div className="py-12 text-center">
            <p className="text-base text-[#F3F4F6]">{t.emptySessions}</p>
            <p className="mt-1 text-sm text-[#9CA3AF]">{t.emptySessionsHint}</p>
          </div>
        ) : (
          sessions.map((sess) => (
            <SessionRow
              key={sess.id}
              session={sess}
              screening={screeningSessionId === sess.id}
              onOpenReview={onOpenReview}
              onOpenFolder={(id) => void handleOpenShootFolder(id)}
              onAutoScreen={(s) => void handleAutoScreen(s)}
              onResetReview={(s) => void handleResetReview(s)}
              onQuickQuarantine={(id) => void handleQuickQuarantine(id)}
            />
          ))
        )}
      </div>

      {(verifyingPlan || activeDraft) && (
        <QuarantineModal
          draft={activeDraft}
          verifying={verifyingPlan}
          onClose={() => {
            setVerifyingPlan(false);
            setActiveDraft(null);
          }}
          onApplied={(moved, planId) => {
            setActiveDraft(null);
            setLastAppliedPlanId(planId);
            setNotice(t.modalSuccess(moved));
            void queryClient.invalidateQueries({ queryKey: ["sessions"] });
            void queryClient.invalidateQueries({ queryKey: ["session-summary"] });
          }}
          onReturned={(returned) => {
            setActiveDraft(null);
            setLastAppliedPlanId(null);
            setNotice(t.modalReturnSuccess(returned));
            void queryClient.invalidateQueries({ queryKey: ["sessions"] });
            void queryClient.invalidateQueries({ queryKey: ["session-summary"] });
            void queryClient.invalidateQueries({ queryKey: ["session-photos"] });
          }}
        />
      )}
    </div>
  );
}
