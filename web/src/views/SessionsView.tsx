import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  type DraftObviousResponse,
  type SessionInfo,
  type SessionSummary,
} from "../api";
import { QuarantineModal } from "../components/QuarantineModal";
import { verdictForPhoto, isPhotoFinalized } from "../components/photoMarks";
import {
  Button,
  EmptyState,
  Icon,
  IconButton,
  Loading,
  Notice,
  NoticeDock,
  ProgressLine,
  StatusDot,
  Thumb,
} from "../components/ui";
import { photoImageUrl } from "../api";
import { useI18n } from "../i18n";
import { Folder, Layers, Lock, Play, Refresh, Search, Trash } from "reicon-react";

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

  const previewQ = useQuery({
    queryKey: ["session-photos", session.id, "preview", lang],
    queryFn: () =>
      api.listSessionPhotos(session.id, {
        sort: "name",
        limit: 12,
        lang,
        includeQuarantined: false,
      }),
  });
  const preview = previewQ.data?.items ?? [];

  const statusText = !s
    ? null
    : isAllDone
      ? t.sessionStatusDone
      : isReadyTrash
        ? t.sessionStatusReadyTrash(actionableRed)
        : isOnlyUnsure
          ? t.sessionStatusUnsure(yellowCount)
          : t.sessionStatusInProgress(unreviewed);

  return (
    <article>
      <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-4">
        <div className="min-w-0 flex-1">
          <button
            type="button"
            onClick={() => onOpenReview(session.id)}
            className="block max-w-full truncate text-left text-lg font-semibold text-ink hover:underline hover:decoration-1 hover:underline-offset-4"
          >
            {session.name}
          </button>
          <p className="mt-1 truncate text-sm text-ink-2" title={session.root_path}>
            {session.root_path}
          </p>
        </div>

        <div className="flex min-w-0 flex-wrap items-center gap-2 sm:justify-end">
          <IconButton
            icon={Folder}
            label={t.openShootFolder}
            variant="secondary"
            onClick={() => onOpenFolder(session.id)}
          />
          {s && (reviewed > 0 || yellowCount > 0 || actionableRed > 0) && (
            <Button variant="ghost" icon={Refresh} onClick={() => onResetReview(session)}>
              {t.resetReview}
            </Button>
          )}
          <Button
            variant="secondary"
            icon={Search}
            disabled={screening}
            onClick={() => onAutoScreen(session)}
          >
            {screening ? t.autoScreening : t.autoScreen}
          </Button>
          {actionableRed > 0 && (
            <Button variant="mark" icon={Trash} onClick={() => onQuickQuarantine(session.id)}>
              {t.oneClickWithCount(actionableRed)}
            </Button>
          )}
          <Button variant="primary" onClick={() => onOpenReview(session.id)}>
            {t.openReview}
          </Button>
        </div>
      </div>

      {/* Contact sheet: the first frames of the shoot with the marks already made */}
      <button
        type="button"
        tabIndex={-1}
        aria-hidden="true"
        onClick={() => onOpenReview(session.id)}
        className="mt-6 grid w-full grid-cols-[repeat(auto-fill,minmax(104px,1fr))] grid-rows-[80px] auto-rows-[0px] gap-x-2 gap-y-0 overflow-hidden"
      >
        {previewQ.isLoading
          ? Array.from({ length: 8 }).map((_, i) => (
              <span key={i} className="h-20 rounded-chip bg-table" />
            ))
          : preview.map((item) => (
              <Thumb
                key={item.photo_id}
                src={photoImageUrl(item.photo_id, "grid")}
                verdict={verdictForPhoto(item)}
                veiled={isPhotoFinalized(item)}
                className="h-20"
              />
            ))}
      </button>

      {s && (
        <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm tabular-nums">
          <span className={isReadyTrash ? "font-medium text-reject-ink" : "font-medium text-ink"}>
            {statusText}
          </span>
          {screening && (
            <span className="fade-enter flex items-center gap-2 text-ink-2">
              <ProgressLine indeterminate className="w-16" />
              {t.autoScreening}
            </span>
          )}
          {unreviewed > 0 && reviewTotal > 0 && (
            <span className="flex items-center gap-2 text-ink-2">
              <ProgressLine value={progressPct} className="w-24" />
              <span>{t.sessionProgressRatio(reviewed, reviewTotal)}</span>
            </span>
          )}
          <span className="flex flex-wrap items-center gap-x-4 gap-y-1 text-ink-2">
            {(keptCount > 0 || isAllDone) && (
              <span className="inline-flex items-center gap-1" title={t.sessionKeptCount(keptCount)}>
                <StatusDot verdict="keep" />
                {keptCount}
              </span>
            )}
            {actionableRed > 0 && (
              <span className="inline-flex items-center gap-1" title={t.bucketRed}>
                <StatusDot verdict="reject" />
                {actionableRed}
              </span>
            )}
            {yellowCount > 0 && (
              <span className="inline-flex items-center gap-1" title={t.bucketYellow}>
                <StatusDot verdict="unsure" />
                {yellowCount}
              </span>
            )}
            {exemptCount > 0 && (
              <span className="inline-flex items-center gap-1" title={t.bucketExempt}>
                <Icon as={Lock} />
                {exemptCount}
              </span>
            )}
            {s.burst_count > 0 && (
              <span className="inline-flex items-center gap-1" title={t.bucketBursts}>
                <Icon as={Layers} />
                {s.burst_count}
              </span>
            )}
            {quarantined > 0 && (
              <span className="inline-flex items-center gap-1" title={t.quarantinedBadge}>
                <Icon as={Trash} />
                {quarantined}
              </span>
            )}
          </span>
        </div>
      )}
    </article>
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
  const scanActive = busy || (liveProgress !== null && liveProgress.status !== "completed");
  const scanPct =
    liveProgress?.total && liveProgress.total > 0
      ? Math.min(100, Math.round(((liveProgress.done ?? 0) / liveProgress.total) * 100))
      : undefined;

  return (
    <div className="mx-auto max-w-5xl px-6 pb-32 pt-12 sm:px-12">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-lg font-semibold text-ink">{t.sessionsTitle}</h1>
        <div className="flex items-center gap-2">
          <Button variant="ghost" icon={Play} disabled={busy} onClick={() => void handleRunDemo()}>
            {t.runDemo}
          </Button>
          <Button
            variant="primary"
            icon={Folder}
            disabled={busy}
            onClick={() => void handlePickFolder()}
          >
            {t.pickFolder}
          </Button>
        </div>
      </div>

      <form
        className="intent mt-8 flex items-center gap-2"
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
          aria-label={t.scanPathPlaceholder}
          className="field h-12 min-w-0 flex-1"
        />
        {folderInput.trim() && (
          <Button type="submit" variant="primary" size="lg" disabled={busy} className="fade-enter">
            {t.startScan}
          </Button>
        )}
      </form>

      {scanActive && (
        <div className="fade-enter mt-6 flex items-center gap-4 text-sm tabular-nums">
          <span className="shrink-0 font-medium text-ink">
            {t.scanning}
            {liveProgress?.total ? ` ${liveProgress.done ?? 0} / ${liveProgress.total}` : ""}
          </span>
          <ProgressLine
            value={scanPct}
            indeterminate={scanPct === undefined}
            className="min-w-0 flex-1"
          />
          {liveProgress?.stage && <span className="shrink-0 text-ink-2">{liveProgress.stage}</span>}
          {liveProgress?.jobId && (
            <Button variant="secondary" size="sm" onClick={() => void handlePauseResume()}>
              {liveProgress.status === "paused" ? t.resumeScan : t.pauseScan}
            </Button>
          )}
        </div>
      )}

      <div className="mt-16 flex flex-col gap-16">
        {sessionsQ.isLoading ? (
          <Loading label={t.loading} />
        ) : sessions.length === 0 ? (
          <EmptyState title={t.emptySessions}>
            <Button variant="secondary" icon={Play} disabled={busy} onClick={() => void handleRunDemo()}>
              {t.runDemo}
            </Button>
            <Button
              variant="primary"
              icon={Folder}
              disabled={busy}
              onClick={() => void handlePickFolder()}
            >
              {t.pickFolder}
            </Button>
          </EmptyState>
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

      <NoticeDock>
        {notice && (
          <Notice
            onDismiss={() => setNotice(null)}
            dismissLabel={t.uiDismiss}
            actions={
              lastAppliedPlanId
                ? [
                    {
                      label: t.openTrashFolder,
                      onClick: () => void api.openPlanFolder(lastAppliedPlanId),
                    },
                  ]
                : []
            }
          >
            {notice}
          </Notice>
        )}
        {error && (
          <Notice tone="error" onDismiss={() => setError(null)} dismissLabel={t.uiDismiss}>
            {t.errorPrefix}
            {error}
          </Notice>
        )}
      </NoticeDock>

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
