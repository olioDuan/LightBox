import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  api,
  photoImageUrl,
  type Bucket,
  type DraftObviousResponse,
  type PhotoItem,
} from "../api";
import { QuarantineModal } from "../components/QuarantineModal";
import { useI18n, type Messages } from "../i18n";

type FilterTab =
  | "all"
  | "revisit"
  | "unreviewed"
  | "yellow"
  | "red"
  | "kept"
  | "exempt"
  | "bursts";

type LastAction =
  | { kind: "verdict"; photoId: string }
  | { kind: "skip"; photoId: string }
  | null;

interface ReviewViewProps {
  sessionId: string;
  onBack: () => void;
  onGoInsights: () => void;
}

function formatSeq(seqNo: number, total: number): string {
  const width = total >= 100 ? 3 : 2;
  return `#${String(seqNo).padStart(width, "0")}`;
}

interface SlideVisual {
  borderClass: string;
  label: string | null;
  labelClass: string;
  isFinalized: boolean;
}

/**
 * Finalized photos get a subtle gray veil in the filmstrip (`保留`, `待删除`, `已保护`).
 * Undecided photos (`存疑`, `跳过`, unreviewed) remain full-color so they stand out for comparison & revisit.
 */
function isPhotoFinalized(item: PhotoItem): boolean {
  if (item.bucket === "red") return true;
  if (item.bucket === "exempt" || item.is_protected) return true;
  if (
    item.bucket === "normal" &&
    item.source === "user" &&
    !item.reason_codes.includes("user_reset")
  ) {
    return true;
  }
  return false;
}

function getSlideVisual(
  item: PhotoItem,
  isSkipped: boolean,
  t: Messages,
): SlideVisual {
  if (item.bucket === "red") {
    return {
      borderClass: "border-[#DC4C46]/80",
      label: t.stripReject,
      labelClass: "text-[#FCA5A5]",
      isFinalized: true,
    };
  }
  if (item.bucket === "yellow") {
    return {
      borderClass: "border-[#D99B26]",
      label: t.stripUnsure,
      labelClass: "text-[#FCD34D]",
      isFinalized: false,
    };
  }
  if (item.bucket === "exempt" || item.is_protected) {
    return {
      borderClass: "border-[#5B8DEF]/80",
      label: t.stripProtected,
      labelClass: "text-[#93C5FD]",
      isFinalized: true,
    };
  }
  if (isPhotoFinalized(item)) {
    return {
      borderClass: "border-[#3D9A6B]/80",
      label: t.stripKept,
      labelClass: "text-[#6EE7B7]",
      isFinalized: true,
    };
  }
  if (isSkipped) {
    return {
      borderClass: "border-dashed border-[#93C5FD]/85",
      label: t.stripSkipped,
      labelClass: "text-[#93C5FD]",
      isFinalized: false,
    };
  }
  return {
    borderClass: "border-[#33363B]",
    label: null,
    labelClass: "text-[#6B7280]",
    isFinalized: false,
  };
}

export function ReviewView({ sessionId, onBack, onGoInsights }: ReviewViewProps) {
  const { lang, t } = useI18n();
  const queryClient = useQueryClient();

  // Default to full shoot sequence (`all`): previously rejected photos are grouped on the far left
  // when entering `all`, while newly marked photos stay in place for instant ← comparison.
  const [tab, setTab] = useState<FilterTab>("all");
  const [allEntryToken, setAllEntryToken] = useState(1);
  const [leftRedSnapshot, setLeftRedSnapshot] = useState<{ token: number; ids: string[] } | null>(
    null,
  );
  const [revisitIds, setRevisitIds] = useState<string[]>([]);
  const [showCompleteModal, setShowCompleteModal] = useState(false);
  const [skippedIds, setSkippedIds] = useState<string[]>([]);
  const [lastAction, setLastAction] = useState<LastAction>(null);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [rescueBannerPhotoId, setRescueBannerPhotoId] = useState<string | null>(null);
  const [lastAppliedPlanId, setLastAppliedPlanId] = useState<string | null>(null);
  const [burstIdx, setBurstIdx] = useState(0);
  const [selectedBurstPickId, setSelectedBurstPickId] = useState<string | null>(null);
  const [burstZoomed, setBurstZoomed] = useState(false);
  const [activeDraft, setActiveDraft] = useState<DraftObviousResponse | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const positionedTokenRef = useRef<number>(0);

  // Reset per-session state when switching sessions
  useEffect(() => {
    setTab("all");
    setLeftRedSnapshot(null);
    setRevisitIds([]);
    setShowCompleteModal(false);
    setSkippedIds([]);
    setLastAction(null);
    setSelectedIdx(0);
    setToast(null);
    positionedTokenRef.current = 0;
    setAllEntryToken((tok) => tok + 1);
  }, [sessionId]);

  const summaryQ = useQuery({
    queryKey: ["session-summary", sessionId, lang],
    queryFn: () => api.getSessionSummary(sessionId, lang),
  });

  // Always keep full active photo list fresh (used by `all`, `revisit`, and the completion modal stats)
  const allPhotosQ = useQuery({
    queryKey: ["session-photos", sessionId, "all", lang],
    queryFn: () =>
      api.listSessionPhotos(sessionId, {
        bucket: undefined,
        sort: "name",
        limit: 2000,
        lang,
        includeQuarantined: false,
      }),
  });

  const bucketPhotosQ = useQuery({
    queryKey: ["session-photos", sessionId, tab, lang],
    queryFn: () =>
      api.listSessionPhotos(sessionId, {
        bucket: tab,
        sort: "name",
        limit: 2000,
        lang,
        includeQuarantined: false,
      }),
    enabled: tab !== "all" && tab !== "revisit" && tab !== "bursts",
  });

  const burstsQ = useQuery({
    queryKey: ["session-bursts", sessionId],
    queryFn: () => api.listSessionBursts(sessionId),
  });

  const allActivePhotos = allPhotosQ.data?.items ?? [];
  const rawBucketPhotos = bucketPhotosQ.data?.items ?? [];
  const photosLoading =
    tab === "all" || tab === "revisit" ? allPhotosQ.isLoading : bucketPhotosQ.isLoading;
  const skippedSet = useMemo(() => new Set(skippedIds), [skippedIds]);

  // Live session-wide breakdown (always sums cleanly to allActivePhotos.length)
  const sessionStats = useMemo(() => {
    let kept = 0;
    let reject = 0;
    let unsure = 0;
    let skippedOnly = 0;
    let unseen = 0;
    const undecided: PhotoItem[] = [];
    for (const p of allActivePhotos) {
      if (p.bucket === "red") {
        reject += 1;
      } else if (p.bucket === "yellow") {
        unsure += 1;
        undecided.push(p);
      } else if (isPhotoFinalized(p)) {
        kept += 1;
      } else {
        undecided.push(p);
        if (skippedSet.has(p.photo_id)) {
          skippedOnly += 1;
        } else {
          unseen += 1;
        }
      }
    }
    return {
      total: allActivePhotos.length,
      kept,
      reject,
      unsure,
      skippedOnly,
      unseen,
      pendingCount: skippedOnly + unseen,
      undecided,
    };
  }, [allActivePhotos, skippedSet]);

  // Partition `all` so photos already rejected before entering `all` sit on the far left,
  // while photos rejected during this active pass stay in place for immediate ← comparison.
  const { orderedAllPhotos, leftRedCount } = useMemo(() => {
    const effectiveLeftRedSet =
      leftRedSnapshot?.token === allEntryToken
        ? new Set(leftRedSnapshot.ids)
        : new Set(allActivePhotos.filter((p) => p.bucket === "red").map((p) => p.photo_id));
    if (effectiveLeftRedSet.size === 0) {
      return { orderedAllPhotos: allActivePhotos, leftRedCount: 0 };
    }
    const leftReds: PhotoItem[] = [];
    const mainSeq: PhotoItem[] = [];
    for (const item of allActivePhotos) {
      if (effectiveLeftRedSet.has(item.photo_id)) {
        leftReds.push(item);
      } else {
        mainSeq.push(item);
      }
    }
    return { orderedAllPhotos: [...leftReds, ...mainSeq], leftRedCount: leftReds.length };
  }, [allActivePhotos, allEntryToken, leftRedSnapshot]);

  // Freeze the left-rejected snapshot once fresh `all` data has settled, and place the cursor
  // at the first undecided photo right after the pre-rejected left partition.
  useEffect(() => {
    if (tab !== "all" || allPhotosQ.isLoading || allPhotosQ.isFetching || !allPhotosQ.data) return;
    if (positionedTokenRef.current === allEntryToken) return;
    positionedTokenRef.current = allEntryToken;

    const preRedIds = allActivePhotos.filter((p) => p.bucket === "red").map((p) => p.photo_id);
    setLeftRedSnapshot({ token: allEntryToken, ids: preRedIds });

    const preRedSet = new Set(preRedIds);
    const mainSeq = allActivePhotos.filter((p) => !preRedSet.has(p.photo_id));
    const firstUndecidedInMain = mainSeq.findIndex((p) => !isPhotoFinalized(p));
    if (firstUndecidedInMain >= 0) {
      setSelectedIdx(preRedIds.length + firstUndecidedInMain);
    } else if (mainSeq.length > 0) {
      setSelectedIdx(preRedIds.length);
    } else {
      setSelectedIdx(0);
    }
  }, [allActivePhotos, allEntryToken, allPhotosQ.data, allPhotosQ.isFetching, allPhotosQ.isLoading, tab]);

  const photos = useMemo(() => {
    if (tab === "all") {
      return orderedAllPhotos;
    }
    if (tab === "revisit") {
      const byId = new Map<string, PhotoItem>();
      for (const item of allActivePhotos) {
        byId.set(item.photo_id, item);
      }
      const list: PhotoItem[] = [];
      for (const id of revisitIds) {
        const found = byId.get(id);
        if (found) list.push(found);
      }
      return list;
    }
    if (tab === "unreviewed" && skippedIds.length > 0) {
      const normalQueue: PhotoItem[] = [];
      const byId = new Map<string, PhotoItem>();
      for (const item of rawBucketPhotos) {
        if (skippedSet.has(item.photo_id)) {
          byId.set(item.photo_id, item);
        } else {
          normalQueue.push(item);
        }
      }
      const tail: PhotoItem[] = [];
      for (const id of skippedIds) {
        const found = byId.get(id);
        if (found) tail.push(found);
      }
      return [...normalQueue, ...tail];
    }
    return rawBucketPhotos;
  }, [allActivePhotos, orderedAllPhotos, rawBucketPhotos, revisitIds, skippedIds, skippedSet, tab]);

  const bursts = burstsQ.data?.items ?? [];
  const summary = summaryQ.data;
  const totalForSeq = summary?.total_photos ?? photos.length;

  const clampedIdx = photos.length > 0 ? Math.min(selectedIdx, photos.length - 1) : 0;
  const activePhoto: PhotoItem | undefined = photos[clampedIdx];

  const clampedBurstIdx = bursts.length > 0 ? Math.min(burstIdx, bursts.length - 1) : 0;
  const activeBurst = bursts[clampedBurstIdx];

  useEffect(() => {
    if (!activeBurst) {
      setSelectedBurstPickId(null);
      return;
    }
    const existingPick = activeBurst.members.find((m) => m.is_pick) ?? activeBurst.members[0];
    setSelectedBurstPickId(existingPick?.photo_id ?? null);
  }, [activeBurst?.id]);

  // Preload ±5 surrounding loupe images for instant (<100ms) ←/→ scrubbing
  useEffect(() => {
    if (tab === "bursts" || photos.length === 0) return;
    const start = Math.max(0, clampedIdx - 5);
    const end = Math.min(photos.length - 1, clampedIdx + 5);
    for (let i = start; i <= end; i += 1) {
      const img = new Image();
      img.src = photoImageUrl(photos[i].photo_id, "loupe");
    }
  }, [clampedIdx, photos, tab]);

  // Virtualized horizontal filmstrip (slide-mount cards with fixed #01 number + status below image)
  const stripParentRef = useRef<HTMLDivElement | null>(null);
  const filmstripVirtualizer = useVirtualizer({
    horizontal: true,
    count: photos.length,
    getScrollElement: () => stripParentRef.current,
    estimateSize: () => 124,
    overscan: 8,
  });

  // Allow mouse wheel (vertical deltaY) to scroll the bottom filmstrip horizontally
  useEffect(() => {
    const el = stripParentRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        e.preventDefault();
        el.scrollLeft += e.deltaY;
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [tab]);

  useEffect(() => {
    if (tab !== "bursts" && photos.length > 0) {
      filmstripVirtualizer.scrollToIndex(clampedIdx, { align: "auto" });
    }
  }, [clampedIdx, filmstripVirtualizer, photos.length, tab]);

  const refreshSession = useCallback(async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["session-summary", sessionId] }),
      queryClient.invalidateQueries({ queryKey: ["session-photos", sessionId] }),
      queryClient.invalidateQueries({ queryKey: ["session-bursts", sessionId] }),
    ]);
  }, [queryClient, sessionId]);

  const enterAllTab = useCallback(() => {
    setShowCompleteModal(false);
    setRescueBannerPhotoId(null);
    setToast(null);
    setTab("all");
    setAllEntryToken((tok) => tok + 1);
  }, []);

  const enterRevisitPass = useCallback(() => {
    const ids = sessionStats.undecided.map((p) => p.photo_id);
    if (ids.length === 0) {
      enterAllTab();
      return;
    }
    setShowCompleteModal(false);
    setRescueBannerPhotoId(null);
    setToast(null);
    setRevisitIds(ids);
    setTab("revisit");
    setSelectedIdx(0);
  }, [enterAllTab, sessionStats.undecided]);

  const applyVerdict = useCallback(
    async (photo: PhotoItem, bucket: Bucket, reasonTag?: string) => {
      if (photo.is_protected && bucket === "red") return;
      setError(null);
      const wasRed = photo.bucket === "red";
      const willDisappear =
        tab === "unreviewed" ||
        (tab === "kept" && bucket !== "normal" && bucket !== "exempt") ||
        (tab === "yellow" && bucket !== "yellow") ||
        (tab === "red" && bucket !== "red");
      const wasLast = clampedIdx >= photos.length - 1;

      try {
        await api.setVerdict([photo.photo_id], bucket, reasonTag);

        // Once explicitly tagged (keep / unsure / reject), clear skipped flag and record Undo
        setSkippedIds((prev) => prev.filter((id) => id !== photo.photo_id));
        setLastAction({ kind: "verdict", photoId: photo.photo_id });
        setLastAppliedPlanId(null);

        if (wasRed && bucket === "normal" && !reasonTag) {
          setRescueBannerPhotoId(photo.photo_id);
        } else {
          setRescueBannerPhotoId(null);
        }

        if (bucket === "red" && (!wasLast || willDisappear)) {
          const nextRejectCount =
            (summary?.actionable_reject_count ?? 0) + (photo.bucket === "red" ? 0 : 1);
          setToast(t.rejectedToast(nextRejectCount));
        } else {
          setToast(null);
        }

        await refreshSession();

        // Linear non-looping advance: finishing the last photo opens the Smart Completion Modal
        if (!reasonTag) {
          if (willDisappear) {
            setSelectedIdx(Math.max(0, Math.min(clampedIdx, photos.length - 2)));
          } else if (wasLast) {
            setToast(null);
            setShowCompleteModal(true);
          } else {
            setSelectedIdx(Math.min(photos.length - 1, clampedIdx + 1));
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [clampedIdx, photos.length, refreshSession, summary, t, tab],
  );

  const handleSkip = useCallback(
    (photo: PhotoItem) => {
      setError(null);
      setRescueBannerPhotoId(null);
      setLastAppliedPlanId(null);

      const seqLabel = formatSeq(photo.seq_no, totalForSeq);
      const wasLast = clampedIdx >= photos.length - 1;

      // Mark undecided photos as skipped so they stay bright in place and appear in the Revisit pass
      if (!isPhotoFinalized(photo) && photo.bucket !== "yellow") {
        setSkippedIds((prev) => [...prev.filter((id) => id !== photo.photo_id), photo.photo_id]);
        setLastAction({ kind: "skip", photoId: photo.photo_id });
      }

      if (wasLast) {
        setToast(null);
        setShowCompleteModal(true);
        return;
      }

      if (tab === "unreviewed") {
        setToast(t.skippedToast(seqLabel));
      } else {
        setToast(null);
        setSelectedIdx(Math.min(photos.length - 1, clampedIdx + 1));
      }
    },
    [clampedIdx, photos.length, t, tab, totalForSeq],
  );

  const handleUndo = useCallback(
    async (fallbackPhotoId?: string) => {
      setError(null);
      setShowCompleteModal(false);
      if (lastAction?.kind === "skip") {
        const targetIdx = photos.findIndex((p) => p.photo_id === lastAction.photoId);
        setSkippedIds((prev) => prev.filter((id) => id !== lastAction.photoId));
        if (targetIdx >= 0) setSelectedIdx(targetIdx);
        setLastAction(null);
        setToast(null);
        return;
      }
      const targetId = lastAction?.kind === "verdict" ? lastAction.photoId : fallbackPhotoId;
      if (!targetId) return;
      try {
        await api.undoVerdict(targetId);
        const targetIdx = photos.findIndex((p) => p.photo_id === targetId);
        if (targetIdx >= 0) setSelectedIdx(targetIdx);
        setRescueBannerPhotoId(null);
        setLastAction(null);
        setToast(null);
        await refreshSession();
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [lastAction, photos, refreshSession],
  );

  const handleConfirmBurst = useCallback(async () => {
    if (!activeBurst || !selectedBurstPickId) return;
    setError(null);
    try {
      await api.confirmBurst(activeBurst.id, selectedBurstPickId);
      await refreshSession();
      if (clampedBurstIdx < bursts.length - 1) {
        setBurstIdx(clampedBurstIdx + 1);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [activeBurst, bursts.length, clampedBurstIdx, refreshSession, selectedBurstPickId]);

  const handleOpenQuarantine = useCallback(async () => {
    setError(null);
    setToast(null);
    setShowCompleteModal(false);
    try {
      const draft = await api.draftObviousRejects(sessionId, "subfolder", lang);
      setActiveDraft(draft);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [lang, sessionId]);

  const handleResetReview = useCallback(async () => {
    if (!window.confirm(t.resetReviewConfirm)) return;
    setError(null);
    setToast(null);
    setShowCompleteModal(false);
    try {
      await api.resetSessionReview(sessionId);
      setSkippedIds([]);
      setRevisitIds([]);
      setLeftRedSnapshot(null);
      setLastAction(null);
      setTab("all");
      setSelectedIdx(0);
      positionedTokenRef.current = 0;
      setAllEntryToken((tok) => tok + 1);
      await refreshSession();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [refreshSession, sessionId, t.resetReviewConfirm]);

  // Global keyboard shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (activeDraft) return;
      if (showCompleteModal) {
        if (e.key === "Escape") {
          e.preventDefault();
          setShowCompleteModal(false);
        }
        return;
      }
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;

      if (tab === "bursts") {
        if (!activeBurst) return;
        const visibleMembers = activeBurst.members.slice(0, 4);
        if (e.key >= "1" && e.key <= "4") {
          const idx = Number(e.key) - 1;
          if (visibleMembers[idx]) {
            e.preventDefault();
            setSelectedBurstPickId(visibleMembers[idx].photo_id);
          }
          return;
        }
        if (e.key === "ArrowRight") {
          e.preventDefault();
          void handleConfirmBurst();
          return;
        }
        if (e.key === "ArrowLeft") {
          e.preventDefault();
          setBurstIdx((prev) => Math.max(0, prev - 1));
          return;
        }
        if (e.key === " " || e.key.toLowerCase() === "z") {
          e.preventDefault();
          setBurstZoomed((z) => !z);
        }
        return;
      }

      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
        if (lastAction || activePhoto?.photo_id) {
          e.preventDefault();
          void handleUndo(activePhoto?.photo_id);
        }
        return;
      }

      if (!activePhoto || e.metaKey || e.ctrlKey || e.altKey) return;

      switch (e.key) {
        case "ArrowLeft":
          e.preventDefault();
          setSelectedIdx((i) => Math.max(0, i - 1));
          setRescueBannerPhotoId(null);
          break;
        case "ArrowRight":
          e.preventDefault();
          setSelectedIdx((i) => Math.min(photos.length - 1, i + 1));
          setRescueBannerPhotoId(null);
          break;
        case "p":
        case "P":
          e.preventDefault();
          void applyVerdict(activePhoto, "normal");
          break;
        case "u":
        case "U":
          e.preventDefault();
          void applyVerdict(activePhoto, "yellow");
          break;
        case "s":
        case "S":
          e.preventDefault();
          handleSkip(activePhoto);
          break;
        case "x":
        case "X":
          e.preventDefault();
          void applyVerdict(activePhoto, "red");
          break;
        default:
          break;
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    activeBurst,
    activeDraft,
    activePhoto,
    applyVerdict,
    handleConfirmBurst,
    handleSkip,
    handleUndo,
    lastAction,
    photos.length,
    showCompleteModal,
    tab,
  ]);

  const activeCounts = summary?.active_counts_by_bucket ?? summary?.counts_by_bucket ?? {};
  const actionableRed = summary?.actionable_reject_count ?? activeCounts.red ?? 0;
  const yellowCount = activeCounts.yellow ?? 0;
  const exemptCount = activeCounts.exempt ?? 0;
  const keptCount = summary?.reviewed_count ?? 0;
  const unreviewedCount = summary?.unreviewed_count ?? 0;
  const burstCount = summary?.burst_count ?? bursts.length;
  const quarantinedCount = summary?.quarantined_count ?? 0;

  const tabs: Array<{ id: FilterTab; label: string; count?: number }> = useMemo(() => {
    const list: Array<{ id: FilterTab; label: string; count?: number }> = [
      { id: "all", label: t.bucketAllPhotos, count: summary?.active_total_photos ?? 0 },
    ];
    if (tab === "revisit") {
      list.push({ id: "revisit", label: t.bucketRevisit, count: photos.length });
    }
    list.push(
      { id: "unreviewed", label: t.bucketAll, count: unreviewedCount },
      { id: "yellow", label: t.bucketYellow, count: yellowCount },
      { id: "red", label: t.bucketRed, count: activeCounts.red ?? 0 },
      { id: "kept", label: t.bucketNormal, count: keptCount },
    );
    if (exemptCount > 0 || tab === "exempt") {
      list.push({ id: "exempt", label: t.bucketExempt, count: exemptCount });
    }
    if (burstCount > 0 || tab === "bursts") {
      list.push({ id: "bursts", label: t.bucketBursts, count: burstCount });
    }
    return list;
  }, [
    activeCounts.red,
    burstCount,
    exemptCount,
    keptCount,
    photos.length,
    summary?.active_total_photos,
    t,
    tab,
    unreviewedCount,
    yellowCount,
  ]);

  return (
    <div className="flex h-[calc(100vh-52px)] flex-col bg-[#1B1C1E]">
      {/* Top workspace toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#33363B] bg-[#242629] px-5 py-2.5">
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={onBack}
            className="text-sm text-[#9CA3AF] hover:text-[#F3F4F6]"
          >
            {t.backToSessions}
          </button>
          <span className="text-[#4B5563]">/</span>
          <span className="text-sm font-medium text-[#F3F4F6]">
            {summary?.session.name ?? sessionId}
          </span>

          <div className="ml-1 flex flex-wrap items-center gap-1">
            {tabs.map((item) => {
              const active = tab === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => {
                    if (item.id === "all") {
                      enterAllTab();
                    } else {
                      setShowCompleteModal(false);
                      setTab(item.id);
                      setSelectedIdx(0);
                      setRescueBannerPhotoId(null);
                    }
                  }}
                  className={`rounded-md px-2.5 py-1 text-xs tabular-nums ${
                    active
                      ? "bg-[#F3F4F6] font-medium text-[#1B1C1E]"
                      : "text-[#D1D5DB] hover:bg-[#2C2F34]"
                  }`}
                >
                  {item.label}
                  {item.count !== undefined ? ` ${item.count}` : ""}
                </button>
              );
            })}
          </div>
        </div>

        <button
          type="button"
          disabled={actionableRed === 0}
          onClick={() => void handleOpenQuarantine()}
          className={`shrink-0 rounded-md px-3.5 py-1.5 text-sm font-medium tabular-nums ${
            actionableRed > 0
              ? "bg-[#DC4C46] text-white hover:bg-[#C63E38]"
              : "cursor-not-allowed border border-[#33363B] bg-[#1B1C1E] text-[#6B7280]"
          }`}
        >
          {t.oneClickWithCount(actionableRed)}
        </button>
      </div>

      {/* Always-visible quiet review progress strip */}
      {summary && tab !== "bursts" && (
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#33363B] bg-[#1F2124] px-5 py-1.5 text-xs text-[#9CA3AF] tabular-nums">
          <div className="flex flex-wrap items-center gap-4">
            {tab === "revisit" ? (
              <span className="text-[#FCD34D]">{t.revisitBanner(photos.length)}</span>
            ) : unreviewedCount === 0 ? (
              <span
                className={
                  actionableRed > 0
                    ? "text-[#FCA5A5]"
                    : yellowCount > 0
                      ? "text-[#FCD34D]"
                      : "text-[#86EFAC]"
                }
              >
                {actionableRed > 0
                  ? t.progressDoneWithTrash(actionableRed)
                  : yellowCount > 0
                    ? t.progressDoneWithUnsure(yellowCount)
                    : t.progressAllDone}
              </span>
            ) : (
              <span>
                {t.progressSummary(
                  summary.reviewed_count,
                  unreviewedCount,
                  actionableRed,
                  yellowCount,
                )}
              </span>
            )}
            {tab === "all" && leftRedCount > 0 && (
              <button
                type="button"
                onClick={() => {
                  setSelectedIdx(0);
                  setRescueBannerPhotoId(null);
                }}
                className="text-[#FCA5A5]/90 hover:underline"
              >
                {t.leftRedGroupedHint(leftRedCount)}
              </button>
            )}
            {quarantinedCount > 0 && (
              <button
                type="button"
                onClick={onGoInsights}
                className="text-[#93C5FD] hover:underline"
              >
                {t.alreadyQuarantinedLink(quarantinedCount)}
              </button>
            )}
          </div>
          {tab !== "revisit" &&
            sessionStats.undecided.length > 0 &&
            (sessionStats.unsure > 0 ||
              sessionStats.skippedOnly > 0 ||
              sessionStats.kept > 0 ||
              sessionStats.reject > 0) && (
              <button
                type="button"
                onClick={enterRevisitPass}
                className="rounded border border-[#D99B26]/50 bg-[#D99B26]/10 px-2.5 py-0.5 text-[#FCD34D] hover:bg-[#D99B26]/20"
              >
                {t.progressRevisitBtn(sessionStats.undecided.length)}
              </button>
            )}
        </div>
      )}

      {/* Action feedback toast with instant 1-click Undo or Open Trash Folder */}
      {toast && (
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#3D9A6B]/40 bg-[#3D9A6B]/15 px-5 py-2 text-xs text-[#A7F3D0]">
          <span>{toast}</span>
          <div className="flex items-center gap-2">
            {lastAction && (
              <button
                type="button"
                onClick={() => void handleUndo()}
                className="rounded border border-[#A7F3D0]/40 px-2 py-0.5 text-[#ECFDF5] hover:bg-[#3D9A6B]/30"
              >
                {t.actionUndo} (⌘Z)
              </button>
            )}
            {lastAppliedPlanId && (
              <>
                <button
                  type="button"
                  onClick={() => void api.openPlanFolder(lastAppliedPlanId)}
                  className="rounded border border-[#A7F3D0]/40 px-2 py-0.5 text-[#ECFDF5] hover:bg-[#3D9A6B]/30"
                >
                  {t.openTrashFolder}
                </button>
                <button
                  type="button"
                  onClick={onGoInsights}
                  className="rounded border border-[#A7F3D0]/40 px-2 py-0.5 text-[#ECFDF5] hover:bg-[#3D9A6B]/30"
                >
                  {t.goTrashBin}
                </button>
              </>
            )}
          </div>
        </div>
      )}
      {error && (
        <div className="border-b border-[#DC4C46]/40 bg-[#DC4C46]/15 px-5 py-2 text-xs text-[#FCA5A5]">
          {t.errorPrefix}
          {error}
        </div>
      )}

      {/* Main Content */}
      {tab === "bursts" ? (
        <div className="flex flex-1 flex-col overflow-hidden">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#33363B] bg-[#202530] px-5 py-2.5 text-xs text-[#BFDBFE]">
            <span>{t.burstBanner}</span>
            {activeBurst && (
              <div className="flex items-center gap-2">
                <span className="tabular-nums text-[#F3F4F6]">
                  {t.burstGroupLabel(clampedBurstIdx + 1, bursts.length)}
                </span>
                <span className="rounded bg-[#2C2F34] px-2 py-0.5 text-[#D1D5DB]">
                  {activeBurst.confirmed ? t.burstConfirmed : t.burstUnconfirmed}
                </span>
              </div>
            )}
          </div>

          {bursts.length === 0 ? (
            <div className="flex flex-1 items-center justify-center text-sm text-[#9CA3AF]">
              {t.burstNoGroups}
            </div>
          ) : (
            <div className="flex flex-1 flex-col overflow-hidden p-5">
              <div className="grid flex-1 grid-cols-2 gap-4 overflow-hidden lg:grid-cols-4">
                {activeBurst.members.slice(0, 4).map((member, i) => {
                  const chosen = member.photo_id === selectedBurstPickId;
                  return (
                    <button
                      key={member.photo_id}
                      type="button"
                      onClick={() => setSelectedBurstPickId(member.photo_id)}
                      className={`flex flex-col overflow-hidden rounded-md border bg-[#242629] text-left ${
                        chosen ? "border-[#3D9A6B] ring-2 ring-[#3D9A6B]/50" : "border-[#33363B]"
                      }`}
                    >
                      <div className="flex items-center justify-between border-b border-[#33363B] px-3 py-1.5 text-xs">
                        <span className="font-medium text-[#F3F4F6] tabular-nums">#{i + 1}</span>
                        <div className="flex items-center gap-1.5">
                          {member.rank === 1 && (
                            <span className="rounded bg-[#3D9A6B]/20 px-1.5 py-0.5 text-[#86EFAC]">
                              {t.burstRecommended}
                            </span>
                          )}
                          {chosen && (
                            <span className="rounded bg-[#3D9A6B] px-1.5 py-0.5 text-white">
                              {t.burstSelected}
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex min-h-[220px] flex-1 items-center justify-center overflow-hidden bg-[#161719] p-2">
                        <img
                          src={photoImageUrl(
                            member.photo_id,
                            burstZoomed ? "crop" : "loupe",
                            burstZoomed ? "sharpest" : undefined,
                          )}
                          alt=""
                          className="max-h-full max-w-full object-contain"
                        />
                      </div>
                    </button>
                  );
                })}
              </div>

              <div className="mt-4 flex items-center justify-between">
                <button
                  type="button"
                  onClick={() => setBurstZoomed((z) => !z)}
                  className="rounded-md border border-[#33363B] bg-[#242629] px-3.5 py-2 text-xs text-[#F3F4F6] hover:bg-[#2C2F34]"
                >
                  {t.burstSyncZoom}
                </button>
                <button
                  type="button"
                  onClick={() => void handleConfirmBurst()}
                  className="rounded-md bg-[#3D9A6B] px-4 py-2 text-sm font-medium text-white hover:bg-[#32825A]"
                >
                  {t.burstConfirmNext}
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="flex flex-1 flex-col overflow-hidden">
          {/* Center Loupe Stage */}
          <div className="relative flex flex-1 items-center justify-center overflow-hidden bg-[#161719] p-4">
            {photosLoading ? (
              <span className="text-sm text-[#9CA3AF]">{t.loading}</span>
            ) : !activePhoto ? (
              <div className="max-w-md space-y-4 rounded-lg border border-[#33363B] bg-[#242629] p-6 text-center">
                <h2 className="text-lg font-medium text-[#F3F4F6]">
                  {unreviewedCount === 0 ? t.reviewCompleteTitle : t.noPhotosInBucket}
                </h2>
                <p className="text-sm text-[#9CA3AF] tabular-nums">
                  {unreviewedCount > 0
                    ? t.reviewRemainingDesc(unreviewedCount, actionableRed)
                    : actionableRed > 0 || yellowCount > 0
                      ? t.reviewCompleteDesc(keptCount, yellowCount, actionableRed)
                      : t.reviewCompleteClean(keptCount)}
                </p>
                <div className="flex flex-wrap items-center justify-center gap-2.5 pt-1">
                  {tab !== "all" && (summary?.active_total_photos ?? 0) > 0 && (
                    <button
                      type="button"
                      onClick={enterAllTab}
                      className="rounded-md bg-[#F3F4F6] px-4 py-2 text-sm font-medium text-[#1B1C1E] hover:bg-white"
                    >
                      {t.reviewViewAll}
                    </button>
                  )}
                  {actionableRed > 0 && (
                    <button
                      type="button"
                      onClick={() => void handleOpenQuarantine()}
                      className="rounded-md bg-[#DC4C46] px-4 py-2 text-sm font-medium text-white hover:bg-[#C63E38] tabular-nums"
                    >
                      {t.reviewCompleteConfirm(actionableRed)}
                    </button>
                  )}
                  {yellowCount > 0 && tab !== "yellow" && (
                    <button
                      type="button"
                      onClick={() => {
                        setTab("yellow");
                        setSelectedIdx(0);
                      }}
                      className="rounded-md border border-[#D99B26]/60 bg-[#D99B26]/15 px-4 py-2 text-sm font-medium text-[#FCD34D] hover:bg-[#D99B26]/25 tabular-nums"
                    >
                      {t.reviewCompleteUnsure(yellowCount)}
                    </button>
                  )}
                  {tab === "unreviewed" && (summary?.active_total_photos ?? 0) > 0 && (
                    <button
                      type="button"
                      onClick={() => void handleResetReview()}
                      className="rounded-md border border-[#33363B] px-4 py-2 text-sm text-[#D1D5DB] hover:bg-[#2C2F34]"
                    >
                      {t.resetReview}
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={onBack}
                    className="rounded-md border border-[#33363B] px-4 py-2 text-sm text-[#D1D5DB] hover:bg-[#2C2F34]"
                  >
                    {t.backToSessions}
                  </button>
                </div>
              </div>
            ) : (
              <>
                {photos.length > 1 && (
                  <button
                    type="button"
                    aria-label="Previous photo"
                    disabled={clampedIdx <= 0}
                    onClick={() => {
                      setSelectedIdx((i) => Math.max(0, i - 1));
                      setRescueBannerPhotoId(null);
                    }}
                    className="absolute left-4 top-1/2 z-10 flex h-11 w-11 -translate-y-1/2 items-center justify-center rounded-full border border-white/15 bg-black/50 text-[#F3F4F6] transition hover:bg-black/75 disabled:cursor-not-allowed disabled:opacity-25"
                  >
                    <svg
                      aria-hidden="true"
                      viewBox="0 0 20 20"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      className="h-5 w-5"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12.5 4.5 7 10l5.5 5.5" />
                    </svg>
                  </button>
                )}

                <img
                  key={activePhoto.photo_id}
                  src={photoImageUrl(activePhoto.photo_id, "loupe")}
                  alt={activePhoto.stem}
                  className="max-h-full max-w-full select-none object-contain"
                />

                {photos.length > 1 && (
                  <button
                    type="button"
                    aria-label="Next photo"
                    disabled={clampedIdx >= photos.length - 1}
                    onClick={() => {
                      setSelectedIdx((i) => Math.min(photos.length - 1, i + 1));
                      setRescueBannerPhotoId(null);
                    }}
                    className="absolute right-4 top-1/2 z-10 flex h-11 w-11 -translate-y-1/2 items-center justify-center rounded-full border border-white/15 bg-black/50 text-[#F3F4F6] transition hover:bg-black/75 disabled:cursor-not-allowed disabled:opacity-25"
                  >
                    <svg
                      aria-hidden="true"
                      viewBox="0 0 20 20"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      className="h-5 w-5"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 4.5 13 10l-5.5 5.5" />
                    </svg>
                  </button>
                )}
              </>
            )}
          </div>

          {/* Compact Reason + 5-Action Bar */}
          {activePhoto && (
            <div className="border-t border-[#33363B] bg-[#242629] px-5 py-2.5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex min-w-0 flex-wrap items-center gap-2.5 text-xs">
                  <span className="rounded bg-[#1B1C1E] px-2 py-0.5 font-mono font-medium text-[#F3F4F6] tabular-nums">
                    {formatSeq(activePhoto.seq_no, totalForSeq)}
                  </span>
                  <span className="font-medium text-[#F3F4F6]">{activePhoto.stem}</span>
                  <span className="tabular-nums text-[#9CA3AF]">
                    ({clampedIdx + 1} / {photos.length})
                  </span>
                  {(() => {
                    const visual = getSlideVisual(
                      activePhoto,
                      skippedSet.has(activePhoto.photo_id),
                      t,
                    );
                    return visual.label ? (
                      <span
                        className={`rounded px-2 py-0.5 font-medium ${
                          activePhoto.bucket === "red"
                            ? "border border-[#DC4C46]/60 bg-[#DC4C46]/20 text-[#FCA5A5]"
                            : `bg-[#1B1C1E] ${visual.labelClass}`
                        }`}
                      >
                        {visual.label}
                      </span>
                    ) : null;
                  })()}
                  {activePhoto.reason_labels.length > 0 && (
                    <span className="text-[#9CA3AF]">
                      {activePhoto.reason_labels.join("；")}
                    </span>
                  )}
                </div>

                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <button
                    type="button"
                    onClick={() => void applyVerdict(activePhoto, "normal")}
                    className="rounded border border-[#3D9A6B]/60 bg-[#3D9A6B]/15 px-3 py-1.5 font-medium text-[#86EFAC] hover:bg-[#3D9A6B]/25"
                  >
                    {t.actionKeep} (P)
                  </button>
                  <button
                    type="button"
                    onClick={() => void applyVerdict(activePhoto, "yellow")}
                    className="rounded border border-[#D99B26]/60 bg-[#D99B26]/15 px-3 py-1.5 font-medium text-[#FCD34D] hover:bg-[#D99B26]/25"
                  >
                    {t.actionUnsure} (U)
                  </button>
                  <button
                    type="button"
                    onClick={() => handleSkip(activePhoto)}
                    className="rounded border border-[#33363B] bg-[#1B1C1E] px-3 py-1.5 text-[#D1D5DB] hover:bg-[#2C2F34]"
                  >
                    {t.actionSkip} (S)
                  </button>
                  <button
                    type="button"
                    disabled={activePhoto.is_protected}
                    onClick={() => void applyVerdict(activePhoto, "red")}
                    className="rounded border border-[#DC4C46]/60 bg-[#DC4C46]/15 px-3 py-1.5 font-medium text-[#FCA5A5] hover:bg-[#DC4C46]/25 disabled:opacity-40"
                  >
                    {t.actionReject} (X)
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleUndo(activePhoto.photo_id)}
                    className="rounded border border-[#33363B] bg-[#1B1C1E] px-2.5 py-1.5 text-[#9CA3AF] hover:bg-[#2C2F34] hover:text-[#F3F4F6]"
                  >
                    {t.actionUndo} (⌘Z)
                  </button>
                </div>
              </div>

              {/* Non-blocking quick rescue reason strip when rescuing a red photo */}
              {rescueBannerPhotoId === activePhoto.photo_id && (
                <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-[#33363B] pt-2 text-xs">
                  <span className="text-[#9CA3AF]">{t.rescueReasonPrompt}</span>
                  {t.rescueReasons.map((r) => (
                    <button
                      key={r.tag}
                      type="button"
                      onClick={() => void applyVerdict(activePhoto, "normal", r.tag)}
                      className="rounded border border-[#33363B] bg-[#1B1C1E] px-2.5 py-1 text-[#D1D5DB] hover:bg-[#2C2F34]"
                    >
                      {r.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Virtualized Bottom Filmstrip — Slide-Mount Cards + Mouse-Wheel Horizontal Scroll */}
          <div
            ref={stripParentRef}
            className="h-32 shrink-0 overflow-x-auto border-t border-[#33363B] bg-[#1B1C1E] px-3 py-2"
          >
            <div
              className="relative h-full"
              style={{ width: `${filmstripVirtualizer.getTotalSize()}px` }}
            >
              {filmstripVirtualizer.getVirtualItems().map((vItem) => {
                const item = photos[vItem.index];
                if (!item) return null;
                const isSelected = vItem.index === clampedIdx;
                const visual = getSlideVisual(item, skippedSet.has(item.photo_id), t);
                const isPartitionBoundary =
                  tab === "all" && leftRedCount > 0 && vItem.index === leftRedCount;
                return (
                  <button
                    key={item.photo_id}
                    type="button"
                    onClick={() => {
                      setSelectedIdx(vItem.index);
                      setRescueBannerPhotoId(null);
                    }}
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: `${vItem.size - 8}px`,
                      height: "100%",
                      transform: `translateX(${vItem.start}px)`,
                    }}
                    className={`group flex flex-col overflow-hidden rounded border bg-[#242629] ${
                      visual.borderClass
                    } ${isSelected ? "ring-2 ring-[#F3F4F6]" : ""} ${
                      isPartitionBoundary ? "shadow-[-5px_0_0_0_rgba(220,76,70,0.45)]" : ""
                    }`}
                  >
                    <div className="relative min-h-0 flex-1 w-full overflow-hidden bg-[#141517]">
                      <img
                        src={photoImageUrl(item.photo_id, "grid")}
                        alt=""
                        loading="lazy"
                        className={`h-full w-full object-cover transition ${
                          visual.isFinalized
                            ? isSelected
                              ? "grayscale-[45%] brightness-90"
                              : "grayscale opacity-60 group-hover:grayscale-[30%] group-hover:opacity-90"
                            : ""
                        }`}
                      />
                      {visual.isFinalized && (
                        <div
                          className={`pointer-events-none absolute inset-0 transition ${
                            isSelected
                              ? "bg-[#6B7280]/20"
                              : "bg-[#6B7280]/40 group-hover:bg-[#6B7280]/15"
                          }`}
                        />
                      )}
                    </div>
                    <div className="flex h-5 w-full shrink-0 items-center justify-between border-t border-[#2E3136] bg-[#18191B] px-1.5 text-[10px] leading-none tabular-nums">
                      <span className="font-mono text-[#9CA3AF]">
                        {formatSeq(item.seq_no, totalForSeq)}
                      </span>
                      {visual.label && (
                        <span className={`font-medium ${visual.labelClass}`}>
                          {visual.label}
                        </span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Smart Round-Completion Modal (locks background controls against misclicks) */}
      {showCompleteModal && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-40 flex items-center justify-center bg-black/75 p-4 backdrop-blur-[2px]"
        >
          <div className="w-full max-w-lg rounded-xl border border-[#33363B] bg-[#242629] p-6 shadow-2xl">
            <div className="space-y-1.5">
              <h2 className="text-lg font-semibold text-[#F3F4F6]">
                {sessionStats.undecided.length === 0
                  ? t.completeModalAllDoneTitle
                  : t.completeModalReachedEndTitle}
              </h2>
              <p className="text-sm text-[#9CA3AF]">
                {sessionStats.undecided.length === 0
                  ? t.completeModalAllDoneSubtitle(sessionStats.total)
                  : t.completeModalPendingSubtitle(sessionStats.undecided.length)}
              </p>
            </div>

            <div className="mt-5 grid grid-cols-2 gap-2.5 sm:grid-cols-4">
              <div className="rounded-lg border border-[#3D9A6B]/35 bg-[#3D9A6B]/10 px-3 py-2.5 text-center">
                <div className="text-xl font-semibold text-[#86EFAC] tabular-nums">
                  {sessionStats.kept}
                </div>
                <div className="mt-0.5 text-xs text-[#9CA3AF]">{t.completeStatKept}</div>
              </div>
              <div className="rounded-lg border border-[#DC4C46]/35 bg-[#DC4C46]/10 px-3 py-2.5 text-center">
                <div className="text-xl font-semibold text-[#FCA5A5] tabular-nums">
                  {sessionStats.reject}
                </div>
                <div className="mt-0.5 text-xs text-[#9CA3AF]">{t.completeStatReject}</div>
              </div>
              <div className="rounded-lg border border-[#D99B26]/35 bg-[#D99B26]/10 px-3 py-2.5 text-center">
                <div className="text-xl font-semibold text-[#FCD34D] tabular-nums">
                  {sessionStats.unsure}
                </div>
                <div className="mt-0.5 text-xs text-[#9CA3AF]">{t.completeStatUnsure}</div>
              </div>
              <div className="rounded-lg border border-[#5B8DEF]/35 bg-[#5B8DEF]/10 px-3 py-2.5 text-center">
                <div className="text-xl font-semibold text-[#93C5FD] tabular-nums">
                  {sessionStats.pendingCount}
                </div>
                <div className="mt-0.5 text-xs text-[#9CA3AF]">{t.completeStatPending}</div>
              </div>
            </div>

            <div className="mt-5 flex flex-wrap items-center justify-end gap-2.5">
              <button
                type="button"
                onClick={() => setShowCompleteModal(false)}
                className="rounded-md border border-[#33363B] bg-[#1B1C1E] px-4 py-2 text-sm text-[#D1D5DB] hover:bg-[#2C2F34]"
              >
                {t.completeActionClose}
              </button>
              {tab === "revisit" && (
                <button
                  type="button"
                  onClick={enterAllTab}
                  className="rounded-md border border-[#33363B] bg-[#1B1C1E] px-4 py-2 text-sm text-[#D1D5DB] hover:bg-[#2C2F34]"
                >
                  {t.reviewViewAll}
                </button>
              )}
              {sessionStats.reject > 0 && (
                <button
                  type="button"
                  onClick={() => void handleOpenQuarantine()}
                  className={`rounded-md px-4 py-2 text-sm font-medium tabular-nums ${
                    sessionStats.undecided.length === 0
                      ? "bg-[#DC4C46] text-white hover:bg-[#C63E38]"
                      : "border border-[#DC4C46]/60 bg-[#DC4C46]/15 text-[#FCA5A5] hover:bg-[#DC4C46]/25"
                  }`}
                >
                  {t.completeActionQuarantine(sessionStats.reject)}
                </button>
              )}
              {sessionStats.undecided.length > 0 && (
                <button
                  type="button"
                  onClick={enterRevisitPass}
                  className="rounded-md bg-[#D99B26] px-4 py-2 text-sm font-medium text-[#1B1C1E] hover:bg-[#E5A935] tabular-nums"
                >
                  {t.completeActionRevisit(sessionStats.undecided.length)}
                </button>
              )}
            </div>

            <p className="mt-4 border-t border-[#33363B] pt-3 text-xs text-[#9CA3AF]">
              {t.completeModalHint}
            </p>
          </div>
        </div>
      )}

      {activeDraft && (
        <QuarantineModal
          draft={activeDraft}
          onClose={() => setActiveDraft(null)}
          onApplied={(moved, planId) => {
            setActiveDraft(null);
            setLastAction(null);
            setLastAppliedPlanId(planId);
            setToast(t.modalSuccess(moved));
            positionedTokenRef.current = 0;
            setAllEntryToken((tok) => tok + 1);
            void refreshSession();
          }}
          onReturned={(returned) => {
            setActiveDraft(null);
            setLastAction(null);
            setLastAppliedPlanId(null);
            setToast(t.modalReturnSuccess(returned));
            positionedTokenRef.current = 0;
            setAllEntryToken((tok) => tok + 1);
            void refreshSession();
          }}
        />
      )}
    </div>
  );
}
