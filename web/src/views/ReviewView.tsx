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
import { verdictForPhoto } from "../components/photoMarks";
import {
  Button,
  Dialog,
  EmptyState,
  Icon,
  IconButton,
  Kbd,
  Loading,
  Notice,
  NoticeDock,
  ProgressLine,
  Tabs,
  StatusDot,
  Thumb,
  VERDICT_STYLE,
  useJustMarked,
  type ReiconGlyph,
  type Verdict,
} from "../components/ui";
import { useI18n, type Messages } from "../i18n";
import {
  ArrowLeft,
  Check,
  ChevronLeft,
  ChevronRight,
  HelpCircle,
  Lock,
  Maximize,
  SkipNext,
  Sparkles,
  Trash,
  Undo,
  X,
} from "reicon-react";

/** Verdict action: tinted at rest, filled when hovered or when it is the photo's current verdict. */
function VerdictButton({
  verdict,
  icon: Glyph,
  label,
  hotkey,
  current,
  disabled,
  onClick,
}: {
  verdict: Verdict;
  icon: ReiconGlyph;
  label: string;
  hotkey: string;
  current: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  const v = VERDICT_STYLE[verdict];
  const filled = current && verdict !== "skip";
  return (
    <button
      type="button"
      disabled={disabled}
      aria-pressed={current}
      aria-keyshortcuts={hotkey}
      onClick={onClick}
      data-verdict={verdict}
      className={`verdict-btn group inline-flex h-12 min-w-[112px] items-center justify-center gap-2 rounded-btn px-4 text-sm font-semibold transition-[background-color,color,border-color] duration-150 disabled:pointer-events-none disabled:opacity-40 ${
        verdict === "skip" ? "border-[1.5px] border-dashed" : ""
      }`}
      style={
        {
          "--v": v.color,
          "--v-soft": v.soft,
          "--v-ink": v.ink,
          background: filled ? v.color : verdict === "skip" ? "transparent" : v.soft,
          color: filled ? (verdict === "unsure" ? "var(--color-ink)" : "#fff") : v.ink,
          borderColor: verdict === "skip" ? (current ? "var(--color-ink-2)" : v.color) : undefined,
        } as React.CSSProperties
      }
    >
      <Glyph size={18} strokeWidth={verdict === "skip" ? 1.5 : 2} />
      <span>{label}</span>
      <kbd className="ml-1 font-sans text-xs font-medium opacity-60">{hotkey}</kbd>
    </button>
  );
}

/** i18n strip labels carry their own glyph prefix (✓ / ? / ✕); the UI draws marks instead. */
function plainLabel(label: string | null): string {
  return (label ?? "").replace(/^[✓?✕×]\s*/, "");
}

function plainLink(label: string): string {
  return label.replace(/\s*→\s*$/, "");
}

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
  const [verifyingPlan, setVerifyingPlan] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const positionedTokenRef = useRef<number>(0);
  const [justMarkedId, setJustMarkedId] = useJustMarked();

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
    estimateSize: () => 120,
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
        setJustMarkedId(photo.photo_id);
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
    [clampedIdx, photos.length, refreshSession, setJustMarkedId, summary, t, tab],
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
    setActiveDraft(null);
    setVerifyingPlan(true);
    try {
      const draft = await api.draftObviousRejects(sessionId, "subfolder", lang);
      setActiveDraft(draft);
      if ((draft.verifier_report?.flagged_photos?.length ?? 0) > 0) {
        void refreshSession();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setVerifyingPlan(false);
    }
  }, [lang, refreshSession, sessionId]);

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
      if (activeDraft || verifyingPlan) return;
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

  const progressPct =
    summary && summary.active_total_photos > 0
      ? Math.round(((summary.active_total_photos - unreviewedCount) / summary.active_total_photos) * 100)
      : 0;
  const activeVisual = activePhoto
    ? getSlideVisual(activePhoto, skippedSet.has(activePhoto.photo_id), t)
    : null;
  const activeVerdict = activePhoto
    ? verdictForPhoto(activePhoto, skippedSet.has(activePhoto.photo_id))
    : null;

  return (
    <div className="flex h-[calc(100vh-64px)] flex-col bg-paper">
      {/* Row 1 — back arrow sits under the logo mark, the session name under the wordmark */}
      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2 px-4 sm:px-8">
        <div className="flex min-w-0 flex-1 flex-wrap items-center gap-x-6 gap-y-2">
          <div className="flex min-w-0 items-center">
            <IconButton
              icon={ArrowLeft}
              label={t.backToSessions}
              onClick={onBack}
              className="-ml-3"
            />
            <h1 className="-ml-1 max-w-[20rem] truncate text-lg font-semibold text-ink">
              {summary?.session.name ?? sessionId}
            </h1>
          </div>
          {summary && tab !== "bursts" && (
            <div className="flex min-w-0 items-center gap-4 text-sm tabular-nums">
              <ProgressLine value={progressPct} className="w-24 shrink-0" />
              {tab === "revisit" ? (
                <span className="truncate font-medium text-ink">{t.revisitBanner(photos.length)}</span>
              ) : unreviewedCount === 0 ? (
                <span
                  className={`truncate font-medium ${
                    actionableRed > 0
                      ? "text-reject-ink"
                      : yellowCount > 0
                        ? "text-unsure-ink"
                        : "text-keep-ink"
                  }`}
                >
                  {actionableRed > 0
                    ? t.progressDoneWithTrash(actionableRed)
                    : yellowCount > 0
                      ? t.progressDoneWithUnsure(yellowCount)
                      : t.progressAllDone}
                </span>
              ) : (
                <span className="truncate text-ink-2">
                  {t.progressSummary(summary.reviewed_count, unreviewedCount, actionableRed, yellowCount)}
                </span>
              )}
            </div>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-2">
          {tab !== "revisit" &&
            tab !== "bursts" &&
            sessionStats.undecided.length > 0 &&
            (sessionStats.unsure > 0 ||
              sessionStats.skippedOnly > 0 ||
              sessionStats.kept > 0 ||
              sessionStats.reject > 0) && (
              <Button variant="secondary" onClick={enterRevisitPass}>
                {t.progressRevisitBtn(sessionStats.undecided.length)}
              </Button>
            )}
          {actionableRed > 0 && (
            <Button variant="mark" icon={Trash} onClick={() => void handleOpenQuarantine()}>
              {t.oneClickWithCount(actionableRed)}
            </Button>
          )}
        </div>
      </div>

      {/* Row 2 — filters as one segmented control, each verdict carrying its color */}
      <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2 px-4 pb-4 pt-2 sm:px-8">
        <Tabs
          variant="segmented"
          value={tab}
          items={tabs.map((item) => ({
            id: item.id,
            label: item.label,
            count: item.count,
            attention: item.id === "red" && (item.count ?? 0) > 0,
            verdict:
              item.id === "yellow"
                ? "unsure"
                : item.id === "red"
                  ? "reject"
                  : item.id === "kept"
                    ? "keep"
                    : item.id === "exempt"
                      ? "protected"
                      : undefined,
          }))}
          onChange={(id) => {
            if (id === "all") {
              enterAllTab();
            } else {
              setShowCompleteModal(false);
              setTab(id);
              setSelectedIdx(0);
              setRescueBannerPhotoId(null);
            }
          }}
        />
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
          {tab === "all" && leftRedCount > 0 && (
            <button
              type="button"
              onClick={() => {
                setSelectedIdx(0);
                setRescueBannerPhotoId(null);
              }}
              className="inline-flex items-center gap-1 text-ink-2 hover:text-ink"
            >
              <StatusDot verdict="reject" className="-ml-2" />
              {t.leftRedGroupedHint(leftRedCount)}
            </button>
          )}
          {quarantinedCount > 0 && (
            <button
              type="button"
              onClick={onGoInsights}
              className="inline-flex items-center gap-1 text-ink-2 hover:text-ink"
            >
              <Icon as={Trash} />
              {plainLink(t.alreadyQuarantinedLink(quarantinedCount))}
            </button>
          )}
        </div>
      </div>

      {/* Main Content */}
      {tab === "bursts" ? (
        <div className="flex min-h-0 flex-1 flex-col bg-table px-4 py-6 sm:px-8">
          {activeBurst && (
            <div className="flex items-center gap-4 pb-6 text-sm tabular-nums">
              <span className="font-semibold text-ink">
                {t.burstGroupLabel(clampedBurstIdx + 1, bursts.length)}
              </span>
              <span className={activeBurst.confirmed ? "text-ink-2" : "text-ink-3"}>
                {activeBurst.confirmed ? t.burstConfirmed : t.burstUnconfirmed}
              </span>
            </div>
          )}

          {bursts.length === 0 ? (
            <EmptyState title={t.burstNoGroups} className="flex-1" />
          ) : (
            <>
              <div className="grid min-h-0 flex-1 grid-cols-2 content-center gap-4 overflow-y-auto lg:grid-cols-4">
                {activeBurst.members.slice(0, 4).map((member, i) => {
                  const chosen = member.photo_id === selectedBurstPickId;
                  return (
                    <button
                      key={member.photo_id}
                      type="button"
                      aria-pressed={chosen}
                      onClick={() => setSelectedBurstPickId(member.photo_id)}
                      className="group flex min-h-0 flex-col gap-2 text-left"
                    >
                      <div
                        key={chosen ? `pick-${member.photo_id}` : member.photo_id}
                        className={`relative aspect-[4/3] w-full overflow-hidden rounded-tile border-2 bg-paper bg-clip-padding transition-[border-color] duration-200 ${
                          chosen ? "verdict-pop border-keep" : "border-transparent group-hover:border-line"
                        }`}
                      >
                        <img
                          src={photoImageUrl(
                            member.photo_id,
                            burstZoomed ? "crop" : "loupe",
                            burstZoomed ? "sharpest" : undefined,
                          )}
                          alt=""
                          className="h-full w-full object-contain"
                        />
                        {chosen && (
                          <span className="absolute right-2 top-2 flex h-6 w-6 items-center justify-center rounded-full bg-keep text-white">
                            <Check size={16} strokeWidth={2} />
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <Kbd className={chosen ? "bg-keep text-white" : "bg-paper"}>{i + 1}</Kbd>
                        {member.rank === 1 && (
                          <span className="inline-flex items-center gap-1 text-ink-2">
                            <Icon as={Sparkles} />
                            {t.burstRecommended}
                          </span>
                        )}
                        {chosen && <span className="ml-auto font-medium text-keep-ink">{t.burstSelected}</span>}
                      </div>
                    </button>
                  );
                })}
              </div>

              <div className="flex items-center justify-between gap-4 pt-6">
                <Button
                  variant={burstZoomed ? "primary" : "secondary"}
                  icon={Maximize}
                  aria-pressed={burstZoomed}
                  onClick={() => setBurstZoomed((z) => !z)}
                >
                  {t.burstSyncZoom}
                </Button>
                <Button variant="primary" onClick={() => void handleConfirmBurst()}>
                  {t.burstConfirmNext}
                </Button>
              </div>
            </>
          )}
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col">
          {/* Loupe stage */}
          <div className="intent relative flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-table p-6 sm:px-20">
            {photosLoading ? (
              <Loading label={t.loading} />
            ) : !activePhoto ? (
              <div className="flex max-w-md flex-col items-center gap-4 text-center">
                <h2 className="text-lg font-semibold text-ink">
                  {unreviewedCount === 0 ? t.reviewCompleteTitle : t.noPhotosInBucket}
                </h2>
                <p className="text-sm text-ink-2 tabular-nums">
                  {unreviewedCount > 0
                    ? t.reviewRemainingDesc(unreviewedCount, actionableRed)
                    : actionableRed > 0 || yellowCount > 0
                      ? t.reviewCompleteDesc(keptCount, yellowCount, actionableRed)
                      : t.reviewCompleteClean(keptCount)}
                </p>
                <div className="flex flex-wrap items-center justify-center gap-2 pt-4">
                  {tab !== "all" && (summary?.active_total_photos ?? 0) > 0 && (
                    <Button variant="primary" onClick={enterAllTab}>
                      {t.reviewViewAll}
                    </Button>
                  )}
                  {actionableRed > 0 && (
                    <Button variant="mark" icon={Trash} onClick={() => void handleOpenQuarantine()}>
                      {t.reviewCompleteConfirm(actionableRed)}
                    </Button>
                  )}
                  {yellowCount > 0 && tab !== "yellow" && (
                    <Button
                      variant="secondary"
                      onClick={() => {
                        setTab("yellow");
                        setSelectedIdx(0);
                      }}
                    >
                      {t.reviewCompleteUnsure(yellowCount)}
                    </Button>
                  )}
                  {tab === "unreviewed" && (summary?.active_total_photos ?? 0) > 0 && (
                    <Button variant="ghost" onClick={() => void handleResetReview()}>
                      {t.resetReview}
                    </Button>
                  )}
                  <Button variant="ghost" onClick={onBack}>
                    {t.backToSessions}
                  </Button>
                </div>
              </div>
            ) : (
              <>
                {photos.length > 1 && (
                  <button
                    type="button"
                    aria-label={t.uiPrevPhoto}
                    title={t.uiPrevPhoto}
                    disabled={clampedIdx <= 0}
                    onClick={() => {
                      setSelectedIdx((i) => Math.max(0, i - 1));
                      setRescueBannerPhotoId(null);
                    }}
                    className="reveal absolute left-4 top-1/2 z-10 flex h-12 w-12 -translate-y-1/2 items-center justify-center rounded-full bg-paper text-ink transition hover:bg-table-2 disabled:invisible sm:left-6"
                  >
                    <Icon as={ChevronLeft} />
                  </button>
                )}

                <img
                  key={activePhoto.photo_id}
                  src={photoImageUrl(activePhoto.photo_id, "loupe")}
                  alt={activePhoto.stem}
                  className="loupe-enter max-h-full max-w-full select-none rounded-[4px] object-contain"
                />

                {photos.length > 1 && (
                  <button
                    type="button"
                    aria-label={t.uiNextPhoto}
                    title={t.uiNextPhoto}
                    disabled={clampedIdx >= photos.length - 1}
                    onClick={() => {
                      setSelectedIdx((i) => Math.min(photos.length - 1, i + 1));
                      setRescueBannerPhotoId(null);
                    }}
                    className="reveal absolute right-4 top-1/2 z-10 flex h-12 w-12 -translate-y-1/2 items-center justify-center rounded-full bg-paper text-ink transition hover:bg-table-2 disabled:invisible sm:right-6"
                  >
                    <Icon as={ChevronRight} />
                  </button>
                )}
              </>
            )}
          </div>

          {/* Caption + verdict bar */}
          {activePhoto && (
            <div className="bg-paper px-4 py-4 sm:px-8">
              <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-4">
                <div className="flex min-w-0 flex-1 items-center gap-4 text-sm tabular-nums">
                  <span className="text-lg font-semibold text-ink">
                    {formatSeq(activePhoto.seq_no, totalForSeq)}
                  </span>
                  <div className="min-w-0">
                    <div className="flex min-w-0 items-center gap-2">
                      <span className="truncate font-medium text-ink">{activePhoto.stem}</span>
                      <span className="shrink-0 text-ink-3">
                        {clampedIdx + 1} / {photos.length}
                      </span>
                    </div>
                    <div className="flex min-w-0 items-center gap-2 text-ink-2">
                      {activeVisual?.label && activeVerdict && (
                        <span
                          className="inline-flex shrink-0 items-center gap-1 font-medium"
                          style={{ color: VERDICT_STYLE[activeVerdict].ink }}
                        >
                          {activeVerdict === "protected" ? (
                            <Icon as={Lock} className="-ml-1" />
                          ) : (
                            <StatusDot verdict={activeVerdict} className="-ml-2" />
                          )}
                          {plainLabel(activeVisual.label)}
                        </span>
                      )}
                      {activePhoto.reason_labels.length > 0 && (
                        <span className="truncate" title={activePhoto.reason_labels.join("；")}>
                          {activePhoto.reason_labels.join("；")}
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <VerdictButton
                    verdict="keep"
                    icon={Check}
                    label={t.actionKeep}
                    hotkey="P"
                    current={activeVerdict === "keep"}
                    onClick={() => void applyVerdict(activePhoto, "normal")}
                  />
                  <VerdictButton
                    verdict="unsure"
                    icon={HelpCircle}
                    label={t.actionUnsure}
                    hotkey="U"
                    current={activeVerdict === "unsure"}
                    onClick={() => void applyVerdict(activePhoto, "yellow")}
                  />
                  <VerdictButton
                    verdict="skip"
                    icon={SkipNext}
                    label={t.actionSkip}
                    hotkey="S"
                    current={activeVerdict === "skip"}
                    onClick={() => handleSkip(activePhoto)}
                  />
                  <VerdictButton
                    verdict="reject"
                    icon={X}
                    label={t.actionReject}
                    hotkey="X"
                    current={activeVerdict === "reject"}
                    disabled={activePhoto.is_protected}
                    onClick={() => void applyVerdict(activePhoto, "red")}
                  />
                  <IconButton
                    icon={Undo}
                    label={`${t.actionUndo} ⌘Z`}
                    onClick={() => void handleUndo(activePhoto.photo_id)}
                  />
                </div>
              </div>

              {/* Optional rescue reason after pulling a photo back from the reject pile */}
              {rescueBannerPhotoId === activePhoto.photo_id && (
                <div className="fade-enter mt-4 flex flex-wrap items-center gap-2 text-sm">
                  <span className="mr-2 text-ink-2">{t.rescueReasonPrompt}</span>
                  {t.rescueReasons.map((r) => (
                    <Button
                      key={r.tag}
                      variant="secondary"
                      size="sm"
                      onClick={() => void applyVerdict(activePhoto, "normal", r.tag)}
                    >
                      {r.label}
                    </Button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Virtualized filmstrip — every frame framed in its verdict color */}
          <div
            ref={stripParentRef}
            className="scrollbar-none h-[124px] shrink-0 overflow-x-auto bg-paper px-4 pb-4 sm:px-8"
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
                const verdict = verdictForPhoto(item, skippedSet.has(item.photo_id));
                return (
                  <button
                    key={item.photo_id}
                    type="button"
                    aria-current={isSelected ? "true" : undefined}
                    aria-label={`${formatSeq(item.seq_no, totalForSeq)} ${item.stem} ${plainLabel(visual.label)}`}
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
                    className="group flex flex-col gap-1 pt-1 text-left"
                  >
                    {isPartitionBoundary && (
                      <span
                        aria-hidden="true"
                        className="absolute -left-1 top-1 h-20 w-[2px] rounded-full bg-reject"
                      />
                    )}
                    <Thumb
                      key={`${item.photo_id}:${verdict ?? "none"}`}
                      src={photoImageUrl(item.photo_id, "grid")}
                      verdict={verdict}
                      pop={justMarkedId === item.photo_id}
                      veiled={visual.isFinalized && !isSelected}
                      className={`h-20 w-full transition-[box-shadow,transform] duration-200 ${
                        isSelected
                          ? "shadow-[0_0_0_2px_var(--color-paper),0_0_0_4px_var(--color-ink)]"
                          : "group-hover:-translate-y-0.5"
                      }`}
                    />
                    <span className="flex h-5 items-center justify-between px-1 text-xs tabular-nums">
                      <span className={isSelected ? "font-semibold text-ink" : "text-ink-3"}>
                        {formatSeq(item.seq_no, totalForSeq)}
                      </span>
                      {verdict === "protected" ? (
                        <Lock size={14} strokeWidth={1.5} className="text-ink-3" />
                      ) : verdict ? (
                        <span className="font-medium" style={{ color: VERDICT_STYLE[verdict].ink }}>
                          {plainLabel(visual.label)}
                        </span>
                      ) : null}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}

      <NoticeDock bottom={tab === "bursts" ? 112 : 280}>
        {toast && (
          <Notice
            onDismiss={() => setToast(null)}
            dismissLabel={t.uiDismiss}
            actions={[
              ...(lastAction ? [{ label: `${t.actionUndo} ⌘Z`, onClick: () => void handleUndo() }] : []),
              ...(lastAppliedPlanId
                ? [
                    { label: t.openTrashFolder, onClick: () => void api.openPlanFolder(lastAppliedPlanId) },
                    { label: t.goTrashBin, onClick: onGoInsights },
                  ]
                : []),
            ]}
          >
            {toast}
          </Notice>
        )}
        {error && (
          <Notice tone="error" onDismiss={() => setError(null)} dismissLabel={t.uiDismiss}>
            {t.errorPrefix}
            {error}
          </Notice>
        )}
      </NoticeDock>

      {/* Round-completion dialog (locks background controls against misclicks) */}
      {showCompleteModal && (
        <Dialog labelledBy="complete-title" width={600}>
          <div className="p-8 sm:p-12">
            <h2 id="complete-title" className="text-lg font-semibold text-ink">
              {sessionStats.undecided.length === 0
                ? t.completeModalAllDoneTitle
                : t.completeModalReachedEndTitle}
            </h2>
            <p className="mt-1 text-sm text-ink-2 tabular-nums">
              {sessionStats.undecided.length === 0
                ? t.completeModalAllDoneSubtitle(sessionStats.total)
                : t.completeModalPendingSubtitle(sessionStats.undecided.length)}
            </p>

            <dl className="mt-12 grid grid-cols-2 gap-x-8 gap-y-8 sm:grid-cols-4">
              {(
                [
                  ["keep", sessionStats.kept, t.completeStatKept],
                  ["reject", sessionStats.reject, t.completeStatReject],
                  ["unsure", sessionStats.unsure, t.completeStatUnsure],
                  ["skip", sessionStats.pendingCount, t.completeStatPending],
                ] as const
              ).map(([kind, value, label]) => (
                <div key={label}>
                  <dd className="text-[32px] font-semibold leading-10 text-ink tabular-nums">{value}</dd>
                  <dt className="mt-1 flex items-center gap-1 text-sm text-ink-2">
                    <StatusDot verdict={kind} className="-ml-2" />
                    {label}
                  </dt>
                </div>
              ))}
            </dl>

            <div className="mt-12 flex flex-wrap items-center justify-end gap-2">
              <Button variant="ghost" onClick={() => setShowCompleteModal(false)}>
                {t.completeActionClose}
              </Button>
              {tab === "revisit" && (
                <Button variant="secondary" onClick={enterAllTab}>
                  {t.reviewViewAll}
                </Button>
              )}
              {sessionStats.reject > 0 && (
                <Button
                  variant={sessionStats.undecided.length === 0 ? "mark" : "secondary"}
                  icon={Trash}
                  onClick={() => void handleOpenQuarantine()}
                >
                  {t.completeActionQuarantine(sessionStats.reject)}
                </Button>
              )}
              {sessionStats.undecided.length > 0 && (
                <Button variant="primary" onClick={enterRevisitPass}>
                  {t.completeActionRevisit(sessionStats.undecided.length)}
                </Button>
              )}
            </div>
          </div>
        </Dialog>
      )}

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
