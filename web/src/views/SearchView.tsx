import { useState, type FormEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, photoImageUrl, type SearchMatchItem, type SearchStrictness } from "../api";
import { Button, EmptyState, Icon, IconButton, Loading, Notice, NoticeDock, Tabs } from "../components/ui";
import { useI18n } from "../i18n";
import { ArrowLeft, Camera, ChevronDown, Location, Search, X } from "reicon-react";

interface SearchViewProps {
  onOpenReview: (sessionId: string) => void;
}

function formatDate(ts: number | null): string {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  if (Number.isNaN(d.getTime())) return "";
  return d.toISOString().slice(0, 10);
}

function PhotoSearchCard({
  item,
  showScore,
  onFindSimilar,
  onOpenReview,
}: {
  item: SearchMatchItem;
  showScore: boolean;
  onFindSimilar: (photoId: string) => void;
  onOpenReview: (sessionId: string) => void;
}) {
  const { t } = useI18n();
  const dateStr = formatDate(item.capture_time);

  return (
    <article className="intent group flex min-w-0 flex-col gap-2">
      <div className="relative aspect-[4/3] w-full overflow-hidden rounded-tile bg-table">
        <button
          type="button"
          onClick={() => onFindSimilar(item.photo_id)}
          className="block h-full w-full"
          aria-label={`${t.searchFindSimilar} ${item.stem}`}
        >
          <img
            src={photoImageUrl(item.photo_id, "grid")}
            alt={item.stem}
            loading="lazy"
            className="h-full w-full object-cover transition-transform duration-500 ease-out group-hover:scale-[1.03]"
          />
        </button>

        <div className="reveal pointer-events-none absolute inset-x-2 top-2 flex items-start justify-between gap-2">
          {item.scene_label ? (
            <span className="rounded-chip bg-paper/90 px-2 py-1 text-xs font-medium text-ink">
              {item.scene_label}
            </span>
          ) : (
            <span />
          )}
          {showScore && (
            <span className="rounded-chip bg-paper/90 px-2 py-1 text-xs font-medium text-ink tabular-nums">
              {item.score.toFixed(3)}
            </span>
          )}
        </div>

        <div className="reveal absolute inset-x-2 bottom-2 flex items-center justify-between gap-2">
          <Button size="sm" variant="primary" onClick={() => onFindSimilar(item.photo_id)}>
            {t.searchFindSimilar}
          </Button>
          <Button
            size="sm"
            variant="secondary"
            className="bg-paper/90"
            onClick={() => onOpenReview(item.session_id)}
          >
            {t.searchOpenSession}
          </Button>
        </div>
      </div>

      <div className="min-w-0 px-1">
        <div className="truncate text-sm font-medium text-ink" title={item.stem}>
          {item.stem}
        </div>
        <div className="flex min-w-0 items-center gap-2 text-sm text-ink-2">
          <span className="truncate">{item.session_name}</span>
          {dateStr && <span className="shrink-0 tabular-nums text-ink-3">{dateStr}</span>}
        </div>
        {item.location_label && (
          <div className="-ml-1 flex min-w-0 items-center text-sm text-ink-2">
            <Location size={16} strokeWidth={1.5} className="mx-1 shrink-0" />
            <span className="truncate">{item.location_label}</span>
          </div>
        )}
        {item.exif_brief && (
          <div className="truncate text-sm text-ink-3 tabular-nums" title={item.exif_brief}>
            {item.exif_brief}
          </div>
        )}
      </div>
    </article>
  );
}

function PhotoGrid({ children, dim = false }: { children: React.ReactNode; dim?: boolean }) {
  return (
    <div
      className={`grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 ${
        dim ? "opacity-70" : ""
      }`}
    >
      {children}
    </div>
  );
}

export function SearchView({ onOpenReview }: SearchViewProps) {
  const { lang, t } = useI18n();
  const queryClient = useQueryClient();

  const [draftQuery, setDraftQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [location, setLocation] = useState("");
  const [ignoreLocation, setIgnoreLocation] = useState(false);
  const [strictness, setStrictness] = useState<SearchStrictness>("balanced");
  const [showWeak, setShowWeak] = useState(false);
  const [camera, setCamera] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [lightBucket, setLightBucket] = useState("");
  const [similarAnchorId, setSimilarAnchorId] = useState<string | null>(null);

  const [indexing, setIndexing] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const sessionsQ = useQuery({
    queryKey: ["sessions"],
    queryFn: () => api.listSessions(),
  });
  const sessions = sessionsQ.data?.items ?? [];

  const searchQ = useQuery({
    queryKey: [
      "search",
      submittedQuery,
      sessionId,
      location,
      ignoreLocation,
      strictness,
      camera,
      dateFrom,
      dateTo,
      lightBucket,
      lang,
    ],
    queryFn: () =>
      api.searchPhotos({
        q: submittedQuery,
        sessionId: sessionId || undefined,
        location: location || undefined,
        ignoreLocation,
        strictness,
        camera: camera || undefined,
        dateFrom: dateFrom || undefined,
        dateTo: dateTo || undefined,
        lightBucket: lightBucket || undefined,
        limit: 48,
        lang,
      }),
    enabled: similarAnchorId === null,
  });

  const similarQ = useQuery({
    queryKey: ["similar", similarAnchorId, sessionId, strictness, lang],
    queryFn: () =>
      api.findSimilarPhotos(similarAnchorId!, {
        limit: 18,
        sessionId: sessionId || undefined,
        strictness,
        lang,
      }),
    enabled: similarAnchorId !== null,
  });

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setSimilarAnchorId(null);
    setIgnoreLocation(false);
    setShowWeak(false);
    setSubmittedQuery(draftQuery.trim());
  }

  function handleClearQuery() {
    setDraftQuery("");
    setSubmittedQuery("");
    setIgnoreLocation(false);
    setShowWeak(false);
    setSimilarAnchorId(null);
  }

  function handleResetFilters() {
    setSessionId("");
    setLocation("");
    setIgnoreLocation(false);
    setStrictness("balanced");
    setShowWeak(false);
    setCamera("");
    setDateFrom("");
    setDateTo("");
    setLightBucket("");
    setSimilarAnchorId(null);
  }

  async function handleBuildIndex() {
    const targets = sessionId ? [sessionId] : sessions.map((s) => s.id);
    if (targets.length === 0) return;
    setIndexing(true);
    setError(null);
    setNotice(null);
    try {
      let totalIndexed = 0;
      let totalPhotos = 0;
      for (const sid of targets) {
        const res = await api.embedSession(sid);
        totalIndexed += res.indexed_count;
        totalPhotos += res.total_photos;
      }
      setNotice(t.searchBuildIndexDone(totalIndexed, totalPhotos));
      await queryClient.invalidateQueries({ queryKey: ["search"] });
      await queryClient.invalidateQueries({ queryKey: ["similar"] });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIndexing(false);
    }
  }

  const searchData = searchQ.data;
  const cameras = searchData?.cameras ?? [];
  const locations = searchData?.locations ?? [];
  const hasFilters = Boolean(
    sessionId || location || camera || dateFrom || dateTo || lightBucket || strictness !== "balanced",
  );
  const needsIndex =
    sessions.length > 0 &&
    searchData !== undefined &&
    searchData.total_candidates > searchData.indexed_count;

  const modeBadge =
    searchData?.mode === "semantic"
      ? t.searchModeSemantic
      : searchData?.mode === "fts_exif"
        ? t.searchModeFallback
        : t.searchModeFilter;

  const showWeakToggle = (n: number) => (
    <Button variant="ghost" iconRight={ChevronDown} onClick={() => setShowWeak((v) => !v)}>
      {showWeak ? t.searchHideWeak : t.searchShowWeak(n)}
    </Button>
  );

  const renderCards = (items: SearchMatchItem[], showScore: boolean) =>
    items.map((item) => (
      <PhotoSearchCard
        key={item.photo_id}
        item={item}
        showScore={showScore}
        onFindSimilar={(pid) => {
          setSimilarAnchorId(pid);
          setShowWeak(false);
        }}
        onOpenReview={onOpenReview}
      />
    ));

  const queryError =
    error ??
    (searchQ.error instanceof Error
      ? searchQ.error.message
      : similarQ.error instanceof Error
        ? similarQ.error.message
        : null);

  return (
    <div className="mx-auto max-w-6xl px-6 pb-32 pt-12 sm:px-12">
      {/* Search field is the page */}
      <form onSubmit={handleSubmit} className="relative">
        <Search
          size={20}
          strokeWidth={1.5}
          className="pointer-events-none absolute left-5 top-1/2 -translate-y-1/2 text-ink-2"
          aria-hidden="true"
        />
        <input
          type="search"
          value={draftQuery}
          onChange={(e) => setDraftQuery(e.target.value)}
          placeholder={t.searchPlaceholder}
          aria-label={t.searchAction}
          className="field h-16 w-full rounded-panel pl-14 pr-40 text-base [&::-webkit-search-cancel-button]:hidden"
        />
        <div className="absolute right-2 top-1/2 flex -translate-y-1/2 items-center gap-1">
          {(draftQuery || submittedQuery) && (
            <IconButton icon={X} label={t.searchClear} onClick={handleClearQuery} />
          )}
          <Button type="submit" variant="primary" size="lg" className="h-12 rounded-[14px]">
            {t.searchAction}
          </Button>
        </div>
      </form>

      {/* Filters */}
      <div className="mt-6 flex flex-wrap items-center gap-2">
        <select
          aria-label={t.searchAllSessions}
          value={sessionId}
          onChange={(e) => {
            setSessionId(e.target.value);
            setShowWeak(false);
          }}
          className={`field h-10 ${sessionId ? "font-medium" : "text-ink-2"}`}
        >
          <option value="">{t.searchAllSessions}</option>
          {sessions.map((s) => (
            <option key={s.id} value={s.id}>
              {s.name}
            </option>
          ))}
        </select>

        {locations.length > 0 && (
          <select
            aria-label={t.searchAllLocations}
            value={location}
            onChange={(e) => {
              setLocation(e.target.value);
              setIgnoreLocation(false);
              setShowWeak(false);
            }}
            className={`field h-10 ${location ? "font-medium" : "text-ink-2"}`}
          >
            <option value="">{t.searchAllLocations}</option>
            {locations.map((loc) => (
              <option key={loc.key} value={loc.key}>
                {loc.label}
              </option>
            ))}
          </select>
        )}

        <select
          aria-label={t.searchAllCameras}
          value={camera}
          onChange={(e) => setCamera(e.target.value)}
          className={`field h-10 ${camera ? "font-medium" : "text-ink-2"}`}
        >
          <option value="">{t.searchAllCameras}</option>
          {cameras.map((cam) => (
            <option key={cam} value={cam}>
              {cam}
            </option>
          ))}
        </select>

        <select
          aria-label={t.searchAllLights}
          value={lightBucket}
          onChange={(e) => setLightBucket(e.target.value)}
          className={`field h-10 ${lightBucket ? "font-medium" : "text-ink-2"}`}
        >
          <option value="">{t.searchAllLights}</option>
          {Object.entries(t.searchLightLabels).map(([key, label]) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
        </select>

        <span className="flex items-center gap-1">
          <input
            type="date"
            value={dateFrom}
            onChange={(e) => setDateFrom(e.target.value)}
            aria-label={t.searchDateFrom}
            title={t.searchDateFrom}
            className={`field h-10 ${dateFrom ? "" : "text-ink-2"}`}
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
            className={`field h-10 ${dateTo ? "" : "text-ink-2"}`}
          />
        </span>

        <div className="flex items-center" title={t.searchStrictnessLabel}>
          <Tabs
            size="sm"
            value={strictness}
            items={[
              { id: "strict", label: t.searchStrictnessStrict },
              { id: "balanced", label: t.searchStrictnessBalanced },
              { id: "relaxed", label: t.searchStrictnessRelaxed },
            ]}
            onChange={(val) => {
              setStrictness(val);
              setShowWeak(false);
            }}
          />
        </div>

        {hasFilters && (
          <Button variant="ghost" size="sm" onClick={handleResetFilters}>
            {t.searchResetFilters}
          </Button>
        )}
      </div>

      {/* Detected place in the query */}
      {similarAnchorId === null &&
        submittedQuery &&
        (searchData?.detected_location_label || ignoreLocation) && (
          <div className="fade-enter mt-6 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
            <span className="flex items-center gap-1 text-ink">
              <Icon as={Location} className="text-ink-2" />
              {ignoreLocation
                ? t.searchLocationIgnoredBanner
                : t.searchDetectedLocation(
                    searchData!.detected_location_label!,
                    searchData!.effective_visual_query,
                  )}
            </span>
            {ignoreLocation ? (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setIgnoreLocation(false);
                  setShowWeak(false);
                }}
              >
                {t.searchRestoreLocationBtn}
              </Button>
            ) : (
              !location && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setIgnoreLocation(true);
                    setShowWeak(false);
                  }}
                >
                  {t.searchIgnoreLocationBtn}
                </Button>
              )
            )}
          </div>
        )}

      {similarAnchorId !== null ? (
        <section className="mt-12 space-y-8">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex min-w-0 items-center gap-4">
              <IconButton
                icon={ArrowLeft}
                label={t.searchBackToResults}
                onClick={() => setSimilarAnchorId(null)}
              />
              <img
                src={photoImageUrl(similarAnchorId, "grid")}
                alt=""
                className="h-16 w-24 shrink-0 rounded-chip bg-table object-cover"
              />
              <div className="min-w-0">
                <h2 className="truncate text-lg font-semibold text-ink">
                  {similarQ.data?.anchor.stem ?? similarAnchorId}
                </h2>
                <p className="flex min-w-0 items-center gap-4 text-sm text-ink-2">
                  <span className="shrink-0">{t.searchSimilarHeader}</span>
                  {similarQ.data?.anchor.exif_brief && (
                    <span className="truncate text-ink-3">{similarQ.data.anchor.exif_brief}</span>
                  )}
                </p>
              </div>
            </div>
            {similarQ.data && (
              <span className="text-sm text-ink-3">
                {similarQ.data.mode === "semantic"
                  ? t.searchSimilarModeSemantic
                  : t.searchSimilarModeFallback}
              </span>
            )}
          </div>

          {similarQ.isLoading ? (
            <Loading label={t.loading} />
          ) : (similarQ.data?.similar.length ?? 0) === 0 ? (
            <EmptyState title={t.searchSimilarEmpty}>
              {(similarQ.data?.weak_similar.length ?? 0) > 0 &&
                showWeakToggle(similarQ.data!.weak_similar.length)}
            </EmptyState>
          ) : (
            <>
              <PhotoGrid>{renderCards(similarQ.data?.similar ?? [], true)}</PhotoGrid>
              {(similarQ.data?.weak_similar.length ?? 0) > 0 && (
                <div>{showWeakToggle(similarQ.data!.weak_similar.length)}</div>
              )}
            </>
          )}

          {showWeak && (similarQ.data?.weak_similar.length ?? 0) > 0 && (
            <PhotoGrid dim>{renderCards(similarQ.data!.weak_similar, true)}</PhotoGrid>
          )}
        </section>
      ) : (
        <section className="mt-12 space-y-8">
          {searchData && (
            <div className="flex flex-wrap items-center justify-between gap-4 text-sm tabular-nums">
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
                <span className="font-medium text-ink">
                  {t.searchMatchCount(searchData.matches.length)}
                </span>
                {searchData.score_threshold !== null && (
                  <span className="text-ink-3">
                    {t.searchThresholdHint(searchData.score_threshold.toFixed(3))}
                  </span>
                )}
                <span className="text-ink-3">{modeBadge}</span>
              </div>

              {sessions.length > 0 && (
                <div className="flex items-center gap-4">
                  <span className="text-ink-3">
                    {t.searchIndexedSummary(searchData.indexed_count, searchData.total_candidates)}
                  </span>
                  {(needsIndex || searchData.indexed_count === 0) && (
                    <Button
                      variant="secondary"
                      size="sm"
                      icon={Camera}
                      disabled={indexing}
                      onClick={() => void handleBuildIndex()}
                    >
                      {indexing ? t.searchBuildingIndex : t.searchBuildIndex}
                    </Button>
                  )}
                </div>
              )}
            </div>
          )}

          {searchQ.isLoading ? (
            <Loading label={t.loading} />
          ) : (searchData?.matches.length ?? 0) === 0 ? (
            <EmptyState title={t.searchNoResults}>
              {(searchData?.weak_matches.length ?? 0) > 0 &&
                showWeakToggle(searchData!.weak_matches.length)}
              {hasFilters && (
                <Button variant="secondary" onClick={handleResetFilters}>
                  {t.searchResetFilters}
                </Button>
              )}
            </EmptyState>
          ) : (
            <>
              <PhotoGrid>
                {renderCards(searchData?.matches ?? [], searchData?.mode === "semantic")}
              </PhotoGrid>
              {(searchData?.weak_matches.length ?? 0) > 0 && (
                <div>{showWeakToggle(searchData!.weak_matches.length)}</div>
              )}
            </>
          )}

          {showWeak && (searchData?.weak_matches.length ?? 0) > 0 && (
            <PhotoGrid dim>
              {renderCards(searchData!.weak_matches, searchData?.mode === "semantic")}
            </PhotoGrid>
          )}
        </section>
      )}

      <NoticeDock>
        {notice && (
          <Notice onDismiss={() => setNotice(null)} dismissLabel={t.uiDismiss}>
            {notice}
          </Notice>
        )}
        {queryError && (
          <Notice tone="error" onDismiss={error ? () => setError(null) : undefined} dismissLabel={t.uiDismiss}>
            {t.errorPrefix}
            {queryError}
          </Notice>
        )}
      </NoticeDock>
    </div>
  );
}
