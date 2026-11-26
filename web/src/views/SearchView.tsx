import { useState, type FormEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, photoImageUrl, type SearchMatchItem, type SearchStrictness } from "../api";
import { useI18n } from "../i18n";

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
    <div className="group flex flex-col overflow-hidden rounded-lg border border-[#33363B] bg-[#242629] transition hover:border-[#4B5563]">
      <div className="relative aspect-[4/3] w-full overflow-hidden bg-[#1B1C1E]">
        <img
          src={photoImageUrl(item.photo_id, "grid")}
          alt={item.stem}
          loading="lazy"
          className="h-full w-full object-cover transition duration-200 group-hover:scale-[1.02]"
        />

        {/* Top badges: cosine score & scene tag */}
        <div className="pointer-events-none absolute inset-x-2 top-2 flex items-start justify-between gap-1.5">
          {item.scene_label ? (
            <span className="rounded bg-[#1B1C1E]/85 px-2 py-0.5 text-[11px] text-[#E5E7EB] backdrop-blur-sm">
              {item.scene_label}
            </span>
          ) : (
            <span />
          )}
          {showScore && (
            <span className="rounded bg-[#1B1C1E]/85 px-1.5 py-0.5 font-mono text-[11px] text-[#93C5FD] tabular-nums backdrop-blur-sm">
              {item.score.toFixed(3)}
            </span>
          )}
        </div>

        {/* Hover action overlay */}
        <div className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-1.5 bg-gradient-to-t from-black/80 via-black/45 to-transparent px-2.5 pb-2 pt-6 opacity-0 transition group-hover:opacity-100 focus-within:opacity-100">
          <button
            type="button"
            onClick={() => onFindSimilar(item.photo_id)}
            className="rounded bg-[#5B8DEF] px-2.5 py-1 text-xs font-medium text-white hover:bg-[#4A7CE0]"
          >
            {t.searchFindSimilar}
          </button>
          <button
            type="button"
            onClick={() => onOpenReview(item.session_id)}
            className="rounded border border-[#4B5563] bg-[#1B1C1E]/90 px-2 py-1 text-xs text-[#E5E7EB] hover:bg-[#2C2F34]"
          >
            {t.searchOpenSession}
          </button>
        </div>
      </div>

      <div className="flex flex-1 flex-col justify-between gap-1 p-2.5">
        <div className="flex items-center justify-between gap-2">
          <span className="truncate text-xs font-medium text-[#F3F4F6]" title={item.stem}>
            {item.stem}
          </span>
          <button
            type="button"
            onClick={() => onFindSimilar(item.photo_id)}
            className="shrink-0 text-[11px] text-[#93C5FD] hover:underline"
          >
            {t.searchFindSimilar}
          </button>
        </div>
        <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 truncate text-[11px] text-[#9CA3AF]">
          {item.location_label && (
            <span className="inline-flex items-center rounded bg-[#3D9A6B]/15 px-1.5 py-0.25 text-[10px] font-medium text-[#A7F3D0]">
              📍 {item.location_label}
            </span>
          )}
          <span className="truncate">
            {item.session_name}
            {dateStr ? ` · ${dateStr}` : ""}
          </span>
        </div>
        {item.exif_brief && (
          <div className="truncate text-[11px] text-[#6B7280] tabular-nums">{item.exif_brief}</div>
        )}
      </div>
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

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div className="space-y-1">
          <h1 className="text-2xl font-medium tracking-tight text-[#F3F4F6]">{t.searchTitle}</h1>
          <p className="text-sm text-[#9CA3AF]">{t.searchSubtitle}</p>
        </div>

        {sessions.length > 0 && (
          <div className="flex items-center gap-3">
            {searchData && (
              <span className="text-xs text-[#9CA3AF] tabular-nums">
                {t.searchIndexedSummary(searchData.indexed_count, searchData.total_candidates)}
              </span>
            )}
            {(needsIndex || (searchData && searchData.indexed_count === 0)) && (
              <button
                type="button"
                disabled={indexing}
                onClick={() => void handleBuildIndex()}
                className="rounded-md border border-[#5B8DEF]/50 bg-[#5B8DEF]/15 px-3 py-1.5 text-xs font-medium text-[#93C5FD] hover:bg-[#5B8DEF]/25 disabled:opacity-50"
              >
                {indexing ? t.searchBuildingIndex : t.searchBuildIndex}
              </button>
            )}
          </div>
        )}
      </div>

      {/* Search bar */}
      <form onSubmit={handleSubmit} className="mt-5 flex flex-col gap-2 sm:flex-row">
        <input
          type="text"
          value={draftQuery}
          onChange={(e) => setDraftQuery(e.target.value)}
          placeholder={t.searchPlaceholder}
          className="flex-1 rounded-md border border-[#33363B] bg-[#242629] px-3.5 py-2.5 text-sm text-[#F3F4F6] placeholder-[#6B7280] focus:border-[#5B8DEF] focus:outline-none"
        />
        <div className="flex gap-2">
          <button
            type="submit"
            className="rounded-md bg-[#F3F4F6] px-4 py-2.5 text-sm font-medium text-[#1B1C1E] hover:bg-white"
          >
            {t.searchAction}
          </button>
          {(draftQuery || submittedQuery) && (
            <button
              type="button"
              onClick={handleClearQuery}
              className="rounded-md border border-[#33363B] bg-[#242629] px-3 py-2.5 text-sm text-[#D1D5DB] hover:bg-[#2C2F34]"
            >
              {t.searchClear}
            </button>
          )}
        </div>
      </form>

      {/* Filter controls: Session / Location / Camera / Light / Date / Match strictness */}
      <div className="mt-3 flex flex-wrap items-center gap-2.5">
        <select
          aria-label={t.searchAllSessions}
          value={sessionId}
          onChange={(e) => {
            setSessionId(e.target.value);
            setShowWeak(false);
          }}
          className="rounded-md border border-[#33363B] bg-[#242629] px-2.5 py-1.5 text-xs text-[#E5E7EB] focus:border-[#5B8DEF] focus:outline-none"
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
            className="rounded-md border border-[#33363B] bg-[#242629] px-2.5 py-1.5 text-xs text-[#E5E7EB] focus:border-[#5B8DEF] focus:outline-none"
          >
            <option value="">{t.searchAllLocations}</option>
            {locations.map((loc) => (
              <option key={loc.key} value={loc.key}>
                📍 {loc.label}
              </option>
            ))}
          </select>
        )}

        <select
          aria-label={t.searchAllCameras}
          value={camera}
          onChange={(e) => setCamera(e.target.value)}
          className="rounded-md border border-[#33363B] bg-[#242629] px-2.5 py-1.5 text-xs text-[#E5E7EB] focus:border-[#5B8DEF] focus:outline-none"
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
          className="rounded-md border border-[#33363B] bg-[#242629] px-2.5 py-1.5 text-xs text-[#E5E7EB] focus:border-[#5B8DEF] focus:outline-none"
        >
          <option value="">{t.searchAllLights}</option>
          {Object.entries(t.searchLightLabels).map(([key, label]) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
        </select>

        <label className="flex items-center gap-1.5 text-xs text-[#9CA3AF]">
          <span>{t.searchDateFrom}</span>
          <input
            type="date"
            value={dateFrom}
            onChange={(e) => setDateFrom(e.target.value)}
            className="rounded-md border border-[#33363B] bg-[#242629] px-2 py-1 text-xs text-[#E5E7EB] focus:border-[#5B8DEF] focus:outline-none"
          />
        </label>

        <label className="flex items-center gap-1.5 text-xs text-[#9CA3AF]">
          <span>{t.searchDateTo}</span>
          <input
            type="date"
            value={dateTo}
            onChange={(e) => setDateTo(e.target.value)}
            className="rounded-md border border-[#33363B] bg-[#242629] px-2 py-1 text-xs text-[#E5E7EB] focus:border-[#5B8DEF] focus:outline-none"
          />
        </label>

        {/* Strictness segmented control */}
        <div className="flex items-center gap-1 rounded-md border border-[#33363B] bg-[#242629] p-0.5 text-xs">
          <span className="px-1.5 text-[11px] text-[#9CA3AF]">{t.searchStrictnessLabel}</span>
          {(
            [
              ["strict", t.searchStrictnessStrict],
              ["balanced", t.searchStrictnessBalanced],
              ["relaxed", t.searchStrictnessRelaxed],
            ] as const
          ).map(([val, label]) => (
            <button
              key={val}
              type="button"
              onClick={() => {
                setStrictness(val);
                setShowWeak(false);
              }}
              className={`rounded px-2 py-0.5 text-[11px] transition ${
                strictness === val
                  ? "bg-[#F3F4F6] font-medium text-[#1B1C1E]"
                  : "text-[#D1D5DB] hover:bg-[#2C2F34]"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {hasFilters && (
          <button
            type="button"
            onClick={handleResetFilters}
            className="rounded-md border border-[#33363B] px-2.5 py-1.5 text-xs text-[#9CA3AF] hover:bg-[#242629] hover:text-[#F3F4F6]"
          >
            {t.searchResetFilters}
          </button>
        )}
      </div>

      {/* Active smart location parse banner */}
      {similarAnchorId === null &&
        submittedQuery &&
        (searchData?.detected_location_label || ignoreLocation) && (
          <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-md border border-[#3D9A6B]/40 bg-[#3D9A6B]/10 px-3.5 py-2 text-xs text-[#A7F3D0]">
            <span>
              {ignoreLocation
                ? t.searchLocationIgnoredBanner
                : t.searchDetectedLocation(
                    searchData!.detected_location_label!,
                    searchData!.effective_visual_query,
                  )}
            </span>
            {ignoreLocation ? (
              <button
                type="button"
                onClick={() => {
                  setIgnoreLocation(false);
                  setShowWeak(false);
                }}
                className="rounded border border-[#A7F3D0]/40 px-2.5 py-0.5 text-[#ECFDF5] hover:bg-[#3D9A6B]/25"
              >
                {t.searchRestoreLocationBtn}
              </button>
            ) : (
              !location && (
                <button
                  type="button"
                  onClick={() => {
                    setIgnoreLocation(true);
                    setShowWeak(false);
                  }}
                  className="rounded border border-[#A7F3D0]/40 px-2.5 py-0.5 text-[#ECFDF5] hover:bg-[#3D9A6B]/25"
                >
                  {t.searchIgnoreLocationBtn}
                </button>
              )
            )}
          </div>
        )}

      {notice && (
        <div className="mt-4 rounded-md border border-[#3D9A6B]/50 bg-[#3D9A6B]/10 px-4 py-2.5 text-sm text-[#A7F3D0]">
          {notice}
        </div>
      )}

      {(error || searchQ.error || similarQ.error) && (
        <div className="mt-4 rounded-md border border-[#DC4C46]/50 bg-[#DC4C46]/10 px-4 py-2.5 text-sm text-[#FCA5A5]">
          {t.errorPrefix}
          {error ??
            (searchQ.error instanceof Error
              ? searchQ.error.message
              : similarQ.error instanceof Error
                ? similarQ.error.message
                : "")}
        </div>
      )}

      {/* Similar photos view */}
      {similarAnchorId !== null ? (
        <div className="mt-6 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-[#33363B] bg-[#242629] p-3">
            <div className="flex min-w-0 items-center gap-3">
              <img
                src={photoImageUrl(similarAnchorId, "grid")}
                alt=""
                className="h-14 w-20 shrink-0 rounded bg-[#1B1C1E] object-cover"
              />
              <div className="min-w-0 space-y-0.5">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-medium text-[#93C5FD]">
                    {t.searchSimilarHeader}
                  </span>
                  {similarQ.data && (
                    <span className="rounded bg-[#1B1C1E] px-2 py-0.5 text-[11px] text-[#9CA3AF]">
                      {similarQ.data.mode === "semantic"
                        ? t.searchSimilarModeSemantic
                        : t.searchSimilarModeFallback}
                    </span>
                  )}
                </div>
                <div className="truncate text-sm font-medium text-[#F3F4F6]">
                  {similarQ.data?.anchor.stem ?? similarAnchorId}
                </div>
                {similarQ.data?.anchor.exif_brief && (
                  <div className="truncate text-xs text-[#9CA3AF]">
                    {similarQ.data.anchor.exif_brief}
                  </div>
                )}
              </div>
            </div>

            <button
              type="button"
              onClick={() => setSimilarAnchorId(null)}
              className="rounded-md border border-[#33363B] bg-[#1B1C1E] px-3 py-1.5 text-xs text-[#E5E7EB] hover:bg-[#2C2F34]"
            >
              {t.searchBackToResults}
            </button>
          </div>

          {similarQ.isLoading ? (
            <p className="py-10 text-sm text-[#9CA3AF]">{t.loading}</p>
          ) : (similarQ.data?.similar.length ?? 0) === 0 ? (
            <div className="space-y-3 py-12 text-center text-sm text-[#9CA3AF]">
              <p>{t.searchSimilarEmpty}</p>
              {(similarQ.data?.weak_similar.length ?? 0) > 0 && (
                <button
                  type="button"
                  onClick={() => setShowWeak((v) => !v)}
                  className="rounded-md border border-[#33363B] bg-[#242629] px-3.5 py-1.5 text-xs text-[#D1D5DB] hover:bg-[#2C2F34]"
                >
                  {showWeak
                    ? t.searchHideWeak
                    : t.searchShowWeak(similarQ.data!.weak_similar.length)}
                </button>
              )}
            </div>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
                {(similarQ.data?.similar ?? []).map((item) => (
                  <PhotoSearchCard
                    key={item.photo_id}
                    item={item}
                    showScore
                    onFindSimilar={(pid) => {
                      setSimilarAnchorId(pid);
                      setShowWeak(false);
                    }}
                    onOpenReview={onOpenReview}
                  />
                ))}
              </div>
              {(similarQ.data?.weak_similar.length ?? 0) > 0 && (
                <div className="border-t border-[#33363B] pt-3">
                  <button
                    type="button"
                    onClick={() => setShowWeak((v) => !v)}
                    className="rounded-md border border-[#33363B] bg-[#242629] px-3.5 py-1.5 text-xs text-[#9CA3AF] hover:bg-[#2C2F34] hover:text-[#F3F4F6]"
                  >
                    {showWeak
                      ? t.searchHideWeak
                      : t.searchShowWeak(similarQ.data!.weak_similar.length)}
                  </button>
                </div>
              )}
            </div>
          )}

          {showWeak && (similarQ.data?.weak_similar.length ?? 0) > 0 && (
            <div className="space-y-2.5 pt-2 opacity-80">
              <p className="text-xs text-[#9CA3AF]">{t.searchWeakSectionHint}</p>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
                {similarQ.data!.weak_similar.map((item) => (
                  <PhotoSearchCard
                    key={item.photo_id}
                    item={item}
                    showScore
                    onFindSimilar={(pid) => {
                      setSimilarAnchorId(pid);
                      setShowWeak(false);
                    }}
                    onOpenReview={onOpenReview}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        /* Search results grid */
        <div className="mt-6 space-y-4">
          {searchData && (
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#33363B] pb-2 text-xs text-[#9CA3AF]">
              <div className="flex flex-wrap items-center gap-2.5">
                <span className="tabular-nums">{t.searchMatchCount(searchData.matches.length)}</span>
                {searchData.score_threshold !== null && (
                  <span className="rounded bg-[#1F2124] px-2 py-0.5 font-mono text-[11px] text-[#9CA3AF] tabular-nums">
                    {t.searchThresholdHint(searchData.score_threshold.toFixed(3))}
                  </span>
                )}
              </div>
              <span className="rounded bg-[#242629] px-2 py-0.5 text-[#D1D5DB]">{modeBadge}</span>
            </div>
          )}

          {searchQ.isLoading ? (
            <p className="py-10 text-sm text-[#9CA3AF]">{t.loading}</p>
          ) : (searchData?.matches.length ?? 0) === 0 ? (
            <div className="space-y-3 py-14 text-center">
              <p className="text-base text-[#F3F4F6]">{t.searchNoResults}</p>
              <p className="text-sm text-[#9CA3AF]">{t.searchNoResultsHint}</p>
              {(searchData?.weak_matches.length ?? 0) > 0 && (
                <button
                  type="button"
                  onClick={() => setShowWeak((v) => !v)}
                  className="rounded-md border border-[#33363B] bg-[#242629] px-3.5 py-1.5 text-xs text-[#D1D5DB] hover:bg-[#2C2F34]"
                >
                  {showWeak
                    ? t.searchHideWeak
                    : t.searchShowWeak(searchData!.weak_matches.length)}
                </button>
              )}
            </div>
          ) : (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
                {(searchData?.matches ?? []).map((item) => (
                  <PhotoSearchCard
                    key={item.photo_id}
                    item={item}
                    showScore={searchData?.mode === "semantic"}
                    onFindSimilar={(pid) => {
                      setSimilarAnchorId(pid);
                      setShowWeak(false);
                    }}
                    onOpenReview={onOpenReview}
                  />
                ))}
              </div>
              {(searchData?.weak_matches.length ?? 0) > 0 && (
                <div className="border-t border-[#33363B] pt-3">
                  <button
                    type="button"
                    onClick={() => setShowWeak((v) => !v)}
                    className="rounded-md border border-[#33363B] bg-[#242629] px-3.5 py-1.5 text-xs text-[#9CA3AF] hover:bg-[#2C2F34] hover:text-[#F3F4F6]"
                  >
                    {showWeak
                      ? t.searchHideWeak
                      : t.searchShowWeak(searchData!.weak_matches.length)}
                  </button>
                </div>
              )}
            </div>
          )}

          {showWeak && (searchData?.weak_matches.length ?? 0) > 0 && (
            <div className="space-y-2.5 pt-2 opacity-80">
              <p className="text-xs text-[#9CA3AF]">{t.searchWeakSectionHint}</p>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
                {searchData!.weak_matches.map((item) => (
                  <PhotoSearchCard
                    key={item.photo_id}
                    item={item}
                    showScore={searchData?.mode === "semantic"}
                    onFindSimilar={(pid) => {
                      setSimilarAnchorId(pid);
                      setShowWeak(false);
                    }}
                    onOpenReview={onOpenReview}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
