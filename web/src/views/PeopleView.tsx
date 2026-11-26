import { useState, type FormEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  photoFaceCropUrl,
  photoImageUrl,
  type PeopleGroupsResponse,
  type PeoplePhotoItem,
} from "../api";
import { useI18n } from "../i18n";

interface PeopleViewProps {
  onOpenReview: (sessionId: string) => void;
}

type ActiveSelection =
  | { kind: "person"; personId: string }
  | { kind: "cluster"; clusterId: string }
  | { kind: "group_photo" }
  | { kind: "unnamed" }
  | { kind: "no_face" };

function ExportPeopleModal({
  data,
  sessionId,
  onClose,
  onSuccess,
}: {
  data: PeopleGroupsResponse;
  sessionId: string;
  onClose: () => void;
  onSuccess: (msg: string) => void;
}) {
  const { lang, t } = useI18n();
  const [targetDir, setTargetDir] = useState("");
  const [selectedKeys, setSelectedKeys] = useState<string[]>(() => {
    const keys: string[] = [];
    for (const p of data.people) {
      if (p.photo_count > 0) keys.push(p.group_key);
    }
    if (data.group_photo_total_count > 0) keys.push("group_photo");
    return keys;
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const exportableOptions: Array<{ key: string; label: string; count: number }> = [
    ...data.people
      .filter((p) => p.photo_count > 0)
      .map((p) => ({ key: p.group_key, label: p.name, count: p.photo_count })),
    ...(data.group_photo_total_count > 0
      ? [{ key: "group_photo", label: t.peopleGroupPhotos, count: data.group_photo_total_count }]
      : []),
    ...(data.unnamed_photos.length > 0
      ? [{ key: "unnamed", label: t.peopleUnnamed, count: data.unnamed_photos.length }]
      : []),
    ...(data.no_face_photos.length > 0
      ? [{ key: "no_face", label: t.peopleNoFace, count: data.no_face_photos.length }]
      : []),
  ];

  function toggleKey(key: string) {
    setSelectedKeys((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
    );
  }

  async function handlePickFolder() {
    setError(null);
    try {
      const res = await api.pickFolder();
      if (!res.canceled && res.path) {
        setTargetDir(res.path);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleExport(e: FormEvent) {
    e.preventDefault();
    if (!targetDir.trim() || selectedKeys.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.exportPeopleGroups({
        target_dir: targetDir.trim(),
        session_id: sessionId || undefined,
        group_keys: selectedKeys,
        lang,
      });
      onSuccess(t.peopleExportDone(res.exported_photos, res.exported_files, res.target_dir));
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="w-full max-w-lg rounded-xl border border-[#33363B] bg-[#1F2124] p-5 shadow-2xl">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-medium text-[#F3F4F6]">{t.peopleExportModalTitle}</h2>
            <p className="mt-1 text-xs text-[#9CA3AF]">{t.peopleExportModalDesc}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded px-2 py-1 text-xs text-[#9CA3AF] hover:bg-[#2C2F34] hover:text-[#F3F4F6]"
          >
            ✕
          </button>
        </div>

        <form onSubmit={(e) => void handleExport(e)} className="mt-4 space-y-4">
          <div className="space-y-1.5">
            <label className="block text-xs font-medium text-[#D1D5DB]">
              {t.peopleExportTargetLabel}
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={targetDir}
                onChange={(e) => setTargetDir(e.target.value)}
                placeholder={t.peopleExportTargetPlaceholder}
                className="flex-1 rounded-md border border-[#33363B] bg-[#1B1C1E] px-3 py-2 text-xs text-[#F3F4F6] placeholder-[#6B7280] focus:border-[#5B8DEF] focus:outline-none"
              />
              <button
                type="button"
                onClick={() => void handlePickFolder()}
                className="shrink-0 rounded-md border border-[#33363B] bg-[#242629] px-3 py-2 text-xs text-[#E5E7EB] hover:bg-[#2C2F34]"
              >
                {t.peopleExportPickFolder}
              </button>
            </div>
          </div>

          <div className="space-y-1.5">
            <div className="text-xs font-medium text-[#D1D5DB]">{t.peopleExportSelectGroups}</div>
            <div className="max-h-48 space-y-1.5 overflow-y-auto rounded-md border border-[#33363B] bg-[#1B1C1E] p-2.5">
              {exportableOptions.map((opt) => (
                <label
                  key={opt.key}
                  className="flex cursor-pointer items-center justify-between gap-2 rounded px-2 py-1 text-xs hover:bg-[#242629]"
                >
                  <span className="flex items-center gap-2 text-[#E5E7EB]">
                    <input
                      type="checkbox"
                      checked={selectedKeys.includes(opt.key)}
                      onChange={() => toggleKey(opt.key)}
                      className="rounded border-[#4B5563]"
                    />
                    <span>{opt.label}</span>
                  </span>
                  <span className="text-[#9CA3AF] tabular-nums">{t.photoCount(opt.count)}</span>
                </label>
              ))}
            </div>
          </div>

          <p className="rounded-md border border-[#3D9A6B]/40 bg-[#3D9A6B]/10 px-3 py-2 text-xs text-[#A7F3D0]">
            {t.peopleExportSafety}
          </p>

          {error && (
            <p className="rounded-md border border-[#DC4C46]/50 bg-[#DC4C46]/10 px-3 py-2 text-xs text-[#FCA5A5]">
              {t.errorPrefix}
              {error}
            </p>
          )}

          <div className="flex justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={onClose}
              className="rounded-md border border-[#33363B] px-3.5 py-2 text-xs text-[#D1D5DB] hover:bg-[#2C2F34]"
            >
              {t.modalCancel}
            </button>
            <button
              type="submit"
              disabled={busy || !targetDir.trim() || selectedKeys.length === 0}
              className="rounded-md bg-[#3D9A6B] px-4 py-2 text-xs font-medium text-white hover:bg-[#32825A] disabled:opacity-50"
            >
              {busy ? t.peopleExporting : t.peopleExportConfirm}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export function PeopleView({ onOpenReview }: PeopleViewProps) {
  const { t } = useI18n();
  const queryClient = useQueryClient();

  const [sessionId, setSessionId] = useState("");
  const [groupPersonFilter, setGroupPersonFilter] = useState<string>("");
  const [selection, setSelection] = useState<ActiveSelection>({ kind: "group_photo" });

  const [nameInput, setNameInput] = useState("");
  const [mergeTarget, setMergeTarget] = useState("");
  const [indexing, setIndexing] = useState(false);
  const [mutating, setMutating] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const sessionsQ = useQuery({
    queryKey: ["sessions"],
    queryFn: () => api.listSessions(),
  });
  const sessions = sessionsQ.data?.items ?? [];

  const peopleQ = useQuery({
    queryKey: ["people", sessionId, groupPersonFilter],
    queryFn: () =>
      api.listPeopleGroups({
        sessionId: sessionId || undefined,
        groupPersonId: groupPersonFilter || undefined,
      }),
  });
  const data = peopleQ.data;

  async function refreshPeople() {
    await queryClient.invalidateQueries({ queryKey: ["people"] });
  }

  async function handleIndexPeople() {
    const targets = sessionId ? [sessionId] : sessions.map((s) => s.id);
    if (targets.length === 0) return;
    setIndexing(true);
    setError(null);
    setNotice(null);
    try {
      let totalFaces = 0;
      let totalNewly = 0;
      let lastClusters = 0;
      for (const sid of targets) {
        const res = await api.indexSessionPeople(sid);
        totalFaces += res.valid_face_count;
        totalNewly += res.newly_embedded_faces;
        lastClusters = res.unnamed_cluster_count;
      }
      setNotice(t.peopleIndexDone(totalFaces, totalNewly, lastClusters));
      await refreshPeople();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIndexing(false);
    }
  }

  async function handleNameCluster(clusterId: string, e: FormEvent) {
    e.preventDefault();
    if (!nameInput.trim()) return;
    setMutating(true);
    setError(null);
    setNotice(null);
    try {
      const clean = nameInput.trim();
      await api.labelPeople({ action: "name", name: clean, cluster_id: clusterId });
      setNameInput("");
      setNotice(t.peopleNameSuccess(clean));
      await refreshPeople();
      setSelection({ kind: "unnamed" });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setMutating(false);
    }
  }

  async function handleRenamePerson(personId: string, e: FormEvent) {
    e.preventDefault();
    if (!nameInput.trim()) return;
    setMutating(true);
    setError(null);
    setNotice(null);
    try {
      const clean = nameInput.trim();
      await api.labelPeople({ action: "name", name: clean, person_id: personId });
      setNameInput("");
      setNotice(t.peopleNameSuccess(clean));
      await refreshPeople();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setMutating(false);
    }
  }

  async function handleMerge(source: { personId?: string; clusterId?: string }) {
    if (!mergeTarget) return;
    setMutating(true);
    setError(null);
    setNotice(null);
    try {
      const [targetKind, targetId] = mergeTarget.split(":", 2);
      if (source.personId && targetKind === "person") {
        await api.labelPeople({
          action: "merge",
          source_person_id: source.personId,
          target_person_id: targetId,
        });
        setSelection({ kind: "person", personId: targetId });
      } else if (source.clusterId && targetKind === "person") {
        await api.labelPeople({
          action: "merge",
          source_cluster_id: source.clusterId,
          target_person_id: targetId,
        });
        setSelection({ kind: "person", personId: targetId });
      } else if (source.clusterId && targetKind === "cluster") {
        await api.labelPeople({
          action: "merge",
          source_cluster_id: source.clusterId,
          target_cluster_id: targetId,
        });
        setSelection({ kind: "cluster", clusterId: targetId });
      }
      setMergeTarget("");
      setNotice(t.peopleMergeSuccess);
      await refreshPeople();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setMutating(false);
    }
  }

  async function handleIgnoreCluster(clusterId: string) {
    setMutating(true);
    setError(null);
    setNotice(null);
    try {
      await api.labelPeople({ action: "ignore", cluster_id: clusterId });
      setNotice(t.peopleIgnoreSuccess);
      setSelection({ kind: "unnamed" });
      await refreshPeople();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setMutating(false);
    }
  }

  async function handleIgnoreFace(contentFp: string, faceIdx: number) {
    setMutating(true);
    setError(null);
    setNotice(null);
    try {
      await api.labelPeople({
        action: "ignore",
        faces: [{ content_fp: contentFp, face_idx: faceIdx }],
      });
      setNotice(t.peopleIgnoreSuccess);
      await refreshPeople();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setMutating(false);
    }
  }

  // Resolve currently active group photos & title
  let activeTitle = t.peopleGroupPhotos;
  let activePhotos: PeoplePhotoItem[] = [];
  const selectedPerson =
    selection.kind === "person"
      ? data?.people.find((p) => p.person_id === selection.personId)
      : undefined;
  const selectedClusterIndex =
    selection.kind === "cluster"
      ? (data?.unnamed_clusters.findIndex((c) => c.cluster_id === selection.clusterId) ?? -1)
      : -1;
  const selectedCluster =
    selectedClusterIndex >= 0 ? data?.unnamed_clusters[selectedClusterIndex] : undefined;

  if (data) {
    if (selectedPerson) {
      activeTitle = selectedPerson.name;
      activePhotos = selectedPerson.photos;
    } else if (selectedCluster) {
      activeTitle = t.peopleClusterLabel(selectedClusterIndex + 1, selectedCluster.face_count);
      activePhotos = selectedCluster.photos;
    } else if (selection.kind === "unnamed") {
      activeTitle = t.peopleUnnamed;
      activePhotos = data.unnamed_photos;
    } else if (selection.kind === "no_face") {
      activeTitle = t.peopleNoFace;
      activePhotos = data.no_face_photos;
    } else {
      activeTitle = t.peopleGroupPhotos;
      activePhotos = data.group_photos;
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      {/* Header & actions */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div className="space-y-1">
          <h1 className="text-2xl font-medium tracking-tight text-[#F3F4F6]">{t.peopleTitle}</h1>
          <p className="text-sm text-[#9CA3AF]">{t.peopleSubtitle}</p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <select
            aria-label={t.searchAllSessions}
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
            className="rounded-md border border-[#33363B] bg-[#242629] px-2.5 py-2 text-xs text-[#E5E7EB] focus:border-[#5B8DEF] focus:outline-none"
          >
            <option value="">{t.searchAllSessions}</option>
            {sessions.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>

          <button
            type="button"
            disabled={indexing || sessions.length === 0}
            onClick={() => void handleIndexPeople()}
            className="rounded-md border border-[#5B8DEF]/50 bg-[#5B8DEF]/15 px-3.5 py-2 text-xs font-medium text-[#93C5FD] hover:bg-[#5B8DEF]/25 disabled:opacity-50"
          >
            {indexing ? t.peopleIndexing : t.peopleIndexAction}
          </button>

          <button
            type="button"
            disabled={!data || data.total_kept_photos === 0}
            onClick={() => setShowExportModal(true)}
            className="rounded-md bg-[#3D9A6B] px-3.5 py-2 text-xs font-medium text-white hover:bg-[#32825A] disabled:opacity-50"
          >
            {t.peopleExportAction}
          </button>
        </div>
      </div>

      {notice && (
        <div className="mt-4 rounded-md border border-[#3D9A6B]/50 bg-[#3D9A6B]/10 px-4 py-2.5 text-sm text-[#A7F3D0]">
          {notice}
        </div>
      )}

      {(error || peopleQ.error) && (
        <div className="mt-4 rounded-md border border-[#DC4C46]/50 bg-[#DC4C46]/10 px-4 py-2.5 text-sm text-[#FCA5A5]">
          {t.errorPrefix}
          {error ?? (peopleQ.error instanceof Error ? peopleQ.error.message : "")}
        </div>
      )}

      {/* Main split layout: sidebar + photo grid */}
      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-[260px_1fr]">
        {/* Sidebar */}
        <aside className="space-y-5 rounded-lg border border-[#33363B] bg-[#242629] p-3.5">
          {/* Named people */}
          <div>
            <div className="mb-2 px-1 text-[11px] font-medium uppercase tracking-wider text-[#9CA3AF]">
              {t.peopleNamedSection}
            </div>
            {(data?.people.length ?? 0) === 0 ? (
              <p className="px-1 text-xs text-[#6B7280]">—</p>
            ) : (
              <div className="space-y-1">
                {(data?.people ?? []).map((p) => {
                  const active = selection.kind === "person" && selection.personId === p.person_id;
                  return (
                    <button
                      key={p.person_id}
                      type="button"
                      onClick={() => {
                        setSelection({ kind: "person", personId: p.person_id });
                        setNameInput(p.name);
                        setMergeTarget("");
                      }}
                      className={`flex w-full items-center justify-between gap-2 rounded-md px-2 py-1.5 text-left text-xs transition ${
                        active
                          ? "bg-[#33363B] text-[#F3F4F6]"
                          : "text-[#D1D5DB] hover:bg-[#2C2F34]"
                      }`}
                    >
                      <span className="flex min-w-0 items-center gap-2">
                        {p.cover_photo_id && p.cover_face_idx !== null ? (
                          <img
                            src={photoFaceCropUrl(p.cover_photo_id, p.cover_face_idx, 64)}
                            alt={p.name}
                            className="h-7 w-7 shrink-0 rounded-full bg-[#1B1C1E] object-cover"
                          />
                        ) : (
                          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[#1B1C1E] text-[11px] text-[#9CA3AF]">
                            {p.name.slice(0, 1)}
                          </span>
                        )}
                        <span className="truncate font-medium">{p.name}</span>
                      </span>
                      <span className="shrink-0 text-[#9CA3AF] tabular-nums">{p.photo_count}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Special groups */}
          <div className="border-t border-[#33363B] pt-3">
            <div className="mb-2 px-1 text-[11px] font-medium uppercase tracking-wider text-[#9CA3AF]">
              {t.peopleSpecialSection}
            </div>
            <div className="space-y-1">
              <button
                type="button"
                onClick={() => {
                  setSelection({ kind: "group_photo" });
                  setMergeTarget("");
                }}
                className={`flex w-full items-center justify-between rounded-md px-2.5 py-2 text-xs ${
                  selection.kind === "group_photo"
                    ? "bg-[#33363B] font-medium text-[#F3F4F6]"
                    : "text-[#D1D5DB] hover:bg-[#2C2F34]"
                }`}
              >
                <span>{t.peopleGroupPhotos}</span>
                <span className="text-[#9CA3AF] tabular-nums">
                  {data?.group_photo_total_count ?? 0}
                </span>
              </button>
              <button
                type="button"
                onClick={() => {
                  setSelection({ kind: "unnamed" });
                  setMergeTarget("");
                }}
                className={`flex w-full items-center justify-between rounded-md px-2.5 py-2 text-xs ${
                  selection.kind === "unnamed"
                    ? "bg-[#33363B] font-medium text-[#F3F4F6]"
                    : "text-[#D1D5DB] hover:bg-[#2C2F34]"
                }`}
              >
                <span>{t.peopleUnnamed}</span>
                <span className="text-[#9CA3AF] tabular-nums">
                  {data?.unnamed_photos.length ?? 0}
                </span>
              </button>
              <button
                type="button"
                onClick={() => {
                  setSelection({ kind: "no_face" });
                  setMergeTarget("");
                }}
                className={`flex w-full items-center justify-between rounded-md px-2.5 py-2 text-xs ${
                  selection.kind === "no_face"
                    ? "bg-[#33363B] font-medium text-[#F3F4F6]"
                    : "text-[#D1D5DB] hover:bg-[#2C2F34]"
                }`}
              >
                <span>{t.peopleNoFace}</span>
                <span className="text-[#9CA3AF] tabular-nums">
                  {data?.no_face_photos.length ?? 0}
                </span>
              </button>
            </div>
          </div>

          {/* Unnamed clusters */}
          {(data?.unnamed_clusters.length ?? 0) > 0 && (
            <div className="border-t border-[#33363B] pt-3">
              <div className="mb-2 px-1 text-[11px] font-medium uppercase tracking-wider text-[#9CA3AF]">
                {t.peopleClustersSection} ({data?.unnamed_clusters.length ?? 0})
              </div>
              <div className="max-h-72 space-y-1 overflow-y-auto">
                {(data?.unnamed_clusters ?? []).map((clu, idx) => {
                  const active =
                    selection.kind === "cluster" && selection.clusterId === clu.cluster_id;
                  return (
                    <button
                      key={clu.cluster_id}
                      type="button"
                      onClick={() => {
                        setSelection({ kind: "cluster", clusterId: clu.cluster_id });
                        setNameInput("");
                        setMergeTarget("");
                      }}
                      className={`flex w-full items-center justify-between gap-2 rounded-md px-2 py-1.5 text-left text-xs ${
                        active
                          ? "bg-[#33363B] text-[#F3F4F6]"
                          : "text-[#D1D5DB] hover:bg-[#2C2F34]"
                      }`}
                    >
                      <span className="flex min-w-0 items-center gap-2">
                        <img
                          src={photoFaceCropUrl(clu.cover_photo_id, clu.cover_face_idx, 64)}
                          alt=""
                          className="h-7 w-7 shrink-0 rounded-full bg-[#1B1C1E] object-cover"
                        />
                        <span className="truncate">
                          {t.peopleClusterLabel(idx + 1, clu.face_count)}
                        </span>
                      </span>
                      <span className="shrink-0 text-[#9CA3AF] tabular-nums">
                        {clu.photos.length}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </aside>

        {/* Main content area */}
        <section className="min-w-0 space-y-4">
          {/* Active group toolbar */}
          <div className="flex flex-col gap-3 rounded-lg border border-[#33363B] bg-[#242629] p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2.5">
                <h2 className="text-base font-medium text-[#F3F4F6]">{activeTitle}</h2>
                <span className="rounded bg-[#1B1C1E] px-2 py-0.5 text-xs text-[#9CA3AF] tabular-nums">
                  {t.peopleFullCount(activePhotos.length)}
                </span>
              </div>
            </div>

            {/* Group photo single-person filter chips */}
            {selection.kind === "group_photo" && (data?.people.length ?? 0) > 0 && (
              <div className="flex flex-wrap items-center gap-1.5">
                <button
                  type="button"
                  onClick={() => setGroupPersonFilter("")}
                  className={`rounded-full px-3 py-1 text-xs ${
                    groupPersonFilter === ""
                      ? "bg-[#5B8DEF] font-medium text-white"
                      : "border border-[#33363B] bg-[#1B1C1E] text-[#D1D5DB] hover:bg-[#2C2F34]"
                  }`}
                >
                  {t.peopleFilterAllGroups}
                </button>
                {(data?.people ?? []).map((p) => (
                  <button
                    key={p.person_id}
                    type="button"
                    onClick={() =>
                      setGroupPersonFilter(groupPersonFilter === p.person_id ? "" : p.person_id)
                    }
                    className={`rounded-full px-3 py-1 text-xs ${
                      groupPersonFilter === p.person_id
                        ? "bg-[#5B8DEF] font-medium text-white"
                        : "border border-[#33363B] bg-[#1B1C1E] text-[#D1D5DB] hover:bg-[#2C2F34]"
                    }`}
                  >
                    {t.peopleFilterContains(p.name)}
                    {p.group_photo_count > 0 ? ` (${p.group_photo_count})` : ""}
                  </button>
                ))}
              </div>
            )}

            {/* Unnamed cluster controls: Name / Merge / Ignore */}
            {selectedCluster && (
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <form
                  onSubmit={(e) => void handleNameCluster(selectedCluster.cluster_id, e)}
                  className="flex items-center gap-1.5"
                >
                  <input
                    type="text"
                    value={nameInput}
                    onChange={(e) => setNameInput(e.target.value)}
                    placeholder={t.peopleNamePlaceholder}
                    className="w-48 rounded-md border border-[#33363B] bg-[#1B1C1E] px-2.5 py-1.5 text-xs text-[#F3F4F6] placeholder-[#6B7280] focus:border-[#5B8DEF] focus:outline-none"
                  />
                  <button
                    type="submit"
                    disabled={mutating || !nameInput.trim()}
                    className="rounded-md bg-[#F3F4F6] px-3 py-1.5 text-xs font-medium text-[#1B1C1E] hover:bg-white disabled:opacity-50"
                  >
                    {t.peopleNameAction}
                  </button>
                </form>

                <div className="flex items-center gap-1.5">
                  <select
                    value={mergeTarget}
                    onChange={(e) => setMergeTarget(e.target.value)}
                    className="rounded-md border border-[#33363B] bg-[#1B1C1E] px-2 py-1.5 text-xs text-[#E5E7EB]"
                  >
                    <option value="">{t.peopleMergePlaceholder}</option>
                    {(data?.people ?? []).map((p) => (
                      <option key={p.person_id} value={`person:${p.person_id}`}>
                        {p.name}
                      </option>
                    ))}
                    {(data?.unnamed_clusters ?? [])
                      .filter((c) => c.cluster_id !== selectedCluster.cluster_id)
                      .map((c, idx) => (
                        <option key={c.cluster_id} value={`cluster:${c.cluster_id}`}>
                          {t.peopleClusterLabel(idx + 1, c.face_count)}
                        </option>
                      ))}
                  </select>
                  <button
                    type="button"
                    disabled={mutating || !mergeTarget}
                    onClick={() => void handleMerge({ clusterId: selectedCluster.cluster_id })}
                    className="rounded-md border border-[#33363B] bg-[#1B1C1E] px-2.5 py-1.5 text-xs text-[#E5E7EB] hover:bg-[#2C2F34] disabled:opacity-50"
                  >
                    {t.peopleMergeAction}
                  </button>
                </div>

                <button
                  type="button"
                  disabled={mutating}
                  onClick={() => void handleIgnoreCluster(selectedCluster.cluster_id)}
                  className="rounded-md border border-[#DC4C46]/50 bg-[#DC4C46]/10 px-2.5 py-1.5 text-xs text-[#FCA5A5] hover:bg-[#DC4C46]/20 disabled:opacity-50"
                >
                  {t.peopleIgnoreCluster}
                </button>
              </div>
            )}

            {/* Named person controls: Rename / Merge into another person */}
            {selectedPerson && (
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <form
                  onSubmit={(e) => void handleRenamePerson(selectedPerson.person_id, e)}
                  className="flex items-center gap-1.5"
                >
                  <input
                    type="text"
                    value={nameInput}
                    onChange={(e) => setNameInput(e.target.value)}
                    placeholder={t.peopleNamePlaceholder}
                    className="w-44 rounded-md border border-[#33363B] bg-[#1B1C1E] px-2.5 py-1.5 text-xs text-[#F3F4F6] focus:border-[#5B8DEF] focus:outline-none"
                  />
                  <button
                    type="submit"
                    disabled={mutating || !nameInput.trim()}
                    className="rounded-md border border-[#33363B] bg-[#1B1C1E] px-3 py-1.5 text-xs text-[#E5E7EB] hover:bg-[#2C2F34] disabled:opacity-50"
                  >
                    {t.peopleRenameAction}
                  </button>
                </form>

                {(data?.people.length ?? 0) > 1 && (
                  <div className="flex items-center gap-1.5">
                    <select
                      value={mergeTarget}
                      onChange={(e) => setMergeTarget(e.target.value)}
                      className="rounded-md border border-[#33363B] bg-[#1B1C1E] px-2 py-1.5 text-xs text-[#E5E7EB]"
                    >
                      <option value="">{t.peopleMergePlaceholder}</option>
                      {(data?.people ?? [])
                        .filter((p) => p.person_id !== selectedPerson.person_id)
                        .map((p) => (
                          <option key={p.person_id} value={`person:${p.person_id}`}>
                            {p.name}
                          </option>
                        ))}
                    </select>
                    <button
                      type="button"
                      disabled={mutating || !mergeTarget}
                      onClick={() => void handleMerge({ personId: selectedPerson.person_id })}
                      className="rounded-md border border-[#33363B] bg-[#1B1C1E] px-2.5 py-1.5 text-xs text-[#E5E7EB] hover:bg-[#2C2F34] disabled:opacity-50"
                    >
                      {t.peopleMergeAction}
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Photo grid (full untruncated list) */}
          {peopleQ.isLoading ? (
            <p className="py-10 text-sm text-[#9CA3AF]">{t.loading}</p>
          ) : activePhotos.length === 0 ? (
            <div className="rounded-lg border border-dashed border-[#33363B] py-14 text-center">
              <p className="text-base text-[#F3F4F6]">{t.peopleEmptyGroup}</p>
              <p className="mt-1 text-sm text-[#9CA3AF]">{t.peopleEmptyHint}</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-4">
              {activePhotos.map((item) => (
                <div
                  key={item.photo_id}
                  className="group flex flex-col overflow-hidden rounded-lg border border-[#33363B] bg-[#242629]"
                >
                  <div className="relative aspect-[4/3] w-full bg-[#1B1C1E]">
                    <img
                      src={photoImageUrl(item.photo_id, "grid")}
                      alt={item.stem}
                      loading="lazy"
                      className="h-full w-full object-cover"
                    />
                    <button
                      type="button"
                      onClick={() => onOpenReview(item.session_id)}
                      className="absolute right-2 top-2 rounded bg-[#1B1C1E]/85 px-2 py-0.5 text-[11px] text-[#E5E7EB] opacity-0 transition hover:bg-[#2C2F34] group-hover:opacity-100 focus:opacity-100"
                    >
                      {t.searchOpenSession}
                    </button>
                  </div>

                  <div className="flex flex-1 flex-col justify-between gap-1.5 p-2.5">
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate text-xs font-medium text-[#F3F4F6]">
                        {item.stem}
                      </span>
                      <span className="shrink-0 text-[11px] text-[#9CA3AF]">
                        {item.session_name}
                      </span>
                    </div>

                    {item.faces.length > 0 && (
                      <div className="flex flex-wrap items-center gap-1">
                        {item.faces.map((f) => (
                          <span
                            key={f.face_idx}
                            className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] ${
                              f.status === "ignored"
                                ? "bg-[#1B1C1E] text-[#6B7280] line-through"
                                : f.person_name
                                  ? "bg-[#3D9A6B]/20 text-[#A7F3D0]"
                                  : "bg-[#5B8DEF]/20 text-[#93C5FD]"
                            }`}
                          >
                            <span>
                              {f.status === "ignored"
                                ? t.peopleIgnoredBadge
                                : (f.person_name ?? `${t.peopleUnnamedFace} #${f.face_idx + 1}`)}
                            </span>
                            {f.status !== "ignored" && (
                              <button
                                type="button"
                                disabled={mutating}
                                title={t.peopleIgnoreFace}
                                onClick={() => void handleIgnoreFace(f.content_fp, f.face_idx)}
                                className="ml-0.5 text-[#9CA3AF] hover:text-[#FCA5A5]"
                              >
                                ×
                              </button>
                            )}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>

      {showExportModal && data && (
        <ExportPeopleModal
          data={data}
          sessionId={sessionId}
          onClose={() => setShowExportModal(false)}
          onSuccess={(msg) => setNotice(msg)}
        />
      )}
    </div>
  );
}
