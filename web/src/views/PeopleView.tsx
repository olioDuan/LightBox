import { useEffect, useMemo, useRef, useState, type DragEvent, type FormEvent, type ReactNode } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  photoFaceCropUrl,
  photoImageUrl,
  type FaceSlot,
  type PeopleGroupsResponse,
  type PeoplePhotoItem,
  type SimilarFaceCandidate,
} from "../api";
import {
  Button,
  Dialog,
  EmptyState,
  Icon,
  Loading,
  Menu,
  Notice,
  NoticeDock,
  Tabs,
  type MenuItem,
} from "../components/ui";
import { useI18n } from "../i18n";
import {
  Check,
  ChevronDown,
  EyeOff,
  Export,
  Folder,
  Link,
  Scissor,
  Scan,
  Users,
} from "reicon-react";

/** Drop the parenthetical explanation some labels carry. */
function headline(label: string): string {
  return label.split(/\s*[（(]/)[0];
}

function Avatar({ src, label, size = 28 }: { src: string | null; label: string; size?: number }) {
  const [failed, setFailed] = useState(false);
  return src && !failed ? (
    <img
      src={src}
      alt=""
      onError={() => setFailed(true)}
      className="shrink-0 rounded-full bg-table object-cover"
      style={{ width: size, height: size }}
    />
  ) : (
    <span
      className="flex shrink-0 items-center justify-center rounded-full bg-table text-xs font-medium text-ink-2"
      style={{ width: size, height: size }}
    >
      {label.slice(0, 1)}
    </span>
  );
}

function CheckDot({ checked }: { checked: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-[6px] transition-colors ${
        checked ? "bg-ink text-white" : "shadow-[inset_0_0_0_1.5px_var(--color-ink-3)]"
      }`}
    >
      {checked && <Check size={14} strokeWidth={2} />}
    </span>
  );
}

function FaceChip({
  label,
  tone,
  open,
  disabled,
  onToggle,
  onClose,
  menuItems,
  menuHeader,
}: {
  label: string;
  tone: "named" | "unnamed" | "ignored";
  open: boolean;
  disabled: boolean;
  onToggle: () => void;
  onClose: () => void;
  menuItems: MenuItem[];
  menuHeader: ReactNode;
}) {
  const ref = useRef<HTMLButtonElement | null>(null);
  return (
    <>
      <button
        ref={ref}
        type="button"
        disabled={disabled}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={onToggle}
        className={`inline-flex h-8 items-center gap-1 rounded-chip pl-3 pr-1 text-sm transition-colors ${
          tone === "ignored"
            ? "cursor-default text-ink-3 line-through"
            : tone === "named"
              ? `font-medium text-ink ${open ? "bg-table-2" : "bg-table hover:bg-table-2"}`
              : `text-ink-2 ${open ? "bg-table-2" : "bg-table hover:bg-table-2 hover:text-ink"}`
        }`}
      >
        <span className="max-w-[10rem] truncate">{label}</span>
        {tone !== "ignored" && <ChevronDown size={16} strokeWidth={1.5} className="text-ink-3" />}
      </button>
      {tone !== "ignored" && (
        <Menu
          anchorRef={ref}
          open={open}
          onClose={onClose}
          items={menuItems}
          header={menuHeader}
          width={256}
        />
      )}
    </>
  );
}

interface PeopleViewProps {
  onOpenReview: (sessionId: string) => void;
}

type ActiveSelection =
  | { kind: "person"; personId: string }
  | { kind: "cluster"; clusterId: string }
  | { kind: "group_photo" }
  | { kind: "no_face" };

type EntityRef = { kind: "person"; id: string } | { kind: "cluster"; id: string };

function entityKey(ref: EntityRef): string {
  return `${ref.kind}:${ref.id}`;
}

function parseEntityKey(key: string): EntityRef | null {
  const [kind, id] = key.split(":", 2);
  if ((kind === "person" || kind === "cluster") && id) {
    return { kind, id };
  }
  return null;
}

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
    <Dialog labelledBy="export-title" width={560} onDismiss={busy ? undefined : onClose}>
      <form onSubmit={(e) => void handleExport(e)} className="flex min-h-0 flex-col p-8 sm:p-12">
        <h2 id="export-title" className="text-lg font-semibold text-ink" title={t.peopleExportModalDesc}>
          {t.peopleExportModalTitle}
        </h2>

        <div className="mt-8 flex gap-2">
          <input
            type="text"
            value={targetDir}
            onChange={(e) => setTargetDir(e.target.value)}
            placeholder={t.peopleExportTargetPlaceholder}
            aria-label={t.peopleExportTargetLabel}
            className="field min-w-0 flex-1"
          />
          <Button variant="secondary" icon={Folder} onClick={() => void handlePickFolder()}>
            {t.peopleExportPickFolder}
          </Button>
        </div>

        <div
          role="group"
          aria-label={t.peopleExportSelectGroups}
          className="-mx-2 mt-8 max-h-64 overflow-y-auto"
        >
          {exportableOptions.map((opt) => {
            const checked = selectedKeys.includes(opt.key);
            return (
              <label
                key={opt.key}
                className="flex h-12 cursor-pointer items-center justify-between gap-4 rounded-btn px-2 text-sm hover:bg-table"
              >
                <span className="flex min-w-0 items-center gap-4">
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleKey(opt.key)}
                    className="peer sr-only"
                  />
                  <CheckDot checked={checked} />
                  <span className={`truncate ${checked ? "font-medium text-ink" : "text-ink-2"}`}>
                    {opt.label}
                  </span>
                </span>
                <span className="shrink-0 text-ink-3 tabular-nums">{t.photoCount(opt.count)}</span>
              </label>
            );
          })}
        </div>

        {error && (
          <p className="mt-6 text-sm text-mark">
            {t.errorPrefix}
            {error}
          </p>
        )}

        <div className="mt-8 flex justify-end gap-2" title={t.peopleExportSafety}>
          <Button variant="ghost" onClick={onClose}>
            {t.modalCancel}
          </Button>
          <Button
            type="submit"
            variant="primary"
            icon={Export}
            disabled={busy || !targetDir.trim() || selectedKeys.length === 0}
          >
            {busy ? t.peopleExporting : t.peopleExportConfirm}
          </Button>
        </div>
      </form>
    </Dialog>
  );
}

export function PeopleView({ onOpenReview }: PeopleViewProps) {
  const { t } = useI18n();
  const queryClient = useQueryClient();

  const [sessionId, setSessionId] = useState("");
  const [groupPersonFilter, setGroupPersonFilter] = useState<string>("");
  const [groupHeadcountFilter, setGroupHeadcountFilter] = useState<number | "5+" | null>(null);
  const [selection, setSelection] = useState<ActiveSelection>({ kind: "group_photo" });

  const [nameInput, setNameInput] = useState("");
  const [mergeTarget, setMergeTarget] = useState("");
  const [batchMergeMode, setBatchMergeMode] = useState(false);
  const [batchSelectedKeys, setBatchSelectedKeys] = useState<string[]>([]);
  const [draggedKey, setDraggedKey] = useState<string | null>(null);
  const [dropHoverKey, setDropHoverKey] = useState<string | null>(null);
  const [activeFaceMenu, setActiveFaceMenu] = useState<{
    photoId: string;
    faceIdx: number;
  } | null>(null);

  const [indexing, setIndexing] = useState(false);
  const [mutating, setMutating] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Close inline face correction popover on Escape
  useEffect(() => {
    if (!activeFaceMenu) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setActiveFaceMenu(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [activeFaceMenu]);

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

  // Consistent 1-based numbering for unnamed faces (`人脸 1`, `人脸 2`, ...) across sidebar, dropdowns, and cards
  const clusterIndexById = useMemo(() => {
    const map = new Map<string, number>();
    (data?.unnamed_clusters ?? []).forEach((clu, idx) => {
      map.set(clu.cluster_id, idx + 1);
    });
    return map;
  }, [data?.unnamed_clusters]);

  const clusterById = useMemo(() => {
    const map = new Map<string, PeopleGroupsResponse["unnamed_clusters"][number]>();
    for (const clu of data?.unnamed_clusters ?? []) {
      map.set(clu.cluster_id, clu);
    }
    return map;
  }, [data?.unnamed_clusters]);

  const personById = useMemo(() => {
    const map = new Map<string, PeopleGroupsResponse["people"][number]>();
    for (const p of data?.people ?? []) {
      map.set(p.person_id, p);
    }
    return map;
  }, [data?.people]);

  function getEntityLabel(ref: EntityRef): string {
    if (ref.kind === "person") {
      return personById.get(ref.id)?.name ?? ref.id;
    }
    const idx = clusterIndexById.get(ref.id);
    return idx ? t.peopleClusterLabel(idx) : t.peopleUnnamedFace;
  }

  function getCandidateLabel(cand: SimilarFaceCandidate): string {
    if (cand.kind === "person") {
      return cand.name ?? personById.get(cand.target_id)?.name ?? cand.target_id;
    }
    const idx = clusterIndexById.get(cand.target_id);
    return idx ? t.peopleClusterLabel(idx) : t.peopleUnnamedFace;
  }

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
      const res = (await api.labelPeople({
        action: "name",
        name: clean,
        cluster_id: clusterId,
      })) as { person?: { id?: string } };
      setNameInput("");
      setNotice(t.peopleNameSuccess(clean));
      await refreshPeople();
      if (res.person?.id) {
        setSelection({ kind: "person", personId: res.person.id });
      } else {
        setSelection({ kind: "group_photo" });
      }
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

  async function performPairMerge(source: EntityRef, target: EntityRef) {
    if (source.kind === target.kind && source.id === target.id) return;
    // Always preserve named person if one side is named and the other is an unnamed face
    let finalSource = source;
    let finalTarget = target;
    if (source.kind === "person" && target.kind === "cluster") {
      finalSource = target;
      finalTarget = source;
    }

    setMutating(true);
    setError(null);
    setNotice(null);
    setActiveFaceMenu(null);
    try {
      if (finalTarget.kind === "person") {
        await api.labelPeople({
          action: "merge",
          source_person_id: finalSource.kind === "person" ? finalSource.id : undefined,
          source_cluster_id: finalSource.kind === "cluster" ? finalSource.id : undefined,
          target_person_id: finalTarget.id,
        });
        setSelection({ kind: "person", personId: finalTarget.id });
      } else {
        await api.labelPeople({
          action: "merge",
          source_cluster_id: finalSource.id,
          target_cluster_id: finalTarget.id,
        });
        setSelection({ kind: "cluster", clusterId: finalTarget.id });
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

  async function handleBatchMergeConfirm() {
    const refs = batchSelectedKeys
      .map(parseEntityKey)
      .filter((r): r is EntityRef => r !== null);
    if (refs.length < 2) return;

    // Prefer the first named person as target so we keep their name; otherwise the first selected face
    const target = refs.find((r) => r.kind === "person") ?? refs[0];
    const sources = refs.filter((r) => !(r.kind === target.kind && r.id === target.id));
    const sourcePersonIds = sources.filter((r) => r.kind === "person").map((r) => r.id);
    const sourceClusterIds = sources.filter((r) => r.kind === "cluster").map((r) => r.id);

    setMutating(true);
    setError(null);
    setNotice(null);
    try {
      if (target.kind === "person") {
        await api.labelPeople({
          action: "merge",
          source_person_ids: sourcePersonIds.length > 0 ? sourcePersonIds : undefined,
          source_cluster_ids: sourceClusterIds.length > 0 ? sourceClusterIds : undefined,
          target_person_id: target.id,
        });
        setSelection({ kind: "person", personId: target.id });
      } else {
        await api.labelPeople({
          action: "merge",
          source_cluster_ids: sourceClusterIds,
          target_cluster_id: target.id,
        });
        setSelection({ kind: "cluster", clusterId: target.id });
      }
      setBatchSelectedKeys([]);
      setBatchMergeMode(false);
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
      const remaining = (data?.unnamed_clusters ?? []).filter((c) => c.cluster_id !== clusterId);
      if (remaining.length > 0) {
        setSelection({ kind: "cluster", clusterId: remaining[0].cluster_id });
      } else {
        setSelection({ kind: "group_photo" });
      }
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
    setActiveFaceMenu(null);
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

  async function handleSplitFace(contentFp: string, faceIdx: number) {
    setMutating(true);
    setError(null);
    setNotice(null);
    setActiveFaceMenu(null);
    try {
      await api.labelPeople({
        action: "split",
        faces: [{ content_fp: contentFp, face_idx: faceIdx }],
      });
      setNotice(t.peopleFaceSplitSuccess);
      await refreshPeople();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setMutating(false);
    }
  }

  async function handleReassignFace(contentFp: string, faceIdx: number, target: EntityRef) {
    setMutating(true);
    setError(null);
    setNotice(null);
    setActiveFaceMenu(null);
    try {
      await api.labelPeople({
        action: "reassign",
        faces: [{ content_fp: contentFp, face_idx: faceIdx }],
        target_person_id: target.kind === "person" ? target.id : undefined,
        target_cluster_id: target.kind === "cluster" ? target.id : undefined,
      });
      setNotice(t.peopleFaceReassignSuccess(getEntityLabel(target)));
      await refreshPeople();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setMutating(false);
    }
  }

  function toggleBatchSelect(key: string) {
    setBatchSelectedKeys((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
    );
  }

  function handleEntityDragStart(e: DragEvent, ref: EntityRef) {
    const key = entityKey(ref);
    setDraggedKey(key);
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", key);
  }

  function handleEntityDragOver(e: DragEvent, ref: EntityRef) {
    const targetKey = entityKey(ref);
    if (!draggedKey || draggedKey === targetKey) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    if (dropHoverKey !== targetKey) {
      setDropHoverKey(targetKey);
    }
  }

  function handleEntityDrop(e: DragEvent, targetRef: EntityRef) {
    e.preventDefault();
    const raw = draggedKey ?? e.dataTransfer.getData("text/plain");
    setDraggedKey(null);
    setDropHoverKey(null);
    const sourceRef = raw ? parseEntityKey(raw) : null;
    if (!sourceRef) return;
    void performPairMerge(sourceRef, targetRef);
  }

  function handleEntityDragEnd() {
    setDraggedKey(null);
    setDropHoverKey(null);
  }

  // Distinct headcount filter options present in group photos (`2`, `3`, `4`, `5+`)
  const availableHeadcounts = useMemo(() => {
    const exact = new Set<number>();
    let hasFivePlus = false;
    for (const item of data?.group_photos ?? []) {
      if (item.valid_face_count >= 5) {
        hasFivePlus = true;
      } else if (item.valid_face_count >= 2) {
        exact.add(item.valid_face_count);
      }
    }
    const opts: Array<number | "5+"> = [...Array.from(exact).sort((a, b) => a - b)];
    if (hasFivePlus) opts.push("5+");
    return opts;
  }, [data?.group_photos]);

  // Resolve currently active group photos, title, and similar-face suggestions
  let activeTitle = t.peopleGroupPhotos;
  let activePhotos: PeoplePhotoItem[] = [];
  let activeSimilar: SimilarFaceCandidate[] = [];
  let activeEntityRef: EntityRef | null = null;

  const selectedPerson =
    selection.kind === "person" ? personById.get(selection.personId) : undefined;
  const selectedCluster =
    selection.kind === "cluster" ? clusterById.get(selection.clusterId) : undefined;
  const selectedClusterIdx =
    selectedCluster ? (clusterIndexById.get(selectedCluster.cluster_id) ?? 1) : -1;

  if (data) {
    if (selectedPerson) {
      activeTitle = selectedPerson.name;
      activePhotos = selectedPerson.photos;
      activeSimilar = selectedPerson.similar_candidates ?? [];
      activeEntityRef = { kind: "person", id: selectedPerson.person_id };
    } else if (selectedCluster) {
      activeTitle = t.peopleClusterLabel(selectedClusterIdx);
      activePhotos = selectedCluster.photos;
      activeSimilar = selectedCluster.similar_candidates ?? [];
      activeEntityRef = { kind: "cluster", id: selectedCluster.cluster_id };
    } else if (selection.kind === "no_face") {
      activeTitle = t.peopleNoFace;
      activePhotos = data.no_face_photos;
    } else {
      activeTitle = t.peopleGroupPhotos;
      activePhotos = data.group_photos.filter((item) => {
        if (groupHeadcountFilter === null) return true;
        if (groupHeadcountFilter === "5+") return item.valid_face_count >= 5;
        return item.valid_face_count === groupHeadcountFilter;
      });
    }
  }

  const totalMergeableEntities =
    (data?.people.length ?? 0) + (data?.unnamed_clusters.length ?? 0);

  function renderFaceLabel(f: FaceSlot): string {
    if (f.status === "ignored") return t.peopleIgnoredBadge;
    if (f.person_name) return f.person_name;
    if (f.cluster_id) {
      const idx = clusterIndexById.get(f.cluster_id);
      if (idx) return t.peopleClusterLabel(idx);
    }
    return t.peopleClusterLabel(f.face_idx + 1);
  }

  const renderEntityRow = (
    ref: EntityRef,
    label: string,
    count: number,
    avatar: string | null,
    active: boolean,
    onSelect: () => void,
  ) => {
    const key = entityKey(ref);
    const checked = batchSelectedKeys.includes(key);
    const isDropTarget = dropHoverKey === key;
    return (
      <div
        key={key}
        draggable={!batchMergeMode && !mutating}
        onDragStart={(e) => handleEntityDragStart(e, ref)}
        onDragOver={(e) => handleEntityDragOver(e, ref)}
        onDragLeave={() => {
          if (dropHoverKey === key) setDropHoverKey(null);
        }}
        onDrop={(e) => handleEntityDrop(e, ref)}
        onDragEnd={handleEntityDragEnd}
        className={`rounded-btn transition-shadow ${
          isDropTarget ? "shadow-[inset_0_0_0_2px_var(--color-ink)]" : ""
        } ${draggedKey === key ? "opacity-40" : ""}`}
      >
        <button
          type="button"
          aria-pressed={batchMergeMode ? checked : active}
          onClick={() => {
            if (batchMergeMode) {
              toggleBatchSelect(key);
              return;
            }
            onSelect();
          }}
          className={`flex h-12 w-full items-center justify-between gap-2 rounded-btn px-2 text-left text-sm transition-colors ${
            active || checked ? "bg-table text-ink" : "text-ink-2 hover:bg-table hover:text-ink"
          } ${batchMergeMode ? "" : "cursor-grab active:cursor-grabbing"}`}
        >
          <span className="flex min-w-0 items-center gap-2">
            {batchMergeMode && <CheckDot checked={checked} />}
            <Avatar src={avatar} label={label} />
            <span className={`truncate ${active || checked ? "font-medium" : ""}`}>
              {isDropTarget ? t.peopleDropToMerge(label) : label}
            </span>
          </span>
          <span className="shrink-0 text-ink-3 tabular-nums">{count}</span>
        </button>
      </div>
    );
  };

  const specialRow = (kind: "group_photo" | "no_face", label: string, count: number) => {
    const active = selection.kind === kind;
    return (
      <button
        type="button"
        aria-pressed={active}
        onClick={() => {
          setSelection({ kind });
          setMergeTarget("");
          setActiveFaceMenu(null);
        }}
        className={`flex h-12 w-full items-center justify-between gap-2 rounded-btn px-2 text-left text-sm transition-colors ${
          active ? "bg-table font-medium text-ink" : "text-ink-2 hover:bg-table hover:text-ink"
        }`}
      >
        <span className="flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center">
            <Icon as={kind === "group_photo" ? Users : Scan} />
          </span>
          {label}
        </span>
        <span className="text-ink-3 tabular-nums">{count}</span>
      </button>
    );
  };

  const mergeOptions = (excludeKey: string) => [
    ...(data?.people ?? [])
      .filter((p) => `person:${p.person_id}` !== excludeKey)
      .map((p) => ({ value: `person:${p.person_id}`, label: p.name })),
    ...(data?.unnamed_clusters ?? [])
      .filter((c) => `cluster:${c.cluster_id}` !== excludeKey)
      .map((c) => {
        const num = clusterIndexById.get(c.cluster_id) ?? 1;
        return {
          value: `cluster:${c.cluster_id}`,
          label: `${t.peopleClusterLabel(num)} (${t.photoCount(c.photos.length)})`,
        };
      }),
  ];

  const renderMergeControl = (source: EntityRef) => (
    <div className="flex items-center gap-2">
      <select
        value={mergeTarget}
        onChange={(e) => setMergeTarget(e.target.value)}
        aria-label={t.peopleMergePlaceholder}
        className={`field ${mergeTarget ? "" : "text-ink-2"}`}
      >
        <option value="">{t.peopleMergePlaceholder}</option>
        {mergeOptions(entityKey(source)).map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      {mergeTarget && (
        <Button
          variant="secondary"
          icon={Link}
          disabled={mutating}
          className="fade-enter"
          onClick={() => {
            const targetRef = parseEntityKey(mergeTarget);
            if (targetRef) void performPairMerge(source, targetRef);
          }}
        >
          {t.peopleMergeAction}
        </Button>
      )}
    </div>
  );

  return (
    <div className="mx-auto max-w-6xl px-6 pb-32 pt-12 sm:px-12">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-lg font-semibold text-ink">{t.peopleTitle}</h1>
        <div className="flex flex-wrap items-center gap-2">
          <select
            aria-label={t.searchAllSessions}
            value={sessionId}
            onChange={(e) => {
              setSessionId(e.target.value);
              setGroupHeadcountFilter(null);
            }}
            className={`field ${sessionId ? "font-medium" : "text-ink-2"}`}
          >
            <option value="">{t.searchAllSessions}</option>
            {sessions.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
          <Button
            variant="secondary"
            icon={Scan}
            disabled={indexing || sessions.length === 0}
            onClick={() => void handleIndexPeople()}
          >
            {indexing ? t.peopleIndexing : t.peopleIndexAction}
          </Button>
          <Button
            variant="primary"
            icon={Export}
            disabled={!data || data.total_kept_photos === 0}
            onClick={() => setShowExportModal(true)}
          >
            {t.peopleExportAction}
          </Button>
        </div>
      </div>

      <div className="mt-12 grid grid-cols-1 gap-12 lg:grid-cols-[248px_1fr]">
        {/* Sidebar: people, faces waiting for a name, special groups — separated by space only */}
        <aside className="flex flex-col gap-8 lg:sticky lg:top-24 lg:max-h-[calc(100vh-128px)] lg:self-start lg:overflow-y-auto">
          {totalMergeableEntities >= 2 && (
            <div className="flex flex-col gap-2" title={batchMergeMode ? t.peopleBatchMergeHint : t.peopleDragHint}>
              <Button
                variant={batchMergeMode ? "secondary" : "ghost"}
                size="sm"
                icon={Link}
                className="self-start"
                onClick={() => {
                  setBatchMergeMode((m) => !m);
                  setBatchSelectedKeys([]);
                }}
              >
                {batchMergeMode ? t.peopleBatchMergeCancel : t.peopleBatchMergeStart}
              </Button>
              {batchMergeMode && (
                <Button
                  variant="primary"
                  disabled={mutating || batchSelectedKeys.length < 2}
                  onClick={() => void handleBatchMergeConfirm()}
                >
                  {t.peopleBatchMergeConfirm(batchSelectedKeys.length)}
                </Button>
              )}
            </div>
          )}

          {(data?.people.length ?? 0) > 0 && (
            <div className="flex flex-col" aria-label={t.peopleNamedSection}>
              {(data?.people ?? []).map((p) =>
                renderEntityRow(
                  { kind: "person", id: p.person_id },
                  p.name,
                  p.photo_count,
                  p.cover_photo_id && p.cover_face_idx !== null
                    ? photoFaceCropUrl(p.cover_photo_id, p.cover_face_idx, 64)
                    : null,
                  !batchMergeMode && selection.kind === "person" && selection.personId === p.person_id,
                  () => {
                    setSelection({ kind: "person", personId: p.person_id });
                    setNameInput(p.name);
                    setMergeTarget("");
                    setActiveFaceMenu(null);
                  },
                ),
              )}
            </div>
          )}

          {(data?.unnamed_clusters.length ?? 0) > 0 && (
            <div className="flex flex-col" aria-label={t.peopleClustersSection}>
              {(data?.unnamed_clusters ?? []).map((clu, idx) =>
                renderEntityRow(
                  { kind: "cluster", id: clu.cluster_id },
                  t.peopleClusterLabel(idx + 1),
                  clu.photos.length,
                  photoFaceCropUrl(clu.cover_photo_id, clu.cover_face_idx, 64),
                  !batchMergeMode && selection.kind === "cluster" && selection.clusterId === clu.cluster_id,
                  () => {
                    setSelection({ kind: "cluster", clusterId: clu.cluster_id });
                    setNameInput("");
                    setMergeTarget("");
                    setActiveFaceMenu(null);
                  },
                ),
              )}
            </div>
          )}

          <div className="flex flex-col" aria-label={t.peopleSpecialSection}>
            {specialRow("group_photo", t.peopleGroupPhotos, data?.group_photo_total_count ?? 0)}
            {specialRow("no_face", t.peopleNoFace, data?.no_face_photos.length ?? 0)}
          </div>
        </aside>

        {/* Main */}
        <section className="min-w-0">
          <div className="flex flex-col gap-6">
            <div className="flex items-baseline gap-4">
              <h2 className="truncate text-lg font-semibold text-ink">{activeTitle}</h2>
              <span className="shrink-0 text-sm text-ink-3 tabular-nums">
                {t.peopleFullCount(activePhotos.length)}
              </span>
            </div>

            {selection.kind === "group_photo" && (
              <div className="flex flex-col gap-2">
                {availableHeadcounts.length > 0 && (
                  <Tabs
                    size="sm"
                    value={groupHeadcountFilter === null ? "all" : String(groupHeadcountFilter)}
                    items={[
                      { id: "all", label: t.peopleHeadcountAll },
                      ...availableHeadcounts.map((hc) => ({
                        id: String(hc),
                        label: hc === "5+" ? t.peopleHeadcountFilterMore : t.peopleHeadcountFilter(hc),
                      })),
                    ]}
                    onChange={(id) => {
                      if (id === "all") {
                        setGroupHeadcountFilter(null);
                        return;
                      }
                      const hc: number | "5+" = id === "5+" ? "5+" : Number(id);
                      setGroupHeadcountFilter(groupHeadcountFilter === hc ? null : hc);
                    }}
                  />
                )}
                {(data?.people.length ?? 0) > 0 && (
                  <Tabs
                    size="sm"
                    value={groupPersonFilter || "all"}
                    items={[
                      { id: "all", label: t.peopleFilterAllGroups },
                      ...(data?.people ?? []).map((p) => ({
                        id: p.person_id,
                        label: t.peopleFilterContains(p.name),
                        count: p.group_photo_count > 0 ? p.group_photo_count : undefined,
                      })),
                    ]}
                    onChange={(id) => {
                      if (id === "all") {
                        setGroupPersonFilter("");
                        return;
                      }
                      setGroupPersonFilter(groupPersonFilter === id ? "" : id);
                    }}
                  />
                )}
              </div>
            )}

            {selectedCluster && (
              <div className="flex flex-wrap items-center gap-2">
                <form
                  onSubmit={(e) => void handleNameCluster(selectedCluster.cluster_id, e)}
                  className="flex items-center gap-2"
                >
                  <input
                    type="text"
                    value={nameInput}
                    onChange={(e) => setNameInput(e.target.value)}
                    placeholder={t.peopleNamePlaceholder}
                    aria-label={t.peopleNameAction}
                    className="field w-64"
                  />
                  <Button type="submit" variant="primary" disabled={mutating || !nameInput.trim()}>
                    {t.peopleNameAction}
                  </Button>
                </form>
                {renderMergeControl({ kind: "cluster", id: selectedCluster.cluster_id })}
                <Button
                  variant="ghost"
                  icon={EyeOff}
                  disabled={mutating}
                  onClick={() => void handleIgnoreCluster(selectedCluster.cluster_id)}
                >
                  {headline(t.peopleIgnoreCluster)}
                </Button>
              </div>
            )}

            {selectedPerson && (
              <div className="flex flex-wrap items-center gap-2">
                <form
                  onSubmit={(e) => void handleRenamePerson(selectedPerson.person_id, e)}
                  className="flex items-center gap-2"
                >
                  <input
                    type="text"
                    value={nameInput}
                    onChange={(e) => setNameInput(e.target.value)}
                    placeholder={t.peopleNamePlaceholder}
                    aria-label={t.peopleRenameAction}
                    className="field w-56"
                  />
                  {nameInput.trim() && nameInput.trim() !== selectedPerson.name && (
                    <Button type="submit" variant="secondary" disabled={mutating} className="fade-enter">
                      {t.peopleRenameAction}
                    </Button>
                  )}
                </form>
                {totalMergeableEntities > 1 &&
                  renderMergeControl({ kind: "person", id: selectedPerson.person_id })}
              </div>
            )}

            {activeEntityRef && activeSimilar.length > 0 && (
              <div className="flex flex-col gap-4">
                <p className="text-sm font-medium text-ink" title={t.peopleSimilarTitle}>
                  {headline(t.peopleSimilarTitle)}
                </p>
                <div className="flex flex-wrap gap-x-8 gap-y-4">
                  {activeSimilar.map((cand) => {
                    const label = getCandidateLabel(cand);
                    const pct = Math.max(1, Math.min(99, Math.round(cand.similarity * 100)));
                    return (
                      <div key={`${cand.kind}:${cand.target_id}`} className="flex items-center gap-4">
                        <Avatar
                          size={40}
                          label={label}
                          src={
                            cand.cover_photo_id && cand.cover_face_idx !== null
                              ? photoFaceCropUrl(cand.cover_photo_id, cand.cover_face_idx, 64)
                              : null
                          }
                        />
                        <div className="min-w-0 text-sm">
                          <div className="font-medium text-ink">{label}</div>
                          <div className="text-ink-3 tabular-nums">
                            {t.photoCount(cand.photo_count)}
                            <span className="ml-2">{t.peopleSimilarScore(pct)}</span>
                          </div>
                        </div>
                        <Button
                          variant="secondary"
                          size="sm"
                          disabled={mutating}
                          onClick={() => {
                            if (!activeEntityRef) return;
                            const candRef: EntityRef = { kind: cand.kind, id: cand.target_id };
                            void performPairMerge(candRef, activeEntityRef);
                          }}
                        >
                          {t.peopleSimilarMergeBtn}
                        </Button>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* Photo grid */}
          <div className="mt-12">
            {peopleQ.isLoading ? (
              <Loading label={t.loading} />
            ) : activePhotos.length === 0 ? (
              <EmptyState title={t.peopleEmptyGroup}>
                {sessions.length > 0 && (
                  <Button
                    variant="secondary"
                    icon={Scan}
                    disabled={indexing}
                    onClick={() => void handleIndexPeople()}
                  >
                    {indexing ? t.peopleIndexing : t.peopleIndexAction}
                  </Button>
                )}
              </EmptyState>
            ) : (
              <div className="grid grid-cols-1 gap-x-4 gap-y-8 sm:grid-cols-2 xl:grid-cols-3">
                {activePhotos.map((item) => (
                  <article key={item.photo_id} className="intent flex min-w-0 flex-col gap-2">
                    <div className="relative aspect-[4/3] w-full overflow-hidden rounded-tile bg-table">
                      <img
                        src={photoImageUrl(item.photo_id, "grid")}
                        alt={item.stem}
                        loading="lazy"
                        className="h-full w-full object-cover"
                      />
                      {item.valid_face_count >= 2 && (
                        <span className="absolute left-2 top-2 rounded-chip bg-paper/90 px-2 py-1 text-xs font-medium text-ink tabular-nums">
                          {t.peopleHeadcountBadge(item.valid_face_count)}
                        </span>
                      )}
                      <Button
                        size="sm"
                        variant="secondary"
                        className="reveal absolute right-2 top-2 bg-paper/90"
                        onClick={() => onOpenReview(item.session_id)}
                      >
                        {t.searchOpenSession}
                      </Button>
                    </div>

                    <div className="min-w-0 px-1 text-sm">
                      <div className="truncate font-medium text-ink" title={item.stem}>
                        {item.stem}
                      </div>
                      <div className="truncate text-ink-3">{item.session_name}</div>
                    </div>

                    {item.faces.length > 0 && (
                      <div className="flex flex-wrap items-center gap-1">
                        {item.faces.map((f) => {
                          const menuOpen =
                            activeFaceMenu?.photoId === item.photo_id &&
                            activeFaceMenu?.faceIdx === f.face_idx;
                          const canSplit =
                            f.person_id !== null ||
                            (f.cluster_id !== null &&
                              (clusterById.get(f.cluster_id)?.face_count ?? 0) > 1);
                          const reassignTargets: MenuItem[] = [
                            ...(data?.people ?? [])
                              .filter((p) => p.person_id !== f.person_id)
                              .map((p) => ({
                                key: `person:${p.person_id}`,
                                label: p.name,
                                icon: (
                                  <Avatar
                                    size={20}
                                    label={p.name}
                                    src={
                                      p.cover_photo_id && p.cover_face_idx !== null
                                        ? photoFaceCropUrl(p.cover_photo_id, p.cover_face_idx, 64)
                                        : null
                                    }
                                  />
                                ),
                                disabled: mutating,
                                onSelect: () =>
                                  void handleReassignFace(f.content_fp, f.face_idx, {
                                    kind: "person",
                                    id: p.person_id,
                                  }),
                              })),
                            ...(data?.unnamed_clusters ?? [])
                              .filter((c) => c.cluster_id !== f.cluster_id)
                              .map((c) => {
                                const num = clusterIndexById.get(c.cluster_id) ?? 1;
                                return {
                                  key: `cluster:${c.cluster_id}`,
                                  label: t.peopleClusterLabel(num),
                                  icon: (
                                    <Avatar
                                      size={20}
                                      label={t.peopleClusterLabel(num)}
                                      src={photoFaceCropUrl(c.cover_photo_id, c.cover_face_idx, 64)}
                                    />
                                  ),
                                  disabled: mutating,
                                  onSelect: () =>
                                    void handleReassignFace(f.content_fp, f.face_idx, {
                                      kind: "cluster",
                                      id: c.cluster_id,
                                    }),
                                };
                              }),
                          ];
                          const menuItems: MenuItem[] = [
                            ...(canSplit
                              ? [
                                  {
                                    key: "split",
                                    label: headline(t.peopleFaceSplitAction),
                                    icon: <Icon as={Scissor} />,
                                    disabled: mutating,
                                    onSelect: () => void handleSplitFace(f.content_fp, f.face_idx),
                                  },
                                ]
                              : []),
                            ...(totalMergeableEntities > 1 && reassignTargets.length > 0
                              ? [
                                  {
                                    key: "reassign",
                                    label: t.peopleFaceReassignTitle,
                                    icon: <Icon as={Users} />,
                                    submenu: reassignTargets,
                                  },
                                ]
                              : []),
                            {
                              key: "ignore",
                              label: t.peopleIgnoreFace,
                              icon: <Icon as={EyeOff} />,
                              tone: "mark" as const,
                              groupStart: true,
                              disabled: mutating,
                              onSelect: () => void handleIgnoreFace(f.content_fp, f.face_idx),
                            },
                          ];
                          return (
                            <FaceChip
                              key={f.face_idx}
                              label={renderFaceLabel(f)}
                              tone={
                                f.status === "ignored" ? "ignored" : f.person_name ? "named" : "unnamed"
                              }
                              open={menuOpen}
                              disabled={mutating || f.status === "ignored"}
                              onToggle={() =>
                                setActiveFaceMenu(
                                  menuOpen ? null : { photoId: item.photo_id, faceIdx: f.face_idx },
                                )
                              }
                              onClose={() => setActiveFaceMenu(null)}
                              menuItems={menuItems}
                              menuHeader={
                                <div className="flex items-center gap-2">
                                  <Avatar
                                    size={32}
                                    label={renderFaceLabel(f)}
                                    src={photoFaceCropUrl(item.photo_id, f.face_idx, 64)}
                                  />
                                  <span className="truncate text-sm font-medium text-ink">
                                    {renderFaceLabel(f)}
                                  </span>
                                </div>
                              }
                            />
                          );
                        })}
                      </div>
                    )}
                  </article>
                ))}
              </div>
            )}
          </div>
        </section>
      </div>

      <NoticeDock>
        {notice && (
          <Notice onDismiss={() => setNotice(null)} dismissLabel={t.uiDismiss}>
            {notice}
          </Notice>
        )}
        {(error || peopleQ.error) && (
          <Notice
            tone="error"
            onDismiss={error ? () => setError(null) : undefined}
            dismissLabel={t.uiDismiss}
          >
            {t.errorPrefix}
            {error ?? (peopleQ.error instanceof Error ? peopleQ.error.message : "")}
          </Notice>
        )}
      </NoticeDock>

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
