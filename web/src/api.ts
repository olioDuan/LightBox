import type { Lang } from "./i18n";

export type Bucket = "red" | "yellow" | "normal" | "exempt";

export interface SessionInfo {
  id: string;
  name: string;
  root_path: string;
  sensitivity: string;
  engine_version?: string;
  created_at?: number;
  last_scanned_at: number | null;
}

export interface SessionSummary {
  session: SessionInfo;
  total_photos: number;
  active_total_photos: number;
  counts_by_bucket: Record<string, number>;
  active_counts_by_bucket: Record<string, number>;
  reviewed_count: number;
  unreviewed_count: number;
  actionable_reject_count: number;
  top_reasons: Array<{ code: string; count: number; label: string }>;
  camera_distribution: Record<string, number>;
  low_res_preview_count: number;
  burst_count: number;
  unconfirmed_burst_count: number;
  quarantined_count: number;
}

export interface PhotoItem {
  photo_id: string;
  stem: string;
  rel_dir: string;
  bucket: Bucket;
  source: string;
  confidence: number;
  reason_codes: string[];
  reason_labels: string[];
  decision_version: number;
  is_protected: boolean;
  is_quarantined: boolean;
  burst_id: string | null;
  preview_w: number | null;
  preview_h: number | null;
  face_count: number;
  seq_no: number;
}

export interface BurstMember {
  photo_id: string;
  rank: number;
  is_pick: boolean;
  ai_note: string | null;
}

export interface BurstGroup {
  id: string;
  kind: string;
  confirmed: boolean;
  members: BurstMember[];
}

export interface PlanItem {
  photo_id: string;
  decision_version: number;
  src_paths: string[];
  dst_paths: string[];
  status: string;
  done_at: number | null;
  stem?: string;
  reason_codes?: string[];
  reason_labels?: string[];
}

export interface PlanDetail {
  id: string;
  session_id: string;
  session_name?: string | null;
  action_type: string;
  plan_hash: string;
  status: string;
  quarantine_dir: string | null;
  approved_by: string | null;
  approved_at: number | null;
  verifier_report: Record<string, unknown> | null;
  item_count: number;
  items: PlanItem[];
}

export interface DraftObviousResponse {
  plan: PlanDetail | null;
  selected_count: number;
  tech_reject_count: number;
  burst_non_pick_count: number;
  manual_reject_count: number;
  high_ratio_warning: boolean;
  excluded_already_quarantined: number;
  excluded_unconfirmed_burst: number;
  excluded_in_applying_plan: number;
  superseded_plan_ids: string[];
  verifier_report: Record<string, unknown>;
}

export interface ScanResponse {
  session_id: string;
  job_id: string;
  status: string;
  total_photos: number;
  counts_by_bucket: Record<string, number>;
  burst_count: number;
  elapsed_s: number;
}

export interface ApplyPlanResponse {
  plan_id: string;
  plan_hash: string;
  status: string;
  quarantine_dir: string;
  moved_photo_count: number;
  moved_file_count: number;
}

export interface RestorePlanResponse {
  plan_id: string;
  restored_photo_ids: string[];
  restored_file_count: number;
  plan_status: string;
}

export interface SearchMatchItem {
  photo_id: string;
  session_id: string;
  session_name: string;
  stem: string;
  rel_dir: string;
  bucket: Bucket;
  confidence: number;
  reason_codes: string[];
  reason_labels: string[];
  camera_model: string | null;
  lens_model: string | null;
  capture_time: number | null;
  light_bucket: string | null;
  focal_eq: number | null;
  scene_tag: string | null;
  scene_label: string | null;
  exif_brief: string;
  score: number;
  similarity: number;
  relation: string;
  gps_lat: number | null;
  gps_lon: number | null;
  location_key: string | null;
  location_label: string | null;
  location_source: "gps" | "folder" | null;
}

export type SearchStrictness = "strict" | "balanced" | "relaxed" | "all";

export interface SearchLocationOption {
  key: string;
  label: string;
}

export interface SearchPhotosResponse {
  query: string;
  mode: "semantic" | "filter" | "fts_exif";
  semantic_ready: boolean;
  indexed_count: number;
  total_candidates: number;
  cameras: string[];
  locations: SearchLocationOption[];
  detected_location_key: string | null;
  detected_location_label: string | null;
  effective_visual_query: string;
  strictness: SearchStrictness;
  score_threshold: number | null;
  matches: SearchMatchItem[];
  weak_matches: SearchMatchItem[];
}

export interface SimilarPhotosResponse {
  anchor: SearchMatchItem;
  anchor_id: string;
  session_id: string;
  mode: "semantic" | "burst_dhash";
  semantic_ready: boolean;
  strictness: SearchStrictness;
  score_threshold: number | null;
  similar: SearchMatchItem[];
  weak_similar: SearchMatchItem[];
}

export interface EmbedSessionResponse {
  session_id: string;
  indexed_count: number;
  skipped_count: number;
  tagged_count: number;
  total_photos: number;
  model_id: string;
  elapsed_s: number;
}

export interface FaceSlot {
  content_fp: string;
  face_idx: number;
  bbox: number[];
  area_ratio: number;
  det_score: number | null;
  status: "assigned" | "unassigned" | "ignored";
  source: string;
  person_id: string | null;
  person_name: string | null;
  cluster_id: string | null;
  confidence: number | null;
}

export interface PeoplePhotoItem {
  photo_id: string;
  session_id: string;
  session_name: string;
  stem: string;
  rel_dir: string;
  content_fp: string;
  capture_time: number | null;
  valid_face_count: number;
  faces: FaceSlot[];
}

export interface SimilarFaceCandidate {
  kind: "person" | "cluster";
  target_id: string;
  name: string | null;
  cover_photo_id: string | null;
  cover_face_idx: number | null;
  photo_count: number;
  similarity: number;
}

export interface PersonGroupInfo {
  person_id: string;
  group_key: string;
  name: string;
  is_self: boolean;
  cover_photo_id: string | null;
  cover_face_idx: number | null;
  photo_count: number;
  group_photo_count: number;
  photos: PeoplePhotoItem[];
  similar_candidates: SimilarFaceCandidate[];
}

export interface UnnamedClusterInfo {
  cluster_id: string;
  cover_photo_id: string;
  cover_content_fp: string;
  cover_face_idx: number;
  face_count: number;
  solo_photo_count: number;
  faces: Array<{ content_fp: string; face_idx: number }>;
  photos: PeoplePhotoItem[];
  similar_candidates: SimilarFaceCandidate[];
}

export interface PeopleGroupsResponse {
  session_id: string | null;
  group_person_id: string | null;
  total_kept_photos: number;
  indexed_face_count: number;
  people: PersonGroupInfo[];
  group_photos: PeoplePhotoItem[];
  group_photo_total_count: number;
  unnamed_photos: PeoplePhotoItem[];
  unnamed_clusters: UnnamedClusterInfo[];
  no_face_photos: PeoplePhotoItem[];
}

export interface PeopleIndexResponse {
  session_id: string;
  kept_photo_count: number;
  valid_face_count: number;
  newly_embedded_faces: number;
  skipped_cached_faces: number;
  named_people_count: number;
  unnamed_cluster_count: number;
  model_id: string;
  threshold: number;
  elapsed_s: number;
}

export interface PeopleExportResponse {
  target_dir: string;
  exported_photos: number;
  exported_files: number;
  groups: Array<{
    group_key: string;
    folder_name: string;
    photo_count: number;
    file_count: number;
  }>;
}

export interface DigestScope {
  mode: "session" | "cross_session";
  session_id: string | null;
  session_name: string | null;
  session_count: number;
  date_from: string | null;
  date_to: string | null;
  min_samples: number;
}

export interface DigestOverview {
  total_shots: number;
  tech_reject_count: number;
  tech_reject_rate: number;
  burst_cull_count: number;
  burst_cull_rate: number;
  subjective_reject_count: number;
  subjective_reject_rate: number;
  kept_count: number;
  non_burst_total: number;
  kept_rate_excl_burst: number;
  kept_rate_total: number;
  uncertain_count: number;
  uncertain_rate: number;
}

export interface DigestReasonStat {
  code: string;
  label: string;
  count: number;
  rate_of_tech_rejects: number;
  rate_of_total: number;
}

export interface DigestSliceBucket {
  key: string;
  label: string;
  total: number;
  non_burst_total: number;
  tech_reject: number;
  tech_reject_rate: number;
  burst_cull: number;
  burst_cull_rate: number;
  subjective_reject: number;
  subjective_reject_rate: number;
  kept: number;
  kept_rate_excl_burst: number;
  kept_rate_total: number;
  uncertain: number;
  uncertain_rate: number;
  meets_min_samples: boolean;
  top_reason_code: string | null;
  top_reason_label: string | null;
  top_reason_count: number;
}

export interface DigestDimension {
  key: string;
  label: string;
  buckets: DigestSliceBucket[];
}

export interface DigestConditionHighlight {
  dimension_key: string;
  dimension_label: string;
  bucket_key: string;
  bucket_label: string;
  total: number;
  non_burst_total: number;
  tech_reject: number;
  tech_reject_rate: number;
  tech_reject_delta: number;
  kept: number;
  kept_rate_excl_burst: number;
  kept_rate_delta: number;
  kept_rate_total: number;
  top_reason_code: string | null;
  top_reason_label: string | null;
}

export interface DigestStableSetup {
  focal_bucket: string;
  focal_label: string;
  shutter_bucket: string;
  shutter_label: string;
  total: number;
  non_burst_total: number;
  kept: number;
  tech_reject: number;
  burst_cull: number;
  subjective_reject: number;
  uncertain: number;
  kept_rate_excl_burst: number;
  kept_rate_total: number;
  tech_reject_rate: number;
}

export interface ShootingDigestResponse {
  scope: DigestScope;
  lang: string;
  overview: DigestOverview;
  top_tech_reasons: DigestReasonStat[];
  dimensions: DigestDimension[];
  high_tech_reject_conditions: DigestConditionHighlight[];
  high_kept_conditions: DigestConditionHighlight[];
  stable_setups: DigestStableSetup[];
}

let csrfToken: string | null = null;

export function setCsrfToken(token: string | null): void {
  csrfToken = token;
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const method = (init?.method ?? "GET").toUpperCase();
  const headers = new Headers(init?.headers);
  if (init?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (method !== "GET" && method !== "HEAD" && csrfToken) {
    headers.set("X-Lightbox-CSRF", csrfToken);
  }
  const resp = await fetch(path, {
    ...init,
    method,
    headers,
    credentials: "same-origin",
  });
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const errJson = (await resp.json()) as { detail?: unknown };
      if (typeof errJson.detail === "string") {
        detail = errJson.detail;
      }
    } catch {
      // ignore non-JSON error body
    }
    throw new Error(detail);
  }
  return (await resp.json()) as T;
}

export function photoImageUrl(
  photoId: string,
  size: "grid" | "loupe" | "crop",
  region?: "face0" | "sharpest",
): string {
  const params = new URLSearchParams({ size });
  if (region) {
    params.set("region", region);
  }
  return `/api/photos/${encodeURIComponent(photoId)}/image?${params.toString()}`;
}

export function photoFaceCropUrl(photoId: string, faceIdx: number, size = 160): string {
  const params = new URLSearchParams({
    face_idx: String(faceIdx),
    size: String(size),
  });
  return `/api/photos/${encodeURIComponent(photoId)}/face-crop?${params.toString()}`;
}

export const api = {
  async checkAuth(): Promise<{ authenticated: boolean; csrf_token: string } | null> {
    const resp = await fetch("/api/auth/me", { credentials: "same-origin" });
    if (resp.status === 401) {
      return null;
    }
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const data = (await resp.json()) as { authenticated: boolean; csrf_token: string };
    csrfToken = data.csrf_token;
    return data;
  },

  async confirmLocalSession(): Promise<{ authenticated: boolean; csrf_token: string }> {
    const resp = await fetch("/api/auth/session", {
      method: "POST",
      credentials: "same-origin",
      headers: { "X-Lightbox-Local-Confirm": "1" },
    });
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const data = (await resp.json()) as { authenticated: boolean; csrf_token: string };
    csrfToken = data.csrf_token;
    return data;
  },

  listSessions: () => apiFetch<{ items: SessionInfo[] }>("/api/sessions"),

  getSessionSummary: (sessionId: string, lang: Lang) =>
    apiFetch<SessionSummary>(
      `/api/sessions/${encodeURIComponent(sessionId)}/summary?lang=${encodeURIComponent(lang)}`,
    ),

  listSessionPhotos: (
    sessionId: string,
    opts: {
      bucket?: string;
      sort?: string;
      limit?: number;
      offset?: number;
      lang: Lang;
      includeQuarantined?: boolean;
    },
  ) => {
    const params = new URLSearchParams({
      sort: opts.sort ?? "uncertainty",
      limit: String(opts.limit ?? 200),
      offset: String(opts.offset ?? 0),
      lang: opts.lang,
      include_quarantined: opts.includeQuarantined ? "true" : "false",
    });
    if (opts.bucket && opts.bucket !== "all") {
      params.set("bucket", opts.bucket);
    }
    return apiFetch<{ items: PhotoItem[] }>(
      `/api/sessions/${encodeURIComponent(sessionId)}/photos?${params.toString()}`,
    );
  },

  listSessionBursts: (sessionId: string) =>
    apiFetch<{ items: BurstGroup[] }>(
      `/api/sessions/${encodeURIComponent(sessionId)}/bursts`,
    ),

  pickFolder: () =>
    apiFetch<{ path: string | null; canceled: boolean }>("/api/system/pick-folder", {
      method: "POST",
    }),

  runDemoBootstrap: () =>
    apiFetch<ScanResponse & { demo_dir: string; generated_photos: number }>("/api/demo/bootstrap", {
      method: "POST",
      body: JSON.stringify({}),
    }),

  scanFolder: (folderPath: string, wait = true) =>
    apiFetch<ScanResponse>("/api/sessions/scan", {
      method: "POST",
      body: JSON.stringify({ folder_path: folderPath, wait }),
    }),

  pauseJob: (jobId: string) =>
    apiFetch<{ job_id: string; status: string }>(`/api/jobs/${encodeURIComponent(jobId)}/pause`, {
      method: "POST",
    }),

  resumeJob: (jobId: string) =>
    apiFetch<ScanResponse>(`/api/jobs/${encodeURIComponent(jobId)}/resume`, {
      method: "POST",
    }),

  setVerdict: (
    photoIds: string[],
    bucket: Bucket,
    reasonTag?: string,
    reasonCodes?: string[],
  ) =>
    apiFetch<{ updated: Array<{ photo_id: string; bucket: Bucket; decision_version: number }> }>(
      "/api/verdicts",
      {
        method: "POST",
        body: JSON.stringify({
          photo_ids: photoIds,
          bucket,
          reason_tag: reasonTag ?? null,
          ...(reasonCodes ? { reason_codes: reasonCodes } : {}),
        }),
      },
    ),

  undoVerdict: (photoId: string) =>
    apiFetch<{ undone: boolean; photo_id: string; bucket?: Bucket }>("/api/verdicts/undo", {
      method: "POST",
      body: JSON.stringify({ photo_id: photoId }),
    }),

  confirmBurst: (burstId: string, pickPhotoId: string) =>
    apiFetch<{ burst_id: string; pick_photo_id: string; confirmed: boolean }>(
      `/api/bursts/${encodeURIComponent(burstId)}/confirm`,
      {
        method: "POST",
        body: JSON.stringify({ pick_photo_id: pickPhotoId }),
      },
    ),

  draftObviousRejects: (
    sessionId: string,
    strategy?: "outside_library" | "sibling" | "subfolder",
    lang: Lang = "zh-CN",
  ) =>
    apiFetch<DraftObviousResponse>(
      `/api/sessions/${encodeURIComponent(sessionId)}/plans/obvious-rejects?lang=${encodeURIComponent(lang)}`,
      {
        method: "POST",
        body: JSON.stringify(strategy ? { strategy } : {}),
      },
    ),

  createPlan: (
    sessionId: string,
    photoIds: string[],
    strategy?: "outside_library" | "sibling" | "subfolder",
  ) =>
    apiFetch<{ plan: PlanDetail }>(
      `/api/sessions/${encodeURIComponent(sessionId)}/plans`,
      {
        method: "POST",
        body: JSON.stringify({
          photo_ids: photoIds,
          ...(strategy ? { strategy } : {}),
        }),
      },
    ),

  openSessionFolder: (sessionId: string) =>
    apiFetch<{ opened: boolean; path: string }>(
      `/api/sessions/${encodeURIComponent(sessionId)}/open-folder`,
      {
        method: "POST",
      },
    ),

  resetSessionReview: (sessionId: string) =>
    apiFetch<{ session_id: string; reset_count: number }>(
      `/api/sessions/${encodeURIComponent(sessionId)}/reset-review`,
      {
        method: "POST",
      },
    ),

  autoScreenSession: (sessionId: string) =>
    apiFetch<{
      session_id: string;
      total_active: number;
      flagged_red: number;
      flagged_yellow: number;
      unchanged_normal: number;
    }>(`/api/sessions/${encodeURIComponent(sessionId)}/auto-screen`, {
      method: "POST",
    }),

  listPlans: (sessionId?: string) => {
    const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
    return apiFetch<{ items: PlanDetail[] }>(`/api/plans${qs}`);
  },

  approveAndApplyPlan: (planId: string, planHash: string) =>
    apiFetch<ApplyPlanResponse>(
      `/api/plans/${encodeURIComponent(planId)}/approve-and-apply`,
      {
        method: "POST",
        body: JSON.stringify({ plan_hash: planHash }),
      },
    ),

  restorePlan: (planId: string, photoIds?: string[]) =>
    apiFetch<RestorePlanResponse>(`/api/plans/${encodeURIComponent(planId)}/restore`, {
      method: "POST",
      body: JSON.stringify(photoIds ? { photo_ids: photoIds } : {}),
    }),

  openPlanFolder: (planId: string) =>
    apiFetch<{ opened: boolean; path: string }>(
      `/api/plans/${encodeURIComponent(planId)}/open-folder`,
      {
        method: "POST",
      },
    ),

  updateLanguage: (language: Lang) =>
    apiFetch<{ language: Lang }>("/api/settings", {
      method: "PUT",
      body: JSON.stringify({ language }),
    }),

  searchPhotos: (opts: {
    q?: string;
    sessionId?: string;
    dateFrom?: string;
    dateTo?: string;
    camera?: string;
    lightBucket?: string;
    location?: string;
    ignoreLocation?: boolean;
    strictness?: SearchStrictness;
    limit?: number;
    lang: Lang;
  }) => {
    const params = new URLSearchParams({
      limit: String(opts.limit ?? 48),
      lang: opts.lang,
    });
    if (opts.q && opts.q.trim()) params.set("q", opts.q.trim());
    if (opts.sessionId) params.set("session_id", opts.sessionId);
    if (opts.dateFrom) params.set("date_from", opts.dateFrom);
    if (opts.dateTo) params.set("date_to", opts.dateTo);
    if (opts.camera) params.set("camera", opts.camera);
    if (opts.lightBucket) params.set("light_bucket", opts.lightBucket);
    if (opts.location) params.set("location", opts.location);
    if (opts.ignoreLocation) params.set("ignore_location", "true");
    if (opts.strictness) params.set("strictness", opts.strictness);
    return apiFetch<SearchPhotosResponse>(`/api/search?${params.toString()}`);
  },

  findSimilarPhotos: (
    photoId: string,
    opts: { limit?: number; sessionId?: string; strictness?: SearchStrictness; lang: Lang },
  ) => {
    const params = new URLSearchParams({
      limit: String(opts.limit ?? 18),
      lang: opts.lang,
    });
    if (opts.sessionId) params.set("session_id", opts.sessionId);
    if (opts.strictness) params.set("strictness", opts.strictness);
    return apiFetch<SimilarPhotosResponse>(
      `/api/photos/${encodeURIComponent(photoId)}/similar?${params.toString()}`,
    );
  },

  embedSession: (sessionId: string) =>
    apiFetch<EmbedSessionResponse>(
      `/api/sessions/${encodeURIComponent(sessionId)}/embed`,
      { method: "POST" },
    ),

  indexSessionPeople: (sessionId: string, threshold?: number) =>
    apiFetch<PeopleIndexResponse>(
      `/api/sessions/${encodeURIComponent(sessionId)}/people/index`,
      {
        method: "POST",
        body: JSON.stringify(threshold !== undefined ? { threshold } : {}),
      },
    ),

  listPeopleGroups: (opts?: { sessionId?: string; groupPersonId?: string }) => {
    const params = new URLSearchParams();
    if (opts?.sessionId) params.set("session_id", opts.sessionId);
    if (opts?.groupPersonId) params.set("group_person_id", opts.groupPersonId);
    const qs = params.toString();
    return apiFetch<PeopleGroupsResponse>(`/api/people${qs ? `?${qs}` : ""}`);
  },

  labelPeople: (payload: {
    action: "name" | "merge" | "ignore" | "split" | "reassign";
    name?: string;
    is_self?: boolean;
    cluster_id?: string;
    person_id?: string;
    faces?: Array<{ content_fp: string; face_idx: number }>;
    source_cluster_id?: string;
    source_cluster_ids?: string[];
    target_cluster_id?: string;
    source_person_id?: string;
    source_person_ids?: string[];
    target_person_id?: string;
    threshold?: number;
  }) =>
    apiFetch<Record<string, unknown>>("/api/people/label", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  exportPeopleGroups: (payload: {
    target_dir: string;
    session_id?: string;
    group_keys?: string[];
    include_unnamed?: boolean;
    include_no_face?: boolean;
    lang: Lang;
  }) =>
    apiFetch<PeopleExportResponse>("/api/people/export", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getShootingDigest: (opts: {
    sessionId?: string;
    dateFrom?: string;
    dateTo?: string;
    minSamples?: number;
    lang: Lang;
  }) => {
    const params = new URLSearchParams({ lang: opts.lang });
    if (opts.sessionId) params.set("session_id", opts.sessionId);
    if (opts.dateFrom) params.set("date_from", opts.dateFrom);
    if (opts.dateTo) params.set("date_to", opts.dateTo);
    if (opts.minSamples !== undefined) params.set("min_samples", String(opts.minSamples));
    return apiFetch<ShootingDigestResponse>(`/api/insights/digest?${params.toString()}`);
  },
};
