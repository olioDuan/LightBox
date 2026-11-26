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
};
