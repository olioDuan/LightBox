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

export interface VerifierFlaggedPhoto {
  photo_id: string;
  stem: string;
  reason: string;
  action: string;
}

export interface VerifierLayer2Info {
  status: "passed" | "flagged" | "skipped";
  skipped_reason: string | null;
  model_id: string | null;
  provider: string | null;
  family: string | null;
  cross_family: boolean;
  checked_count: number;
  flagged_count: number;
  summary: string;
  llm_call_id: string | null;
  cost_usd: number;
  cache_hit: boolean;
}

export interface VerifierReport {
  layer?: "deterministic" | "two_layer" | string;
  passed?: boolean;
  exempt_count?: number;
  unconfirmed_burst_non_pick_count?: number;
  incomplete_atomic_pair_count?: number;
  selected_count?: number;
  initial_selected_count?: number;
  total_session_photos?: number;
  reject_ratio?: number;
  high_ratio_warning?: boolean;
  warnings?: string[];
  layer2?: VerifierLayer2Info;
  flagged_photos?: VerifierFlaggedPhoto[];
  [key: string]: unknown;
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
  verifier_report: VerifierReport | null;
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
  verifier_report: VerifierReport;
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

export interface AgentTodoItem {
  id: string;
  title: string;
  status: "pending" | "in_progress" | "done" | "skipped" | string;
}

export interface AgentEventItem {
  seq: number;
  event_type: string;
  payload: Record<string, unknown>;
  created_at: number;
}

export interface AgentSessionSummaryItem {
  agent_session_id: string;
  session_id: string;
  mode: "chat" | "supervisor" | string;
  status: "running" | "awaiting_user" | "completed" | "failed" | string;
  stage: "explore" | "review" | "propose" | string;
  step: number;
  spent_usd: number;
  latest_user_message: string;
  created_at: number;
}

export interface AgentLatestPlan {
  plan_id: string;
  plan_hash: string;
  count: number;
  status: string;
  approve_url: string;
}

export interface AgentPendingAsk {
  tool_call: { id: string; name: string; arguments: Record<string, unknown> };
  reason: string;
  ask_kind?: "budget" | "unpriced_triage" | string;
  scope?: "session" | "monthly" | string;
  suggested_budget_usd?: number;
  needs_monthly_bypass?: boolean;
  model_id?: string;
  unpriced_max_photos?: number;
  total_units?: number;
  completed_results?: Array<Record<string, unknown>>;
  remaining_calls?: Array<Record<string, unknown>>;
}

export interface AgentSessionDetail {
  agent_session_id: string;
  session_id: string;
  mode: "chat" | "supervisor" | string;
  status: "running" | "awaiting_user" | "completed" | "failed" | string;
  stage: "explore" | "review" | "propose" | string;
  step: number;
  answer: string;
  todos: AgentTodoItem[];
  constraints: string[];
  spent_usd: number;
  viewed_photos: number;
  context_tokens_est: number;
  stage_rebuilds: number;
  pending_ask: AgentPendingAsk | null;
  latest_plan: AgentLatestPlan | null;
  verifier_report: VerifierReport | null;
  error: string | null;
  events: AgentEventItem[];
}

export interface LlmTraceItem {
  id: string;
  role: string;
  provider: string;
  model: string;
  in_tokens: number;
  out_tokens: number;
  cached_tokens: number;
  cost_usd: number;
  latency_ms: number;
  cache_hit: boolean;
  session_id: string | null;
  status?: "ok" | "error" | string;
  error_class?: string | null;
  connection_fingerprint?: string | null;
  created_at: number;
}

export interface LlmTracesResponse {
  session_id: string | null;
  monthly_spent_usd: number;
  monthly_budget_usd: number;
  session_spent_usd: number | null;
  session_budget_usd: number;
  model_profile: string;
  total_calls: number;
  cache_hits: number;
  cache_hit_rate: number;
  total_in_tokens: number;
  total_out_tokens: number;
  total_cached_tokens: number;
  total_cost_usd: number;
  items: LlmTraceItem[];
}

export interface MemoryRuleItem {
  rule_id: string;
  status: "proposed" | "accepted" | "rejected" | string;
  rule_text: string;
  rationale: string | null;
  evidence_ids: number[];
  condition_dsl_json: string | null;
  hits: number;
  conflicts: number;
  created_at: number;
  updated_at: number;
}

export interface MemoryStateResponse {
  preferences_md: string;
  preferences_exists: boolean;
  rules: MemoryRuleItem[];
  proposed_count: number;
  accepted_count: number;
  rejected_count: number;
  new_override_count: number;
  total_override_count: number;
  min_overrides_threshold: number;
  ready_to_consolidate: boolean;
  last_consolidated_at: number | null;
}

export interface ConsolidateMemoryResponse {
  result: {
    status: string;
    new_override_count: number;
    analyzed_override_count: number;
    proposed_rules: MemoryRuleItem[];
    llm_call_id: string | null;
    cost_usd: number;
    model_id: string | null;
  };
  state: MemoryStateResponse;
}

export type AiProvider = "anthropic" | "openai" | "gemini" | "ollama" | "custom";
export type AiConnectionStatus = "unconfigured" | "configured_unverified" | "connected" | "error";
export type CredentialSource = "keyring" | "env" | "dotenv" | "none";
export type CostClassification = "known_free" | "known_paid" | "price_unknown";

export interface ProbeCheckResult {
  ok: boolean;
  error: string | null;
}

export interface AgentProbeReport {
  vision: ProbeCheckResult;
  tools: ProbeCheckResult;
  json_schema: ProbeCheckResult;
}

export interface TriageProbeReport {
  vision: ProbeCheckResult;
  json_schema: ProbeCheckResult;
  shared_with_agent?: boolean;
}

export interface AiConnectionErrorInfo {
  error_class: string;
  error_code?: string | null;
  message: string;
  occurred_at?: number;
  role?: string;
}

export interface AiTransientNotice {
  error_class: string;
  occurred_at: number;
  role?: string;
}

export interface AiPresetInfo {
  agent_model: string;
  triage_model: string;
  default_base_url: string;
  requires_api_key: boolean;
}

export interface AiConnectionState {
  status: AiConnectionStatus;
  configured: boolean;
  provider: AiProvider | null;
  agent_model: string | null;
  triage_model: string | null;
  base_url: string | null;
  normalized_endpoint?: string;
  reviewer: string;
  review_mode: "rules_only" | "two_layer" | string;
  is_local_free: boolean;
  input_cost_per_mtok: number | null;
  output_cost_per_mtok: number | null;
  triage_input_cost_per_mtok: number | null;
  triage_output_cost_per_mtok: number | null;
  cost_classification: CostClassification | null;
  triage_cost_classification: CostClassification | null;
  max_session_calls: number;
  max_triage_photos_unpriced: number;
  credential_source: CredentialSource;
  can_delete_credential: boolean;
  key_suffix: string | null;
  fingerprint: string | null;
  verified_at: number | null;
  agent_ready: boolean;
  triage_ready: boolean;
  agent_probes: AgentProbeReport | null;
  triage_probes: TriageProbeReport | null;
  agent_error: AiConnectionErrorInfo | null;
  triage_error: AiConnectionErrorInfo | null;
  error: AiConnectionErrorInfo | null;
  transient_notice: AiTransientNotice | null;
  presets: Record<string, AiPresetInfo>;
}

export interface AiConnectionTestResponse {
  overall_ok: boolean;
  status: AiConnectionStatus;
  provider: AiProvider;
  agent_model: string;
  triage_model: string;
  agent_ready: boolean;
  triage_ready: boolean;
  agent_probes: AgentProbeReport;
  triage_probes: TriageProbeReport;
  ollama_confirmed: boolean;
  cost_classification: CostClassification;
  fingerprint: string;
  verified_at: number | null;
  error_class: string | null;
  error_code: string | null;
  error_message: string | null;
  is_persistent_error: boolean;
}

export interface DeleteAiConnectionResponse extends AiConnectionState {
  deactivated_provider: AiProvider | null;
  credential_deleted: boolean;
  remaining_credential_source: CredentialSource;
  remaining_key_suffix: string | null;
  remaining_can_delete_credential: boolean;
}

export interface OllamaModelItem {
  name: string;
  size_bytes: number | null;
  context_length_limit: number | null;
}

export interface OllamaModelsResponse {
  base_url: string;
  openai_v1_url: string;
  reachable: boolean;
  ollama_confirmed: boolean;
  models: OllamaModelItem[];
  context_note: string;
  error: string | null;
}

export type McpClientId = "claude_desktop" | "cursor" | "codex" | "claude_code";
export type McpClientStatus =
  | "configured"
  | "not_configured"
  | "not_installed"
  | "invalid_path"
  | "parse_error";

export interface McpClientInfo {
  client_id: McpClientId;
  status: McpClientStatus;
  config_exists: boolean;
  lightbox_configured: boolean;
  command_valid: boolean;
  directory_valid: boolean;
  config_path: string;
}

export interface McpClientsResponse {
  clients: McpClientInfo[];
}

export interface McpSnippetResponse {
  client_id: McpClientId;
  format: "shell" | "toml" | "json";
  config_path: string;
  restart_required: boolean;
  snippet: string;
}

export interface AppSettings {
  language: Lang;
  sensitivity: "conservative" | "standard";
  model_profile: "local" | "balanced" | "best";
  auto_supervisor: boolean;
  budget: {
    session_budget_usd: number;
    monthly_budget_usd: number;
  };
  quarantine: {
    strategy: "outside_library" | "sibling" | "subfolder";
    auto_cleanup_days: number;
  };
  cache: {
    max_gb: number;
  };
  ai: Record<string, unknown> | null;
}

export interface TriageEstimateResponse {
  session_id: string;
  dry_run: true;
  yellow_count: number;
  burst_count: number;
  total_yellow_available: number;
  total_bursts_available: number;
  skipped_bursts_due_to_limit: number;
  note: string | null;
  total_units: number;
  est_usd: number;
  profile: string;
  model_alias: string;
  model_id: string;
  provider: string;
  vlm_backend_ready: boolean;
  session_spent_usd: number;
  session_budget_usd: number;
  monthly_spent_usd: number;
  monthly_budget_usd: number;
  cost_classification: CostClassification;
  requires_unpriced_confirmation: boolean;
  unpriced_max_photos: number;
}

export interface TriageRunResponse {
  job_id: string;
  session_id: string;
  status: string;
  paused_reason?: string | null;
  yellow_total?: number;
  burst_total?: number;
  reviewed_yellow?: number;
  reviewed_bursts?: number;
  failed_count?: number;
  cache_hits?: number;
  spent_usd?: number;
  elapsed_s?: number;
}

export class ApiError extends Error {
  status: number;
  errorClass?: string;

  constructor(message: string, status: number, errorClass?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.errorClass = errorClass;
  }
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
    let errorClass: string | undefined;
    try {
      const errJson = (await resp.json()) as { detail?: unknown };
      if (typeof errJson.detail === "string") {
        detail = errJson.detail;
      } else if (errJson.detail && typeof errJson.detail === "object" && !Array.isArray(errJson.detail)) {
        const obj = errJson.detail as { message?: unknown; error_class?: unknown };
        if (typeof obj.error_class === "string") {
          errorClass = obj.error_class;
        }
        if (typeof obj.message === "string" && obj.message) {
          detail = obj.message;
        }
      } else if (Array.isArray(errJson.detail) && errJson.detail.length > 0) {
        const first = errJson.detail[0] as { msg?: unknown };
        if (typeof first?.msg === "string") {
          detail = first.msg;
        }
      }
    } catch {
      // ignore non-JSON error body
    }
    throw new ApiError(detail, resp.status, errorClass);
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
    skipLlmVerify = false,
  ) =>
    apiFetch<{ plan: PlanDetail; verifier_report?: VerifierReport }>(
      `/api/sessions/${encodeURIComponent(sessionId)}/plans`,
      {
        method: "POST",
        body: JSON.stringify({
          photo_ids: photoIds,
          ...(strategy ? { strategy } : {}),
          ...(skipLlmVerify ? { skip_llm_verify: true } : {}),
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

  getSettings: () => apiFetch<AppSettings>("/api/settings"),

  updateSettings: (updates: {
    language?: Lang;
    sensitivity?: "conservative" | "standard";
    model_profile?: "local" | "balanced" | "best";
    auto_supervisor?: boolean;
    budget?: {
      session_budget_usd?: number;
      monthly_budget_usd?: number;
    };
  }) =>
    apiFetch<AppSettings>("/api/settings", {
      method: "PUT",
      body: JSON.stringify(updates),
    }),

  updateLanguage: (language: Lang) =>
    apiFetch<{ language: Lang }>("/api/settings", {
      method: "PUT",
      body: JSON.stringify({ language }),
    }),

  getAiConnection: () => apiFetch<AiConnectionState>("/api/ai/connection"),

  testAiConnection: (payload: {
    provider?: AiProvider;
    agent_model?: string;
    triage_model?: string;
    base_url?: string | null;
    api_key?: string;
    is_local_free?: boolean;
  }) =>
    apiFetch<AiConnectionTestResponse>("/api/ai/connection/test", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  saveAiConnection: (payload: {
    provider: AiProvider;
    agent_model?: string;
    triage_model?: string;
    base_url?: string | null;
    api_key?: string;
    reviewer?: string;
    is_local_free?: boolean;
    input_cost_per_mtok?: number | null;
    output_cost_per_mtok?: number | null;
    triage_input_cost_per_mtok?: number | null;
    triage_output_cost_per_mtok?: number | null;
    max_session_calls?: number;
    max_triage_photos_unpriced?: number;
  }) =>
    apiFetch<AiConnectionState>("/api/ai/connection", {
      method: "PUT",
      body: JSON.stringify(payload),
    }),

  deleteAiConnection: (deleteCredential = true) =>
    apiFetch<DeleteAiConnectionResponse>(
      `/api/ai/connection?delete_credential=${deleteCredential ? "true" : "false"}`,
      { method: "DELETE" },
    ),

  getOllamaModels: (baseUrl?: string) => {
    const qs = baseUrl ? `?base_url=${encodeURIComponent(baseUrl)}` : "";
    return apiFetch<OllamaModelsResponse>(`/api/ai/ollama/models${qs}`);
  },

  getMcpClients: () => apiFetch<McpClientsResponse>("/api/ai/mcp/clients"),

  getMcpSnippet: (client: McpClientId) =>
    apiFetch<McpSnippetResponse>("/api/ai/mcp/snippet", {
      method: "POST",
      body: JSON.stringify({ client }),
    }),

  estimateTriage: (
    sessionId: string,
    opts?: { maxPhotos?: number; profile?: string; budgetUsd?: number },
  ) =>
    apiFetch<TriageEstimateResponse>(
      `/api/sessions/${encodeURIComponent(sessionId)}/triage`,
      {
        method: "POST",
        body: JSON.stringify({
          dry_run: true,
          ...(opts?.maxPhotos !== undefined ? { max_photos: opts.maxPhotos } : {}),
          ...(opts?.profile ? { profile: opts.profile } : {}),
          ...(opts?.budgetUsd !== undefined ? { budget_usd: opts.budgetUsd } : {}),
        }),
      },
    ),

  runTriage: (
    sessionId: string,
    opts?: {
      maxPhotos?: number;
      profile?: string;
      budgetUsd?: number;
      wait?: boolean;
      confirmUnpriced?: boolean;
    },
  ) =>
    apiFetch<TriageRunResponse>(
      `/api/sessions/${encodeURIComponent(sessionId)}/triage`,
      {
        method: "POST",
        body: JSON.stringify({
          dry_run: false,
          wait: opts?.wait ?? true,
          confirm_unpriced: opts?.confirmUnpriced ?? false,
          ...(opts?.maxPhotos !== undefined ? { max_photos: opts.maxPhotos } : {}),
          ...(opts?.profile ? { profile: opts.profile } : {}),
          ...(opts?.budgetUsd !== undefined ? { budget_usd: opts.budgetUsd } : {}),
        }),
      },
    ),

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

  getPlan: (planId: string, lang: Lang = "zh-CN") =>
    apiFetch<PlanDetail>(
      `/api/plans/${encodeURIComponent(planId)}?lang=${encodeURIComponent(lang)}`,
    ),

  listAgentSessions: (sessionId?: string) => {
    const qs = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
    return apiFetch<{ items: AgentSessionSummaryItem[] }>(`/api/agent/sessions${qs}`);
  },

  getAgentSession: (agentSessionId: string) =>
    apiFetch<AgentSessionDetail>(`/api/agent/sessions/${encodeURIComponent(agentSessionId)}`),

  createAgentSession: (payload: {
    session_id: string;
    mode?: "chat" | "supervisor";
    message?: string;
    constraints?: string[];
    lang: Lang;
  }) =>
    apiFetch<AgentSessionDetail>("/api/agent/sessions", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  sendAgentMessage: (
    agentSessionId: string,
    payload: {
      message: string;
      approve_ask?: boolean;
      constraints?: string[];
      lang: Lang;
    },
  ) =>
    apiFetch<AgentSessionDetail>(
      `/api/agent/sessions/${encodeURIComponent(agentSessionId)}/messages`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      },
    ),

  getLlmTraces: (opts?: { sessionId?: string; limit?: number }) => {
    const params = new URLSearchParams();
    if (opts?.sessionId) params.set("session_id", opts.sessionId);
    if (opts?.limit !== undefined) params.set("limit", String(opts.limit));
    const qs = params.toString();
    return apiFetch<LlmTracesResponse>(`/api/insights/traces${qs ? `?${qs}` : ""}`);
  },

  getMemoryPreferences: () => apiFetch<MemoryStateResponse>("/api/memory/preferences"),

  saveMemoryPreferences: (content: string) =>
    apiFetch<MemoryStateResponse>("/api/memory/preferences", {
      method: "PUT",
      body: JSON.stringify({ content }),
    }),

  consolidateMemory: (force = false) =>
    apiFetch<ConsolidateMemoryResponse>("/api/memory/consolidate", {
      method: "POST",
      body: JSON.stringify({ force }),
    }),

  decideMemoryRule: (
    ruleId: string,
    payload: { decision: "accept" | "reject"; rule_text?: string },
  ) =>
    apiFetch<{ rule: MemoryRuleItem; state: MemoryStateResponse }>(
      `/api/memory/rules/${encodeURIComponent(ruleId)}/decision`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      },
    ),
};
