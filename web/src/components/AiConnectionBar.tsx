import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ApiError,
  api,
  type AiConnectionState,
  type AiConnectionTestResponse,
  type AiProvider,
  type OllamaModelsResponse,
  type ProbeCheckResult,
} from "../api";
import { useI18n } from "../i18n";
import { Button } from "./ui";
import { Check, ChevronDown, ChevronUp, Refresh, Trash, X } from "reicon-react";

interface AiConnectionBarProps {
  open: boolean;
  onConnectionLoaded?: (state: AiConnectionState) => void;
}

const PROVIDERS: AiProvider[] = ["anthropic", "openai", "gemini", "ollama", "custom"];

function formatVerifiedTime(ts: number | null): string | null {
  if (!ts) return null;
  try {
    const d = new Date(ts * 1000);
    if (Number.isNaN(d.getTime())) return null;
    const now = new Date();
    const pad2 = (n: number) => String(n).padStart(2, "0");
    const hm = `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
    const sameDay =
      d.getFullYear() === now.getFullYear() &&
      d.getMonth() === now.getMonth() &&
      d.getDate() === now.getDate();
    if (sameDay && Math.abs(now.getTime() - d.getTime()) < 24 * 60 * 60 * 1000) {
      return hm;
    }
    const md = `${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
    if (d.getFullYear() === now.getFullYear()) {
      return `${md} ${hm}`;
    }
    return `${d.getFullYear()}-${md} ${hm}`;
  } catch {
    return null;
  }
}

const CONNECTIVITY_OR_ACCOUNT_ERROR_CLASSES = new Set([
  "auth_failed",
  "insufficient_balance",
  "model_or_endpoint_unavailable",
  "endpoint_binding_mismatch",
  "service_unreachable",
  "network_timeout",
  "rate_limited",
]);

function isDisplayableErrorCode(
  code: string | null | undefined,
  errorClass: string | null | undefined,
  knownClasses: Record<string, string>,
): boolean {
  if (!code) return false;
  const trimmed = code.trim();
  if (!trimmed) return false;
  if (errorClass && trimmed === errorClass) return false;
  if (trimmed in knownClasses) return false;
  // Never leak Python internal exception class names (e.g. ModelError, ValueError, RuntimeError)
  if (/^[A-Z][A-Za-z0-9_]*(Error|Exception)$/.test(trimmed)) return false;
  return true;
}

function parseOptionalNumber(raw: string): number | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : Number.NaN;
}

function formatKeySuffix(suffix: string | null | undefined): string {
  if (!suffix) return "";
  return suffix.startsWith("*") || suffix.startsWith("•") || suffix.startsWith("·")
    ? suffix
    : `····${suffix}`;
}

function ProbeTag({
  label,
  probe,
  fallbackUntested,
  untestedLabel,
}: {
  label: string;
  probe: ProbeCheckResult;
  fallbackUntested?: boolean;
  untestedLabel: string;
}) {
  const status =
    probe.status ?? (probe.ok ? "passed" : fallbackUntested ? "untested" : "failed");
  if (status === "untested") {
    return (
      <span className="inline-flex items-center gap-1 text-xs text-ink-3">
        <span>{label}</span>
        <span>{untestedLabel}</span>
      </span>
    );
  }
  const ok = status === "passed";
  return (
    <span className={`inline-flex items-center gap-1 text-xs ${ok ? "text-keep-ink" : "text-reject-ink"}`}>
      {ok ? <Check size={12} strokeWidth={2} /> : <X size={12} strokeWidth={2} />}
      <span>{label}</span>
    </span>
  );
}

export function AiConnectionBar({ open, onConnectionLoaded }: AiConnectionBarProps) {
  const { t } = useI18n();
  const queryClient = useQueryClient();

  const connQ = useQuery({
    queryKey: ["ai-connection"],
    queryFn: () => api.getAiConnection(),
    enabled: open,
  });

  const conn = connQ.data ?? null;

  useEffect(() => {
    if (conn) {
      onConnectionLoaded?.(conn);
    }
  }, [conn, onConnectionLoaded]);

  const [expanded, setExpanded] = useState(false);
  const [initialized, setInitialized] = useState(false);

  // Form fields
  const [provider, setProvider] = useState<AiProvider>("anthropic");
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [agentModel, setAgentModel] = useState("");
  const [triageModel, setTriageModel] = useState("");
  const [isLocalFree, setIsLocalFree] = useState(false);
  const [showPricing, setShowPricing] = useState(false);
  const [inputCost, setInputCost] = useState("");
  const [outputCost, setOutputCost] = useState("");
  const [triageInputCost, setTriageInputCost] = useState("");
  const [triageOutputCost, setTriageOutputCost] = useState("");

  // Action states
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [confirmDeleteCred, setConfirmDeleteCred] = useState(false);
  const [ollamaLoading, setOllamaLoading] = useState(false);
  const [ollamaData, setOllamaData] = useState<OllamaModelsResponse | null>(null);
  const [testResult, setTestResult] = useState<AiConnectionTestResponse | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const [formNotice, setFormNotice] = useState<string | null>(null);

  function syncFormFromConn(data: AiConnectionState) {
    const nextProvider: AiProvider = data.provider ?? "anthropic";
    const preset = data.presets[nextProvider];
    setProvider(nextProvider);
    setApiKey("");
    setBaseUrl(data.base_url ?? (nextProvider === "ollama" ? preset?.default_base_url ?? "http://127.0.0.1:11434" : ""));
    setAgentModel(data.agent_model ?? preset?.agent_model ?? "");
    setTriageModel(data.triage_model ?? preset?.triage_model ?? "");
    setIsLocalFree(Boolean(data.is_local_free));
    setInputCost(data.input_cost_per_mtok != null ? String(data.input_cost_per_mtok) : "");
    setOutputCost(data.output_cost_per_mtok != null ? String(data.output_cost_per_mtok) : "");
    setTriageInputCost(
      data.triage_input_cost_per_mtok != null ? String(data.triage_input_cost_per_mtok) : "",
    );
    setTriageOutputCost(
      data.triage_output_cost_per_mtok != null ? String(data.triage_output_cost_per_mtok) : "",
    );
    setShowPricing(
      data.input_cost_per_mtok != null ||
        data.output_cost_per_mtok != null ||
        data.triage_input_cost_per_mtok != null ||
        data.triage_output_cost_per_mtok != null,
    );
  }

  useEffect(() => {
    if (conn && !initialized) {
      syncFormFromConn(conn);
      setInitialized(true);
    }
  }, [conn, initialized]);

  function handleProviderChange(next: AiProvider) {
    setProvider(next);
    setApiKey("");
    setTestResult(null);
    setFormError(null);
    setFormNotice(null);
    setConfirmDeleteCred(false);
    setOllamaData(null);
    if (conn && conn.configured && conn.provider === next) {
      syncFormFromConn(conn);
      return;
    }
    const preset = conn?.presets[next];
    setAgentModel(preset?.agent_model ?? "");
    setTriageModel(preset?.triage_model ?? "");
    setBaseUrl(next === "ollama" ? preset?.default_base_url ?? "http://127.0.0.1:11434" : "");
    setIsLocalFree(false);
    setInputCost("");
    setOutputCost("");
    setTriageInputCost("");
    setTriageOutputCost("");
    setShowPricing(false);
  }

  function formatErrorClass(
    errorClass: string | null | undefined,
    errorCode?: string | null,
    fallbackMsg?: string | null,
  ): string {
    if (errorClass) {
      const mapped = t.aiErrorClassLabels[errorClass] ?? errorClass;
      const suffix = isDisplayableErrorCode(errorCode, errorClass, t.aiErrorClassLabels)
        ? ` (${errorCode})`
        : "";
      return `${mapped}${suffix}`;
    }
    return fallbackMsg || t.aiErrorClassLabels.request_failed;
  }

  function formatApiError(err: unknown): string {
    if (err instanceof ApiError && err.errorClass && t.aiErrorClassLabels[err.errorClass]) {
      return t.aiErrorClassLabels[err.errorClass];
    }
    return err instanceof Error ? err.message : String(err);
  }

  function validateRequiredFields(): boolean {
    if (provider === "custom" && !baseUrl.trim()) {
      setFormError(t.aiErrorClassLabels.missing_base_url);
      return false;
    }
    if (provider === "custom" && !agentModel.trim()) {
      setFormError(t.aiErrorClassLabels.missing_model);
      return false;
    }
    return true;
  }

  function validatePricingInputs(): {
    ok: boolean;
    input_cost_per_mtok: number | null;
    output_cost_per_mtok: number | null;
    triage_input_cost_per_mtok: number | null;
    triage_output_cost_per_mtok: number | null;
  } {
    const inVal = parseOptionalNumber(inputCost);
    const outVal = parseOptionalNumber(outputCost);
    const tInVal = parseOptionalNumber(triageInputCost);
    const tOutVal = parseOptionalNumber(triageOutputCost);
    if (
      Number.isNaN(inVal) ||
      Number.isNaN(outVal) ||
      Number.isNaN(tInVal) ||
      Number.isNaN(tOutVal)
    ) {
      setFormError(t.aiPricingPairError);
      return {
        ok: false,
        input_cost_per_mtok: null,
        output_cost_per_mtok: null,
        triage_input_cost_per_mtok: null,
        triage_output_cost_per_mtok: null,
      };
    }
    if ((inVal === null) !== (outVal === null) || (tInVal === null) !== (tOutVal === null)) {
      setFormError(t.aiPricingPairError);
      return {
        ok: false,
        input_cost_per_mtok: null,
        output_cost_per_mtok: null,
        triage_input_cost_per_mtok: null,
        triage_output_cost_per_mtok: null,
      };
    }
    return {
      ok: true,
      input_cost_per_mtok: inVal,
      output_cost_per_mtok: outVal,
      triage_input_cost_per_mtok: tInVal,
      triage_output_cost_per_mtok: tOutVal,
    };
  }

  async function persistConnection(costs: {
    input_cost_per_mtok: number | null;
    output_cost_per_mtok: number | null;
    triage_input_cost_per_mtok: number | null;
    triage_output_cost_per_mtok: number | null;
  }) {
    const saved = await api.saveAiConnection({
      provider,
      agent_model: agentModel.trim() || undefined,
      triage_model: triageModel.trim() || undefined,
      base_url: baseUrl.trim() || null,
      ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}),
      is_local_free: provider === "custom" ? isLocalFree : false,
      input_cost_per_mtok: costs.input_cost_per_mtok,
      output_cost_per_mtok: costs.output_cost_per_mtok,
      triage_input_cost_per_mtok: costs.triage_input_cost_per_mtok,
      triage_output_cost_per_mtok: costs.triage_output_cost_per_mtok,
    });
    setApiKey("");
    syncFormFromConn(saved);
    queryClient.setQueryData(["ai-connection"], saved);
    void queryClient.invalidateQueries({ queryKey: ["settings"] });
    void queryClient.invalidateQueries({ queryKey: ["llm-traces"] });
    return saved;
  }

  async function handleTestAndConnect() {
    setFormError(null);
    setFormNotice(null);
    setConfirmDeleteCred(false);
    if (!validateRequiredFields()) return;
    const costs = validatePricingInputs();
    if (!costs.ok) return;

    setTesting(true);
    try {
      const res = await api.testAiConnection({
        provider,
        agent_model: agentModel.trim() || undefined,
        triage_model: triageModel.trim() || undefined,
        base_url: baseUrl.trim() || null,
        ...(apiKey.trim() ? { api_key: apiKey.trim() } : {}),
        is_local_free: provider === "custom" ? isLocalFree : false,
      });
      setTestResult(res);
      if (res.overall_ok) {
        await persistConnection(costs);
        setExpanded(false);
      } else {
        void queryClient.invalidateQueries({ queryKey: ["ai-connection"] });
        if (res.error_class || res.error_message) {
          setFormError(formatErrorClass(res.error_class, res.error_code, res.error_message));
        }
      }
    } catch (err) {
      setFormError(formatApiError(err));
    } finally {
      setTesting(false);
    }
  }

  async function handleSaveOnly() {
    setFormError(null);
    setFormNotice(null);
    setConfirmDeleteCred(false);
    if (!validateRequiredFields()) return;
    const costs = validatePricingInputs();
    if (!costs.ok) return;

    setSaving(true);
    try {
      await persistConnection(costs);
      setTestResult(null);
      setExpanded(false);
    } catch (err) {
      setFormError(formatApiError(err));
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(deleteCredential: boolean) {
    setFormError(null);
    setFormNotice(null);
    setConfirmDeleteCred(false);
    setDeleting(true);
    try {
      const res = await api.deleteAiConnection(deleteCredential);
      setTestResult(null);
      setApiKey("");
      queryClient.setQueryData(["ai-connection"], res);
      void queryClient.invalidateQueries({ queryKey: ["settings"] });
      if (res.remaining_credential_source === "env") {
        setFormNotice(t.aiDeleteRemainingEnvNotice(res.remaining_key_suffix));
      }
    } catch (err) {
      setFormError(formatApiError(err));
    } finally {
      setDeleting(false);
    }
  }

  async function handleFetchOllama() {
    setOllamaLoading(true);
    setFormError(null);
    try {
      const data = await api.getOllamaModels(baseUrl.trim() || undefined);
      setOllamaData(data);
      if (data.reachable && data.models.length > 0) {
        if (!agentModel.trim()) setAgentModel(data.models[0].name);
        if (!triageModel.trim()) setTriageModel(data.models[0].name);
      }
    } catch (err) {
      setFormError(formatApiError(err));
    } finally {
      setOllamaLoading(false);
    }
  }

  const displayStatus = testing ? "testing" : (conn?.status ?? "unconfigured");
  const statusDotClass =
    displayStatus === "connected"
      ? "bg-keep"
      : displayStatus === "error"
        ? "bg-reject"
        : displayStatus === "configured_unverified" || displayStatus === "testing"
          ? "bg-unsure"
          : "bg-ink-3";

  const verifiedTime = formatVerifiedTime(conn?.verified_at ?? null);
  const sameProviderAsSaved = Boolean(conn?.configured && conn.provider === provider);
  const savedBaseNormalized = (conn?.base_url ?? "").trim().replace(/\/+$/, "");
  const draftBaseNormalized = baseUrl.trim().replace(/\/+$/, "");
  const endpointChangedWithoutKey =
    sameProviderAsSaved &&
    provider !== "ollama" &&
    draftBaseNormalized !== savedBaseNormalized &&
    !apiKey.trim();

  const activeAgentProbes = testResult?.agent_probes ?? conn?.agent_probes ?? null;
  const activeTriageProbes = testResult?.triage_probes ?? conn?.triage_probes ?? null;
  const showRoleDegraded =
    Boolean(conn?.configured) &&
    !conn?.agent_ready &&
    Boolean(conn?.triage_ready) &&
    (conn?.agent_probes !== null || conn?.agent_error !== null);

  const errorBannerText = (() => {
    if (!conn || conn.status !== "error" || !conn.error) return null;
    const base = formatErrorClass(conn.error.error_class, conn.error.error_code, conn.error.message);
    const rolePrefix =
      conn.error.role === "triage_vlm"
        ? `${t.aiRoleTriage}: `
        : conn.error.role === "agent"
          ? `${t.aiRoleAgent}: `
          : "";
    return `${rolePrefix}${base}`;
  })();

  const activeErrorClass = testResult?.error_class ?? conn?.error?.error_class ?? null;
  const fallbackProbesUntested = Boolean(
    activeErrorClass && CONNECTIVITY_OR_ACCOUNT_ERROR_CLASSES.has(activeErrorClass),
  );
  const showBannerError = Boolean(errorBannerText) && !(expanded && formError);

  const transientText = (() => {
    if (!conn?.transient_notice) return null;
    const cls = conn.transient_notice.error_class;
    return `${t.aiTransientPrefix}${t.aiErrorClassLabels[cls] ?? cls}`;
  })();

  const busy = testing || saving || deleting;

  return (
    <div className="flex flex-col gap-3 rounded-panel bg-table px-4 py-3">
      {/* Compact single-surface status bar */}
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1 space-y-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-sm">
            <span
              className={`h-2 w-2 shrink-0 rounded-full ${statusDotClass} ${testing ? "animate-pulse" : ""}`}
            />
            <span className="font-medium text-ink">
              {t.aiStatusLabels[displayStatus] ?? displayStatus}
            </span>
            {conn?.configured && conn.agent_model && (
              <span className="truncate text-ink-2">{conn.agent_model}</span>
            )}
            {displayStatus === "connected" && verifiedTime && (
              <span className="text-xs text-ink-3 tabular-nums">{t.aiVerifiedAt(verifiedTime)}</span>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-ink-2">
            <span>
              {t.aiCredentialSourceLabels[conn?.credential_source ?? "none"]}
              {conn?.key_suffix ? ` · ${formatKeySuffix(conn.key_suffix)}` : ""}
            </span>
            {conn?.configured && (
              <span>
                {conn.review_mode === "two_layer"
                  ? t.aiReviewModeTwoLayer
                  : t.aiReviewModeRulesOnly}
              </span>
            )}
            {conn?.configured && conn.cost_classification && (
              <span
                className={
                  conn.cost_classification === "price_unknown"
                    ? "font-medium text-unsure-ink"
                    : "text-ink-2"
                }
              >
                {t.aiCostLabels[conn.cost_classification]}
              </span>
            )}
          </div>

          {showBannerError && <p className="text-xs text-reject-ink">{errorBannerText}</p>}
          {!errorBannerText && showRoleDegraded && (
            <p className="text-xs text-unsure-ink">{t.aiAgentDisabledHint}</p>
          )}
          {transientText && <p className="text-xs text-unsure-ink">{transientText}</p>}
          {formNotice && <p className="text-xs text-ink-2">{formNotice}</p>}
        </div>

        <Button
          variant="ghost"
          size="sm"
          iconRight={expanded ? ChevronUp : ChevronDown}
          onClick={() => {
            setExpanded((prev) => !prev);
            setFormError(null);
            setConfirmDeleteCred(false);
            if (!expanded && conn) {
              syncFormFromConn(conn);
            }
          }}
        >
          {expanded ? t.aiCollapseBtn : conn?.configured ? t.aiEditBtn : t.aiSetupBtn}
        </Button>
      </div>

      {/* Inline expandable configuration form (API Key & Ollama only; no MCP here) */}
      {expanded && (
        <div className="fade-enter flex flex-col gap-3 pt-1">
          {/* Provider */}
          <label className="flex flex-col gap-1 text-xs text-ink-2">
            <span>{t.aiFieldProvider}</span>
            <select
              value={provider}
              disabled={busy}
              onChange={(e) => handleProviderChange(e.target.value as AiProvider)}
              className="field h-9 bg-paper px-3 text-sm"
            >
              {PROVIDERS.map((p) => (
                <option key={p} value={p}>
                  {t.aiProviderLabels[p]}
                </option>
              ))}
            </select>
          </label>

          {/* API Key (non-Ollama) */}
          {provider !== "ollama" && (
            <label className="flex flex-col gap-1 text-xs text-ink-2">
              <span>{t.aiFieldApiKey}</span>
              <input
                type="password"
                autoComplete="off"
                value={apiKey}
                disabled={busy}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={
                  sameProviderAsSaved && conn?.key_suffix
                    ? t.aiFieldApiKeyKeepHint(formatKeySuffix(conn.key_suffix))
                    : t.aiFieldApiKeyPlaceholder
                }
                className="field h-9 bg-paper px-3 text-sm"
              />
            </label>
          )}

          {/* Base URL (Ollama with model fetch, Custom required, Cloud optional) */}
          <div className="flex flex-col gap-1 text-xs text-ink-2">
            <span>
              {provider === "ollama"
                ? t.aiFieldOllamaBaseUrl
                : provider === "custom"
                  ? t.aiFieldBaseUrlRequired
                  : t.aiFieldBaseUrl}
            </span>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={baseUrl}
                disabled={busy}
                required={provider === "custom"}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder={
                  provider === "ollama"
                    ? "http://127.0.0.1:11434"
                    : provider === "custom"
                      ? t.aiFieldBaseUrlRequiredPlaceholder
                      : conn?.presets[provider]?.default_base_url || t.aiFieldBaseUrlOptional
                }
                className="field h-9 min-w-0 flex-1 bg-paper px-3 text-sm"
              />
              {provider === "ollama" && (
                <Button
                  variant="secondary"
                  size="sm"
                  icon={Refresh}
                  disabled={busy || ollamaLoading}
                  onClick={() => void handleFetchOllama()}
                  className="h-9 bg-paper hover:bg-table-2"
                >
                  {ollamaLoading ? t.aiOllamaFetching : t.aiOllamaFetchBtn}
                </Button>
              )}
            </div>
            {endpointChangedWithoutKey && (
              <span className="text-xs text-unsure-ink">{t.aiFieldBaseUrlRebindWarn}</span>
            )}
          </div>

          {/* Ollama discovered model picker + num_ctx note */}
          {provider === "ollama" && ollamaData && (
            <div className="flex flex-col gap-1.5 text-xs">
              {!ollamaData.reachable ? (
                <span className="text-reject-ink">{t.aiOllamaUnreachable}</span>
              ) : ollamaData.models.length === 0 ? (
                <span className="text-ink-3">{t.aiOllamaNoModels}</span>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {ollamaData.models.slice(0, 8).map((m) => {
                    const active = agentModel === m.name;
                    return (
                      <button
                        key={m.name}
                        type="button"
                        onClick={() => {
                          setAgentModel(m.name);
                          setTriageModel(m.name);
                        }}
                        className={`inline-flex h-7 items-center gap-1.5 rounded-chip px-2.5 text-xs transition-colors ${
                          active
                            ? "bg-ink text-white"
                            : "bg-paper text-ink-2 hover:bg-table-2 hover:text-ink"
                        }`}
                      >
                        <span>{m.name}</span>
                        {m.context_length_limit && (
                          <span className={active ? "text-white/70" : "text-ink-3"}>
                            {t.aiOllamaContextLimit(m.context_length_limit)}
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              )}
              <span className="text-ink-3">{t.aiOllamaContextNote}</span>
            </div>
          )}

          {/* Dual models: Agent + Triage */}
          <div className="grid grid-cols-2 gap-2">
            <label className="flex flex-col gap-1 text-xs text-ink-2">
              <span>{t.aiFieldAgentModel}</span>
              <input
                type="text"
                value={agentModel}
                disabled={busy}
                onChange={(e) => setAgentModel(e.target.value)}
                className="field h-9 bg-paper px-3 text-sm"
              />
            </label>
            <label className="flex flex-col gap-1 text-xs text-ink-2">
              <span>{t.aiFieldTriageModel}</span>
              <input
                type="text"
                value={triageModel}
                disabled={busy}
                onChange={(e) => setTriageModel(e.target.value)}
                className="field h-9 bg-paper px-3 text-sm"
              />
            </label>
          </div>

          {/* Local free flag & optional custom pricing for non-Ollama */}
          {provider !== "ollama" && (
            <div className="flex flex-col gap-2">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                {provider === "custom" ? (
                  <label className="inline-flex cursor-pointer items-center gap-2 text-ink-2">
                    <input
                      type="checkbox"
                      checked={isLocalFree}
                      disabled={busy}
                      onChange={(e) => setIsLocalFree(e.target.checked)}
                      className="h-3.5 w-3.5 accent-ink"
                    />
                    <span>{t.aiFieldLocalFree}</span>
                  </label>
                ) : (
                  <span />
                )}
                {!isLocalFree && (
                  <button
                    type="button"
                    onClick={() => setShowPricing((p) => !p)}
                    className="text-ink-2 underline-offset-2 hover:text-ink hover:underline"
                  >
                    {t.aiPricingToggle}
                  </button>
                )}
              </div>

              {showPricing && !isLocalFree && (
                <div className="grid grid-cols-2 gap-2">
                  <label className="flex flex-col gap-1 text-xs text-ink-2">
                    <span>{t.aiFieldInputCost}</span>
                    <input
                      type="number"
                      min="0"
                      step="any"
                      value={inputCost}
                      disabled={busy}
                      onChange={(e) => setInputCost(e.target.value)}
                      placeholder="0.00"
                      className="field h-8 bg-paper px-3 text-xs tabular-nums"
                    />
                  </label>
                  <label className="flex flex-col gap-1 text-xs text-ink-2">
                    <span>{t.aiFieldOutputCost}</span>
                    <input
                      type="number"
                      min="0"
                      step="any"
                      value={outputCost}
                      disabled={busy}
                      onChange={(e) => setOutputCost(e.target.value)}
                      placeholder="0.00"
                      className="field h-8 bg-paper px-3 text-xs tabular-nums"
                    />
                  </label>
                  <label className="flex flex-col gap-1 text-xs text-ink-2">
                    <span>{t.aiFieldTriageInputCost}</span>
                    <input
                      type="number"
                      min="0"
                      step="any"
                      value={triageInputCost}
                      disabled={busy}
                      onChange={(e) => setTriageInputCost(e.target.value)}
                      placeholder="0.00"
                      className="field h-8 bg-paper px-3 text-xs tabular-nums"
                    />
                  </label>
                  <label className="flex flex-col gap-1 text-xs text-ink-2">
                    <span>{t.aiFieldTriageOutputCost}</span>
                    <input
                      type="number"
                      min="0"
                      step="any"
                      value={triageOutputCost}
                      disabled={busy}
                      onChange={(e) => setTriageOutputCost(e.target.value)}
                      placeholder="0.00"
                      className="field h-8 bg-paper px-3 text-xs tabular-nums"
                    />
                  </label>
                </div>
              )}
            </div>
          )}

          {/* Per-role probe results (flat rows, zero nested boxes) */}
          {(activeAgentProbes || activeTriageProbes) && (
            <div className="flex flex-col gap-1 text-xs text-ink-2">
              {activeAgentProbes && (
                <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5">
                  <span className="font-medium text-ink">{t.aiRoleAgent}</span>
                  <ProbeTag
                    label={t.aiProbeVision}
                    probe={activeAgentProbes.vision}
                    fallbackUntested={fallbackProbesUntested}
                    untestedLabel={t.aiProbeUntested}
                  />
                  <ProbeTag
                    label={t.aiProbeTools}
                    probe={activeAgentProbes.tools}
                    fallbackUntested={fallbackProbesUntested}
                    untestedLabel={t.aiProbeUntested}
                  />
                  <ProbeTag
                    label={t.aiProbeJson}
                    probe={activeAgentProbes.json_schema}
                    fallbackUntested={fallbackProbesUntested}
                    untestedLabel={t.aiProbeUntested}
                  />
                </div>
              )}
              {activeTriageProbes && (
                <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5">
                  <span className="font-medium text-ink">{t.aiRoleTriage}</span>
                  <ProbeTag
                    label={t.aiProbeVision}
                    probe={activeTriageProbes.vision}
                    fallbackUntested={fallbackProbesUntested}
                    untestedLabel={t.aiProbeUntested}
                  />
                  <ProbeTag
                    label={t.aiProbeJson}
                    probe={activeTriageProbes.json_schema}
                    fallbackUntested={fallbackProbesUntested}
                    untestedLabel={t.aiProbeUntested}
                  />
                </div>
              )}
            </div>
          )}

          {formError && <p className="text-xs text-reject-ink">{formError}</p>}

          {/* Action buttons */}
          <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
            <div className="flex items-center gap-2">
              <Button
                variant="primary"
                size="sm"
                disabled={busy}
                onClick={() => void handleTestAndConnect()}
              >
                {testing ? t.aiTestingBtn : t.aiTestAndConnectBtn}
              </Button>
              <Button
                variant="secondary"
                size="sm"
                disabled={busy}
                onClick={() => void handleSaveOnly()}
                className="bg-paper hover:bg-table-2"
              >
                {saving ? t.aiSavingBtn : t.aiSaveBtn}
              </Button>
            </div>

            {conn?.configured && !confirmDeleteCred && (
              <div className="flex items-center gap-3">
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={busy}
                  onClick={() => void handleDelete(false)}
                >
                  {t.aiDeactivateBtn}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  icon={Trash}
                  disabled={busy || !conn.can_delete_credential}
                  title={!conn.can_delete_credential ? t.aiEnvDeleteDisabledHint : undefined}
                  onClick={() => setConfirmDeleteCred(true)}
                  className={conn.can_delete_credential ? "text-reject-ink hover:text-reject" : ""}
                >
                  {t.aiDeleteCredentialBtn}
                </Button>
              </div>
            )}
          </div>

          {conn?.configured && confirmDeleteCred && (
            <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
              <span className="text-xs text-reject-ink">{t.aiDeleteCredentialConfirmText}</span>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={busy}
                  onClick={() => setConfirmDeleteCred(false)}
                >
                  {t.aiDeleteCredentialCancelBtn}
                </Button>
                <Button
                  variant="mark"
                  size="sm"
                  disabled={busy}
                  onClick={() => void handleDelete(true)}
                >
                  {t.aiDeleteCredentialConfirmBtn}
                </Button>
              </div>
            </div>
          )}

          {conn?.configured && conn.credential_source === "env" && (
            <p className="text-xs text-ink-3">{t.aiEnvDeleteDisabledHint}</p>
          )}
        </div>
      )}
    </div>
  );
}
