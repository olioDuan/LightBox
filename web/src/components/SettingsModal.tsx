import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type McpClientId, type McpClientStatus, type McpSnippetResponse } from "../api";
import { useI18n, type Lang } from "../i18n";
import { Button, Dialog, IconButton, Loading, Tabs } from "./ui";
import { Check, Copy, Refresh, X } from "reicon-react";

interface SettingsModalProps {
  open: boolean;
  onClose: () => void;
}

type SettingsTab = "general" | "mcp";

function mcpStatusDot(status: McpClientStatus): string {
  if (status === "configured") return "bg-keep";
  if (status === "invalid_path" || status === "parse_error") return "bg-unsure";
  return "bg-ink-3";
}

export function SettingsModal({ open, onClose }: SettingsModalProps) {
  const { lang, setLang, t } = useI18n();
  const queryClient = useQueryClient();

  const [tab, setTab] = useState<SettingsTab>("general");

  // General form state
  const [sensitivity, setSensitivity] = useState<"conservative" | "standard">("conservative");
  const [modelProfile, setModelProfile] = useState<"local" | "balanced" | "best">("balanced");
  const [sessionBudget, setSessionBudget] = useState("0.50");
  const [monthlyBudget, setMonthlyBudget] = useState("10.00");
  const [saving, setSaving] = useState(false);
  const [saveNotice, setSaveNotice] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  // MCP panel state
  const [selectedClient, setSelectedClient] = useState<McpClientId>("claude_desktop");
  const [snippet, setSnippet] = useState<McpSnippetResponse | null>(null);
  const [snippetLoading, setSnippetLoading] = useState(false);
  const [snippetError, setSnippetError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const settingsQ = useQuery({
    queryKey: ["settings"],
    queryFn: () => api.getSettings(),
    enabled: open,
  });

  const aiConnQ = useQuery({
    queryKey: ["ai-connection"],
    queryFn: () => api.getAiConnection(),
    enabled: open,
  });

  const mcpClientsQ = useQuery({
    queryKey: ["mcp-clients"],
    queryFn: () => api.getMcpClients(),
    enabled: open && tab === "mcp",
  });

  useEffect(() => {
    if (!settingsQ.data) return;
    setSensitivity(settingsQ.data.sensitivity);
    setModelProfile(settingsQ.data.model_profile);
    setSessionBudget(String(settingsQ.data.budget.session_budget_usd));
    setMonthlyBudget(String(settingsQ.data.budget.monthly_budget_usd));
  }, [settingsQ.data]);

  useEffect(() => {
    if (!open || tab !== "mcp") return;
    let cancelled = false;
    setSnippetLoading(true);
    setSnippetError(null);
    setCopied(false);
    void api
      .getMcpSnippet(selectedClient)
      .then((res) => {
        if (!cancelled) setSnippet(res);
      })
      .catch((err) => {
        if (!cancelled) setSnippetError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setSnippetLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, tab, selectedClient]);

  if (!open) return null;

  const aiManaged = Boolean(aiConnQ.data?.configured || settingsQ.data?.ai);

  async function handleSaveGeneral() {
    setSaving(true);
    setSaveError(null);
    setSaveNotice(null);
    const parsedSession = Number(sessionBudget);
    const parsedMonthly = Number(monthlyBudget);
    try {
      const updated = await api.updateSettings({
        language: lang,
        sensitivity,
        ...(!aiManaged ? { model_profile: modelProfile } : {}),
        budget: {
          ...(Number.isFinite(parsedSession) && parsedSession >= 0
            ? { session_budget_usd: parsedSession }
            : {}),
          ...(Number.isFinite(parsedMonthly) && parsedMonthly >= 0
            ? { monthly_budget_usd: parsedMonthly }
            : {}),
        },
      });
      queryClient.setQueryData(["settings"], updated);
      void queryClient.invalidateQueries({ queryKey: ["llm-traces"] });
      setSaveNotice(t.settingsSavedNotice);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  async function handleCopySnippet() {
    if (!snippet) return;
    try {
      await navigator.clipboard.writeText(snippet.snippet);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      // fallback selection copy
      const ta = document.createElement("textarea");
      ta.value = snippet.snippet;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    }
  }

  const clients = mcpClientsQ.data?.clients ?? [];

  return (
    <Dialog labelledBy="settings-modal-title" onDismiss={onClose} width={600}>
      <div className="flex max-h-[88vh] flex-col">
        {/* Header */}
        <div className="flex shrink-0 flex-col gap-4 px-8 pb-4 pt-8">
          <div className="flex items-center justify-between gap-4">
            <h2 id="settings-modal-title" className="text-lg font-semibold text-ink">
              {t.settingsTitle}
            </h2>
            <IconButton icon={X} label={t.uiDismiss} onClick={onClose} />
          </div>
          <Tabs
            variant="segmented"
            value={tab}
            onChange={(next) => {
              setTab(next);
              setSaveNotice(null);
              setSaveError(null);
            }}
            items={[
              { id: "general", label: t.settingsTabGeneral },
              { id: "mcp", label: t.settingsTabMcp },
            ]}
          />
        </div>

        {/* Content */}
        <div className="min-h-0 flex-1 overflow-y-auto px-8 pb-8 pt-4">
          {tab === "general" ? (
            settingsQ.isLoading ? (
              <Loading label={t.loading} className="py-12" />
            ) : (
              <div className="flex flex-col gap-6">
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <label className="flex flex-col gap-2 text-sm text-ink-2">
                    <span>{t.settingsLanguage}</span>
                    <select
                      value={lang}
                      onChange={(e) => setLang(e.target.value as Lang)}
                      className="field"
                    >
                      <option value="zh-CN">中文</option>
                      <option value="en-US">English</option>
                    </select>
                  </label>

                  <label className="flex flex-col gap-2 text-sm text-ink-2">
                    <span>{t.settingsSensitivity}</span>
                    <select
                      value={sensitivity}
                      onChange={(e) =>
                        setSensitivity(e.target.value as "conservative" | "standard")
                      }
                      className="field"
                    >
                      <option value="conservative">{t.settingsSensitivityConservative}</option>
                      <option value="standard">{t.settingsSensitivityStandard}</option>
                    </select>
                  </label>
                </div>

                <label className="flex flex-col gap-2 text-sm text-ink-2">
                  <span className="flex items-center justify-between gap-2">
                    <span>{t.settingsModelProfile}</span>
                    {aiManaged && (
                      <span className="inline-flex items-center gap-1.5 rounded-chip bg-table px-2 py-0.5 text-xs font-medium text-ink">
                        <span className="h-1.5 w-1.5 rounded-full bg-keep" />
                        {t.settingsProfileManagedByAi}
                      </span>
                    )}
                  </span>
                  <select
                    value={modelProfile}
                    disabled={aiManaged}
                    onChange={(e) =>
                      setModelProfile(e.target.value as "local" | "balanced" | "best")
                    }
                    className="field"
                  >
                    <option value="local">{t.settingsProfileLocal}</option>
                    <option value="balanced">{t.settingsProfileBalanced}</option>
                    <option value="best">{t.settingsProfileBest}</option>
                  </select>
                </label>

                <div className="grid grid-cols-2 gap-4">
                  <label className="flex flex-col gap-2 text-sm text-ink-2">
                    <span>{t.settingsSessionBudget}</span>
                    <input
                      type="number"
                      min="0"
                      step="0.05"
                      value={sessionBudget}
                      onChange={(e) => setSessionBudget(e.target.value)}
                      className="field tabular-nums"
                    />
                  </label>
                  <label className="flex flex-col gap-2 text-sm text-ink-2">
                    <span>{t.settingsMonthlyBudget}</span>
                    <input
                      type="number"
                      min="0"
                      step="0.5"
                      value={monthlyBudget}
                      onChange={(e) => setMonthlyBudget(e.target.value)}
                      className="field tabular-nums"
                    />
                  </label>
                </div>

                {saveError && <p className="text-sm text-reject-ink">{saveError}</p>}

                <div className="flex items-center justify-between gap-4 pt-2">
                  <span className="text-sm text-keep-ink">{saveNotice ?? ""}</span>
                  <Button
                    variant="primary"
                    disabled={saving}
                    onClick={() => void handleSaveGeneral()}
                  >
                    {saving ? t.aiSavingBtn : t.settingsSaveBtn}
                  </Button>
                </div>
              </div>
            )
          ) : (
            <div className="flex flex-col gap-6">
              <div className="flex items-center justify-between gap-4">
                <p className="text-xs text-ink-3">{t.mcpScopeHint}</p>
                <Button
                  variant="ghost"
                  size="sm"
                  icon={Refresh}
                  disabled={mcpClientsQ.isFetching}
                  onClick={() => void mcpClientsQ.refetch()}
                >
                  {t.mcpRefreshBtn}
                </Button>
              </div>

              {/* Client status list (flat rows, no nested borders) */}
              {mcpClientsQ.isLoading ? (
                <Loading label={t.loading} className="py-8" />
              ) : (
                <ul className="flex flex-col gap-1">
                  {clients.map((c) => {
                    const active = selectedClient === c.client_id;
                    return (
                      <li key={c.client_id}>
                        <button
                          type="button"
                          onClick={() => setSelectedClient(c.client_id)}
                          className={`flex h-11 w-full items-center justify-between gap-4 rounded-btn px-4 text-left text-sm transition-colors ${
                            active ? "bg-table font-medium text-ink" : "text-ink-2 hover:bg-table/60 hover:text-ink"
                          }`}
                        >
                          <span className="truncate">
                            {t.mcpClientNames[c.client_id] ?? c.client_id}
                          </span>
                          <span className="inline-flex shrink-0 items-center gap-2 text-xs text-ink-2">
                            <span className={`h-2 w-2 rounded-full ${mcpStatusDot(c.status)}`} />
                            <span>{t.mcpClientStatuses[c.status] ?? c.status}</span>
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}

              {/* Selected client's native copy snippet */}
              {snippetLoading ? (
                <Loading label={t.loading} className="py-6" />
              ) : snippetError ? (
                <p className="text-sm text-reject-ink">{snippetError}</p>
              ) : snippet ? (
                <div className="flex flex-col gap-3 rounded-panel bg-table p-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-sm font-medium text-ink">
                          {t.mcpClientNames[snippet.client_id] ?? snippet.client_id}
                        </span>
                        {snippet.restart_required && (
                          <span className="rounded-chip bg-paper px-2 py-0.5 text-xs text-ink-2">
                            {t.mcpRestartBadge}
                          </span>
                        )}
                      </div>
                      <p className="mt-0.5 truncate text-xs text-ink-3" title={snippet.config_path}>
                        {snippet.config_path}
                      </p>
                    </div>
                    <Button
                      variant="primary"
                      size="sm"
                      icon={copied ? Check : Copy}
                      onClick={() => void handleCopySnippet()}
                    >
                      {copied ? t.mcpCopiedBtn : t.mcpCopyBtn}
                    </Button>
                  </div>

                  <pre className="overflow-x-auto whitespace-pre rounded-btn bg-paper p-3 font-mono text-xs leading-5 text-ink">
                    {snippet.snippet}
                  </pre>
                </div>
              ) : null}
            </div>
          )}
        </div>
      </div>
    </Dialog>
  );
}
