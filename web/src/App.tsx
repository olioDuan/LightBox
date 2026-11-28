import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { AgentDrawer } from "./components/AgentDrawer";
import { SettingsModal } from "./components/SettingsModal";
import { Button, IconButton, Loading, Notice, NoticeDock, Tabs, type TabItem } from "./components/ui";
import { waitForDesktopBridge } from "./desktop";
import { Settings, Sparkles } from "reicon-react";
import { getMessages, I18nContext, type Lang } from "./i18n";
import { InsightsView } from "./views/InsightsView";
import { PeopleView } from "./views/PeopleView";
import { ReviewView } from "./views/ReviewView";
import { SearchView } from "./views/SearchView";
import { SessionsView } from "./views/SessionsView";

type InsightsTab = "digest" | "trash" | "traces" | "memory";

type Route =
  | { name: "sessions" }
  | { name: "review"; sessionId: string }
  | { name: "search" }
  | { name: "people" }
  | { name: "insights"; tab: InsightsTab };

function parseRoute(pathname: string, search = ""): Route {
  const reviewMatch = pathname.match(/^\/sessions\/([^/]+)\/review\/?$/);
  if (reviewMatch) {
    return { name: "review", sessionId: decodeURIComponent(reviewMatch[1]) };
  }
  if (pathname === "/search" || pathname === "/search/") {
    return { name: "search" };
  }
  if (pathname === "/people" || pathname === "/people/") {
    return { name: "people" };
  }
  if (pathname === "/insights" || pathname === "/insights/") {
    const params = new URLSearchParams(search);
    const raw = params.get("tab");
    const tab: InsightsTab =
      raw === "trash"
        ? "trash"
        : raw === "traces"
          ? "traces"
          : raw === "memory"
            ? "memory"
            : "digest";
    return { name: "insights", tab };
  }
  return { name: "sessions" };
}

export function App() {
  const [route, setRoute] = useState<Route>(() =>
    parseRoute(window.location.pathname, window.location.search),
  );
  const [lang, setLangState] = useState<Lang>("zh-CN");
  const [authState, setAuthState] = useState<"checking" | "authed" | "needs_confirm">("checking");
  const [authError, setAuthError] = useState<string | null>(null);
  const [agentOpen, setAgentOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [serverDisconnected, setServerDisconnected] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const [closeBlockedNotice, setCloseBlockedNotice] = useState(false);
  const healthFailuresRef = useRef(0);

  // Global ⌘K / Ctrl+K shortcut to toggle the built-in Agent side drawer
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        if (authState !== "authed") return;
        e.preventDefault();
        setAgentOpen((prev) => !prev);
      } else if (e.key === "Escape" && agentOpen) {
        setAgentOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [agentOpen, authState]);

  const navigate = (path: string) => {
    window.history.pushState({}, "", path);
    const url = new URL(path, window.location.origin);
    setRoute(parseRoute(url.pathname, url.search));
  };

  useEffect(() => {
    const onPop = () => setRoute(parseRoute(window.location.pathname, window.location.search));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const res = await api.checkAuth();
        if (cancelled) return;
        if (res) {
          setAuthState("authed");
          return;
        }
        // Only auto-establish session when running inside the pywebview desktop bridge
        const bridge = await waitForDesktopBridge();
        if (cancelled) return;
        if (bridge !== null) {
          await api.confirmLocalSession();
          if (!cancelled) {
            setAuthState("authed");
          }
          return;
        }
        setAuthState("needs_confirm");
      } catch (err) {
        if (cancelled) return;
        setAuthError(err instanceof Error ? err.message : String(err));
        setAuthState("needs_confirm");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Listen for desktop close-blocked notifications while file moves are in flight
  useEffect(() => {
    const onCloseBlocked = () => setCloseBlockedNotice(true);
    window.addEventListener("lightbox:close-blocked", onCloseBlocked);
    return () => window.removeEventListener("lightbox:close-blocked", onCloseBlocked);
  }, []);

  // Probe `/healthz`; only show disconnected banner after 3 consecutive failures
  useEffect(() => {
    if (authState !== "authed") return;
    let cancelled = false;
    let inFlight = false;

    const checkHealth = async () => {
      if (cancelled || inFlight) return;
      inFlight = true;
      try {
        const ok = await api.probeHealthz();
        if (cancelled) return;
        if (ok) {
          healthFailuresRef.current = 0;
          setServerDisconnected(false);
        } else {
          healthFailuresRef.current += 1;
          if (healthFailuresRef.current >= 3) {
            setServerDisconnected(true);
          }
        }
      } finally {
        inFlight = false;
      }
    };

    const timer = window.setInterval(() => {
      void checkHealth();
    }, 5000);
    const onOnline = () => void checkHealth();
    const onVisibility = () => {
      if (document.visibilityState === "visible") {
        void checkHealth();
      }
    };
    window.addEventListener("online", onOnline);
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
      window.removeEventListener("online", onOnline);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [authState]);

  async function handleReconnect() {
    if (reconnecting) return;
    setReconnecting(true);
    try {
      const bridge = await waitForDesktopBridge();
      if (bridge?.reconnect) {
        await bridge.reconnect();
        return;
      }
      const ok = await api.probeHealthz();
      if (ok) {
        healthFailuresRef.current = 0;
        setServerDisconnected(false);
        window.location.reload();
      }
    } catch {
      // keep disconnect banner visible so user can retry
    } finally {
      setReconnecting(false);
    }
  }

  const setLang = (next: Lang) => {
    setLangState(next);
    document.documentElement.lang = next;
    if (authState === "authed") {
      void api.updateLanguage(next).catch(() => {
        // non-blocking preference sync
      });
    }
  };

  const t = useMemo(() => getMessages(lang), [lang]);
  const i18nValue = useMemo(() => ({ lang, setLang, t }), [lang, t]);

  async function handleConfirmLocal() {
    setAuthError(null);
    try {
      await api.confirmLocalSession();
      setAuthState("authed");
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : String(err));
    }
  }

  const navItems: TabItem<"sessions" | "search" | "people" | "insights">[] = [
    { id: "sessions", label: t.navSessions },
    { id: "search", label: t.navSearch },
    { id: "people", label: t.navPeople },
    { id: "insights", label: t.navRestore },
  ];
  const navValue = route.name === "review" ? "sessions" : route.name;

  return (
    <I18nContext.Provider value={i18nValue}>
      <div className="min-h-screen bg-paper text-ink">
        <header className="sticky top-0 z-30 flex h-16 items-center justify-between gap-4 bg-paper px-4 sm:px-8">
          <div className="flex min-w-0 items-center gap-4 md:gap-8">
            <button
              type="button"
              onClick={() => navigate("/")}
              aria-label={t.appName}
              className="flex items-center gap-2 text-lg font-semibold tracking-tight text-ink"
            >
              <span aria-hidden="true" className="h-4 w-4 rounded-[5px] border-[1.5px] border-ink" />
              <span className="hidden sm:inline">{t.appName}</span>
            </button>
            {authState === "authed" && (
              <nav className="scrollbar-none -mx-1 min-w-0 overflow-x-auto px-1">
                <Tabs
                  items={navItems}
                  value={navValue}
                  size="sm"
                  wrap={false}
                  onChange={(id) =>
                    navigate(id === "sessions" ? "/" : id === "insights" ? "/insights" : `/${id}`)
                  }
                />
              </nav>
            )}
          </div>

          <div className="flex shrink-0 items-center gap-1">
            {authState === "authed" && (
              <>
                <Button
                  variant={agentOpen ? "primary" : "ghost"}
                  size="sm"
                  icon={Sparkles}
                  aria-pressed={agentOpen}
                  aria-keyshortcuts="Meta+K Control+K"
                  title={`${t.navAgentBtn} ⌘K`}
                  onClick={() => setAgentOpen((prev) => !prev)}
                >
                  <span className="hidden md:inline">{t.navAgentBtn}</span>
                </Button>
                <IconButton
                  icon={Settings}
                  label={t.settingsBtnLabel}
                  size="sm"
                  variant={settingsOpen ? "secondary" : "ghost"}
                  onClick={() => setSettingsOpen(true)}
                />
              </>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setLang(lang === "zh-CN" ? "en-US" : "zh-CN")}
            >
              {lang === "zh-CN" ? "EN" : "中文"}
            </Button>
          </div>
        </header>

        {authState === "checking" ? (
          <Loading label={t.loading} className="h-[calc(100vh-64px)]" />
        ) : authState === "needs_confirm" ? (
          <div className="mx-auto flex min-h-[calc(100vh-64px)] max-w-sm flex-col items-center justify-center gap-8 px-6 pb-16 text-center">
            <h1 className="text-lg font-semibold text-ink">{t.authPromptTitle}</h1>
            {authError && <p className="text-sm text-mark">{authError}</p>}
            <Button variant="primary" size="lg" onClick={() => void handleConfirmLocal()}>
              {t.authPromptAction}
            </Button>
          </div>
        ) : route.name === "review" ? (
          <ReviewView
            sessionId={route.sessionId}
            onBack={() => navigate("/")}
            onGoInsights={() => navigate("/insights?tab=trash")}
            onOpenAgent={() => setAgentOpen(true)}
          />
        ) : route.name === "search" ? (
          <SearchView onOpenReview={(id) => navigate(`/sessions/${encodeURIComponent(id)}/review`)} />
        ) : route.name === "people" ? (
          <PeopleView onOpenReview={(id) => navigate(`/sessions/${encodeURIComponent(id)}/review`)} />
        ) : route.name === "insights" ? (
          <InsightsView
            activeTab={route.tab}
            onChangeTab={(tab) =>
              navigate(tab === "digest" ? "/insights" : `/insights?tab=${tab}`)
            }
          />
        ) : (
          <SessionsView onOpenReview={(id) => navigate(`/sessions/${encodeURIComponent(id)}/review`)} />
        )}

        {authState === "authed" && (
          <>
            <AgentDrawer
              open={agentOpen}
              activeSessionId={route.name === "review" ? route.sessionId : null}
              onClose={() => setAgentOpen(false)}
            />
            <SettingsModal
              open={settingsOpen}
              onClose={() => setSettingsOpen(false)}
            />
          </>
        )}

        {(serverDisconnected || closeBlockedNotice) && (
          <NoticeDock bottom={24}>
            {closeBlockedNotice && (
              <Notice
                tone="error"
                onDismiss={() => setCloseBlockedNotice(false)}
                dismissLabel={t.uiDismiss}
              >
                {t.desktopCloseBlockedToast}
              </Notice>
            )}
            {serverDisconnected && (
              <Notice
                tone="error"
                actions={[
                  {
                    label: reconnecting ? t.desktopReconnectingBtn : t.desktopReconnectBtn,
                    onClick: () => void handleReconnect(),
                  },
                ]}
              >
                {t.desktopDisconnectedTitle}
              </Notice>
            )}
          </NoticeDock>
        )}
      </div>
    </I18nContext.Provider>
  );
}
