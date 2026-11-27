import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { AgentDrawer } from "./components/AgentDrawer";
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
    void api
      .checkAuth()
      .then((res) => {
        if (cancelled) return;
        setAuthState(res ? "authed" : "needs_confirm");
      })
      .catch((err) => {
        if (cancelled) return;
        setAuthError(err instanceof Error ? err.message : String(err));
        setAuthState("needs_confirm");
      });
    return () => {
      cancelled = true;
    };
  }, []);

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

  return (
    <I18nContext.Provider value={i18nValue}>
      <div className="min-h-screen bg-[#1B1C1E] text-[#F3F4F6]">
        {/* Quiet top navigation bar */}
        <header className="flex h-[52px] items-center justify-between border-b border-[#33363B] bg-[#1B1C1E] px-5">
          <div className="flex items-center gap-6">
            <button
              type="button"
              onClick={() => navigate("/")}
              className="text-base font-semibold tracking-tight text-[#F3F4F6]"
            >
              {t.appName}
            </button>
            {authState === "authed" && (
              <nav className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => navigate("/")}
                  className={`rounded-md px-3 py-1.5 text-sm ${
                    route.name === "sessions"
                      ? "bg-[#242629] text-[#F3F4F6]"
                      : "text-[#9CA3AF] hover:text-[#F3F4F6]"
                  }`}
                >
                  {t.navSessions}
                </button>
                <button
                  type="button"
                  onClick={() => navigate("/search")}
                  className={`rounded-md px-3 py-1.5 text-sm ${
                    route.name === "search"
                      ? "bg-[#242629] text-[#F3F4F6]"
                      : "text-[#9CA3AF] hover:text-[#F3F4F6]"
                  }`}
                >
                  {t.navSearch}
                </button>
                <button
                  type="button"
                  onClick={() => navigate("/people")}
                  className={`rounded-md px-3 py-1.5 text-sm ${
                    route.name === "people"
                      ? "bg-[#242629] text-[#F3F4F6]"
                      : "text-[#9CA3AF] hover:text-[#F3F4F6]"
                  }`}
                >
                  {t.navPeople}
                </button>
                <button
                  type="button"
                  onClick={() => navigate("/insights")}
                  className={`rounded-md px-3 py-1.5 text-sm ${
                    route.name === "insights"
                      ? "bg-[#242629] text-[#F3F4F6]"
                      : "text-[#9CA3AF] hover:text-[#F3F4F6]"
                  }`}
                >
                  {t.navRestore}
                </button>
              </nav>
            )}
          </div>

          <div className="flex items-center gap-2">
            {authState === "authed" && (
              <button
                type="button"
                onClick={() => setAgentOpen((prev) => !prev)}
                className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs transition ${
                  agentOpen
                    ? "border-[#5B8DEF]/60 bg-[#5B8DEF]/20 text-[#93C5FD]"
                    : "border-[#33363B] bg-[#242629] text-[#E5E7EB] hover:bg-[#2C2F34]"
                }`}
              >
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-[#5B8DEF]" />
                <span>{t.navAgentBtn}</span>
                <kbd className="font-mono text-[10px] text-[#9CA3AF]">⌘K</kbd>
              </button>
            )}
            <button
              type="button"
              onClick={() => setLang(lang === "zh-CN" ? "en-US" : "zh-CN")}
              className="rounded-md border border-[#33363B] px-2.5 py-1 text-xs text-[#D1D5DB] hover:bg-[#242629]"
            >
              {lang === "zh-CN" ? "EN" : "中文"}
            </button>
          </div>
        </header>

        {authState === "checking" ? (
          <div className="flex h-[calc(100vh-52px)] items-center justify-center text-sm text-[#9CA3AF]">
            {t.loading}
          </div>
        ) : authState === "needs_confirm" ? (
          <div className="mx-auto flex max-w-md flex-col items-start gap-4 px-6 py-24">
            <h1 className="text-xl font-medium text-[#F3F4F6]">{t.authPromptTitle}</h1>
            <p className="text-sm leading-relaxed text-[#9CA3AF]">{t.authPromptDesc}</p>
            {authError && <p className="text-sm text-[#FCA5A5]">{authError}</p>}
            <button
              type="button"
              onClick={() => void handleConfirmLocal()}
              className="rounded-md bg-[#F3F4F6] px-4 py-2 text-sm font-medium text-[#1B1C1E] hover:bg-white"
            >
              {t.authPromptAction}
            </button>
          </div>
        ) : route.name === "review" ? (
          <ReviewView
            sessionId={route.sessionId}
            onBack={() => navigate("/")}
            onGoInsights={() => navigate("/insights?tab=trash")}
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
          <AgentDrawer
            open={agentOpen}
            activeSessionId={route.name === "review" ? route.sessionId : null}
            onClose={() => setAgentOpen(false)}
          />
        )}
      </div>
    </I18nContext.Provider>
  );
}
