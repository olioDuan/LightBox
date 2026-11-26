import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import { getMessages, I18nContext, type Lang } from "./i18n";
import { InsightsView } from "./views/InsightsView";
import { ReviewView } from "./views/ReviewView";
import { SessionsView } from "./views/SessionsView";

type Route =
  | { name: "sessions" }
  | { name: "review"; sessionId: string }
  | { name: "insights" };

function parseRoute(pathname: string): Route {
  const reviewMatch = pathname.match(/^\/sessions\/([^/]+)\/review\/?$/);
  if (reviewMatch) {
    return { name: "review", sessionId: decodeURIComponent(reviewMatch[1]) };
  }
  if (pathname === "/insights" || pathname === "/insights/") {
    return { name: "insights" };
  }
  return { name: "sessions" };
}

export function App() {
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.pathname));
  const [lang, setLangState] = useState<Lang>("zh-CN");
  const [authState, setAuthState] = useState<"checking" | "authed" | "needs_confirm">("checking");
  const [authError, setAuthError] = useState<string | null>(null);

  const navigate = (path: string) => {
    window.history.pushState({}, "", path);
    setRoute(parseRoute(path));
  };

  useEffect(() => {
    const onPop = () => setRoute(parseRoute(window.location.pathname));
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

          <button
            type="button"
            onClick={() => setLang(lang === "zh-CN" ? "en-US" : "zh-CN")}
            className="rounded-md border border-[#33363B] px-2.5 py-1 text-xs text-[#D1D5DB] hover:bg-[#242629]"
          >
            {lang === "zh-CN" ? "EN" : "中文"}
          </button>
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
            onGoInsights={() => navigate("/insights")}
          />
        ) : route.name === "insights" ? (
          <InsightsView />
        ) : (
          <SessionsView onOpenReview={(id) => navigate(`/sessions/${encodeURIComponent(id)}/review`)} />
        )}
      </div>
    </I18nContext.Provider>
  );
}
