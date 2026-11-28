/**
 * Shared `pywebview` desktop bridge detection with bounded `pywebviewready` wait.
 *
 * Used by desktop-only auto-login, native OS folder picker, and 3-failure reconnect
 * so all three paths share the exact same readiness decision and never hang browser mode.
 *
 * Only a resolved non-null bridge is cached: if an early check times out (e.g. slow
 * WebView2 injection on Windows), subsequent calls re-check `window.pywebview?.api`
 * and re-wait instead of permanently locking the UI into `null`.
 */

export interface DesktopJsApiBridge {
  pick_folder?: () => Promise<{ path: string | null; canceled: boolean }>;
  reconnect?: () => Promise<{ ok: boolean; url?: string }>;
  get_lifecycle_status?: () => Promise<{ owns_server: boolean; in_flight_file_ops: number }>;
}

declare global {
  interface Window {
    pywebview?: {
      api?: DesktopJsApiBridge;
    };
  }
}

let cachedBridge: DesktopJsApiBridge | null = null;
let inflightBridgeWait: Promise<DesktopJsApiBridge | null> | null = null;

function readBridgeNow(): DesktopJsApiBridge | null {
  if (typeof window === "undefined") return null;
  return window.pywebview?.api ?? null;
}

/**
 * Wait for `window.pywebview.api` (listening for `pywebviewready` up to `timeoutMs`).
 * Resolves to the bridge object in desktop mode, or `null` in standard browsers / on timeout.
 */
export function waitForDesktopBridge(timeoutMs = 450): Promise<DesktopJsApiBridge | null> {
  if (cachedBridge) {
    return Promise.resolve(cachedBridge);
  }
  const existing = readBridgeNow();
  if (existing) {
    cachedBridge = existing;
    return Promise.resolve(existing);
  }
  if (typeof window === "undefined") {
    return Promise.resolve(null);
  }
  if (inflightBridgeWait) {
    return inflightBridgeWait;
  }

  inflightBridgeWait = new Promise<DesktopJsApiBridge | null>((resolve) => {
    let settled = false;

    const finish = (bridge: DesktopJsApiBridge | null) => {
      if (settled) return;
      settled = true;
      inflightBridgeWait = null;
      window.removeEventListener("pywebviewready", onReady);
      window.clearTimeout(timer);
      if (bridge) {
        cachedBridge = bridge;
      }
      resolve(bridge);
    };

    const onReady = () => {
      finish(readBridgeNow());
    };

    window.addEventListener("pywebviewready", onReady, { once: true });
    const timer = window.setTimeout(() => {
      finish(readBridgeNow());
    }, timeoutMs);
  });

  return inflightBridgeWait;
}
