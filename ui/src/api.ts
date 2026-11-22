/**
 * Where quackd lives. Resolution order:
 *   1. `window.__QUACK_PORT__` — a host page or test harness pins it.
 *   2. The Tauri shell's `quack_port` command — the desktop app
 *      picks 1234, or 1235 when something else (LM Studio, another
 *      quackd) already owns 1234, and spawns the daemon itself.
 *   3. 1234 — plain web dev against a hand-started daemon.
 *
 * The Tauri global only exists when the app runs under Tauri with
 * `withGlobalTauri` enabled; the typeof guards keep the plain web
 * build working with no Tauri packages installed. The result is
 * cached: the port cannot change during a session.
 */

interface TauriCore {
  invoke: (cmd: string, args?: Record<string, unknown>) => Promise<unknown>;
}

declare global {
  interface Window {
    __QUACK_PORT__?: number;
    __TAURI__?: { core?: TauriCore };
  }
}

let cached: Promise<string> | null = null;

/** Base URL of the daemon, e.g. `http://127.0.0.1:1234`. */
export function apiBase(): Promise<string> {
  if (!cached) {
    cached = resolveApiBase();
  }
  return cached;
}

async function resolveApiBase(): Promise<string> {
  if (typeof window !== "undefined") {
    if (typeof window.__QUACK_PORT__ === "number") {
      return `http://127.0.0.1:${window.__QUACK_PORT__}`;
    }
    const invoke = window.__TAURI__?.core?.invoke;
    if (typeof invoke === "function") {
      try {
        const port: unknown = await invoke("quack_port");
        if (typeof port === "number" && port > 0) {
          return `http://127.0.0.1:${port}`;
        }
      } catch {
        // Not under Tauri after all — fall through to the default.
      }
    }
  }
  return "http://127.0.0.1:1234";
}
