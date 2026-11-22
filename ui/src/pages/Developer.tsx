import { useCallback, useEffect, useState } from "react";
import {
  ChevronDown,
  ChevronUp,
  Copy,
  FileJson,
  Loader2,
  Settings2,
  X,
} from "lucide-react";
import Toggle from "../components/Toggle";
import { apiBase } from "../api";
import type {
  LoadedInstance,
  ModelEntry,
  ServerSettings,
  SettingsView,
} from "../types";

const ENDPOINTS = [
  "POST /v1/chat/completions",
  "POST /v1/completions",
  "POST /v1/embeddings",
  "POST /v1/messages",
  "GET /v1/models",
  "GET /api/v1/models",
];

interface LoadedRow {
  instance: LoadedInstance;
  entry: ModelEntry | null;
  /** The instance's context length from /api/v1/status, if any. */
  contextLength: number | null;
}

const SERVER_DEFAULTS: ServerSettings = {
  require_auth: false,
  serve_lan: true,
  cors: true,
  jit: true,
  jit_ttl_minutes: 10,
  jit_only_last: false,
};

/** Developer (Model Server): status card with the Server Settings
 *  popover (persisted through the settings API; JIT is enforced
 *  by quackd), loaded-models list with a detail panel, supported
 *  endpoints, and the Server Logs strip fed by GET /api/v1/logs. */
export default function Developer() {
  const [running, setRunning] = useState(true);
  const [reachable, setReachable] = useState("http://127.0.0.1:1234");
  const [port, setPort] = useState(1234);
  const [rows, setRows] = useState<LoadedRow[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [server, setServer] = useState<ServerSettings>(SERVER_DEFAULTS);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const [logsOpen, setLogsOpen] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [unloading, setUnloading] = useState(false);
  const [copied, setCopied] = useState(false);

  const fetchAll = useCallback(async () => {
    try {
      const base = await apiBase();
      setReachable(base);
      const [modelsRes, statusRes] = await Promise.all([
        fetch(`${base}/api/v1/models`),
        fetch(`${base}/api/v1/status`),
      ]);
      if (modelsRes.ok) {
        const models = (await modelsRes.json()) as ModelEntry[];
        const ctxByKey = new Map<string, number | null>();
        if (statusRes.ok) {
          const status = (await statusRes.json()) as {
            port: number;
            instances: {
              instance_id: string;
              context_length: number | null;
            }[];
          };
          setPort(status.port);
          for (const inst of status.instances) {
            ctxByKey.set(inst.instance_id, inst.context_length);
          }
        }
        const flat: LoadedRow[] = [];
        for (const entry of models) {
          for (const instance of entry.loaded_instances) {
            flat.push({
              instance,
              entry,
              contextLength: ctxByKey.get(instance.instance_id) ?? null,
            });
          }
        }
        setRows(flat);
      }
    } catch {
      // Page renders its empty state when quackd is unreachable.
    }
  }, []);

  useEffect(() => {
    void fetchAll();
    const timer = setInterval(() => void fetchAll(), 5000);
    return () => clearInterval(timer);
  }, [fetchAll]);

  // Server settings draft from the settings API.
  useEffect(() => {
    void (async () => {
      try {
        const base = await apiBase();
        const res = await fetch(`${base}/api/v1/settings`);
        if (!res.ok) return;
        const view = (await res.json()) as SettingsView;
        setServer(view.server);
        setPort(view.port);
      } catch {
        // Defaults stay.
      }
    })();
  }, []);

  // Logs: poll while the strip is expanded.
  useEffect(() => {
    if (!logsOpen) return;
    let cancelled = false;
    const fetchLogs = async () => {
      try {
        const base = await apiBase();
        const res = await fetch(`${base}/api/v1/logs`);
        if (!res.ok) return;
        const data = (await res.json()) as { lines: string[] };
        if (!cancelled) setLogs(data.lines);
      } catch {
        // Strip just keeps its last lines.
      }
    };
    void fetchLogs();
    const timer = setInterval(() => void fetchLogs(), 3000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [logsOpen]);

  async function saveServer() {
    setSaving(true);
    setSaveMsg(null);
    try {
      const base = await apiBase();
      const res = await fetch(`${base}/api/v1/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ server }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const view = (await res.json()) as SettingsView;
      setServer(view.server);
      setSaveMsg("Saved.");
    } catch (e) {
      setSaveMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  async function unloadSelected() {
    if (!selected) return;
    setUnloading(true);
    try {
      const base = await apiBase();
      await fetch(`${base}/api/v1/models/unload`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instance_id: selected.instance.instance_id }),
      });
      setSelectedId(null);
      await fetchAll();
    } catch {
      // Row stays; the next poll reflects reality.
    } finally {
      setUnloading(false);
    }
  }

  const selected = rows.find((r) => r.instance.instance_id === selectedId) ?? null;

  function serverRow(
    label: string,
    hint: string,
    field: keyof ServerSettings,
  ) {
    return (
      <div className="flex items-center justify-between gap-4 py-2">
        <div>
          <p className="text-[13px]">{label}</p>
          <p className="text-[11px] text-muted">{hint}</p>
        </div>
        <Toggle
          on={!!server[field]}
          onClick={() => setServer((s) => ({ ...s, [field]: !s[field] }))}
        />
      </div>
    );
  }

  return (
    <div className="flex h-full">
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="min-h-0 flex-1 overflow-auto">
          <div className="mx-auto w-full max-w-4xl px-8 py-6">
            {/* Server status card */}
            <section className="relative rounded-xl border border-edge bg-panel p-5">
              <div className="flex items-start justify-between">
                <div>
                  <h2 className="text-[15px] font-semibold">Model Server</h2>
                  <p className="mt-1 flex items-center gap-2 text-muted">
                    Status:
                    {running ? (
                      <span className="flex items-center gap-1.5 font-medium text-ok">
                        <span className="h-2 w-2 rounded-full bg-ok" />
                        Running
                      </span>
                    ) : (
                      <span className="font-medium">Stopped</span>
                    )}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setSettingsOpen((v) => !v)}
                    className="flex items-center gap-1.5 rounded-lg border border-edge px-3 py-1.5 text-[13px] transition-colors hover:bg-panel-2"
                  >
                    <Settings2 size={13} />
                    Server Settings
                  </button>
                  <button
                    type="button"
                    title="mcp.json — view only in QuackStudio"
                    className="flex items-center gap-1.5 rounded-lg border border-edge px-3 py-1.5 text-[13px] text-muted transition-colors hover:bg-panel-2"
                  >
                    <FileJson size={13} />
                    mcp.json
                  </button>
                  <Toggle on={running} onClick={() => setRunning((v) => !v)} />
                </div>
              </div>

              {/* Server Settings popover */}
              {settingsOpen && (
                <div className="absolute top-16 right-5 z-20 w-96 rounded-xl border border-edge bg-panel-2 p-4 shadow-2xl">
                  <div className="flex items-center justify-between">
                    <p className="text-[14px] font-semibold">
                      Server Settings
                    </p>
                    <button
                      type="button"
                      title="Close"
                      onClick={() => setSettingsOpen(false)}
                      className="flex h-6 w-6 items-center justify-center rounded-md text-muted hover:text-ink"
                    >
                      <X size={13} />
                    </button>
                  </div>
                  <div className="mt-1 flex items-center justify-between py-2">
                    <div>
                      <p className="text-[13px]">Server Port</p>
                      <p className="text-[11px] text-muted">
                        Chosen by the app at launch (1234, or 1235 when
                        taken) — display only.
                      </p>
                    </div>
                    <code className="rounded bg-app-bg px-2 py-0.5 font-mono text-[12px]">
                      {port}
                    </code>
                  </div>
                  <div className="divide-y divide-edge/60">
                    {serverRow(
                      "Require Authentication",
                      "Stored — quackd's loopback listener does not enforce tokens yet.",
                      "require_auth",
                    )}
                    {serverRow(
                      "Serve on Local Network",
                      "Stored preference.",
                      "serve_lan",
                    )}
                    {serverRow(
                      "Enable CORS",
                      "Stored — the listener currently sends permissive CORS headers.",
                      "cors",
                    )}
                    {serverRow(
                      "Just-in-Time Model Loading",
                      "Enforced: gates the chat proxy's auto-load of unloaded models.",
                      "jit",
                    )}
                    {serverRow(
                      "Only Keep Last JIT Loaded Model",
                      "Stored — no JIT eviction is implemented yet.",
                      "jit_only_last",
                    )}
                  </div>
                  <div className="flex items-center justify-between py-2">
                    <div>
                      <p className="text-[13px]">Max idle TTL (minutes)</p>
                      <p className="text-[11px] text-muted">
                        Stored — idle eviction is not implemented yet.
                      </p>
                    </div>
                    <input
                      type="number"
                      min={1}
                      max={1440}
                      value={server.jit_ttl_minutes}
                      onChange={(e) =>
                        setServer((s) => ({
                          ...s,
                          jit_ttl_minutes: Math.max(
                            1,
                            Number(e.target.value) || 1,
                          ),
                        }))
                      }
                      className="w-20 rounded-lg border border-edge bg-app-bg px-2 py-1 text-right text-[13px] outline-none"
                    />
                  </div>
                  <div className="mt-2 flex items-center gap-3 border-t border-edge pt-3">
                    <button
                      type="button"
                      disabled={saving}
                      onClick={() => void saveServer()}
                      className="flex items-center gap-1.5 rounded-lg bg-primary px-3.5 py-1.5 font-medium text-white transition-opacity disabled:opacity-40"
                    >
                      {saving && (
                        <Loader2 size={13} className="animate-spin" />
                      )}
                      Save
                    </button>
                    {saveMsg && (
                      <span className="text-[12px] text-muted">{saveMsg}</span>
                    )}
                  </div>
                </div>
              )}

              <div className="mt-4 flex items-center gap-2 border-t border-edge pt-4 text-muted">
                <span>Reachable at:</span>
                <code className="rounded bg-app-bg px-2 py-0.5 font-mono text-[12px] text-ink">
                  {reachable}
                </code>
                <button
                  type="button"
                  title="Copy"
                  onClick={() => {
                    void navigator.clipboard
                      ?.writeText(reachable)
                      .catch(() => {});
                    setCopied(true);
                    setTimeout(() => setCopied(false), 1200);
                  }}
                  className="flex h-6 w-6 items-center justify-center rounded-md text-muted transition-colors hover:bg-panel-2 hover:text-ink"
                >
                  <Copy size={12} />
                </button>
                {copied && <span className="text-[11px]">Copied</span>}
              </div>
            </section>

            {/* Loaded models */}
            <section className="mt-6">
              <div className="flex items-center justify-between">
                <h3 className="text-[14px] font-semibold">Loaded Models</h3>
                <button
                  type="button"
                  onClick={() =>
                    window.dispatchEvent(new Event("quack:open-load"))
                  }
                  className="rounded-lg bg-primary px-3.5 py-1.5 font-medium text-white transition-opacity hover:opacity-90"
                >
                  Load Model
                </button>
              </div>
              {rows.length === 0 ? (
                <div className="mt-3 rounded-xl border border-dashed border-edge px-5 py-8 text-center text-muted">
                  No models loaded.
                  <br />
                  Use QuackStudio from other apps or from your code.
                </div>
              ) : (
                <div className="mt-3 flex flex-col gap-2">
                  {rows.map((row) => (
                    <button
                      key={row.instance.instance_id}
                      type="button"
                      onClick={() => setSelectedId(row.instance.instance_id)}
                      className={`rounded-xl border px-4 py-3 text-left transition-colors ${
                        row.instance.instance_id === selectedId
                          ? "border-accent bg-panel"
                          : "border-edge bg-panel hover:bg-panel-2"
                      }`}
                    >
                      <p className="font-medium break-all">
                        {row.entry?.display_name ?? row.instance.model_key}
                      </p>
                      <p className="mt-0.5 text-[12px] text-muted">
                        {row.instance.backend} · port {row.instance.port} ·
                        pid {row.instance.pid}
                      </p>
                    </button>
                  ))}
                </div>
              )}
            </section>

            {/* Supported endpoints */}
            <section className="mt-6">
              <h3 className="text-[14px] font-semibold">
                Supported Endpoints
              </h3>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                {ENDPOINTS.map((ep) => (
                  <code
                    key={ep}
                    className="rounded-full border border-edge bg-panel px-2.5 py-1 font-mono text-[11px] text-muted"
                  >
                    {ep}
                  </code>
                ))}
                <span className="rounded-full border border-primary/50 bg-primary/15 px-2.5 py-1 text-[11px] font-medium text-primary">
                  NEW · REST API v1
                </span>
              </div>
            </section>
          </div>
        </div>

        {/* Server Logs strip */}
        <div className="shrink-0 border-t border-edge">
          <button
            type="button"
            onClick={() => setLogsOpen((v) => !v)}
            className="flex w-full items-center justify-between px-5 py-2.5 text-left"
          >
            <span className="text-[13px] font-medium">Server Logs</span>
            {logsOpen ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
          </button>
          {logsOpen && (
            <div className="h-44 overflow-auto border-t border-edge bg-app-bg px-5 py-2">
              {logs.length === 0 ? (
                <p className="py-2 font-mono text-[12px] text-muted">
                  No log lines yet.
                </p>
              ) : (
                logs.map((line, i) => (
                  <p
                    key={i}
                    className="font-mono text-[12px] whitespace-pre-wrap text-muted"
                  >
                    {line}
                  </p>
                ))
              )}
            </div>
          )}
        </div>
      </div>

      {/* Loaded-model detail panel */}
      {selected && (
        <aside className="flex w-80 shrink-0 flex-col overflow-auto border-l border-edge bg-panel px-5 py-4">
          <div className="flex items-start gap-2">
            <p className="min-w-0 flex-1 text-[15px] font-medium break-words">
              {selected.entry?.display_name ?? selected.instance.model_key}
            </p>
            <button
              type="button"
              title="Close"
              onClick={() => setSelectedId(null)}
              className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-muted transition-colors hover:bg-panel-2 hover:text-ink"
            >
              <X size={15} />
            </button>
          </div>
          <dl className="mt-4 flex flex-col gap-3 text-[13px]">
            {[
              ["Model key", selected.instance.model_key],
              ["Instance", selected.instance.instance_id],
              ["Backend", selected.instance.backend],
              ["Port", String(selected.instance.port)],
              ["PID", String(selected.instance.pid)],
              [
                "Context",
                selected.contextLength
                  ? `${selected.contextLength} tokens`
                  : "engine default",
              ],
              [
                "Format",
                selected.entry
                  ? selected.entry.format.toUpperCase()
                  : "—",
              ],
            ].map(([label, value]) => (
              <div key={label} className="flex items-baseline gap-3">
                <dt className="w-20 shrink-0 text-muted">{label}</dt>
                <dd className="min-w-0 break-words">{value}</dd>
              </div>
            ))}
          </dl>
          {selected.entry?.format === "maya" ? (
            <p className="mt-5 rounded-xl border border-edge bg-panel-2 px-3.5 py-3 text-[12px] text-muted">
              Served by Maya — Maya manages this model's lifecycle, so
              there is no local unload.
            </p>
          ) : (
            <button
              type="button"
              disabled={unloading}
              onClick={() => void unloadSelected()}
              className="mt-5 flex items-center justify-center gap-2 rounded-lg bg-panel-2 px-3 py-2 text-[13px] transition-colors hover:bg-edge disabled:cursor-not-allowed disabled:opacity-40"
            >
              {unloading && <Loader2 size={13} className="animate-spin" />}
              Unload
            </button>
          )}
        </aside>
      )}
    </div>
  );
}
