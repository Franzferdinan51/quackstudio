import { useCallback, useEffect, useState } from "react";
import { Loader2, Plus, Trash2 } from "lucide-react";
import type {
  GeneralSettings,
  MayaProvider,
  ModelDefaults,
  ServerSettings,
  SettingsView,
  StatusView,
} from "../types";
import { formatGB } from "../types";
import { apiBase } from "../api";
import Toggle from "../components/Toggle";
import type { UiPrefs } from "../store";
import {
  INTEGRATIONS,
  loadPrefs,
  loadToolConfirm,
  savePrefs,
  saveToolConfirm,
} from "../store";

const SOURCE_NOTES: Record<SettingsView["maya_providers_source"], string> = {
  env: "Set by the QUACK_MAYA_PROVIDERS environment variable — it wins over the settings file until unset.",
  settings: "Loaded from ~/.quackstudio/settings.json.",
  default: "Built-in defaults — Duckets' six Maya endpoints on the Mac.",
};

type Section =
  | "general"
  | "appearance"
  | "developer"
  | "chat"
  | "defaults"
  | "integrations"
  | "link"
  | "runtime"
  | "hardware";

const SECTIONS: { id: Section; label: string; group?: string }[] = [
  { id: "general", label: "General" },
  { id: "appearance", label: "Appearance" },
  { id: "developer", label: "Developer" },
  { id: "chat", label: "Chat" },
  { id: "defaults", label: "Model Defaults" },
  { id: "integrations", label: "Integrations" },
  { id: "link", label: "LM Link" },
  { id: "runtime", label: "Runtime", group: "System" },
  { id: "hardware", label: "Hardware", group: "System" },
];

type SaveMessage = { kind: "ok" | "error"; text: string };

function errorFrom(data: unknown, fallback: string): string {
  if (
    data !== null &&
    typeof data === "object" &&
    "error" in data &&
    typeof (data as { error: unknown }).error === "string"
  ) {
    return (data as { error: string }).error;
  }
  return fallback;
}

function InfoRow({
  label,
  hint,
  value,
}: {
  label: string;
  hint: string;
  value: string;
}) {
  return (
    <div className="flex items-center justify-between gap-6 px-4 py-3">
      <div>
        <p>{label}</p>
        <p className="mt-0.5 text-[12px] text-muted">{hint}</p>
      </div>
      <code className="shrink-0 rounded bg-app-bg px-2 py-0.5 font-mono text-[12px]">
        {value}
      </code>
    </div>
  );
}

function ToggleRow({
  label,
  hint,
  on,
  onClick,
}: {
  label: string;
  hint?: string;
  on: boolean;
  onClick: () => void;
}) {
  return (
    <div className="flex items-center justify-between gap-6 px-4 py-3">
      <div>
        <p>{label}</p>
        {hint && <p className="mt-0.5 text-[12px] text-muted">{hint}</p>}
      </div>
      <Toggle on={on} onClick={onClick} />
    </div>
  );
}

function Card({ children }: { children: React.ReactNode }) {
  return (
    <div className="divide-y divide-edge rounded-xl border border-edge bg-panel">
      {children}
    </div>
  );
}

/** Settings: nine sections like LM Studio's. Daemon-backed
 *  sections (General HF proxy, Developer server knobs, Model
 *  Defaults, LM Link's Maya providers) persist through the
 *  settings API; Appearance/Chat/Integrations preferences are
 *  local UI state; Runtime/Hardware are read-only status views. */
export default function SettingsPage({ status }: { status: StatusView | null }) {
  const [section, setSection] = useState<Section>("general");
  const [view, setView] = useState<SettingsView | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [draft, setDraft] = useState<MayaProvider[]>([]);
  const [server, setServer] = useState<ServerSettings | null>(null);
  const [general, setGeneral] = useState<GeneralSettings | null>(null);
  const [defaults, setDefaults] = useState<ModelDefaults | null>(null);
  const [prefs, setPrefsState] = useState<UiPrefs>(() => loadPrefs());
  const [toolConfirm, setToolConfirm] = useState<Record<string, boolean>>(
    () => loadToolConfirm(),
  );
  const [allowLoading, setAllowLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<SaveMessage | null>(null);

  const fetchSettings = useCallback(async () => {
    try {
      const base = await apiBase();
      const res = await fetch(`${base}/api/v1/settings`);
      const data: unknown = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error(errorFrom(data, `settings failed (HTTP ${res.status})`));
      }
      const settings = data as SettingsView;
      setView(settings);
      setDraft(settings.maya_providers.map((p) => ({ ...p })));
      setServer(settings.server);
      setGeneral(settings.general);
      setDefaults(settings.defaults);
      setLoadError(null);
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "cannot reach quackd");
    }
  }, []);

  useEffect(() => {
    void fetchSettings();
  }, [fetchSettings]);

  async function postSettings(body: Record<string, unknown>, okText: string) {
    if (saving) return;
    setSaving(true);
    setSaveMsg(null);
    try {
      const base = await apiBase();
      const res = await fetch(`${base}/api/v1/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data: unknown = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error(errorFrom(data, `save failed (HTTP ${res.status})`));
      }
      const settings = data as SettingsView;
      setView(settings);
      setDraft(settings.maya_providers.map((p) => ({ ...p })));
      setServer(settings.server);
      setGeneral(settings.general);
      setDefaults(settings.defaults);
      setSaveMsg({ kind: "ok", text: settings.note ?? okText });
    } catch (e) {
      setSaveMsg({
        kind: "error",
        text: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setSaving(false);
    }
  }

  function updatePrefs(patch: Partial<UiPrefs>) {
    const next = { ...prefs, ...patch };
    setPrefsState(next);
    savePrefs(next);
    // Theme applies live (App also reads it at startup).
    if (patch.theme) {
      const light =
        patch.theme === "light" ||
        (patch.theme === "auto" &&
          window.matchMedia?.("(prefers-color-scheme: light)").matches);
      document.documentElement.classList.toggle("light", !!light);
    }
  }

  function updateRow(index: number, field: keyof MayaProvider, value: string) {
    setDraft((prev) =>
      prev.map((p, i) => (i === index ? { ...p, [field]: value } : p)),
    );
    setSaveMsg(null);
  }

  const saveButton = (onClick: () => void) => (
    <div className="mt-3 flex items-center gap-3">
      <button
        type="button"
        onClick={onClick}
        disabled={saving}
        className="flex items-center gap-1.5 rounded-lg bg-primary px-3.5 py-1.5 font-medium text-white transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
      >
        {saving && <Loader2 size={13} className="animate-spin" />}
        Save
      </button>
      {saveMsg && (
        <span
          className={`text-[12px] ${saveMsg.kind === "ok" ? "text-ok" : "text-red-400"}`}
        >
          {saveMsg.text}
        </span>
      )}
    </div>
  );

  return (
    <div className="flex h-full">
      {/* Section nav */}
      <nav className="flex w-52 shrink-0 flex-col border-r border-edge px-2.5 py-4">
        <div className="flex flex-col gap-0.5">
          {SECTIONS.map((s) => (
            <div key={s.id}>
              {s.group && (
                <p className="mt-3 px-2.5 pb-1 text-[11px] font-semibold tracking-wide text-muted uppercase">
                  {s.group}
                </p>
              )}
              <button
                type="button"
                onClick={() => setSection(s.id)}
                className={`w-full rounded-lg px-2.5 py-1.5 text-left text-[13px] transition-colors ${
                  section === s.id
                    ? "bg-panel-2 text-ink"
                    : "text-muted hover:bg-panel hover:text-ink"
                }`}
              >
                {s.label}
              </button>
            </div>
          ))}
        </div>
        <div className="mt-auto flex items-center gap-2 px-2.5">
          <span className="flex h-7 w-7 items-center justify-center rounded-full bg-accent/30 text-[12px] font-semibold text-accent">
            d
          </span>
          <span className="text-[13px]">duckets</span>
        </div>
      </nav>

      {/* Section content */}
      <div className="min-w-0 flex-1 overflow-auto px-8 py-6">
        <div className="mx-auto flex max-w-2xl flex-col gap-6">
          {loadError && section !== "appearance" && section !== "chat" && (
            <p className="rounded-xl border border-edge bg-panel px-4 py-3 text-red-400">
              Cannot load settings: {loadError}. Is quackd running?
            </p>
          )}

          {section === "general" && view && general && (
            <>
              <section>
                <h2 className="text-[15px] font-semibold">App</h2>
                <div className="mt-3">
                  <Card>
                    <InfoRow
                      label="QuackStudio version"
                      hint="The daemon (quackd) reports its own version."
                      value={view.version}
                    />
                    <InfoRow
                      label="Daemon port"
                      hint="quackd listens on loopback; the desktop app picks 1235 when 1234 is taken."
                      value={String(view.port)}
                    />
                    <InfoRow
                      label="Models directory"
                      hint="Shared with LM Studio — existing downloads appear automatically, and new downloads land here."
                      value={view.models_dir}
                    />
                    <InfoRow
                      label="Backends directory"
                      hint="Where backend packs (llama.cpp, MLX, …) are discovered."
                      value={view.backends_dir}
                    />
                    <ToggleRow
                      label="Use Hugging Face Proxy"
                      hint="Stored in settings — quackd currently talks to huggingface.co directly."
                      on={general.hf_proxy}
                      onClick={() =>
                        setGeneral({ hf_proxy: !general.hf_proxy })
                      }
                    />
                  </Card>
                </div>
                {saveButton(() =>
                  void postSettings({ general }, "Saved."),
                )}
              </section>
              <section>
                <h2 className="text-[15px] font-semibold">About</h2>
                <div className="mt-3 rounded-xl border border-edge bg-panel px-4 py-3">
                  <p className="font-medium">Personal project.</p>
                  <p className="mt-1 text-muted">
                    QuackStudio is a personal app, built for one person's
                    own setup and published as-is. It may not remain
                    maintained — development happens in bursts, and issues
                    or pull requests may go unanswered for long stretches,
                    or forever. Use it at your own risk; no support is
                    promised.
                  </p>
                </div>
              </section>
            </>
          )}

          {section === "appearance" && (
            <section>
              <h2 className="text-[15px] font-semibold">Appearance</h2>
              <div className="mt-3">
                <Card>
                  <div className="flex items-center justify-between gap-6 px-4 py-3">
                    <div>
                      <p>Color Theme</p>
                      <p className="mt-0.5 text-[12px] text-muted">
                        Light applies a basic light palette; Auto follows
                        the system.
                      </p>
                    </div>
                    <select
                      value={prefs.theme}
                      onChange={(e) =>
                        updatePrefs({
                          theme: e.target.value as UiPrefs["theme"],
                        })
                      }
                      className="rounded-lg border border-edge bg-app-bg px-2.5 py-1.5 outline-none"
                    >
                      <option value="auto">Auto</option>
                      <option value="dark">Dark</option>
                      <option value="light">Light</option>
                    </select>
                  </div>
                  <div className="flex items-center justify-between gap-6 px-4 py-3">
                    <div>
                      <p>Navigation Bar Position</p>
                      <p className="mt-0.5 text-[12px] text-muted">
                        Stored — only Left is implemented.
                      </p>
                    </div>
                    <select
                      value={prefs.navPosition}
                      onChange={(e) =>
                        updatePrefs({
                          navPosition: e.target.value as UiPrefs["navPosition"],
                        })
                      }
                      className="rounded-lg border border-edge bg-app-bg px-2.5 py-1.5 outline-none"
                    >
                      <option value="left">Left</option>
                    </select>
                  </div>
                </Card>
              </div>
            </section>
          )}

          {section === "developer" && server && (
            <section>
              <h2 className="text-[15px] font-semibold">Developer</h2>
              <div className="mt-3">
                <Card>
                  <ToggleRow
                    label="Just-in-Time Model Loading"
                    hint="Enforced by quackd: gates the chat proxy's auto-load. Shared with Model Server → Server Settings."
                    on={server.jit}
                    onClick={() => setServer({ ...server, jit: !server.jit })}
                  />
                  <div className="flex items-center justify-between gap-6 px-4 py-3">
                    <div>
                      <p>Max idle TTL (minutes)</p>
                      <p className="mt-0.5 text-[12px] text-muted">
                        Stored — idle eviction is not implemented yet.
                      </p>
                    </div>
                    <input
                      type="number"
                      min={1}
                      max={1440}
                      value={server.jit_ttl_minutes}
                      onChange={(e) =>
                        setServer({
                          ...server,
                          jit_ttl_minutes: Math.max(
                            1,
                            Number(e.target.value) || 1,
                          ),
                        })
                      }
                      className="w-20 rounded-lg border border-edge bg-app-bg px-2 py-1 text-right outline-none"
                    />
                  </div>
                  <div className="flex items-center justify-between gap-6 px-4 py-3">
                    <div>
                      <p>Extension Packs Download Channel</p>
                      <p className="mt-0.5 text-[12px] text-muted">
                        Stored locally — pack updates are manual.
                      </p>
                    </div>
                    <select
                      value="beta"
                      disabled
                      className="rounded-lg border border-edge bg-app-bg px-2.5 py-1.5 outline-none disabled:opacity-60"
                    >
                      <option value="beta">Beta</option>
                      <option value="stable">Stable</option>
                    </select>
                  </div>
                </Card>
              </div>
              {saveButton(() => void postSettings({ server }, "Saved."))}
            </section>
          )}

          {section === "chat" && (
            <section>
              <h2 className="text-[15px] font-semibold">Chat</h2>
              <div className="mt-3">
                <Card>
                  <div className="flex items-center justify-between gap-6 px-4 py-3">
                    <div>
                      <p>Sidebar Sort</p>
                      <p className="mt-0.5 text-[12px] text-muted">
                        Orders the chat list in the expanded sidebar.
                      </p>
                    </div>
                    <select
                      value={prefs.chatSort}
                      onChange={(e) =>
                        updatePrefs({
                          chatSort: e.target.value as UiPrefs["chatSort"],
                        })
                      }
                      className="rounded-lg border border-edge bg-app-bg px-2.5 py-1.5 outline-none"
                    >
                      <option value="created-desc">Date created (newest)</option>
                      <option value="created-asc">Date created (oldest)</option>
                      <option value="name">Name</option>
                    </select>
                  </div>
                  <ToggleRow
                    label="AI-generated Chat Names"
                    hint="Stored — chats are currently named from their first message."
                    on={prefs.aiChatNames}
                    onClick={() =>
                      updatePrefs({ aiChatNames: !prefs.aiChatNames })
                    }
                  />
                </Card>
              </div>
            </section>
          )}

          {section === "defaults" && defaults && (
            <section>
              <h2 className="text-[15px] font-semibold">Model Defaults</h2>
              <div className="mt-3">
                <Card>
                  <div className="flex items-center justify-between gap-6 px-4 py-3">
                    <div>
                      <p>Default Context Length</p>
                      <p className="mt-0.5 text-[12px] text-muted">
                        Wired: seeds the Library Load tab's initial
                        context value. Blank = 8192.
                      </p>
                    </div>
                    <input
                      type="number"
                      min={1024}
                      max={262144}
                      step={1024}
                      placeholder="8192"
                      value={defaults.context_length ?? ""}
                      onChange={(e) =>
                        setDefaults({
                          ...defaults,
                          context_length:
                            e.target.value === ""
                              ? null
                              : Number(e.target.value),
                        })
                      }
                      className="w-28 rounded-lg border border-edge bg-app-bg px-2 py-1 text-right outline-none"
                    />
                  </div>
                  <div className="flex items-center justify-between gap-6 px-4 py-3">
                    <div>
                      <p>Image Input Max (px)</p>
                      <p className="mt-0.5 text-[12px] text-muted">
                        Stored — image inputs are not resized yet.
                      </p>
                    </div>
                    <input
                      type="number"
                      min={256}
                      max={8192}
                      step={256}
                      placeholder="2048"
                      value={defaults.image_max_px ?? ""}
                      onChange={(e) =>
                        setDefaults({
                          ...defaults,
                          image_max_px:
                            e.target.value === ""
                              ? null
                              : Number(e.target.value),
                        })
                      }
                      className="w-28 rounded-lg border border-edge bg-app-bg px-2 py-1 text-right outline-none"
                    />
                  </div>
                </Card>
              </div>
              {saveButton(() => void postSettings({ defaults }, "Saved."))}
            </section>
          )}

          {section === "integrations" && (
            <section>
              <h2 className="text-[15px] font-semibold">Integrations</h2>
              <p className="mt-1 text-[12px] text-muted">
                Tool Call Confirmation — allow the following tools to
                run without confirmation. Stored locally; these plugins
                are not connected to quackd.
              </p>
              <div className="mt-3 flex flex-col gap-3">
                {INTEGRATIONS.map((integration) => (
                  <div
                    key={integration.id}
                    className="rounded-xl border border-edge bg-panel px-4 py-3"
                  >
                    <p className="font-medium">{integration.id}</p>
                    {integration.tools.map((tool) => {
                      const key = `${integration.id}:${tool}`;
                      return (
                        <div
                          key={tool}
                          className="mt-2 flex items-center justify-between"
                        >
                          <span className="font-mono text-[12px] text-muted">
                            {tool}
                          </span>
                          <Toggle
                            on={!!toolConfirm[key]}
                            onClick={() => {
                              const next = {
                                ...toolConfirm,
                                [key]: !toolConfirm[key],
                              };
                              setToolConfirm(next);
                              saveToolConfirm(next);
                            }}
                          />
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </section>
          )}

          {section === "link" && view && (
            <>
              <section>
                <h2 className="text-[15px] font-semibold">LM Link</h2>
                <div className="mt-3">
                  <Card>
                    <InfoRow
                      label="Device name"
                      hint="This machine's hostname, as reported by quackd."
                      value={status?.host.name ?? "—"}
                    />
                    <InfoRow
                      label="Device identifier"
                      hint="The daemon's listen address — QuackStudio has no LM Link device id."
                      value={
                        status ? `127.0.0.1:${status.port}` : "—"
                      }
                    />
                    <ToggleRow
                      label="Allow loading models on this machine"
                      hint="Stored locally — local loads are always allowed in QuackStudio."
                      on={allowLoading}
                      onClick={() => setAllowLoading((v) => !v)}
                    />
                  </Card>
                </div>
              </section>

              <section>
                <h2 className="text-[15px] font-semibold">
                  Maya providers (linked services)
                </h2>
                <p className="mt-1 text-[12px] text-muted">
                  Remote Maya servers whose models appear in the library
                  as remote entries — QuackStudio's link layer.{" "}
                  {SOURCE_NOTES[view.maya_providers_source]}
                </p>
                <div className="mt-3 rounded-xl border border-edge bg-panel px-4 py-3">
                  <div className="flex flex-col gap-2">
                    {draft.map((provider, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <input
                          value={provider.name}
                          onChange={(e) => updateRow(i, "name", e.target.value)}
                          placeholder="name"
                          className="w-36 shrink-0 rounded-lg border border-edge bg-app-bg px-2.5 py-1.5 outline-none placeholder:text-muted"
                        />
                        <input
                          value={provider.base_url}
                          onChange={(e) =>
                            updateRow(i, "base_url", e.target.value)
                          }
                          placeholder="http://100.68.208.113:8085"
                          className="min-w-0 flex-1 rounded-lg border border-edge bg-app-bg px-2.5 py-1.5 font-mono text-[12px] outline-none placeholder:text-muted"
                        />
                        <button
                          type="button"
                          title="Remove provider"
                          onClick={() => {
                            setDraft((prev) =>
                              prev.filter((_, j) => j !== i),
                            );
                            setSaveMsg(null);
                          }}
                          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-muted transition-colors hover:bg-panel-2 hover:text-red-400"
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    ))}
                    {draft.length === 0 && (
                      <p className="text-muted">
                        No providers — Maya models will not appear in the
                        library.
                      </p>
                    )}
                  </div>
                  <div className="mt-3 flex items-center gap-3 border-t border-edge pt-3">
                    <button
                      type="button"
                      onClick={() => {
                        setDraft((prev) => [
                          ...prev,
                          { name: "", base_url: "" },
                        ]);
                        setSaveMsg(null);
                      }}
                      className="flex items-center gap-1.5 rounded-lg border border-edge px-3 py-1.5 transition-colors hover:bg-panel-2"
                    >
                      <Plus size={13} />
                      Add provider
                    </button>
                    {saveButton(() =>
                      void postSettings(
                        { maya_providers: draft },
                        "Saved — the new provider list is live.",
                      ),
                    )}
                  </div>
                </div>
              </section>
            </>
          )}

          {section === "runtime" && (
            <section>
              <h2 className="text-[15px] font-semibold">
                Runtime Selections
              </h2>
              <p className="mt-1 text-[12px] text-muted">
                Read-only — the packs quackd found on disk.
              </p>
              <div className="mt-3">
                <Card>
                  <InfoRow
                    label="GGUF"
                    hint="llama.cpp pack (Metal) — the llama-server quackd spawns."
                    value={status ? `llama.cpp ${status.packs.llama}` : "—"}
                  />
                  <InfoRow
                    label="MLX"
                    hint="oMLX runtime — quackd drives its Python's mlx_vlm.server."
                    value={status?.packs.mlx ?? "—"}
                  />
                  <InfoRow
                    label="Splash"
                    hint="Served through Maya/Strata today; a native pack is pending."
                    value={
                      status
                        ? `${status.packs.splash} — via Maya/Strata (pending)`
                        : "—"
                    }
                  />
                </Card>
              </div>
            </section>
          )}

          {section === "hardware" && (
            <section>
              <h2 className="text-[15px] font-semibold">Hardware</h2>
              <p className="mt-1 text-[12px] text-muted">
                Read-only — reported by quackd from the host.
              </p>
              <div className="mt-3">
                <Card>
                  <InfoRow
                    label="CPU"
                    hint="Apple Silicon, ARM64."
                    value={status?.host.chip ?? "—"}
                  />
                  <InfoRow
                    label="Memory Capacity"
                    hint="Unified Memory — RAM and VRAM are the same pool."
                    value={
                      status
                        ? `RAM ${formatGB(status.host.total_ram_bytes)} · VRAM ${formatGB(status.host.total_ram_bytes)}`
                        : "—"
                    }
                  />
                  <InfoRow
                    label="GPU"
                    hint="Metal."
                    value={status?.host.gpu ?? "—"}
                  />
                  <InfoRow
                    label="Loaded model weights"
                    hint="Sum of loaded models' file sizes — the sidebar RAM pill."
                    value={
                      status ? formatGB(status.host.loaded_bytes) : "—"
                    }
                  />
                  <InfoRow
                    label="quackd memory"
                    hint="The daemon's own peak RSS (getrusage) — the sidebar's second pill."
                    value={
                      status
                        ? `${(status.host.daemon_rss_bytes / 1_000_000).toFixed(0)} MB`
                        : "—"
                    }
                  />
                </Card>
              </div>
            </section>
          )}

          {!view && !loadError && section !== "appearance" && section !== "chat" && section !== "integrations" && (
            <p className="flex items-center gap-2 text-muted">
              <Loader2 size={14} className="animate-spin" />
              Loading settings…
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
