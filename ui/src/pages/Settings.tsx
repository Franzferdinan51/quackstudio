import { useCallback, useEffect, useState } from "react";
import { Loader2, Plus, Trash2 } from "lucide-react";
import type { MayaProvider, SettingsView } from "../types";
import { apiBase } from "../api";

const SOURCE_NOTES: Record<SettingsView["maya_providers_source"], string> = {
  env: "Set by the QUACK_MAYA_PROVIDERS environment variable — it wins over the settings file until unset.",
  settings: "Loaded from ~/.quackstudio/settings.json.",
  default: "Built-in defaults — Duckets' six Maya endpoints on the Mac.",
};

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

/** One read-only App-section row. */
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

/** Settings: the daemon's effective settings. App facts are
 *  read-only; the Maya provider list is editable and persists to
 *  ~/.quackstudio/settings.json through quackd. */
export default function SettingsPage() {
  const [view, setView] = useState<SettingsView | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [draft, setDraft] = useState<MayaProvider[]>([]);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<SaveMessage | null>(null);

  const fetchSettings = useCallback(async () => {
    try {
      const base = await apiBase();
      const res = await fetch(`${base}/api/v1/settings`);
      const data: unknown = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error(
          errorFrom(data, `settings failed (HTTP ${res.status})`),
        );
      }
      const settings = data as SettingsView;
      setView(settings);
      setDraft(settings.maya_providers.map((p) => ({ ...p })));
      setLoadError(null);
    } catch (e) {
      setLoadError(
        e instanceof Error ? e.message : "cannot reach quackd",
      );
    }
  }, []);

  useEffect(() => {
    void fetchSettings();
  }, [fetchSettings]);

  function updateRow(index: number, field: keyof MayaProvider, value: string) {
    setDraft((prev) =>
      prev.map((p, i) => (i === index ? { ...p, [field]: value } : p)),
    );
    setSaveMsg(null);
  }

  function removeRow(index: number) {
    setDraft((prev) => prev.filter((_, i) => i !== index));
    setSaveMsg(null);
  }

  function addRow() {
    setDraft((prev) => [...prev, { name: "", base_url: "" }]);
    setSaveMsg(null);
  }

  async function save() {
    if (saving) return;
    setSaving(true);
    setSaveMsg(null);
    try {
      const base = await apiBase();
      const res = await fetch(`${base}/api/v1/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ maya_providers: draft }),
      });
      const data: unknown = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error(errorFrom(data, `save failed (HTTP ${res.status})`));
      }
      const settings = data as SettingsView;
      setView(settings);
      setDraft(settings.maya_providers.map((p) => ({ ...p })));
      setSaveMsg({
        kind: "ok",
        text: settings.note ?? "Saved — the new provider list is live.",
      });
    } catch (e) {
      setSaveMsg({
        kind: "error",
        text: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="h-full overflow-auto px-8 py-6">
      <div className="mx-auto flex max-w-2xl flex-col gap-6">
        {loadError && (
          <p className="rounded-xl border border-edge bg-panel px-4 py-3 text-red-400">
            Cannot load settings: {loadError}. Is quackd running?
          </p>
        )}
        {!view && !loadError && (
          <p className="flex items-center gap-2 text-muted">
            <Loader2 size={14} className="animate-spin" />
            Loading settings…
          </p>
        )}

        {view && (
          <>
            {/* App */}
            <section>
              <h2 className="text-[15px] font-semibold">App</h2>
              <div className="mt-3 divide-y divide-edge rounded-xl border border-edge bg-panel">
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
              </div>
            </section>

            {/* Maya Providers */}
            <section>
              <h2 className="text-[15px] font-semibold">Maya Providers</h2>
              <p className="mt-1 text-[12px] text-muted">
                Remote Maya servers whose models appear in the library
                as remote entries. {SOURCE_NOTES[view.maya_providers_source]}
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
                        onClick={() => removeRow(i)}
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
                    onClick={addRow}
                    className="flex items-center gap-1.5 rounded-lg border border-edge px-3 py-1.5 transition-colors hover:bg-panel-2"
                  >
                    <Plus size={13} />
                    Add provider
                  </button>
                  <button
                    type="button"
                    onClick={() => void save()}
                    disabled={saving}
                    className="flex items-center gap-1.5 rounded-lg bg-primary px-3.5 py-1.5 font-medium text-white transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {saving && <Loader2 size={13} className="animate-spin" />}
                    Save
                  </button>
                  {saveMsg && (
                    <span
                      className={`min-w-0 text-[12px] ${
                        saveMsg.kind === "ok" ? "text-ok" : "text-red-400"
                      }`}
                    >
                      {saveMsg.text}
                    </span>
                  )}
                </div>
              </div>
            </section>

            {/* About */}
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
      </div>
    </div>
  );
}
