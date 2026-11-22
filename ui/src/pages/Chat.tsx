import { useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  ChevronDown,
  Loader2,
  MoreHorizontal,
  Paperclip,
  Plus,
  SlidersHorizontal,
  SplitSquareHorizontal,
  Wrench,
  X,
} from "lucide-react";
import type { ModelEntry } from "../types";
import { formatGB } from "../types";
import { apiBase } from "../api";
import type { ChatSession, ChatsState, Preset } from "../store";
import {
  INTEGRATIONS,
  loadInference,
  loadIntegrationEnabled,
  loadPresets,
  loadPrefs,
  newChatSession,
  saveIntegrationEnabled,
  savePresets,
  savePrefs,
} from "../store";
import Toggle from "../components/Toggle";

type LoadPhase = "none" | "loading" | "ready" | "error";
type PanelTab = "integrations" | "configuration";

interface Usage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

/** Chat page: tab strip (one tab per chat), model picker popover,
 *  streaming chat over quackd's OpenAI-compat proxy, and the
 *  right panel — Integrations (toggles + composer chips) and
 *  Configuration (presets, system prompt, temperature / top P —
 *  wired into the chat request). Chat sessions live in App state
 *  and persist to localStorage. */
export default function Chat({
  chatsState,
  onChatsState,
  panelOpen,
  onNewChat,
}: {
  chatsState: ChatsState;
  onChatsState: (
    next: ChatsState | ((prev: ChatsState) => ChatsState),
  ) => void;
  panelOpen: boolean;
  onNewChat: () => void;
}) {
  const chat =
    chatsState.chats.find((c) => c.id === chatsState.activeId) ??
    chatsState.chats[0];
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [pickerFilter, setPickerFilter] = useState("");
  const [phase, setPhase] = useState<LoadPhase>("none");
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [panelTab, setPanelTab] = useState<PanelTab>("integrations");
  const [integrationFilter, setIntegrationFilter] = useState("");
  const [enabled, setEnabled] = useState<Record<string, boolean>>(() =>
    loadIntegrationEnabled(),
  );
  const [presets, setPresets] = useState<Record<string, Preset>>(() =>
    loadPresets(),
  );
  const [presetName, setPresetName] = useState<string>("");
  const [usage, setUsage] = useState<Usage | null>(null);
  const [noteDraft, setNoteDraft] = useState("");
  const [manualParams, setManualParams] = useState(
    () => loadPrefs().manualParams,
  );
  const [instanceCtx, setInstanceCtx] = useState<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const selected = models.find((m) => m.key === chat?.modelKey) ?? null;

  function updateChat(patch: Partial<ChatSession>) {
    if (!chat) return;
    const id = chat.id;
    onChatsState((prev) => ({
      ...prev,
      chats: prev.chats.map((c) => (c.id === id ? { ...c, ...patch } : c)),
    }));
  }

  // The loaded instance's context length (from /api/v1/status),
  // the denominator for the composer's context-% indicator.
  async function refreshInstanceCtx(modelKey: string) {
    try {
      const base = await apiBase();
      const res = await fetch(`${base}/api/v1/status`);
      if (!res.ok) return;
      const data = (await res.json()) as {
        instances?: { model_key: string; context_length: number | null }[];
      };
      const inst = data.instances?.find((i) => i.model_key === modelKey);
      setInstanceCtx(inst?.context_length ?? null);
    } catch {
      // Status is best-effort for this indicator.
    }
  }

  // Load the library; adopt a "Use in New Chat" handoff or an
  // already-loaded instance for the active chat.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const base = await apiBase();
      fetch(`${base}/api/v1/models`)
        .then((res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json() as Promise<ModelEntry[]>;
        })
        .then((data) => {
          if (cancelled) return;
          setModels(data);
          const pending = sessionStorage.getItem("quack.useInChat");
          if (pending) {
            sessionStorage.removeItem("quack.useInChat");
            const wanted = data.find((m) => m.key === pending);
            if (wanted) {
              void selectModel(wanted);
              return;
            }
          }
          const current = data.find((m) => m.key === chat?.modelKey);
          if (current && current.loaded_instances.length > 0) {
            setPhase("ready");
          } else if (!chat?.modelKey) {
            const already = data.find((m) => m.loaded_instances.length > 0);
            if (already) {
              updateChat({ modelKey: already.key });
              setPhase("ready");
            }
          }
        })
        .catch(() => {
          if (!cancelled) {
            setError("cannot reach quackd — is the daemon running?");
          }
        });
    })();
    return () => {
      cancelled = true;
    };
    // Runs once on mount; chat identity changes are handled by the
    // phase effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Switching chats: reflect the newly active chat's model state.
  useEffect(() => {
    setError(null);
    setUsage(null);
    if (!chat?.modelKey) {
      setPhase("none");
      return;
    }
    const entry = models.find((m) => m.key === chat.modelKey);
    if (entry && entry.loaded_instances.length > 0) {
      setPhase("ready");
      void refreshInstanceCtx(chat.modelKey);
    } else if (entry) setPhase("none");
  }, [chat?.id, chat?.modelKey, models]);

  // Keep the newest message in view while streaming.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [chat?.messages]);

  async function selectModel(model: ModelEntry) {
    setPickerOpen(false);
    setError(null);
    // Fresh chat with this model: seed sampling from the model's
    // stored inference defaults (Library → Inference tab).
    const patch: Partial<ChatSession> = { modelKey: model.key };
    if (chat && chat.messages.length === 0) {
      const inf = loadInference(model.key);
      patch.temperature = inf.temperature;
      patch.topP = inf.topP;
    }
    updateChat(patch);
    if (manualParams) {
      sessionStorage.setItem("quack.loadModel", model.key);
      window.dispatchEvent(new Event("quack:open-load"));
      return;
    }
    if (model.loaded_instances.length > 0) {
      setPhase("ready");
      void refreshInstanceCtx(model.key);
      return;
    }
    setPhase("loading");
    try {
      const base = await apiBase();
      const res = await fetch(`${base}/api/v1/models/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: model.key }),
      });
      const data: unknown = await res.json().catch(() => null);
      if (!res.ok) {
        const message =
          data !== null &&
          typeof data === "object" &&
          "error" in data &&
          typeof (data as { error: unknown }).error === "string"
            ? (data as { error: string }).error
            : `load failed (HTTP ${res.status})`;
        throw new Error(message);
      }
      setPhase("ready");
      void refreshInstanceCtx(model.key);
    } catch (e) {
      setPhase("error");
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function closeChat(id: string) {
    const remaining = chatsState.chats.filter((c) => c.id !== id);
    if (remaining.length === 0) {
      const fresh = newChatSession();
      onChatsState({ ...chatsState, chats: [fresh], activeId: fresh.id });
      return;
    }
    onChatsState({
      ...chatsState,
      chats: remaining,
      activeId:
        chatsState.activeId === id ? remaining[0].id : chatsState.activeId,
    });
  }

  async function send() {
    const text = input.trim();
    if (!text || !chat || !selected || phase !== "ready" || sending) return;
    const history = [...chat.messages, { role: "user" as const, content: text }];
    // Name the chat after its first user message (LM Studio's
    // behaviour when AI chat names are off).
    const namePatch =
      chat.messages.length === 0 && chat.name === "Unnamed Chat"
        ? { name: text.slice(0, 42) }
        : {};
    updateChat({
      messages: [...history, { role: "assistant", content: "" }],
      ...namePatch,
    });
    setInput("");
    setSending(true);
    setError(null);
    const chatId = chat.id;
    const appendDelta = (delta: string) => {
      onChatsState((prev) => ({
        ...prev,
        chats: prev.chats.map((c) => {
          if (c.id !== chatId) return c;
          const msgs = [...c.messages];
          const last = msgs[msgs.length - 1];
          msgs[msgs.length - 1] = { ...last, content: last.content + delta };
          return { ...c, messages: msgs };
        }),
      }));
    };
    try {
      const base = await apiBase();
      const payload: { role: string; content: string }[] = [];
      if (chat.systemPrompt.trim()) {
        payload.push({ role: "system", content: chat.systemPrompt });
      }
      payload.push(...history);
      const res = await fetch(`${base}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: selected.key,
          messages: payload,
          stream: true,
          temperature: chat.temperature,
          top_p: chat.topP,
        }),
      });
      if (!res.ok || !res.body) {
        const detail = await res.text().catch(() => "");
        throw new Error(`chat failed (HTTP ${res.status}) ${detail}`.trim());
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data:")) continue;
          const payloadText = trimmed.slice("data:".length).trim();
          if (payloadText === "[DONE]") continue;
          try {
            const json = JSON.parse(payloadText) as {
              choices?: { delta?: { content?: string } }[];
              usage?: Usage;
            };
            if (json.usage) setUsage(json.usage);
            const delta = json.choices?.[0]?.delta?.content ?? "";
            if (delta) appendDelta(delta);
          } catch {
            // Split JSON fragment — skip it (the proxy forwards
            // SSE verbatim, so this should not happen).
          }
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSending(false);
    }
  }

  function toggleIntegration(id: string) {
    const next = { ...enabled, [id]: !enabled[id] };
    setEnabled(next);
    saveIntegrationEnabled(next);
  }

  function applyPreset(name: string) {
    setPresetName(name);
    const preset = presets[name];
    if (!preset || !chat) return;
    updateChat({
      systemPrompt: preset.systemPrompt,
      temperature: preset.temperature,
      topP: preset.topP,
      reasoningBudget: preset.reasoningBudget,
    });
  }

  function savePresetAs() {
    if (!chat) return;
    const name = window.prompt("Preset name:");
    if (!name) return;
    const next = {
      ...presets,
      [name]: {
        systemPrompt: chat.systemPrompt,
        temperature: chat.temperature,
        topP: chat.topP,
        reasoningBudget: chat.reasoningBudget,
      },
    };
    setPresets(next);
    savePresets(next);
    setPresetName(name);
  }

  if (!chat) return null;

  const pillLabel =
    phase === "loading"
      ? `Loading ${selected?.display_name ?? "model"}…`
      : selected
        ? selected.display_name
        : "Select a model to load";

  const pickerModels = models.filter((m) => {
    const needle = pickerFilter.trim().toLowerCase();
    return (
      needle === "" ||
      `${m.publisher}/${m.display_name}`.toLowerCase().includes(needle)
    );
  });

  // Context indicator: last response's prompt tokens against the
  // loaded instance's context length (from /api/v1/status; the
  // 262144 production max when the instance didn't declare one),
  // else 0%.
  const ctxWindow = instanceCtx ?? 262144;
  const ctxPct =
    usage?.prompt_tokens != null
      ? Math.min(100, Math.round((usage.prompt_tokens / ctxWindow) * 1000) / 10)
      : 0;

  const enabledIntegrations = INTEGRATIONS.filter((i) => enabled[i.id]);

  return (
    <div className="flex h-full">
      <div className="relative flex min-w-0 flex-1 flex-col">
        {/* Tab strip */}
        <div className="flex items-center gap-1 border-b border-edge px-3 pt-2">
          <div className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto">
            {chatsState.chats.map((c) => (
              <span
                key={c.id}
                className={`flex shrink-0 items-center gap-1.5 rounded-t-lg px-3 py-1.5 text-[12px] ${
                  c.id === chat.id
                    ? "bg-panel text-ink"
                    : "text-muted hover:bg-panel/60 hover:text-ink"
                }`}
              >
                <button
                  type="button"
                  onClick={() =>
                    onChatsState({ ...chatsState, activeId: c.id })
                  }
                  className="max-w-40 truncate"
                >
                  {c.name}
                </button>
                <button
                  type="button"
                  title="Close chat"
                  onClick={() => closeChat(c.id)}
                  className="text-muted hover:text-ink"
                >
                  <X size={11} />
                </button>
              </span>
            ))}
            <button
              type="button"
              title="New chat"
              onClick={onNewChat}
              className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-muted hover:bg-panel hover:text-ink"
            >
              <Plus size={13} />
            </button>
          </div>
          <button
            type="button"
            title="Split view (decorative)"
            className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-muted hover:text-ink"
          >
            <SplitSquareHorizontal size={13} />
          </button>
          <button
            type="button"
            title="More options (decorative)"
            className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-muted hover:text-ink"
          >
            <MoreHorizontal size={13} />
          </button>
        </div>

        {/* Model picker pill + popover */}
        <div className="relative flex justify-center pt-3">
          <button
            type="button"
            onClick={() => setPickerOpen((open) => !open)}
            className="flex items-center gap-2 rounded-full border border-edge bg-panel px-4 py-1.5 transition-colors hover:bg-panel-2"
          >
            {phase === "loading" && (
              <Loader2 size={14} className="animate-spin text-muted" />
            )}
            <span>{pillLabel}</span>
            <span className="text-muted">(⌘L)</span>
            <ChevronDown size={14} className="text-muted" />
          </button>
          {pickerOpen && (
            <>
              <div
                className="fixed inset-0 z-10"
                onClick={() => setPickerOpen(false)}
              />
              <div className="absolute top-full z-20 mt-2 w-[26rem] overflow-hidden rounded-xl border border-edge bg-panel shadow-xl">
                <div className="border-b border-edge px-3 py-2">
                  <input
                    autoFocus
                    value={pickerFilter}
                    onChange={(e) => setPickerFilter(e.target.value)}
                    placeholder="Filter models…"
                    className="w-full bg-transparent outline-none placeholder:text-muted"
                  />
                </div>
                <div className="max-h-72 overflow-auto">
                  {pickerModels.length === 0 && (
                    <p className="px-4 py-3 text-muted">
                      No models match.
                    </p>
                  )}
                  {pickerModels.map((m) => {
                    const loadable =
                      m.format === "gguf" ||
                      m.format === "mlx" ||
                      m.format === "maya";
                    return (
                      <button
                        key={m.key}
                        type="button"
                        disabled={!loadable}
                        onClick={() => void selectModel(m)}
                        className="flex w-full items-center gap-2 px-4 py-2.5 text-left transition-colors enabled:hover:bg-panel-2 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        <span className="min-w-0 flex-1 truncate">
                          {m.display_name}
                          <span className="block truncate text-[11px] text-muted">
                            {m.format === "maya"
                              ? (m.loaded_instances[0]?.backend ??
                                "Maya · remote")
                              : `${m.publisher} · ${formatGB(m.size_bytes)}`}
                          </span>
                        </span>
                        {m.loaded_instances.length > 0 && (
                          <span className="rounded bg-accent/20 px-1.5 py-0.5 text-[10px] text-accent">
                            loaded
                          </span>
                        )}
                        {m.format === "maya" ? (
                          <span className="rounded bg-primary/20 px-1.5 py-0.5 text-[10px] text-primary">
                            Maya · remote
                          </span>
                        ) : (
                          <span className="rounded bg-panel-2 px-1.5 py-0.5 text-[10px] uppercase text-muted">
                            {m.format}
                            {!loadable && " · soon"}
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
                <label className="flex cursor-pointer items-center gap-2 border-t border-edge px-4 py-2.5 text-[12px] text-muted">
                  <input
                    type="checkbox"
                    checked={manualParams}
                    onChange={(e) => {
                      setManualParams(e.target.checked);
                      savePrefs({
                        ...loadPrefs(),
                        manualParams: e.target.checked,
                      });
                    }}
                    className="accent-accent"
                  />
                  Manually choose model load parameters
                </label>
              </div>
            </>
          )}
        </div>

        {/* Messages / empty state */}
        {chat.messages.length === 0 ? (
          <div className="flex flex-1 items-center justify-center px-6">
            <div className="max-w-sm text-center">
              <p className="text-[15px] font-medium">
                {phase === "ready" && selected
                  ? selected.display_name
                  : phase === "loading"
                    ? "Loading model…"
                    : "No model loaded"}
              </p>
              <p className="mt-1 text-muted">
                {phase === "ready"
                  ? "Say something to start the conversation."
                  : "Pick a model from the Library, or press ⌘L to load one and start chatting."}
              </p>
            </div>
          </div>
        ) : (
          <div ref={scrollRef} className="min-h-0 flex-1 overflow-auto px-6 py-6">
            <div className="mx-auto flex max-w-3xl flex-col gap-5">
              {chat.messages.map((m, i) =>
                m.role === "user" ? (
                  <div
                    key={i}
                    className="max-w-[80%] self-end rounded-2xl bg-panel-2 px-4 py-2.5 whitespace-pre-wrap"
                  >
                    {m.content}
                  </div>
                ) : (
                  <div key={i} className="whitespace-pre-wrap">
                    {m.content}
                    {sending && i === chat.messages.length - 1 && (
                      <Loader2
                        size={13}
                        className="ml-1.5 inline animate-spin text-muted"
                      />
                    )}
                  </div>
                ),
              )}
            </div>
          </div>
        )}

        {/* Composer */}
        <div className="mx-auto w-full max-w-3xl px-6 pb-5">
          {error && (
            <p className="pb-2 text-center text-[12px] text-red-400">{error}</p>
          )}
          <div className="rounded-2xl border border-edge bg-panel p-3 shadow-lg">
            <textarea
              rows={3}
              value={input}
              disabled={phase !== "ready"}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void send();
                }
              }}
              placeholder="Send a message to the model..."
              className="w-full resize-none bg-transparent outline-none placeholder:text-muted disabled:cursor-not-allowed"
            />
            {enabledIntegrations.length > 0 && (
              <div className="flex flex-wrap gap-1.5 pb-2">
                {enabledIntegrations.map((integration) => (
                  <span
                    key={integration.id}
                    className="flex items-center gap-1.5 rounded-full bg-primary/20 px-2.5 py-1 text-[11px] text-primary"
                  >
                    {integration.id}
                    <button
                      type="button"
                      title="Disable integration"
                      onClick={() => toggleIntegration(integration.id)}
                      className="hover:text-ink"
                    >
                      <X size={11} />
                    </button>
                  </span>
                ))}
              </div>
            )}
            <div className="flex items-center justify-between">
              <button
                type="button"
                disabled
                className="flex h-7 w-7 items-center justify-center rounded-full text-muted disabled:cursor-not-allowed"
                title="Attach a file"
              >
                <Paperclip size={15} />
              </button>
              <div className="flex items-center gap-3">
                <span
                  className="text-[11px] text-muted"
                  title={
                    usage?.prompt_tokens != null
                      ? `Last response used ${usage.prompt_tokens} prompt tokens of a ${ctxWindow} window`
                      : "Context usage — no response yet"
                  }
                >
                  {ctxPct}% context
                </span>
                <button
                  type="button"
                  onClick={() => void send()}
                  disabled={phase !== "ready" || sending || input.trim() === ""}
                  className="flex h-7 w-7 items-center justify-center rounded-full bg-primary text-white transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
                  title="Send"
                >
                  <ArrowUp size={15} />
                </button>
              </div>
            </div>
          </div>
          <p className="mt-2 text-center text-[11px] text-muted">
            QuackStudio runs models locally — nothing leaves this machine.
          </p>
        </div>
      </div>

      {/* Right panel: Integrations / Configuration */}
      {panelOpen && (
        <aside className="flex w-80 shrink-0 flex-col border-l border-edge bg-panel">
          <div className="flex gap-1 border-b border-edge p-2">
            {(
              [
                ["integrations", "Integrations", Wrench],
                ["configuration", "Configuration", SlidersHorizontal],
              ] as [PanelTab, string, typeof Wrench][]
            ).map(([value, label, Icon]) => (
              <button
                key={value}
                type="button"
                onClick={() => setPanelTab(value)}
                className={`flex flex-1 items-center justify-center gap-1.5 rounded-lg px-2 py-1.5 text-[12px] transition-colors ${
                  panelTab === value
                    ? "bg-primary text-white"
                    : "text-muted hover:bg-panel-2 hover:text-ink"
                }`}
              >
                <Icon size={13} />
                {label}
              </button>
            ))}
          </div>

          {panelTab === "integrations" && (
            <div className="min-h-0 flex-1 overflow-auto px-4 py-3">
              <button
                type="button"
                disabled
                title="Plugin installation is not implemented in QuackStudio"
                className="w-full rounded-lg border border-edge bg-panel-2 py-2 text-[13px] text-muted disabled:cursor-not-allowed"
              >
                + Install
              </button>
              <input
                value={integrationFilter}
                onChange={(e) => setIntegrationFilter(e.target.value)}
                placeholder="Filter plugins…"
                className="mt-3 w-full rounded-lg border border-edge bg-app-bg px-3 py-1.5 outline-none placeholder:text-muted"
              />
              <div className="mt-2 flex flex-col divide-y divide-edge/60">
                {INTEGRATIONS.filter(
                  (i) =>
                    integrationFilter.trim() === "" ||
                    i.id.toLowerCase().includes(integrationFilter.toLowerCase()),
                ).map((integration) => (
                  <div
                    key={integration.id}
                    className="flex items-center justify-between py-2.5"
                  >
                    <span className="truncate text-[13px]">
                      {integration.id}
                    </span>
                    <Toggle
                      on={!!enabled[integration.id]}
                      onClick={() => toggleIntegration(integration.id)}
                    />
                  </div>
                ))}
              </div>
              <p className="mt-3 text-[11px] text-muted">
                Toggles are local UI state — these plugins are not
                connected to quackd.
              </p>
            </div>
          )}

          {panelTab === "configuration" && (
            <div className="min-h-0 flex-1 overflow-auto px-4 py-3">
              <p className="text-[12px] font-medium text-muted">Preset</p>
              <select
                value={presetName}
                onChange={(e) => applyPreset(e.target.value)}
                className="mt-1.5 w-full rounded-lg border border-edge bg-app-bg px-2.5 py-1.5 outline-none"
              >
                <option value="">Unsaved Preset</option>
                {Object.keys(presets).map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
              <div className="mt-2 flex gap-2">
                <button
                  type="button"
                  disabled={presetName === ""}
                  onClick={() => {
                    if (!chat) return;
                    const inf = selected
                      ? loadInference(selected.key)
                      : null;
                    updateChat({
                      systemPrompt: "",
                      temperature: inf?.temperature ?? 0.8,
                      topP: inf?.topP ?? 1,
                      reasoningBudget: "Unrestricted",
                    });
                    setPresetName("");
                  }}
                  className="flex-1 rounded-lg border border-edge px-2 py-1.5 text-[12px] transition-colors hover:bg-panel-2 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Discard unsaved
                </button>
                <button
                  type="button"
                  onClick={savePresetAs}
                  className="flex-1 rounded-lg border border-edge px-2 py-1.5 text-[12px] transition-colors hover:bg-panel-2"
                >
                  Save Preset As…
                </button>
              </div>

              <p className="mt-5 text-[12px] font-medium text-muted">
                System Prompt
              </p>
              <textarea
                rows={4}
                value={chat.systemPrompt}
                onChange={(e) => updateChat({ systemPrompt: e.target.value })}
                placeholder="You are a helpful assistant…"
                className="mt-1.5 w-full resize-none rounded-lg border border-edge bg-app-bg px-2.5 py-2 outline-none placeholder:text-muted"
              />
              <p className="mt-1 text-[11px] text-muted">
                Sent as the system message with every request in this
                chat.
              </p>

              <p className="mt-5 text-[12px] font-medium text-muted">
                Reasoning Budget
              </p>
              <select
                value={chat.reasoningBudget}
                onChange={(e) => updateChat({ reasoningBudget: e.target.value })}
                className="mt-1.5 w-full rounded-lg border border-edge bg-app-bg px-2.5 py-1.5 outline-none"
              >
                {["Unrestricted", "High", "Medium", "Low", "Off"].map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
              <p className="mt-1 text-[11px] text-muted">
                Stored per chat; not sent to the engine.
              </p>

              <p className="mt-5 text-[12px] font-medium text-muted">
                Settings
              </p>
              <div className="mt-2">
                <div className="flex items-center justify-between text-[13px]">
                  <span>Temperature</span>
                  <span className="text-muted">
                    {chat.temperature.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.05}
                  value={chat.temperature}
                  onChange={(e) =>
                    updateChat({ temperature: Number(e.target.value) })
                  }
                  className="mt-1 w-full accent-accent"
                />
              </div>
              <div className="mt-3">
                <div className="flex items-center justify-between text-[13px]">
                  <span>Top P</span>
                  <span className="text-muted">{chat.topP.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={chat.topP}
                  onChange={(e) => updateChat({ topP: Number(e.target.value) })}
                  className="mt-1 w-full accent-accent"
                />
              </div>

              <p className="mt-5 text-[12px] font-medium text-muted">
                Conversation Notes
              </p>
              <div className="mt-1.5 flex flex-col gap-1.5">
                {chat.notes.map((note, i) => (
                  <p
                    key={i}
                    className="rounded-lg bg-panel-2 px-2.5 py-1.5 text-[12px]"
                  >
                    {note}
                  </p>
                ))}
                <div className="flex gap-1.5">
                  <input
                    value={noteDraft}
                    onChange={(e) => setNoteDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && noteDraft.trim()) {
                        updateChat({ notes: [...chat.notes, noteDraft.trim()] });
                        setNoteDraft("");
                      }
                    }}
                    placeholder="Add a note"
                    className="min-w-0 flex-1 rounded-lg border border-edge bg-app-bg px-2.5 py-1.5 outline-none placeholder:text-muted"
                  />
                  <button
                    type="button"
                    disabled={noteDraft.trim() === ""}
                    onClick={() => {
                      updateChat({ notes: [...chat.notes, noteDraft.trim()] });
                      setNoteDraft("");
                    }}
                    className="rounded-lg border border-edge px-2.5 text-[12px] transition-colors hover:bg-panel-2 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    Add
                  </button>
                </div>
              </div>
            </div>
          )}
        </aside>
      )}
    </div>
  );
}
