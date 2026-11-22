import { useEffect, useRef, useState } from "react";
import { ArrowUp, ChevronDown, Loader2, Paperclip } from "lucide-react";
import type { ModelEntry } from "../types";
import { formatGB } from "../types";
import { apiBase } from "../api";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

type LoadPhase = "none" | "loading" | "ready" | "error";

/** Chat page: model picker pill (loads through quackd's format
 *  dispatch — llama.cpp for GGUF, oMLX for MLX; Maya models are
 *  remote and always ready), streaming chat over the
 *  OpenAI-compat proxy. */
export default function Chat() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [selected, setSelected] = useState<ModelEntry | null>(null);
  const [phase, setPhase] = useState<LoadPhase>("none");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Load the library; if quackd already has an instance running
  // (e.g. loaded from the CLI), adopt it so the chat just works.
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
          // Library's "Use in New Chat" leaves the model key in
          // sessionStorage; adopt (and load) it on arrival.
          const pending = sessionStorage.getItem("quack.useInChat");
          if (pending) {
            sessionStorage.removeItem("quack.useInChat");
            const wanted = data.find((m) => m.key === pending);
            if (wanted) {
              void selectModel(wanted);
              return;
            }
          }
          const already = data.find((m) => m.loaded_instances.length > 0);
          if (already) {
            setSelected(already);
            setPhase("ready");
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
  }, []);

  // Keep the newest message in view while streaming.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  async function selectModel(model: ModelEntry) {
    setPickerOpen(false);
    setError(null);
    setSelected(model);
    if (model.loaded_instances.length > 0) {
      setPhase("ready");
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
    } catch (e) {
      setPhase("error");
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function send() {
    const text = input.trim();
    if (!text || !selected || phase !== "ready" || sending) return;
    const history: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages([...history, { role: "assistant", content: "" }]);
    setInput("");
    setSending(true);
    setError(null);
    try {
      const base = await apiBase();
      const res = await fetch(`${base}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: selected.key,
          messages: history,
          stream: true,
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
          const payload = trimmed.slice("data:".length).trim();
          if (payload === "[DONE]") continue;
          try {
            const json = JSON.parse(payload) as {
              choices?: { delta?: { content?: string } }[];
            };
            const delta = json.choices?.[0]?.delta?.content ?? "";
            if (delta) {
              setMessages((prev) => {
                const next = [...prev];
                const last = next[next.length - 1];
                next[next.length - 1] = {
                  ...last,
                  content: last.content + delta,
                };
                return next;
              });
            }
          } catch {
            // A chunk boundary split a JSON payload; the proxy
            // forwards SSE verbatim, so this should not happen —
            // skip the fragment rather than kill the stream.
          }
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSending(false);
    }
  }

  const pillLabel =
    phase === "loading"
      ? `Loading ${selected?.display_name ?? "model"}…`
      : selected
        ? selected.display_name
        : "Select a model to load";

  return (
    <div className="relative flex h-full flex-col">
      {/* Model picker pill + dropdown */}
      <div className="relative flex justify-center pt-4">
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
            <div className="absolute top-full z-20 mt-2 max-h-80 w-96 overflow-auto rounded-xl border border-edge bg-panel shadow-xl">
              {models.length === 0 && (
                <p className="px-4 py-3 text-muted">No models in the library.</p>
              )}
              {models.map((m) => {
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
          </>
        )}
      </div>

      {/* Messages / empty state */}
      {messages.length === 0 ? (
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
            {messages.map((m, i) =>
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
                  {sending && i === messages.length - 1 && (
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
          <div className="flex items-center justify-between">
            <button
              type="button"
              disabled
              className="flex h-7 w-7 items-center justify-center rounded-full text-muted disabled:cursor-not-allowed"
              title="Attach a file"
            >
              <Paperclip size={15} />
            </button>
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
        <p className="mt-2 text-center text-[11px] text-muted">
          QuackStudio runs models locally — nothing leaves this machine.
        </p>
      </div>
    </div>
  );
}
