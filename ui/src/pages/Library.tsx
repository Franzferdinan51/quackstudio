import { useCallback, useEffect, useState } from "react";
import { Check, Loader2, MessageSquare, Search, Upload, X } from "lucide-react";
import type { LibraryCategory, ModelEntry } from "../types";
import { formatGB } from "../types";
import { apiBase } from "../api";
import type { InferenceDefaults } from "../store";
import { loadInference, saveInference } from "../store";

/** Shown (clearly labeled) only when quackd is unreachable. */
const SAMPLE_ROWS: ModelEntry[] = [
  {
    key: "sample/llama-3.1-8b-instruct",
    publisher: "sample",
    display_name: "llama-3.1-8b-instruct",
    format: "gguf",
    size_bytes: 4_920_000_000,
    path: "",
    arch: "llama",
    vision: false,
    remote_url: null,
    loaded_instances: [],
  },
  {
    key: "sample/qwen2.5-7b-instruct-4bit",
    publisher: "sample",
    display_name: "qwen2.5-7b-instruct-4bit",
    format: "mlx",
    size_bytes: 4_270_000_000,
    path: "",
    arch: "Qwen2ForCausalLM",
    vision: true,
    remote_url: null,
    loaded_instances: [],
  },
  {
    key: "sample/arete-27b-splash",
    publisher: "sample",
    display_name: "arete-27b-splash",
    format: "splash",
    size_bytes: 14_800_000_000,
    path: "",
    arch: null,
    vision: true,
    remote_url: null,
    loaded_instances: [],
  },
  {
    key: "maya/ik2/inkling-2-mini",
    publisher: "maya",
    display_name: "inkling-2-mini",
    format: "maya",
    size_bytes: 0,
    path: "",
    arch: null,
    vision: false,
    remote_url: "http://100.68.208.113:8085",
    loaded_instances: [
      {
        instance_id: "maya/ik2/inkling-2-mini",
        model_key: "maya/ik2/inkling-2-mini",
        backend: "maya (ik2)",
        port: 8085,
        pid: 0,
      },
    ],
  },
];

type DetailTab = "info" | "load" | "inference";
type LoadMessage = { kind: "ok" | "error"; text: string };

interface LoadAdvanced {
  nThreads: string;
  evalBatch: string;
  physBatch: string;
}

const ADV_KEY = (key: string) => `quack.loadAdv.${key}`;

function loadAdvanced(key: string): LoadAdvanced {
  try {
    const raw = localStorage.getItem(ADV_KEY(key));
    if (raw) return JSON.parse(raw) as LoadAdvanced;
  } catch {
    // fall through to blanks
  }
  return { nThreads: "", evalBatch: "", physBatch: "" };
}

/** Category membership, from the fields the scanner gives us.
 *  Approximations (no type field exists on ModelEntry):
 *  embedding = name mentions nomic/embed; tokenizer/audio
 *  (speech_tokenizer) is neither an LLM nor an embedding;
 *  decision = d1-/decision in the key; drafters = draft in name. */
function inCategory(m: ModelEntry, category: LibraryCategory): boolean {
  const name = `${m.key} ${m.display_name}`.toLowerCase();
  const isEmbedding = name.includes("nomic") || name.includes("embed");
  const isTokenizer = name.includes("tokenizer");
  switch (category) {
    case "all":
      return true;
    case "llms":
      return !isEmbedding && !isTokenizer;
    case "embedding":
      return isEmbedding;
    case "decision":
      return name.includes("d1-") || name.includes("decision");
    case "drafters":
      return name.includes("drafter") || name.includes("draft");
  }
}

/** Library (My Models): category-filtered model table fed by
 *  quackd, footer totals, and a detail panel (Info / Load /
 *  Inference) for the selected model. */
export default function Library({ category }: { category: LibraryCategory }) {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [offline, setOffline] = useState(false);
  const [filter, setFilter] = useState("");
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [tab, setTab] = useState<DetailTab>("info");
  const [ctxLength, setCtxLength] = useState(8192);
  const [defaultCtx, setDefaultCtx] = useState(8192);
  const [modelsDir, setModelsDir] = useState("");
  const [gpuLayers, setGpuLayers] = useState(99);
  const [advanced, setAdvanced] = useState<LoadAdvanced>({
    nThreads: "",
    evalBatch: "",
    physBatch: "",
  });
  const [inference, setInference] = useState<InferenceDefaults | null>(null);
  const [loadBusy, setLoadBusy] = useState(false);
  const [loadMsg, setLoadMsg] = useState<LoadMessage | null>(null);
  const [unloadingId, setUnloadingId] = useState<string | null>(null);

  const fetchModels = useCallback(async () => {
    try {
      const base = await apiBase();
      const res = await fetch(`${base}/api/v1/models`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as ModelEntry[];
      setModels(data);
      setOffline(false);
    } catch {
      setOffline(true);
    }
  }, []);

  useEffect(() => {
    void fetchModels();
  }, [fetchModels]);

  // Settings defaults: the Load tab's initial context length and
  // the footer models-dir path.
  useEffect(() => {
    void (async () => {
      try {
        const base = await apiBase();
        const res = await fetch(`${base}/api/v1/settings`);
        if (!res.ok) return;
        const view = (await res.json()) as {
          models_dir: string;
          defaults?: { context_length: number | null };
        };
        setModelsDir(view.models_dir);
        if (view.defaults?.context_length) {
          setDefaultCtx(view.defaults.context_length);
          setCtxLength(view.defaults.context_length);
        }
      } catch {
        // Defaults are best-effort.
      }
    })();
  }, []);

  // Downloads pane / model picker "Load Model" handoff: a model
  // key in sessionStorage + this event (or already set at mount).
  const adoptLoadHandoff = useCallback(() => {
    const pending = sessionStorage.getItem("quack.loadModel");
    if (pending) {
      sessionStorage.removeItem("quack.loadModel");
      setSelectedKey(pending);
      setTab("load");
    }
  }, []);
  useEffect(() => {
    adoptLoadHandoff();
    window.addEventListener("quack:open-load", adoptLoadHandoff);
    return () =>
      window.removeEventListener("quack:open-load", adoptLoadHandoff);
  }, [adoptLoadHandoff]);

  // Fresh selection → back to Info (unless a load handoff just
  // chose the Load tab), default load config, no stale result.
  useEffect(() => {
    setCtxLength(defaultCtx);
    setGpuLayers(99);
    setLoadMsg(null);
    if (selectedKey) {
      setAdvanced(loadAdvanced(selectedKey));
      setInference(loadInference(selectedKey));
    }
  }, [selectedKey, defaultCtx]);

  const rows = offline ? SAMPLE_ROWS : models;
  const needle = filter.trim().toLowerCase();
  const visible = rows.filter(
    (m) =>
      inCategory(m, category) &&
      (needle === "" ||
        `${m.publisher}/${m.display_name}`.toLowerCase().includes(needle)),
  );
  const selected = rows.find((m) => m.key === selectedKey) ?? null;
  const localRows = rows.filter((m) => m.format !== "maya");
  const totalBytes = localRows.reduce((sum, m) => sum + m.size_bytes, 0);

  async function loadSelected() {
    if (!selected || loadBusy) return;
    setLoadBusy(true);
    setLoadMsg(null);
    try {
      const base = await apiBase();
      const body: Record<string, unknown> = {
        model: selected.key,
        context_length: ctxLength,
        gpu_offload: gpuLayers,
      };
      // Advanced knobs: forwarded to llama-server as -t/-b/-ub by
      // quackd for GGUF loads; stored-but-ignored for MLX.
      const nThreads = parseInt(advanced.nThreads, 10);
      const evalBatch = parseInt(advanced.evalBatch, 10);
      const physBatch = parseInt(advanced.physBatch, 10);
      if (Number.isFinite(nThreads)) body.n_threads = nThreads;
      if (Number.isFinite(evalBatch)) body.eval_batch_size = evalBatch;
      if (Number.isFinite(physBatch)) body.physical_batch_size = physBatch;
      const res = await fetch(`${base}/api/v1/models/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
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
      const ok = data as {
        instance_id?: string;
        load_time_seconds?: number;
      };
      setLoadMsg({
        kind: "ok",
        text: `Loaded as ${ok.instance_id ?? "?"} in ${(ok.load_time_seconds ?? 0).toFixed(1)}s`,
      });
      await fetchModels();
    } catch (e) {
      setLoadMsg({
        kind: "error",
        text: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setLoadBusy(false);
    }
  }

  async function unloadInstance(instanceId: string) {
    setUnloadingId(instanceId);
    setLoadMsg(null);
    try {
      const base = await apiBase();
      const res = await fetch(`${base}/api/v1/models/unload`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instance_id: instanceId }),
      });
      if (!res.ok) {
        const data: unknown = await res.json().catch(() => null);
        const message =
          data !== null &&
          typeof data === "object" &&
          "error" in data &&
          typeof (data as { error: unknown }).error === "string"
            ? (data as { error: string }).error
            : `unload failed (HTTP ${res.status})`;
        throw new Error(message);
      }
      setLoadMsg({ kind: "ok", text: `Unloaded ${instanceId}` });
      await fetchModels();
    } catch (e) {
      setLoadMsg({
        kind: "error",
        text: e instanceof Error ? e.message : String(e),
      });
    } finally {
      setUnloadingId(null);
    }
  }

  function useInNewChat(model: ModelEntry) {
    sessionStorage.setItem("quack.useInChat", model.key);
    window.dispatchEvent(new Event("quack:use-in-chat"));
  }

  function updateAdvanced(patch: Partial<LoadAdvanced>) {
    const next = { ...advanced, ...patch };
    setAdvanced(next);
    if (selectedKey) {
      try {
        localStorage.setItem(ADV_KEY(selectedKey), JSON.stringify(next));
      } catch {
        // best-effort
      }
    }
  }

  function updateInference(patch: Partial<InferenceDefaults>) {
    if (!inference || !selectedKey) return;
    const next = { ...inference, ...patch };
    setInference(next);
    saveInference(selectedKey, next);
  }

  return (
    <div className="flex h-full">
      {/* List column */}
      <div className="flex min-w-0 flex-1 flex-col px-6 py-4">
        {/* Filter bar */}
        <div className="flex items-center gap-3">
          <div className="flex w-72 items-center gap-2 rounded-lg border border-edge bg-panel px-3 py-1.5">
            <Search size={14} className="shrink-0 text-muted" />
            <input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter models…"
              className="w-full bg-transparent outline-none placeholder:text-muted"
            />
          </div>
          <span className="text-muted">
            {visible.length} model{visible.length === 1 ? "" : "s"}
          </span>
          {offline && (
            <span className="rounded-full border border-edge bg-panel px-2.5 py-0.5 text-[11px] text-muted">
              daemon offline — showing sample rows
            </span>
          )}
        </div>

        {/* Table */}
        <div className="mt-4 min-h-0 flex-1 overflow-auto rounded-xl border border-edge">
          <table className="w-full border-collapse text-left">
            <thead className="sticky top-0 bg-panel">
              <tr className="border-b border-edge text-[11px] uppercase tracking-wide text-muted">
                <th className="px-4 py-2 font-medium">Model</th>
                <th className="px-4 py-2 font-medium">Device</th>
                <th className="px-4 py-2 font-medium">Arch</th>
                <th className="px-4 py-2 font-medium">Params</th>
                <th className="px-4 py-2 font-medium">Size</th>
                <th className="px-4 py-2 font-medium">Publisher</th>
                <th className="px-4 py-2 font-medium">LLM</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((m) => (
                <tr
                  key={m.key}
                  onClick={() => {
                    setSelectedKey(m.key);
                    setTab("info");
                  }}
                  className={`cursor-pointer border-b border-edge/50 transition-colors last:border-0 hover:bg-panel ${
                    m.key === selectedKey ? "bg-panel-2" : ""
                  }`}
                >
                  <td className="px-4 py-2.5">
                    <span className="font-medium">{m.display_name}</span>
                    {offline && (
                      <span className="ml-2 rounded bg-panel-2 px-1.5 py-0.5 text-[10px] text-muted">
                        sample
                      </span>
                    )}
                    {m.vision && (
                      <span className="ml-2 rounded bg-accent/20 px-1.5 py-0.5 text-[10px] text-accent">
                        vision
                      </span>
                    )}
                    {m.format === "maya" && (
                      <span className="ml-2 rounded bg-primary/20 px-1.5 py-0.5 text-[10px] text-primary">
                        Remote · Maya
                      </span>
                    )}
                    {m.loaded_instances.length > 0 && (
                      <span className="ml-2 rounded bg-accent/20 px-1.5 py-0.5 text-[10px] text-accent">
                        loaded
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-2.5 text-muted">
                    {m.format === "maya" ? "Remote" : "Local"}
                  </td>
                  <td className="px-4 py-2.5 text-muted">{m.arch ?? "—"}</td>
                  <td className="px-4 py-2.5 text-muted">—</td>
                  <td className="px-4 py-2.5 text-muted">
                    {m.format === "maya" ? "—" : formatGB(m.size_bytes)}
                  </td>
                  <td className="px-4 py-2.5 text-muted">{m.publisher}</td>
                  <td className="px-4 py-2.5">
                    <span className="flex items-center gap-1.5">
                      <Check size={13} className="text-ok" />
                      <span className="text-muted">
                        {m.format.toUpperCase()}
                      </span>
                    </span>
                  </td>
                </tr>
              ))}
              {visible.length === 0 && !offline && (
                <tr>
                  <td colSpan={7} className="px-4 py-10 text-center text-muted">
                    No models in this category. Download one from Model
                    Search.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Footer totals */}
        <p className="pt-3 text-[12px] text-muted">
          You have {localRows.length} local models, taking up{" "}
          {formatGB(totalBytes)} of disk space
          {modelsDir && (
            <>
              {" · "}
              <span className="font-mono text-[11px]">{modelsDir}</span>
            </>
          )}
        </p>
      </div>

      {/* Detail panel */}
      {selected && (
        <aside className="flex w-[380px] shrink-0 flex-col overflow-auto border-l border-edge bg-panel px-5 py-4">
          <div className="flex items-start gap-2">
            <div className="min-w-0 flex-1">
              <p className="text-[15px] font-medium break-words">
                {selected.display_name}
              </p>
              <p className="mt-0.5 text-[12px] text-muted">
                {selected.publisher} · {selected.format.toUpperCase()}
                {selected.vision ? " · vision" : ""}
              </p>
            </div>
            <button
              type="button"
              title="Close"
              onClick={() => setSelectedKey(null)}
              className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-muted transition-colors hover:bg-panel-2 hover:text-ink"
            >
              <X size={15} />
            </button>
          </div>

          <div className="mt-4 flex gap-2">
            <button
              type="button"
              disabled={selected.format === "splash"}
              title={
                selected.format === "splash"
                  ? "Splash arrives with the Strata pack (P3)"
                  : undefined
              }
              onClick={() => useInNewChat(selected)}
              className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-panel-2 px-3 py-2 text-[13px] transition-colors hover:bg-edge disabled:cursor-not-allowed disabled:opacity-40"
            >
              <MessageSquare size={14} />
              Use in New Chat
            </button>
            <button
              type="button"
              onClick={() => setTab("load")}
              className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-panel-2 px-3 py-2 text-[13px] transition-colors hover:bg-edge"
            >
              <Upload size={14} />
              Load Model
            </button>
          </div>

          {/* Tabs */}
          <div className="mt-5 flex gap-4 border-b border-edge">
            {(
              [
                ["info", "Info"],
                ["load", "Load"],
                ["inference", "Inference"],
              ] as [DetailTab, string][]
            ).map(([value, label]) => (
              <button
                key={value}
                type="button"
                onClick={() => setTab(value)}
                className={`border-b-2 px-1 pb-2 text-[13px] transition-colors ${
                  tab === value
                    ? "border-accent text-ink"
                    : "border-transparent text-muted hover:text-ink"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {tab === "info" && (
            <dl className="mt-4 flex flex-col gap-3 text-[13px]">
              {[
                ["Format", selected.format.toUpperCase()],
                ["Arch", selected.arch ?? "—"],
                [
                  "Size",
                  selected.format === "maya"
                    ? "—"
                    : formatGB(selected.size_bytes),
                ],
                ["Vision", selected.vision ? "Yes" : "No"],
                ["Publisher", selected.publisher],
                ["Key", selected.key],
              ].map(([label, value]) => (
                <div key={label} className="flex items-baseline gap-3">
                  <dt className="w-20 shrink-0 text-muted">{label}</dt>
                  <dd className="min-w-0 break-words">{value}</dd>
                </div>
              ))}
              {selected.remote_url ? (
                <div className="flex items-baseline gap-3">
                  <dt className="w-20 shrink-0 text-muted">Endpoint</dt>
                  <dd className="min-w-0 text-[12px] break-all text-muted">
                    {selected.remote_url}
                  </dd>
                </div>
              ) : (
                <div className="flex items-baseline gap-3">
                  <dt className="w-20 shrink-0 text-muted">Path</dt>
                  <dd className="min-w-0 text-[12px] break-all text-muted">
                    {selected.path || "—"}
                  </dd>
                </div>
              )}
            </dl>
          )}

          {tab === "inference" && inference && (
            <div className="mt-4 flex flex-col gap-4">
              <p className="text-[12px] text-muted">
                Defaults for fresh chats with this model (stored
                locally). Temperature and Top P seed the Chat
                Configuration panel; the rest are recorded for when
                the engines accept them.
              </p>
              {(
                [
                  ["temperature", "Temperature", 0, 2, 0.05],
                  ["topP", "Top P", 0, 1, 0.05],
                  ["topK", "Top K", 0, 200, 1],
                  ["minP", "Min P", 0, 1, 0.01],
                  ["repeatPenalty", "Repeat Penalty", 0.5, 2, 0.05],
                ] as [
                  keyof InferenceDefaults,
                  string,
                  number,
                  number,
                  number,
                ][]
              ).map(([field, label, min, max, step]) => (
                <div key={field}>
                  <div className="flex items-center justify-between">
                    <label className="text-[13px] text-muted">{label}</label>
                    <input
                      type="number"
                      min={min}
                      max={max}
                      step={step}
                      value={inference[field]}
                      onChange={(e) =>
                        updateInference({
                          [field]: Number(e.target.value) || 0,
                        })
                      }
                      className="w-24 rounded-lg border border-edge bg-panel-2 px-2 py-1 text-right text-[13px] outline-none"
                    />
                  </div>
                  <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={inference[field]}
                    onChange={(e) =>
                      updateInference({ [field]: Number(e.target.value) })
                    }
                    className="mt-2 w-full accent-accent"
                  />
                </div>
              ))}
            </div>
          )}

          {tab === "load" && selected.format === "maya" && (
            <div className="mt-4 flex flex-col gap-5">
              <p className="rounded-xl border border-edge bg-panel-2 px-3.5 py-3 text-[12px] text-muted">
                Served by Maya at {selected.remote_url} — nothing to
                load locally; pick it in Chat. Maya manages this
                model's lifecycle, so there are no load or unload
                controls here.
              </p>
            </div>
          )}

          {tab === "load" && selected.format !== "maya" && (
            <div className="mt-4 flex flex-col gap-5">
              {selected.loaded_instances.map((inst) => (
                <div
                  key={inst.instance_id}
                  className="rounded-xl border border-edge bg-panel-2 px-3.5 py-3"
                >
                  <p className="text-[13px] font-medium break-all">
                    {inst.instance_id}
                  </p>
                  <p className="mt-0.5 text-[12px] text-muted">
                    {inst.backend} · port {inst.port} · pid {inst.pid}
                  </p>
                  <button
                    type="button"
                    disabled={unloadingId === inst.instance_id}
                    onClick={() => void unloadInstance(inst.instance_id)}
                    className="mt-2.5 flex items-center gap-1.5 rounded-lg bg-panel px-3 py-1.5 text-[12px] transition-colors hover:bg-edge disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {unloadingId === inst.instance_id && (
                      <Loader2 size={13} className="animate-spin" />
                    )}
                    Unload
                  </button>
                </div>
              ))}

              <div>
                <div className="flex items-center justify-between">
                  <label
                    htmlFor="qs-ctx"
                    className="text-[13px] text-muted"
                  >
                    Context Length
                  </label>
                  <input
                    id="qs-ctx"
                    type="number"
                    min={1024}
                    max={262144}
                    step={1024}
                    value={ctxLength}
                    onChange={(e) =>
                      setCtxLength(
                        Math.min(
                          262144,
                          Math.max(1024, Number(e.target.value) || 1024),
                        ),
                      )
                    }
                    className="w-24 rounded-lg border border-edge bg-panel-2 px-2 py-1 text-right text-[13px] outline-none"
                  />
                </div>
                <input
                  type="range"
                  min={1024}
                  max={262144}
                  step={1024}
                  value={ctxLength}
                  onChange={(e) => setCtxLength(Number(e.target.value))}
                  className="mt-2 w-full accent-accent"
                />
                {selected.format !== "gguf" && (
                  <p className="mt-1 text-[11px] text-muted">
                    Context length is passed to llama.cpp only; the MLX
                    server uses its own default.
                  </p>
                )}
              </div>

              <div>
                <div className="flex items-center justify-between">
                  <label
                    htmlFor="qs-gpu"
                    className="text-[13px] text-muted"
                  >
                    GPU Offload (layers)
                  </label>
                  <span className="text-[13px]">{gpuLayers}</span>
                </div>
                <input
                  id="qs-gpu"
                  type="range"
                  min={0}
                  max={99}
                  step={1}
                  value={gpuLayers}
                  disabled={selected.format !== "gguf"}
                  onChange={(e) => setGpuLayers(Number(e.target.value))}
                  className="mt-2 w-full accent-accent disabled:cursor-not-allowed disabled:opacity-40"
                />
                {selected.format !== "gguf" && (
                  <p className="mt-1 text-[11px] text-muted">
                    GPU offload layers apply to GGUF (llama.cpp) models
                    only.
                  </p>
                )}
              </div>

              {/* Advanced (llama.cpp flags -t / -b / -ub) */}
              <div className="rounded-xl border border-edge px-3.5 py-3">
                <p className="text-[13px] font-medium">Advanced</p>
                {(
                  [
                    ["nThreads", "CPU Thread Pool Size"],
                    ["evalBatch", "Evaluation Batch Size"],
                    ["physBatch", "Physical Batch Size"],
                  ] as [keyof LoadAdvanced, string][]
                ).map(([field, label]) => (
                  <div
                    key={field}
                    className="mt-2.5 flex items-center justify-between gap-3"
                  >
                    <label className="text-[13px] text-muted">{label}</label>
                    <input
                      type="number"
                      min={1}
                      placeholder="engine default"
                      value={advanced[field]}
                      onChange={(e) =>
                        updateAdvanced({ [field]: e.target.value })
                      }
                      className="w-28 rounded-lg border border-edge bg-panel-2 px-2 py-1 text-right text-[13px] outline-none placeholder:text-[11px] placeholder:text-muted"
                    />
                  </div>
                ))}
                <p className="mt-2.5 text-[11px] text-muted">
                  {selected.format === "gguf"
                    ? "Forwarded to llama-server (-t / -b / -ub) when set."
                    : "MLX: not applied — mlx_vlm.server has no equivalents; values are stored per model."}
                </p>
              </div>

              {selected.format === "splash" && (
                <p className="rounded-xl border border-edge bg-panel-2 px-3.5 py-3 text-[12px] text-muted">
                  Splash arrives with the Strata pack (P3) — this model
                  can't be loaded yet.
                </p>
              )}

              <button
                type="button"
                disabled={selected.format === "splash" || loadBusy}
                onClick={() => void loadSelected()}
                className="flex items-center justify-center gap-2 rounded-lg bg-primary px-3 py-2 text-[13px] font-medium text-white transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
              >
                {loadBusy && <Loader2 size={14} className="animate-spin" />}
                {loadBusy ? "Loading…" : "Load Model"}
              </button>

              {loadMsg && (
                <p
                  className={`max-h-40 overflow-auto text-[12px] break-words whitespace-pre-wrap ${
                    loadMsg.kind === "ok" ? "text-ok" : "text-red-400"
                  }`}
                >
                  {loadMsg.text}
                </p>
              )}
            </div>
          )}
        </aside>
      )}
    </div>
  );
}
