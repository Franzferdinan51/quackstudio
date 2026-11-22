import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  ChevronRight,
  Cog,
  Copy,
  ExternalLink,
  Eye,
  FolderOpen,
  Loader2,
  MessageSquare,
  MoreHorizontal,
  Pin,
  Search,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import type {
  InferenceConfigWire,
  LibraryCategory,
  LoadConfigWire,
  ModelEntry,
} from "../types";
import { formatGB } from "../types";
import { apiBase } from "../api";
import Toggle from "../components/Toggle";
import {
  emptyModelConfig,
  forgetModelConfig,
  isCustomized,
  saveModelConfig,
  syncModelConfigs,
  type ModelConfigMap,
} from "../modelConfigs";

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
    file_name: "llama-3.1-8b-instruct-q4_k_m.gguf",
    quantization: "Q4_K_M",
    max_context: 131072,
    domain: "llm",
    warnings: [],
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
    file_name: "model.safetensors",
    quantization: "4-bit",
    max_context: 32768,
    domain: "llm",
    warnings: [],
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
    file_name: null,
    quantization: null,
    max_context: null,
    domain: "llm",
    warnings: [],
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
    file_name: null,
    quantization: null,
    max_context: null,
    domain: "llm",
    warnings: [],
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

// ---------------------------------------------------------------------------
// Load form (the Load tab's editable state; "" / null = unset)
// ---------------------------------------------------------------------------

interface LoadForm {
  contextLength: string;
  gpuOffload: string;
  cpuThreads: string;
  evalBatch: string;
  physBatch: string;
  maxConcurrent: string;
  unifiedKvCache: boolean | null;
  contextCheckpoints: string;
  reasoningBudgetMessage: string;
  ropeBaseOn: boolean;
  ropeBase: string;
  ropeScaleOn: boolean;
  ropeScale: string;
  kvOffload: boolean | null;
  keepInMemory: boolean | null;
  tryMmap: boolean | null;
  onDemand: string;
  seedOn: boolean;
  seed: string;
  numExperts: string;
  numLayersOffload: string;
  moeExpert: string;
  speculative: string;
  chatTemplate: string;
  flashAttention: boolean | null;
  cacheK: string;
  cacheV: string;
  argsOverrideOn: boolean;
  argsOverride: string;
}

function wireToLoadForm(w: LoadConfigWire, defaultCtx: number): LoadForm {
  const s = (v?: number) => (v === undefined ? "" : String(v));
  return {
    contextLength: s(w.context_length ?? defaultCtx),
    gpuOffload: s(w.gpu_offload ?? 99),
    cpuThreads: s(w.cpu_threads),
    evalBatch: s(w.eval_batch_size),
    physBatch: s(w.physical_batch_size),
    maxConcurrent: s(w.max_concurrent),
    unifiedKvCache: w.unified_kv_cache ?? null,
    contextCheckpoints: s(w.context_checkpoints),
    reasoningBudgetMessage: w.reasoning_budget_message ?? "",
    ropeBaseOn: w.rope_freq_base !== undefined,
    ropeBase: s(w.rope_freq_base),
    ropeScaleOn: w.rope_freq_scale !== undefined,
    ropeScale: s(w.rope_freq_scale),
    kvOffload: w.kv_offload ?? null,
    keepInMemory: w.keep_in_memory ?? null,
    tryMmap: w.try_mmap ?? null,
    onDemand: w.on_demand_tensor_loading ?? "auto",
    seedOn: w.seed !== undefined,
    seed: s(w.seed),
    numExperts: s(w.num_experts),
    numLayersOffload: s(w.num_layers_offload),
    moeExpert: s(w.moe_expert),
    speculative: w.speculative_decoding ?? "off",
    chatTemplate: w.chat_template ?? "",
    flashAttention: w.flash_attention ?? null,
    cacheK: w.cache_type_k ?? "",
    cacheV: w.cache_type_v ?? "",
    argsOverrideOn: w.args_override !== undefined,
    argsOverride: w.args_override ?? "",
  };
}

function loadFormToWire(f: LoadForm, defaultCtx: number): LoadConfigWire {
  const num = (s: string): number | undefined => {
    if (s.trim() === "") return undefined;
    const n = Number(s);
    return Number.isFinite(n) ? n : undefined;
  };
  const w: LoadConfigWire = {};
  const ctx = num(f.contextLength);
  if (ctx !== undefined && ctx !== defaultCtx) w.context_length = ctx;
  const gpu = num(f.gpuOffload);
  if (gpu !== undefined && gpu !== 99) w.gpu_offload = gpu;
  const cpu = num(f.cpuThreads);
  if (cpu !== undefined) w.cpu_threads = cpu;
  const eb = num(f.evalBatch);
  if (eb !== undefined) w.eval_batch_size = eb;
  const pb = num(f.physBatch);
  if (pb !== undefined) w.physical_batch_size = pb;
  const mc = num(f.maxConcurrent);
  if (mc !== undefined) w.max_concurrent = mc;
  if (f.unifiedKvCache !== null) w.unified_kv_cache = f.unifiedKvCache;
  const cc = num(f.contextCheckpoints);
  if (cc !== undefined) w.context_checkpoints = cc;
  if (f.reasoningBudgetMessage.trim()) {
    w.reasoning_budget_message = f.reasoningBudgetMessage;
  }
  if (f.ropeBaseOn) {
    const v = num(f.ropeBase);
    if (v !== undefined) w.rope_freq_base = v;
  }
  if (f.ropeScaleOn) {
    const v = num(f.ropeScale);
    if (v !== undefined) w.rope_freq_scale = v;
  }
  if (f.kvOffload !== null) w.kv_offload = f.kvOffload;
  if (f.keepInMemory !== null) w.keep_in_memory = f.keepInMemory;
  if (f.tryMmap !== null) w.try_mmap = f.tryMmap;
  if (f.onDemand !== "auto") w.on_demand_tensor_loading = f.onDemand;
  if (f.seedOn) {
    const v = num(f.seed);
    if (v !== undefined) w.seed = v;
  }
  const ne = num(f.numExperts);
  if (ne !== undefined) w.num_experts = ne;
  const nl = num(f.numLayersOffload);
  if (nl !== undefined) w.num_layers_offload = nl;
  const me = num(f.moeExpert);
  if (me !== undefined) w.moe_expert = me;
  if (f.speculative !== "off") w.speculative_decoding = f.speculative;
  if (f.chatTemplate.trim()) w.chat_template = f.chatTemplate;
  if (f.flashAttention !== null) w.flash_attention = f.flashAttention;
  if (f.cacheK) w.cache_type_k = f.cacheK;
  if (f.cacheV) w.cache_type_v = f.cacheV;
  if (f.argsOverrideOn && f.argsOverride.trim()) {
    w.args_override = f.argsOverride;
  }
  return w;
}

// ---------------------------------------------------------------------------
// Inference form
// ---------------------------------------------------------------------------

interface InferenceForm {
  systemPrompt: string;
  reasoningBudgetOn: boolean;
  reasoningBudget: string;
  temperature: number;
  limitResponse: boolean;
  maxResponseTokens: string;
  contextOverflow: string;
  stopStrings: string[];
  topK: number;
  repeatOn: boolean;
  repeatPenalty: number;
  presenceOn: boolean;
  presencePenalty: number;
  topPOn: boolean;
  topP: number;
  minPOn: boolean;
  minP: number;
  structuredOutput: boolean;
}

const INF_DEFAULTS: InferenceForm = {
  systemPrompt: "",
  reasoningBudgetOn: false,
  reasoningBudget: "",
  temperature: 0.8,
  limitResponse: false,
  maxResponseTokens: "",
  contextOverflow: "truncate_middle",
  stopStrings: [],
  topK: 40,
  repeatOn: true,
  repeatPenalty: 1.1,
  presenceOn: false,
  presencePenalty: 0,
  topPOn: true,
  topP: 0.95,
  minPOn: true,
  minP: 0.05,
  structuredOutput: false,
};

function wireToInferenceForm(w: InferenceConfigWire): InferenceForm {
  return {
    systemPrompt: w.system_prompt ?? "",
    reasoningBudgetOn: w.reasoning_budget_enabled ?? false,
    reasoningBudget:
      w.reasoning_budget === undefined ? "" : String(w.reasoning_budget),
    temperature: w.temperature ?? INF_DEFAULTS.temperature,
    limitResponse: w.limit_response_length ?? false,
    maxResponseTokens:
      w.max_response_tokens === undefined
        ? ""
        : String(w.max_response_tokens),
    contextOverflow: w.context_overflow ?? "truncate_middle",
    stopStrings: w.stop_strings ?? [],
    topK: w.top_k ?? 40,
    repeatOn: w.repeat_penalty_enabled ?? true,
    repeatPenalty: w.repeat_penalty ?? 1.1,
    presenceOn: w.presence_penalty_enabled ?? false,
    presencePenalty: w.presence_penalty ?? 0,
    topPOn: w.top_p_enabled ?? true,
    topP: w.top_p ?? 0.95,
    minPOn: w.min_p_enabled ?? true,
    minP: w.min_p ?? 0.05,
    structuredOutput: w.structured_output ?? false,
  };
}

function inferenceFormToWire(f: InferenceForm): InferenceConfigWire {
  const w: InferenceConfigWire = {};
  if (f.systemPrompt.trim()) w.system_prompt = f.systemPrompt;
  if (f.reasoningBudgetOn) {
    w.reasoning_budget_enabled = true;
    const n = Number(f.reasoningBudget);
    if (f.reasoningBudget.trim() && Number.isFinite(n)) {
      w.reasoning_budget = n;
    }
  }
  if (f.temperature !== INF_DEFAULTS.temperature) {
    w.temperature = f.temperature;
  }
  if (f.limitResponse) {
    w.limit_response_length = true;
    const n = Number(f.maxResponseTokens);
    if (f.maxResponseTokens.trim() && Number.isFinite(n)) {
      w.max_response_tokens = n;
    }
  }
  if (f.stopStrings.length > 0) w.stop_strings = f.stopStrings;
  if (f.topK !== INF_DEFAULTS.topK) w.top_k = f.topK;
  if (!f.repeatOn || f.repeatPenalty !== INF_DEFAULTS.repeatPenalty) {
    w.repeat_penalty_enabled = f.repeatOn;
    w.repeat_penalty = f.repeatPenalty;
  }
  if (f.presenceOn || f.presencePenalty !== INF_DEFAULTS.presencePenalty) {
    w.presence_penalty_enabled = f.presenceOn;
    w.presence_penalty = f.presencePenalty;
  }
  if (!f.topPOn || f.topP !== INF_DEFAULTS.topP) {
    w.top_p_enabled = f.topPOn;
    w.top_p = f.topP;
  }
  if (!f.minPOn || f.minP !== INF_DEFAULTS.minP) {
    w.min_p_enabled = f.minPOn;
    w.min_p = f.minP;
  }
  if (f.structuredOutput) w.structured_output = true;
  // The overflow mode rides along with any other customization,
  // so the displayed mode is the applied one.
  if (Object.keys(w).length > 0 || f.contextOverflow !== "truncate_middle") {
    w.context_overflow = f.contextOverflow;
  }
  return w;
}

// ---------------------------------------------------------------------------
// Category membership (unchanged approximations)
// ---------------------------------------------------------------------------

/** Category membership, from the fields the scanner gives us:
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

// ---------------------------------------------------------------------------
// Small building blocks
// ---------------------------------------------------------------------------

function Section({
  title,
  children,
  onReset,
  defaultOpen = true,
}: {
  title: string;
  children: ReactNode;
  onReset?: () => void;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-b border-edge/60 pb-4">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => setOpen(!open)}
          className="flex flex-1 items-center gap-1.5 text-left text-[13px] font-medium"
        >
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          {title}
        </button>
        {onReset && (
          <button
            type="button"
            onClick={onReset}
            className="text-[11px] text-muted transition-colors hover:text-ink"
          >
            Reset
          </button>
        )}
      </div>
      {open && <div className="mt-3 flex flex-col gap-3.5">{children}</div>}
    </div>
  );
}

/** Label + control row with the per-field reset (trash) button,
 *  shown only while the field carries a non-default value. */
function Field({
  label,
  experimental,
  note,
  onReset,
  children,
}: {
  label: string;
  experimental?: boolean;
  note?: string;
  onReset?: () => void;
  children: ReactNode;
}) {
  return (
    <div>
      <div className="flex items-center gap-2">
        <span className="flex-1 text-[13px] text-muted">
          {label}
          {experimental && (
            <span className="ml-1.5 rounded bg-panel-2 px-1 py-0.5 text-[9px] uppercase tracking-wide text-muted">
              Experimental
            </span>
          )}
        </span>
        {onReset && (
          <button
            type="button"
            title={`Reset ${label} to default`}
            onClick={onReset}
            className="text-muted transition-colors hover:text-red-400"
          >
            <Trash2 size={13} />
          </button>
        )}
        {children}
      </div>
      {note && <p className="mt-1 text-[11px] text-muted">{note}</p>}
    </div>
  );
}

const numInputCls =
  "w-24 rounded-lg border border-edge bg-panel-2 px-2 py-1 text-right text-[13px] outline-none";
const textInputCls =
  "w-full rounded-lg border border-edge bg-panel-2 px-2 py-1.5 text-[13px] outline-none placeholder:text-muted";
const selectCls =
  "rounded-lg border border-edge bg-panel-2 px-2 py-1 text-[13px] outline-none";

// ---------------------------------------------------------------------------
// Library page
// ---------------------------------------------------------------------------

/** Library (My Models): category-filtered model table fed by
 *  quackd, footer totals + models-directory menu, and a detail
 *  panel (Info / Load / Inference) for the selected model. Load
 *  and inference edits persist to the daemon's per-model config
 *  store (debounced), which is also what the chat proxy applies. */
export default function Library({ category }: { category: LibraryCategory }) {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [offline, setOffline] = useState(false);
  const [configs, setConfigs] = useState<ModelConfigMap>({});
  const [filter, setFilter] = useState("");
  const [formatFilter, setFormatFilter] = useState("all");
  const [showWarnings, setShowWarnings] = useState(false);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [tab, setTab] = useState<DetailTab>("info");
  const [defaultCtx, setDefaultCtx] = useState(8192);
  const [modelsDir, setModelsDir] = useState("");
  const [loadForm, setLoadForm] = useState<LoadForm>(() =>
    wireToLoadForm({}, 8192),
  );
  const [infForm, setInfForm] = useState<InferenceForm>(INF_DEFAULTS);
  const [loadBusy, setLoadBusy] = useState(false);
  const [loadMsg, setLoadMsg] = useState<LoadMessage | null>(null);
  const [unloadingId, setUnloadingId] = useState<string | null>(null);
  const [menuKey, setMenuKey] = useState<string | null>(null);
  const [menuPos, setMenuPos] = useState({ x: 0, y: 0 });
  const [footerMenu, setFooterMenu] = useState(false);
  const [deleteKey, setDeleteKey] = useState<string | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [actionMsg, setActionMsg] = useState<string | null>(null);
  const [domainOpen, setDomainOpen] = useState(false);
  const [templateOpen, setTemplateOpen] = useState(false);
  const [stopDraft, setStopDraft] = useState("");
  const hydratedKey = useRef<string | null>(null);

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

  // Per-model configs from the daemon store (also migrates the
  // old localStorage stores and mirrors inference for Chat).
  useEffect(() => {
    void syncModelConfigs().then(setConfigs);
  }, []);

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
        }
      } catch {
        // Defaults are best-effort.
      }
    })();
  }, []);

  // Downloads pane / model picker "Load Model" handoff.
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

  // Fresh selection → hydrate both forms from the stored config.
  useEffect(() => {
    if (!selectedKey) return;
    const cfg = configs[selectedKey] ?? emptyModelConfig();
    setLoadForm(wireToLoadForm(cfg.load, defaultCtx));
    setInfForm(wireToInferenceForm(cfg.inference));
    hydratedKey.current = selectedKey;
    setLoadMsg(null);
    setActionMsg(null);
    // Hydration is keyed on selection (and the settings default
    // arriving); config writes from our own saves must not
    // re-hydrate mid-edit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedKey, defaultCtx]);

  // Debounced persistence of both forms to the daemon store.
  const loadWire = useMemo(
    () => loadFormToWire(loadForm, defaultCtx),
    [loadForm, defaultCtx],
  );
  useEffect(() => {
    if (!selectedKey || hydratedKey.current !== selectedKey) return;
    const current = configs[selectedKey]?.load ?? {};
    if (JSON.stringify(current) === JSON.stringify(loadWire)) return;
    const t = setTimeout(() => {
      void saveModelConfig(selectedKey, { load: loadWire })
        .then((updated) =>
          setConfigs((prev) => ({ ...prev, [selectedKey]: updated })),
        )
        .catch(() => {});
    }, 450);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadWire, selectedKey]);

  const infWire = useMemo(() => inferenceFormToWire(infForm), [infForm]);
  useEffect(() => {
    if (!selectedKey || hydratedKey.current !== selectedKey) return;
    const current = configs[selectedKey]?.inference ?? {};
    if (JSON.stringify(current) === JSON.stringify(infWire)) return;
    const t = setTimeout(() => {
      void saveModelConfig(selectedKey, { inference: infWire })
        .then((updated) =>
          setConfigs((prev) => ({ ...prev, [selectedKey]: updated })),
        )
        .catch(() => {});
    }, 450);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [infWire, selectedKey]);

  const rows = offline ? SAMPLE_ROWS : models;
  const needle = filter.trim().toLowerCase();
  const visible = rows
    .filter(
      (m) =>
        inCategory(m, category) &&
        (formatFilter === "all" || m.format === formatFilter) &&
        (!showWarnings || m.warnings.length > 0) &&
        (needle === "" ||
          `${m.publisher}/${m.display_name}`.toLowerCase().includes(needle)),
    )
    .sort((a, b) => {
      const pa = configs[a.key]?.pinned ? 0 : 1;
      const pb = configs[b.key]?.pinned ? 0 : 1;
      return pa - pb;
    });
  const selected = rows.find((m) => m.key === selectedKey) ?? null;
  const localRows = rows.filter((m) => m.format !== "maya");
  const totalBytes = localRows.reduce((sum, m) => sum + m.size_bytes, 0);
  const warningCount = rows.filter((m) => m.warnings.length > 0).length;
  const drafters = rows.filter(
    (m) =>
      m.format === "gguf" &&
      inCategory(m, "drafters") &&
      m.key !== selectedKey,
  );

  async function loadSelected() {
    if (!selected || loadBusy) return;
    setLoadBusy(true);
    setLoadMsg(null);
    try {
      const base = await apiBase();
      const body: Record<string, unknown> = {
        model: selected.key,
        ...loadWire,
      };
      // The displayed context/GPU values always ride along, even
      // when they equal the defaults (the saved wire omits those).
      const ctx = Number(loadForm.contextLength);
      if (Number.isFinite(ctx)) body.context_length = ctx;
      const gpu = Number(loadForm.gpuOffload);
      if (Number.isFinite(gpu)) body.gpu_offload = gpu;
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

  // --- More Actions menu -------------------------------------------

  async function revealModel(model: ModelEntry) {
    setMenuKey(null);
    try {
      const base = await apiBase();
      const res = await fetch(`${base}/api/v1/models/reveal`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: model.key }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setActionMsg(`Revealed ${model.key} in Finder`);
    } catch {
      setActionMsg("Reveal failed — is quackd running on the Mac?");
    }
  }

  async function togglePin(model: ModelEntry) {
    setMenuKey(null);
    const next = !(configs[model.key]?.pinned ?? false);
    try {
      const updated = await saveModelConfig(model.key, { pinned: next });
      setConfigs((prev) => ({ ...prev, [model.key]: updated }));
      setActionMsg(next ? `Pinned ${model.key}` : `Unpinned ${model.key}`);
    } catch {
      setActionMsg("Pin failed — config could not be saved");
    }
  }

  async function copyText(text: string, label: string) {
    setMenuKey(null);
    try {
      await navigator.clipboard.writeText(text);
      setActionMsg(`${label} copied`);
    } catch {
      setActionMsg("Copy failed — clipboard unavailable");
    }
  }

  async function confirmDelete() {
    if (!deleteKey || deleteBusy) return;
    setDeleteBusy(true);
    setDeleteError(null);
    try {
      const base = await apiBase();
      const res = await fetch(`${base}/api/v1/models/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: deleteKey }),
      });
      const data: unknown = await res.json().catch(() => null);
      if (!res.ok) {
        const message =
          data !== null &&
          typeof data === "object" &&
          "error" in data &&
          typeof (data as { error: unknown }).error === "string"
            ? (data as { error: string }).error
            : `delete failed (HTTP ${res.status})`;
        throw new Error(message);
      }
      forgetModelConfig(deleteKey);
      setConfigs((prev) => {
        const next = { ...prev };
        delete next[deleteKey];
        return next;
      });
      if (selectedKey === deleteKey) setSelectedKey(null);
      setActionMsg(`Deleted ${deleteKey}`);
      setDeleteKey(null);
      await fetchModels();
    } catch (e) {
      setDeleteError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleteBusy(false);
    }
  }

  async function revealModelsDir() {
    setFooterMenu(false);
    try {
      const base = await apiBase();
      const res = await fetch(`${base}/api/v1/models/reveal`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: modelsDir }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setActionMsg("Opened the models directory in Finder");
    } catch {
      setActionMsg("Open in Finder failed — is quackd running on the Mac?");
    }
  }

  const patchLoad = (patch: Partial<LoadForm>) =>
    setLoadForm((f) => ({ ...f, ...patch }));
  const patchInf = (patch: Partial<InferenceForm>) =>
    setInfForm((f) => ({ ...f, ...patch }));

  const menuItemCls =
    "flex w-full items-center gap-2 px-3 py-1.5 text-left text-[13px] transition-colors hover:bg-panel";
  const chipCls =
    "max-w-[230px] truncate rounded-md bg-panel-2 px-2 py-0.5 text-right text-[12px]";

  return (
    <div className="flex h-full">
      {/* Menu backdrop */}
      {menuKey !== null && (
        <div
          className="fixed inset-0 z-20"
          onClick={() => {
            setMenuKey(null);
            setFooterMenu(false);
          }}
        />
      )}

      {/* List column */}
      <div className="flex min-w-0 flex-1 flex-col px-6 py-4">
        {/* Header bar: My Models + format combobox + warnings */}
        <div className="flex items-center gap-3">
          <h2 className="text-[15px] font-semibold">My Models</h2>
          <select
            value={formatFilter}
            onChange={(e) => setFormatFilter(e.target.value)}
            className={selectCls}
          >
            <option value="all">All</option>
            <option value="gguf">GGUF</option>
            <option value="mlx">MLX</option>
            <option value="splash">Splash</option>
            <option value="maya">Maya</option>
          </select>
          <button
            type="button"
            onClick={() => setShowWarnings(!showWarnings)}
            className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-[12px] transition-colors ${
              showWarnings
                ? "border-accent bg-accent/15 text-ink"
                : "border-edge bg-panel text-muted hover:text-ink"
            }`}
          >
            <AlertTriangle size={13} />
            Show warnings
            {warningCount > 0 && <span>({warningCount})</span>}
          </button>
          {offline && (
            <span className="rounded-full border border-edge bg-panel px-2.5 py-0.5 text-[11px] text-muted">
              daemon offline — showing sample rows
            </span>
          )}
        </div>

        {/* Filter bar */}
        <div className="mt-3 flex items-center gap-3">
          <div className="flex w-72 items-center gap-2 rounded-lg border border-edge bg-panel px-3 py-1.5">
            <Search size={14} className="shrink-0 text-muted" />
            <input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter models… (⌘F)"
              className="w-full bg-transparent outline-none placeholder:text-muted"
            />
          </div>
          <span className="text-muted">
            {visible.length} model{visible.length === 1 ? "" : "s"}
          </span>
          {actionMsg && (
            <span className="truncate text-[12px] text-muted">{actionMsg}</span>
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
                <th className="px-4 py-2 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((m) => {
                const customized = isCustomized(configs[m.key]);
                const pinned = configs[m.key]?.pinned ?? false;
                return (
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
                      {pinned && (
                        <Pin
                          size={11}
                          className="ml-1.5 inline text-muted"
                        />
                      )}
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
                      {m.warnings.length > 0 && (
                        <span className="ml-2 rounded bg-yellow-500/15 px-1.5 py-0.5 text-[10px] text-yellow-400">
                          warning
                        </span>
                      )}
                      {m.key === selectedKey && (
                        <span className="block text-[11px] text-muted">
                          {m.key}
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
                    <td
                      className="px-4 py-2.5"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <span className="flex items-center gap-1">
                        {m.format !== "maya" && (
                          <button
                            type="button"
                            title="More Actions"
                            onClick={(e) => {
                              setFooterMenu(false);
                              setMenuKey(menuKey === m.key ? null : m.key);
                              setMenuPos({ x: e.clientX, y: e.clientY });
                            }}
                            className="flex h-7 w-7 items-center justify-center rounded-lg text-muted transition-colors hover:bg-panel-2 hover:text-ink"
                          >
                            <MoreHorizontal size={15} />
                          </button>
                        )}
                        <button
                          type="button"
                          title="Edit model default config"
                          onClick={() => {
                            setSelectedKey(m.key);
                            setTab("load");
                          }}
                          className="relative flex h-7 w-7 items-center justify-center rounded-lg text-muted transition-colors hover:bg-panel-2 hover:text-ink"
                        >
                          <Cog size={15} />
                          {customized && (
                            <span className="absolute top-0.5 right-0.5 h-1.5 w-1.5 rounded-full bg-red-500" />
                          )}
                        </button>
                      </span>
                      {menuKey === m.key && (
                        <div
                          className="fixed z-30 w-60 -translate-x-full rounded-xl border border-edge bg-panel-2 py-1 shadow-xl"
                          style={{ left: menuPos.x, top: menuPos.y }}
                        >
                          <button
                            type="button"
                            className={menuItemCls}
                            onClick={() => void revealModel(m)}
                          >
                            <FolderOpen size={14} /> Reveal in Finder
                          </button>
                          <button
                            type="button"
                            className={menuItemCls}
                            onClick={() => void togglePin(m)}
                          >
                            <Pin size={14} />{" "}
                            {pinned ? "Unpin from Top" : "Pin to Top"}
                          </button>
                          <button
                            type="button"
                            className={menuItemCls}
                            onClick={() =>
                              void copyText(m.key, "Default identifier")
                            }
                          >
                            <Copy size={14} /> Copy Default Identifier
                          </button>
                          <button
                            type="button"
                            className={menuItemCls}
                            onClick={() => void copyText(m.path, "Absolute path")}
                          >
                            <Copy size={14} /> Copy Absolute Path
                          </button>
                          <button
                            type="button"
                            className={menuItemCls}
                            onClick={() => {
                              setMenuKey(null);
                              window.open(
                                `https://huggingface.co/${m.key}`,
                                "_blank",
                              );
                            }}
                          >
                            <ExternalLink size={14} /> Open on Hugging Face 🤗
                          </button>
                          <button
                            type="button"
                            className={`${menuItemCls} text-red-400`}
                            onClick={() => {
                              setMenuKey(null);
                              setDeleteError(null);
                              setDeleteKey(m.key);
                            }}
                          >
                            <Trash2 size={14} /> Delete
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })}
              {visible.length === 0 && !offline && (
                <tr>
                  <td colSpan={8} className="px-4 py-10 text-center text-muted">
                    {showWarnings
                      ? "No models with warnings."
                      : "No models in this category. Download one from Model Search."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Footer totals + models directory */}
        <div className="flex items-center gap-3 pt-3">
          <p className="min-w-0 flex-1 text-[12px] text-muted">
            You have {localRows.length} local models, taking up{" "}
            {formatGB(totalBytes)} of disk space
          </p>
          {modelsDir && (
            <>
              <input
                readOnly
                value={modelsDir}
                title={modelsDir}
                className="w-72 truncate rounded-lg border border-edge bg-panel px-2 py-1 font-mono text-[11px] text-muted outline-none"
              />
              <span className="relative">
                <button
                  type="button"
                  title="Models directory options"
                  onClick={(e) => {
                    setMenuKey(null);
                    setFooterMenu(!footerMenu);
                    setMenuPos({ x: e.clientX, y: e.clientY });
                  }}
                  className="flex h-7 w-7 items-center justify-center rounded-lg text-muted transition-colors hover:bg-panel hover:text-ink"
                >
                  <MoreHorizontal size={15} />
                </button>
                {footerMenu && (
                  <div
                    className="fixed z-30 w-52 -translate-x-full rounded-xl border border-edge bg-panel-2 py-1 shadow-xl"
                    style={{ left: menuPos.x, top: menuPos.y - 84 }}
                  >
                    <button
                      type="button"
                      className={menuItemCls}
                      onClick={() => void revealModelsDir()}
                    >
                      <FolderOpen size={14} /> Open in Finder
                    </button>
                    <button
                      type="button"
                      className={menuItemCls}
                      onClick={() => {
                        setFooterMenu(false);
                        void fetchModels();
                      }}
                    >
                      <Search size={14} /> Rescan
                    </button>
                  </div>
                )}
              </span>
            </>
          )}
        </div>
      </div>

      {/* Detail panel */}
      {selected && (
        <aside className="flex w-[380px] shrink-0 flex-col overflow-auto border-l border-edge bg-panel px-5 py-4">
          <div className="flex items-start gap-2">
            <div className="min-w-0 flex-1">
              <p className="flex flex-wrap items-center gap-2 text-[15px] font-medium break-words">
                {selected.display_name}
                {isCustomized(configs[selected.key]) && (
                  <span className="rounded-full bg-primary/20 px-2 py-0.5 text-[10px] font-medium text-primary">
                    Customized
                  </span>
                )}
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
            <div className="mt-4">
              <p className="text-[13px] font-medium">Model Information</p>
              <dl className="mt-3 flex flex-col gap-2.5 text-[13px]">
                {(
                  [
                    ["Model", selected.key],
                    ["File", selected.file_name ?? "—"],
                    ["Format", selected.format.toUpperCase()],
                    ...(selected.quantization
                      ? [["Quantization", selected.quantization]]
                      : []),
                    ...(selected.arch ? [["Arch", selected.arch]] : []),
                    ["Domain", selected.domain],
                    [
                      "Size on disk",
                      selected.format === "maya"
                        ? "—"
                        : formatGB(selected.size_bytes),
                    ],
                  ] as [string, string][]
                ).map(([label, value]) => (
                  <div key={label} className="flex items-center gap-3">
                    <dt className="flex-1 text-muted">{label}</dt>
                    <dd className={chipCls} title={value}>
                      {value}
                    </dd>
                  </div>
                ))}
                {selected.vision && (
                  <div className="flex items-center gap-3">
                    <dt className="flex-1 text-muted">Capabilities</dt>
                    <dd className="flex items-center gap-1 rounded-md bg-panel-2 px-2 py-0.5 text-[12px]">
                      <Eye size={12} /> Vision
                    </dd>
                  </div>
                )}
                {selected.remote_url && (
                  <div className="flex items-center gap-3">
                    <dt className="flex-1 text-muted">Endpoint</dt>
                    <dd className={chipCls} title={selected.remote_url}>
                      {selected.remote_url}
                    </dd>
                  </div>
                )}
              </dl>
              {selected.warnings.length > 0 && (
                <div className="mt-4 flex flex-col gap-1.5">
                  {selected.warnings.map((w) => (
                    <p
                      key={w}
                      className="flex items-start gap-1.5 rounded-lg bg-yellow-500/10 px-2.5 py-2 text-[12px] text-yellow-400"
                    >
                      <AlertTriangle size={13} className="mt-0.5 shrink-0" />
                      {w}
                    </p>
                  ))}
                </div>
              )}
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

              {selected.format !== "gguf" && (
                <p className="rounded-xl border border-edge bg-panel-2 px-3.5 py-3 text-[12px] text-muted">
                  {selected.format === "splash"
                    ? "Splash: these settings are stored per model but not applied — the Splash engine sizes context and memory itself (auto)."
                    : "MLX: these settings are stored per model but not applied — mlx_vlm.server has no equivalents."}
                </p>
              )}

              <Section
                title="Context and Offload"
                onReset={() =>
                  patchLoad({
                    contextLength: String(defaultCtx),
                    gpuOffload: "99",
                  })
                }
              >
                <div>
                  <Field
                    label="Context Length"
                    onReset={
                      loadForm.contextLength !== String(defaultCtx)
                        ? () =>
                            patchLoad({ contextLength: String(defaultCtx) })
                        : undefined
                    }
                  >
                    <input
                      type="number"
                      min={1024}
                      step={1024}
                      value={loadForm.contextLength}
                      onChange={(e) =>
                        patchLoad({ contextLength: e.target.value })
                      }
                      className={numInputCls}
                    />
                  </Field>
                  <input
                    type="range"
                    min={1024}
                    max={selected.max_context ?? 262144}
                    step={1024}
                    value={
                      Number(loadForm.contextLength) || defaultCtx
                    }
                    onChange={(e) =>
                      patchLoad({ contextLength: e.target.value })
                    }
                    className="mt-2 w-full accent-accent"
                  />
                  {selected.max_context && (
                    <p className="mt-1 text-[11px] text-muted">
                      Model supports up to{" "}
                      {selected.max_context.toLocaleString()} tokens
                    </p>
                  )}
                </div>
                <div>
                  <Field
                    label="GPU Offload"
                    onReset={
                      loadForm.gpuOffload !== "99"
                        ? () => patchLoad({ gpuOffload: "99" })
                        : undefined
                    }
                  >
                    <input
                      type="number"
                      min={0}
                      max={99}
                      value={loadForm.gpuOffload}
                      onChange={(e) =>
                        patchLoad({ gpuOffload: e.target.value })
                      }
                      className={numInputCls}
                    />
                  </Field>
                  <input
                    type="range"
                    min={0}
                    max={99}
                    step={1}
                    value={Number(loadForm.gpuOffload) || 0}
                    onChange={(e) =>
                      patchLoad({ gpuOffload: e.target.value })
                    }
                    className="mt-2 w-full accent-accent"
                  />
                  <p className="mt-1 text-[11px] text-muted">
                    Layers offloaded to the GPU (99 = all).
                  </p>
                </div>
              </Section>

              <Section
                title="Advanced"
                onReset={() =>
                  patchLoad({
                    cpuThreads: "",
                    evalBatch: "",
                    physBatch: "",
                    maxConcurrent: "",
                    unifiedKvCache: null,
                    contextCheckpoints: "",
                    reasoningBudgetMessage: "",
                    ropeBaseOn: false,
                    ropeBase: "",
                    ropeScaleOn: false,
                    ropeScale: "",
                    kvOffload: null,
                    keepInMemory: null,
                    tryMmap: null,
                    onDemand: "auto",
                    seedOn: false,
                    seed: "",
                    numExperts: "",
                    numLayersOffload: "",
                    moeExpert: "",
                    speculative: "off",
                    chatTemplate: "",
                    flashAttention: null,
                    cacheK: "",
                    cacheV: "",
                  })
                }
              >
                <div>
                  <Field
                    label="CPU Thread Pool Size"
                    onReset={
                      loadForm.cpuThreads !== ""
                        ? () => patchLoad({ cpuThreads: "" })
                        : undefined
                    }
                  >
                    <input
                      type="number"
                      min={0}
                      placeholder="default"
                      value={loadForm.cpuThreads}
                      onChange={(e) =>
                        patchLoad({ cpuThreads: e.target.value })
                      }
                      className={numInputCls}
                    />
                  </Field>
                  <input
                    type="range"
                    min={0}
                    max={64}
                    step={1}
                    value={Number(loadForm.cpuThreads) || 0}
                    onChange={(e) =>
                      patchLoad({
                        cpuThreads:
                          e.target.value === "0" ? "" : e.target.value,
                      })
                    }
                    className="mt-2 w-full accent-accent"
                  />
                </div>
                <Field
                  label="Evaluation Batch Size"
                  onReset={
                    loadForm.evalBatch !== ""
                      ? () => patchLoad({ evalBatch: "" })
                      : undefined
                  }
                >
                  <input
                    type="number"
                    min={1}
                    placeholder="default"
                    value={loadForm.evalBatch}
                    onChange={(e) => patchLoad({ evalBatch: e.target.value })}
                    className={numInputCls}
                  />
                </Field>
                <Field
                  label="Physical Batch Size"
                  onReset={
                    loadForm.physBatch !== ""
                      ? () => patchLoad({ physBatch: "" })
                      : undefined
                  }
                >
                  <input
                    type="number"
                    min={1}
                    placeholder="default"
                    value={loadForm.physBatch}
                    onChange={(e) => patchLoad({ physBatch: e.target.value })}
                    className={numInputCls}
                  />
                </Field>
                <Field
                  label="Max Concurrent Predictions"
                  experimental
                  onReset={
                    loadForm.maxConcurrent !== ""
                      ? () => patchLoad({ maxConcurrent: "" })
                      : undefined
                  }
                >
                  <input
                    type="number"
                    min={1}
                    placeholder="default"
                    value={loadForm.maxConcurrent}
                    onChange={(e) =>
                      patchLoad({ maxConcurrent: e.target.value })
                    }
                    className={numInputCls}
                  />
                </Field>
                <Field
                  label="Unified KV Cache"
                  experimental
                  onReset={
                    loadForm.unifiedKvCache !== null
                      ? () => patchLoad({ unifiedKvCache: null })
                      : undefined
                  }
                >
                  <Toggle
                    on={loadForm.unifiedKvCache ?? true}
                    onClick={() =>
                      patchLoad({
                        unifiedKvCache: !(loadForm.unifiedKvCache ?? true),
                      })
                    }
                  />
                </Field>
                <Field
                  label="Context Checkpoints"
                  onReset={
                    loadForm.contextCheckpoints !== ""
                      ? () => patchLoad({ contextCheckpoints: "" })
                      : undefined
                  }
                >
                  <input
                    type="number"
                    min={0}
                    placeholder="default"
                    value={loadForm.contextCheckpoints}
                    onChange={(e) =>
                      patchLoad({ contextCheckpoints: e.target.value })
                    }
                    className={numInputCls}
                  />
                </Field>
                <div>
                  <Field
                    label="Reasoning Budget Message"
                    onReset={
                      loadForm.reasoningBudgetMessage !== ""
                        ? () => patchLoad({ reasoningBudgetMessage: "" })
                        : undefined
                    }
                  >
                    <span />
                  </Field>
                  <input
                    type="text"
                    placeholder="Enter text here"
                    value={loadForm.reasoningBudgetMessage}
                    onChange={(e) =>
                      patchLoad({ reasoningBudgetMessage: e.target.value })
                    }
                    className={`mt-1.5 ${textInputCls}`}
                  />
                </div>
                <Field
                  label="RoPE Frequency Base"
                  onReset={
                    loadForm.ropeBaseOn
                      ? () => patchLoad({ ropeBaseOn: false, ropeBase: "" })
                      : undefined
                  }
                >
                  <span className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={loadForm.ropeBaseOn}
                      onChange={(e) =>
                        patchLoad({ ropeBaseOn: e.target.checked })
                      }
                      className="accent-accent"
                    />
                    {loadForm.ropeBaseOn ? (
                      <input
                        type="number"
                        step="any"
                        value={loadForm.ropeBase}
                        onChange={(e) =>
                          patchLoad({ ropeBase: e.target.value })
                        }
                        className="w-20 rounded-lg border border-edge bg-panel-2 px-2 py-1 text-right text-[13px] outline-none"
                      />
                    ) : (
                      <span className="text-[13px] text-muted">Auto</span>
                    )}
                  </span>
                </Field>
                <Field
                  label="RoPE Frequency Scale"
                  onReset={
                    loadForm.ropeScaleOn
                      ? () =>
                          patchLoad({ ropeScaleOn: false, ropeScale: "" })
                      : undefined
                  }
                >
                  <span className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={loadForm.ropeScaleOn}
                      onChange={(e) =>
                        patchLoad({ ropeScaleOn: e.target.checked })
                      }
                      className="accent-accent"
                    />
                    {loadForm.ropeScaleOn ? (
                      <input
                        type="number"
                        step="any"
                        value={loadForm.ropeScale}
                        onChange={(e) =>
                          patchLoad({ ropeScale: e.target.value })
                        }
                        className="w-20 rounded-lg border border-edge bg-panel-2 px-2 py-1 text-right text-[13px] outline-none"
                      />
                    ) : (
                      <span className="text-[13px] text-muted">Auto</span>
                    )}
                  </span>
                </Field>
                <Field
                  label="Offload KV Cache to GPU Memory"
                  onReset={
                    loadForm.kvOffload !== null
                      ? () => patchLoad({ kvOffload: null })
                      : undefined
                  }
                >
                  <Toggle
                    on={loadForm.kvOffload ?? true}
                    onClick={() =>
                      patchLoad({ kvOffload: !(loadForm.kvOffload ?? true) })
                    }
                  />
                </Field>
                <Field
                  label="Keep Model in Memory"
                  onReset={
                    loadForm.keepInMemory !== null
                      ? () => patchLoad({ keepInMemory: null })
                      : undefined
                  }
                >
                  <Toggle
                    on={loadForm.keepInMemory ?? true}
                    onClick={() =>
                      patchLoad({
                        keepInMemory: !(loadForm.keepInMemory ?? true),
                      })
                    }
                  />
                </Field>
                <Field
                  label="Try mmap()"
                  onReset={
                    loadForm.tryMmap !== null
                      ? () => patchLoad({ tryMmap: null })
                      : undefined
                  }
                >
                  <Toggle
                    on={loadForm.tryMmap ?? true}
                    onClick={() =>
                      patchLoad({ tryMmap: !(loadForm.tryMmap ?? true) })
                    }
                  />
                </Field>
                <Field
                  label="On-Demand Tensor Loading"
                  onReset={
                    loadForm.onDemand !== "auto"
                      ? () => patchLoad({ onDemand: "auto" })
                      : undefined
                  }
                >
                  <select
                    value={loadForm.onDemand}
                    onChange={(e) => patchLoad({ onDemand: e.target.value })}
                    className={selectCls}
                  >
                    <option value="auto">Auto</option>
                    <option value="on">On</option>
                    <option value="off">Off</option>
                  </select>
                </Field>
                <Field
                  label="Seed"
                  onReset={
                    loadForm.seedOn
                      ? () => patchLoad({ seedOn: false, seed: "" })
                      : undefined
                  }
                >
                  <span className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={loadForm.seedOn}
                      onChange={(e) =>
                        patchLoad({ seedOn: e.target.checked })
                      }
                      className="accent-accent"
                    />
                    {loadForm.seedOn ? (
                      <input
                        type="number"
                        value={loadForm.seed}
                        onChange={(e) => patchLoad({ seed: e.target.value })}
                        className="w-20 rounded-lg border border-edge bg-panel-2 px-2 py-1 text-right text-[13px] outline-none"
                      />
                    ) : (
                      <span className="text-[13px] text-muted">
                        Random Seed
                      </span>
                    )}
                  </span>
                </Field>
                <div>
                  <Field
                    label="Number of Experts"
                    note="Stored — not applied: llama.cpp's --n-cpu-moe counts layers, not experts."
                    onReset={
                      loadForm.numExperts !== ""
                        ? () => patchLoad({ numExperts: "" })
                        : undefined
                    }
                  >
                    <input
                      type="number"
                      min={0}
                      placeholder="default"
                      value={loadForm.numExperts}
                      onChange={(e) =>
                        patchLoad({ numExperts: e.target.value })
                      }
                      className={numInputCls}
                    />
                  </Field>
                  <input
                    type="range"
                    min={0}
                    max={64}
                    step={1}
                    value={Number(loadForm.numExperts) || 0}
                    onChange={(e) =>
                      patchLoad({
                        numExperts:
                          e.target.value === "0" ? "" : e.target.value,
                      })
                    }
                    className="mt-2 w-full accent-accent"
                  />
                </div>
                <div>
                  <Field
                    label="Number of Layers to Offload"
                    experimental
                    note="Stored — not applied: GPU layer offload is the GPU Offload field above."
                    onReset={
                      loadForm.numLayersOffload !== ""
                        ? () => patchLoad({ numLayersOffload: "" })
                        : undefined
                    }
                  >
                    <input
                      type="number"
                      min={0}
                      placeholder="default"
                      value={loadForm.numLayersOffload}
                      onChange={(e) =>
                        patchLoad({ numLayersOffload: e.target.value })
                      }
                      className={numInputCls}
                    />
                  </Field>
                  <input
                    type="range"
                    min={0}
                    max={128}
                    step={1}
                    value={Number(loadForm.numLayersOffload) || 0}
                    onChange={(e) =>
                      patchLoad({
                        numLayersOffload:
                          e.target.value === "0" ? "" : e.target.value,
                      })
                    }
                    className="mt-2 w-full accent-accent"
                  />
                </div>
                <Field
                  label="MoE Expert"
                  experimental
                  note="Stored — not applied: no llama.cpp equivalent."
                  onReset={
                    loadForm.moeExpert !== ""
                      ? () => patchLoad({ moeExpert: "" })
                      : undefined
                  }
                >
                  <input
                    type="number"
                    min={0}
                    placeholder="default"
                    value={loadForm.moeExpert}
                    onChange={(e) => patchLoad({ moeExpert: e.target.value })}
                    className={numInputCls}
                  />
                </Field>
                <Field
                  label="Speculative Decoding"
                  onReset={
                    loadForm.speculative !== "off"
                      ? () => patchLoad({ speculative: "off" })
                      : undefined
                  }
                >
                  <select
                    value={loadForm.speculative}
                    onChange={(e) =>
                      patchLoad({ speculative: e.target.value })
                    }
                    className={selectCls}
                  >
                    <option value="off">Off</option>
                    {drafters.map((d) => (
                      <option key={d.key} value={d.key}>
                        {d.display_name}
                      </option>
                    ))}
                  </select>
                </Field>
                <div>
                  <button
                    type="button"
                    onClick={() => setTemplateOpen(!templateOpen)}
                    className="flex w-full items-center gap-2 text-left"
                  >
                    <span className="flex-1 text-[13px] text-muted">
                      Chat Template
                    </span>
                    {loadForm.chatTemplate !== "" && (
                      <span
                        role="button"
                        tabIndex={0}
                        title="Reset Chat Template to default"
                        onClick={(e) => {
                          e.stopPropagation();
                          patchLoad({ chatTemplate: "" });
                        }}
                        className="text-muted transition-colors hover:text-red-400"
                      >
                        <Trash2 size={13} />
                      </span>
                    )}
                    {templateOpen ? (
                      <ChevronDown size={14} />
                    ) : (
                      <ChevronRight size={14} />
                    )}
                  </button>
                  {templateOpen && (
                    <textarea
                      rows={5}
                      value={loadForm.chatTemplate}
                      onChange={(e) =>
                        patchLoad({ chatTemplate: e.target.value })
                      }
                      placeholder="Custom Jinja chat template (empty = the model's own)"
                      className={`mt-1.5 font-mono text-[12px] ${textInputCls}`}
                    />
                  )}
                </div>
                <Field
                  label="Flash Attention"
                  onReset={
                    loadForm.flashAttention !== null
                      ? () => patchLoad({ flashAttention: null })
                      : undefined
                  }
                >
                  <Toggle
                    on={loadForm.flashAttention ?? false}
                    onClick={() =>
                      patchLoad({
                        flashAttention: !(loadForm.flashAttention ?? false),
                      })
                    }
                  />
                </Field>
                <Field
                  label="K Cache Quantization"
                  experimental
                  onReset={
                    loadForm.cacheK !== ""
                      ? () => patchLoad({ cacheK: "" })
                      : undefined
                  }
                >
                  <select
                    value={loadForm.cacheK}
                    onChange={(e) => patchLoad({ cacheK: e.target.value })}
                    className={selectCls}
                  >
                    <option value="">Default</option>
                    <option value="f16">F16</option>
                    <option value="bf16">BF16</option>
                    <option value="q8_0">Q8_0</option>
                    <option value="q4_0">Q4_0</option>
                  </select>
                </Field>
                <Field
                  label="V Cache Quantization"
                  experimental
                  onReset={
                    loadForm.cacheV !== ""
                      ? () => patchLoad({ cacheV: "" })
                      : undefined
                  }
                >
                  <select
                    value={loadForm.cacheV}
                    onChange={(e) => patchLoad({ cacheV: e.target.value })}
                    className={selectCls}
                  >
                    <option value="">Default</option>
                    <option value="f16">F16</option>
                    <option value="bf16">BF16</option>
                    <option value="q8_0">Q8_0</option>
                    <option value="q4_0">Q4_0</option>
                  </select>
                </Field>
                <p className="text-[11px] leading-relaxed text-muted">
                  KV Cache Quantization is an experimental feature that
                  may cause issues with some models. Flash Attention
                  must be enabled for V cache quantization to take
                  effect.
                </p>
              </Section>

              <Section title="llama.cpp Arguments Override">
                <Field
                  label="Enable override"
                  onReset={
                    loadForm.argsOverrideOn
                      ? () =>
                          patchLoad({ argsOverrideOn: false, argsOverride: "" })
                      : undefined
                  }
                >
                  <Toggle
                    on={loadForm.argsOverrideOn}
                    onClick={() =>
                      patchLoad({ argsOverrideOn: !loadForm.argsOverrideOn })
                    }
                  />
                </Field>
                {loadForm.argsOverrideOn && (
                  <input
                    type="text"
                    value={loadForm.argsOverride}
                    onChange={(e) =>
                      patchLoad({ argsOverride: e.target.value })
                    }
                    placeholder="e.g. --cache-ram 0 --prio 2"
                    className={`font-mono text-[12px] ${textInputCls}`}
                  />
                )}
                <p className="text-[11px] text-muted">
                  Extra arguments appended verbatim to the llama-server
                  command line when this model loads (GGUF only).
                </p>
              </Section>

              <button
                type="button"
                disabled={loadBusy}
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

          {tab === "inference" && (
            <div className="mt-4 flex flex-col gap-5">
              <Section title="System Prompt">
                <textarea
                  rows={6}
                  value={infForm.systemPrompt}
                  onChange={(e) =>
                    patchInf({ systemPrompt: e.target.value })
                  }
                  placeholder='Example, "Only answer in rhymes"'
                  className={`${textInputCls} resize-none`}
                />
                <p className="text-right text-[11px] text-muted">
                  Token count: N/A
                </p>
              </Section>

              <Section title="Reasoning">
                <Field label="Reasoning Budget">
                  <span className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={infForm.reasoningBudgetOn}
                      onChange={(e) =>
                        patchInf({ reasoningBudgetOn: e.target.checked })
                      }
                      className="accent-accent"
                    />
                    {infForm.reasoningBudgetOn ? (
                      <input
                        type="number"
                        min={0}
                        placeholder="tokens"
                        value={infForm.reasoningBudget}
                        onChange={(e) =>
                          patchInf({ reasoningBudget: e.target.value })
                        }
                        className="w-20 rounded-lg border border-edge bg-panel-2 px-2 py-1 text-right text-[13px] outline-none"
                      />
                    ) : (
                      <span className="text-[13px] text-muted">
                        Unrestricted
                      </span>
                    )}
                  </span>
                </Field>
                <p className="text-[11px] text-muted">
                  Stored per model; the engines we drive take no
                  per-request reasoning budget, so this is recorded
                  for the model's defaults only.
                </p>
              </Section>

              <Section title="Settings">
                <div>
                  <Field label="Temperature">
                    <input
                      type="number"
                      min={0}
                      max={2}
                      step={0.05}
                      value={infForm.temperature}
                      onChange={(e) =>
                        patchInf({ temperature: Number(e.target.value) || 0 })
                      }
                      className={numInputCls}
                    />
                  </Field>
                  <input
                    type="range"
                    min={0}
                    max={2}
                    step={0.05}
                    value={infForm.temperature}
                    onChange={(e) =>
                      patchInf({ temperature: Number(e.target.value) })
                    }
                    className="mt-2 w-full accent-accent"
                  />
                </div>
                <Field label="Limit Response Length">
                  <span className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={infForm.limitResponse}
                      onChange={(e) =>
                        patchInf({ limitResponse: e.target.checked })
                      }
                      className="accent-accent"
                    />
                    {infForm.limitResponse && (
                      <input
                        type="number"
                        min={1}
                        placeholder="max tokens"
                        value={infForm.maxResponseTokens}
                        onChange={(e) =>
                          patchInf({ maxResponseTokens: e.target.value })
                        }
                        className="w-24 rounded-lg border border-edge bg-panel-2 px-2 py-1 text-right text-[13px] outline-none"
                      />
                    )}
                  </span>
                </Field>
                <Field label="Context Overflow">
                  <select
                    value={infForm.contextOverflow}
                    onChange={(e) =>
                      patchInf({ contextOverflow: e.target.value })
                    }
                    className={selectCls}
                  >
                    <option value="truncate_middle">Truncate Middle</option>
                    <option value="stop">Stop</option>
                  </select>
                </Field>
                <div>
                  <Field label="Stop Strings">
                    <span />
                  </Field>
                  <input
                    type="text"
                    value={stopDraft}
                    onChange={(e) => setStopDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && stopDraft.trim()) {
                        e.preventDefault();
                        patchInf({
                          stopStrings: [
                            ...infForm.stopStrings,
                            stopDraft.trim(),
                          ],
                        });
                        setStopDraft("");
                      }
                    }}
                    placeholder="Enter a string and press ↵"
                    className={`mt-1.5 ${textInputCls}`}
                  />
                  {infForm.stopStrings.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {infForm.stopStrings.map((s) => (
                        <span
                          key={s}
                          className="flex items-center gap-1 rounded-md bg-panel-2 px-2 py-0.5 text-[12px]"
                        >
                          {s}
                          <button
                            type="button"
                            title="Remove"
                            onClick={() =>
                              patchInf({
                                stopStrings: infForm.stopStrings.filter(
                                  (x) => x !== s,
                                ),
                              })
                            }
                            className="text-muted hover:text-red-400"
                          >
                            <X size={11} />
                          </button>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                <div>
                  <p className="text-[13px] text-muted">CPU Threads</p>
                  <p className="mt-1 text-[12px] leading-relaxed text-muted">
                    ⓘ CPU thread count for llama.cpp engines is
                    configured in Load settings. Go to Load → Advanced
                    Load Params → CPU Thread Pool Size.
                  </p>
                  <button
                    type="button"
                    onClick={() => setTab("load")}
                    className="mt-2 rounded-lg border border-edge px-2.5 py-1 text-[12px] transition-colors hover:bg-panel-2"
                  >
                    Go to setting →
                  </button>
                </div>
              </Section>

              <Section title="Reasoning Parsing">
                <p className="text-[12px] leading-relaxed text-muted">
                  ⓘ Reasoning parsing is handled by the engine's own
                  thinking controls for llama.cpp engines; there is no
                  separate parser configuration. See Inference →
                  Reasoning above.
                </p>
              </Section>

              <Section title="Sampling">
                <Field label="Top K Sampling">
                  <input
                    type="number"
                    min={0}
                    value={infForm.topK}
                    onChange={(e) =>
                      patchInf({ topK: Number(e.target.value) || 0 })
                    }
                    className={numInputCls}
                  />
                </Field>
                <Field label="Repeat Penalty">
                  <span className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={infForm.repeatOn}
                      onChange={(e) =>
                        patchInf({ repeatOn: e.target.checked })
                      }
                      className="accent-accent"
                    />
                    <input
                      type="number"
                      step={0.05}
                      value={infForm.repeatPenalty}
                      onChange={(e) =>
                        patchInf({
                          repeatPenalty: Number(e.target.value) || 0,
                        })
                      }
                      className="w-20 rounded-lg border border-edge bg-panel-2 px-2 py-1 text-right text-[13px] outline-none"
                    />
                  </span>
                </Field>
                <Field label="Presence Penalty">
                  <span className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={infForm.presenceOn}
                      onChange={(e) =>
                        patchInf({ presenceOn: e.target.checked })
                      }
                      className="accent-accent"
                    />
                    <input
                      type="number"
                      step={0.1}
                      value={infForm.presencePenalty}
                      onChange={(e) =>
                        patchInf({
                          presencePenalty: Number(e.target.value) || 0,
                        })
                      }
                      className="w-20 rounded-lg border border-edge bg-panel-2 px-2 py-1 text-right text-[13px] outline-none"
                    />
                  </span>
                </Field>
                <div>
                  <Field label="Top P Sampling">
                    <span className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={infForm.topPOn}
                        onChange={(e) =>
                          patchInf({ topPOn: e.target.checked })
                        }
                        className="accent-accent"
                      />
                      <input
                        type="number"
                        min={0}
                        max={1}
                        step={0.05}
                        value={infForm.topP}
                        onChange={(e) =>
                          patchInf({ topP: Number(e.target.value) || 0 })
                        }
                        className="w-20 rounded-lg border border-edge bg-panel-2 px-2 py-1 text-right text-[13px] outline-none"
                      />
                    </span>
                  </Field>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={infForm.topP}
                    onChange={(e) =>
                      patchInf({ topP: Number(e.target.value) })
                    }
                    className="mt-2 w-full accent-accent"
                  />
                </div>
                <div>
                  <Field label="Min P Sampling">
                    <span className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={infForm.minPOn}
                        onChange={(e) =>
                          patchInf({ minPOn: e.target.checked })
                        }
                        className="accent-accent"
                      />
                      <input
                        type="number"
                        min={0}
                        max={1}
                        step={0.01}
                        value={infForm.minP}
                        onChange={(e) =>
                          patchInf({ minP: Number(e.target.value) || 0 })
                        }
                        className="w-20 rounded-lg border border-edge bg-panel-2 px-2 py-1 text-right text-[13px] outline-none"
                      />
                    </span>
                  </Field>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={infForm.minP}
                    onChange={(e) =>
                      patchInf({ minP: Number(e.target.value) })
                    }
                    className="mt-2 w-full accent-accent"
                  />
                </div>
              </Section>

              <Section title="Structured Output">
                <Field label="Structured Output">
                  <Toggle
                    on={infForm.structuredOutput}
                    onClick={() =>
                      patchInf({ structuredOutput: !infForm.structuredOutput })
                    }
                  />
                </Field>
                <p className="text-[11px] text-muted">
                  When on, chat requests for this model that don't
                  specify a response format are sent with
                  response_format: json_object.
                </p>
              </Section>

              <Section title="Speculative Decoding">
                <p className="text-[12px] leading-relaxed text-muted">
                  ⓘ Speculative decoding is configured in Load
                  settings for llama.cpp engines. Go to Load →
                  Advanced Load Params → Speculative Decoding.
                </p>
                <p>
                  <button
                    type="button"
                    onClick={() => setTab("load")}
                    className="rounded-lg border border-edge px-2.5 py-1 text-[12px] transition-colors hover:bg-panel-2"
                  >
                    Go to setting →
                  </button>
                </p>
              </Section>

              <Section title="Prompt Template">
                <p className="text-[12px] leading-relaxed text-muted">
                  ⓘ Chat template moved to Load settings for
                  llama.cpp engines. Go to Load → Advanced Load
                  Params → Chat Template.
                </p>
                <p>
                  <button
                    type="button"
                    onClick={() => setTab("load")}
                    className="rounded-lg border border-edge px-2.5 py-1 text-[12px] transition-colors hover:bg-panel-2"
                  >
                    Go to setting →
                  </button>
                </p>
              </Section>

              <Section title="Additional Stop Strings">
                <p className="text-[12px] leading-relaxed text-muted">
                  ⓘ Additional stop strings from prompt templates use
                  the regular Stop Strings setting. Use Inference →
                  Settings → Stop Strings above.
                </p>
              </Section>
            </div>
          )}

          {/* Domain Control strip (panel bottom, all tabs) */}
          <div className="mt-auto pt-4">
            <div className="border-t border-edge pt-2">
              <button
                type="button"
                onClick={() => setDomainOpen(!domainOpen)}
                className="flex w-full items-center gap-1.5 text-left text-[13px] text-muted transition-colors hover:text-ink"
              >
                {domainOpen ? (
                  <ChevronDown size={14} />
                ) : (
                  <ChevronRight size={14} />
                )}
                Domain Control
              </button>
              {domainOpen && (
                <dl className="mt-2.5 flex flex-col gap-2 text-[13px]">
                  <div className="flex items-center gap-3">
                    <dt className="flex-1 text-muted">Domain</dt>
                    <dd className={chipCls}>{selected.domain}</dd>
                  </div>
                  <div className="flex items-center gap-3">
                    <dt className="flex-1 text-muted">Load state</dt>
                    <dd className={chipCls}>
                      {selected.format === "maya"
                        ? "Served remotely by Maya"
                        : selected.loaded_instances.length > 0
                          ? `Loaded (${selected.loaded_instances
                              .map((i) => i.backend)
                              .join(", ")})`
                          : "Not loaded"}
                    </dd>
                  </div>
                </dl>
              )}
            </div>
          </div>
        </aside>
      )}

      {/* Delete confirmation */}
      {deleteKey && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60">
          <div className="w-[380px] rounded-2xl border border-edge bg-panel p-5 shadow-xl">
            <p className="text-[15px] font-medium">Delete this model?</p>
            <p className="mt-2 text-[13px] leading-relaxed text-muted">
              {deleteKey} will be removed from disk (its folder under
              the models directory), along with its saved QuackStudio
              config. This cannot be undone.
            </p>
            {deleteError && (
              <p className="mt-3 text-[13px] text-red-400">{deleteError}</p>
            )}
            <div className="mt-5 flex gap-2">
              <button
                type="button"
                disabled={deleteBusy}
                onClick={() => setDeleteKey(null)}
                className="flex-1 rounded-lg border border-edge px-3 py-2 text-[13px] transition-colors hover:bg-panel-2 disabled:opacity-40"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={deleteBusy}
                onClick={() => void confirmDelete()}
                className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-red-600 px-3 py-2 text-[13px] font-medium text-white transition-opacity disabled:opacity-40"
              >
                {deleteBusy && (
                  <Loader2 size={14} className="animate-spin" />
                )}
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
