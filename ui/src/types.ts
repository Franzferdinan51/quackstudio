/** A loaded model instance (quack-core's Instance). */
export interface LoadedInstance {
  instance_id: string;
  model_key: string;
  backend: string;
  port: number;
  pid: number;
}

/** Shape served by quackd at GET /api/v1/models (quack-core's ModelEntry + loaded_instances). */
export interface ModelEntry {
  key: string;
  publisher: string;
  display_name: string;
  format: "gguf" | "mlx" | "splash" | "maya";
  size_bytes: number;
  path: string;
  arch: string | null;
  vision: boolean;
  /** Name of the main payload file (main GGUF, largest safetensors…). */
  file_name: string | null;
  /** Quantization label when the scanner could derive one. */
  quantization: string | null;
  /** Context ceiling from the model's own metadata. */
  max_context: number | null;
  /** Serving domain: "llm" | "embedding". */
  domain: string;
  /** Scanner flags (incomplete download, missing payload, …). */
  warnings: string[];
  /** Base URL of the serving provider for remote (Maya) models; null locally. */
  remote_url: string | null;
  loaded_instances: LoadedInstance[];
}

/** Per-model load config — the daemon's `model-configs.json`
 *  `load` section and the load request's flattened fields. Every
 *  field optional; absent = engine default / not customized.
 *  Flag mappings live in quackd (verified vs llama-server
 *  2.80.1); num_experts / num_layers_offload / moe_expert are
 *  stored-only. */
export interface LoadConfigWire {
  context_length?: number;
  gpu_offload?: number;
  cpu_threads?: number;
  eval_batch_size?: number;
  physical_batch_size?: number;
  max_concurrent?: number;
  unified_kv_cache?: boolean;
  context_checkpoints?: number;
  reasoning_budget_message?: string;
  rope_freq_base?: number;
  rope_freq_scale?: number;
  kv_offload?: boolean;
  keep_in_memory?: boolean;
  try_mmap?: boolean;
  on_demand_tensor_loading?: string;
  seed?: number;
  num_experts?: number;
  num_layers_offload?: number;
  moe_expert?: number;
  speculative_decoding?: string;
  chat_template?: string;
  flash_attention?: boolean;
  cache_type_k?: string;
  cache_type_v?: string;
  args_override?: string;
}

/** Per-model inference defaults — the daemon's `inference`
 *  section. The chat proxy fills request fields the client left
 *  unset from these. */
export interface InferenceConfigWire {
  system_prompt?: string;
  reasoning_budget_enabled?: boolean;
  reasoning_budget?: number;
  temperature?: number;
  limit_response_length?: boolean;
  max_response_tokens?: number;
  context_overflow?: string;
  stop_strings?: string[];
  top_k?: number;
  repeat_penalty_enabled?: boolean;
  repeat_penalty?: number;
  presence_penalty_enabled?: boolean;
  presence_penalty?: number;
  top_p_enabled?: boolean;
  top_p?: number;
  min_p_enabled?: boolean;
  min_p?: number;
  structured_output?: boolean;
}

/** One model's persisted config (GET/POST /api/v1/model-configs). */
export interface ModelConfigWire {
  load: LoadConfigWire;
  inference: InferenceConfigWire;
  pinned: boolean;
}

/** A Maya provider entry (quack-core's MayaProvider). */
export interface MayaProvider {
  name: string;
  base_url: string;
}

/** Server behaviour knobs (settings.json `server`). Only `jit`
 *  is enforced by quackd today; the rest are stored preferences. */
export interface ServerSettings {
  require_auth: boolean;
  serve_lan: boolean;
  cors: boolean;
  jit: boolean;
  jit_ttl_minutes: number;
  jit_only_last: boolean;
}

/** settings.json `general`. */
export interface GeneralSettings {
  hf_proxy: boolean;
}

/** settings.json `defaults`. `context_length` seeds the Library
 *  Load tab; `image_max_px` is stored only. */
export interface ModelDefaults {
  context_length: number | null;
  image_max_px: number | null;
}

/** GET /api/v1/settings. POST returns the same shape, plus an
 *  optional `note` (e.g. when the env var overrides the file). */
export interface SettingsView {
  version: string;
  port: number;
  models_dir: string;
  backends_dir: string;
  maya_providers: MayaProvider[];
  maya_providers_source: "env" | "settings" | "default";
  server: ServerSettings;
  general: GeneralSettings;
  defaults: ModelDefaults;
  note?: string;
}

/** One loaded instance as reported in GET /api/v1/status. */
export interface StatusInstance {
  instance_id: string;
  model_key: string;
  backend: string;
  port: number;
  pid: number;
  context_length: number | null;
  size_bytes: number;
}

/** GET /api/v1/status. */
export interface StatusView {
  version: string;
  port: number;
  models_dir: string;
  backends_dir: string;
  model_count: number;
  loaded_instances: number;
  instances: StatusInstance[];
  maya_providers: number;
  maya_providers_online: number;
  host: {
    name: string;
    chip: string;
    gpu: string;
    total_ram_bytes: number;
    /** Sum of loaded models' on-disk sizes — the loaded-weights
     *  proxy shown in the sidebar RAM pill. */
    loaded_bytes: number;
    /** quackd's own peak RSS (getrusage) — the second pill. */
    daemon_rss_bytes: number;
    cpu_percent: number | null;
  };
  packs: { llama: string; mlx: string; splash: string };
}

/** One Hugging Face search result (GET /api/v1/discover/search). */
export interface DiscoverModel {
  id: string;
  downloads: number | null;
  likes: number | null;
  pipeline_tag: string | null;
}

/** One downloadable .gguf file in a repo (GET /api/v1/discover/files). */
export interface DiscoverFile {
  filename: string;
  size: number | null;
}

/** A download job (GET /api/v1/downloads, POST /api/v1/models/download). */
export interface DownloadJob {
  id: string;
  repo: string;
  filename: string;
  bytes_done: number;
  bytes_total: number | null;
  status: "downloading" | "done" | "error";
  error?: string;
}

export function formatGB(bytes: number): string {
  return `${(bytes / 1_000_000_000).toFixed(2)} GB`;
}

/** Library sidebar categories. "all" is My Models / View All. */
export type LibraryCategory =
  | "all"
  | "llms"
  | "embedding"
  | "decision"
  | "drafters";
