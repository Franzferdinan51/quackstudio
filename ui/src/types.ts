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
  /** Base URL of the serving provider for remote (Maya) models; null locally. */
  remote_url: string | null;
  loaded_instances: LoadedInstance[];
}

/** A Maya provider entry (quack-core's MayaProvider). */
export interface MayaProvider {
  name: string;
  base_url: string;
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
  note?: string;
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
