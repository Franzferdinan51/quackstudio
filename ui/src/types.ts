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
  format: "gguf" | "mlx";
  size_bytes: number;
  path: string;
  arch: string | null;
  vision: boolean;
  loaded_instances: LoadedInstance[];
}

export const DAEMON_URL = "http://127.0.0.1:1234";

export function formatGB(bytes: number): string {
  return `${(bytes / 1_000_000_000).toFixed(2)} GB`;
}
