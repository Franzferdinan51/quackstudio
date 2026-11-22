/**
 * Per-model configs live in quackd (`model-configs.json` in the
 * settings dir, GET/POST /api/v1/model-configs): load overrides,
 * inference defaults, and Library pins, keyed by model key.
 *
 * This module is the UI's client for that store. It also keeps
 * the localStorage inference mirror (store.ts `quack.inference.*`)
 * in sync, because the Chat page reads its per-model defaults
 * from there — the mirror is a cache of the daemon's truth, and
 * the daemon's chat proxy applies the same defaults server-side
 * to any client.
 */

import { apiBase } from "./api";
import type {
  InferenceConfigWire,
  LoadConfigWire,
  ModelConfigWire,
} from "./types";
import {
  INFERENCE_FALLBACK,
  loadJSON,
  saveInference,
  type InferenceDefaults,
} from "./store";

export type ModelConfigMap = Record<string, ModelConfigWire>;

export function emptyModelConfig(): ModelConfigWire {
  return { load: {}, inference: {}, pinned: false };
}

/** A model counts as Customized (LM Studio's pill + red gear
 *  dot) when it carries any stored load or inference override. */
export function isCustomized(cfg: ModelConfigWire | undefined): boolean {
  if (!cfg) return false;
  return (
    Object.keys(cfg.load).length > 0 ||
    Object.keys(cfg.inference).length > 0 ||
    (cfg.inference.stop_strings?.length ?? 0) > 0
  );
}

/** The daemon's inference wire shape → the flat defaults the
 *  Chat page seeds from. */
export function inferenceWireToDefaults(
  wire: InferenceConfigWire,
): InferenceDefaults {
  return {
    temperature: wire.temperature ?? INFERENCE_FALLBACK.temperature,
    topP: wire.top_p ?? INFERENCE_FALLBACK.topP,
    topK: wire.top_k ?? INFERENCE_FALLBACK.topK,
    minP: wire.min_p ?? INFERENCE_FALLBACK.minP,
    repeatPenalty: wire.repeat_penalty ?? INFERENCE_FALLBACK.repeatPenalty,
    systemPrompt: wire.system_prompt ?? "",
    presencePenalty: wire.presence_penalty ?? 0,
  };
}

let cache: ModelConfigMap | null = null;
let inflight: Promise<ModelConfigMap> | null = null;

async function fetchAll(base: string): Promise<ModelConfigMap> {
  try {
    const res = await fetch(`${base}/api/v1/model-configs`);
    if (!res.ok) return {};
    return (await res.json()) as ModelConfigMap;
  } catch {
    return {};
  }
}

/** One-time migration of the P5 localStorage stores
 *  (`quack.loadAdv.*`, `quack.inference.*`) into the daemon
 *  store, for model keys the daemon has no config for yet. */
async function migrateLocal(
  base: string,
  map: ModelConfigMap,
): Promise<void> {
  const keys = new Set<string>();
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k?.startsWith("quack.loadAdv.")) {
        keys.add(k.slice("quack.loadAdv.".length));
      }
      if (k?.startsWith("quack.inference.")) {
        keys.add(k.slice("quack.inference.".length));
      }
    }
  } catch {
    return;
  }
  for (const key of keys) {
    if (map[key]) continue;
    const load: LoadConfigWire = {};
    const adv = loadJSON<{
      nThreads?: string;
      evalBatch?: string;
      physBatch?: string;
    }>(`quack.loadAdv.${key}`, {});
    const num = (s?: string) => {
      const n = parseInt(s ?? "", 10);
      return Number.isFinite(n) ? n : undefined;
    };
    const cpu = num(adv.nThreads);
    const eb = num(adv.evalBatch);
    const pb = num(adv.physBatch);
    if (cpu !== undefined) load.cpu_threads = cpu;
    if (eb !== undefined) load.eval_batch_size = eb;
    if (pb !== undefined) load.physical_batch_size = pb;

    const inference: InferenceConfigWire = {};
    const inf = loadJSON<Partial<InferenceDefaults>>(
      `quack.inference.${key}`,
      {},
    );
    if (inf.temperature !== undefined) inference.temperature = inf.temperature;
    if (inf.topP !== undefined) {
      inference.top_p = inf.topP;
      inference.top_p_enabled = true;
    }
    if (inf.topK !== undefined) inference.top_k = inf.topK;
    if (inf.minP !== undefined) {
      inference.min_p = inf.minP;
      inference.min_p_enabled = true;
    }
    if (inf.repeatPenalty !== undefined) {
      inference.repeat_penalty = inf.repeatPenalty;
      inference.repeat_penalty_enabled = true;
    }
    if (inf.systemPrompt) inference.system_prompt = inf.systemPrompt;

    if (Object.keys(load).length === 0 && Object.keys(inference).length === 0) {
      continue;
    }
    try {
      const res = await fetch(`${base}/api/v1/model-configs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key, load, inference }),
      });
      if (res.ok) {
        map[key] = (await res.json()) as ModelConfigWire;
      }
    } catch {
      // Migration is best-effort; the local values remain local.
    }
  }
}

/** Fetch (once) + migrate + mirror. Subsequent calls return the
 *  cached map, which saveModelConfig keeps current. */
export function syncModelConfigs(): Promise<ModelConfigMap> {
  if (cache) return Promise.resolve(cache);
  if (!inflight) {
    inflight = (async () => {
      const base = await apiBase();
      const map = await fetchAll(base);
      await migrateLocal(base, map);
      for (const [key, cfg] of Object.entries(map)) {
        saveInference(key, inferenceWireToDefaults(cfg.inference));
      }
      cache = map;
      return map;
    })();
  }
  return inflight;
}

/** Replace one section (or the pin) of a model's config in the
 *  daemon store; updates the cache and the inference mirror. */
export async function saveModelConfig(
  key: string,
  patch: Partial<ModelConfigWire>,
): Promise<ModelConfigWire> {
  const base = await apiBase();
  const res = await fetch(`${base}/api/v1/model-configs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key, ...patch }),
  });
  if (!res.ok) throw new Error(`config save failed (HTTP ${res.status})`);
  const updated = (await res.json()) as ModelConfigWire;
  if (cache) cache[key] = updated;
  if (patch.inference) {
    saveInference(key, inferenceWireToDefaults(updated.inference));
  }
  return updated;
}

/** Forget a model's config locally after its folder is deleted
 *  (the daemon drops its own copy in the delete call). */
export function forgetModelConfig(key: string): void {
  if (cache) delete cache[key];
}
