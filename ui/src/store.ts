/**
 * Client-side persisted state (localStorage). The daemon owns
 * library/server/settings truth; what lives here is UI state:
 * chat sessions, integration toggles, presets, per-model
 * inference defaults, UI preferences, download history.
 */

export function loadJSON<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function saveJSON(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage full/blocked — UI state is best-effort.
  }
}

// ---------------------------------------------------------------------------
// Chats
// ---------------------------------------------------------------------------

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatSession {
  id: string;
  name: string;
  folderId: string | null;
  createdAt: number;
  messages: ChatMessage[];
  modelKey: string | null;
  systemPrompt: string;
  temperature: number;
  topP: number;
  reasoningBudget: string;
  notes: string[];
}

export interface ChatFolder {
  id: string;
  name: string;
}

export interface ChatsState {
  chats: ChatSession[];
  folders: ChatFolder[];
  activeId: string;
}

const CHATS_KEY = "quack.chats.v1";

export function newChatSession(name = "Unnamed Chat"): ChatSession {
  return {
    id: `chat-${Date.now()}-${Math.floor(Math.random() * 1e6)}`,
    name,
    folderId: null,
    createdAt: Date.now(),
    messages: [],
    modelKey: null,
    systemPrompt: "",
    temperature: 0.8,
    topP: 1,
    reasoningBudget: "Unrestricted",
    notes: [],
  };
}

export function loadChats(): ChatsState {
  const stored = loadJSON<ChatsState | null>(CHATS_KEY, null);
  if (stored && stored.chats.length > 0) return stored;
  const first = newChatSession();
  return { chats: [first], folders: [], activeId: first.id };
}

export function saveChats(state: ChatsState): void {
  saveJSON(CHATS_KEY, state);
}

// ---------------------------------------------------------------------------
// Integrations (the Chat right panel + composer chips + Settings)
// ---------------------------------------------------------------------------

export interface IntegrationDef {
  id: string;
  /** Tool names, for the Settings confirmation allowlist. */
  tools: string[];
  defaultOn: boolean;
}

/** The MCP servers from Duckets' LM Studio, as QuackStudio's
 *  built-in integration list. Toggles are local UI state — these
 *  plugins are not actually connected to quackd. */
export const INTEGRATIONS: IntegrationDef[] = [
  { id: "js-code-sandbox", tools: ["run_code"], defaultOn: false },
  { id: "rag-v1", tools: ["search_documents"], defaultOn: true },
  { id: "mcp/agent-teams", tools: ["dispatch_agent"], defaultOn: false },
  { id: "mcp/brave-search", tools: ["web_search"], defaultOn: true },
  { id: "mcp/browserclaw", tools: ["browse_page"], defaultOn: false },
  { id: "mcp/cannaai", tools: ["analyze_plant"], defaultOn: false },
  { id: "mcp/clawdwatch", tools: ["get_status"], defaultOn: false },
  { id: "mcp/context7", tools: ["query_docs"], defaultOn: true },
  { id: "mcp/time", tools: ["current_time"], defaultOn: false },
  { id: "mcp/everything", tools: ["echo"], defaultOn: false },
];

const INTEGRATIONS_KEY = "quack.integrations.v1";
const TOOL_CONFIRM_KEY = "quack.toolConfirm.v1";

export function loadIntegrationEnabled(): Record<string, boolean> {
  const stored = loadJSON<Record<string, boolean>>(INTEGRATIONS_KEY, {});
  const out: Record<string, boolean> = {};
  for (const def of INTEGRATIONS) {
    out[def.id] = stored[def.id] ?? def.defaultOn;
  }
  return out;
}

export function saveIntegrationEnabled(map: Record<string, boolean>): void {
  saveJSON(INTEGRATIONS_KEY, map);
}

/** "Run without confirmation" per `integration:tool`, default off. */
export function loadToolConfirm(): Record<string, boolean> {
  return loadJSON<Record<string, boolean>>(TOOL_CONFIRM_KEY, {});
}

export function saveToolConfirm(map: Record<string, boolean>): void {
  saveJSON(TOOL_CONFIRM_KEY, map);
}

// ---------------------------------------------------------------------------
// Configuration presets + per-model inference defaults
// ---------------------------------------------------------------------------

export interface Preset {
  systemPrompt: string;
  temperature: number;
  topP: number;
  reasoningBudget: string;
}

const PRESETS_KEY = "quack.presets.v1";

export function loadPresets(): Record<string, Preset> {
  return loadJSON<Record<string, Preset>>(PRESETS_KEY, {});
}

export function savePresets(map: Record<string, Preset>): void {
  saveJSON(PRESETS_KEY, map);
}

export interface InferenceDefaults {
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  repeatPenalty: number;
}

export const INFERENCE_FALLBACK: InferenceDefaults = {
  temperature: 0.8,
  topP: 1,
  topK: 40,
  minP: 0.05,
  repeatPenalty: 1.1,
};

export function loadInference(modelKey: string): InferenceDefaults {
  return {
    ...INFERENCE_FALLBACK,
    ...loadJSON<Partial<InferenceDefaults>>(
      `quack.inference.${modelKey}`,
      {},
    ),
  };
}

export function saveInference(
  modelKey: string,
  values: InferenceDefaults,
): void {
  saveJSON(`quack.inference.${modelKey}`, values);
}

// ---------------------------------------------------------------------------
// UI preferences
// ---------------------------------------------------------------------------

export interface UiPrefs {
  theme: "auto" | "dark" | "light";
  navPosition: "left";
  chatSort: "created-desc" | "created-asc" | "name";
  aiChatNames: boolean;
  /** Model picker: choose load parameters instead of instant load. */
  manualParams: boolean;
}

const PREFS_KEY = "quack.uiPrefs.v1";

export const PREFS_FALLBACK: UiPrefs = {
  theme: "auto",
  navPosition: "left",
  chatSort: "created-desc",
  aiChatNames: false,
  manualParams: false,
};

export function loadPrefs(): UiPrefs {
  return { ...PREFS_FALLBACK, ...loadJSON<Partial<UiPrefs>>(PREFS_KEY, {}) };
}

export function savePrefs(prefs: UiPrefs): void {
  saveJSON(PREFS_KEY, prefs);
}
