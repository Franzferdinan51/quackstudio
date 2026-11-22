import { useEffect, useState } from "react";
import { Check, Search } from "lucide-react";
import type { ModelEntry } from "../types";
import { DAEMON_URL, formatGB } from "../types";

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
    loaded_instances: [],
  },
];

/** Library (My Models): filter bar + model table fed by quackd. */
export default function Library() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [offline, setOffline] = useState(false);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    let cancelled = false;
    fetch(`${DAEMON_URL}/api/v1/models`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<ModelEntry[]>;
      })
      .then((data) => {
        if (!cancelled) setModels(data);
      })
      .catch(() => {
        if (!cancelled) setOffline(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const rows = offline ? SAMPLE_ROWS : models;
  const needle = filter.trim().toLowerCase();
  const visible = rows.filter(
    (m) =>
      needle === "" ||
      `${m.publisher}/${m.display_name}`.toLowerCase().includes(needle),
  );

  return (
    <div className="flex h-full flex-col px-6 py-4">
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
                className="border-b border-edge/50 transition-colors last:border-0 hover:bg-panel"
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
                </td>
                <td className="px-4 py-2.5 text-muted">Local</td>
                <td className="px-4 py-2.5 text-muted">{m.arch ?? "—"}</td>
                <td className="px-4 py-2.5 text-muted">—</td>
                <td className="px-4 py-2.5 text-muted">
                  {formatGB(m.size_bytes)}
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
                  No models found. Download one from Discover.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
