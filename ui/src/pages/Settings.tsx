import { useState } from "react";
import Toggle from "../components/Toggle";

const NAV_ITEMS = [
  "General",
  "Appearance",
  "Developer",
  "Chat",
  "Model Defaults",
  "Integrations",
  "LM Link",
  "Runtime",
  "Hardware",
] as const;

interface ToggleRow {
  label: string;
  hint: string;
  initial: boolean;
}

const GENERAL_ROWS: ToggleRow[] = [
  {
    label: "Launch QuackStudio at login",
    hint: "Start the app and daemon when you log in.",
    initial: false,
  },
  {
    label: "Show icon in menu bar",
    hint: "Quick access to loaded models and server status.",
    initial: true,
  },
  {
    label: "Automatically check for updates",
    hint: "Includes backend runtime packs (llama.cpp, MLX, Strata, Maya).",
    initial: true,
  },
  {
    label: "Start server on launch",
    hint: "Run quackd automatically when the app opens.",
    initial: true,
  },
];

/** Settings skeleton: left nav + General pane with toggle rows.
 *  Other panes and persistence land in P4. */
export default function SettingsPage() {
  const [active, setActive] = useState<(typeof NAV_ITEMS)[number]>("General");
  const [values, setValues] = useState<boolean[]>(
    GENERAL_ROWS.map((r) => r.initial),
  );

  return (
    <div className="flex h-full">
      {/* Left nav */}
      <aside className="w-52 shrink-0 border-r border-edge px-3 py-4">
        <ul className="space-y-0.5">
          {NAV_ITEMS.map((item) => (
            <li key={item}>
              <button
                type="button"
                onClick={() => setActive(item)}
                className={`w-full rounded-lg px-3 py-1.5 text-left transition-colors ${
                  active === item
                    ? "bg-accent/20 font-medium text-ink"
                    : "text-muted hover:bg-panel hover:text-ink"
                }`}
              >
                {item}
              </button>
            </li>
          ))}
        </ul>
      </aside>

      {/* Pane */}
      <div className="min-w-0 flex-1 overflow-auto px-8 py-6">
        {active === "General" ? (
          <div className="max-w-2xl">
            <h2 className="text-[15px] font-semibold">General</h2>
            <div className="mt-4 divide-y divide-edge rounded-xl border border-edge bg-panel">
              {GENERAL_ROWS.map((row, i) => (
                <div
                  key={row.label}
                  className="flex items-center justify-between gap-6 px-4 py-3"
                >
                  <div>
                    <p>{row.label}</p>
                    <p className="mt-0.5 text-[12px] text-muted">{row.hint}</p>
                  </div>
                  <Toggle
                    on={values[i]}
                    onClick={() =>
                      setValues((prev) =>
                        prev.map((v, j) => (j === i ? !v : v)),
                      )
                    }
                  />
                </div>
              ))}
              <div className="flex items-center justify-between gap-6 px-4 py-3">
                <div>
                  <p>Models directory</p>
                  <p className="mt-0.5 text-[12px] text-muted">
                    Shared with LM Studio — existing downloads appear
                    automatically.
                  </p>
                </div>
                <code className="shrink-0 rounded bg-app-bg px-2 py-0.5 font-mono text-[12px]">
                  ~/.lmstudio/models
                </code>
              </div>
            </div>
          </div>
        ) : (
          <div className="max-w-2xl">
            <h2 className="text-[15px] font-semibold">{active}</h2>
            <p className="mt-4 text-muted">
              This pane is a skeleton — it lands in P4.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
