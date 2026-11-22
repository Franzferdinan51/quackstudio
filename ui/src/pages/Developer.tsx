import { useEffect, useState } from "react";
import Toggle from "../components/Toggle";
import { apiBase } from "../api";

const ENDPOINTS = [
  "POST /v1/chat/completions",
  "POST /v1/completions",
  "POST /v1/embeddings",
  "POST /v1/messages",
  "GET /v1/models",
  "GET /api/v1/models",
];

/** Developer (Model Server) page skeleton. The toggle is visual in
 *  P0; real start/stop wires to quackd in P2. */
export default function Developer() {
  const [running, setRunning] = useState(true);
  const [reachable, setReachable] = useState("http://127.0.0.1:1234");

  // The daemon's real base (the Tauri shell may have put it on
  // 1235 when LM Studio owns 1234).
  useEffect(() => {
    let cancelled = false;
    void apiBase().then((base) => {
      if (!cancelled) setReachable(base);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="h-full overflow-auto">
      <div className="mx-auto w-full max-w-4xl px-8 py-6">
        {/* Server status card */}
        <section className="rounded-xl border border-edge bg-panel p-5">
          <div className="flex items-start justify-between">
            <div>
              <h2 className="text-[15px] font-semibold">Model Server</h2>
              <p className="mt-1 flex items-center gap-2 text-muted">
                Status:
                {running ? (
                  <span className="flex items-center gap-1.5 font-medium text-ok">
                    <span className="h-2 w-2 rounded-full bg-ok" />
                    Running
                  </span>
                ) : (
                  <span className="font-medium">Stopped</span>
                )}
              </p>
            </div>
            <Toggle on={running} onClick={() => setRunning((v) => !v)} />
          </div>
          <div className="mt-4 flex items-center gap-2 border-t border-edge pt-4 text-muted">
            <span>Reachable at:</span>
            <code className="rounded bg-app-bg px-2 py-0.5 font-mono text-[12px] text-ink">
              {reachable}
            </code>
          </div>
        </section>

        {/* Loaded models */}
        <section className="mt-6">
          <div className="flex items-center justify-between">
            <h3 className="text-[14px] font-semibold">Loaded Models</h3>
            <button
              type="button"
              className="rounded-lg bg-primary px-3.5 py-1.5 font-medium text-white transition-opacity hover:opacity-90"
            >
              Load Model
            </button>
          </div>
          <div className="mt-3 rounded-xl border border-dashed border-edge px-5 py-8 text-center text-muted">
            No models loaded.
            <br />
            Use QuackStudio from other apps or from your code.
          </div>
        </section>

        {/* Supported endpoints */}
        <section className="mt-6">
          <h3 className="text-[14px] font-semibold">Supported Endpoints</h3>
          <div className="mt-3 flex flex-wrap gap-2">
            {ENDPOINTS.map((ep) => (
              <code
                key={ep}
                className="rounded-full border border-edge bg-panel px-2.5 py-1 font-mono text-[11px] text-muted"
              >
                {ep}
              </code>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
