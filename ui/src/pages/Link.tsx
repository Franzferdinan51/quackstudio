import { useEffect, useState } from "react";
import { MonitorSmartphone } from "lucide-react";
import type { ModelEntry, StatusView } from "../types";
import { formatGB } from "../types";
import { apiBase } from "../api";

/** LM Link: QuackStudio has no LM Link protocol — the honest
 *  equivalent of "linked devices" here is the Maya providers.
 *  This device card comes from /api/v1/status host fields; the
 *  right panel lists the Maya remote models (our remote models). */
export default function LinkPage({ status }: { status: StatusView | null }) {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    void (async () => {
      try {
        const base = await apiBase();
        const res = await fetch(`${base}/api/v1/models`);
        if (res.ok) setModels((await res.json()) as ModelEntry[]);
      } catch {
        // Remote list just stays empty.
      }
    })();
  }, []);

  const remote = models.filter(
    (m) =>
      m.format === "maya" &&
      (filter.trim() === "" ||
        m.display_name.toLowerCase().includes(filter.trim().toLowerCase())),
  );
  const host = status?.host;

  return (
    <div className="flex h-full">
      <div className="min-w-0 flex-1 overflow-auto px-8 py-6">
        <div className="mx-auto max-w-2xl">
          <h2 className="flex items-center gap-2 text-[16px] font-semibold">
            <span className="h-2.5 w-2.5 rounded-full bg-muted" />0 Connected
            Devices
          </h2>

          {/* This device */}
          <section className="mt-5 rounded-xl border border-edge bg-panel p-5">
            <div className="flex items-center gap-4">
              <span className="flex h-14 w-14 items-center justify-center rounded-xl bg-panel-2 text-muted">
                <MonitorSmartphone size={26} />
              </span>
              <div>
                <p className="text-[15px] font-medium">
                  {host?.name ?? "This Mac"}
                </p>
                <p className="text-[12px] text-muted">This device</p>
              </div>
            </div>
            <div className="mt-4 grid grid-cols-2 gap-3">
              <div className="rounded-lg bg-panel-2 px-3.5 py-3">
                <p className="text-[11px] tracking-wide text-muted uppercase">
                  GPU
                </p>
                <p className="mt-0.5 text-[13px]">
                  {host?.gpu ?? "Apple Silicon (Metal)"}
                </p>
                <p className="text-[12px] text-muted">{host?.chip ?? "—"}</p>
              </div>
              <div className="rounded-lg bg-panel-2 px-3.5 py-3">
                <p className="text-[11px] tracking-wide text-muted uppercase">
                  Memory
                </p>
                <p className="mt-0.5 text-[13px]">
                  {host ? formatGB(host.total_ram_bytes) : "—"}
                </p>
                <p className="text-[12px] text-muted">
                  Unified — VRAM = RAM on Apple Silicon
                </p>
              </div>
            </div>
            <dl className="mt-4 flex flex-col gap-2.5 text-[13px]">
              {[
                ["Owner", "duckets"],
                [
                  "Status",
                  status
                    ? `${status.maya_providers_online}/${status.maya_providers} Maya providers online`
                    : "quackd unreachable",
                ],
                ["Daemon port", status ? String(status.port) : "—"],
              ].map(([label, value]) => (
                <div key={label} className="flex items-baseline gap-3">
                  <dt className="w-24 shrink-0 text-muted">{label}</dt>
                  <dd>{value}</dd>
                </div>
              ))}
            </dl>
          </section>

          {/* Network devices */}
          <section className="mt-6">
            <h3 className="text-[14px] font-semibold">Network Devices</h3>
            <div className="mt-3 rounded-xl border border-dashed border-edge px-5 py-8 text-center text-muted">
              No network devices — QuackStudio Link is not implemented
              yet; Maya providers serve as linked services.
            </div>
          </section>
        </div>
      </div>

      {/* Remote models */}
      <aside className="flex w-80 shrink-0 flex-col border-l border-edge bg-panel px-4 py-4">
        <h3 className="text-[14px] font-semibold">Models on Remote Device</h3>
        <p className="mt-0.5 text-[12px] text-muted">
          Served by the Maya providers — QuackStudio's remote models.
        </p>
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter models…"
          className="mt-3 w-full rounded-lg border border-edge bg-app-bg px-3 py-1.5 outline-none placeholder:text-muted"
        />
        <div className="mt-3 flex min-h-0 flex-1 flex-col gap-2 overflow-auto">
          {remote.map((m) => (
            <div
              key={m.key}
              className="rounded-xl border border-edge bg-panel-2 px-3.5 py-2.5"
            >
              <p className="truncate text-[13px] font-medium">
                {m.display_name}
              </p>
              <p className="truncate text-[12px] text-muted">
                {m.loaded_instances[0]?.backend ?? "maya"} · {m.remote_url}
              </p>
            </div>
          ))}
          {remote.length === 0 && (
            <p className="py-4 text-center text-[12px] text-muted">
              No Maya models are answering right now.
            </p>
          )}
        </div>
      </aside>
    </div>
  );
}
