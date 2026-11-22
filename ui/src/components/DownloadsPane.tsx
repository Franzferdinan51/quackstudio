import { useCallback, useEffect, useState } from "react";
import { FolderOpen, Loader2, Play, X } from "lucide-react";
import type { DownloadJob } from "../types";
import { formatGB } from "../types";
import { apiBase } from "../api";
import { loadJSON, saveJSON } from "../store";

const HISTORY_KEY = "quack.downloadsHistory.v1";

function pct(job: DownloadJob): number | null {
  if (!job.bytes_total) return null;
  return Math.min(100, Math.round((job.bytes_done / job.bytes_total) * 100));
}

/** Downloads slide-over (top-right Downloads icon). Live jobs come
 *  from quackd's GET /api/v1/downloads; completed jobs are also
 *  remembered in localStorage so the pane survives a reload (the
 *  daemon forgets finished jobs on restart). */
export default function DownloadsPane({
  onClose,
  onLoadModel,
}: {
  onClose: () => void;
  onLoadModel: (modelKey: string) => void;
}) {
  const [jobs, setJobs] = useState<DownloadJob[]>([]);
  const [history, setHistory] = useState<DownloadJob[]>(() =>
    loadJSON<DownloadJob[]>(HISTORY_KEY, []),
  );
  const [cleared, setCleared] = useState<string[]>([]);

  const fetchJobs = useCallback(async () => {
    try {
      const base = await apiBase();
      const res = await fetch(`${base}/api/v1/downloads`);
      if (!res.ok) return;
      const data = (await res.json()) as DownloadJob[];
      setJobs(data);
      // Remember finished jobs across reloads.
      const done = data.filter((j) => j.status === "done");
      if (done.length > 0) {
        setHistory((prev) => {
          const merged = [...prev];
          for (const job of done) {
            if (!merged.some((h) => h.id === job.id)) merged.push(job);
          }
          saveJSON(HISTORY_KEY, merged);
          return merged;
        });
      }
    } catch {
      // Daemon unreachable — history still renders.
    }
  }, []);

  useEffect(() => {
    void fetchJobs();
    const timer = setInterval(() => void fetchJobs(), 2000);
    return () => clearInterval(timer);
  }, [fetchJobs]);

  const liveIds = new Set(jobs.map((j) => j.id));
  const finished: DownloadJob[] = [
    ...jobs.filter((j) => j.status !== "downloading"),
    ...history.filter((h) => !liveIds.has(h.id)),
  ].filter((j) => !cleared.includes(j.id));
  const active = jobs.filter((j) => j.status === "downloading");

  function clearFinished() {
    setCleared(finished.map((j) => j.id));
    setHistory([]);
    saveJSON(HISTORY_KEY, []);
  }

  function row(job: DownloadJob) {
    const p = pct(job);
    // A completed download lands at <models_dir>/<repo>/<filename>;
    // its library key is the repo id for single-file GGUF repos.
    const modelKey = job.repo;
    return (
      <div
        key={job.id}
        className="rounded-xl border border-edge bg-panel px-3.5 py-3"
      >
        <p className="truncate text-[13px] font-medium" title={job.id}>
          {job.repo} : {job.filename}
        </p>
        <p className="mt-0.5 text-[12px] text-muted">
          {job.status === "error"
            ? `Failed — ${job.error ?? "unknown error"}`
            : job.bytes_total
              ? `${formatGB(job.bytes_done)} of ${formatGB(job.bytes_total)}`
              : formatGB(job.bytes_done)}
          {job.status === "downloading" && p !== null ? ` · ${p}%` : ""}
          {job.status === "done" ? " · complete" : ""}
        </p>
        {job.status === "downloading" && (
          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-panel-2">
            <div
              className="h-full rounded-full bg-primary transition-all"
              style={{ width: `${p ?? 5}%` }}
            />
          </div>
        )}
        {job.status === "done" && (
          <button
            type="button"
            onClick={() => onLoadModel(modelKey)}
            className="mt-2 flex items-center gap-1.5 rounded-lg bg-panel-2 px-2.5 py-1.5 text-[12px] transition-colors hover:bg-edge"
          >
            <Play size={12} />
            Load Model
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div className="flex-1 bg-black/40" onClick={onClose} />
      <aside className="flex w-[380px] shrink-0 flex-col border-l border-edge bg-app-bg shadow-2xl">
        <div className="flex items-center justify-between border-b border-edge px-4 py-3">
          <h2 className="text-[14px] font-semibold">Downloads</h2>
          <button
            type="button"
            title="Close"
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-muted transition-colors hover:bg-panel hover:text-ink"
          >
            <X size={15} />
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-auto px-4 py-3">
          {active.length > 0 && (
            <>
              <p className="mb-2 flex items-center gap-1.5 text-[12px] font-medium text-muted">
                <Loader2 size={12} className="animate-spin" />
                Downloading
              </p>
              <div className="flex flex-col gap-2">{active.map(row)}</div>
            </>
          )}
          <div className="mt-4 flex items-center justify-between">
            <p className="text-[12px] font-medium text-muted">Completed</p>
            {finished.length > 0 && (
              <button
                type="button"
                onClick={clearFinished}
                className="text-[12px] text-muted transition-colors hover:text-ink"
              >
                Clear
              </button>
            )}
          </div>
          <div className="mt-2 flex flex-col gap-2">
            {finished.map(row)}
            {active.length === 0 && finished.length === 0 && (
              <p className="py-6 text-center text-muted">
                No downloads yet. Find a model in Model Search.
              </p>
            )}
          </div>
        </div>
        <button
          type="button"
          disabled
          title="Opens the models directory in Finder — not wired in the web UI"
          className="flex items-center gap-2 border-t border-edge px-4 py-3 text-left text-muted disabled:cursor-not-allowed"
        >
          <FolderOpen size={14} />
          Open downloads directory
        </button>
      </aside>
    </div>
  );
}
