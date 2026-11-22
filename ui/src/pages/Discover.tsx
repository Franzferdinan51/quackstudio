import { useCallback, useEffect, useState } from "react";
import { Check, Download, Heart, Loader2, Search } from "lucide-react";
import type { DiscoverFile, DiscoverModel, DownloadJob } from "../types";
import { formatGB } from "../types";
import { apiBase } from "../api";

/** 10704 → "10.7k", 2_400_000 → "2.4M". */
function formatCount(n: number | null): string {
  if (n === null) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

/** Pull the daemon's `{"error": …}` message out of a response body. */
function errorFrom(data: unknown, fallback: string): string {
  if (
    data !== null &&
    typeof data === "object" &&
    "error" in data &&
    typeof (data as { error: unknown }).error === "string"
  ) {
    return (data as { error: string }).error;
  }
  return fallback;
}

function progressPct(job: DownloadJob): number | null {
  if (job.bytes_total === null || job.bytes_total === 0) return null;
  return Math.min(100, (job.bytes_done / job.bytes_total) * 100);
}

/** One job row in the downloads strip. */
function JobRow({ job }: { job: DownloadJob }) {
  const pct = progressPct(job);
  return (
    <div className="px-4 py-2.5">
      <div className="flex items-baseline gap-2">
        <span className="min-w-0 flex-1 truncate font-medium">
          {job.filename}
          <span className="ml-2 text-[11px] font-normal text-muted">
            {job.repo}
          </span>
        </span>
        {job.status === "downloading" && (
          <span className="shrink-0 text-muted">
            {formatGB(job.bytes_done)}
            {job.bytes_total !== null &&
              ` / ${formatGB(job.bytes_total)}`}
            {pct !== null && ` · ${pct.toFixed(0)}%`}
          </span>
        )}
        {job.status === "done" && (
          <span className="flex shrink-0 items-center gap-1 text-ok">
            <Check size={13} />
            In Library
          </span>
        )}
        {job.status === "error" && (
          <span className="shrink-0 text-red-400">
            Failed{job.error ? ` — ${job.error}` : ""}
          </span>
        )}
      </div>
      {job.status === "downloading" && (
        <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-edge">
          <div
            className={`h-full bg-primary transition-all ${
              pct === null ? "w-full animate-pulse" : ""
            }`}
            style={pct !== null ? { width: `${pct}%` } : undefined}
          />
        </div>
      )}
    </div>
  );
}

/** Discover: live Hugging Face search (proxied by quackd), a
 *  repo's .gguf files, and background download jobs with progress.
 *  Layout follows the captured Discover modal: search on top,
 *  results left, download options right. */
export default function Discover() {
  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [results, setResults] = useState<DiscoverModel[] | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);

  const [selectedRepo, setSelectedRepo] = useState<string | null>(null);
  const [files, setFiles] = useState<DiscoverFile[] | null>(null);
  const [filesLoading, setFilesLoading] = useState(false);
  const [filesError, setFilesError] = useState<string | null>(null);

  const [jobs, setJobs] = useState<DownloadJob[]>([]);
  const [startingFile, setStartingFile] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const fetchJobs = useCallback(async () => {
    try {
      const base = await apiBase();
      const res = await fetch(`${base}/api/v1/downloads`);
      if (!res.ok) return;
      setJobs((await res.json()) as DownloadJob[]);
    } catch {
      // Daemon unreachable — the strip just stays empty; search
      // surfaces its own error.
    }
  }, []);

  useEffect(() => {
    void fetchJobs();
  }, [fetchJobs]);

  // Poll while any job is downloading.
  const anyDownloading = jobs.some((j) => j.status === "downloading");
  useEffect(() => {
    if (!anyDownloading) return;
    const timer = setInterval(() => {
      void fetchJobs();
    }, 1500);
    return () => clearInterval(timer);
  }, [anyDownloading, fetchJobs]);

  async function runSearch() {
    const q = query.trim();
    if (!q || searching) return;
    setSearching(true);
    setSearchError(null);
    try {
      const base = await apiBase();
      const res = await fetch(
        `${base}/api/v1/discover/search?q=${encodeURIComponent(q)}`,
      );
      const data: unknown = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error(errorFrom(data, `search failed (HTTP ${res.status})`));
      }
      setResults(data as DiscoverModel[]);
    } catch (e) {
      setResults([]);
      setSearchError(e instanceof Error ? e.message : String(e));
    } finally {
      setSearching(false);
    }
  }

  async function selectRepo(repo: string) {
    setSelectedRepo(repo);
    setFiles(null);
    setFilesError(null);
    setActionError(null);
    setFilesLoading(true);
    try {
      const base = await apiBase();
      const res = await fetch(
        `${base}/api/v1/discover/files?repo=${encodeURIComponent(repo)}`,
      );
      const data: unknown = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error(
          errorFrom(data, `file list failed (HTTP ${res.status})`),
        );
      }
      setFiles(data as DiscoverFile[]);
    } catch (e) {
      setFilesError(e instanceof Error ? e.message : String(e));
    } finally {
      setFilesLoading(false);
    }
  }

  async function startDownload(file: DiscoverFile) {
    if (!selectedRepo || startingFile) return;
    setStartingFile(file.filename);
    setActionError(null);
    try {
      const base = await apiBase();
      const res = await fetch(`${base}/api/v1/models/download`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          repo: selectedRepo,
          filename: file.filename,
        }),
      });
      const data: unknown = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error(
          errorFrom(data, `download failed to start (HTTP ${res.status})`),
        );
      }
      await fetchJobs();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setStartingFile(null);
    }
  }

  function jobFor(filename: string): DownloadJob | undefined {
    return jobs.find(
      (j) => j.repo === selectedRepo && j.filename === filename,
    );
  }

  return (
    <div className="flex h-full flex-col px-6 py-4">
      {/* Search bar */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void runSearch();
        }}
        className="mx-auto flex w-full max-w-2xl items-center gap-2 rounded-lg border border-edge bg-panel px-3 py-2"
      >
        <Search size={15} className="shrink-0 text-muted" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search Hugging Face models to download…"
          className="w-full bg-transparent text-[14px] outline-none placeholder:text-muted"
        />
        <button
          type="submit"
          disabled={searching || query.trim() === ""}
          className="flex shrink-0 items-center gap-1.5 rounded-lg bg-primary px-3 py-1 text-[12px] font-medium text-white transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
        >
          {searching && <Loader2 size={13} className="animate-spin" />}
          Search
        </button>
      </form>

      {/* Downloads strip */}
      {jobs.length > 0 && (
        <div className="mx-auto mt-4 w-full max-w-5xl divide-y divide-edge/60 rounded-xl border border-edge bg-panel">
          {jobs.map((job) => (
            <JobRow key={job.id} job={job} />
          ))}
        </div>
      )}

      {/* Results + detail */}
      <div className="mx-auto mt-4 flex w-full max-w-5xl min-h-0 flex-1 gap-4">
        {/* Results list */}
        <div className="w-[380px] shrink-0 overflow-auto rounded-xl border border-edge">
          {results === null && (
            <p className="px-4 py-8 text-center text-muted">
              Search Hugging Face for a model to download — results
              sort by downloads, like the hub.
            </p>
          )}
          {searchError && (
            <p className="px-4 py-8 text-center text-red-400">
              {searchError}
            </p>
          )}
          {results !== null && !searchError && results.length === 0 && (
            <p className="px-4 py-8 text-center text-muted">
              No models matched that search.
            </p>
          )}
          <ul className="divide-y divide-edge/50">
            {(results ?? []).map((model) => (
              <li key={model.id}>
                <button
                  type="button"
                  onClick={() => void selectRepo(model.id)}
                  className={`w-full px-4 py-3 text-left transition-colors hover:bg-panel ${
                    model.id === selectedRepo ? "bg-panel-2" : ""
                  }`}
                >
                  <p className="truncate font-medium">{model.id}</p>
                  <p className="mt-1 flex items-center gap-3 text-[11px] text-muted">
                    <span className="flex items-center gap-1">
                      <Download size={11} />
                      {formatCount(model.downloads)}
                    </span>
                    <span className="flex items-center gap-1">
                      <Heart size={11} />
                      {formatCount(model.likes)}
                    </span>
                    {model.pipeline_tag && (
                      <span className="rounded bg-panel-2 px-1.5 py-0.5">
                        {model.pipeline_tag}
                      </span>
                    )}
                  </p>
                </button>
              </li>
            ))}
          </ul>
        </div>

        {/* Detail / download options */}
        <div className="min-w-0 flex-1 overflow-auto rounded-xl border border-edge bg-panel px-5 py-4">
          {selectedRepo === null ? (
            <p className="py-8 text-center text-muted">
              Pick a model to see its downloadable GGUF files.
            </p>
          ) : (
            <>
              <h3 className="truncate text-[15px] font-semibold">
                {selectedRepo}
              </h3>
              <p className="mt-1 text-[12px] text-muted">
                Download Options — GGUF files in this repo. Downloads
                land in the shared models directory and appear in My
                Models when they finish.
              </p>
              {actionError && (
                <p className="mt-3 text-[12px] text-red-400">
                  {actionError}
                </p>
              )}
              {filesLoading && (
                <p className="mt-6 flex items-center gap-2 text-muted">
                  <Loader2 size={14} className="animate-spin" />
                  Listing files…
                </p>
              )}
              {filesError && (
                <p className="mt-6 text-red-400">{filesError}</p>
              )}
              {files !== null && files.length === 0 && (
                <p className="mt-6 text-muted">
                  No GGUF files in this repo — QuackStudio downloads
                  GGUF (MLX downloads land in P5).
                </p>
              )}
              <ul className="mt-4 space-y-2">
                {(files ?? []).map((file) => {
                  const job = jobFor(file.filename);
                  const pct = job ? progressPct(job) : null;
                  return (
                    <li
                      key={file.filename}
                      className="flex items-center gap-3 rounded-xl border border-edge bg-panel-2 px-4 py-3"
                    >
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-medium">
                          {file.filename}
                        </p>
                        <p className="mt-0.5 text-[11px] text-muted">
                          {file.size !== null
                            ? formatGB(file.size)
                            : "size unknown"}
                          {job?.status === "downloading" &&
                            ` · ${formatGB(job.bytes_done)} downloaded`}
                          {job?.status === "error" &&
                            ` · ${job.error ?? "download failed"}`}
                        </p>
                        {job?.status === "downloading" && (
                          <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-edge">
                            <div
                              className={`h-full bg-primary transition-all ${
                                pct === null ? "w-full animate-pulse" : ""
                              }`}
                              style={
                                pct !== null
                                  ? { width: `${pct}%` }
                                  : undefined
                              }
                            />
                          </div>
                        )}
                      </div>
                      {job?.status === "done" ? (
                        <span className="flex shrink-0 items-center gap-1.5 rounded-lg border border-edge px-3 py-1.5 text-ok">
                          <Check size={13} />
                          In Library
                        </span>
                      ) : job?.status === "downloading" ? (
                        <span className="flex shrink-0 items-center gap-1.5 rounded-lg border border-edge px-3 py-1.5 text-muted">
                          <Loader2 size={13} className="animate-spin" />
                          {pct !== null ? `${pct.toFixed(0)}%` : "…"}
                        </span>
                      ) : (
                        <button
                          type="button"
                          disabled={startingFile !== null}
                          onClick={() => void startDownload(file)}
                          className="flex shrink-0 items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 font-medium text-white transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
                        >
                          {startingFile === file.filename ? (
                            <Loader2 size={13} className="animate-spin" />
                          ) : (
                            <Download size={13} />
                          )}
                          {job?.status === "error" ? "Retry" : "Download"}
                        </button>
                      )}
                    </li>
                  );
                })}
              </ul>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
