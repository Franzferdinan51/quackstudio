import { Download, Search, Sparkles } from "lucide-react";

interface Pick {
  name: string;
  publisher: string;
  blurb: string;
  size: string;
}

/** Staff-pick-style skeleton rows. The real model.yaml catalog
 *  browser + download jobs land in P4. */
const STAFF_PICKS: Pick[] = [
  {
    name: "qwen3-8b",
    publisher: "lmstudio-community",
    blurb: "Fast general-purpose chat model with strong tool use.",
    size: "5.0 GB",
  },
  {
    name: "llama-3.1-8b-instruct",
    publisher: "lmstudio-community",
    blurb: "Meta's workhorse instruct model, GGUF Q4_K_M.",
    size: "4.9 GB",
  },
  {
    name: "gemma-3-12b-it",
    publisher: "lmstudio-community",
    blurb: "Multimodal 12B from Google — text and image input.",
    size: "7.3 GB",
  },
];

/** Discover page skeleton: search field + staff picks list. */
export default function Discover() {
  return (
    <div className="flex h-full flex-col px-6 py-4">
      <div className="mx-auto flex w-full max-w-2xl items-center gap-2 rounded-lg border border-edge bg-panel px-3 py-2">
        <Search size={15} className="shrink-0 text-muted" />
        <input
          placeholder="Search models to download…"
          className="w-full bg-transparent text-[14px] outline-none placeholder:text-muted"
        />
      </div>

      <div className="mx-auto mt-6 w-full max-w-2xl">
        <h3 className="flex items-center gap-2 text-[14px] font-semibold">
          <Sparkles size={15} className="text-accent" />
          Staff Picks
        </h3>
        <p className="mt-1 text-[12px] text-muted">
          Skeleton — the live catalog arrives in P4.
        </p>
        <ul className="mt-3 space-y-2">
          {STAFF_PICKS.map((pick) => (
            <li
              key={pick.name}
              className="flex items-center gap-4 rounded-xl border border-edge bg-panel px-4 py-3"
            >
              <div className="min-w-0 flex-1">
                <p className="truncate font-medium">
                  {pick.name}
                  <span className="ml-2 text-[12px] font-normal text-muted">
                    {pick.publisher}
                  </span>
                </p>
                <p className="mt-0.5 truncate text-muted">{pick.blurb}</p>
              </div>
              <span className="shrink-0 text-muted">{pick.size}</span>
              <button
                type="button"
                className="flex shrink-0 items-center gap-1.5 rounded-lg border border-edge px-3 py-1.5 transition-colors hover:bg-panel-2"
              >
                <Download size={13} />
                Get
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
