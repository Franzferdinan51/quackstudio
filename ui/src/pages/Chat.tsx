import { ArrowUp, ChevronDown, Paperclip } from "lucide-react";

/** Chat page skeleton: model picker pill + empty-state composer.
 *  Real loading/chat lands with the backend packs (P1+). */
export default function Chat() {
  return (
    <div className="flex h-full flex-col">
      {/* Model picker pill */}
      <div className="flex justify-center pt-4">
        <button
          type="button"
          className="flex items-center gap-2 rounded-full border border-edge bg-panel px-4 py-1.5 transition-colors hover:bg-panel-2"
        >
          <span>Select a model to load</span>
          <span className="text-muted">(⌘L)</span>
          <ChevronDown size={14} className="text-muted" />
        </button>
      </div>

      {/* Empty state */}
      <div className="flex flex-1 items-center justify-center px-6">
        <div className="max-w-sm text-center">
          <p className="text-[15px] font-medium">No model loaded</p>
          <p className="mt-1 text-muted">
            Pick a model from the Library, or press ⌘L to load one and
            start chatting.
          </p>
        </div>
      </div>

      {/* Composer */}
      <div className="mx-auto w-full max-w-3xl px-6 pb-5">
        <div className="rounded-2xl border border-edge bg-panel p-3 shadow-lg">
          <textarea
            rows={3}
            disabled
            placeholder="Send a message to the model..."
            className="w-full resize-none bg-transparent outline-none placeholder:text-muted disabled:cursor-not-allowed"
          />
          <div className="flex items-center justify-between">
            <button
              type="button"
              disabled
              className="flex h-7 w-7 items-center justify-center rounded-full text-muted disabled:cursor-not-allowed"
              title="Attach a file"
            >
              <Paperclip size={15} />
            </button>
            <button
              type="button"
              disabled
              className="flex h-7 w-7 items-center justify-center rounded-full bg-primary/40 text-white disabled:cursor-not-allowed"
              title="Send"
            >
              <ArrowUp size={15} />
            </button>
          </div>
        </div>
        <p className="mt-2 text-center text-[11px] text-muted">
          QuackStudio runs models locally — nothing leaves this machine.
        </p>
      </div>
    </div>
  );
}
