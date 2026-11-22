# QuackStudio — Design

_An LM Studio reimplementation in Rust. Same backends, same UI, plus Strata and Maya as first-class backends. Working name: **QuackStudio** (Duckets can rename anytime). Started 2026-11-21 at Duckets' order._

Sources: `recon-rea.md` (the actual app on the Mac), `recon-github.md` (public repos + docs), `screenshots/` (5 pages captured and reviewed 2026-11-21), Hermes' backend-interface draft (pending — folds in at scaffold).

## What LM Studio actually is (the spec we're matching)

- Electron 38 app, UI ships unpacked (webpack, React + Tailwind family). App core since 0.4.0 is a headless daemon (**llmster**); the desktop app and the `lms` CLI are clients of it.
- Backends are **versioned packs** under `~/.lmstudio/extensions/backends/<engine>-<os>-<arch>@<version>/`, each with a `backend-manifest.json` declaring domains/formats. Engines load in-process via a **private** engine protocol — third parties cannot register engines with the real app. We define our own protocol in the same pack shape, which is exactly what makes Strata and Maya first-class here.
- One HTTP port (default 1234) serves four families: native REST v1, OpenAI-compat, Anthropic-compat, Jev-compat. The GUI/CLI/SDKs talk a separate WebSocket protocol (the public `lmstudio-js` schema).
- Models live in a two-level `publisher/model` directory; the catalog standard is **model.yaml**; per-model load config is a KV field set (contextLength, gpuOffload, evalBatchSize, …) with per-model overrides.
- On Duckets' Mac today: llama.cpp 2.80.1, mlx-llm 1.14.0, omlx 0.5.0, splash 1.3.0 packs; 18 local models / 134 GB at `~/.lmstudio/models` → `/Volumes/Sabrent/lmstudio-models`.

## Architecture

Three pieces, mirroring their split (daemon + app + CLI) but all Rust:

1. **`quackd`** — headless Rust daemon. Owns: the HTTP API server, the model library index, backend pack management, model load/unload lifecycle (instances, like theirs), download jobs. Runs with or without the desktop app, exactly like llmster.
2. **Desktop app** — Tauri shell (Rust) + React/Tailwind front-end replicating the captured UI. Talks to `quackd` over the same local API the CLI uses — no privileged back door, so UI/CLI/API can never drift apart.
3. **`quack` CLI** — Rust binary, `lms`-shaped command tree (subset at v1, growing): `server start|stop|status`, `ps`, `load`, `unload`, `ls`, `get`, `runtime ls|select`.

### Backend packs

Same layout as theirs: `~/.quackstudio/extensions/backends/<engine>-<os>-<arch>@<version>/backend-manifest.json`. Our engine protocol is JSON-RPC over stdio between `quackd` and a pack supervisor process; a pack must provide: `list_models`, `load`, `unload`, `chat` (streaming), `embeddings`, `status`. Packs at v1:

| Pack | What it wraps | Notes |
|---|---|---|
| **llama.cpp** | The `llama-server` binary from a llama.cpp pack | GGUF. Same binary lineage LM Studio ships; per-model load config maps to its flags (ctx, GPU offload layers, batch sizes). |
| **MLX** | `lmstudio-ai/mlx-engine` (MIT, Python on mlx-lm/mlx-vlm) | Apple Silicon safetensors + vision. We reuse their open engine rather than rewriting MLX inference. |
| **Strata** | Duckets' custom Strata engine (the Maya-integrated strata kernels) | First-class pack, not a Maya sub-feature. Pack manifest declares its real format/arch support from the strata build on the Mac. |
| **Maya** | Remote backend — Maya providers (:8085 Inkling, :8090 front door, per-provider ports) | Models on Maya appear in the library as remote entries; load/unload map to Maya's own lifecycle where it has one, chat proxies streaming. Zero local RAM cost. |

### Model library

- Reads the **same models directory** LM Studio uses (`~/.lmstudio/models`) by default, so Duckets' existing 18 models appear on first launch with zero copying. Our own additional directory is supported; writes (downloads) go to ours unless told otherwise.
- Index built from the filesystem (publisher/model, GGUF + mmproj siblings, MLX folders) + model.yaml metadata where present; arch/params/quant/capabilities (vision, tool use, reasoning) derived the way their Library table shows them.
- Per-model config overrides stored as JSON under our `.internal/` equivalent, same KV field names as theirs so presets and muscle memory transfer.

### HTTP API (tier 1 — v1 scope)

On the daemon's port (default **1234**, conflict-checked against a running LM Studio; configurable):

- REST v1: `GET /api/v1/models` (all downloaded + `loaded_instances`), `POST /api/v1/models/load` (returns `instance_id`), `POST /api/v1/models/unload` (by instance), `POST /api/v1/chat`, `POST /api/v1/models/download` + status.
- OpenAI-compat: `GET /v1/models` (loaded only), `POST /v1/chat/completions` (SSE streaming, tool calls, `reasoning_content`), `POST /v1/completions`, `POST /v1/embeddings`, `POST /v1/responses`.
- Anthropic-compat: `POST /v1/messages`.
- JIT auto-load behaviour matches theirs (request for a downloaded-but-unloaded model loads it when JIT is on).
- Optional Bearer API tokens, off by default, same as theirs.

Tier 2 (later): the WebSocket SDK protocol from `lmstudio-js`, so the real `lms` CLI and SDKs can drive QuackStudio.

### UI (from `screenshots/`)

React + Tailwind, dark theme throughout. Tokens observed: near-black app background, slightly lighter card/panel surfaces, 1px soft borders, purple accent (selected nav, badges), blue primary buttons, green status text, small gray secondary text, rounded-lg cards, thin left icon rail (Chat, Developer, Library, Discover, Settings gear at the foot).

Pages, in build order:
1. **Chat** — model picker pill (⌘L), tabs, composer with integration chips + context ring, right panel (Integrations / Configuration).
2. **Library (My Models)** — filter bar, model table (Device/Arch/Params/Publisher/LLM/Actions), right detail panel with Info/Load/Inference tabs and the load-config sliders.
3. **Developer (Model Server)** — status pill + toggle, Server Settings, Reachable-at row, Load Model, Loaded Models list, Supported endpoints bar, Server Logs strip.
4. **Discover** — search modal: staff-picks/catalog list left, model detail + Download Options + README right.
5. **Settings** — modal sheet: General, Appearance, Developer, Chat, Model Defaults, Integrations, LM Link, Runtime, Hardware.

## Build phases

- **P0 — Scaffold:** Cargo workspace (`quackd`, app, `quack` CLI), Tauri shell with the rail + page skeletons skinned from the screenshots, daemon answering `/api/v1/models` from the real models dir.
- **P1 — First fire:** llama.cpp pack end-to-end (load a GGUF, chat streaming in UI + `/v1/chat/completions`), Library table real.
- **P2 — MLX + load config:** MLX pack via mlx-engine; Library detail panel sliders wired to real load configs; Developer page live (start/stop server, loaded list, logs).
- **P3 — Strata + Maya packs:** both selectable in the same backend/runtime UI; Maya remote models in the library.
- **P4 — Discover + downloads + Settings:** model.yaml catalog browse, download jobs with progress, settings pages functional.
- **P5 — Publish:** name/branding final, README with screenshots, GitHub repo under Franzferdinan51, signed Mac build.

## Non-goals (v1)

- LM Link, account system, Hub publishing (`lms push`), RAG/document chat, MCP plugin marketplace — later, if ever.
- Windows/Linux builds: Mac-first (that's where the backends and Duckets live); Tauri keeps the door open.

## Open items

- Hermes' backend-interface draft (~/Desktop/lmstudio-clone/ on the Mac) — reconcile with the pack protocol at P0.
- Ghidra 12.2 update would unblock REA decompilation of their `llm_engine.node` — not blocking; our protocol is our own.
- Chunk→view mapping of their 127 renderer chunks (string analysis) if we ever need finer UI detail than the screenshots give.

## P3 findings (2026-11-21)

**Maya pack = remote proxy, as implemented.** There is no Maya engine process for QuackStudio to host: each Maya provider is already an OpenAI-compatible HTTP server on the Mac (per-provider ports — glm :8080, minimax :8081, qwen :8082, qwen-native :8083, hy4 :8084, ik2 :8085; the :8090 front door does not route every family, notably not ik2, so the pack talks to the per-provider endpoints directly). quackd polls each provider's `/v1/models` (1.5 s timeout, 30 s cache; an unanswering provider contributes nothing and never fails the listing) and merges the results into the library as remote entries keyed `maya/<provider>/<model-id>`, each carrying a synthetic instance — a Maya model is "loaded" exactly when its provider answers. Chat resolves a Maya key (or the provider's bare model id), rewrites the body's `model` field to the provider's own id, and streams the provider's response through untouched. Load is a verify (200 when the provider serves the model, 502 when it is not answering); unload refuses with a 400 — Maya manages its own lifecycle, and faking an unload would be a lie. Providers are configurable via `QUACK_MAYA_PROVIDERS` (`name=url` pairs). Cost to this machine: zero processes, zero RAM.

**Splash/Strata: no standalone server exists to wrap.** LM Studio's splash pack ships a bare amphibian Python interpreter driven by the host's private engine protocol — verified 2026-11-21: the pack's `engine_protocol_server` is just the amphibian `python` binary, with no standalone server CLI in the pack. So loading Splash models inside QuackStudio needs one of two real projects: an REA pass on that private engine protocol, or a native Strata pack speaking our own pack protocol. Neither is a wrapper job. Until then the scanner already lists Splash models in the library (loads return 501), and Strata-powered models remain reachable through the Maya pack, since Maya serves them itself.

## P6 findings (2026-11-21) — Splash pack: the P3 verdict was wrong

**Splash loads natively in QuackStudio.** The REA pass on the pack (see the P6 REA report) found there is no private host protocol to reimplement: the pack's amphibian Python runs `splash_engine`, a readable-Python OpenAI-compatible HTTP server on loopback, and the genuinely private binary protocol (SPLH v8) lives entirely inside the pack, between that server and its native engine. So the pack is a process launcher plus the existing proxy, exactly like the MLX pack: quackd resolves the newest `splash-*` pack under the backends dir, reads the interpreter from its `backend-manifest.json` (`engine_protocol_server.executable_relative_path`, relative to `vendor/`), and spawns `<python> -I -B -u -m splash_engine <modelRoot> --no-webui --tokenizer <modelRoot>/tokenizer --model <owner/repo> --binary <appLayer>/engine/splash --host 127.0.0.1 --port <p> --max-memory auto --max-context auto --idle-release off --decode-share 0 --allow-idle-sleep` with `SPLASH_API_KEY=<per-instance key>`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_VERBOSITY=error` and `PYTHONHOME`/`PYTHONPATH` stripped — LM Studio's own host argv, recovered from llmster's bundle. Readiness is `GET /ready` (600 s budget, matching their host), the actualized context comes from `GET /status` → `maximum_context_tokens` (262144 for the Arete packages), chat is the existing OpenAI proxy plus the instance's Bearer key, and unload is SIGINT (their documented graceful signal) with the usual SIGKILL fallback. The Library context/GPU sliders are not applied (the spawn uses `auto`, the engine sizes itself) and are labeled as such. Verified live 2026-11-21: `duckets/Arete-27B-Splash` loaded in 45 s, answered "4" to 2+2 streamed and non-streamed (121.5 tok/s on the non-streamed run), unloaded cleanly with RAM fully recovered. Strata-powered models remain reachable through the Maya pack as well.
