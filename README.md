# QuackStudio

An LM Studio reimplementation in Rust — same backends, same desktop UI,
plus **Strata** and **Maya** as first-class backends. Design and
rationale: [DESIGN.md](DESIGN.md).

> **Personal project.** QuackStudio is a personal app, built for one
> person's own setup and published as-is. It may not remain maintained —
> development happens in bursts, and issues or pull requests may go
> unanswered for long stretches, or forever. Use it at your own risk;
> no support is promised.

Three pieces, mirroring LM Studio's split (daemon + app + CLI), all Rust:

- **`quackd`** — headless daemon (their `llmster` analogue). Owns the
  HTTP API, the model library index, backend pack management, and the
  model load/unload lifecycle.
- **Desktop app** — Tauri shell + React/Tailwind UI replicating the
  captured LM Studio pages. Talks to `quackd` over the same public
  HTTP API the CLI uses — no privileged back door.
- **`quack` CLI** — `lms`-shaped subset: `ls`, `ps`, `server status`, …

Backends are versioned **packs** with a `backend-manifest.json`
(llama.cpp, MLX, Strata, Maya) — the pack protocol is ours, in their
shape, which is what makes Strata and Maya first-class. Backend packs
land in P1+; P0 is the scaffold: workspace, daemon library endpoint,
UI skeletons, CLI.

## Layout

```
Cargo.toml                  workspace (resolver 2)
crates/
  quack-core/               shared types: ModelEntry + scan_library,
                            LoadConfig (LM Studio field names),
                            PackManifest + scan_packs
  quackd/                   daemon: axum server on :1234
  quack-cli/                `quack` binary (clap + ureq)
src-tauri/                  Tauri v2 shell (productName QuackStudio)
ui/                         Vite + React + TS + Tailwind v4 front-end
```

## Run it

Daemon (serves the library from `~/.lmstudio/models` by default, so an
existing LM Studio library shows up with zero copying):

```sh
cargo run -p quackd
# env: QUACK_PORT=1234  QUACK_MODELS_DIR=~/.lmstudio/models
curl http://127.0.0.1:1234/api/v1/models
curl http://127.0.0.1:1234/api/v1/status
```

CLI (daemon must be running):

```sh
cargo run -p quack-cli -- ls
cargo run -p quack-cli -- ps
cargo run -p quack-cli -- server status
# env: QUACK_URL=http://127.0.0.1:1234
```

UI in a browser (dev mode; Library fetches live data from quackd):

```sh
cd ui
npm install
npm run dev        # http://localhost:5173
npm run build      # type-check + production build into ui/dist
```

Desktop app (requires the Tauri CLI, e.g. `cargo install tauri-cli
--version '^2'` or `npm i -g @tauri-apps/cli`):

```sh
cd src-tauri
cargo tauri dev    # beforeDevCommand starts the UI dev server
cargo tauri build
```

## API (P0)

| Route | Status |
|---|---|
| `GET /api/v1/models` | library index, each entry with `loaded_instances: []` |
| `GET /api/v1/status` | `{version, port, models_dir, model_count}` |
| `GET /v1/models` | OpenAI shape, loaded only (empty in P0) |
| `POST /api/v1/models/load` · `/unload` | 501 — backend packs land in P1 |

## Build phases

P0 scaffold (this) → P1 llama.cpp pack end-to-end → P2 MLX + load
config → P3 Strata + Maya packs → P4 Discover/downloads/Settings →
P5 publish to GitHub under Franzferdinan51. Full detail in
[../DESIGN.md](../DESIGN.md).
