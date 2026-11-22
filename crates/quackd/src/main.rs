//! quackd — the QuackStudio daemon.
//!
//! P4 surface (see ../DESIGN.md):
//!   GET  /api/v1/models          library index + loaded instances,
//!                                with Maya remote models merged in
//!   GET  /api/v1/status          daemon status
//!   POST /api/v1/models/load     spawn a backend engine for a model
//!   POST /api/v1/models/unload   stop an instance by id
//!   GET  /api/v1/discover/search Hugging Face model search (proxy)
//!   GET  /api/v1/discover/files  a repo's downloadable .gguf files
//!   POST /api/v1/models/download start a background download job
//!   GET  /api/v1/downloads       download jobs with progress
//!   GET  /api/v1/settings        effective settings
//!   POST /api/v1/settings        persist + apply settings
//!   GET  /api/v1/model-configs   per-model configs (load +
//!                                inference + pinned), persisted
//!                                in the settings dir
//!   POST /api/v1/model-configs   upsert one model's config
//!   POST /api/v1/models/reveal   reveal a model folder (or the
//!                                models dir) in Finder
//!   POST /api/v1/models/delete   delete a local model's folder
//!   GET  /v1/models              OpenAI-compat list (loaded only,
//!                                plus Maya remote models)
//!   POST /v1/chat/completions    proxy to the instance's engine (SSE passthrough)
//!
//! Loads dispatch on the model's format, mirroring LM Studio's
//! backend preferences:
//!   GGUF       → llama.cpp pack: the newest `llama-server` under
//!                the backends dir, one child process per instance.
//!   MLX        → oMLX pack: the newest oMLX build's Python running
//!                `mlx_vlm.server`, one child per instance.
//!   Splash     → Splash pack: the pack's amphibian Python running
//!                the `splash_engine` server (OpenAI-compatible
//!                HTTP on loopback, per-instance API key), one
//!                child process per instance.
//!   Maya       → remote pack: Maya providers are OpenAI-compatible
//!                servers of their own; their models are listed as
//!                remote library entries (key `maya/<provider>/<id>`),
//!                are "loaded" whenever the provider answers, and
//!                chat proxies straight to the provider. No local
//!                process, no local RAM. Load/unload stay honest:
//!                load just verifies the provider serves the model,
//!                unload refuses (Maya manages its own lifecycle).
//! Chat requests for a downloaded-but-unloaded local model are
//! JIT-loaded through the same dispatch, like LM Studio.
//!
//! Configuration via environment:
//!   QUACK_PORT          listen port          (default 1234)
//!   QUACK_MODELS_DIR    models directory     (default ~/.lmstudio/models)
//!   QUACK_BACKENDS_DIR  backend packs dir    (default ~/.lmstudio/extensions/backends)
//!   QUACK_LLAMA_SERVER  llama-server binary  (overrides pack discovery)
//!   QUACK_MLX_PYTHON    MLX python           (overrides oMLX discovery;
//!                        fallback ~/mlx-venv/bin/python)
//!   QUACK_MAYA_PROVIDERS  Maya providers as comma-separated
//!                        `name=url` pairs. Beats the settings file,
//!                        which beats the defaults (Duckets' six
//!                        per-provider endpoints on the Mac).
//!   QUACK_SETTINGS_DIR  settings directory (default
//!                        ~/.quackstudio; holds settings.json)

use axum::{
    body::Body,
    extract::{Query, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use quack_core::{
    default_maya_providers, find_llama_server, find_omlx_python, find_splash_pack, main_gguf_path,
    scan_library, Instance, MayaProvider, ModelEntry, ModelFormat,
};
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use tokio::process::{Child, Command};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;

/// First loopback port handed out to engine processes.
const FIRST_ENGINE_PORT: u16 = 18100;
/// How long a llama.cpp load waits for the engine's `/health`.
const LOAD_HEALTH_TIMEOUT: Duration = Duration::from_secs(180);
/// How long an MLX load waits for the engine to answer. MLX loads
/// are slower than llama.cpp's, so the budget is larger.
const MLX_HEALTH_TIMEOUT: Duration = Duration::from_secs(300);
/// How long a Splash load waits for `GET /ready`. LM Studio's own
/// host allows 10 minutes for a Splash load (a 27B package reads
/// ~17 GB of weights before the engine reports ready); match it.
const SPLASH_READY_TIMEOUT: Duration = Duration::from_secs(600);

struct LoadedInstance {
    instance: Instance,
    child: Child,
    /// Base URL the chat proxy posts to for this instance
    /// (`http://127.0.0.1:<port>` for local engines; the Maya
    /// resolution path builds the same shape for remote ones).
    base_url: String,
    /// Per-instance engine log (stdout+stderr) under the temp dir;
    /// its tail is reported when a load fails.
    #[allow(dead_code)]
    log_path: PathBuf,
    /// Size of the model's files on disk — the honest proxy for
    /// the memory a loaded instance holds (weights are mmap'd /
    /// loaded whole by both engines we drive).
    size_bytes: u64,
    /// Context length the instance was loaded with, when the
    /// caller specified one (llama.cpp only today).
    context_length: Option<u32>,
    /// Per-instance engine API key, sent as `Authorization:
    /// Bearer …` by the chat proxy (Splash pack only today;
    /// `None` for engines that take no key).
    api_key: Option<String>,
    /// Signal that asks this instance's engine to shut down
    /// gracefully: SIGTERM for llama.cpp / MLX, SIGINT for the
    /// Splash pack (its host's documented graceful signal; the
    /// server treats both the same, SIGINT is the contract).
    stop_signal: i32,
}

struct AppState {
    port: u16,
    models_dir: PathBuf,
    backends_dir: PathBuf,
    settings_dir: PathBuf,
    instances: RwLock<HashMap<String, LoadedInstance>>,
    next_port: AtomicU16,
    next_instance: AtomicU64,
    http: reqwest::Client,
    /// Effective Maya provider list; POST /api/v1/settings swaps
    /// it live, so it sits behind a lock unlike the static config.
    maya_providers: RwLock<Vec<MayaProvider>>,
    /// Where the effective list came from:
    /// "env" | "settings" | "default".
    maya_source: RwLock<&'static str>,
    maya_cache: RwLock<MayaCache>,
    /// In-flight and finished download jobs, by job id
    /// (`<repo>/<filename>`).
    downloads: RwLock<HashMap<String, Arc<RwLock<DownloadJob>>>>,
    /// Ring buffer of the daemon's own recent log lines (newest
    /// last), served at GET /api/v1/logs for the Model Server
    /// page's Server Logs strip.
    logs: RwLock<VecDeque<String>>,
    /// Persisted non-provider settings (server behaviour, general
    /// flags, model defaults) — the rest of settings.json beside
    /// the Maya provider list.
    persisted: RwLock<PersistedSettings>,
    /// Per-model configs (Library detail panel: load overrides,
    /// inference defaults, pins), persisted in the settings dir
    /// as `model-configs.json`, keyed by library model key.
    model_configs: RwLock<HashMap<String, ModelConfig>>,
}

/// How many daemon log lines the ring buffer keeps.
const LOG_CAP: usize = 200;

/// Append one line to the daemon log ring buffer (and echo it to
/// stdout, like the startup line always has).
async fn log_line(state: &Arc<AppState>, line: String) {
    println!("{line}");
    let mut logs = state.logs.write().await;
    if logs.len() >= LOG_CAP {
        logs.pop_front();
    }
    logs.push_back(line);
}

/// A library entry as served by `/api/v1/models`: the entry itself
/// plus its currently loaded instances.
#[derive(Serialize)]
struct ModelWithInstances {
    #[serde(flatten)]
    entry: ModelEntry,
    loaded_instances: Vec<Instance>,
}

fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(serde_json::json!({ "error": message.into() })),
    )
        .into_response()
}

fn error_body(message: String) -> serde_json::Value {
    serde_json::json!({ "error": message })
}

// ---------------------------------------------------------------------------
// Host facts (for /api/v1/status — the sidebar pills, LM Link page
// and Settings → Hardware all read these)
// ---------------------------------------------------------------------------

/// Read a string sysctl by name (macOS). `None` off macOS or on
/// any failure — callers substitute an honest placeholder.
#[cfg(target_os = "macos")]
fn sysctl_string(name: &str) -> Option<String> {
    let cname = std::ffi::CString::new(name).ok()?;
    let mut buf = vec![0u8; 256];
    let mut len = buf.len();
    // SAFETY: sysctlbyname with a valid name, a buffer of `len`
    // bytes, and no new value. Standard read-only query.
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || len == 0 {
        return None;
    }
    buf.truncate(len);
    if buf.last() == Some(&0) {
        buf.pop();
    }
    Some(String::from_utf8_lossy(&buf).into_owned())
}

#[cfg(not(target_os = "macos"))]
fn sysctl_string(_name: &str) -> Option<String> {
    None
}

/// Read a u64 sysctl by name (macOS `hw.memsize`).
#[cfg(target_os = "macos")]
fn sysctl_u64(name: &str) -> Option<u64> {
    let cname = std::ffi::CString::new(name).ok()?;
    let mut value: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: as above; the output slot is exactly one u64.
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            &mut value as *mut u64 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    Some(value)
}

#[cfg(not(target_os = "macos"))]
fn sysctl_u64(_name: &str) -> Option<u64> {
    None
}

/// This machine's hostname (`gethostname`), for the LM Link
/// "This device" card.
fn host_name() -> String {
    let mut buf = vec![0u8; 256];
    // SAFETY: gethostname writes at most `len` bytes into buf.
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
    if rc != 0 {
        return "unknown".to_string();
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// The daemon process's own peak RSS (`getrusage` ru_maxrss —
/// bytes on macOS). Peak, not current: getrusage offers nothing
/// else, and the field name in /api/v1/status says so.
fn daemon_peak_rss_bytes() -> u64 {
    // SAFETY: getrusage fills the struct; RUSAGE_SELF is valid.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) != 0 {
            return 0;
        }
        if usage.ru_maxrss < 0 {
            0
        } else if cfg!(target_os = "macos") {
            usage.ru_maxrss as u64
        } else {
            // Linux reports ru_maxrss in KiB.
            (usage.ru_maxrss as u64) * 1024
        }
    }
}

/// The host block of GET /api/v1/status.
fn host_info() -> serde_json::Value {
    serde_json::json!({
        "name": host_name(),
        "chip": sysctl_string("machdep.cpu.brand_string").unwrap_or_else(|| "unknown".to_string()),
        "gpu": "Apple Silicon (Metal)",
        "total_ram_bytes": sysctl_u64("hw.memsize").unwrap_or(0),
        "daemon_rss_bytes": daemon_peak_rss_bytes(),
        // No honest per-process CPU% without sampling state; the
        // UI shows the daemon RSS in the second pill instead.
        "cpu_percent": null,
    })
}

/// The packs block of GET /api/v1/status: what quackd found on
/// disk, in the shape Settings → Runtime renders.
fn packs_info(state: &AppState) -> serde_json::Value {
    let llama = match llama_server_binary(state) {
        Some(path) => path
            .parent()
            .and_then(|dir| dir.file_name())
            .map(|name| {
                name.to_string_lossy()
                    .rsplit('-')
                    .next()
                    .unwrap_or("")
                    .to_string()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "found".to_string()),
        None => "not found".to_string(),
    };
    let mlx = match mlx_python(state) {
        Some(path) => path
            .parent()
            .and_then(|bin| bin.parent())
            .and_then(|dir| dir.file_name())
            .map(|name| format!("omlx ({})", name.to_string_lossy()))
            .unwrap_or_else(|| "omlx".to_string()),
        None => "not found".to_string(),
    };
    let splash = match find_splash_pack(&state.backends_dir) {
        Some(pack) => pack.version,
        None => "not bundled".to_string(),
    };
    serde_json::json!({
        "llama": llama,
        "mlx": mlx,
        "splash": splash,
    })
}

// ---------------------------------------------------------------------------
// Maya pack (remote providers)
// ---------------------------------------------------------------------------

/// How long a provider's `/v1/models` result is reused before a
/// refetch. Keeps the library listing snappy and avoids hammering
/// Maya on every poll.
const MAYA_CACHE_TTL: Duration = Duration::from_secs(30);
/// Per-provider fetch budget: an unanswering provider must never
/// stall the whole library listing.
const MAYA_FETCH_TIMEOUT: Duration = Duration::from_millis(1500);

/// One Maya provider's last-known state. `models: None` means the
/// provider is not answering (offline, or not serving `/v1/models`)
/// and contributes nothing until the next successful fetch.
#[derive(Clone)]
struct MayaProviderStatus {
    name: String,
    base_url: String,
    port: u16,
    models: Option<Vec<String>>,
}

#[derive(Default)]
struct MayaCache {
    fetched_at: Option<Instant>,
    providers: Vec<MayaProviderStatus>,
}

/// The port a provider's base URL serves on (80 when the URL has
/// no explicit port).
fn provider_port(base_url: &str) -> u16 {
    let after_scheme = base_url.split("://").nth(1).unwrap_or(base_url);
    let host_port = after_scheme.split('/').next().unwrap_or("");
    host_port
        .rsplit(':')
        .next()
        .and_then(|p| p.parse().ok())
        .unwrap_or(80)
}

/// Fetch one provider's model list. Any failure (unreachable,
/// non-2xx, malformed body) yields `models: None` — never an
/// error that could take down the whole listing.
async fn fetch_provider_models(
    http: &reqwest::Client,
    provider: &MayaProvider,
) -> MayaProviderStatus {
    let mut status = MayaProviderStatus {
        name: provider.name.clone(),
        base_url: provider.base_url.clone(),
        port: provider_port(&provider.base_url),
        models: None,
    };
    let url = format!("{}/v1/models", provider.base_url.trim_end_matches('/'));
    let fetched = async {
        let resp = http
            .get(&url)
            .timeout(MAYA_FETCH_TIMEOUT)
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let value: serde_json::Value = resp.json().await.ok()?;
        let data = value.get("data")?.as_array()?;
        let mut ids = Vec::new();
        for item in data {
            if let Some(id) = item.get("id").and_then(|v| v.as_str()) {
                ids.push(id.to_string());
            }
        }
        Some(ids)
    }
    .await;
    status.models = fetched;
    status
}

/// Current per-provider state, from the 30 s cache when fresh.
async fn maya_snapshot(state: &Arc<AppState>) -> Vec<MayaProviderStatus> {
    {
        let cache = state.maya_cache.read().await;
        if let Some(fetched_at) = cache.fetched_at {
            if fetched_at.elapsed() < MAYA_CACHE_TTL {
                return cache.providers.clone();
            }
        }
    }
    let providers = state.maya_providers.read().await.clone();
    let fetches = providers
        .iter()
        .map(|provider| fetch_provider_models(&state.http, provider));
    let providers: Vec<MayaProviderStatus> = futures_util::future::join_all(fetches).await;
    let mut cache = state.maya_cache.write().await;
    cache.fetched_at = Some(Instant::now());
    cache.providers = providers.clone();
    providers
}

/// A Maya model as a library entry: key `maya/<provider>/<id>`,
/// always carrying one synthetic instance (the provider itself is
/// the running engine — pid 0, there is no local process).
fn maya_entry(status: &MayaProviderStatus, model_id: &str) -> ModelWithInstances {
    let key = format!("maya/{}/{}", status.name, model_id);
    ModelWithInstances {
        entry: ModelEntry {
            key: key.clone(),
            publisher: "maya".to_string(),
            display_name: model_id.to_string(),
            format: ModelFormat::Maya,
            size_bytes: 0,
            path: PathBuf::new(),
            arch: None,
            vision: false,
            file_name: None,
            quantization: None,
            max_context: None,
            domain: "llm".to_string(),
            warnings: Vec::new(),
            remote_url: Some(status.base_url.clone()),
        },
        loaded_instances: vec![Instance {
            instance_id: key.clone(),
            model_key: key,
            backend: format!("maya ({})", status.name),
            port: status.port,
            pid: 0,
        }],
    }
}

// ---------------------------------------------------------------------------
// Library / status
// ---------------------------------------------------------------------------

async fn list_models(State(state): State<Arc<AppState>>) -> Json<Vec<ModelWithInstances>> {
    let instances = state.instances.read().await;
    let mut models: Vec<ModelWithInstances> = scan_library(&state.models_dir)
        .into_iter()
        .map(|entry| ModelWithInstances {
            loaded_instances: instances
                .values()
                .filter(|loaded| loaded.instance.model_key == entry.key)
                .map(|loaded| loaded.instance.clone())
                .collect(),
            entry,
        })
        .collect();
    drop(instances);
    // Maya pack: merge each answering provider's models as remote
    // entries. An offline provider contributes nothing.
    for status in maya_snapshot(&state).await {
        if let Some(ids) = &status.models {
            for id in ids {
                models.push(maya_entry(&status, id));
            }
        }
    }
    Json(models)
}

async fn status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let model_count = scan_library(&state.models_dir).len();
    let instances = state.instances.read().await;
    let loaded_count = instances.len();
    // Sum of the loaded models' on-disk sizes: the honest proxy
    // for loaded weight memory (both engines load weights whole).
    let loaded_bytes: u64 = instances.values().map(|l| l.size_bytes).sum();
    let instance_list: Vec<serde_json::Value> = instances
        .values()
        .map(|l| {
            serde_json::json!({
                "instance_id": l.instance.instance_id,
                "model_key": l.instance.model_key,
                "backend": l.instance.backend,
                "port": l.instance.port,
                "pid": l.instance.pid,
                "context_length": l.context_length,
                "size_bytes": l.size_bytes,
            })
        })
        .collect();
    drop(instances);
    let snapshot = maya_snapshot(&state).await;
    let maya_online = snapshot.iter().filter(|s| s.models.is_some()).count();
    let mut host = host_info();
    host["loaded_bytes"] = serde_json::json!(loaded_bytes);
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "port": state.port,
        "models_dir": state.models_dir.to_string_lossy(),
        "backends_dir": state.backends_dir.to_string_lossy(),
        "model_count": model_count,
        "loaded_instances": loaded_count,
        "instances": instance_list,
        "maya_providers": state.maya_providers.read().await.len(),
        "maya_providers_online": maya_online,
        "host": host,
        "packs": packs_info(&state),
    }))
}

/// GET /api/v1/logs — the daemon's recent log lines (ring buffer,
/// oldest first), for the Model Server page's Server Logs strip.
async fn get_logs(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let logs = state.logs.read().await;
    Json(serde_json::json!({ "lines": logs.iter().collect::<Vec<_>>() }))
}

async fn openai_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    // OpenAI-compat lists *loaded* models only — local instances
    // plus Maya remotes (loaded by definition when answering).
    let instances = state.instances.read().await;
    let mut data: Vec<serde_json::Value> = instances
        .values()
        .map(|loaded| {
            serde_json::json!({
                "id": loaded.instance.model_key,
                "object": "model",
                "created": 0,
                "owned_by": "quackstudio",
            })
        })
        .collect();
    drop(instances);
    for status in maya_snapshot(&state).await {
        if let Some(ids) = &status.models {
            for id in ids {
                data.push(serde_json::json!({
                    "id": format!("maya/{}/{}", status.name, id),
                    "object": "model",
                    "created": 0,
                    "owned_by": "maya",
                }));
            }
        }
    }
    Json(serde_json::json!({ "object": "list", "data": data }))
}

// ---------------------------------------------------------------------------
// Load / unload
// ---------------------------------------------------------------------------

/// Resolve a load/chat model query: a full `publisher/name` key or
/// a bare model (directory) name.
fn resolve_model(state: &AppState, query: &str) -> Option<ModelEntry> {
    scan_library(&state.models_dir)
        .into_iter()
        .find(|e| e.key == query || e.display_name == query)
}

fn llama_server_binary(state: &AppState) -> Option<PathBuf> {
    if let Ok(path) = std::env::var("QUACK_LLAMA_SERVER") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    find_llama_server(&state.backends_dir)
}

/// Last `max_chars` characters of a text file ("" when unreadable).
fn log_tail(path: &PathBuf, max_chars: usize) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let start = text.char_indices().rev().nth(max_chars).map(|(i, _)| i).unwrap_or(0);
    text[start..].trim().to_string()
}

/// Terminate an engine child: SIGTERM, a short grace period, then
/// SIGKILL. Always reaps the child.
async fn terminate_child(child: &mut Child) {
    terminate_child_with(child, libc::SIGTERM).await;
}

/// Terminate an engine child with a chosen graceful signal, a
/// short grace period, then SIGKILL. Always reaps the child.
async fn terminate_child_with(child: &mut Child, signal: i32) {
    if let Some(pid) = child.id() {
        // SAFETY: kill(2) with a pid we spawned and have not reaped;
        // the worst case is ESRCH if it already exited.
        unsafe {
            libc::kill(pid as i32, signal);
        }
        for _ in 0..20 {
            match child.try_wait() {
                Ok(Some(_)) => return,
                _ => tokio::time::sleep(Duration::from_millis(100)).await,
            }
        }
    }
    let _ = child.kill().await;
    let _ = child.wait().await;
}

/// A failed load: HTTP status + the `{"error": …}` body.
type LoadError = (StatusCode, serde_json::Value);

fn load_error(status: StatusCode, message: String) -> LoadError {
    (status, error_body(message))
}

/// Create the per-instance engine log under the temp dir; returns
/// (stdout handle, stderr handle, path). Its tail is reported when
/// a load fails.
fn create_engine_log(
    tag: &str,
    entry: &ModelEntry,
    port: u16,
) -> Result<(std::fs::File, std::fs::File, PathBuf), LoadError> {
    let log_path = std::env::temp_dir().join(format!(
        "quackd-{tag}-{}-{}.log",
        entry.display_name.replace('/', "_"),
        port
    ));
    let log_file = std::fs::File::create(&log_path).map_err(|e| {
        load_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("cannot create engine log {}: {e}", log_path.display()),
        )
    })?;
    let log_file_err = log_file.try_clone().map_err(|e| {
        load_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("cannot clone engine log handle: {e}"),
        )
    })?;
    Ok((log_file, log_file_err, log_path))
}

/// Poll the engine's health endpoints until one answers 2xx.
/// Returns false on timeout, and fails fast if the child exits
/// first.
async fn wait_for_health(
    state: &AppState,
    child: &mut Child,
    port: u16,
    timeout: Duration,
    paths: &[&str],
) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return false;
        }
        for path in paths {
            let url = format!("http://127.0.0.1:{port}{path}");
            if let Ok(resp) = state.http.get(&url).send().await {
                if resp.status().is_success() {
                    return true;
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    false
}

/// Register a healthy engine child as a loaded instance.
#[allow(clippy::too_many_arguments)]
async fn register_instance(
    state: &Arc<AppState>,
    entry: &ModelEntry,
    backend: &str,
    port: u16,
    pid: u32,
    child: Child,
    log_path: PathBuf,
    context_length: Option<u32>,
    api_key: Option<String>,
    stop_signal: i32,
) -> Instance {
    let n = state.next_instance.fetch_add(1, Ordering::SeqCst) + 1;
    let instance = Instance {
        instance_id: format!("{}#{}", entry.key, n),
        model_key: entry.key.clone(),
        backend: backend.to_string(),
        port,
        pid,
    };
    state.instances.write().await.insert(
        instance.instance_id.clone(),
        LoadedInstance {
            instance: instance.clone(),
            child,
            base_url: format!("http://127.0.0.1:{port}"),
            log_path,
            size_bytes: entry.size_bytes,
            context_length,
            api_key,
            stop_signal,
        },
    );
    instance
}

/// Spawn the backend engine for a model, dispatching on format:
/// GGUF → llama.cpp, MLX → oMLX (`mlx_vlm.server`), Splash → the
/// Splash pack's `splash_engine` server. On success the instance
/// is registered and returned. Maya entries never reach here:
/// they are not in the local library scan, and their loads are
/// answered by `maya_load` before local resolution. The load
/// config applies to the GGUF spawner; the MLX and Splash
/// spawners have no equivalents and ignore it (their Library
/// panels label the fields as not applied).
async fn spawn_instance(
    state: &Arc<AppState>,
    entry: &ModelEntry,
    config: &StoredLoadConfig,
) -> Result<Instance, LoadError> {
    match entry.format {
        ModelFormat::Gguf => spawn_llama_instance(state, entry, config).await,
        ModelFormat::Mlx => spawn_mlx_instance(state, entry).await,
        ModelFormat::Splash => spawn_splash_instance(state, entry).await,
        ModelFormat::Maya => Err(load_error(
            StatusCode::BAD_REQUEST,
            "Maya models are served by Maya and cannot be loaded locally".to_string(),
        )),
    }
}

/// Split a raw arguments string the way a shell would (single /
/// double quotes, backslash escapes) — for the Library Load
/// tab's llama.cpp Arguments Override field.
fn shell_split(input: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut escaped = false;
    let mut started = false;
    for ch in input.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        match (quote, ch) {
            (_, '\\') => escaped = true,
            (Some(q), c) if c == q => quote = None,
            (Some(_), c) => current.push(c),
            (None, '\'' | '"') => {
                quote = Some(ch);
                started = true;
            }
            (None, c) if c.is_whitespace() => {
                if started || !current.is_empty() {
                    args.push(std::mem::take(&mut current));
                    started = false;
                }
            }
            (None, c) => current.push(c),
        }
    }
    if escaped {
        current.push('\\');
    }
    if started || !current.is_empty() {
        args.push(current);
    }
    args
}

/// Spawn llama-server for a GGUF entry and wait for its `/health`.
/// Every load-config field with a verified llama-server 2.80.1
/// flag is applied (see StoredLoadConfig's field notes); the
/// stored-only fields (num_experts, num_layers_offload,
/// moe_expert) never reach the command line.
async fn spawn_llama_instance(
    state: &Arc<AppState>,
    entry: &ModelEntry,
    cfg: &StoredLoadConfig,
) -> Result<Instance, LoadError> {
    let server = llama_server_binary(state).ok_or_else(|| {
        load_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "no llama.cpp backend pack with a llama-server binary under {}",
                state.backends_dir.display()
            ),
        )
    })?;
    let model_path = main_gguf_path(entry).ok_or_else(|| {
        load_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("no main GGUF file found under {}", entry.path.display()),
        )
    })?;

    // Speculative decoding: resolve the drafter model key to its
    // main GGUF file before spawning, so a bad key fails the load
    // with a clear message instead of a dead engine.
    let draft_path = match cfg.speculative_decoding.as_deref() {
        None | Some("") | Some("off") => None,
        Some(drafter_key) => {
            let drafter = resolve_model(state, drafter_key).ok_or_else(|| {
                load_error(
                    StatusCode::BAD_REQUEST,
                    format!("speculative decoding drafter not in the library: {drafter_key}"),
                )
            })?;
            Some(main_gguf_path(&drafter).ok_or_else(|| {
                load_error(
                    StatusCode::BAD_REQUEST,
                    format!("speculative decoding drafter {drafter_key} has no main GGUF file"),
                )
            })?)
        }
    };

    let port = state.next_port.fetch_add(1, Ordering::SeqCst);
    let (log_file, log_file_err, log_path) = create_engine_log("llama", entry, port)?;
    let context_length = cfg.context_length;

    let mut cmd = Command::new(&server);
    cmd.arg("--model")
        .arg(&model_path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--ctx-size")
        .arg(context_length.unwrap_or(8192).to_string())
        .arg("--n-gpu-layers")
        .arg(cfg.gpu_offload.unwrap_or(999).to_string())
        .arg("--jinja");
    // Advanced knobs (Library Load tab) — only when set, so
    // llama-server's own defaults rule otherwise. Flag names
    // verified against the installed llama-server 2.80.1 --help.
    if let Some(n) = cfg.cpu_threads {
        cmd.arg("-t").arg(n.to_string());
    }
    if let Some(n) = cfg.eval_batch_size {
        cmd.arg("-b").arg(n.to_string());
    }
    if let Some(n) = cfg.physical_batch_size {
        cmd.arg("-ub").arg(n.to_string());
    }
    if let Some(n) = cfg.max_concurrent {
        cmd.arg("--parallel").arg(n.to_string());
    }
    match cfg.unified_kv_cache {
        Some(true) => {
            cmd.arg("--kv-unified");
        }
        Some(false) => {
            cmd.arg("--no-kv-unified");
        }
        None => {}
    }
    if let Some(n) = cfg.context_checkpoints {
        cmd.arg("--ctx-checkpoints").arg(n.to_string());
    }
    if let Some(message) = &cfg.reasoning_budget_message {
        if !message.is_empty() {
            cmd.arg("--reasoning-budget-message").arg(message);
        }
    }
    if let Some(v) = cfg.rope_freq_base {
        cmd.arg("--rope-freq-base").arg(v.to_string());
    }
    if let Some(v) = cfg.rope_freq_scale {
        cmd.arg("--rope-freq-scale").arg(v.to_string());
    }
    match cfg.kv_offload {
        Some(true) => {
            cmd.arg("--kv-offload");
        }
        Some(false) => {
            cmd.arg("--no-kv-offload");
        }
        None => {}
    }
    // Keep-in-memory / try-mmap combine into --load-mode
    // (llama.cpp's current spelling of --mlock / --mmap). An
    // unset partner takes LM Studio's default (on).
    if cfg.keep_in_memory.is_some() || cfg.try_mmap.is_some() {
        let keep = cfg.keep_in_memory.unwrap_or(true);
        let mmap = cfg.try_mmap.unwrap_or(true);
        let mode = match (keep, mmap) {
            (true, true) => "mmap+mlock",
            (true, false) => "mlock",
            (false, true) => "mmap",
            (false, false) => "none",
        };
        cmd.arg("--load-mode").arg(mode);
    }
    if let Some(mode) = &cfg.on_demand_tensor_loading {
        if matches!(mode.as_str(), "auto" | "on" | "off") {
            cmd.arg("--lazy-mode").arg(mode);
        }
    }
    if let Some(seed) = cfg.seed {
        cmd.arg("--seed").arg(seed.to_string());
    }
    if let Some(draft) = &draft_path {
        cmd.arg("--model-draft").arg(draft);
    }
    if let Some(template) = &cfg.chat_template {
        if !template.is_empty() {
            cmd.arg("--chat-template").arg(template);
        }
    }
    if let Some(on) = cfg.flash_attention {
        cmd.arg("--flash-attn").arg(if on { "on" } else { "off" });
    }
    if let Some(t) = &cfg.cache_type_k {
        cmd.arg("--cache-type-k").arg(t);
    }
    if let Some(t) = &cfg.cache_type_v {
        cmd.arg("--cache-type-v").arg(t);
    }
    // The Arguments Override goes last, verbatim (shell-split),
    // so it can refine anything above it.
    if let Some(extra) = &cfg.args_override {
        for arg in shell_split(extra) {
            cmd.arg(arg);
        }
    }
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .spawn()
        .map_err(|e| {
            load_error(
                StatusCode::BAD_GATEWAY,
                format!("failed to spawn llama-server ({}): {e}", server.display()),
            )
        })?;
    let pid = child.id().unwrap_or(0);

    if !wait_for_health(state, &mut child, port, LOAD_HEALTH_TIMEOUT, &["/health"]).await {
        terminate_child(&mut child).await;
        return Err(load_error(
            StatusCode::BAD_GATEWAY,
            format!(
                "llama-server did not become healthy on port {port}; log tail: {}",
                log_tail(&log_path, 2000)
            ),
        ));
    }

    Ok(register_instance(
        state,
        entry,
        "llama.cpp",
        port,
        pid,
        child,
        log_path,
        context_length,
        None,
        libc::SIGTERM,
    )
    .await)
}

/// The MLX interpreter: `QUACK_MLX_PYTHON` override first, then
/// the newest oMLX build's Python (it imports `mlx_vlm`), then
/// the standalone `~/mlx-venv` fallback.
fn mlx_python(state: &AppState) -> Option<PathBuf> {
    if let Ok(path) = std::env::var("QUACK_MLX_PYTHON") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    if let Some(python) = find_omlx_python(&state.backends_dir) {
        return Some(python);
    }
    let venv_python = home_dir().join("mlx-venv").join("bin").join("python");
    if venv_python.is_file() {
        return Some(venv_python);
    }
    None
}

/// Spawn `mlx_vlm.server` (via the oMLX Python) for an MLX entry
/// and wait for it to answer. `context_length` / `gpu_offload`
/// have no `mlx_vlm.server` equivalent in P2 and are ignored.
async fn spawn_mlx_instance(
    state: &Arc<AppState>,
    entry: &ModelEntry,
) -> Result<Instance, LoadError> {
    let python = mlx_python(state).ok_or_else(|| {
        load_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "no MLX python found (no oMLX build under {}, no ~/mlx-venv); set QUACK_MLX_PYTHON",
                state.backends_dir.display()
            ),
        )
    })?;

    let port = state.next_port.fetch_add(1, Ordering::SeqCst);
    let (log_file, log_file_err, log_path) = create_engine_log("mlx", entry, port)?;

    let mut child = Command::new(&python)
        .arg("-m")
        .arg("mlx_vlm.server")
        .arg("--model")
        .arg(&entry.path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .spawn()
        .map_err(|e| {
            load_error(
                StatusCode::BAD_GATEWAY,
                format!("failed to spawn mlx_vlm.server ({}): {e}", python.display()),
            )
        })?;
    let pid = child.id().unwrap_or(0);

    // Current mlx_vlm.server builds expose /health; fall back to
    // the OpenAI model list on builds that don't.
    if !wait_for_health(
        state,
        &mut child,
        port,
        MLX_HEALTH_TIMEOUT,
        &["/health", "/v1/models"],
    )
    .await
    {
        terminate_child(&mut child).await;
        return Err(load_error(
            StatusCode::BAD_GATEWAY,
            format!(
                "mlx_vlm.server did not become healthy on port {port}; log tail: {}",
                log_tail(&log_path, 2000)
            ),
        ));
    }

    Ok(register_instance(
        state,
        entry,
        "mlx (omlx)",
        port,
        pid,
        child,
        log_path,
        None,
        None,
        libc::SIGTERM,
    )
    .await)
}

/// A random per-instance API key (32 bytes from `/dev/urandom`,
/// hex-encoded). The Splash server takes it via `SPLASH_API_KEY`
/// and the chat proxy presents it as a Bearer token.
fn random_api_key() -> Result<String, LoadError> {
    let mut buf = [0u8; 32];
    std::io::Read::read_exact(&mut std::fs::File::open("/dev/urandom").map_err(|e| {
        load_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("cannot open /dev/urandom for an instance API key: {e}"),
        )
    })?, &mut buf)
    .map_err(|e| {
        load_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("cannot read /dev/urandom for an instance API key: {e}"),
        )
    })?;
    Ok(buf.iter().map(|b| format!("{b:02x}")).collect())
}

/// Poll the Splash server's `GET /ready` (with the instance key)
/// until its JSON body reports `{"status":"ready"}`. Fails fast
/// if the child exits first.
async fn wait_for_splash_ready(
    state: &AppState,
    child: &mut Child,
    port: u16,
    api_key: &str,
    timeout: Duration,
) -> bool {
    let deadline = Instant::now() + timeout;
    let url = format!("http://127.0.0.1:{port}/ready");
    while Instant::now() < deadline {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return false;
        }
        if let Ok(resp) = state
            .http
            .get(&url)
            .header(header::AUTHORIZATION, format!("Bearer {api_key}"))
            .send()
            .await
        {
            if resp.status().is_success() {
                if let Ok(value) = resp.json::<serde_json::Value>().await {
                    if value.get("status").and_then(|v| v.as_str()) == Some("ready") {
                        return true;
                    }
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    false
}

/// Spawn the Splash pack's server for a Splash entry and wait
/// for readiness. The spawn mirrors LM Studio's own host
/// (`buildSpawnCommand` for pack ≥ 1.3.0, recovered in the P6
/// REA pass): the pack's amphibian Python runs `-m splash_engine`
/// with the package root positional, memory/context on `auto`
/// (the Library sliders are not applied — the engine sizes
/// itself), and a per-instance API key in the environment.
/// Readiness is `GET /ready`; the actualized context length is
/// then read from `GET /status` (`maximum_context_tokens`).
/// `context_length` / `gpu_offload` / knobs have no Splash
/// equivalents and are ignored, like MLX.
async fn spawn_splash_instance(
    state: &Arc<AppState>,
    entry: &ModelEntry,
) -> Result<Instance, LoadError> {
    let pack = find_splash_pack(&state.backends_dir).ok_or_else(|| {
        load_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "no Splash backend pack with a usable server under {}",
                state.backends_dir.display()
            ),
        )
    })?;
    let api_key = random_api_key()?;

    let port = state.next_port.fetch_add(1, Ordering::SeqCst);
    let (log_file, log_file_err, log_path) = create_engine_log("splash", entry, port)?;

    let mut child = Command::new(&pack.python)
        .arg("-I")
        .arg("-B")
        .arg("-u")
        .arg("-m")
        .arg("splash_engine")
        .arg(&entry.path)
        .arg("--no-webui")
        .arg("--tokenizer")
        .arg(entry.path.join("tokenizer"))
        .arg("--model")
        .arg(&entry.key)
        .arg("--binary")
        .arg(&pack.binary)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--max-memory")
        .arg("auto")
        .arg("--max-context")
        .arg("auto")
        .arg("--idle-release")
        .arg("off")
        .arg("--decode-share")
        .arg("0")
        .arg("--allow-idle-sleep")
        .env("SPLASH_API_KEY", &api_key)
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_VERBOSITY", "error")
        .env_remove("PYTHONHOME")
        .env_remove("PYTHONPATH")
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_file_err))
        .spawn()
        .map_err(|e| {
            load_error(
                StatusCode::BAD_GATEWAY,
                format!(
                    "failed to spawn splash_engine ({}): {e}",
                    pack.python.display()
                ),
            )
        })?;
    let pid = child.id().unwrap_or(0);

    if !wait_for_splash_ready(state, &mut child, port, &api_key, SPLASH_READY_TIMEOUT).await {
        terminate_child(&mut child).await;
        return Err(load_error(
            StatusCode::BAD_GATEWAY,
            format!(
                "splash_engine did not become ready on port {port}; log tail: {}",
                log_tail(&log_path, 2000)
            ),
        ));
    }

    // The actualized context length, per the host's own
    // handshake (GET /status → maximum_context_tokens).
    let context_length = async {
        let resp = state
            .http
            .get(format!("http://127.0.0.1:{port}/status"))
            .header(header::AUTHORIZATION, format!("Bearer {api_key}"))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let value = resp.json::<serde_json::Value>().await.ok()?;
        value
            .get("maximum_context_tokens")
            .and_then(|v| v.as_u64())
            .and_then(|n| u32::try_from(n).ok())
    }
    .await;

    Ok(register_instance(
        state,
        entry,
        "splash",
        port,
        pid,
        child,
        log_path,
        context_length,
        Some(api_key),
        libc::SIGINT,
    )
    .await)
}

#[derive(serde::Deserialize)]
struct LoadRequest {
    model: String,
    /// The request's load-config fields ride flattened alongside
    /// `model` (same snake_case names as the stored config).
    /// Fields present here override the model's stored config
    /// for this load; absent fields inherit it.
    #[serde(flatten)]
    config: StoredLoadConfig,
    /// Legacy alias for `cpu_threads` (the P5 UI's name).
    n_threads: Option<u32>,
}

/// Overlay `over` onto `base`: every field set in `over` wins.
/// Implemented through JSON so the field list lives in exactly
/// one place (the struct definition).
fn merge_load_config(base: &StoredLoadConfig, over: &StoredLoadConfig) -> StoredLoadConfig {
    let mut merged = serde_json::to_value(base).unwrap_or(serde_json::Value::Null);
    let over = serde_json::to_value(over).unwrap_or(serde_json::Value::Null);
    if let (Some(m), Some(o)) = (merged.as_object_mut(), over.as_object()) {
        for (key, value) in o {
            if !value.is_null() {
                m.insert(key.clone(), value.clone());
            }
        }
    }
    serde_json::from_value(merged).unwrap_or_default()
}

/// The effective load config for a model: its stored config with
/// the request's fields overlaid.
async fn effective_load_config(
    state: &Arc<AppState>,
    key: &str,
    request: Option<&StoredLoadConfig>,
) -> StoredLoadConfig {
    let stored = state
        .model_configs
        .read()
        .await
        .get(key)
        .map(|config| config.load.clone())
        .unwrap_or_default();
    match request {
        Some(over) => merge_load_config(&stored, over),
        None => stored,
    }
}

/// Load for a Maya key (`maya/<provider>/<model-id>`): there is
/// nothing to spawn — the model is loaded exactly when its
/// provider answers and lists it. 200 with the synthetic instance
/// then; 502 when the provider is not answering; 404 when the
/// provider is unknown or does not serve that model.
async fn maya_load(state: &Arc<AppState>, key: &str, rest: &str) -> Response {
    let Some((provider_name, model_id)) = rest.split_once('/') else {
        return error_response(
            StatusCode::NOT_FOUND,
            format!("malformed Maya model key: {key}"),
        );
    };
    let snapshot = maya_snapshot(state).await;
    let Some(provider) = snapshot.iter().find(|s| s.name == provider_name) else {
        return error_response(
            StatusCode::NOT_FOUND,
            format!("unknown Maya provider: {provider_name}"),
        );
    };
    match &provider.models {
        None => error_response(
            StatusCode::BAD_GATEWAY,
            format!(
                "Maya provider {provider_name} ({}) is not answering",
                provider.base_url
            ),
        ),
        Some(models) if models.iter().any(|m| m == model_id) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "type": "llm",
                "instance_id": key,
                "status": "loaded",
                "load_time_seconds": 0.0,
            })),
        )
            .into_response(),
        Some(_) => error_response(
            StatusCode::NOT_FOUND,
            format!("Maya provider {provider_name} does not serve {model_id}"),
        ),
    }
}

async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadRequest>,
) -> Response {
    // Maya pack: remote models have no local load; verify the
    // provider serves the model instead.
    if let Some(rest) = req.model.strip_prefix("maya/") {
        return maya_load(&state, &req.model, rest).await;
    }
    let Some(entry) = resolve_model(&state, &req.model) else {
        return error_response(
            StatusCode::NOT_FOUND,
            format!("model not found in the library: {}", req.model),
        );
    };
    // Format dispatch (GGUF / MLX / Splash) lives in
    // spawn_instance; the load config applies to GGUF loads and
    // is ignored by the MLX and Splash spawners.
    let mut request_config = req.config.clone();
    if request_config.cpu_threads.is_none() {
        request_config.cpu_threads = req.n_threads;
    }
    let config = effective_load_config(&state, &entry.key, Some(&request_config)).await;
    let started = Instant::now();
    match spawn_instance(&state, &entry, &config).await {
        Ok(instance) => {
            log_line(
                &state,
                format!(
                    "loaded {} as {} ({}) in {:.1}s",
                    entry.key,
                    instance.instance_id,
                    instance.backend,
                    started.elapsed().as_secs_f64()
                ),
            )
            .await;
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "type": "llm",
                    "instance_id": instance.instance_id,
                    "status": "loaded",
                    "load_time_seconds": started.elapsed().as_secs_f64(),
                })),
            )
                .into_response()
        }
        Err((status, body)) => {
            let message = body
                .get("error")
                .and_then(|v| v.as_str())
                .unwrap_or("load failed")
                .to_string();
            log_line(&state, format!("load failed for {}: {message}", entry.key)).await;
            (status, Json(body)).into_response()
        }
    }
}

#[derive(serde::Deserialize)]
struct UnloadRequest {
    instance_id: String,
}

async fn unload_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UnloadRequest>,
) -> Response {
    // Maya instances are synthetic — the model runs on Maya, not
    // here, so there is nothing of ours to stop. Refuse honestly
    // instead of faking an unload.
    if req.instance_id.starts_with("maya/") {
        return error_response(
            StatusCode::BAD_REQUEST,
            "Maya models are managed by Maya — unload it there",
        );
    }
    let loaded = state.instances.write().await.remove(&req.instance_id);
    match loaded {
        None => error_response(
            StatusCode::NOT_FOUND,
            format!("unknown instance: {}", req.instance_id),
        ),
        Some(mut loaded) => {
            terminate_child_with(&mut loaded.child, loaded.stop_signal).await;
            log_line(&state, format!("unloaded {}", req.instance_id)).await;
            (
                StatusCode::OK,
                Json(serde_json::json!({ "status": "unloaded" })),
            )
                .into_response()
        }
    }
}

// ---------------------------------------------------------------------------
// Discover (Hugging Face catalog proxy)
// ---------------------------------------------------------------------------

/// Budget for Hugging Face catalog calls.
const HF_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(serde::Deserialize)]
struct SearchParams {
    q: Option<String>,
}

/// GET /api/v1/discover/search?q=… — Hugging Face model search,
/// trimmed to the fields the Discover page renders
/// (`id`, `downloads`, `likes`, `pipeline_tag`).
async fn discover_search(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchParams>,
) -> Response {
    let query = params.q.unwrap_or_default();
    let resp = match state
        .http
        .get("https://huggingface.co/api/models")
        .query(&[
            ("search", query.as_str()),
            ("limit", "30"),
            ("sort", "downloads"),
            ("direction", "-1"),
        ])
        .timeout(HF_TIMEOUT)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => resp,
        Ok(resp) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("Hugging Face search failed: HTTP {}", resp.status()),
            )
        }
        Err(e) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("Hugging Face search failed: {e}"),
            )
        }
    };
    let value: serde_json::Value = match resp.json().await {
        Ok(value) => value,
        Err(e) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("Hugging Face search returned an unreadable body: {e}"),
            )
        }
    };
    let results: Vec<serde_json::Value> = value
        .as_array()
        .map(|items| {
            items
                .iter()
                .map(|item| {
                    serde_json::json!({
                        "id": item.get("id").cloned().unwrap_or(serde_json::Value::Null),
                        "downloads": item.get("downloads").cloned().unwrap_or(serde_json::Value::Null),
                        "likes": item.get("likes").cloned().unwrap_or(serde_json::Value::Null),
                        "pipeline_tag": item.get("pipeline_tag").cloned().unwrap_or(serde_json::Value::Null),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Json(results).into_response()
}

#[derive(serde::Deserialize)]
struct FilesParams {
    repo: Option<String>,
}

/// GET /api/v1/discover/files?repo=owner/name — the repo's `.gguf`
/// files, with sizes when Hugging Face reports them (`blobs=true`
/// asks for blob metadata; a missing size stays `null`).
async fn discover_files(
    State(state): State<Arc<AppState>>,
    Query(params): Query<FilesParams>,
) -> Response {
    let Some(repo) = params.repo.filter(|r| !r.trim().is_empty()) else {
        return error_response(StatusCode::BAD_REQUEST, "missing repo parameter");
    };
    let url = format!("https://huggingface.co/api/models/{repo}");
    let resp = match state
        .http
        .get(&url)
        .query(&[("blobs", "true")])
        .timeout(HF_TIMEOUT)
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => resp,
        Ok(resp) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("Hugging Face repo lookup failed: HTTP {}", resp.status()),
            )
        }
        Err(e) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("Hugging Face repo lookup failed: {e}"),
            )
        }
    };
    let value: serde_json::Value = match resp.json().await {
        Ok(value) => value,
        Err(e) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("Hugging Face repo lookup returned an unreadable body: {e}"),
            )
        }
    };
    let files: Vec<serde_json::Value> = value
        .get("siblings")
        .and_then(|siblings| siblings.as_array())
        .map(|siblings| {
            siblings
                .iter()
                .filter_map(|sibling| {
                    let filename = sibling.get("rfilename")?.as_str()?;
                    if !filename.to_lowercase().ends_with(".gguf") {
                        return None;
                    }
                    Some(serde_json::json!({
                        "filename": filename,
                        "size": sibling.get("size").cloned().unwrap_or(serde_json::Value::Null),
                    }))
                })
                .collect()
        })
        .unwrap_or_default();
    Json(files).into_response()
}

// ---------------------------------------------------------------------------
// Downloads
// ---------------------------------------------------------------------------

/// One model-file download job. Jobs live in memory only; a
/// finished file lands in the models directory and the library
/// scanner picks it up from there.
#[derive(Clone, Serialize)]
struct DownloadJob {
    /// `<repo>/<filename>`.
    id: String,
    repo: String,
    filename: String,
    bytes_done: u64,
    /// From the response's content-length, when the server sends
    /// one (Hugging Face's resolve endpoint does).
    bytes_total: Option<u64>,
    /// "downloading" | "done" | "error"
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(serde::Deserialize)]
struct DownloadRequest {
    repo: String,
    filename: String,
}

/// A repo id is exactly `owner/name`, both parts plain names —
/// anything else (extra slashes, `..`) is rejected so a download
/// can never write outside the models directory.
fn valid_repo(repo: &str) -> bool {
    let mut parts = repo.split('/');
    let owner = parts.next().unwrap_or("");
    let name = parts.next().unwrap_or("");
    parts.next().is_none()
        && !owner.is_empty()
        && !name.is_empty()
        && !owner.contains("..")
        && !name.contains("..")
}

/// A bare file name: no separators, no traversal.
fn valid_filename(filename: &str) -> bool {
    !filename.is_empty()
        && !filename.contains('/')
        && !filename.contains('\\')
        && !filename.contains("..")
}

/// POST /api/v1/models/download — start (or re-report) a
/// background download of one repo file into
/// `<models_dir>/<repo>/<filename>`.
async fn start_download(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DownloadRequest>,
) -> Response {
    let repo = req.repo.trim().to_string();
    let filename = req.filename.trim().to_string();
    if !valid_repo(&repo) {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!("invalid repo (want owner/name): {repo}"),
        );
    }
    if !valid_filename(&filename) {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!("invalid filename: {filename}"),
        );
    }
    let id = format!("{repo}/{filename}");
    // A job already downloading this file is returned as-is —
    // never two writers on one destination.
    {
        let jobs = state.downloads.read().await;
        if let Some(existing) = jobs.get(&id) {
            let job = existing.read().await;
            if job.status == "downloading" {
                return Json(job.clone()).into_response();
            }
        }
    }
    let job = Arc::new(RwLock::new(DownloadJob {
        id: id.clone(),
        repo,
        filename,
        bytes_done: 0,
        bytes_total: None,
        status: "downloading".to_string(),
        error: None,
    }));
    state.downloads.write().await.insert(id.clone(), job.clone());
    log_line(&state, format!("download started: {id}")).await;
    tokio::spawn(run_download(state.clone(), job.clone()));
    let snapshot = job.read().await.clone();
    Json(snapshot).into_response()
}

/// The download worker: stream the file to `<filename>.part`,
/// rename into place on success, and keep the job's counters
/// current as chunks land.
async fn run_download(state: Arc<AppState>, job: Arc<RwLock<DownloadJob>>) {
    let (repo, filename) = {
        let job = job.read().await;
        (job.repo.clone(), job.filename.clone())
    };
    let id = format!("{repo}/{filename}");
    match download_file(&state, &job, &repo, &filename).await {
        Ok(()) => {
            let mut job = job.write().await;
            job.status = "done".to_string();
            if job.bytes_total.is_none() {
                job.bytes_total = Some(job.bytes_done);
            }
            drop(job);
            log_line(&state, format!("download finished: {id}")).await;
        }
        Err(message) => {
            let mut job = job.write().await;
            job.status = "error".to_string();
            job.error = Some(message.clone());
            drop(job);
            log_line(&state, format!("download failed: {id}: {message}")).await;
        }
    }
}

async fn download_file(
    state: &Arc<AppState>,
    job: &Arc<RwLock<DownloadJob>>,
    repo: &str,
    filename: &str,
) -> Result<(), String> {
    let dest_dir = state.models_dir.join(repo);
    tokio::fs::create_dir_all(&dest_dir)
        .await
        .map_err(|e| format!("cannot create {}: {e}", dest_dir.display()))?;
    let part_path = dest_dir.join(format!("{filename}.part"));
    let final_path = dest_dir.join(filename);
    let url = format!("https://huggingface.co/{repo}/resolve/main/{filename}");
    let resp = state
        .http
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("download request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("download failed: HTTP {}", resp.status()));
    }
    job.write().await.bytes_total = resp.content_length();
    let mut file = tokio::fs::File::create(&part_path)
        .await
        .map_err(|e| format!("cannot create {}: {e}", part_path.display()))?;
    let mut stream = resp.bytes_stream();
    let mut written: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(e) => {
                let _ = tokio::fs::remove_file(&part_path).await;
                return Err(format!("download interrupted: {e}"));
            }
        };
        if let Err(e) = file.write_all(&chunk).await {
            let _ = tokio::fs::remove_file(&part_path).await;
            return Err(format!("cannot write {}: {e}", part_path.display()));
        }
        written += chunk.len() as u64;
        job.write().await.bytes_done = written;
    }
    // Close before the rename so no buffered handle straddles it.
    drop(file);
    tokio::fs::rename(&part_path, &final_path)
        .await
        .map_err(|e| format!("cannot move the finished download into place: {e}"))?;
    Ok(())
}

/// GET /api/v1/downloads — every known job, sorted by id.
async fn list_downloads(State(state): State<Arc<AppState>>) -> Json<Vec<DownloadJob>> {
    let jobs = state.downloads.read().await;
    let mut list: Vec<DownloadJob> = Vec::with_capacity(jobs.len());
    for job in jobs.values() {
        list.push(job.read().await.clone());
    }
    list.sort_by(|a, b| a.id.cmp(&b.id));
    Json(list)
}

// ---------------------------------------------------------------------------
// Per-model configs (Library detail panel)
// ---------------------------------------------------------------------------

/// Per-model load configuration: the Library Load tab's fields,
/// persisted in `model-configs.json` and sent (flattened) on load
/// requests. Every field is optional — `None` means "not set",
/// which is both the engine default at spawn time and "not
/// customized" in the UI. Flag mappings were verified against
/// the installed llama-server 2.80.1 `--help`; fields with no
/// honest flag stay stored-only and the UI labels them as such.
#[derive(Debug, Clone, Default, PartialEq, Serialize, serde::Deserialize)]
#[serde(default)]
struct StoredLoadConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    context_length: Option<u32>,
    /// GPU offload as a layer count (`--n-gpu-layers`).
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_offload: Option<u32>,
    /// `-t` CPU thread pool size.
    #[serde(skip_serializing_if = "Option::is_none")]
    cpu_threads: Option<u32>,
    /// `-b` evaluation (logical) batch size.
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_batch_size: Option<u32>,
    /// `-ub` physical batch size.
    #[serde(skip_serializing_if = "Option::is_none")]
    physical_batch_size: Option<u32>,
    /// `--parallel` server slots.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_concurrent: Option<u32>,
    /// `--kv-unified` / `--no-kv-unified`.
    #[serde(skip_serializing_if = "Option::is_none")]
    unified_kv_cache: Option<bool>,
    /// `--ctx-checkpoints`.
    #[serde(skip_serializing_if = "Option::is_none")]
    context_checkpoints: Option<u32>,
    /// `--reasoning-budget-message`.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_budget_message: Option<String>,
    /// `--rope-freq-base`.
    #[serde(skip_serializing_if = "Option::is_none")]
    rope_freq_base: Option<f32>,
    /// `--rope-freq-scale`.
    #[serde(skip_serializing_if = "Option::is_none")]
    rope_freq_scale: Option<f32>,
    /// KV cache offload to GPU (llama.cpp's default); `false`
    /// maps to `--no-kv-offload`.
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_offload: Option<bool>,
    /// Combined with `try_mmap` into `--load-mode` (mlock side).
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_in_memory: Option<bool>,
    /// Combined with `keep_in_memory` into `--load-mode`.
    #[serde(skip_serializing_if = "Option::is_none")]
    try_mmap: Option<bool>,
    /// `--lazy-mode`: "auto" | "on" | "off".
    #[serde(skip_serializing_if = "Option::is_none")]
    on_demand_tensor_loading: Option<String>,
    /// `--seed` (the UI sends it only while Seed is enabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    /// Stored only: llama.cpp's `--n-cpu-moe` counts *layers* of
    /// MoE weights kept on CPU, not experts — no honest mapping.
    #[serde(skip_serializing_if = "Option::is_none")]
    num_experts: Option<u32>,
    /// Stored only: GPU layer offload is `gpu_offload` above;
    /// this LM Studio field has no separate flag.
    #[serde(skip_serializing_if = "Option::is_none")]
    num_layers_offload: Option<u32>,
    /// Stored only: no llama.cpp equivalent.
    #[serde(skip_serializing_if = "Option::is_none")]
    moe_expert: Option<u32>,
    /// Drafter model key for `--model-draft`; "off"/empty = off.
    #[serde(skip_serializing_if = "Option::is_none")]
    speculative_decoding: Option<String>,
    /// `--chat-template`.
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template: Option<String>,
    /// `--flash-attn on|off`.
    #[serde(skip_serializing_if = "Option::is_none")]
    flash_attention: Option<bool>,
    /// `--cache-type-k` (llama.cpp type name, e.g. "q8_0").
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_type_k: Option<String>,
    /// `--cache-type-v`.
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_type_v: Option<String>,
    /// Raw extra llama-server arguments (shell-split, appended
    /// last) — the Arguments Override field.
    #[serde(skip_serializing_if = "Option::is_none")]
    args_override: Option<String>,
}

/// Per-model inference defaults: the Library Inference tab's
/// fields. The chat proxy merges these into requests for the
/// model, filling only fields the request itself leaves unset.
#[derive(Debug, Clone, Default, PartialEq, Serialize, serde::Deserialize)]
#[serde(default)]
struct InferenceConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    system_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_budget_enabled: Option<bool>,
    /// Stored only: the proxy has no per-request reasoning-budget
    /// field to fill on the engines we drive.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_budget: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    limit_response_length: Option<bool>,
    /// Sent as `max_tokens` when `limit_response_length` is on.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_response_tokens: Option<u32>,
    /// "truncate_middle" makes the proxy drop middle messages
    /// when a conversation outgrows the instance's context.
    #[serde(skip_serializing_if = "Option::is_none")]
    context_overflow: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    stop_strings: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_penalty_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_p_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_p: Option<f32>,
    /// When on, the proxy fills `response_format:
    /// {"type": "json_object"}` on requests that lack one.
    #[serde(skip_serializing_if = "Option::is_none")]
    structured_output: Option<bool>,
}

impl InferenceConfig {
    fn is_empty(&self) -> bool {
        *self == InferenceConfig::default()
    }
}

/// One model's persisted config: load overrides + inference
/// defaults + the Library pin.
#[derive(Debug, Clone, Default, Serialize, serde::Deserialize)]
#[serde(default)]
struct ModelConfig {
    load: StoredLoadConfig,
    inference: InferenceConfig,
    pinned: bool,
}

/// The config store at startup (`<settings_dir>/model-configs.json`).
/// A missing or corrupt file means an empty store.
fn read_model_configs(dir: &Path) -> HashMap<String, ModelConfig> {
    let Ok(text) = std::fs::read_to_string(dir.join("model-configs.json")) else {
        return HashMap::new();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

/// Persist the whole store. Called with the in-memory map already
/// updated; a failed write is reported to the caller.
async fn save_model_configs(state: &Arc<AppState>) -> Result<(), String> {
    let text = {
        let configs = state.model_configs.read().await;
        serde_json::to_string_pretty(&*configs).unwrap_or_default()
    };
    tokio::fs::create_dir_all(&state.settings_dir)
        .await
        .map_err(|e| format!("cannot create the settings directory: {e}"))?;
    tokio::fs::write(state.settings_dir.join("model-configs.json"), text)
        .await
        .map_err(|e| format!("cannot write model-configs.json: {e}"))
}

/// GET /api/v1/model-configs — every model's persisted config.
async fn get_model_configs(
    State(state): State<Arc<AppState>>,
) -> Json<HashMap<String, ModelConfig>> {
    Json(state.model_configs.read().await.clone())
}

#[derive(serde::Deserialize)]
struct ModelConfigRequest {
    key: String,
    load: Option<StoredLoadConfig>,
    inference: Option<InferenceConfig>,
    pinned: Option<bool>,
}

/// POST /api/v1/model-configs — replace the provided sections of
/// one model's config and persist. Answers with the model's
/// full updated config.
async fn update_model_config(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ModelConfigRequest>,
) -> Response {
    let key = req.key.trim().to_string();
    if key.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "model key must not be empty");
    }
    let updated = {
        let mut configs = state.model_configs.write().await;
        let entry = configs.entry(key.clone()).or_default();
        if let Some(load) = req.load {
            entry.load = load;
        }
        if let Some(inference) = req.inference {
            entry.inference = inference;
        }
        if let Some(pinned) = req.pinned {
            entry.pinned = pinned;
        }
        entry.clone()
    };
    if let Err(message) = save_model_configs(&state).await {
        return error_response(StatusCode::INTERNAL_SERVER_ERROR, message);
    }
    log_line(&state, format!("model config updated: {key}")).await;
    Json(updated).into_response()
}

#[derive(serde::Deserialize)]
struct RevealRequest {
    key: Option<String>,
    path: Option<String>,
}

/// POST /api/v1/models/reveal — reveal a model's folder (by key)
/// or the models directory itself (by path, e.g. from the
/// Library footer's directory menu) in Finder. Paths outside
/// the models directory are refused.
async fn reveal_path(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RevealRequest>,
) -> Response {
    let target: PathBuf = if let Some(key) = &req.key {
        if key.starts_with("maya/") {
            return error_response(
                StatusCode::BAD_REQUEST,
                "Maya models are remote — there is no local folder to reveal",
            );
        }
        match resolve_model(&state, key) {
            Some(entry) => entry.path,
            None => {
                return error_response(
                    StatusCode::NOT_FOUND,
                    format!("model not found in the library: {key}"),
                )
            }
        }
    } else if let Some(path) = &req.path {
        let root = state
            .models_dir
            .canonicalize()
            .unwrap_or_else(|_| state.models_dir.clone());
        let Ok(canon) = PathBuf::from(path).canonicalize() else {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("path does not exist: {path}"),
            );
        };
        if canon != root && !canon.starts_with(&root) {
            return error_response(
                StatusCode::FORBIDDEN,
                "only the models directory and its contents can be revealed",
            );
        }
        canon
    } else {
        return error_response(StatusCode::BAD_REQUEST, "provide a model key or a path");
    };
    #[cfg(target_os = "macos")]
    {
        match std::process::Command::new("open")
            .arg("-R")
            .arg(&target)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(_) => (
                StatusCode::OK,
                Json(serde_json::json!({ "status": "revealed" })),
            )
                .into_response(),
            Err(e) => error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to launch Finder: {e}"),
            ),
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = target;
        error_response(
            StatusCode::NOT_IMPLEMENTED,
            "Reveal in Finder is only available on macOS",
        )
    }
}

#[derive(serde::Deserialize)]
struct DeleteRequest {
    key: String,
}

/// POST /api/v1/models/delete — delete a local model's folder
/// from disk (the Library's More Actions → Delete, behind the
/// UI's confirm dialog). Refuses remote models, loaded models,
/// and anything whose on-disk path is not exactly the scanned
/// `<models_dir>/<publisher>/<model>` directory. The model's
/// persisted config goes with it.
async fn delete_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DeleteRequest>,
) -> Response {
    if req.key.starts_with("maya/") {
        return error_response(
            StatusCode::BAD_REQUEST,
            "Maya models are remote — there is nothing local to delete",
        );
    }
    let Some(entry) = resolve_model(&state, &req.key) else {
        return error_response(
            StatusCode::NOT_FOUND,
            format!("model not found in the library: {}", req.key),
        );
    };
    {
        let instances = state.instances.read().await;
        if instances
            .values()
            .any(|loaded| loaded.instance.model_key == entry.key)
        {
            return error_response(
                StatusCode::CONFLICT,
                format!("{} is loaded — unload it before deleting", entry.key),
            );
        }
    }
    // The path must be exactly the two-level library directory
    // the scanner found — never a caller-shaped path.
    let expected = state
        .models_dir
        .join(&entry.publisher)
        .join(&entry.display_name);
    let path_ok = entry.path == expected
        || entry.path.canonicalize().ok() == expected.canonicalize().ok();
    if !path_ok {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "refusing to delete {}: not a library model directory",
                entry.path.display()
            ),
        );
    }
    if let Err(e) = tokio::fs::remove_dir_all(&entry.path).await {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to delete {}: {e}", entry.path.display()),
        );
    }
    // The persisted config belongs to the deleted model.
    {
        let mut configs = state.model_configs.write().await;
        configs.remove(&entry.key);
    }
    let _ = save_model_configs(&state).await;
    log_line(&state, format!("deleted model {}", entry.key)).await;
    (
        StatusCode::OK,
        Json(serde_json::json!({ "status": "deleted" })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// Server behaviour knobs (Model Server page's Server Settings
/// popover + Settings → Developer). Stored in settings.json.
/// Of these, only `jit` is enforced today (it gates the chat
/// proxy's just-in-time load); the rest are recorded preferences
/// the daemon does not yet act on — auth/CORS are handled by the
/// permissive local-only listener, and idle-TTL eviction is not
/// implemented. The UI labels them as stored settings.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
#[serde(default)]
struct ServerSettings {
    require_auth: bool,
    serve_lan: bool,
    cors: bool,
    jit: bool,
    jit_ttl_minutes: u32,
    jit_only_last: bool,
}

impl Default for ServerSettings {
    fn default() -> Self {
        ServerSettings {
            require_auth: false,
            serve_lan: true,
            cors: true,
            jit: true,
            jit_ttl_minutes: 10,
            jit_only_last: false,
        }
    }
}

/// General app flags stored in settings.json.
#[derive(Debug, Clone, Default, Serialize, serde::Deserialize)]
#[serde(default)]
struct GeneralSettings {
    /// Use LM Studio's Hugging Face proxy for Discover/downloads.
    /// Stored only — quackd talks to huggingface.co directly.
    hf_proxy: bool,
}

/// Model defaults stored in settings.json. `context_length` is
/// wired: it seeds the Library Load tab's initial value.
#[derive(Debug, Clone, Default, Serialize, serde::Deserialize)]
#[serde(default)]
struct ModelDefaults {
    context_length: Option<u32>,
    image_max_px: Option<u32>,
}

/// The persisted settings beside the Maya provider list.
#[derive(Debug, Clone, Default)]
struct PersistedSettings {
    server: ServerSettings,
    general: GeneralSettings,
    defaults: ModelDefaults,
}

/// On-disk settings (`<settings_dir>/settings.json`). Every
/// field defaults so files written by older builds still parse.
#[derive(Serialize, serde::Deserialize, Default)]
struct SettingsFile {
    #[serde(default)]
    maya_providers: Vec<MayaProvider>,
    #[serde(default)]
    server: ServerSettings,
    #[serde(default)]
    general: GeneralSettings,
    #[serde(default)]
    defaults: ModelDefaults,
}

/// The settings directory: `QUACK_SETTINGS_DIR`, else
/// `~/.quackstudio`.
fn settings_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("QUACK_SETTINGS_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    home_dir().join(".quackstudio")
}

/// The whole settings file, when it exists and parses. Entries
/// were validated on save; a hand-corrupted file simply falls
/// back to the defaults.
fn read_settings_file(dir: &Path) -> Option<SettingsFile> {
    let text = std::fs::read_to_string(dir.join("settings.json")).ok()?;
    serde_json::from_str(&text).ok()
}

/// Providers persisted in the settings file. `Some` only when
/// the file exists and actually carries a provider list, so a
/// file written for other settings doesn't clobber the defaults
/// with an empty list.
fn read_settings_providers(dir: &Path) -> Option<Vec<MayaProvider>> {
    let file = read_settings_file(dir)?;
    if file.maya_providers.is_empty() {
        // Distinguish "no file" from "file with an empty list":
        // the raw text tells us whether the key was present.
        let text = std::fs::read_to_string(dir.join("settings.json")).ok()?;
        if !text.contains("maya_providers") {
            return None;
        }
    }
    Some(file.maya_providers)
}

/// Provider resolution at startup: `QUACK_MAYA_PROVIDERS` env
/// beats the settings file, which beats the built-in defaults.
fn resolve_maya_providers(dir: &Path) -> (Vec<MayaProvider>, &'static str) {
    if std::env::var("QUACK_MAYA_PROVIDERS").is_ok() {
        return (maya_providers_from_env(), "env");
    }
    if let Some(providers) = read_settings_providers(dir) {
        return (providers, "settings");
    }
    (default_maya_providers(), "default")
}

/// The non-provider persisted settings at startup.
fn resolve_persisted(dir: &Path) -> PersistedSettings {
    match read_settings_file(dir) {
        Some(file) => PersistedSettings {
            server: file.server,
            general: file.general,
            defaults: file.defaults,
        },
        None => PersistedSettings::default(),
    }
}

/// The settings view both settings endpoints answer with.
async fn settings_view(state: &Arc<AppState>) -> serde_json::Value {
    let providers = state.maya_providers.read().await.clone();
    let source = *state.maya_source.read().await;
    let persisted = state.persisted.read().await.clone();
    serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "port": state.port,
        "models_dir": state.models_dir.to_string_lossy(),
        "backends_dir": state.backends_dir.to_string_lossy(),
        "maya_providers": providers,
        "maya_providers_source": source,
        "server": persisted.server,
        "general": persisted.general,
        "defaults": persisted.defaults,
    })
}

/// GET /api/v1/settings — the effective settings.
async fn get_settings(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(settings_view(&state).await)
}

/// POST /api/v1/settings body: every section is optional; the
/// ones present are validated, persisted, and applied.
#[derive(serde::Deserialize)]
struct SettingsRequest {
    maya_providers: Option<Vec<MayaProvider>>,
    server: Option<ServerSettings>,
    general: Option<GeneralSettings>,
    defaults: Option<ModelDefaults>,
}

/// POST /api/v1/settings — validate, persist, and apply the
/// provided settings sections. The file is written before the
/// live state swaps, so a failed write can never desync memory
/// from storage.
async fn update_settings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SettingsRequest>,
) -> Response {
    // Validate + normalize the provider list when one was sent.
    let providers: Option<Vec<MayaProvider>> = match &req.maya_providers {
        None => None,
        Some(list) => {
            let mut normalized: Vec<MayaProvider> = Vec::with_capacity(list.len());
            for provider in list {
                let name = provider.name.trim();
                let base_url = provider.base_url.trim().trim_end_matches('/');
                if name.is_empty() {
                    return error_response(
                        StatusCode::BAD_REQUEST,
                        "provider names must not be empty",
                    );
                }
                let parsed = match reqwest::Url::parse(base_url) {
                    Ok(url) => url,
                    Err(_) => {
                        return error_response(
                            StatusCode::BAD_REQUEST,
                            format!("invalid provider URL: {}", provider.base_url),
                        )
                    }
                };
                if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
                    return error_response(
                        StatusCode::BAD_REQUEST,
                        format!(
                            "provider URL must be http(s) with a host: {}",
                            provider.base_url
                        ),
                    );
                }
                normalized.push(MayaProvider {
                    name: name.to_string(),
                    base_url: base_url.to_string(),
                });
            }
            Some(normalized)
        }
    };

    // Merge onto the current state: sections not sent keep their
    // live values, and the file keeps the provider list it had
    // when only other sections were posted.
    let persisted = state.persisted.read().await.clone();
    let current_providers = state.maya_providers.read().await.clone();
    let file = SettingsFile {
        maya_providers: providers.clone().unwrap_or(current_providers),
        server: req.server.clone().unwrap_or(persisted.server),
        general: req.general.clone().unwrap_or(persisted.general),
        defaults: req.defaults.clone().unwrap_or(persisted.defaults),
    };
    if let Err(e) = tokio::fs::create_dir_all(&state.settings_dir).await {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("cannot create the settings directory: {e}"),
        );
    }
    let text = serde_json::to_string_pretty(&file).unwrap_or_default();
    if let Err(e) = tokio::fs::write(state.settings_dir.join("settings.json"), text).await {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("cannot write settings: {e}"),
        );
    }

    *state.persisted.write().await = PersistedSettings {
        server: file.server.clone(),
        general: file.general.clone(),
        defaults: file.defaults.clone(),
    };
    if let Some(providers) = providers {
        *state.maya_providers.write().await = providers;
        *state.maya_source.write().await = "settings";
        // The new list invalidates every cached provider result.
        *state.maya_cache.write().await = MayaCache::default();
    }
    log_line(&state, "settings updated".to_string()).await;

    let mut view = settings_view(&state).await;
    if std::env::var("QUACK_MAYA_PROVIDERS").is_ok() {
        view["note"] = serde_json::Value::String(
            "Saved — but QUACK_MAYA_PROVIDERS is set and wins on the next daemon start until it is unset."
                .to_string(),
        );
    }
    Json(view).into_response()
}

// ---------------------------------------------------------------------------
// Chat proxy
// ---------------------------------------------------------------------------

/// Resolve a chat request naming a Maya model — by full entry
/// key (`maya/<provider>/<model-id>`) or by the provider's bare
/// model id — to `(provider base_url, provider model id, library
/// key)`. The model id is what the request body's `model` field
/// must carry for the provider; the library key is the model's
/// per-model config key. `None` when no answering provider
/// serves it.
async fn resolve_maya_target(
    state: &Arc<AppState>,
    requested: &str,
) -> Option<(String, String, String)> {
    let snapshot = maya_snapshot(state).await;
    if let Some(rest) = requested.strip_prefix("maya/") {
        let (provider_name, model_id) = rest.split_once('/')?;
        let provider = snapshot.iter().find(|s| s.name == provider_name)?;
        let models = provider.models.as_ref()?;
        if models.iter().any(|m| m == model_id) {
            return Some((
                provider.base_url.clone(),
                model_id.to_string(),
                requested.to_string(),
            ));
        }
        return None;
    }
    for provider in &snapshot {
        if let Some(models) = &provider.models {
            if models.iter().any(|m| m == requested) {
                return Some((
                    provider.base_url.clone(),
                    requested.to_string(),
                    format!("maya/{}/{}", provider.name, requested),
                ));
            }
        }
    }
    None
}

/// Rough token estimate for a message list (chars / 4 over the
/// string contents) — enough for the context-overflow check,
/// which only needs to know "clearly over", not an exact count.
fn estimate_tokens(messages: &[serde_json::Value]) -> u64 {
    let mut chars: u64 = 0;
    for message in messages {
        match message.get("content") {
            Some(serde_json::Value::String(text)) => {
                chars += text.chars().count() as u64;
            }
            Some(serde_json::Value::Array(parts)) => {
                for part in parts {
                    if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                        chars += text.chars().count() as u64;
                    }
                }
            }
            _ => {}
        }
    }
    chars / 4
}

/// Merge a model's stored inference defaults into a chat request
/// body, in place. Only fields the request itself leaves unset
/// are filled — a client's explicit values always win. Returns
/// true when anything changed (callers then re-serialize instead
/// of forwarding the raw body).
fn apply_inference_defaults(
    body: &mut serde_json::Value,
    cfg: &InferenceConfig,
    instance_context: Option<u32>,
) -> bool {
    if cfg.is_empty() {
        return false;
    }
    let Some(obj) = body.as_object_mut() else {
        return false;
    };
    fn fill(
        obj: &mut serde_json::Map<String, serde_json::Value>,
        key: &str,
        value: Option<serde_json::Value>,
    ) -> bool {
        match value {
            Some(value) if !obj.contains_key(key) => {
                obj.insert(key.to_string(), value);
                true
            }
            _ => false,
        }
    }
    let mut changed = false;
    changed |= fill(obj, "temperature", cfg.temperature.map(|v| serde_json::json!(v)));
    changed |= fill(
        obj,
        "top_p",
        cfg.top_p
            .filter(|_| cfg.top_p_enabled != Some(false))
            .map(|v| serde_json::json!(v)),
    );
    changed |= fill(obj, "top_k", cfg.top_k.map(|v| serde_json::json!(v)));
    changed |= fill(
        obj,
        "min_p",
        cfg.min_p
            .filter(|_| cfg.min_p_enabled != Some(false))
            .map(|v| serde_json::json!(v)),
    );
    changed |= fill(
        obj,
        "repeat_penalty",
        cfg.repeat_penalty
            .filter(|_| cfg.repeat_penalty_enabled == Some(true))
            .map(|v| serde_json::json!(v)),
    );
    changed |= fill(
        obj,
        "presence_penalty",
        cfg.presence_penalty
            .filter(|_| cfg.presence_penalty_enabled == Some(true))
            .map(|v| serde_json::json!(v)),
    );
    if cfg.limit_response_length == Some(true) {
        changed |= fill(
            obj,
            "max_tokens",
            cfg.max_response_tokens.map(|v| serde_json::json!(v)),
        );
    }
    if !cfg.stop_strings.is_empty() {
        changed |= fill(obj, "stop", Some(serde_json::json!(cfg.stop_strings)));
    }
    if cfg.structured_output == Some(true) {
        changed |= fill(
            obj,
            "response_format",
            Some(serde_json::json!({ "type": "json_object" })),
        );
    }

    // Message-level fills: the stored system prompt (only when
    // the conversation carries no system message of its own) and
    // Truncate-Middle context overflow.
    let prompt = cfg
        .system_prompt
        .as_deref()
        .filter(|p| !p.trim().is_empty());
    let truncate = cfg.context_overflow.as_deref() == Some("truncate_middle");
    if prompt.is_some() || truncate {
        if let Some(messages) = obj.get_mut("messages").and_then(|m| m.as_array_mut()) {
            if let Some(prompt) = prompt {
                let has_system = messages
                    .iter()
                    .any(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"));
                if !has_system {
                    messages.insert(
                        0,
                        serde_json::json!({ "role": "system", "content": prompt }),
                    );
                    changed = true;
                }
            }
            if truncate {
                if let Some(limit) = instance_context {
                    // Drop middle non-system messages (never the
                    // last one — the current turn) until the
                    // estimate fits. System messages stay.
                    let mut guard = 0;
                    while estimate_tokens(messages) > limit as u64 && guard < 512 {
                        guard += 1;
                        let droppable: Vec<usize> = messages
                            .iter()
                            .enumerate()
                            .filter(|(_, m)| {
                                m.get("role").and_then(|r| r.as_str()) != Some("system")
                            })
                            .map(|(i, _)| i)
                            .collect();
                        if droppable.len() <= 2 {
                            break;
                        }
                        // The middle of the droppable range,
                        // never its final element.
                        let victim = droppable[(droppable.len() - 1) / 2];
                        messages.remove(victim);
                        changed = true;
                    }
                }
            }
        }
    }
    changed
}

async fn chat_completions(State(state): State<Arc<AppState>>, body: axum::body::Bytes) -> Response {
    let parsed: serde_json::Value =
        serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null);
    let requested = parsed
        .get("model")
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    // The proxy target. Local instances forward the body
    // untouched (rewrite `None`); a Maya target rewrites `model`
    // to the provider's own id, since the client named our
    // `maya/<provider>/<id>` key. The API key is the instance's
    // per-load Bearer token (Splash pack), `None` for keyless
    // engines and Maya. `config_key` + `context_length` feed the
    // per-model inference-defaults merge below.
    struct ChatTarget {
        base_url: String,
        rewrite: Option<String>,
        api_key: Option<String>,
        config_key: Option<String>,
        context_length: Option<u32>,
    }
    let mut target: Option<ChatTarget> = None;

    // 1. A loaded local instance: exact model match (full key or
    //    bare name).
    {
        let instances = state.instances.read().await;
        if let Some(model) = &requested {
            target = instances
                .values()
                .find(|loaded| {
                    loaded.instance.model_key == *model
                        || loaded.instance.model_key.rsplit('/').next() == Some(model.as_str())
                })
                .map(|loaded| ChatTarget {
                    base_url: loaded.base_url.clone(),
                    rewrite: None,
                    api_key: loaded.api_key.clone(),
                    config_key: Some(loaded.instance.model_key.clone()),
                    context_length: loaded.context_length,
                });
        }
    }

    // 2. A Maya remote model (full key or bare provider model
    //    id). A `maya/…` name that resolves nowhere is a hard
    //    502: the client explicitly asked for a remote model
    //    whose provider is offline or does not serve it.
    if target.is_none() {
        if let Some(model) = &requested {
            if let Some((base_url, model_id, key)) = resolve_maya_target(&state, model).await {
                target = Some(ChatTarget {
                    base_url,
                    rewrite: Some(model_id),
                    api_key: None,
                    config_key: Some(key),
                    context_length: None,
                });
            } else if model.starts_with("maya/") {
                return error_response(
                    StatusCode::BAD_GATEWAY,
                    format!(
                        "Maya model {model} is unavailable: its provider is offline or does not serve it"
                    ),
                );
            }
        }
    }

    // 3. The sole loaded local instance, when there is only one.
    if target.is_none() {
        let instances = state.instances.read().await;
        if instances.len() == 1 {
            target = instances.values().next().map(|loaded| ChatTarget {
                base_url: loaded.base_url.clone(),
                rewrite: None,
                api_key: loaded.api_key.clone(),
                config_key: Some(loaded.instance.model_key.clone()),
                context_length: loaded.context_length,
            });
        }
    }

    // 4. JIT: a requested, downloaded-but-unloaded local model
    //    loads on demand through the same format dispatch as
    //    POST /api/v1/models/load. Maya models never JIT — they
    //    are already serving, and step 2 handled them. The Server
    //    Settings JIT toggle gates this step.
    if target.is_none() {
        if let Some(model) = &requested {
            if !state.persisted.read().await.server.jit {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "model {model} is not loaded and just-in-time loading is off (Server Settings)"
                    ),
                );
            }
            if let Some(entry) = resolve_model(&state, model) {
                // JIT loads use the model's stored load config —
                // the same config an explicit Library load uses.
                let config = effective_load_config(&state, &entry.key, None).await;
                match spawn_instance(&state, &entry, &config).await {
                    Ok(instance) => {
                        let loaded = state.instances.read().await;
                        let loaded = loaded.get(&instance.instance_id);
                        target = Some(ChatTarget {
                            base_url: format!("http://127.0.0.1:{}", instance.port),
                            rewrite: None,
                            api_key: loaded.and_then(|l| l.api_key.clone()),
                            config_key: Some(entry.key.clone()),
                            context_length: loaded.and_then(|l| l.context_length),
                        })
                    }
                    Err((status, body)) => return (status, Json(body)).into_response(),
                }
            }
        }
    }

    let Some(target) = target else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "no loaded instance matches the requested model; load a model first",
        );
    };
    let ChatTarget {
        base_url,
        rewrite,
        api_key,
        config_key,
        context_length,
    } = target;
    log_line(
        &state,
        format!(
            "chat → {} (model {})",
            base_url,
            requested.as_deref().unwrap_or("?")
        ),
    )
    .await;

    // Per-model inference defaults: fill whatever the request
    // leaves unset (temperature, sampling, stop strings, system
    // prompt, structured output, context overflow). When nothing
    // is filled and no rewrite is needed, the raw body forwards
    // byte-for-byte, exactly as before.
    let inference = match &config_key {
        Some(key) => state
            .model_configs
            .read()
            .await
            .get(key)
            .map(|config| config.inference.clone()),
        None => None,
    };
    let mut value = parsed.clone();
    let mut merged = false;
    if let Some(inference) = &inference {
        merged = apply_inference_defaults(&mut value, inference, context_length);
    }
    let out_body: Vec<u8> = match &rewrite {
        Some(model_id) => {
            value["model"] = serde_json::Value::String(model_id.clone());
            serde_json::to_vec(&value).unwrap_or_else(|_| body.to_vec())
        }
        None if merged => serde_json::to_vec(&value).unwrap_or_else(|_| body.to_vec()),
        None => body.to_vec(),
    };

    let upstream = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
    let mut request = state
        .http
        .post(&upstream)
        .header(header::CONTENT_TYPE, "application/json")
        .body(out_body);
    if let Some(key) = &api_key {
        request = request.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    let resp = match request.send().await {
        Ok(resp) => resp,
        Err(e) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("failed to reach the engine at {upstream}: {e}"),
            );
        }
    };

    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = resp.headers().get(header::CONTENT_TYPE).cloned();
    let stream = resp
        .bytes_stream()
        .map(|chunk| chunk.map_err(std::io::Error::other));
    let mut builder = Response::builder().status(status);
    if let Some(content_type) = content_type {
        builder = builder.header(header::CONTENT_TYPE, content_type);
    }
    builder
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| {
            error_response(StatusCode::BAD_GATEWAY, "failed to build proxy response")
        })
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn home_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
}

fn default_models_dir() -> PathBuf {
    home_dir().join(".lmstudio").join("models")
}

fn default_backends_dir() -> PathBuf {
    home_dir()
        .join(".lmstudio")
        .join("extensions")
        .join("backends")
}

/// Maya providers from `QUACK_MAYA_PROVIDERS` (comma-separated
/// `name=url` pairs); unparseable entries fall back to Duckets'
/// defaults. Malformed pairs are skipped, not fatal. Startup
/// resolution (env > settings file > defaults) lives in
/// `resolve_maya_providers`.
fn maya_providers_from_env() -> Vec<MayaProvider> {
    match std::env::var("QUACK_MAYA_PROVIDERS") {
        Ok(spec) => spec
            .split(',')
            .filter_map(|pair| {
                let (name, url) = pair.trim().split_once('=')?;
                let name = name.trim();
                let url = url.trim().trim_end_matches('/');
                if name.is_empty() || url.is_empty() {
                    return None;
                }
                Some(MayaProvider {
                    name: name.to_string(),
                    base_url: url.to_string(),
                })
            })
            .collect(),
        Err(_) => default_maya_providers(),
    }
}

#[tokio::main]
async fn main() {
    let port: u16 = std::env::var("QUACK_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1234);
    let models_dir = std::env::var("QUACK_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_models_dir());
    let backends_dir = std::env::var("QUACK_BACKENDS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_backends_dir());
    let settings_dir = settings_dir();
    let (maya_providers, maya_source) = resolve_maya_providers(&settings_dir);
    let persisted = resolve_persisted(&settings_dir);
    let model_configs = read_model_configs(&settings_dir);

    let state = Arc::new(AppState {
        port,
        models_dir,
        backends_dir,
        settings_dir,
        instances: RwLock::new(HashMap::new()),
        next_port: AtomicU16::new(FIRST_ENGINE_PORT),
        next_instance: AtomicU64::new(0),
        http: reqwest::Client::new(),
        maya_providers: RwLock::new(maya_providers),
        maya_source: RwLock::new(maya_source),
        maya_cache: RwLock::new(MayaCache::default()),
        downloads: RwLock::new(HashMap::new()),
        logs: RwLock::new(VecDeque::new()),
        persisted: RwLock::new(persisted),
        model_configs: RwLock::new(model_configs),
    });
    let app = Router::new()
        .route("/api/v1/models", get(list_models))
        .route("/api/v1/status", get(status))
        .route("/api/v1/logs", get(get_logs))
        .route("/api/v1/models/load", post(load_model))
        .route("/api/v1/models/unload", post(unload_model))
        .route(
            "/api/v1/model-configs",
            get(get_model_configs).post(update_model_config),
        )
        .route("/api/v1/models/reveal", post(reveal_path))
        .route("/api/v1/models/delete", post(delete_model))
        .route("/api/v1/discover/search", get(discover_search))
        .route("/api/v1/discover/files", get(discover_files))
        .route("/api/v1/models/download", post(start_download))
        .route("/api/v1/downloads", get(list_downloads))
        .route("/api/v1/settings", get(get_settings).post(update_settings))
        .route("/v1/models", get(openai_models))
        .route("/v1/chat/completions", post(chat_completions))
        // The Tauri dev UI (localhost:5173) talks to us cross-origin.
        .layer(CorsLayer::permissive())
        .with_state(state.clone());

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("quackd: failed to bind listen address");
    log_line(
        &state,
        format!(
            "quackd {} listening on http://{addr} (models dir: {}, backends dir: {})",
            env!("CARGO_PKG_VERSION"),
            state.models_dir.display(),
            state.backends_dir.display()
        ),
    )
    .await;

    // On Ctrl-C, stop every engine child before exiting so no
    // llama-server is ever orphaned holding GPU memory.
    let shutdown_state = state.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = tokio::signal::ctrl_c().await;
            let mut instances = shutdown_state.instances.write().await;
            for (_, mut loaded) in instances.drain() {
                terminate_child_with(&mut loaded.child, loaded.stop_signal).await;
            }
        })
        .await
        .expect("quackd: server error");
}
