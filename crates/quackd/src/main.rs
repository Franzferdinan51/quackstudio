//! quackd — the QuackStudio daemon.
//!
//! P3 surface (see ../DESIGN.md):
//!   GET  /api/v1/models          library index + loaded instances,
//!                                with Maya remote models merged in
//!   GET  /api/v1/status          daemon status
//!   POST /api/v1/models/load     spawn a backend engine for a model
//!   POST /api/v1/models/unload   stop an instance by id
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
//!   Splash     → 501 until the Splash/Strata pack lands.
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
//!                        `name=url` pairs (default: Duckets' six
//!                        per-provider endpoints on the Mac)

use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use quack_core::{
    default_maya_providers, find_llama_server, find_omlx_python, main_gguf_path, scan_library,
    Instance, MayaProvider, ModelEntry, ModelFormat,
};
use serde::Serialize;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
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
}

struct AppState {
    port: u16,
    models_dir: PathBuf,
    backends_dir: PathBuf,
    instances: RwLock<HashMap<String, LoadedInstance>>,
    next_port: AtomicU16,
    next_instance: AtomicU64,
    http: reqwest::Client,
    maya_providers: Vec<MayaProvider>,
    maya_cache: RwLock<MayaCache>,
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
    let fetches = state
        .maya_providers
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
    let loaded_count = state.instances.read().await.len();
    let snapshot = maya_snapshot(&state).await;
    let maya_online = snapshot.iter().filter(|s| s.models.is_some()).count();
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "port": state.port,
        "models_dir": state.models_dir.to_string_lossy(),
        "backends_dir": state.backends_dir.to_string_lossy(),
        "model_count": model_count,
        "loaded_instances": loaded_count,
        "maya_providers": state.maya_providers.len(),
        "maya_providers_online": maya_online,
    }))
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
    if let Some(pid) = child.id() {
        // SAFETY: kill(2) with a pid we spawned and have not reaped;
        // the worst case is ESRCH if it already exited.
        unsafe {
            libc::kill(pid as i32, libc::SIGTERM);
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
async fn register_instance(
    state: &Arc<AppState>,
    entry: &ModelEntry,
    backend: &str,
    port: u16,
    pid: u32,
    child: Child,
    log_path: PathBuf,
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
        },
    );
    instance
}

/// Spawn the backend engine for a model, dispatching on format:
/// GGUF → llama.cpp, MLX → oMLX (`mlx_vlm.server`), Splash → 501
/// until the Splash/Strata pack lands. On success the instance
/// is registered and returned. Maya entries never reach here:
/// they are not in the local library scan, and their loads are
/// answered by `maya_load` before local resolution.
async fn spawn_instance(
    state: &Arc<AppState>,
    entry: &ModelEntry,
    context_length: Option<u32>,
    gpu_offload: Option<u32>,
) -> Result<Instance, LoadError> {
    match entry.format {
        ModelFormat::Gguf => spawn_llama_instance(state, entry, context_length, gpu_offload).await,
        ModelFormat::Mlx => spawn_mlx_instance(state, entry).await,
        ModelFormat::Splash => Err(load_error(
            StatusCode::NOT_IMPLEMENTED,
            "Splash/Strata pack lands in P3".to_string(),
        )),
        ModelFormat::Maya => Err(load_error(
            StatusCode::BAD_REQUEST,
            "Maya models are served by Maya and cannot be loaded locally".to_string(),
        )),
    }
}

/// Spawn llama-server for a GGUF entry and wait for its `/health`.
async fn spawn_llama_instance(
    state: &Arc<AppState>,
    entry: &ModelEntry,
    context_length: Option<u32>,
    gpu_offload: Option<u32>,
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

    let port = state.next_port.fetch_add(1, Ordering::SeqCst);
    let (log_file, log_file_err, log_path) = create_engine_log("llama", entry, port)?;

    let mut child = Command::new(&server)
        .arg("--model")
        .arg(&model_path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--ctx-size")
        .arg(context_length.unwrap_or(8192).to_string())
        .arg("--n-gpu-layers")
        .arg(gpu_offload.unwrap_or(999).to_string())
        .arg("--jinja")
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

    Ok(register_instance(state, entry, "llama.cpp", port, pid, child, log_path).await)
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

    Ok(register_instance(state, entry, "mlx (omlx)", port, pid, child, log_path).await)
}

#[derive(serde::Deserialize)]
struct LoadRequest {
    model: String,
    context_length: Option<u32>,
    /// Passed to llama-server as `--n-gpu-layers` (default 999 =
    /// offload everything, LM Studio's Metal behaviour).
    gpu_offload: Option<f64>,
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
    // Format dispatch (GGUF / MLX / Splash-501) lives in
    // spawn_instance; context_length and gpu_offload are GGUF-only
    // knobs today and are ignored by the MLX spawner.
    let gpu_offload = req.gpu_offload.map(|v| v.round() as u32);
    let started = Instant::now();
    match spawn_instance(&state, &entry, req.context_length, gpu_offload).await {
        Ok(instance) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "type": "llm",
                "instance_id": instance.instance_id,
                "status": "loaded",
                "load_time_seconds": started.elapsed().as_secs_f64(),
            })),
        )
            .into_response(),
        Err((status, body)) => (status, Json(body)).into_response(),
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
            terminate_child(&mut loaded.child).await;
            (
                StatusCode::OK,
                Json(serde_json::json!({ "status": "unloaded" })),
            )
                .into_response()
        }
    }
}

// ---------------------------------------------------------------------------
// Chat proxy
// ---------------------------------------------------------------------------

/// Resolve a chat request naming a Maya model — by full entry
/// key (`maya/<provider>/<model-id>`) or by the provider's bare
/// model id — to `(provider base_url, provider model id)`. The
/// model id is what the request body's `model` field must carry
/// for the provider. `None` when no answering provider serves it.
async fn resolve_maya_target(
    state: &Arc<AppState>,
    requested: &str,
) -> Option<(String, String)> {
    let snapshot = maya_snapshot(state).await;
    if let Some(rest) = requested.strip_prefix("maya/") {
        let (provider_name, model_id) = rest.split_once('/')?;
        let provider = snapshot.iter().find(|s| s.name == provider_name)?;
        let models = provider.models.as_ref()?;
        if models.iter().any(|m| m == model_id) {
            return Some((provider.base_url.clone(), model_id.to_string()));
        }
        return None;
    }
    for provider in &snapshot {
        if let Some(models) = &provider.models {
            if models.iter().any(|m| m == requested) {
                return Some((provider.base_url.clone(), requested.to_string()));
            }
        }
    }
    None
}

async fn chat_completions(State(state): State<Arc<AppState>>, body: axum::body::Bytes) -> Response {
    let parsed: serde_json::Value =
        serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null);
    let requested = parsed
        .get("model")
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    // The proxy target: (base_url, model-id rewrite). Local
    // instances forward the body untouched (rewrite `None`); a
    // Maya target rewrites `model` to the provider's own id,
    // since the client named our `maya/<provider>/<id>` key.
    let mut target: Option<(String, Option<String>)> = None;

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
                .map(|loaded| (loaded.base_url.clone(), None));
        }
    }

    // 2. A Maya remote model (full key or bare provider model
    //    id). A `maya/…` name that resolves nowhere is a hard
    //    502: the client explicitly asked for a remote model
    //    whose provider is offline or does not serve it.
    if target.is_none() {
        if let Some(model) = &requested {
            if let Some((base_url, model_id)) = resolve_maya_target(&state, model).await {
                target = Some((base_url, Some(model_id)));
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
            target = instances
                .values()
                .next()
                .map(|loaded| (loaded.base_url.clone(), None));
        }
    }

    // 4. JIT: a requested, downloaded-but-unloaded local model
    //    loads on demand through the same format dispatch as
    //    POST /api/v1/models/load (a Splash model surfaces the
    //    dispatch's 501 here). Maya models never JIT — they are
    //    already serving, and step 2 handled them.
    if target.is_none() {
        if let Some(model) = &requested {
            if let Some(entry) = resolve_model(&state, model) {
                match spawn_instance(&state, &entry, None, None).await {
                    Ok(instance) => {
                        target = Some((format!("http://127.0.0.1:{}", instance.port), None))
                    }
                    Err((status, body)) => return (status, Json(body)).into_response(),
                }
            }
        }
    }

    let Some((base_url, rewrite)) = target else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "no loaded instance matches the requested model; load a model first",
        );
    };

    let out_body: Vec<u8> = match &rewrite {
        Some(model_id) => {
            let mut value = parsed.clone();
            value["model"] = serde_json::Value::String(model_id.clone());
            serde_json::to_vec(&value).unwrap_or_else(|_| body.to_vec())
        }
        None => body.to_vec(),
    };

    let upstream = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
    let resp = match state
        .http
        .post(&upstream)
        .header(header::CONTENT_TYPE, "application/json")
        .body(out_body)
        .send()
        .await
    {
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
/// `name=url` pairs); unset or unparseable entries fall back to
/// Duckets' defaults. Malformed pairs are skipped, not fatal.
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

    let state = Arc::new(AppState {
        port,
        models_dir,
        backends_dir,
        instances: RwLock::new(HashMap::new()),
        next_port: AtomicU16::new(FIRST_ENGINE_PORT),
        next_instance: AtomicU64::new(0),
        http: reqwest::Client::new(),
        maya_providers: maya_providers_from_env(),
        maya_cache: RwLock::new(MayaCache::default()),
    });
    let app = Router::new()
        .route("/api/v1/models", get(list_models))
        .route("/api/v1/status", get(status))
        .route("/api/v1/models/load", post(load_model))
        .route("/api/v1/models/unload", post(unload_model))
        .route("/v1/models", get(openai_models))
        .route("/v1/chat/completions", post(chat_completions))
        // The Tauri dev UI (localhost:5173) talks to us cross-origin.
        .layer(CorsLayer::permissive())
        .with_state(state.clone());

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("quackd: failed to bind listen address");
    println!(
        "quackd {} listening on http://{addr} (models dir: {}, backends dir: {})",
        env!("CARGO_PKG_VERSION"),
        state.models_dir.display(),
        state.backends_dir.display()
    );

    // On Ctrl-C, stop every engine child before exiting so no
    // llama-server is ever orphaned holding GPU memory.
    let shutdown_state = state.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = tokio::signal::ctrl_c().await;
            let mut instances = shutdown_state.instances.write().await;
            for (_, mut loaded) in instances.drain() {
                terminate_child(&mut loaded.child).await;
            }
        })
        .await
        .expect("quackd: server error");
}
