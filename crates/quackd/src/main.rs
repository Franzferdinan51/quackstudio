//! quackd — the QuackStudio daemon.
//!
//! P1 surface (see ../DESIGN.md):
//!   GET  /api/v1/models          library index + loaded instances
//!   GET  /api/v1/status          daemon status
//!   POST /api/v1/models/load     spawn a backend engine (llama.cpp in P1)
//!   POST /api/v1/models/unload   stop an instance by id
//!   GET  /v1/models              OpenAI-compat list (loaded only)
//!   POST /v1/chat/completions    proxy to the instance's engine (SSE passthrough)
//!
//! The llama.cpp pack is LM Studio's own: quackd finds the newest
//! `llama-server` under the backends dir and runs it as a child
//! process on a loopback port, exactly one child per loaded
//! instance. Chat requests for a downloaded-but-unloaded GGUF are
//! JIT-loaded through the same path, like LM Studio.
//!
//! Configuration via environment:
//!   QUACK_PORT          listen port          (default 1234)
//!   QUACK_MODELS_DIR    models directory     (default ~/.lmstudio/models)
//!   QUACK_BACKENDS_DIR  backend packs dir    (default ~/.lmstudio/extensions/backends)
//!   QUACK_LLAMA_SERVER  llama-server binary  (overrides pack discovery)

use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use quack_core::{find_llama_server, main_gguf_path, scan_library, Instance, ModelEntry, ModelFormat};
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
/// How long a load waits for the engine's `/health` to answer.
const LOAD_HEALTH_TIMEOUT: Duration = Duration::from_secs(180);

struct LoadedInstance {
    instance: Instance,
    child: Child,
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
// Library / status
// ---------------------------------------------------------------------------

async fn list_models(State(state): State<Arc<AppState>>) -> Json<Vec<ModelWithInstances>> {
    let instances = state.instances.read().await;
    let models = scan_library(&state.models_dir)
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
    Json(models)
}

async fn status(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let model_count = scan_library(&state.models_dir).len();
    let loaded_count = state.instances.read().await.len();
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "port": state.port,
        "models_dir": state.models_dir.to_string_lossy(),
        "backends_dir": state.backends_dir.to_string_lossy(),
        "model_count": model_count,
        "loaded_instances": loaded_count,
    }))
}

async fn openai_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    // OpenAI-compat lists *loaded* models only.
    let instances = state.instances.read().await;
    let data: Vec<serde_json::Value> = instances
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

/// Spawn llama-server for a GGUF entry and wait for its `/health`.
/// On success the instance is registered and returned.
async fn spawn_instance(
    state: &Arc<AppState>,
    entry: &ModelEntry,
    context_length: Option<u32>,
    gpu_offload: Option<u32>,
) -> Result<Instance, (StatusCode, serde_json::Value)> {
    let err = |status: StatusCode, message: String| (status, error_body(message));

    if entry.format != ModelFormat::Gguf {
        return Err(err(
            StatusCode::NOT_IMPLEMENTED,
            "MLX pack lands in P2".to_string(),
        ));
    }
    let server = llama_server_binary(state).ok_or_else(|| {
        err(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "no llama.cpp backend pack with a llama-server binary under {}",
                state.backends_dir.display()
            ),
        )
    })?;
    let model_path = main_gguf_path(entry).ok_or_else(|| {
        err(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("no main GGUF file found under {}", entry.path.display()),
        )
    })?;

    let port = state.next_port.fetch_add(1, Ordering::SeqCst);
    let log_path = std::env::temp_dir().join(format!(
        "quackd-llama-{}-{}.log",
        entry.display_name.replace('/', "_"),
        port
    ));
    let log_file = std::fs::File::create(&log_path).map_err(|e| {
        err(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("cannot create engine log {}: {e}", log_path.display()),
        )
    })?;
    let log_file_err = log_file.try_clone().map_err(|e| {
        err(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("cannot clone engine log handle: {e}"),
        )
    })?;

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
            err(
                StatusCode::BAD_GATEWAY,
                format!("failed to spawn llama-server ({}): {e}", server.display()),
            )
        })?;
    let pid = child.id().unwrap_or(0);

    // Poll /health until the engine is ready; fail fast if the
    // process exits first.
    let health_url = format!("http://127.0.0.1:{port}/health");
    let deadline = Instant::now() + LOAD_HEALTH_TIMEOUT;
    let mut healthy = false;
    while Instant::now() < deadline {
        if matches!(child.try_wait(), Ok(Some(_))) {
            break;
        }
        if let Ok(resp) = state.http.get(&health_url).send().await {
            if resp.status().is_success() {
                healthy = true;
                break;
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    if !healthy {
        terminate_child(&mut child).await;
        return Err(err(
            StatusCode::BAD_GATEWAY,
            format!(
                "llama-server did not become healthy on port {port}; log tail: {}",
                log_tail(&log_path, 2000)
            ),
        ));
    }

    let n = state.next_instance.fetch_add(1, Ordering::SeqCst) + 1;
    let instance = Instance {
        instance_id: format!("{}#{}", entry.key, n),
        model_key: entry.key.clone(),
        backend: "llama.cpp".to_string(),
        port,
        pid,
    };
    state.instances.write().await.insert(
        instance.instance_id.clone(),
        LoadedInstance {
            instance: instance.clone(),
            child,
            log_path,
        },
    );
    Ok(instance)
}

#[derive(serde::Deserialize)]
struct LoadRequest {
    model: String,
    context_length: Option<u32>,
    /// Passed to llama-server as `--n-gpu-layers` (default 999 =
    /// offload everything, LM Studio's Metal behaviour).
    gpu_offload: Option<f64>,
}

async fn load_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadRequest>,
) -> Response {
    let Some(entry) = resolve_model(&state, &req.model) else {
        return error_response(
            StatusCode::NOT_FOUND,
            format!("model not found in the library: {}", req.model),
        );
    };
    if entry.format != ModelFormat::Gguf {
        return error_response(StatusCode::NOT_IMPLEMENTED, "MLX pack lands in P2");
    }
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

async fn chat_completions(State(state): State<Arc<AppState>>, body: axum::body::Bytes) -> Response {
    let parsed: serde_json::Value =
        serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null);
    let requested = parsed
        .get("model")
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    // Pick a loaded instance: exact model match first (full key or
    // bare name), then the sole instance when there is only one.
    let mut target: Option<u16> = None;
    {
        let instances = state.instances.read().await;
        if let Some(model) = &requested {
            target = instances
                .values()
                .find(|loaded| {
                    loaded.instance.model_key == *model
                        || loaded.instance.model_key.rsplit('/').next() == Some(model.as_str())
                })
                .map(|loaded| loaded.instance.port);
        }
        if target.is_none() && instances.len() == 1 {
            target = instances.values().next().map(|loaded| loaded.instance.port);
        }
    }

    // JIT: a requested, downloaded-but-unloaded GGUF loads on
    // demand through the same path as POST /api/v1/models/load.
    if target.is_none() {
        if let Some(model) = &requested {
            if let Some(entry) = resolve_model(&state, model) {
                if entry.format == ModelFormat::Gguf {
                    match spawn_instance(&state, &entry, None, None).await {
                        Ok(instance) => target = Some(instance.port),
                        Err((status, body)) => return (status, Json(body)).into_response(),
                    }
                }
            }
        }
    }

    let Some(port) = target else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "no loaded instance matches the requested model; load a model first",
        );
    };

    let upstream = format!("http://127.0.0.1:{port}/v1/chat/completions");
    let resp = match state
        .http
        .post(&upstream)
        .header(header::CONTENT_TYPE, "application/json")
        .body(body.to_vec())
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                format!("failed to reach llama-server on port {port}: {e}"),
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
