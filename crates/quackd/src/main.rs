//! quackd — the QuackStudio daemon.
//!
//! P0 surface (see ../DESIGN.md):
//!   GET  /api/v1/models        library index + (empty) loaded instances
//!   GET  /api/v1/status         daemon status
//!   GET  /v1/models             OpenAI-compat list (loaded only — none in P0)
//!   POST /api/v1/models/load    501 until backend packs land in P1
//!   POST /api/v1/models/unload  501 until backend packs land in P1
//!
//! Configuration via environment:
//!   QUACK_PORT        listen port        (default 1234)
//!   QUACK_MODELS_DIR  models directory   (default ~/.lmstudio/models)

use axum::{
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use quack_core::{scan_library, ModelEntry};
use serde::Serialize;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

struct AppState {
    port: u16,
    models_dir: PathBuf,
}

/// A library entry as served by `/api/v1/models`: the entry itself
/// plus its currently loaded instances (always empty in P0 — no
/// backend packs yet).
#[derive(Serialize)]
struct ModelWithInstances {
    #[serde(flatten)]
    entry: ModelEntry,
    loaded_instances: Vec<serde_json::Value>,
}

async fn list_models(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> Json<Vec<ModelWithInstances>> {
    let models = scan_library(&state.models_dir)
        .into_iter()
        .map(|entry| ModelWithInstances {
            entry,
            loaded_instances: Vec::new(),
        })
        .collect();
    Json(models)
}

async fn status(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let model_count = scan_library(&state.models_dir).len();
    Json(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "port": state.port,
        "models_dir": state.models_dir.to_string_lossy(),
        "model_count": model_count,
    }))
}

async fn openai_models() -> Json<serde_json::Value> {
    // OpenAI-compat lists *loaded* models only; nothing can be
    // loaded until backend packs land in P1.
    Json(serde_json::json!({ "object": "list", "data": [] }))
}

async fn not_implemented() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({ "error": "backend packs land in P1" })),
    )
}

fn default_models_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".lmstudio").join("models")
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

    let state = Arc::new(AppState { port, models_dir });
    let app = Router::new()
        .route("/api/v1/models", get(list_models))
        .route("/api/v1/status", get(status))
        .route("/api/v1/models/load", post(not_implemented))
        .route("/api/v1/models/unload", post(not_implemented))
        .route("/v1/models", get(openai_models))
        // The Tauri dev UI (localhost:5173) talks to us cross-origin.
        .layer(CorsLayer::permissive())
        .with_state(state.clone());

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("quackd: failed to bind listen address");
    println!(
        "quackd {} listening on http://{addr} (models dir: {})",
        env!("CARGO_PKG_VERSION"),
        state.models_dir.display()
    );
    axum::serve(listener, app)
        .await
        .expect("quackd: server error");
}
