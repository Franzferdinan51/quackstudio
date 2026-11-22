//! quack — QuackStudio CLI (an `lms`-shaped subset).
//!
//! Talks to a running quackd over HTTP; there is no privileged
//! channel, exactly like the desktop UI. Daemon base URL comes from
//! QUACK_URL (default http://127.0.0.1:1234).

use clap::{Parser, Subcommand};
use quack_core::ModelEntry;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "quack", version, about = "QuackStudio command-line client")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List downloaded models in the library.
    Ls,
    /// List currently loaded model instances.
    Ps,
    /// Load a model (by publisher/name key or bare name) and print its instance id.
    Load {
        /// Model key (`publisher/name`) or bare model name.
        model: String,
    },
    /// Unload a loaded instance by its instance id (see `quack ps`).
    Unload {
        /// Instance id, e.g. `publisher/model#1`.
        instance_id: String,
    },
    /// Server management.
    Server {
        #[command(subcommand)]
        action: ServerAction,
    },
}

#[derive(Subcommand)]
enum ServerAction {
    /// Show daemon status (version, port, models dir, model count).
    Status,
}

fn daemon_base() -> String {
    std::env::var("QUACK_URL").unwrap_or_else(|_| "http://127.0.0.1:1234".to_string())
}

fn unreachable(url: &str, err: &ureq::Error) -> String {
    format!(
        "cannot reach quackd at {url} ({err}).\n\
         is the server running? start it with: cargo run -p quackd"
    )
}

/// GET a daemon endpoint and return the raw body text.
fn get_body(path: &str) -> Result<String, String> {
    let url = format!("{}{}", daemon_base(), path);
    let mut response = ureq::get(&url)
        .call()
        .map_err(|e| unreachable(&url, &e))?;
    response
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("failed reading response from {url}: {e}"))
}

/// GET /api/v1/models as raw JSON (keeps `loaded_instances`, which
/// the typed ModelEntry intentionally does not model).
fn get_models_json() -> Result<serde_json::Value, String> {
    let body = get_body("/api/v1/models")?;
    serde_json::from_str(&body).map_err(|e| format!("bad JSON from quackd: {e}"))
}

/// POST a JSON body to a daemon endpoint and return
/// `(status, parsed body)`. HTTP error statuses are *not* treated
/// as transport errors here — the daemon's `{"error": …}` body is
/// the useful part, so callers decide what a status means.
fn post_json(path: &str, payload: serde_json::Value) -> Result<(u16, serde_json::Value), String> {
    let url = format!("{}{}", daemon_base(), path);
    let config = ureq::config::Config::builder()
        .http_status_as_error(false)
        // A cold model load can take a while (daemon allows 180 s
        // for the engine's health check alone).
        .timeout_global(Some(std::time::Duration::from_secs(600)))
        .build();
    let agent = ureq::Agent::new_with_config(config);
    let mut response = agent
        .post(&url)
        .send_json(payload)
        .map_err(|e| unreachable(&url, &e))?;
    let status = response.status().as_u16();
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("failed reading response from {url}: {e}"))?;
    let json = serde_json::from_str(&text).unwrap_or(serde_json::Value::Null);
    Ok((status, json))
}

/// Extract the daemon's `{"error": …}` message, if present.
fn daemon_error(json: &serde_json::Value, status: u16) -> String {
    json.get("error")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("request failed (HTTP {status})"))
}

fn cmd_load(model: &str) -> Result<(), String> {
    let (status, json) = post_json(
        "/api/v1/models/load",
        serde_json::json!({ "model": model }),
    )?;
    if status != 200 {
        return Err(daemon_error(&json, status));
    }
    let instance_id = json
        .get("instance_id")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    println!("loaded: {instance_id}");
    if let Some(seconds) = json.get("load_time_seconds").and_then(|v| v.as_f64()) {
        println!("load time: {seconds:.1}s");
    }
    Ok(())
}

fn cmd_unload(instance_id: &str) -> Result<(), String> {
    let (status, json) = post_json(
        "/api/v1/models/unload",
        serde_json::json!({ "instance_id": instance_id }),
    )?;
    if status != 200 {
        return Err(daemon_error(&json, status));
    }
    println!("unloaded: {instance_id}");
    Ok(())
}

fn cmd_ls() -> Result<(), String> {
    let body = get_body("/api/v1/models")?;
    let models: Vec<ModelEntry> =
        serde_json::from_str(&body).map_err(|e| format!("bad JSON from quackd: {e}"))?;
    if models.is_empty() {
        println!("no models in the library");
        return Ok(());
    }
    for m in &models {
        println!(
            "{:<48} {:<5} {:>8.2} GB",
            format!("{}/{}", m.publisher, m.display_name),
            match m.format {
                quack_core::ModelFormat::Gguf => "gguf",
                quack_core::ModelFormat::Mlx => "mlx",
            },
            m.size_bytes as f64 / 1_000_000_000.0
        );
    }
    Ok(())
}

fn cmd_ps() -> Result<(), String> {
    let payload = get_models_json()?;
    let mut any = false;
    if let Some(models) = payload.as_array() {
        for model in models {
            if let Some(instances) = model.get("loaded_instances").and_then(|v| v.as_array()) {
                for instance in instances {
                    any = true;
                    println!(
                        "{}  {}",
                        model.get("key").and_then(|v| v.as_str()).unwrap_or("?"),
                        instance
                    );
                }
            }
        }
    }
    if !any {
        println!("no models loaded");
    }
    Ok(())
}

fn cmd_server_status() -> Result<(), String> {
    let body = get_body("/api/v1/status")?;
    let status: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("bad JSON from quackd: {e}"))?;
    println!("server:    running");
    println!(
        "version:   {}",
        status.get("version").and_then(|v| v.as_str()).unwrap_or("?")
    );
    println!(
        "port:      {}",
        status.get("port").and_then(|v| v.as_u64()).unwrap_or(0)
    );
    println!(
        "models:    {} ({})",
        status
            .get("model_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        status
            .get("models_dir")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
    );
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Ls => cmd_ls(),
        Commands::Ps => cmd_ps(),
        Commands::Load { model } => cmd_load(&model),
        Commands::Unload { instance_id } => cmd_unload(&instance_id),
        Commands::Server { action } => match action {
            ServerAction::Status => cmd_server_status(),
        },
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}
