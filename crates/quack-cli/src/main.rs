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
