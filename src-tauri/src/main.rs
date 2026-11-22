// Prevents an extra console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::{SocketAddr, TcpStream};
use std::sync::Mutex;
use std::time::Duration;

use tauri::{Manager, RunEvent, State};
use tauri_plugin_shell::process::CommandChild;
use tauri_plugin_shell::ShellExt;

/// The daemon this shell is responsible for, and the port the UI
/// talks to. The UI never guesses the port: it asks through the
/// `quack_port` command (see ui/src/api.ts). All model traffic
/// still goes over quackd's public HTTP API — the shell's only
/// privileges are spawning and stopping the daemon.
struct DaemonState {
    port: u16,
    /// False when both candidate ports were already taken: a
    /// server (LM Studio, or a quackd from an earlier launch) is
    /// answering and we spawned nothing of our own.
    spawn_sidecar: bool,
    child: Mutex<Option<CommandChild>>,
}

#[tauri::command]
fn quack_port(state: State<'_, DaemonState>) -> u16 {
    state.port
}

/// True when something accepts a TCP connection on 127.0.0.1:port.
fn port_taken(port: u16) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(250)).is_ok()
}

/// quackd's default port is 1234, but LM Studio (or an already
/// running quackd) may own it — then our daemon takes 1235. When
/// both are taken we spawn nothing and point the UI at 1234, where
/// a server is already answering.
fn pick_port() -> (u16, bool) {
    if !port_taken(1234) {
        (1234, true)
    } else if !port_taken(1235) {
        (1235, true)
    } else {
        (1234, false)
    }
}

/// Stop the sidecar we spawned, if it is still ours to stop.
fn kill_sidecar(app: &tauri::AppHandle) {
    let Some(state) = app.try_state::<DaemonState>() else {
        return;
    };
    // Bind the guard as a named local so it drops at the end of
    // this block, before `state` does.
    let child = {
        let Ok(mut slot) = state.child.lock() else {
            return;
        };
        slot.take()
    };
    if let Some(child) = child {
        let _ = child.kill();
    }
}

fn main() {
    let (port, spawn_sidecar) = pick_port();
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(DaemonState {
            port,
            spawn_sidecar,
            child: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![quack_port])
        .setup(|app| {
            let state = app.state::<DaemonState>();
            if state.spawn_sidecar {
                match app.shell().sidecar("quackd") {
                    Ok(command) => {
                        let spawned = command
                            .env("QUACK_PORT", state.port.to_string())
                            .spawn();
                        match spawned {
                            Ok((_events, child)) => {
                                if let Ok(mut slot) = state.child.lock() {
                                    *slot = Some(child);
                                }
                            }
                            Err(e) => {
                                eprintln!(
                                    "quackstudio: failed to start the quackd sidecar: {e}"
                                );
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("quackstudio: quackd sidecar unavailable: {e}");
                    }
                }
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while running QuackStudio");

    // Never orphan the daemon: it holds engine processes and GPU
    // memory of its own, so it dies with the app.
    app.run(|app_handle, event| match event {
        RunEvent::ExitRequested { .. } | RunEvent::Exit => kill_sidecar(app_handle),
        _ => {}
    });
}
