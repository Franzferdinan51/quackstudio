// Prevents an extra console window on Windows in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // P0: no commands yet. The UI talks to quackd over its public
    // HTTP API (http://127.0.0.1:1234), exactly like the CLI — the
    // shell deliberately has no privileged back door.
    tauri::Builder::default()
        .run(tauri::generate_context!())
        .expect("error while running QuackStudio");
}
