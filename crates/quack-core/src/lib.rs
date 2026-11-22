//! quack-core — shared types for QuackStudio.
//!
//! The model library index (filesystem scan of a two-level
//! `publisher/model` directory, LM Studio layout), per-model load
//! configuration using LM Studio's KV field names, and backend pack
//! manifests (`backend-manifest.json`, same shape LM Studio ships).

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Model library
// ---------------------------------------------------------------------------

/// On-disk model format, as QuackStudio understands it.
/// Mirrors Duckets' backend preferences: GGUF → llama.cpp,
/// safetensors (MLX) → oMLX, Splash → the Splash/Strata pack (P3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    Gguf,
    Mlx,
    Splash,
}

/// One downloaded model in the library.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Canonical `publisher/model` identifier.
    pub key: String,
    pub publisher: String,
    /// Model directory name (what the Library table shows).
    pub display_name: String,
    pub format: ModelFormat,
    /// Recursive size of the model directory, in bytes.
    pub size_bytes: u64,
    /// Path of the model directory on disk.
    pub path: PathBuf,
    /// Architecture string when known (MLX `config.json` today; GGUF
    /// header parsing lands later).
    pub arch: Option<String>,
    /// True when the model ships a vision projector (GGUF `mmproj`)
    /// or declares a vision tower (MLX `vision_config`).
    pub vision: bool,
}

/// Scan a two-level `publisher/model` models directory.
///
/// - A model dir containing `.gguf` files is a GGUF model. It must
///   contain a main file — the largest `.gguf` whose name lacks
///   "mmproj" — and `vision` is set when an `mmproj` file is present.
/// - A model dir with a top-level `config.json` plus `*.safetensors`
///   files is an MLX model; `arch` comes from `config.json`
///   `"architectures"[0]` when parseable.
/// - A model dir with a top-level `manifest.json` plus a `target/`
///   subdirectory is a Splash model (the Splash engine's package
///   layout: `manifest.json` + `target/` + `draft/`); `vision` is
///   set when a `vision/` subdirectory exists.
/// - Anything else is skipped, as are hidden directories at either
///   level.
///
/// Results are sorted by publisher, then display name.
pub fn scan_library(models_dir: &Path) -> Vec<ModelEntry> {
    let mut entries = Vec::new();
    let Ok(publishers) = fs::read_dir(models_dir) else {
        return entries;
    };
    for publisher_entry in publishers.flatten() {
        let publisher = publisher_entry.file_name().to_string_lossy().into_owned();
        if publisher.starts_with('.') {
            continue;
        }
        let publisher_path = publisher_entry.path();
        if !publisher_path.is_dir() {
            continue;
        }
        let Ok(models) = fs::read_dir(&publisher_path) else {
            continue;
        };
        for model_entry in models.flatten() {
            let model_name = model_entry.file_name().to_string_lossy().into_owned();
            if model_name.starts_with('.') {
                continue;
            }
            let model_path = model_entry.path();
            if !model_path.is_dir() {
                continue;
            }
            if let Some(entry) = scan_model_dir(&publisher, &model_name, &model_path) {
                entries.push(entry);
            }
        }
    }
    entries.sort_by(|a, b| {
        a.publisher
            .cmp(&b.publisher)
            .then_with(|| a.display_name.cmp(&b.display_name))
    });
    entries
}

fn scan_model_dir(publisher: &str, name: &str, dir: &Path) -> Option<ModelEntry> {
    let mut files: Vec<(PathBuf, u64)> = Vec::new();
    walk_files(dir, &mut files);
    let size_bytes: u64 = files.iter().map(|(_, size)| *size).sum();
    let file_name = |p: &Path| {
        p.file_name()
            .map(|n| n.to_string_lossy().to_lowercase())
            .unwrap_or_default()
    };

    // Splash: manifest.json + target/ payload directory. Checked
    // first: a Splash package's payload may itself contain weight
    // files that would otherwise trip the GGUF/MLX rules.
    if dir.join("manifest.json").is_file() && dir.join("target").is_dir() {
        return Some(ModelEntry {
            key: format!("{publisher}/{name}"),
            publisher: publisher.to_string(),
            display_name: name.to_string(),
            format: ModelFormat::Splash,
            size_bytes,
            path: dir.to_path_buf(),
            arch: None,
            vision: dir.join("vision").is_dir(),
        });
    }

    // GGUF: needs a main file (largest .gguf that is not an mmproj).
    let ggufs: Vec<&(PathBuf, u64)> = files
        .iter()
        .filter(|(p, _)| file_name(p).ends_with(".gguf"))
        .collect();
    let has_mmproj = files.iter().any(|(p, _)| file_name(p).contains("mmproj"));
    let main_gguf = ggufs
        .iter()
        .filter(|(p, _)| !file_name(p).contains("mmproj"))
        .max_by_key(|(_, size)| *size);
    if main_gguf.is_some() {
        return Some(ModelEntry {
            key: format!("{publisher}/{name}"),
            publisher: publisher.to_string(),
            display_name: name.to_string(),
            format: ModelFormat::Gguf,
            size_bytes,
            path: dir.to_path_buf(),
            arch: None,
            vision: has_mmproj,
        });
    }

    // MLX: config.json + safetensors weights.
    let config_path = dir.join("config.json");
    let has_safetensors = files
        .iter()
        .any(|(p, _)| file_name(p).ends_with(".safetensors"));
    if config_path.is_file() && has_safetensors {
        let config_text = fs::read_to_string(&config_path).unwrap_or_default();
        let arch = serde_json::from_str::<serde_json::Value>(&config_text)
            .ok()
            .and_then(|value| {
                value
                    .get("architectures")?
                    .as_array()?
                    .first()?
                    .as_str()
                    .map(str::to_owned)
            });
        let vision = config_text.contains("\"vision_config\"");
        return Some(ModelEntry {
            key: format!("{publisher}/{name}"),
            publisher: publisher.to_string(),
            display_name: name.to_string(),
            format: ModelFormat::Mlx,
            size_bytes,
            path: dir.to_path_buf(),
            arch,
            vision,
        });
    }

    None
}

/// Recursively collect `(path, size)` for every non-hidden file under
/// `dir`. Unreadable entries are skipped silently — a partially
/// readable library should still index.
fn walk_files(dir: &Path, out: &mut Vec<(PathBuf, u64)>) {
    let Ok(read_dir) = fs::read_dir(dir) else {
        return;
    };
    for entry in read_dir.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let path = entry.path();
        if file_type.is_dir() {
            walk_files(&path, out);
        } else if file_type.is_file() {
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            out.push((path, size));
        }
    }
}

// ---------------------------------------------------------------------------
// Load configuration (LM Studio KV field names)
// ---------------------------------------------------------------------------

/// Per-model load configuration. Field names on the wire are LM
/// Studio's (`contextLength`, `gpuOffload`, …) so presets and muscle
/// memory transfer. Every field is optional: `None` means "engine
/// default", resolved by [`LoadConfig::default_for`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadConfig {
    pub context_length: Option<u32>,
    pub gpu_offload: Option<f32>,
    pub eval_batch_size: Option<u32>,
    pub physical_batch_size: Option<u32>,
    pub flash_attention: Option<bool>,
    pub num_experts: Option<u32>,
}

impl LoadConfig {
    /// Built-in defaults for a model, mirroring LM Studio's
    /// out-of-the-box behaviour per format. Fields a format does not
    /// use stay `None` (the engine ignores them).
    pub fn default_for(entry: &ModelEntry) -> Self {
        match entry.format {
            ModelFormat::Gguf => LoadConfig {
                context_length: Some(4096),
                gpu_offload: Some(1.0),
                eval_batch_size: Some(512),
                physical_batch_size: Some(512),
                flash_attention: Some(false),
                num_experts: None,
            },
            ModelFormat::Mlx => LoadConfig {
                context_length: Some(4096),
                gpu_offload: None,
                eval_batch_size: None,
                physical_batch_size: None,
                flash_attention: None,
                num_experts: None,
            },
            ModelFormat::Splash => LoadConfig {
                context_length: Some(4096),
                gpu_offload: None,
                eval_batch_size: None,
                physical_batch_size: None,
                flash_attention: None,
                num_experts: None,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Backend packs
// ---------------------------------------------------------------------------

/// A backend pack manifest — `backend-manifest.json`, the same shape
/// LM Studio ships under `~/.lmstudio/extensions/backends/`. Parsed
/// loosely: unknown fields are ignored, missing lists default empty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackManifest {
    pub name: String,
    pub version: String,
    /// Capability domains the pack serves, e.g. `["llm", "embeddings"]`.
    #[serde(default)]
    pub domains: Vec<String>,
    /// Model formats the pack can load, e.g. `["gguf"]`.
    #[serde(default)]
    pub formats: Vec<String>,
}

/// Scan a backends directory: every immediate subdirectory holding a
/// parseable `backend-manifest.json` contributes one manifest.
/// Sorted by pack name.
pub fn scan_packs(backends_dir: &Path) -> Vec<PackManifest> {
    let mut packs = Vec::new();
    let Ok(read_dir) = fs::read_dir(backends_dir) else {
        return packs;
    };
    for entry in read_dir.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let manifest_path = path.join("backend-manifest.json");
        if let Ok(text) = fs::read_to_string(&manifest_path) {
            if let Ok(manifest) = serde_json::from_str::<PackManifest>(&text) {
                packs.push(manifest);
            }
        }
    }
    packs.sort_by(|a, b| a.name.cmp(&b.name));
    packs
}

/// Lowercased file name of a path ("" when it has none).
fn file_name_lower(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default()
}

/// The main GGUF file of a GGUF model entry: the largest `.gguf`
/// under the model directory whose name lacks "mmproj" (the same
/// rule the scanner uses to recognise the model). `None` for
/// non-GGUF entries or when no main file exists.
pub fn main_gguf_path(entry: &ModelEntry) -> Option<PathBuf> {
    if entry.format != ModelFormat::Gguf {
        return None;
    }
    let mut files: Vec<(PathBuf, u64)> = Vec::new();
    walk_files(&entry.path, &mut files);
    files
        .iter()
        .filter(|(p, _)| {
            let name = file_name_lower(p);
            name.ends_with(".gguf") && !name.contains("mmproj")
        })
        .max_by_key(|(_, size)| *size)
        .map(|(p, _)| p.clone())
}

// ---------------------------------------------------------------------------
// Loaded instances
// ---------------------------------------------------------------------------

/// A running model instance, tracked by quackd's registry and
/// reported inside each library entry's `loaded_instances`
/// (LM Studio loads by instance, not by model).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Instance {
    pub instance_id: String,
    pub model_key: String,
    /// Backend pack that loaded it, e.g. `"llama.cpp"`.
    pub backend: String,
    /// Loopback port the engine process serves on.
    pub port: u16,
    /// Engine process id.
    pub pid: u32,
}

// ---------------------------------------------------------------------------
// llama.cpp pack discovery
// ---------------------------------------------------------------------------

/// Parse the trailing `major.minor.patch` of a backend pack
/// directory name (everything after the last `-`).
fn parse_pack_version(dir_name: &str) -> Option<(u64, u64, u64)> {
    let tail = dir_name.rsplit('-').next()?;
    let mut parts = tail.split('.');
    let version = (
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
        parts.next()?.parse().ok()?,
    );
    if parts.next().is_some() {
        return None;
    }
    Some(version)
}

/// Find the `llama-server` binary of the newest llama.cpp backend
/// pack under `backends_dir`. Pack directories are named like
/// `llama.cpp-mac-arm64-apple-metal-advsimd-2.80.1` and sit side by
/// side across versions; the highest trailing version wins.
/// `None` when no pack ships a `llama-server` file.
pub fn find_llama_server(backends_dir: &Path) -> Option<PathBuf> {
    let read_dir = fs::read_dir(backends_dir).ok()?;
    let mut best: Option<((u64, u64, u64), PathBuf)> = None;
    for entry in read_dir.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("llama.cpp-") {
            continue;
        }
        let Some(version) = parse_pack_version(&name) else {
            continue;
        };
        let server = entry.path().join("llama-server");
        if !server.is_file() {
            continue;
        }
        if best.as_ref().map_or(true, |(v, _)| version > *v) {
            best = Some((version, server));
        }
    }
    best.map(|(_, path)| path)
}

// ---------------------------------------------------------------------------
// oMLX pack discovery
// ---------------------------------------------------------------------------

/// Find the Python interpreter of the newest oMLX (LM Studio's
/// selected MLX backend) install under `backends_dir`. oMLX builds
/// live under `vendor/_amphibian/` as `app-omlx-mac26-arm64@<N>`
/// directories (several builds side by side); the highest build
/// number with a `bin/python` wins. That interpreter imports
/// `mlx_vlm`, which is what quackd serves MLX models with.
/// `None` when no oMLX build is installed.
pub fn find_omlx_python(backends_dir: &Path) -> Option<PathBuf> {
    let amphibian = backends_dir.join("vendor").join("_amphibian");
    let read_dir = fs::read_dir(&amphibian).ok()?;
    let mut best: Option<(u64, PathBuf)> = None;
    for entry in read_dir.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(build) = name.strip_prefix("app-omlx-mac26-arm64@") else {
            continue;
        };
        let Ok(build) = build.parse::<u64>() else {
            continue;
        };
        let python = entry.path().join("bin").join("python");
        if !python.is_file() {
            continue;
        }
        if best.as_ref().map_or(true, |(b, _)| build > *b) {
            best = Some((build, python));
        }
    }
    best.map(|(_, path)| path)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_bytes(path: &Path, n: usize) {
        let mut f = fs::File::create(path).unwrap();
        f.write_all(&vec![0u8; n]).unwrap();
    }

    /// Fixture tree:
    ///   models/lmstudio-community/Test-GGUF/{test-q4_k_m.gguf, mmproj-test.gguf}
    ///   models/mlx-community/Test-MLX/{config.json, model.safetensors}
    ///   models/duckets/Test-Splash/{manifest.json, target/weights.bin, draft/, vision/}
    ///   models/.hidden/Nope/x.gguf            (skipped: hidden)
    ///   models/someone/not-a-model/readme.txt  (skipped: not a model)
    fn fixture() -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "quack-core-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let gguf_dir = base.join("lmstudio-community").join("Test-GGUF");
        fs::create_dir_all(&gguf_dir).unwrap();
        write_bytes(&gguf_dir.join("test-q4_k_m.gguf"), 100);
        write_bytes(&gguf_dir.join("mmproj-test.gguf"), 10);

        let mlx_dir = base.join("mlx-community").join("Test-MLX");
        fs::create_dir_all(&mlx_dir).unwrap();
        fs::write(
            mlx_dir.join("config.json"),
            r#"{"architectures": ["LlamaForCausalLM"]}"#,
        )
        .unwrap();
        write_bytes(&mlx_dir.join("model.safetensors"), 50);

        let splash_dir = base.join("duckets").join("Test-Splash");
        fs::create_dir_all(splash_dir.join("target")).unwrap();
        fs::create_dir_all(splash_dir.join("draft")).unwrap();
        fs::create_dir_all(splash_dir.join("vision")).unwrap();
        fs::write(splash_dir.join("manifest.json"), "{}").unwrap();
        write_bytes(&splash_dir.join("target").join("weights.bin"), 200);

        let hidden_dir = base.join(".hidden").join("Nope");
        fs::create_dir_all(&hidden_dir).unwrap();
        write_bytes(&hidden_dir.join("x.gguf"), 5);

        let junk_dir = base.join("someone").join("not-a-model");
        fs::create_dir_all(&junk_dir).unwrap();
        fs::write(junk_dir.join("readme.txt"), "hi").unwrap();

        base
    }

    #[test]
    fn scan_library_finds_gguf_mlx_and_splash() {
        let base = fixture();
        let entries = scan_library(&base);
        assert_eq!(entries.len(), 3, "entries: {entries:?}");

        // Sorted by publisher: duckets before lmstudio-community
        // before mlx-community.
        let splash = &entries[0];
        assert_eq!(splash.key, "duckets/Test-Splash");
        assert_eq!(splash.format, ModelFormat::Splash);
        assert!(splash.vision, "vision/ subdir means vision");
        // 200 bytes of payload + the 2-byte manifest.json.
        assert_eq!(splash.size_bytes, 202);
        assert_eq!(splash.arch, None);

        let gguf = &entries[1];
        assert_eq!(gguf.key, "lmstudio-community/Test-GGUF");
        assert_eq!(gguf.publisher, "lmstudio-community");
        assert_eq!(gguf.display_name, "Test-GGUF");
        assert_eq!(gguf.format, ModelFormat::Gguf);
        assert_eq!(gguf.size_bytes, 110);
        assert!(gguf.vision, "mmproj present means vision");
        assert_eq!(gguf.arch, None);

        let mlx = &entries[2];
        assert_eq!(mlx.key, "mlx-community/Test-MLX");
        assert_eq!(mlx.format, ModelFormat::Mlx);
        assert_eq!(mlx.arch.as_deref(), Some("LlamaForCausalLM"));
        assert!(!mlx.vision);
        // 50 bytes of weights + the config.json bytes.
        assert!(mlx.size_bytes > 50);

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn scan_library_missing_dir_is_empty() {
        let entries = scan_library(Path::new("/nonexistent/quack/models"));
        assert!(entries.is_empty());
    }

    #[test]
    fn load_config_uses_lm_studio_field_names() {
        let entry = ModelEntry {
            key: "p/m".into(),
            publisher: "p".into(),
            display_name: "m".into(),
            format: ModelFormat::Gguf,
            size_bytes: 0,
            path: PathBuf::from("/tmp/m"),
            arch: None,
            vision: false,
        };
        let cfg = LoadConfig::default_for(&entry);
        let json = serde_json::to_value(&cfg).unwrap();
        assert_eq!(json["contextLength"], 4096);
        assert_eq!(json["gpuOffload"], 1.0);
        assert_eq!(json["evalBatchSize"], 512);
        assert_eq!(json["physicalBatchSize"], 512);
        assert_eq!(json["flashAttention"], false);
        assert!(json["numExperts"].is_null());
    }

    #[test]
    fn scan_packs_reads_manifests() {
        let base = std::env::temp_dir().join(format!(
            "quack-core-packs-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let pack_dir = base.join("llama.cpp-mac-arm64@2.80.1");
        fs::create_dir_all(&pack_dir).unwrap();
        fs::write(
            pack_dir.join("backend-manifest.json"),
            r#"{"name": "llama.cpp", "version": "2.80.1", "domains": ["llm"], "formats": ["gguf"], "extra": true}"#,
        )
        .unwrap();
        let empty_dir = base.join("no-manifest-here");
        fs::create_dir_all(&empty_dir).unwrap();

        let packs = scan_packs(&base);
        assert_eq!(packs.len(), 1);
        assert_eq!(packs[0].name, "llama.cpp");
        assert_eq!(packs[0].version, "2.80.1");
        assert_eq!(packs[0].formats, vec!["gguf".to_string()]);

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn find_llama_server_picks_highest_version() {
        let base = std::env::temp_dir().join(format!(
            "quack-core-llama-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        for version in ["2.71.0", "2.80.1", "2.9.0"] {
            let dir = base.join(format!(
                "llama.cpp-mac-arm64-apple-metal-advsimd-{version}"
            ));
            fs::create_dir_all(&dir).unwrap();
            write_bytes(&dir.join("llama-server"), 8);
        }
        // A newer-looking dir without the binary must not win, and
        // unrelated packs are ignored.
        let no_binary = base.join("llama.cpp-mac-arm64-apple-metal-advsimd-9.9.9");
        fs::create_dir_all(&no_binary).unwrap();
        let mlx_dir = base.join("mlx-llm-mac-arm64-1.14.0");
        fs::create_dir_all(&mlx_dir).unwrap();
        write_bytes(&mlx_dir.join("llama-server"), 8);

        let found = find_llama_server(&base).expect("a llama-server");
        assert!(
            found.ends_with("llama.cpp-mac-arm64-apple-metal-advsimd-2.80.1/llama-server"),
            "picked: {found:?}"
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn find_omlx_python_picks_highest_build() {
        let base = std::env::temp_dir().join(format!(
            "quack-core-omlx-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let amphibian = base.join("vendor").join("_amphibian");
        for build in [9_u32, 18] {
            let bin = amphibian
                .join(format!("app-omlx-mac26-arm64@{build}"))
                .join("bin");
            fs::create_dir_all(&bin).unwrap();
            write_bytes(&bin.join("python"), 8);
        }
        // The highest build *without* a python must not win, and
        // unrelated apps under _amphibian are ignored.
        fs::create_dir_all(amphibian.join("app-omlx-mac26-arm64@99")).unwrap();
        let other = amphibian.join("app-something-else@100").join("bin");
        fs::create_dir_all(&other).unwrap();
        write_bytes(&other.join("python"), 8);

        let found = find_omlx_python(&base).expect("an omlx python");
        assert!(
            found.ends_with("app-omlx-mac26-arm64@18/bin/python"),
            "picked: {found:?}"
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn main_gguf_path_skips_mmproj() {
        let base = fixture();
        let entries = scan_library(&base);
        let gguf = entries
            .iter()
            .find(|e| e.format == ModelFormat::Gguf)
            .unwrap();
        let main = main_gguf_path(gguf).expect("main gguf");
        assert!(main.ends_with("test-q4_k_m.gguf"), "picked: {main:?}");

        let mlx = entries
            .iter()
            .find(|e| e.format == ModelFormat::Mlx)
            .unwrap();
        assert_eq!(main_gguf_path(mlx), None);

        let _ = fs::remove_dir_all(&base);
    }
}
