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

/// Model format / serving kind, as QuackStudio understands it.
/// Mirrors Duckets' backend preferences: GGUF → llama.cpp,
/// safetensors (MLX) → oMLX, Splash → the Splash/Strata pack.
/// Maya is not an on-disk format: it marks a remote model served
/// by a Maya provider (see [`MayaProvider`]) and merged into the
/// library by quackd — zero local files, zero local RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    Gguf,
    Mlx,
    Splash,
    Maya,
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
    /// Architecture string when known (MLX `config.json`
    /// `architectures[0]`; GGUF `general.architecture` metadata).
    pub arch: Option<String>,
    /// True when the model ships a vision projector (GGUF `mmproj`)
    /// or declares a vision tower (MLX `vision_config`).
    pub vision: bool,
    /// Name of the model's main payload file (the main GGUF, the
    /// largest safetensors shard, the Splash target payload).
    /// `None` when no payload file could be identified.
    #[serde(default)]
    pub file_name: Option<String>,
    /// Quantization label when derivable: from the GGUF file name
    /// (`Q4_K_M`, `Q8_0`, …), the GGUF `general.file_type`
    /// metadata, or an MLX directory name (`4-bit`, `8-bit`).
    #[serde(default)]
    pub quantization: Option<String>,
    /// Context-length ceiling from the model's own metadata
    /// (GGUF `<arch>.context_length`, MLX `max_position_embeddings`).
    /// `None` when the metadata does not declare one.
    #[serde(default)]
    pub max_context: Option<u32>,
    /// Serving domain: `"llm"`, or `"embedding"` for models whose
    /// name marks them as embedding models.
    #[serde(default = "default_domain")]
    pub domain: String,
    /// Scanner flags for this entry (incomplete downloads, missing
    /// payload files, …). Empty for a healthy model. The Library's
    /// "Show warnings" filter lists exactly the entries with a
    /// non-empty list.
    #[serde(default)]
    pub warnings: Vec<String>,
    /// Base URL of the remote server when the model is served
    /// remotely (Maya pack entries); `None` for local models.
    pub remote_url: Option<String>,
}

fn default_domain() -> String {
    "llm".to_string()
}

/// The serving domain a directory name implies: embedding models
/// are the one family the scanner can tell apart by name alone.
fn domain_for_name(name: &str) -> String {
    let lower = name.to_lowercase();
    if lower.contains("embed") || lower.contains("nomic") {
        "embedding".to_string()
    } else {
        "llm".to_string()
    }
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
    let base = |format: ModelFormat| ModelEntry {
        key: format!("{publisher}/{name}"),
        publisher: publisher.to_string(),
        display_name: name.to_string(),
        format,
        size_bytes,
        path: dir.to_path_buf(),
        arch: None,
        vision: false,
        file_name: None,
        quantization: None,
        max_context: None,
        domain: domain_for_name(name),
        warnings: Vec::new(),
        remote_url: None,
    };
    // A leftover `.part` file means a download never finished —
    // the one warning every format shares.
    let mut shared_warnings: Vec<String> = Vec::new();
    if files.iter().any(|(p, _)| file_name(p).ends_with(".part")) {
        shared_warnings.push("Incomplete download: a .part file is present".to_string());
    }

    // Splash: manifest.json + target/ payload directory. Checked
    // first: a Splash package's payload may itself contain weight
    // files that would otherwise trip the GGUF/MLX rules.
    if dir.join("manifest.json").is_file() && dir.join("target").is_dir() {
        let mut entry = base(ModelFormat::Splash);
        entry.vision = dir.join("vision").is_dir();
        entry.warnings = shared_warnings;
        // The main payload file: the largest file under target/.
        entry.file_name = files
            .iter()
            .filter(|(p, _)| p.starts_with(dir.join("target")))
            .max_by_key(|(_, size)| *size)
            .and_then(|(p, _)| p.file_name().map(|n| n.to_string_lossy().into_owned()));
        return Some(entry);
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
    if let Some((main_path, main_size)) = main_gguf {
        let mut entry = base(ModelFormat::Gguf);
        entry.vision = has_mmproj;
        entry.warnings = shared_warnings;
        if *main_size == 0 {
            entry
                .warnings
                .push("Main model file is empty (0 bytes)".to_string());
        }
        entry.file_name = main_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned());
        // Header metadata: architecture, context ceiling, and the
        // file-type quantization fallback.
        let metadata = read_gguf_metadata(main_path);
        entry.arch = metadata.architecture;
        entry.max_context = metadata.context_length;
        entry.quantization = entry
            .file_name
            .as_deref()
            .and_then(quantization_from_file_name)
            .or(metadata.file_type_label);
        return Some(entry);
    }
    // GGUF fragments with no main file (only a vision projector):
    // still list the entry — flagged — instead of hiding it.
    if has_mmproj {
        let mut entry = base(ModelFormat::Gguf);
        entry.vision = true;
        entry.warnings = shared_warnings;
        entry.warnings.push(
            "No main model file found — only a vision projector (mmproj) is present"
                .to_string(),
        );
        entry.file_name = ggufs
            .iter()
            .max_by_key(|(_, size)| *size)
            .and_then(|(p, _)| p.file_name().map(|n| n.to_string_lossy().into_owned()));
        return Some(entry);
    }

    // MLX: config.json + safetensors weights.
    let config_path = dir.join("config.json");
    let has_safetensors = files
        .iter()
        .any(|(p, _)| file_name(p).ends_with(".safetensors"));
    if config_path.is_file() && has_safetensors {
        let config_text = fs::read_to_string(&config_path).unwrap_or_default();
        let config: Option<serde_json::Value> = serde_json::from_str(&config_text).ok();
        let mut entry = base(ModelFormat::Mlx);
        entry.vision = config_text.contains("\"vision_config\"");
        entry.warnings = shared_warnings;
        entry.arch = config.as_ref().and_then(|value| {
            value
                .get("architectures")?
                .as_array()?
                .first()?
                .as_str()
                .map(str::to_owned)
        });
        entry.max_context = config.as_ref().and_then(mlx_max_context);
        entry.quantization = quantization_from_file_name(name);
        entry.file_name = files
            .iter()
            .filter(|(p, _)| file_name(p).ends_with(".safetensors"))
            .max_by_key(|(_, size)| *size)
            .and_then(|(p, _)| p.file_name().map(|n| n.to_string_lossy().into_owned()));
        return Some(entry);
    }

    // Nothing loadable, but the folder carries a scanner warning
    // (an unfinished `.part` download): list it flagged rather
    // than hiding it, so Show warnings can surface it.
    if !shared_warnings.is_empty() {
        let mut entry = base(ModelFormat::Gguf);
        entry.warnings = shared_warnings;
        entry.file_name = files
            .first()
            .and_then(|(p, _)| p.file_name().map(|n| n.to_string_lossy().into_owned()));
        return Some(entry);
    }

    None
}

/// The context ceiling an MLX `config.json` declares:
/// `max_position_embeddings` at the top level or inside
/// `text_config` (vision-language layouts nest it there).
fn mlx_max_context(config: &serde_json::Value) -> Option<u32> {
    let dig = |v: &serde_json::Value| -> Option<u32> {
        v.get("max_position_embeddings")
            .and_then(|n| n.as_u64())
            .and_then(|n| u32::try_from(n).ok())
    };
    dig(config).or_else(|| config.get("text_config").and_then(dig))
}

/// Quantization tokens as they appear in GGUF file names and MLX
/// directory names, longest first so `Q4_K_M` wins over `Q4_0`.
const QUANT_TOKENS: &[&str] = &[
    "Q4_K_XL", "Q5_K_XL", "Q3_K_XL", "Q6_K_XL", "Q2_K_XL",
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q3_K_M", "Q3_K_L", "Q3_K_S", "Q6_K",
    "Q2_K", "Q8_0", "Q5_0", "Q5_1", "Q4_0", "Q4_1", "IQ4_XS", "IQ4_NL", "IQ3_S",
    "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ1_S", "IQ1_M", "BF16", "F16", "F32",
];

/// The quantization label a file or directory name carries, if
/// any. GGUF names embed the token (`…-Q4_K_M.gguf`); MLX names
/// use bit widths (`…-4bit` → `4-bit`).
fn quantization_from_file_name(name: &str) -> Option<String> {
    let upper = name.to_uppercase().replace('-', "_");
    for token in QUANT_TOKENS {
        if upper.contains(token) {
            return Some((*token).to_string());
        }
    }
    let lower = name.to_lowercase();
    for bits in ["4bit", "8bit", "2bit", "3bit", "6bit"] {
        if lower.contains(bits) {
            return Some(format!("{}-bit", &bits[..1]));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// GGUF header metadata
// ---------------------------------------------------------------------------

/// The header facts the library surfaces: `general.architecture`,
/// `<arch>.context_length`, and `general.file_type` (as a label).
#[derive(Debug, Default, PartialEq)]
pub struct GgufMetadata {
    pub architecture: Option<String>,
    pub context_length: Option<u32>,
    pub file_type_label: Option<String>,
}

/// How much of a GGUF file the metadata reader will scan. The
/// metadata block sits at the very start of the file, long before
/// the tensor data; 16 MiB is far beyond any real header.
const GGUF_METADATA_CAP: usize = 16 * 1024 * 1024;

/// Read a GGUF file's header metadata. Any malformation (bad
/// magic, truncation, an unknown value type) yields whatever was
/// gathered before it — the scanner treats metadata as a bonus,
/// never a gate.
pub fn read_gguf_metadata(path: &Path) -> GgufMetadata {
    let Ok(mut file) = fs::File::open(path) else {
        return GgufMetadata::default();
    };
    use std::io::Read;
    let mut buf = Vec::new();
    let _ = file
        .by_ref()
        .take(GGUF_METADATA_CAP as u64)
        .read_to_end(&mut buf);
    parse_gguf_metadata(&buf)
}

/// A little-endian cursor over the header bytes.
struct GgufCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> GgufCursor<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let slice = self.bytes.get(self.pos..end)?;
        self.pos = end;
        Some(slice)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn string(&mut self) -> Option<String> {
        let len = usize::try_from(self.u64()?).ok()?;
        let raw = self.take(len)?;
        Some(String::from_utf8_lossy(raw).into_owned())
    }
    /// Skip one value of the given GGUF metadata type. `false`
    /// when the bytes run out or the type is unknown.
    fn skip_value(&mut self, value_type: u32) -> bool {
        let size = match value_type {
            0 | 1 | 7 => 1,  // UINT8, INT8, BOOL
            2 | 3 => 2,      // UINT16, INT16
            4 | 5 | 6 => 4,  // UINT32, INT32, FLOAT32
            10 | 11 | 12 => 8, // UINT64, INT64, FLOAT64
            8 => return self.string().is_some(), // STRING
            9 => {
                // ARRAY: element type + count, then the elements.
                let Some(elem_type) = self.u32() else {
                    return false;
                };
                let Some(count) = self.u64() else {
                    return false;
                };
                for _ in 0..count {
                    if !self.skip_value(elem_type) {
                        return false;
                    }
                }
                return true;
            }
            _ => return false,
        };
        self.take(size).is_some()
    }
}

/// Parse the metadata block out of raw GGUF header bytes.
fn parse_gguf_metadata(bytes: &[u8]) -> GgufMetadata {
    let mut out = GgufMetadata::default();
    let mut cur = GgufCursor { bytes, pos: 0 };
    if cur.take(4) != Some(b"GGUF".as_slice()) {
        return out;
    }
    let Some(_version) = cur.u32() else { return out };
    let Some(_tensor_count) = cur.u64() else { return out };
    let Some(kv_count) = cur.u64() else { return out };
    let mut file_type: Option<u32> = None;
    for _ in 0..kv_count {
        let Some(key) = cur.string() else { return out };
        let Some(value_type) = cur.u32() else { return out };
        match (key.as_str(), value_type) {
            ("general.architecture", 8) => {
                out.architecture = cur.string();
            }
            ("general.file_type", 4) => {
                file_type = cur.u32();
            }
            (_, 8) if key.ends_with(".context_length") => {
                // Wrong type for a context length; consume it.
                let _ = cur.string();
            }
            _ => {
                // `<arch>.context_length` is an integer (UINT32 in
                // practice); capture it, skip everything else.
                if key.ends_with(".context_length") {
                    match value_type {
                        4 => out.context_length = cur.u32(),
                        10 => {
                            out.context_length = cur.u64().and_then(|n| u32::try_from(n).ok())
                        }
                        _ => {
                            if !cur.skip_value(value_type) {
                                return finish_gguf(out, file_type);
                            }
                        }
                    }
                } else if !cur.skip_value(value_type) {
                    return finish_gguf(out, file_type);
                }
            }
        }
    }
    finish_gguf(out, file_type)
}

fn finish_gguf(mut out: GgufMetadata, file_type: Option<u32>) -> GgufMetadata {
    out.file_type_label = file_type.and_then(gguf_file_type_label);
    out
}

/// `general.file_type` (llama.cpp `llama_ftype`) → label. Only
/// the common values are mapped; anything else stays unlabeled.
fn gguf_file_type_label(file_type: u32) -> Option<String> {
    let label = match file_type {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        10 => "Q2_K",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        30 => "BF16",
        _ => return None,
    };
    Some(label.to_string())
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
            // Remote models are loaded by Maya, not by us; the
            // config is informational only.
            ModelFormat::Maya => LoadConfig {
                context_length: None,
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
// Maya pack (remote backend)
// ---------------------------------------------------------------------------

/// A Maya provider: one of Duckets' Maya model servers. Each is an
/// OpenAI-compatible HTTP server (its own port on the Mac); quackd
/// lists its models as remote library entries and proxies chat to
/// it, so a Maya model costs QuackStudio zero local processes and
/// zero local RAM.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MayaProvider {
    /// Short provider name, used in library keys:
    /// `maya/<name>/<model-id>`.
    pub name: String,
    /// Base URL of the provider's OpenAI-compatible API,
    /// e.g. `http://100.68.208.113:8085` (no trailing slash).
    pub base_url: String,
}

/// Duckets' Maya providers on the Mac (tailnet address), one per
/// model family. The front door on :8090 deliberately does not
/// route every family (it does not route ik2), so the pack talks
/// to the per-provider endpoints directly. `hy4` is often not
/// running; quackd treats an unanswering provider as absent.
pub fn default_maya_providers() -> Vec<MayaProvider> {
    const HOST: &str = "100.68.208.113";
    [
        ("glm", 8080),
        ("minimax", 8081),
        ("qwen", 8082),
        ("qwen-native", 8083),
        ("hy4", 8084),
        ("ik2", 8085),
    ]
    .into_iter()
    .map(|(name, port)| MayaProvider {
        name: name.to_string(),
        base_url: format!("http://{HOST}:{port}"),
    })
    .collect()
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
// Splash pack discovery
// ---------------------------------------------------------------------------

/// A resolved Splash backend pack: everything quackd needs to
/// spawn the pack's OpenAI-compatible server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplashPack {
    /// Pack version from `backend-manifest.json` ("1.3.0").
    pub version: String,
    /// The amphibian Python that runs the server
    /// (`<app layer>/bin/python`).
    pub python: PathBuf,
    /// The app layer root (holds `bin/`, `engine/` and the
    /// server's site-packages).
    pub app_layer: PathBuf,
    /// The native engine binary (`<app layer>/engine/splash`),
    /// passed to the server as `--binary`.
    pub binary: PathBuf,
}

/// Find the newest Splash backend pack under `backends_dir`.
/// Pack directories are named like
/// `splash-mac-arm64-apple-metal-advsimd-1.3.0`; the highest
/// trailing version wins. The Python interpreter comes from the
/// pack's `backend-manifest.json`
/// (`engine_protocol_server.executable_relative_path`, relative
/// to `backends_dir/vendor`); when the manifest can't be read,
/// the newest `app-splash-*` layer under `vendor/_amphibian` is
/// used instead, mirroring [`find_omlx_python`]. `None` when no
/// usable pack is installed.
pub fn find_splash_pack(backends_dir: &Path) -> Option<SplashPack> {
    // 1. Newest pack directory, by trailing version.
    let mut best_pack: Option<((u64, u64, u64), PathBuf)> = None;
    for entry in fs::read_dir(backends_dir).ok()?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("splash-") {
            continue;
        }
        let Some(version) = parse_pack_version(&name) else {
            continue;
        };
        if best_pack.as_ref().map_or(true, |(v, _)| version > *v) {
            best_pack = Some((version, entry.path()));
        }
    }

    // 2. The interpreter: the pack manifest names it; fall back
    //    to the newest splash app layer under vendor/_amphibian.
    let vendor = backends_dir.join("vendor");
    let mut python: Option<PathBuf> = None;
    let mut version: Option<String> = None;
    if let Some((_, pack_dir)) = &best_pack {
        if let Ok(text) = fs::read_to_string(pack_dir.join("backend-manifest.json")) {
            if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(rel) = manifest
                    .get("engine_protocol_server")
                    .and_then(|s| s.get("executable_relative_path"))
                    .and_then(|v| v.as_str())
                {
                    let candidate = vendor.join(rel);
                    if candidate.is_file() {
                        python = Some(candidate);
                    }
                }
                version = manifest
                    .get("version")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned);
            }
        }
    }
    let python = match python {
        Some(p) => p,
        None => {
            let amphibian = vendor.join("_amphibian");
            let mut best_layer: Option<(u64, PathBuf)> = None;
            for entry in fs::read_dir(&amphibian).ok()?.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                let Some(build) = name
                    .strip_prefix("app-splash-")
                    .and_then(|rest| rest.rsplit('@').next())
                    .and_then(|n| n.parse::<u64>().ok())
                else {
                    continue;
                };
                let candidate = entry.path().join("bin").join("python");
                if !candidate.is_file() {
                    continue;
                }
                if best_layer.as_ref().map_or(true, |(b, _)| build > *b) {
                    best_layer = Some((build, candidate));
                }
            }
            best_layer?.1
        }
    };

    // The app layer is the interpreter's grandparent
    // (<layer>/bin/python); the native binary sits in its
    // engine/ directory.
    let app_layer = python.parent()?.parent()?.to_path_buf();
    let binary = app_layer.join("engine").join("splash");
    if !binary.is_file() {
        return None;
    }
    let version = version.unwrap_or_else(|| match &best_pack {
        Some(((a, b, c), _)) => format!("{a}.{b}.{c}"),
        None => "unknown".to_string(),
    });
    Some(SplashPack {
        version,
        python,
        app_layer,
        binary,
    })
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
        // The fixture GGUF is 100 zero bytes with a real name:
        // file/quantization/domain come from the name, and a
        // zero-filled header yields no metadata.
        assert_eq!(gguf.file_name.as_deref(), Some("test-q4_k_m.gguf"));
        assert_eq!(gguf.quantization.as_deref(), Some("Q4_K_M"));
        assert_eq!(gguf.domain, "llm");
        assert!(gguf.warnings.is_empty(), "warnings: {:?}", gguf.warnings);

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
            file_name: None,
            quantization: None,
            max_context: None,
            domain: "llm".to_string(),
            warnings: Vec::new(),
            remote_url: None,
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
    fn find_splash_pack_picks_highest_version_and_manifest_python() {
        let base = std::env::temp_dir().join(format!(
            "quack-core-splash-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // Two pack versions side by side; each manifest points
        // at its own amphibian layer. The newest pack must win.
        for (ver, build) in [("0.0.5", 2), ("1.3.0", 8)] {
            let pack = base.join(format!("splash-mac-arm64-apple-metal-advsimd-{ver}"));
            fs::create_dir_all(&pack).unwrap();
            let manifest = format!(
                "{{\"version\":\"{ver}\",\"engine_protocol_server\":{{\"runtime_kind\":\"splash-server\",\"executable_relative_path\":\"_amphibian/app-splash-mac26-arm64@{build}/bin/python\"}}}}"
            );
            fs::write(pack.join("backend-manifest.json"), manifest).unwrap();
            let layer = base
                .join("vendor")
                .join("_amphibian")
                .join(format!("app-splash-mac26-arm64@{build}"));
            fs::create_dir_all(layer.join("bin")).unwrap();
            write_bytes(&layer.join("bin").join("python"), 8);
            fs::create_dir_all(layer.join("engine")).unwrap();
            write_bytes(&layer.join("engine").join("splash"), 8);
        }

        let found = find_splash_pack(&base).expect("a splash pack");
        assert_eq!(found.version, "1.3.0");
        assert!(
            found
                .python
                .ends_with("app-splash-mac26-arm64@8/bin/python"),
            "picked: {:?}",
            found.python
        );
        assert!(
            found
                .binary
                .ends_with("app-splash-mac26-arm64@8/engine/splash"),
            "picked: {:?}",
            found.binary
        );

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn default_maya_providers_are_the_six_mac_endpoints() {
        let providers = default_maya_providers();
        let names: Vec<&str> = providers.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(
            names,
            ["glm", "minimax", "qwen", "qwen-native", "hy4", "ik2"]
        );
        assert_eq!(providers[0].base_url, "http://100.68.208.113:8080");
        assert_eq!(providers[5].base_url, "http://100.68.208.113:8085");
        // The Maya format serializes lowercase, like the others.
        let entry_format = serde_json::to_value(ModelFormat::Maya).unwrap();
        assert_eq!(entry_format, "maya");
    }

    #[test]
    fn gguf_metadata_parses_header() {
        // Hand-built header: architecture + context length +
        // file_type, plus an array value the parser must skip.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        bytes.extend_from_slice(&4u64.to_le_bytes()); // kv count
        fn kv(bytes: &mut Vec<u8>, key: &str, ty: u32) {
            bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
            bytes.extend_from_slice(key.as_bytes());
            bytes.extend_from_slice(&ty.to_le_bytes());
        }
        kv(&mut bytes, "general.architecture", 8);
        bytes.extend_from_slice(&8u64.to_le_bytes());
        bytes.extend_from_slice(b"testarch");
        kv(&mut bytes, "testarch.context_length", 4);
        bytes.extend_from_slice(&131072u32.to_le_bytes());
        kv(&mut bytes, "general.file_type", 4);
        bytes.extend_from_slice(&15u32.to_le_bytes());
        kv(&mut bytes, "some.array", 9);
        bytes.extend_from_slice(&4u32.to_le_bytes()); // elem: UINT32
        bytes.extend_from_slice(&2u64.to_le_bytes()); // count
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());

        let meta = parse_gguf_metadata(&bytes);
        assert_eq!(meta.architecture.as_deref(), Some("testarch"));
        assert_eq!(meta.context_length, Some(131072));
        assert_eq!(meta.file_type_label.as_deref(), Some("Q4_K_M"));

        // Garbage is not a GGUF.
        assert_eq!(parse_gguf_metadata(b"NOPE"), GgufMetadata::default());
        // Truncation mid-KV keeps what was gathered, never panics.
        let meta = parse_gguf_metadata(&bytes[..40]);
        assert_eq!(meta.architecture, None);
    }

    #[test]
    fn quantization_labels_come_from_names() {
        assert_eq!(
            quantization_from_file_name("Inkling-2-Mini-Q4_K_M.gguf").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(
            quantization_from_file_name("model-q8_0.gguf").as_deref(),
            Some("Q8_0")
        );
        assert_eq!(
            quantization_from_file_name("Ornith-1.5-9B-MLX-4bit").as_deref(),
            Some("4-bit")
        );
        assert_eq!(quantization_from_file_name("plain-model.gguf"), None);
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
