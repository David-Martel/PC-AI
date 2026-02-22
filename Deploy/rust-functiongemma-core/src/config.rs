/// Shared configuration loader for the FunctionGemma subsystem.
///
/// Loads `Config/pcai-functiongemma.json` from the working directory (or the
/// path supplied via the `PCAI_CONFIG_PATH` environment variable) and makes
/// the validated values available to all three crates (core, train, runtime).
///
/// # Environment variable overrides
///
/// | Variable            | Field overridden                        |
/// |---------------------|-----------------------------------------|
/// | `PCAI_CONFIG_PATH`  | Path to the JSON file itself            |
/// | `PCAI_ROUTER_ADDR`  | `runtime.router_addr`                   |
/// | `PCAI_ROUTER_MODEL` | `runtime.router_model`                  |
///
/// # Examples
///
/// ```rust,no_run
/// use rust_functiongemma_core::PcaiConfig;
///
/// let cfg = PcaiConfig::load().expect("failed to load config");
/// println!("router addr: {}", cfg.runtime.router_addr);
/// ```
use crate::error::PcaiError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Top-level config file shape
// ---------------------------------------------------------------------------

/// Full contents of `pcai-functiongemma.json`.
///
/// Only the `runtime` section is fully typed; `train` is kept as a raw
/// [`Value`] because the training crate owns its own strongly-typed
/// representation and may diverge independently.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcaiConfig {
    /// Router runtime settings.
    #[serde(default)]
    pub runtime: RuntimeConfig,

    /// Training settings (kept opaque; the train crate owns the schema).
    #[serde(default)]
    pub train: Value,
}

// ---------------------------------------------------------------------------
// Runtime section
// ---------------------------------------------------------------------------

/// Settings for the FunctionGemma router runtime process.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeConfig {
    /// Socket address the router listens on (e.g. `"127.0.0.1:8000"`).
    pub router_addr: String,

    /// Human-readable model identifier.
    pub router_model: String,

    /// File-system path to the model directory.
    pub router_model_path: String,

    /// Routing engine: `"model"` or `"heuristic"`.
    pub router_engine: String,

    /// Path to the tools JSON schema file.
    pub tools_path: String,

    /// Enable the KV cache.
    pub router_kv_cache: bool,

    /// KV cache quantisation scheme (`"int8"` or `null`).
    pub router_kv_cache_quant: Option<String>,

    /// Maximum KV cache sequence length.
    pub router_kv_cache_max_len: Option<usize>,

    /// Where to store KV cache entries: `"gpu"` or `"cpu"`.
    pub router_kv_cache_store: String,

    /// Enable streaming KV cache eviction.
    pub router_kv_cache_streaming: bool,

    /// Block length for streaming KV eviction (0 = disabled).
    pub router_kv_cache_block_len: usize,

    /// Default maximum tokens to generate per request.
    pub router_default_max_tokens: u32,

    /// Default sampling temperature.
    pub router_default_temperature: f64,

    /// Maximum sequence length accepted by the model.
    pub router_max_seq_len: Option<usize>,

    /// Enable flash attention (requires the `flash-attn` feature).
    pub router_flash_attn: bool,

    /// Enable candle QMatMul quantised projection.
    pub router_candle_qmatmul: bool,

    /// Dtype for candle QMatMul (e.g. `"q4_0"`).
    pub router_candle_qmatmul_dtype: Option<String>,

    /// Device selection: `"auto"`, `"cpu"`, or `"cuda:N"`.
    pub router_device: String,

    /// Explicit CUDA device index (overrides auto-detection).
    pub router_gpu: Option<usize>,

    /// Path to the LoRA adapter directory.
    pub router_lora_path: String,

    /// Allowed CUDA device indices (empty = all).
    pub cuda_visible_devices: Vec<usize>,

    /// Minimum VRAM (MiB) required to use a GPU.
    pub min_vram_mb: Option<u64>,

    /// Enable the CUDA memory pool.
    pub router_cuda_mem_pool: bool,

    /// Release threshold for the CUDA memory pool (MiB).
    pub router_cuda_mem_pool_release_threshold_mb: Option<u64>,

    /// Trim the CUDA memory pool to this size (MiB).
    pub router_cuda_mem_pool_trim_mb: Option<u64>,

    /// Log CUDA memory snapshots around key events.
    pub router_cuda_mem_snapshot: bool,

    /// Enable verbose structured logging.
    pub verbose: bool,

    /// `tracing` filter string (e.g. `"info"`, `"debug,hyper=warn"`).
    pub log_filter: String,

    /// Maximum concurrent inference requests (router queue depth).
    pub router_queue_depth: usize,

    /// Per-request timeout in seconds.
    pub router_request_timeout_secs: u64,

    /// Optional bearer token for API authentication.
    /// Leave empty (the default) to disable authentication.
    #[serde(default)]
    pub api_key: String,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            router_addr: "127.0.0.1:8000".to_string(),
            router_model: "functiongemma-270m-it".to_string(),
            router_model_path: String::new(),
            router_engine: "heuristic".to_string(),
            tools_path: "Config/pcai-tools.json".to_string(),
            router_kv_cache: true,
            router_kv_cache_quant: None,
            router_kv_cache_max_len: None,
            router_kv_cache_store: "gpu".to_string(),
            router_kv_cache_streaming: false,
            router_kv_cache_block_len: 0,
            router_default_max_tokens: 64,
            router_default_temperature: 0.1,
            router_max_seq_len: None,
            router_flash_attn: false,
            router_candle_qmatmul: false,
            router_candle_qmatmul_dtype: None,
            router_device: "auto".to_string(),
            router_gpu: None,
            router_lora_path: String::new(),
            cuda_visible_devices: Vec::new(),
            min_vram_mb: None,
            router_cuda_mem_pool: false,
            router_cuda_mem_pool_release_threshold_mb: None,
            router_cuda_mem_pool_trim_mb: None,
            router_cuda_mem_snapshot: false,
            verbose: false,
            log_filter: "info".to_string(),
            router_queue_depth: 4,
            router_request_timeout_secs: 30,
            api_key: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Loading logic
// ---------------------------------------------------------------------------

impl PcaiConfig {
    /// Resolve the config file path.
    ///
    /// Respects the `PCAI_CONFIG_PATH` env var; otherwise falls back to
    /// `Config/pcai-functiongemma.json` relative to the current working
    /// directory.
    pub fn config_path() -> PathBuf {
        if let Ok(p) = std::env::var("PCAI_CONFIG_PATH") {
            let p = p.trim().to_string();
            if !p.is_empty() {
                return PathBuf::from(p);
            }
        }
        PathBuf::from("Config/pcai-functiongemma.json")
    }

    /// Load and validate the configuration.
    ///
    /// If the file does not exist the function returns a fully-defaulted
    /// config rather than an error, matching the runtime's behaviour.
    ///
    /// Returns `Err(PcaiError)` only if the file exists but cannot be parsed.
    pub fn load() -> Result<Self, PcaiError> {
        Self::load_from(&Self::config_path())
    }

    /// Load from an explicit path.
    ///
    /// If the file does not exist, a fully-defaulted config is returned.
    pub fn load_from(path: &Path) -> Result<Self, PcaiError> {
        let mut cfg = match std::fs::read_to_string(path) {
            Ok(raw) => serde_json::from_str::<PcaiConfig>(&raw)?,
            // Missing file → use defaults (mirrors runtime behaviour)
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => PcaiConfig::default(),
            Err(e) => return Err(e.into()),
        };
        cfg.apply_env_overrides();
        Ok(cfg)
    }

    /// Apply any environment variable overrides to a loaded config.
    fn apply_env_overrides(&mut self) {
        if let Ok(addr) = std::env::var("PCAI_ROUTER_ADDR") {
            let addr = addr.trim().to_string();
            if !addr.is_empty() {
                self.runtime.router_addr = addr;
            }
        }
        if let Ok(model) = std::env::var("PCAI_ROUTER_MODEL") {
            let model = model.trim().to_string();
            if !model.is_empty() {
                self.runtime.router_model = model;
            }
        }
    }
}

impl Default for PcaiConfig {
    fn default() -> Self {
        Self {
            runtime: RuntimeConfig::default(),
            train: Value::Null,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Path to the real config file, relative to the crate root at compile time.
    ///
    /// `cargo test` runs with the working directory set to the crate root, so
    /// `../../Config/pcai-functiongemma.json` is correct for
    /// `Deploy/rust-functiongemma-core/`.
    const REAL_CONFIG_RELATIVE: &str = "../../Config/pcai-functiongemma.json";

    // ------------------------------------------------------------------
    // Loading the actual config file
    // ------------------------------------------------------------------

    #[test]
    fn load_real_config_file() {
        let path = PathBuf::from(REAL_CONFIG_RELATIVE);
        if !path.exists() {
            eprintln!("skipping: config file not found at {}", path.display());
            return;
        }
        let cfg = PcaiConfig::load_from(&path).expect("config should parse");
        // Spot-check a few known values from the JSON.
        assert_eq!(cfg.runtime.router_addr, "127.0.0.1:8000");
        assert_eq!(cfg.runtime.router_model, "functiongemma-270m-it");
        assert_eq!(cfg.runtime.router_default_max_tokens, 64);
        assert!((cfg.runtime.router_default_temperature - 0.1).abs() < f64::EPSILON);
        assert!(cfg.runtime.router_kv_cache);
        assert_eq!(cfg.runtime.router_device, "auto");
    }

    #[test]
    fn real_config_train_section_is_object() {
        let path = PathBuf::from(REAL_CONFIG_RELATIVE);
        if !path.exists() {
            return;
        }
        let cfg = PcaiConfig::load_from(&path).expect("config should parse");
        // The `train` section should deserialise as a JSON object.
        assert!(
            cfg.train.is_object(),
            "expected train to be a JSON object, got: {:?}",
            cfg.train
        );
    }

    // ------------------------------------------------------------------
    // Env var overrides
    // ------------------------------------------------------------------

    /// Run a closure with a specific env var set, then restore the original
    /// value.  Using a process-level lock avoids data races when tests run in
    /// parallel threads.
    fn with_env_var<F: FnOnce()>(key: &str, value: &str, f: F) {
        // SAFETY: only one thread modifies env vars at a time via the mutex.
        use std::sync::Mutex;
        static ENV_LOCK: Mutex<()> = Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap();
        let old = std::env::var(key).ok();
        std::env::set_var(key, value);
        f();
        match old {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }

    #[test]
    fn env_override_router_addr() {
        with_env_var("PCAI_ROUTER_ADDR", "0.0.0.0:9999", || {
            // Load from a non-existent path to get defaults, then apply env.
            let cfg = PcaiConfig::load_from(Path::new("/nonexistent/config.json"))
                .expect("should return defaults on missing file");
            assert_eq!(cfg.runtime.router_addr, "0.0.0.0:9999");
        });
    }

    #[test]
    fn env_override_router_model() {
        with_env_var("PCAI_ROUTER_MODEL", "my-custom-model", || {
            let cfg = PcaiConfig::load_from(Path::new("/nonexistent/config.json"))
                .expect("should return defaults on missing file");
            assert_eq!(cfg.runtime.router_model, "my-custom-model");
        });
    }

    #[test]
    fn empty_env_var_does_not_override() {
        with_env_var("PCAI_ROUTER_ADDR", "", || {
            let cfg =
                PcaiConfig::load_from(Path::new("/nonexistent/config.json")).expect("defaults");
            // Empty override must not replace the default.
            assert_eq!(
                cfg.runtime.router_addr,
                RuntimeConfig::default().router_addr
            );
        });
    }

    // ------------------------------------------------------------------
    // Default values
    // ------------------------------------------------------------------

    #[test]
    fn defaults_are_sensible() {
        let cfg = PcaiConfig::default();
        assert_eq!(cfg.runtime.router_addr, "127.0.0.1:8000");
        assert_eq!(cfg.runtime.router_engine, "heuristic");
        assert_eq!(cfg.runtime.router_default_max_tokens, 64);
        assert!(!cfg.runtime.router_flash_attn);
        assert!(!cfg.runtime.verbose);
        assert!(cfg.runtime.cuda_visible_devices.is_empty());
        assert!(cfg.runtime.router_gpu.is_none());
        assert!(cfg.train.is_null());
    }

    #[test]
    fn missing_file_returns_defaults() {
        let cfg = PcaiConfig::load_from(Path::new("/nonexistent/path/config.json"))
            .expect("should return defaults, not an error");
        assert_eq!(
            cfg.runtime.router_addr,
            RuntimeConfig::default().router_addr
        );
    }

    // ------------------------------------------------------------------
    // Partial JSON (missing optional fields) should not error
    // ------------------------------------------------------------------

    #[test]
    fn partial_json_uses_field_defaults() {
        let json = r#"{"runtime": {"router_addr": "127.0.0.1:1234"}}"#;
        let cfg: PcaiConfig = serde_json::from_str(json).expect("partial config should parse");
        assert_eq!(cfg.runtime.router_addr, "127.0.0.1:1234");
        // Fields not present in the JSON get their defaults.
        assert_eq!(
            cfg.runtime.router_model,
            RuntimeConfig::default().router_model
        );
        assert_eq!(
            cfg.runtime.router_default_max_tokens,
            RuntimeConfig::default().router_default_max_tokens
        );
    }

    #[test]
    fn train_section_absent_is_null() {
        let json = r#"{"runtime": {}}"#;
        let cfg: PcaiConfig = serde_json::from_str(json).expect("should parse");
        assert!(cfg.train.is_null());
    }
}
