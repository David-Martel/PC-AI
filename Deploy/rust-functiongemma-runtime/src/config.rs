use crate::types::RouterEngine;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tracing_subscriber::EnvFilter;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub(crate) struct PcaiConfig {
    pub(crate) runtime: RuntimeConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub(crate) struct RuntimeConfig {
    pub(crate) router_addr: String,
    pub(crate) router_model: String,
    pub(crate) router_model_path: String,
    pub(crate) router_engine: String,
    pub(crate) tools_path: String,
    pub(crate) router_kv_cache: bool,
    pub(crate) router_kv_cache_quant: Option<String>,
    pub(crate) router_kv_cache_max_len: Option<usize>,
    pub(crate) router_kv_cache_store: String,
    pub(crate) router_default_max_tokens: u32,
    pub(crate) router_default_temperature: f64,
    pub(crate) router_seed: Option<u64>,
    pub(crate) router_max_seq_len: Option<usize>,
    pub(crate) router_flash_attn: bool,
    pub(crate) router_candle_qmatmul: bool,
    pub(crate) router_candle_qmatmul_dtype: Option<String>,
    pub(crate) router_device: String,
    pub(crate) router_gpu: Option<usize>,
    pub(crate) router_lora_path: String,
    pub(crate) cuda_visible_devices: Vec<usize>,
    pub(crate) min_vram_mb: Option<u64>,
    pub(crate) router_cuda_mem_pool: bool,
    pub(crate) router_cuda_mem_pool_release_threshold_mb: Option<u64>,
    pub(crate) router_cuda_mem_pool_trim_mb: Option<u64>,
    pub(crate) router_cuda_mem_snapshot: bool,
    pub(crate) verbose: bool,
    pub(crate) log_filter: String,
    pub(crate) router_queue_depth: usize,
    pub(crate) router_request_timeout_secs: u64,
    pub(crate) api_key: String, // pragma: allowlist secret (field name, loaded at runtime)
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
            router_default_max_tokens: 64,
            router_default_temperature: 0.1,
            router_seed: None,
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
            api_key: String::new(), // pragma: allowlist secret (empty default)
        }
    }
}

impl PcaiConfig {
    pub(crate) fn config_path() -> PathBuf {
        if let Ok(p) = std::env::var("PCAI_CONFIG_PATH") {
            let p = p.trim().to_string();
            if !p.is_empty() {
                return PathBuf::from(p);
            }
        }
        PathBuf::from("Config/pcai-functiongemma.json")
    }

    pub(crate) fn load_from(path: &Path) -> anyhow::Result<Self> {
        let raw = match std::fs::read_to_string(path) {
            Ok(raw) => raw,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(PcaiConfig::default()),
            Err(e) => return Err(e.into()),
        };
        let mut cfg: PcaiConfig = serde_json::from_str(&raw)?;
        cfg.apply_env_overrides();
        Ok(cfg)
    }

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

/// Returns the configured API key, or `None` when authentication is disabled.
///
/// Resolution order (first non-empty value wins):
/// 1. `runtime.api_key` in the JSON config file.
/// 2. `PCAI_API_KEY` environment variable.
///
/// Returns `None` if both sources are empty, meaning all requests are allowed
/// without an `Authorization` header.
pub(crate) fn api_key() -> Option<String> {
    let from_config = runtime_config().api_key.trim().to_string();
    if !from_config.is_empty() {
        return Some(from_config);
    }
    let from_env = std::env::var("PCAI_API_KEY").unwrap_or_default();
    if !from_env.trim().is_empty() {
        return Some(from_env.trim().to_string());
    }
    None
}

pub(crate) static RUNTIME_CONFIG: OnceLock<RuntimeConfig> = OnceLock::new();

pub(crate) fn load_runtime_config(path: &Path) -> RuntimeConfig {
    match PcaiConfig::load_from(path) {
        Ok(cfg) => cfg.runtime,
        Err(_) => RuntimeConfig::default(),
    }
}

pub(crate) fn runtime_config() -> &'static RuntimeConfig {
    RUNTIME_CONFIG.get_or_init(|| load_runtime_config(&PcaiConfig::config_path()))
}

pub fn init_runtime_config<P: AsRef<Path>>(path: P) {
    if let Some(existing) = RUNTIME_CONFIG.get() {
        let _ = existing;
        return;
    }
    let loaded = load_runtime_config(path.as_ref());
    let _ = RUNTIME_CONFIG.set(loaded);
}

pub fn runtime_addr() -> anyhow::Result<std::net::SocketAddr> {
    let raw = runtime_config().router_addr.trim();
    if raw.eq_ignore_ascii_case("auto") {
        return Ok("127.0.0.1:0".parse().expect("valid auto address"));
    }
    raw.parse().map_err(anyhow::Error::msg)
}

#[cfg(feature = "model")]
pub(crate) fn router_model_path_override() -> Option<String> {
    let value = runtime_config().router_model_path.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

#[cfg(feature = "model")]
pub(crate) fn router_lora_path_override() -> Option<PathBuf> {
    let value = runtime_config().router_lora_path.trim();
    if value.is_empty() {
        None
    } else {
        Some(PathBuf::from(value))
    }
}

pub(crate) fn build_log_filter() -> EnvFilter {
    let raw = runtime_config().log_filter.trim();
    if raw.is_empty() {
        return EnvFilter::new("info");
    }
    EnvFilter::try_new(raw).unwrap_or_else(|_| EnvFilter::new("info"))
}

pub(crate) fn default_model() -> String {
    runtime_config().router_model.clone()
}

pub(crate) fn router_engine() -> RouterEngine {
    match runtime_config().router_engine.to_lowercase().as_str() {
        "model" => RouterEngine::Model,
        _ => RouterEngine::Heuristic,
    }
}

#[cfg(feature = "model")]
pub(crate) fn is_verbose() -> bool {
    runtime_config().verbose
}
