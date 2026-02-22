use crate::types::RouterEngine;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tracing_subscriber::EnvFilter;

use rust_functiongemma_core::{PcaiConfig, RuntimeConfig};

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

pub(crate) fn router_model_path_override() -> Option<String> {
    let value = runtime_config().router_model_path.trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

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
