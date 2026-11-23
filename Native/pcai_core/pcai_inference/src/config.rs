//! Configuration types for inference engine

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for the inference engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Backend to use
    pub backend: BackendConfig,

    /// Model configuration
    pub model: ModelConfig,

    /// Router/tool-call configuration
    #[serde(default)]
    pub router: RouterConfig,

    /// Server configuration (if feature = "server")
    #[cfg(feature = "server")]
    pub server: Option<ServerConfig>,
}

/// Backend selection
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BackendConfig {
    #[cfg(feature = "llamacpp")]
    LlamaCpp {
        #[serde(default)]
        n_gpu_layers: Option<i32>,
        #[serde(default)]
        n_ctx: Option<usize>,
    },

    #[cfg(feature = "mistralrs-backend")]
    MistralRs {
        #[serde(default)]
        device: Option<String>,
    },
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to model file
    pub path: PathBuf,

    /// Model type hint (optional)
    pub model_type: Option<String>,

    /// Default generation parameters
    #[serde(default)]
    pub generation: GenerationDefaults,
}

/// Default generation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationDefaults {
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    #[serde(default = "default_temperature")]
    pub temperature: f32,

    #[serde(default = "default_top_p")]
    pub top_p: f32,

    #[serde(default)]
    pub stop: Vec<String>,
}

impl Default for GenerationDefaults {
    fn default() -> Self {
        Self {
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            stop: vec![],
        }
    }
}

fn default_max_tokens() -> usize {
    512
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.95
}

/// Router configuration passed to the HTTP service at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    #[serde(default = "default_router_enabled")]
    pub enabled: bool,

    #[serde(default = "default_router_provider")]
    pub provider: String,

    #[serde(default)]
    pub base_url: String,

    #[serde(default = "default_router_model")]
    pub model: String,

    #[serde(default)]
    pub tools_path: String,

    #[serde(default)]
    pub strict: bool,

    #[serde(default)]
    pub force: bool,

    #[serde(default)]
    pub disable: bool,

    #[serde(default = "default_router_temperature")]
    pub default_temperature: f64,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            enabled: default_router_enabled(),
            provider: default_router_provider(),
            base_url: String::new(),
            model: default_router_model(),
            tools_path: String::new(),
            strict: false,
            force: false,
            disable: false,
            default_temperature: default_router_temperature(),
        }
    }
}

fn default_router_enabled() -> bool {
    true
}

fn default_router_provider() -> String {
    "functiongemma".to_string()
}

fn default_router_model() -> String {
    "functiongemma-270m-it".to_string()
}

fn default_router_temperature() -> f64 {
    0.2
}

/// Server configuration
///
/// The server binds to `host:port`, which must be an IP literal (IPv4, IPv6 with or without
/// brackets, or a scoped link-local such as `fe80::1%12`); `localhost` is accepted and means
/// `127.0.0.1`. Binding anything other than a loopback address requires an authentication token.
#[cfg(feature = "server")]
#[derive(Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Host to bind to
    #[serde(default = "default_host")]
    pub host: String,

    /// Port to bind to
    #[serde(default = "default_port")]
    pub port: u16,

    /// Legacy `"cors": true|false` key. It used to switch on a wildcard CORS layer. The value is
    /// still accepted so existing configuration files load, but it no longer enables anything;
    /// use [`ServerConfig::cors_allowed_origins`].
    #[serde(default, rename = "cors", skip_serializing)]
    pub legacy_cors: Option<bool>,

    /// Browser origins (for example `https://tools.example:3000`) that may call the API
    /// cross-origin. Empty (the default) sends no CORS headers and refuses any request that
    /// carries an `Origin` header.
    #[serde(default)]
    pub cors_allowed_origins: Vec<String>,

    /// Extra `Host` header values to accept besides `localhost`, `127.0.0.1`, `[::1]` and the
    /// bind address. Required when binding a wildcard address (`0.0.0.0`, `::`).
    #[serde(default)]
    pub allowed_hosts: Vec<String>,

    /// Bearer token required on every route except `GET /health`. The environment variable
    /// `PCAI_INFERENCE_TOKEN` overrides it. Never serialized, so `to_file` cannot leak it.
    #[serde(default, skip_serializing)]
    pub auth_token: Option<String>,
}

#[cfg(feature = "server")]
impl std::fmt::Debug for ServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerConfig")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("legacy_cors", &self.legacy_cors)
            .field("cors_allowed_origins", &self.cors_allowed_origins)
            .field("allowed_hosts", &self.allowed_hosts)
            .field("auth_token", &self.auth_token.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

#[cfg(feature = "server")]
fn default_host() -> String {
    "127.0.0.1".to_string()
}

#[cfg(feature = "server")]
fn default_port() -> u16 {
    8080
}

#[cfg(feature = "server")]
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            legacy_cors: None,
            cors_allowed_origins: Vec::new(),
            allowed_hosts: Vec::new(),
            auth_token: None,
        }
    }
}

impl InferenceConfig {
    /// Load configuration from a file
    pub fn from_file(path: impl AsRef<std::path::Path>) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to a file
    pub fn to_file(&self, path: impl AsRef<std::path::Path>) -> crate::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_defaults() {
        let defaults = GenerationDefaults::default();
        assert_eq!(defaults.max_tokens, 512);
        assert!((defaults.temperature - 0.7).abs() < f32::EPSILON);
        assert!((defaults.top_p - 0.95).abs() < f32::EPSILON);
        assert!(defaults.stop.is_empty());
    }

    #[test]
    fn test_generation_defaults_serde_roundtrip() {
        let defaults = GenerationDefaults::default();
        let json = serde_json::to_string(&defaults).expect("test: GenerationDefaults must serialize to JSON");
        let deserialized: GenerationDefaults =
            serde_json::from_str(&json).expect("test: GenerationDefaults must roundtrip through JSON");
        assert_eq!(deserialized.max_tokens, defaults.max_tokens);
        assert!((deserialized.temperature - defaults.temperature).abs() < f32::EPSILON);
        assert!((deserialized.top_p - defaults.top_p).abs() < f32::EPSILON);
        assert_eq!(deserialized.stop, defaults.stop);
    }

    #[test]
    fn test_model_config_serde() {
        let config = ModelConfig {
            path: PathBuf::from("models/test.gguf"),
            model_type: Some("llama".to_string()),
            generation: GenerationDefaults::default(),
        };
        let json = serde_json::to_string(&config).expect("test: ModelConfig must serialize to JSON");
        let deserialized: ModelConfig =
            serde_json::from_str(&json).expect("test: ModelConfig must roundtrip through JSON");
        assert_eq!(deserialized.path, PathBuf::from("models/test.gguf"));
        assert_eq!(deserialized.model_type, Some("llama".to_string()));
    }

    #[test]
    fn test_config_from_file_nonexistent() {
        let result = InferenceConfig::from_file("/nonexistent/path/config.json");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::Error::Io(_)));
    }

    #[test]
    fn test_config_from_file_invalid_json() {
        let dir = tempfile::tempdir().expect("test: tempdir creation must succeed");
        let path = dir.path().join("bad.json");
        std::fs::write(&path, "not valid json {{{")
            .expect("test: writing invalid JSON fixture to tempdir must succeed");
        let result = InferenceConfig::from_file(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::Error::Serialization(_)));
    }

    #[cfg(feature = "server")]
    #[test]
    fn test_server_config_defaults() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8080);
        assert!(config.cors_allowed_origins.is_empty());
        assert!(config.allowed_hosts.is_empty());
        assert!(config.auth_token.is_none());
    }

    #[test]
    fn test_generation_defaults_serde_with_overrides() {
        // Partial JSON: only max_tokens is set, rest should use defaults
        let json = r#"{"max_tokens": 1024}"#;
        let deserialized: GenerationDefaults = serde_json::from_str(json)
            .expect("test: GenerationDefaults must deserialize from partial JSON using serde defaults");
        assert_eq!(deserialized.max_tokens, 1024);
        assert!((deserialized.temperature - 0.7).abs() < f32::EPSILON);
        assert!((deserialized.top_p - 0.95).abs() < f32::EPSILON);
        assert!(deserialized.stop.is_empty());
    }

    #[test]
    fn test_model_config_minimal_serde() {
        // Only required field 'path' - model_type defaults to None, generation to Default
        let json = r#"{"path": "model.gguf"}"#;
        let config: ModelConfig =
            serde_json::from_str(json).expect("test: ModelConfig must deserialize with only the required path field");
        assert_eq!(config.path, PathBuf::from("model.gguf"));
        assert!(config.model_type.is_none());
        assert_eq!(config.generation.max_tokens, 512);
    }
}
