//! # pcai-inference
//!
//! Dual-backend LLM inference engine for PC diagnostics.
//!
//! Supports two backends:
//! - llama.cpp via llama-cpp-2 (feature: `llamacpp`)
//! - mistral.rs (feature: `mistralrs-backend`)
//!
//! Optional features:
//! - `cuda`: Enable GPU acceleration
//! - `server`: HTTP server with Axum
//! - `ffi`: C FFI exports for PowerShell integration

pub mod backends;
pub mod config;
pub mod version;

#[cfg(feature = "server")]
pub mod http;

#[cfg(feature = "ffi")]
pub mod ffi;

pub use backends::InferenceBackend;
pub use config::InferenceConfig;

/// Result type for inference operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for inference operations
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An error reported by the active inference backend (e.g. llama.cpp or mistral.rs).
    #[error("Backend error: {0}")]
    Backend(String),

    /// Invalid or missing configuration (bad JSON schema, missing required field, etc.).
    #[error("Configuration error: {0}")]
    Config(String),

    /// A generation was attempted before a model was loaded via `load_model`.
    #[error("Model not loaded")]
    ModelNotLoaded,

    /// The caller supplied a malformed argument (null pointer, oversized prompt, etc.).
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// An I/O error from the standard library (file not found, permission denied, etc.).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// A JSON serialization or deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Any other error that does not fit the categories above.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

// Candle error conversion for mistral.rs backend
// Note: candle_core::Error is not directly accessible, but we can convert via anyhow
// which is already implemented in the Other variant

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_backend() {
        let err = Error::Backend("gpu out of memory".to_string());
        assert_eq!(err.to_string(), "Backend error: gpu out of memory");
    }

    #[test]
    fn test_error_display_model_not_loaded() {
        let err = Error::ModelNotLoaded;
        assert_eq!(err.to_string(), "Model not loaded");
    }

    #[test]
    fn test_error_display_invalid_input() {
        let err = Error::InvalidInput("empty prompt".to_string());
        assert_eq!(err.to_string(), "Invalid input: empty prompt");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
        assert!(err.to_string().contains("file missing"));
    }

    #[test]
    fn test_error_from_serde_json() {
        let bad_json = serde_json::from_str::<serde_json::Value>("not json");
        let serde_err = bad_json.unwrap_err();
        let err: Error = serde_err.into();
        assert!(matches!(err, Error::Serialization(_)));
    }
}
