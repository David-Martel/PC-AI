/// Shared error type for the FunctionGemma subsystem.
///
/// All three crates (core, train, runtime) can convert their local errors into
/// `PcaiError` for a uniform error surface.
///
/// # Examples
///
/// ```rust
/// use rust_functiongemma_core::PcaiError;
///
/// let e: PcaiError = PcaiError::Config("missing router_addr".into());
/// assert!(e.to_string().contains("missing router_addr"));
/// ```
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum PcaiError {
    /// A configuration value is invalid or missing.
    #[error("Config error: {0}")]
    Config(String),

    /// A candle model operation failed.
    ///
    /// Stored as a `String` rather than `candle_core::Error` so that
    /// `PcaiError` remains `Send + Sync` regardless of the platform.
    #[error("Model error: {0}")]
    Model(String),

    /// A template file could not be found or opened.
    #[error("Template not found: {path}")]
    TemplateNotFound {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// A generic I/O error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// A JSON serialisation / deserialisation error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// A catch-all for errors from crates that already use `anyhow`.
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

impl From<candle_core::Error> for PcaiError {
    fn from(e: candle_core::Error) -> Self {
        PcaiError::Model(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Display ---

    #[test]
    fn config_variant_display() {
        let e = PcaiError::Config("missing field".into());
        assert_eq!(e.to_string(), "Config error: missing field");
    }

    #[test]
    fn model_variant_display() {
        let e = PcaiError::Model("tensor shape mismatch".into());
        assert_eq!(e.to_string(), "Model error: tensor shape mismatch");
    }

    #[test]
    fn template_not_found_display() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let e = PcaiError::TemplateNotFound {
            path: PathBuf::from("/tmp/foo.j2"),
            source: io_err,
        };
        assert!(e.to_string().contains("/tmp/foo.j2"));
    }

    #[test]
    fn io_variant_display() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let e: PcaiError = io_err.into();
        assert!(e.to_string().contains("IO error"));
    }

    #[test]
    fn json_variant_display() {
        let json_err: serde_json::Error = serde_json::from_str::<serde_json::Value>("bad{").unwrap_err();
        let e: PcaiError = json_err.into();
        assert!(e.to_string().contains("JSON error"));
    }

    #[test]
    fn other_variant_display() {
        let e: PcaiError = anyhow::anyhow!("something went wrong").into();
        assert_eq!(e.to_string(), "something went wrong");
    }

    // --- From conversions ---

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "not found");
        let _e: PcaiError = io_err.into();
    }

    #[test]
    fn from_serde_json_error() {
        let json_err: serde_json::Error = serde_json::from_str::<i32>("not_a_number").unwrap_err();
        let _e: PcaiError = json_err.into();
    }

    #[test]
    fn from_anyhow_error() {
        let anyhow_err = anyhow::anyhow!("anyhow error");
        let _e: PcaiError = anyhow_err.into();
    }

    #[test]
    fn from_candle_error() {
        // candle_core::Error can be constructed via the public Error::Msg variant.
        let candle_err = candle_core::Error::msg("candle test error");
        let e: PcaiError = candle_err.into();
        assert!(matches!(e, PcaiError::Model(_)));
        assert!(e.to_string().contains("candle test error"));
    }

    // --- Send + Sync (compile-time check) ---

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn pcai_error_is_send_sync() {
        assert_send_sync::<PcaiError>();
    }
}
