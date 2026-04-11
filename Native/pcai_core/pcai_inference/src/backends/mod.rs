//! Inference backend implementations

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

#[cfg(feature = "llamacpp")]
pub mod llamacpp;

#[cfg(feature = "mistralrs-backend")]
pub mod mistralrs;

/// Request for text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// The input prompt to generate a continuation for.
    pub prompt: String,
    /// Maximum number of tokens to generate. `None` uses the backend default.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Sampling temperature (0.0 = greedy, 1.0 = creative). `None` uses the backend default.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold. `None` uses the backend default.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Stop sequences that terminate generation early.
    #[serde(default)]
    pub stop: Vec<String>,
}

/// Response from text generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    /// The generated text continuation.
    pub text: String,
    /// Number of tokens produced by the backend.
    pub tokens_generated: usize,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
}

/// Reason for generation completion
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// A stop sequence was encountered or the end-of-sequence token was generated.
    Stop,
    /// The `max_tokens` limit was reached before a stop condition.
    Length,
    /// The backend returned an error during generation.
    Error,
}

/// Trait for inference backends
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Load a model from the given path
    async fn load_model(&mut self, model_path: &str) -> Result<()>;

    /// Generate text from a prompt
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse>;

    /// Generate text from a prompt with streaming callback.
    /// Default implementation falls back to non-streaming generation.
    async fn generate_streaming(
        &self,
        request: GenerateRequest,
        callback: &mut (dyn FnMut(String) + Send),
    ) -> Result<GenerateResponse> {
        let response = self.generate(request).await?;
        if !response.text.is_empty() {
            callback(response.text.clone());
        }
        Ok(response)
    }

    /// Unload the current model
    async fn unload_model(&mut self) -> Result<()>;

    /// Check if a model is loaded
    fn is_loaded(&self) -> bool;

    /// Get backend name
    fn backend_name(&self) -> &'static str;
}

/// Discriminant used to select which backend implementation to instantiate.
pub enum BackendType {
    /// The llama.cpp backend (enabled by the `llamacpp` feature).
    #[cfg(feature = "llamacpp")]
    LlamaCpp,

    /// The mistral.rs backend (enabled by the `mistralrs-backend` feature).
    #[cfg(feature = "mistralrs-backend")]
    MistralRs,
}

impl BackendType {
    /// Construct a boxed [`InferenceBackend`] for this variant.
    ///
    /// Returns an error if no backend feature is compiled in.
    pub fn create(&self) -> Result<Box<dyn InferenceBackend>> {
        match self {
            #[cfg(feature = "llamacpp")]
            BackendType::LlamaCpp => Ok(Box::new(llamacpp::LlamaCppBackend::new())),

            #[cfg(feature = "mistralrs-backend")]
            BackendType::MistralRs => Ok(Box::new(mistralrs::MistralRsBackend::new())),

            // The catch-all arm is a fallthrough guard for builds compiled without
            // any backend feature — without at least one `#[cfg(feature="...")]`
            // arm matching, the match is non-exhaustive. When `llamacpp` or
            // `mistralrs-backend` IS enabled, clippy's exhaustiveness check
            // considers the `_` pattern unreachable; when BOTH are disabled, it's
            // the only arm. Use `#[cfg_attr]` to silence `unreachable_patterns`
            // only in the configurations where it actually triggers.
            #[cfg_attr(
                any(feature = "llamacpp", feature = "mistralrs-backend"),
                allow(unreachable_patterns)
            )]
            _ => Err(Error::Backend("No backend feature enabled".to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_request_serde_minimal() {
        let json = r#"{"prompt": "Hello"}"#;
        let req: GenerateRequest =
            serde_json::from_str(json).expect("test: minimal GenerateRequest must deserialize from valid JSON");
        assert_eq!(req.prompt, "Hello");
        assert!(req.max_tokens.is_none());
        assert!(req.temperature.is_none());
        assert!(req.top_p.is_none());
        assert!(req.stop.is_empty());
    }

    #[test]
    fn test_generate_request_serde_full() {
        let req = GenerateRequest {
            prompt: "Test prompt".to_string(),
            max_tokens: Some(256),
            temperature: Some(0.8),
            top_p: Some(0.9),
            stop: vec!["END".to_string(), "\n".to_string()],
        };
        let json = serde_json::to_string(&req).expect("test: GenerateRequest must serialize to JSON");
        let deserialized: GenerateRequest =
            serde_json::from_str(&json).expect("test: GenerateRequest must roundtrip through JSON");
        assert_eq!(deserialized.prompt, "Test prompt");
        assert_eq!(deserialized.max_tokens, Some(256));
        assert!((deserialized.temperature.unwrap() - 0.8).abs() < f32::EPSILON);
        assert!((deserialized.top_p.unwrap() - 0.9).abs() < f32::EPSILON);
        assert_eq!(deserialized.stop, vec!["END", "\n"]);
    }

    #[test]
    fn test_generate_response_serde() {
        let resp = GenerateResponse {
            text: "Generated output".to_string(),
            tokens_generated: 42,
            finish_reason: FinishReason::Stop,
        };
        let json = serde_json::to_string(&resp).expect("test: GenerateResponse must serialize to JSON");
        let deserialized: GenerateResponse =
            serde_json::from_str(&json).expect("test: GenerateResponse must roundtrip through JSON");
        assert_eq!(deserialized.text, "Generated output");
        assert_eq!(deserialized.tokens_generated, 42);
        assert!(matches!(deserialized.finish_reason, FinishReason::Stop));
    }

    #[test]
    fn test_finish_reason_serialization() {
        assert_eq!(serde_json::to_string(&FinishReason::Stop).unwrap(), "\"stop\"");
        assert_eq!(serde_json::to_string(&FinishReason::Length).unwrap(), "\"length\"");
        assert_eq!(serde_json::to_string(&FinishReason::Error).unwrap(), "\"error\"");
    }

    #[test]
    fn test_finish_reason_deserialization() {
        let stop: FinishReason =
            serde_json::from_str("\"stop\"").expect("test: FinishReason::Stop must deserialize from \"stop\"");
        assert!(matches!(stop, FinishReason::Stop));
        let length: FinishReason =
            serde_json::from_str("\"length\"").expect("test: FinishReason::Length must deserialize from \"length\"");
        assert!(matches!(length, FinishReason::Length));
        let error: FinishReason =
            serde_json::from_str("\"error\"").expect("test: FinishReason::Error must deserialize from \"error\"");
        assert!(matches!(error, FinishReason::Error));
    }

    #[test]
    fn test_generate_request_default_stop_empty() {
        let req: GenerateRequest = serde_json::from_str(r#"{"prompt": "x"}"#)
            .expect("test: GenerateRequest must deserialize with only prompt field set");
        assert!(req.stop.is_empty());
    }

    #[test]
    #[cfg(not(any(feature = "llamacpp", feature = "mistralrs-backend")))]
    fn test_backend_type_no_features() {
        // With no backend features enabled, BackendType has no variants,
        // so we can't construct one. The enum is effectively empty.
        // This test verifies the module compiles correctly without backends.
        assert!(
            true,
            "BackendType correctly has no constructible variants without features"
        );
    }
}
