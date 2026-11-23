//! Inference backend implementations

use std::sync::atomic::AtomicBool;

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

    /// Generate text from a prompt, stopping early once `cancel` becomes `true`.
    ///
    /// Backends that can observe the flag between tokens should override this and
    /// return the partial output with [`FinishReason::Stop`]. The default
    /// implementation ignores the flag and runs [`generate`](Self::generate) to
    /// completion, so callers must still be prepared to discard a late result.
    async fn generate_cancellable(&self, request: GenerateRequest, cancel: &AtomicBool) -> Result<GenerateResponse> {
        let _ = cancel;
        self.generate(request).await
    }

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

/// Prompt of the post-load generation probe. Short and fixed so the probe is cheap and repeatable.
const PROBE_PROMPT: &str = "Reply with the single word: ok";

/// Tokens the probe asks for. A few are enough to run the prompt and decode steps on every layer.
const PROBE_MAX_TOKENS: usize = 4;

/// Longest the probe may take. Generous because a CPU run of a large model is slow, but finite so
/// a wedged device cannot keep a server from ever reporting its state.
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

/// Why a generation probe failed.
///
/// `Display` carries the full detail and is meant for logs only: it can hold file paths and other
/// diagnostics from the backend. Anything shown to an unauthenticated caller must use the variant,
/// never the text.
#[derive(Debug)]
pub enum ProbeFailure {
    /// The backend returned an error for the probe request.
    Failed(Error),
    /// The probe did not finish within its time limit.
    TimedOut(std::time::Duration),
}

impl std::fmt::Display for ProbeFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Failed(error) => write!(f, "the generation probe failed: {error}"),
            Self::TimedOut(limit) => write!(
                f,
                "the generation probe did not finish within {:.0} s",
                limit.as_secs_f64()
            ),
        }
    }
}

impl std::error::Error for ProbeFailure {}

/// Run one short, deterministic generation to prove the loaded model can answer a request.
///
/// A backend that reports [`InferenceBackend::is_loaded`] has only shown that the weights are in
/// memory. Failures that appear at the first forward pass (for example a layer split across the
/// CPU and a GPU, which mistral.rs reports as a device mismatch) would otherwise surface as an
/// HTTP 500 on every request while the server claims to be ready. Only success or failure
/// matters here, so the output text is not inspected.
///
/// # Errors
///
/// [`ProbeFailure::Failed`] with the backend's own error, or [`ProbeFailure::TimedOut`] when the
/// probe exceeds its time limit.
pub async fn probe_generation(backend: &dyn InferenceBackend) -> std::result::Result<(), ProbeFailure> {
    probe_within(backend, PROBE_TIMEOUT).await
}

async fn probe_within(
    backend: &dyn InferenceBackend,
    limit: std::time::Duration,
) -> std::result::Result<(), ProbeFailure> {
    let request = GenerateRequest {
        prompt: PROBE_PROMPT.to_string(),
        max_tokens: Some(PROBE_MAX_TOKENS),
        temperature: Some(0.0),
        top_p: None,
        stop: Vec::new(),
    };
    match tokio::time::timeout(limit, backend.generate(request)).await {
        Ok(result) => result.map(|_| ()).map_err(ProbeFailure::Failed),
        Err(_) => Err(ProbeFailure::TimedOut(limit)),
    }
}

/// Probe a freshly loaded backend and unload it when the probe fails.
///
/// After an `Err` the backend reports `is_loaded() == false`, so nothing downstream can mistake a
/// model that cannot generate for a ready one. The full error is logged here and returned as a
/// [`ProbeFailure`], never replaced by a success message.
///
/// # Errors
///
/// The [`ProbeFailure`]. Its text is for logs; callers that answer remote clients must publish a
/// fixed message chosen from the variant instead.
pub async fn probe_or_unload(backend: &mut dyn InferenceBackend) -> std::result::Result<(), ProbeFailure> {
    match probe_generation(&*backend).await {
        Ok(()) => Ok(()),
        Err(failure) => {
            tracing::error!(error = %failure, backend = backend.backend_name(), "model loaded but the generation probe failed");
            if let Err(unload_error) = backend.unload_model().await {
                tracing::warn!(%unload_error, "unloading after a failed probe also failed");
            }
            Err(failure)
        }
    }
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

    /// Backend whose first generation fails the way mistral.rs does for a split-device model.
    struct ProbeBackend {
        loaded: bool,
        fail_with: Option<&'static str>,
        hang: bool,
    }

    #[async_trait]
    impl InferenceBackend for ProbeBackend {
        async fn load_model(&mut self, _model_path: &str) -> Result<()> {
            self.loaded = true;
            Ok(())
        }

        async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
            assert_eq!(request.max_tokens, Some(PROBE_MAX_TOKENS));
            assert_eq!(request.temperature, Some(0.0));
            if self.hang {
                std::future::pending::<()>().await;
            }
            match self.fail_with {
                Some(message) => Err(Error::Backend(message.to_string())),
                None => Ok(GenerateResponse {
                    text: "ok".to_string(),
                    tokens_generated: 1,
                    finish_reason: FinishReason::Stop,
                }),
            }
        }

        async fn unload_model(&mut self) -> Result<()> {
            self.loaded = false;
            Ok(())
        }

        fn is_loaded(&self) -> bool {
            self.loaded
        }

        fn backend_name(&self) -> &'static str {
            "probe-test"
        }
    }

    // Protects readiness: a loaded model that cannot generate must not stay loaded.
    // Detects probe_or_unload swallowing the error, keeping the model loaded, or replacing the
    // real error with a success. Needs no model. Breadcrumb: backends::probe_or_unload.
    #[tokio::test]
    async fn probe_failure_unloads_the_backend_and_keeps_the_real_error() {
        let mut backend = ProbeBackend {
            loaded: true,
            fail_with: Some("device mismatch in rms-norm, lhs: Cpu, rhs: Cuda"),
            hang: false,
        };
        let failure = probe_or_unload(&mut backend).await.expect_err("the probe must fail");
        assert!(matches!(failure, ProbeFailure::Failed(_)), "{failure:?}");
        let message = failure.to_string();
        assert!(message.contains("generation probe failed"), "{message}");
        assert!(message.contains("device mismatch in rms-norm"), "{message}");
        assert!(!backend.is_loaded(), "a failed probe must leave the backend not loaded");
    }

    #[tokio::test]
    async fn probe_success_leaves_the_backend_loaded() {
        let mut backend = ProbeBackend {
            loaded: true,
            fail_with: None,
            hang: false,
        };
        probe_or_unload(&mut backend).await.expect("the probe must pass");
        assert!(backend.is_loaded());
    }

    // Protects against a wedged device keeping the server from ever answering /health.
    // Detects removing the probe timeout.
    #[tokio::test]
    async fn probe_that_never_finishes_times_out() {
        let backend = ProbeBackend {
            loaded: true,
            fail_with: None,
            hang: true,
        };
        let error = probe_within(&backend, std::time::Duration::from_millis(50))
            .await
            .expect_err("a hung probe must fail");
        assert!(matches!(error, ProbeFailure::TimedOut(_)), "{error:?}");
        assert!(error.to_string().contains("did not finish"), "{error}");
    }

    #[test]
    #[cfg(not(any(feature = "llamacpp", feature = "mistralrs-backend")))]
    fn test_backend_type_no_features() {
        // With no backend features enabled, BackendType has no variants,
        // so we can't construct one. The enum is effectively empty.
        // This test verifies the module compiles correctly without backends —
        // reaching this point at runtime is the assertion.
    }
}
