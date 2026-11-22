//! mistral.rs backend implementation
//!
//! This backend uses mistral.rs for high-performance LLM inference with support for:
//! - GGUF quantized models
//! - SafeTensors (HuggingFace) models
//! - CUDA acceleration (when available)
//! - CPU fallback when CUDA is not enabled at build time

use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;

use mistralrs::{best_device, GgufModelBuilder, Model, RequestBuilder, TextMessageRole, TextModelBuilder};
use mistralrs_core::ChatCompletionResponse;

use super::{FinishReason, GenerateRequest, GenerateResponse, InferenceBackend};
use crate::{Error, Result};

/// mistral.rs backend implementation
pub struct MistralRsBackend {
    /// The loaded model instance
    model: Option<Arc<Model>>,
    /// Path to the currently loaded model
    model_path: Option<String>,
    /// Whether the current model is a GGUF model
    is_gguf: bool,
    /// Whether to force CPU execution
    force_cpu: bool,
}

impl MistralRsBackend {
    #[inline]
    fn cuda_enabled() -> bool {
        // Keep compatibility with legacy umbrella feature (`cuda`) and explicit backend
        // feature (`cuda-mistralrs`) used by Invoke-PcaiBuild for mistralrs.
        cfg!(feature = "cuda") || cfg!(feature = "cuda-mistralrs")
    }

    /// Create a new mistral.rs backend
    pub fn new() -> Self {
        Self {
            model: None,
            model_path: None,
            is_gguf: false,
            force_cpu: false,
        }
    }

    /// Create a new mistral.rs backend with configuration
    pub fn with_config(force_cpu: bool) -> Self {
        Self {
            model: None,
            model_path: None,
            is_gguf: false,
            force_cpu,
        }
    }

    fn requires_cpu(&self) -> bool {
        // The pinned builder tries CUDA when it was compiled in, even if a
        // device is missing. Check initialization before opening the model.
        self.select_cpu(Self::cuda_enabled(), || match best_device(false) {
            Ok(_) => true,
            Err(error) => {
                tracing::warn!(%error, "CUDA device unavailable, loading the model on CPU");
                false
            }
        })
    }

    fn select_cpu(&self, cuda_compiled: bool, cuda_available: impl FnOnce() -> bool) -> bool {
        if self.force_cpu {
            return true;
        }
        if cuda_compiled {
            return !cuda_available();
        }
        cfg!(target_os = "windows")
    }

    /// Detect if a path is a GGUF model
    fn is_gguf_model(path: &str) -> bool {
        path.to_lowercase().ends_with(".gguf")
    }

    /// Extract model ID and GGUF filename from path
    fn parse_gguf_path(path: &str) -> Result<(String, String)> {
        let path_obj = Path::new(path);

        // If it's a file path, extract directory and filename
        if path_obj.is_file() || path.contains('/') || path.contains('\\') {
            let filename = path_obj
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or_else(|| Error::Backend("Invalid GGUF filename".to_string()))?
                .to_string();

            // Use parent directory or current directory as model ID
            let model_id = path_obj.parent().and_then(|p| p.to_str()).unwrap_or(".").to_string();

            Ok((model_id, filename))
        } else {
            // Assume it's a HuggingFace repo (e.g., "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
            // In this case, the user should specify the GGUF filename separately
            Err(Error::Backend(
                "For HuggingFace GGUF models, please provide full path or use load_gguf_hf".to_string(),
            ))
        }
    }

    /// Load a GGUF model from local path or HuggingFace
    async fn load_gguf_model(&mut self, path: &str) -> Result<()> {
        tracing::info!("Loading GGUF model from: {}", path);

        // Try to parse as local path
        let (model_id, gguf_file) = Self::parse_gguf_path(path)?;

        // mistral.rs 0.8.1 resolves the device using its builder's force_cpu setting.
        let mut builder = GgufModelBuilder::new(&model_id, vec![&gguf_file]).with_logging();
        let force_cpu = self.requires_cpu();
        if force_cpu {
            builder = builder.with_force_cpu();
        }
        tracing::info!(force_cpu, "Loading GGUF with the selected device policy");
        let model = builder
            .build()
            .await
            .map_err(|e| Error::Backend(format!("Failed to load GGUF model: {}", e)))?;

        self.model = Some(Arc::new(model));
        self.model_path = Some(path.to_string());
        self.is_gguf = true;

        tracing::info!("GGUF model loaded successfully");
        Ok(())
    }

    /// Load a SafeTensors model from HuggingFace or local path
    async fn load_safetensors_model(&mut self, path: &str) -> Result<()> {
        tracing::info!("Loading SafeTensors model from: {}", path);

        let mut builder = TextModelBuilder::new(path.to_string()).with_logging();
        let force_cpu = self.requires_cpu();
        if force_cpu {
            builder = builder.with_force_cpu();
        }
        tracing::info!(force_cpu, "Loading text model with the selected device policy");
        let model = builder
            .build()
            .await
            .map_err(|e| Error::Backend(format!("Failed to load SafeTensors model: {}", e)))?;

        self.model = Some(Arc::new(model));
        self.model_path = Some(path.to_string());
        self.is_gguf = false;

        tracing::info!("SafeTensors model loaded successfully");
        Ok(())
    }

    /// Map ChatCompletionResponse to GenerateResponse
    fn map_response(completion: ChatCompletionResponse) -> Result<GenerateResponse> {
        let text = completion.choices[0]
            .message
            .content
            .as_ref()
            .cloned()
            .unwrap_or_default();

        let tokens_generated = completion.usage.completion_tokens;

        let finish_reason = match completion.choices[0].finish_reason.as_str() {
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            _ => FinishReason::Stop,
        };

        Ok(GenerateResponse {
            text,
            tokens_generated,
            finish_reason,
        })
    }
}

impl Default for MistralRsBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl InferenceBackend for MistralRsBackend {
    async fn load_model(&mut self, model_path: &str) -> Result<()> {
        // Unload any existing model first
        if self.model.is_some() {
            self.unload_model().await?;
        }

        // Detect model type and load appropriately
        if Self::is_gguf_model(model_path) {
            self.load_gguf_model(model_path).await
        } else {
            // Assume SafeTensors/HuggingFace model
            self.load_safetensors_model(model_path).await
        }
    }

    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let model = self.model.as_ref().ok_or(Error::ModelNotLoaded)?.clone();

        tracing::debug!("Generating response for prompt (length: {})", request.prompt.len());

        // Build messages - treat prompt as user message
        let mut builder = RequestBuilder::new().add_message(TextMessageRole::User, &request.prompt);

        if let Some(t) = request.temperature {
            builder = builder.set_sampler_temperature(t as f64);
        }
        if let Some(p) = request.top_p {
            builder = builder.set_sampler_topp(p as f64);
        }
        if let Some(max_tokens) = request.max_tokens {
            builder = builder.set_sampler_max_len(max_tokens);
        }
        if !request.stop.is_empty() {
            builder = builder.set_sampler_stop_toks(mistralrs_core::StopTokens::Seqs(request.stop.clone()));
        }

        let response = model
            .send_chat_request(builder)
            .await
            .map_err(|e| Error::Backend(format!("Generation failed: {}", e)))?;

        tracing::debug!("Generated {} tokens", response.usage.completion_tokens);

        Self::map_response(response)
    }

    async fn unload_model(&mut self) -> Result<()> {
        if self.model.is_some() {
            tracing::info!("Unloading model");
            self.model = None;
            self.model_path = None;
            self.is_gguf = false;
        }
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    fn backend_name(&self) -> &'static str {
        "mistral.rs"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Protects explicit CPU mode regardless of the compiled GPU features.
    // Detects changes that allow the automatic device probe in explicit CPU mode.
    // Needs no device or checkpoint. Breadcrumb: MistralRsBackend::requires_cpu.
    #[test]
    fn explicit_cpu_mode_requires_cpu() {
        assert!(MistralRsBackend::with_config(true)
            .select_cpu(true, || { panic!("an explicit CPU request must not initialize CUDA") }));
    }

    // Protects automatic selection for a CUDA artifact with a detachable GPU.
    // Detects retaining CUDA after a failed probe or choosing CPU with a ready device.
    // Needs two synthetic probe outcomes. Breadcrumb: MistralRsBackend::select_cpu.
    #[test]
    fn auto_mode_falls_back_only_when_cuda_is_unavailable() {
        let backend = MistralRsBackend::new();
        assert!(backend.select_cpu(true, || false));
        assert!(!backend.select_cpu(true, || true));
    }

    // Protects default loading with a Windows CPU-only artifact.
    // Detects CPU-only deployments entering the CUDA builder policy.
    // Needs Windows without the backend CUDA feature. Breadcrumb: MistralRsBackend::requires_cpu.
    #[cfg(all(target_os = "windows", not(feature = "cuda-mistralrs")))]
    #[test]
    fn windows_cpu_artifact_defaults_to_cpu_mode() {
        assert!(MistralRsBackend::new().requires_cpu());
    }

    #[test]
    fn test_is_gguf_model() {
        assert!(MistralRsBackend::is_gguf_model("model.gguf"));
        assert!(MistralRsBackend::is_gguf_model("path/to/model.GGUF"));
        assert!(!MistralRsBackend::is_gguf_model("model.safetensors"));
        assert!(!MistralRsBackend::is_gguf_model("microsoft/Phi-3.5-mini"));
    }

    #[test]
    fn test_parse_gguf_path() {
        let result = MistralRsBackend::parse_gguf_path("/path/to/model.gguf");
        assert!(result.is_ok());
        let (model_id, filename) = result.expect("test: parse_gguf_path must succeed for a valid absolute GGUF path");
        assert_eq!(model_id, "/path/to");
        assert_eq!(filename, "model.gguf");
    }

    #[tokio::test]
    async fn test_backend_creation() {
        let backend = MistralRsBackend::new();
        assert!(!backend.is_loaded());
        assert_eq!(backend.backend_name(), "mistral.rs");
    }
}
