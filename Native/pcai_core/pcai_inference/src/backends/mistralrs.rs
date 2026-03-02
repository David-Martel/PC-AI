//! mistral.rs backend implementation
//!
//! This backend uses mistral.rs for high-performance LLM inference with support for:
//! - GGUF quantized models
//! - SafeTensors (HuggingFace) models
//! - Multimodal models (vision + text)
//! - CUDA acceleration (when available)
//! - CPU fallback when CUDA is not enabled at build time

use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;

use mistralrs::{best_device, GgufModelBuilder, Model, TextMessageRole, TextMessages, TextModelBuilder};
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

        // Detect best available device (prefer CUDA, fallback to CPU)
        let default_force_cpu = cfg!(target_os = "windows") && !Self::cuda_enabled();
        let force_cpu = self.force_cpu || default_force_cpu;

        if force_cpu && cfg!(target_os = "windows") && !self.force_cpu {
            tracing::info!(
                "Defaulting to CPU mode on Windows. For GPU support, build with 'cuda-mistralrs' \
                 (or umbrella 'cuda') and ensure CUDA environment is initialized."
            );
        }

        let device = best_device(force_cpu).map_err(|e| Error::Backend(format!("Failed to get device: {}", e)))?;
        tracing::info!("Using device: {:?}", device);

        // Build the model
        let model = GgufModelBuilder::new(&model_id, vec![&gguf_file])
            .with_logging()
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

        // Detect best available device (prefer CUDA, fallback to CPU)
        let default_force_cpu = cfg!(target_os = "windows") && !Self::cuda_enabled();
        let force_cpu = self.force_cpu || default_force_cpu;

        if force_cpu && cfg!(target_os = "windows") && !self.force_cpu {
            tracing::info!(
                "Defaulting to CPU mode on Windows. For GPU support, build with 'cuda-mistralrs' \
                 (or umbrella 'cuda') and ensure CUDA environment is initialized."
            );
        }

        let device = best_device(force_cpu).map_err(|e| Error::Backend(format!("Failed to get device: {}", e)))?;
        tracing::info!("Using device: {:?}", device);

        // Build the model using TextModelBuilder for SafeTensors
        let model = TextModelBuilder::new(path.to_string())
            .with_logging()
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
        // Future enhancement: Parse system/user messages from prompt
        let messages = TextMessages::new().add_message(TextMessageRole::User, &request.prompt);

        // NOTE: Advanced sampling parameters (temperature, top_p, max_tokens, stop sequences)
        // are currently not supported with the simple TextMessages API.
        // TextMessages uses deterministic sampling by default.
        //
        // TODO: Implement a custom RequestLike to support full sampling control.
        // For now, we ignore the request parameters and use deterministic sampling.

        // Send chat request with deterministic sampling
        let response = model
            .send_chat_request(messages)
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
        let (model_id, filename) = result.expect("TODO: Verify unwrap");
        assert_eq!(filename, "model.gguf");
    }

    #[tokio::test]
    async fn test_backend_creation() {
        let backend = MistralRsBackend::new();
        assert!(!backend.is_loaded());
        assert_eq!(backend.backend_name(), "mistral.rs");
    }
}
