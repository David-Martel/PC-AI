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
            .with_device(device)
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
            .with_device(device)
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
        let choice = completion
            .choices
            .first()
            .ok_or_else(|| Error::Backend("Model response contained no choices".to_string()))?;
        let text = choice.message.content.as_ref().cloned().unwrap_or_default();

        let tokens_generated = completion.usage.completion_tokens;

        let finish_reason = match choice.finish_reason.as_str() {
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

    fn response(choices: Vec<mistralrs_core::Choice>) -> ChatCompletionResponse {
        ChatCompletionResponse {
            id: "fixture".into(),
            choices,
            created: 0,
            model: "fixture".into(),
            system_fingerprint: "fixture".into(),
            object: "chat.completion".into(),
            usage: mistralrs_core::Usage {
                completion_tokens: 3,
                prompt_tokens: 2,
                total_tokens: 5,
                avg_tok_per_sec: 0.0,
                avg_prompt_tok_per_sec: 0.0,
                avg_compl_tok_per_sec: 0.0,
                total_time_sec: 0.0,
                total_prompt_time_sec: 0.0,
                total_completion_time_sec: 0.0,
            },
        }
    }

    #[test]
    fn empty_response_returns_error_instead_of_panicking() {
        let result = MistralRsBackend::map_response(response(vec![]));
        assert!(matches!(result, Err(Error::Backend(message)) if message.contains("no choices")));
    }

    #[test]
    fn response_preserves_text_token_count_and_length_finish() {
        let completion = response(vec![mistralrs_core::Choice {
            finish_reason: "length".into(),
            index: 0,
            message: mistralrs_core::ResponseMessage {
                content: Some("fixture text".into()),
                role: "assistant".into(),
                tool_calls: None,
                reasoning_content: None,
            },
            logprobs: None,
        }]);
        let result = MistralRsBackend::map_response(completion).expect("fixture response must map");
        assert_eq!(result.text, "fixture text");
        assert_eq!(result.tokens_generated, 3);
        assert!(matches!(result.finish_reason, FinishReason::Length));
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
