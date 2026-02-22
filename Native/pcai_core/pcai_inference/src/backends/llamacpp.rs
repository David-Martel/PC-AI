//! llama.cpp backend implementation

use async_trait::async_trait;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend as LlamaCppBackend_;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;

use super::{FinishReason, GenerateRequest, GenerateResponse, InferenceBackend};
use crate::{Error, Result};

/// llama.cpp backend implementation
pub struct LlamaCppBackend {
    /// Global llama.cpp backend (initialized once)
    backend: Arc<LlamaCppBackend_>,
    /// Loaded model (heap-allocated for stable address)
    model: Option<Box<LlamaModel>>,
    /// Inference context
    context: Arc<Mutex<Option<LlamaContext<'static>>>>,
    /// Model path
    model_path: Option<PathBuf>,
    /// GPU layers to offload (-1 = all)
    n_gpu_layers: u32,
    /// Context size
    n_ctx: u32,
    /// Batch size
    n_batch: u32,
}

impl LlamaCppBackend {
    /// Create a new llama.cpp backend
    pub fn new() -> Self {
        Self::with_config(u32::MAX, 8192, 2048)
    }

    /// Create a new llama.cpp backend with custom configuration
    pub fn with_config(n_gpu_layers: u32, n_ctx: u32, n_batch: u32) -> Self {
        let backend = LlamaCppBackend_::init().expect("Failed to initialize llama.cpp backend");

        Self {
            backend: Arc::new(backend),
            model: None,
            context: Arc::new(Mutex::new(None)),
            model_path: None,
            n_gpu_layers,
            n_ctx,
            n_batch,
        }
    }

    /// Generate text with streaming callback
    pub async fn generate_streaming_internal<F>(
        &self,
        request: GenerateRequest,
        mut callback: F,
    ) -> Result<GenerateResponse>
    where
        F: FnMut(String) + Send,
    {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::Backend("No model loaded".to_string()))?;

        let context = self.context.clone();
        let mut ctx_guard = context
            .lock()
            .map_err(|e| Error::Backend(format!("Failed to lock context: {}", e)))?;
        let ctx = ctx_guard
            .as_mut()
            .ok_or_else(|| Error::Backend("No context available".to_string()))?;

        // Tokenize prompt
        let tokens_list = model
            .str_to_token(&request.prompt, AddBos::Always)
            .map_err(|e| Error::Backend(format!("Tokenization failed: {:?}", e)))?;

        tracing::info!("Tokenized prompt: {} tokens", tokens_list.len());

        // Create batch and add prompt tokens
        let mut batch = LlamaBatch::new(self.n_batch as usize, 1);
        let last_index = tokens_list.len() as i32 - 1;

        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            let is_last = i == last_index;
            batch
                .add(token, i, &[0], is_last)
                .map_err(|e| Error::Backend(format!("Batch add failed: {:?}", e)))?;
        }

        // Decode prompt
        ctx.decode(&mut batch)
            .map_err(|e| Error::Backend(format!("Decode failed: {:?}", e)))?;

        // Generate tokens
        let max_tokens = request.max_tokens.unwrap_or(512);
        let temperature = request.temperature.unwrap_or(0.7);
        let top_p = request.top_p.unwrap_or(0.9);

        let mut n_cur = batch.n_tokens();
        let mut generated_text = String::new();
        let mut tokens_generated = 0;
        let mut finish_reason = FinishReason::Length;

        // Create sampler chain: temp -> top_p -> dist
        let mut sampler = if temperature <= 0.0 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple(vec![
                LlamaSampler::temp(temperature),
                LlamaSampler::top_p(top_p, 1),
                LlamaSampler::dist(42), // seed
            ])
        };

        while tokens_generated < max_tokens {
            // Sample next token
            let token = sampler.sample(ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            // Check for EOS or stop sequences
            if token == model.token_eos() {
                finish_reason = FinishReason::Stop;
                break;
            }

            // Decode token to text
            let token_str = model
                .token_to_str(token, Special::Tokenize)
                .map_err(|e| Error::Backend(format!("Token decode failed: {:?}", e)))?;

            // Check stop sequences
            if !request.stop.is_empty() {
                let potential_text = format!("{}{}", generated_text, token_str);
                if request.stop.iter().any(|s| potential_text.contains(s)) {
                    finish_reason = FinishReason::Stop;
                    break;
                }
            }

            generated_text.push_str(&token_str);
            callback(token_str);
            tokens_generated += 1;

            // Add token to batch for next iteration
            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| Error::Backend(format!("Batch add failed: {:?}", e)))?;
            n_cur += 1;

            // Decode next token
            ctx.decode(&mut batch)
                .map_err(|e| Error::Backend(format!("Decode failed: {:?}", e)))?;
        }

        Ok(GenerateResponse {
            text: generated_text,
            tokens_generated,
            finish_reason,
        })
    }
}

impl Default for LlamaCppBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl InferenceBackend for LlamaCppBackend {
    async fn load_model(&mut self, model_path: &str) -> Result<()> {
        tracing::info!("Loading model from: {}", model_path);

        // Configure model parameters
        let mut params = LlamaModelParams::default();
        params = params.with_n_gpu_layers(self.n_gpu_layers);

        // Load model
        let model = LlamaModel::load_from_file(&self.backend, model_path, &params)
            .map_err(|e| Error::Backend(format!("Failed to load model: {:?}", e)))?;

        tracing::info!(
            "Model loaded: {} params, {} layers, vocab size: {}",
            model.n_params(),
            model.n_layer(),
            model.n_vocab()
        );

        // Create context
        let mut ctx_params = LlamaContextParams::default();
        ctx_params = ctx_params
            .with_n_ctx(NonZeroU32::new(self.n_ctx))
            .with_n_batch(self.n_batch);

        // Heap-allocate the model to get a stable address
        let model_box = Box::new(model);

        // SAFETY: We leak the box to get a 'static reference, which is required by LlamaContext.
        // This is safe because:
        // 1. We maintain ownership via self.model
        // 2. We drop the context before the model in unload_model()
        // 3. The Box ensures the model has a stable memory address
        let model_static: &'static LlamaModel = Box::leak(model_box);

        let context = model_static
            .new_context(&self.backend, ctx_params)
            .map_err(|e| Error::Backend(format!("Failed to create context: {:?}", e)))?;

        tracing::info!("Context created: n_ctx={}, n_batch={}", self.n_ctx, self.n_batch);

        // Store the leaked box and context
        // We'll manually drop the box in unload_model after dropping the context
        self.model = Some(unsafe { Box::from_raw(model_static as *const _ as *mut LlamaModel) });
        *self
            .context
            .lock()
            .map_err(|e| Error::Backend(format!("Failed to lock context: {}", e)))? = Some(context);
        self.model_path = Some(PathBuf::from(model_path));

        Ok(())
    }

    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        // Use streaming version with no-op callback
        self.generate_streaming_internal(request, |_| {}).await
    }

    async fn generate_streaming(
        &self,
        request: GenerateRequest,
        callback: &mut (dyn FnMut(String) + Send),
    ) -> Result<GenerateResponse> {
        self.generate_streaming_internal(request, |token| callback(token)).await
    }

    async fn unload_model(&mut self) -> Result<()> {
        tracing::info!("Unloading model");

        // Drop context first
        *self
            .context
            .lock()
            .map_err(|e| Error::Backend(format!("Failed to lock context: {}", e)))? = None;

        // Drop model
        self.model = None;
        self.model_path = None;

        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    fn backend_name(&self) -> &'static str {
        "llama.cpp"
    }
}

// Thread-safety: LlamaCppBackend is Send + Sync due to Arc/Mutex
unsafe impl Send for LlamaCppBackend {}
unsafe impl Sync for LlamaCppBackend {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_creation() {
        let backend = LlamaCppBackend::new();
        assert_eq!(backend.backend_name(), "llama.cpp");
        assert!(!backend.is_loaded());
    }

    #[test]
    fn test_backend_with_config() {
        let backend = LlamaCppBackend::with_config(32, 4096, 512);
        assert_eq!(backend.n_gpu_layers, 32);
        assert_eq!(backend.n_ctx, 4096);
        assert_eq!(backend.n_batch, 512);
    }
}
