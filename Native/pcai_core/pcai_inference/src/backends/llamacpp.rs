//! `llamacpp` compatibility backend implemented with llama-rs (`llm` crate).

use async_trait::async_trait;
use llm::{
    InferenceFeedback, InferenceParameters, InferenceRequest, InferenceResponse, InferenceSessionConfig,
    Model, ModelArchitecture, ModelParameters, OutputRequest, TokenizerSource,
};
use rand::{rngs::StdRng, SeedableRng};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::{FinishReason, GenerateRequest, GenerateResponse, InferenceBackend};
use crate::{Error, Result};

#[derive(Debug, Clone, Default)]
struct LlamaRsRuntimeConfig {
    architecture: Option<ModelArchitecture>,
    sampler_options: Option<Vec<String>>,
    seed: Option<u64>,
}

#[derive(Debug, Deserialize, Default)]
struct LlmConfigFile {
    #[serde(rename = "nativeInference", default)]
    native_inference: NativeInferenceConfigFile,
}

#[derive(Debug, Deserialize, Default)]
struct NativeInferenceConfigFile {
    #[serde(default)]
    backends: NativeBackendsConfigFile,
}

#[derive(Debug, Deserialize, Default)]
struct NativeBackendsConfigFile {
    #[serde(default)]
    llamacpp: NativeLlamaCppConfigFile,
}

#[derive(Debug, Deserialize, Default)]
struct NativeLlamaCppConfigFile {
    #[serde(default)]
    settings: NativeLlamaCppSettingsFile,
}

#[derive(Debug, Deserialize, Default)]
struct NativeLlamaCppSettingsFile {
    architecture: Option<String>,
    n_ctx: Option<u32>,
    n_batch: Option<u32>,
    n_gpu_layers: Option<i32>,
    #[serde(default, alias = "samplerOptions")]
    samplers: Vec<String>,
    #[serde(alias = "randomSeed")]
    seed: Option<u64>,
}

/// Compatibility backend for the existing `llamacpp` feature/identifier.
///
/// Internally this backend uses `llama-rs` (`llm`) instead of `llama-cpp-2`.
pub struct LlamaCppBackend {
    /// Loaded model.
    model: Option<Arc<dyn Model>>,
    /// Model path.
    model_path: Option<PathBuf>,
    /// Detected or configured model architecture.
    architecture: Option<ModelArchitecture>,
    /// GPU layers to offload (`u32::MAX` = all layers, `0` = CPU only).
    n_gpu_layers: u32,
    /// Context size.
    n_ctx: u32,
    /// Batch size.
    n_batch: u32,
    /// Runtime controls loaded from config.
    runtime_config: LlamaRsRuntimeConfig,
}

impl LlamaCppBackend {
    /// Create a new backend with defaults.
    pub fn new() -> Self {
        Self::with_config(u32::MAX, 8192, 2048)
    }

    /// Create a new backend with custom configuration.
    pub fn with_config(n_gpu_layers: u32, n_ctx: u32, n_batch: u32) -> Self {
        Self {
            model: None,
            model_path: None,
            architecture: None,
            n_gpu_layers,
            n_ctx,
            n_batch,
            runtime_config: LlamaRsRuntimeConfig::default(),
        }
    }

    fn find_llm_config_path() -> Option<PathBuf> {
        let mut candidates: Vec<PathBuf> = Vec::new();

        if let Ok(cwd) = std::env::current_dir() {
            candidates.push(cwd);
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                candidates.push(parent.to_path_buf());
            }
        }

        for mut start in candidates {
            for _ in 0..8 {
                let candidate = start.join("Config").join("llm-config.json");
                if candidate.exists() {
                    return Some(candidate);
                }
                if !start.pop() {
                    break;
                }
            }
        }

        None
    }

    fn load_llama_cpp_settings() -> Option<NativeLlamaCppSettingsFile> {
        let config_path = Self::find_llm_config_path()?;
        let raw = std::fs::read_to_string(&config_path).ok()?;
        let cfg: LlmConfigFile = serde_json::from_str(&raw).ok()?;
        Some(cfg.native_inference.backends.llamacpp.settings)
    }

    fn apply_runtime_settings(&mut self) {
        let Some(settings) = Self::load_llama_cpp_settings() else {
            return;
        };

        if let Some(n_ctx) = settings.n_ctx {
            self.n_ctx = n_ctx;
        }
        if let Some(n_batch) = settings.n_batch {
            self.n_batch = n_batch;
        }
        if let Some(n_gpu_layers) = settings.n_gpu_layers {
            self.n_gpu_layers = if n_gpu_layers < 0 {
                u32::MAX
            } else {
                n_gpu_layers as u32
            };
        }

        if let Some(arch_text) = settings.architecture.as_deref() {
            match arch_text.parse::<ModelArchitecture>() {
                Ok(arch) => {
                    self.runtime_config.architecture = Some(arch);
                }
                Err(e) => {
                    tracing::warn!(
                        "Invalid nativeInference.backends.llamacpp.settings.architecture='{}': {}",
                        arch_text,
                        e
                    );
                }
            }
        }

        if !settings.samplers.is_empty() {
            self.runtime_config.sampler_options = Some(settings.samplers.clone());
        }
        self.runtime_config.seed = settings.seed;
    }

    fn resolve_architecture(&self, model_path: &Path) -> Result<ModelArchitecture> {
        if let Some(arch) = self.runtime_config.architecture {
            return Ok(arch);
        }

        let name = model_path
            .file_name()
            .map(|n| n.to_string_lossy().to_lowercase())
            .unwrap_or_default();

        let guesses = [
            ("gptneox", "gptneox"),
            ("neox", "gptneox"),
            ("gpt-j", "gptj"),
            ("gptj", "gptj"),
            ("gpt2", "gpt2"),
            ("bloom", "bloom"),
            ("mpt", "mpt"),
            ("falcon", "falcon"),
            ("llama", "llama"),
            ("alpaca", "llama"),
            ("vicuna", "llama"),
        ];

        for (marker, arch_name) in guesses {
            if name.contains(marker) {
                if let Ok(arch) = arch_name.parse::<ModelArchitecture>() {
                    return Ok(arch);
                }
            }
        }

        "llama"
            .parse::<ModelArchitecture>()
            .map_err(|e| Error::Backend(format!("Failed to resolve llama-rs architecture: {}", e)))
    }

    fn model_parameters(&self) -> ModelParameters {
        #[cfg(feature = "cuda-llamacpp")]
        let use_gpu = self.n_gpu_layers != 0;
        #[cfg(not(feature = "cuda-llamacpp"))]
        let use_gpu = false;

        let gpu_layers = if !use_gpu {
            None
        } else if self.n_gpu_layers == u32::MAX {
            None
        } else {
            Some(self.n_gpu_layers as usize)
        };

        ModelParameters {
            context_size: self.n_ctx as usize,
            use_gpu,
            gpu_layers,
            ..Default::default()
        }
    }

    fn session_config(&self) -> InferenceSessionConfig {
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);

        InferenceSessionConfig {
            n_batch: self.n_batch as usize,
            n_threads,
            ..Default::default()
        }
    }

    fn stop_match_index(text: &str, stops: &[String]) -> Option<usize> {
        stops
            .iter()
            .filter(|s| !s.is_empty())
            .filter_map(|s| text.find(s))
            .min()
    }

    /// Generate text with streaming callback.
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
            .cloned()
            .ok_or_else(|| Error::Backend("No model loaded".to_string()))?;

        let max_tokens = request.max_tokens.unwrap_or(512);
        let temperature = request.temperature.unwrap_or(0.7);
        let top_p = request.top_p.unwrap_or(0.9);
        let stop_sequences = request.stop.clone();

        let sampler_options = self
            .runtime_config
            .sampler_options
            .clone()
            .unwrap_or_else(|| {
                vec![
                    format!("temperature:{}", temperature),
                    format!("top-p:p={}", top_p),
                ]
            });

        let sampler = llm::samplers::build_sampler(model.tokenizer().len(), &[], &sampler_options)
            .map_err(|e| Error::Backend(format!("Failed to build llama-rs sampler chain: {}", e)))?;

        let inference_parameters = InferenceParameters { sampler };
        let inference_request = InferenceRequest {
            prompt: request.prompt.as_str().into(),
            parameters: &inference_parameters,
            play_back_previous_tokens: false,
            maximum_token_count: Some(max_tokens),
        };

        let mut output_request = OutputRequest::default();
        let mut session = model.start_session(self.session_config());
        let mut rng = self
            .runtime_config
            .seed
            .map(StdRng::seed_from_u64)
            .unwrap_or_else(StdRng::from_entropy);

        let mut generated_text = String::new();
        let mut tokens_generated = 0usize;
        let mut finish_reason = FinishReason::Length;

        session
            .infer::<std::convert::Infallible>(
                model.as_ref(),
                &mut rng,
                &inference_request,
                &mut output_request,
                |response| match response {
                    InferenceResponse::InferredToken(token) => {
                        tokens_generated += 1;

                        if stop_sequences.is_empty() {
                            generated_text.push_str(&token);
                            if !token.is_empty() {
                                callback(token);
                            }
                            return Ok(InferenceFeedback::Continue);
                        }

                        let mut candidate = generated_text.clone();
                        candidate.push_str(&token);

                        if let Some(stop_idx) = Self::stop_match_index(&candidate, &stop_sequences) {
                            if stop_idx > generated_text.len() {
                                let safe_suffix = &candidate[generated_text.len()..stop_idx];
                                if !safe_suffix.is_empty() {
                                    callback(safe_suffix.to_string());
                                }
                            }
                            generated_text = candidate[..stop_idx].to_string();
                            finish_reason = FinishReason::Stop;
                            return Ok(InferenceFeedback::Halt);
                        }

                        generated_text = candidate;
                        if !token.is_empty() {
                            callback(token);
                        }
                        Ok(InferenceFeedback::Continue)
                    }
                    InferenceResponse::EotToken => {
                        finish_reason = FinishReason::Stop;
                        Ok(InferenceFeedback::Halt)
                    }
                    _ => Ok(InferenceFeedback::Continue),
                },
            )
            .map_err(|e| Error::Backend(format!("Generation failed: {}", e)))?;

        if matches!(finish_reason, FinishReason::Length) && tokens_generated < max_tokens {
            finish_reason = FinishReason::Stop;
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
        self.apply_runtime_settings();

        let path = PathBuf::from(model_path);
        if !path.exists() {
            return Err(Error::Backend(format!(
                "Model path does not exist: {}",
                path.display()
            )));
        }

        tracing::info!("Loading llama-rs model from: {}", path.display());

        let architecture = self.resolve_architecture(&path)?;
        let model_params = self.model_parameters();

        let loaded_model = llm::load_dynamic(
            Some(architecture),
            path.as_path(),
            TokenizerSource::Embedded,
            model_params,
            |_| {},
        )
        .map_err(|e| Error::Backend(format!("Failed to load model via llama-rs: {}", e)))?;

        self.model = Some(Arc::from(loaded_model));
        self.model_path = Some(path);
        self.architecture = Some(architecture);

        Ok(())
    }

    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        self.generate_streaming_internal(request, |_| {}).await
    }

    async fn generate_streaming(
        &self,
        request: GenerateRequest,
        callback: &mut (dyn FnMut(String) + Send),
    ) -> Result<GenerateResponse> {
        self.generate_streaming_internal(request, |token| callback(token))
            .await
    }

    async fn unload_model(&mut self) -> Result<()> {
        self.model = None;
        self.model_path = None;
        self.architecture = None;
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.model.is_some()
    }

    fn backend_name(&self) -> &'static str {
        // Preserve existing external backend name for compatibility with current FFI and PS wrappers.
        "llama.cpp"
    }
}

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
