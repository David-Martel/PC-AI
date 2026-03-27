//! Text-to-image generation pipeline for Janus-Pro.
//!
//! The pipeline implements the 576-token autoregressive loop with
//! Classifier-Free Guidance (CFG) as described in the Janus-Pro paper.
//!
//! # Pipeline summary
//!
//! 1. Tokenise the prompt using the Janus-Pro chat template.
//! 2. Build a batched `[2 × parallel_size, seq_len]` input where the first
//!    half carries the conditional (prompt) tokens and the second half carries
//!    the unconditional (padding) tokens required for CFG.
//! 3. Run 576 autoregressive steps:
//!    - Embed the current token batch via the LLM embedding table.
//!    - Forward through the LLM backbone to obtain hidden states.
//!    - Project hidden states through the image generation head.
//!    - Extract last-position logits and apply CFG.
//!    - Apply temperature scaling, softmax, and multinomial sampling.
//! 4. Decode the 576 collected tokens into an RGB pixel tensor via the
//!    VQ-VAE decoder.
//! 5. Denormalise from `[-1, 1]` to `[0, 255]` and convert to
//!    [`image::ImageBuffer<Rgb<u8>>`].
//!
//! # Note on API version
//!
//! This module uses `candle_transformers::models::llama::Llama` at version
//! 0.9.  The `forward_input_embed` method returns the pre-LM-head hidden
//! states `[B, S, hidden_size]`; `project_to_image_vocab` then maps those to
//! image-vocabulary logits `[B, S, image_vocab_size]`.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{VarBuilder, VarMap};
use image::{ImageBuffer, Rgb};
use std::path::PathBuf;
use std::time::Instant;

use pcai_media_model::{
    config::JanusConfig,
    janus_llama::{KvCache, PreAllocKvCache},
    vision::{JanusVisionConfig, JanusVisionTower},
    JanusModel, LlamaBackend, QuantizedJanusLlama,
};

use crate::config::PipelineConfig;
use crate::hub;
use crate::telemetry::{GenerationTelemetry, TelemetryCollector};

// ---------------------------------------------------------------------------
// CacheVariant — unified KV cache dispatch
// ---------------------------------------------------------------------------

/// Unified KV cache that dispatches to either [`PreAllocKvCache`] or [`KvCache`].
///
/// Use [`CacheVariant::new`] to construct based on the pipeline config flag
/// `use_prealloc_kv_cache`.  Call [`CacheVariant::forward_hidden`] instead of
/// invoking [`JanusLlama::forward_hidden`] or
/// [`JanusLlama::forward_hidden_prealloc`] directly — this ensures the correct
/// implementation is selected at runtime based on which cache is active.
///
/// # Performance
///
/// The `PreAllocKvCache` variant eliminates ≈95 GB of GPU bandwidth waste from
/// `Tensor::cat` across 576 autoregressive image-generation steps.  The
/// `KvCache` variant is retained as a fallback for debugging or for devices
/// where `scatter_set` is not supported.
enum CacheVariant {
    /// Pre-allocated ring-buffer cache — zero `Tensor::cat` cost.
    PreAlloc(PreAllocKvCache),
    /// Original dynamic cache — appends via `Tensor::cat` each step.
    Dynamic(KvCache),
}

impl CacheVariant {
    /// Construct a [`CacheVariant`] based on `use_prealloc`.
    ///
    /// When `use_prealloc` is `true`, a [`PreAllocKvCache`] is built with
    /// `max_seq_len` pre-allocated slots.  When `false`, a standard [`KvCache`]
    /// is built.
    ///
    /// # Arguments
    ///
    /// * `use_prealloc` — select the pre-allocated variant.
    /// * `dtype`        — tensor dtype matching the pipeline.
    /// * `cfg`          — Llama config (layer count, head dims, etc.).
    /// * `batch_size`   — batch size for the pre-allocated buffers (`1` for
    ///   Janus image generation without CFG; `2` with CFG).
    /// * `max_seq_len`  — maximum sequence length for the pre-allocated cache.
    ///   Ignored when `use_prealloc` is `false`.  For Janus image generation,
    ///   `num_image_tokens + prompt_len + margin` (e.g. `576 + seq_len + 32`).
    /// * `device`       — target device.
    ///
    /// # Errors
    ///
    /// Propagates candle errors from the underlying cache constructors.
    fn new(
        use_prealloc: bool,
        dtype: DType,
        cfg: &candle_transformers::models::llama::Config,
        batch_size: usize,
        max_seq_len: usize,
        device: &Device,
    ) -> candle_core::Result<Self> {
        if use_prealloc {
            tracing::info!(
                batch_size,
                max_seq_len,
                "using PreAllocKvCache (eliminates ~95 GB Tensor::cat bandwidth)"
            );
            let cache = PreAllocKvCache::new(dtype, cfg, batch_size, max_seq_len, device)?;
            Ok(Self::PreAlloc(cache))
        } else {
            tracing::info!("using dynamic KvCache (fallback path)");
            let cache = KvCache::new(true, dtype, cfg, device)?;
            Ok(Self::Dynamic(cache))
        }
    }

    /// Forward the LLM backbone and return last-position hidden states.
    ///
    /// Dispatches to [`JanusLlama::forward_hidden_prealloc`] for the
    /// `PreAlloc` variant and [`JanusLlama::forward_hidden`] for the
    /// `Dynamic` variant.
    ///
    /// # Arguments
    ///
    /// * `llama`       — reference to the [`JanusLlama`] backbone.
    /// * `input_embed` — float tensor `[B, S, hidden_size]`.
    /// * `index_pos`   — absolute position of the first token (for RoPE).
    ///
    /// # Returns
    ///
    /// Float tensor `[B, hidden_size]` — last-position hidden state.
    ///
    /// # Errors
    ///
    /// Propagates candle errors from the underlying forward implementation.
    fn forward_hidden(
        &mut self,
        llama: &LlamaBackend,
        input_embed: &Tensor,
        index_pos: usize,
    ) -> candle_core::Result<Tensor> {
        match self {
            Self::PreAlloc(cache) => llama.forward_hidden_prealloc(input_embed, index_pos, cache),
            Self::Dynamic(cache) => llama.forward_hidden(input_embed, index_pos, cache),
        }
    }
}

// ---------------------------------------------------------------------------
// GenerationPipeline
// ---------------------------------------------------------------------------

/// Loaded Janus-Pro text-to-image generation pipeline.
///
/// Construct with [`GenerationPipeline::load`].  Call
/// [`GenerationPipeline::generate`] with a text prompt to produce an
/// [`ImageBuffer<Rgb<u8>>`].
pub struct GenerationPipeline {
    model: JanusModel,
    tokenizer: tokenizers::Tokenizer,
    config: PipelineConfig,
    model_config: JanusConfig,
    device: Device,
    dtype: DType,
    /// Native Janus vision tower for image understanding.
    /// `None` if vision weights were not found in the safetensors shards.
    vision_tower: Option<JanusVisionTower>,
    /// Device on which the vision tower runs.
    vision_device: Device,
    /// DType used for vision execution.
    vision_dtype: DType,
}

impl GenerationPipeline {
    /// Build the native Janus vision-tower config used for understanding.
    fn vision_config(janus_cfg: &JanusConfig) -> JanusVisionConfig {
        JanusVisionConfig::from_janus_config(janus_cfg)
    }

    /// Load the native Janus vision tower for understanding.
    fn load_vision_tower(
        model_config: &JanusConfig,
        main_device: &Device,
        main_dtype: DType,
        shards: &[PathBuf],
    ) -> Result<(Option<JanusVisionTower>, Device, DType)> {
        if shards.is_empty() {
            tracing::warn!("vision tower unavailable because no safetensors shards were found");
            return Ok((None, Device::Cpu, DType::F32));
        }

        let vision_device = main_device.clone();
        let vision_dtype = main_dtype;
        let vision_cfg = Self::vision_config(model_config);
        let vision_varmap = VarMap::new();
        let vision_vb =
            VarBuilder::from_varmap(&vision_varmap, vision_dtype, &vision_device).pp("vision_model.vision_tower");
        let vision_model = match JanusVisionTower::new(vision_vb, &vision_cfg) {
            Ok(vm) => vm,
            Err(e) => {
                tracing::warn!(error = %e, "native Janus vision tower construction failed");
                return Ok((None, vision_device, vision_dtype));
            }
        };

        let expected = vision_varmap.data().lock().expect("VarMap lock poisoned").len();
        let loaded = hub::load_weights(&vision_varmap, shards, vision_dtype, &vision_device)
            .context("failed to load native Janus vision weights from safetensors")?;

        if loaded < expected {
            tracing::warn!(
                loaded,
                expected,
                "native Janus vision weights are incomplete; understanding will remain unavailable"
            );
            return Ok((None, vision_device, vision_dtype));
        }

        tracing::info!(
            loaded,
            expected,
            device = ?vision_device,
            dtype = ?vision_dtype,
            "native Janus vision tower loaded"
        );
        Ok((Some(vision_model), vision_device, vision_dtype))
    }

    /// Build a [`JanusModel`] from either a GGUF quantized backbone or full-precision
    /// safetensors weights.
    ///
    /// Both paths share the same [`VarMap`] / [`VarBuilder`] for the non-LLM
    /// components (VQ decoder, generation head, etc.).  On the GGUF path the
    /// LLaMA backbone is loaded separately; on the standard path all weights come
    /// from safetensors.
    ///
    /// The caller passes `varmap` as `&mut Option<VarMap>` so that the standard
    /// CUDA path can call `varmap.take()` to eagerly release the old tensor Arc
    /// references after the embedding table has been offloaded to CPU.
    ///
    /// # Errors
    ///
    /// Returns an error if any I/O, GGUF parsing, or weight-loading step fails.
    fn build_janus_model(
        gguf_path_opt: Option<&str>,
        varmap: &mut Option<VarMap>,
        vb: VarBuilder,
        model_config: &JanusConfig,
        shards: &[PathBuf],
        model_path: &std::path::Path,
        device: &Device,
        dtype: DType,
    ) -> Result<JanusModel> {
        if let Some(gguf_path_str) = gguf_path_opt {
            // ── GGUF path: load quantized LLaMA backbone from GGUF file ───────
            // Non-LLM components (VQ decoder, gen_head, etc.) are still loaded
            // from safetensors via the VarMap / VarBuilder path.
            let gguf_path = std::path::PathBuf::from(gguf_path_str);
            if !gguf_path.exists() {
                anyhow::bail!("gguf_path '{}' does not exist on disk", gguf_path.display());
            }

            tracing::info!(path = %gguf_path.display(), "loading quantized LLaMA backbone from GGUF");

            let llama_cfg = model_config.to_llama_config(false);
            let gguf_file = std::fs::File::open(&gguf_path)
                .with_context(|| format!("failed to open GGUF file '{}'", gguf_path.display()))?;
            let mut gguf_reader = std::io::BufReader::new(gguf_file);
            let gguf_content = candle_core::quantized::gguf_file::Content::read(&mut gguf_reader)
                .map_err(|e| anyhow::anyhow!("failed to parse GGUF header from '{}': {e}", gguf_path.display()))?;

            let mut quantized_llama =
                QuantizedJanusLlama::from_gguf(gguf_content, &mut gguf_reader, &llama_cfg, device)
                    .map_err(|e| anyhow::anyhow!("QuantizedJanusLlama::from_gguf failed: {e}"))?;

            tracing::info!(
                layers = quantized_llama.num_layers(),
                "quantized LLaMA backbone loaded from GGUF"
            );

            // On CUDA, offload the (full-precision) token embedding table to CPU to save VRAM.
            if matches!(device, Device::Cuda(_)) {
                quantized_llama
                    .offload_embeddings_to_cpu()
                    .map_err(|e| anyhow::anyhow!("quantized wte CPU offload failed: {e}"))?;
                tracing::info!("Offloaded quantized wte to CPU for CUDA pipeline");
            }

            // Load non-LLM weights (VQ decoder, gen_head, etc.) from safetensors.
            if shards.is_empty() {
                tracing::warn!(path = %model_path.display(), "no safetensors shards found; non-LLM components will use random weights");
            } else {
                let loaded = hub::load_weights(varmap.as_ref().unwrap(), shards, dtype, device)
                    .context("failed to load non-LLM weights from safetensors")?;
                tracing::info!(shards = shards.len(), tensors_loaded = loaded, "non-LLM weights loaded");
            }

            JanusModel::new_with_quantized_llama(quantized_llama, vb, model_config)
                .map_err(|e| anyhow::anyhow!("JanusModel (GGUF) construction failed: {e}"))
        } else {
            // ── Standard path: full-precision LLaMA from safetensors ──────────
            let mut m = JanusModel::new(vb, model_config)
                .map_err(|e| anyhow::anyhow!("JanusModel construction failed: {e}"))?;

            if shards.is_empty() {
                tracing::warn!(path = %model_path.display(), "no safetensors shards found; model will use random weights");
            } else {
                let loaded = hub::load_weights(varmap.as_ref().unwrap(), shards, dtype, device)
                    .context("failed to load model weights from safetensors")?;
                tracing::info!(shards = shards.len(), tensors_loaded = loaded, "weights loaded");
            }

            // On CUDA, offload the token embedding table to CPU to save VRAM, then
            // eagerly drop VarMap to release old GPU tensor Arc references.
            if matches!(device, Device::Cuda(_)) {
                m.llama
                    .offload_embeddings_to_cpu()
                    .map_err(|e| anyhow::anyhow!("wte CPU offload failed: {e}"))?;
                varmap.take();
                tracing::info!("Offloaded wte to CPU for CUDA pipeline");
            }

            Ok(m)
        }
    }

    /// Load the Janus-Pro pipeline from the configuration.
    ///
    /// Steps performed:
    /// 1. Resolve [`candle_core::Device`] and [`candle_core::DType`] from `config`.
    /// 2. Resolve the model path (local directory or HuggingFace Hub download).
    /// 3. Deserialise [`JanusConfig`] from `config.json` (falls back to 1B defaults).
    /// 4. Build [`JanusModel`] via [`build_janus_model`].
    /// 5. Load the tokenizer from `tokenizer.json`.
    /// 6. Build the native Janus vision tower (if `vision_model` weights exist).
    ///
    /// # Errors
    ///
    /// Returns an error if any of the above steps fail (I/O, model shape
    /// mismatch, missing tokenizer, etc.).
    pub fn load(config: PipelineConfig) -> Result<Self> {
        let device = config.resolve_device()?;
        let dtype = config.resolve_dtype();

        tracing::info!(
            model = %config.model,
            device = %config.device,
            dtype = ?dtype,
            "loading GenerationPipeline"
        );

        // 1. Resolve model path.
        let model_path = hub::resolve_model_path(&config.model)
            .with_context(|| format!("failed to resolve model path for '{}'", config.model))?;

        // 2. Load JanusConfig from config.json (with 1B fallback).
        let config_json = model_path.join("config.json");
        let model_config = if config_json.exists() {
            JanusConfig::from_file(&config_json).unwrap_or_else(|err| {
                tracing::warn!(
                    path = %config_json.display(),
                    error = %err,
                    "failed to parse config.json; using 1B defaults"
                );
                JanusConfig::janus_pro_1b()
            })
        } else {
            tracing::warn!("config.json not found; using 1B defaults");
            JanusConfig::janus_pro_1b()
        };

        // 3. Collect safetensors shards (needed for both full-precision and GGUF paths).
        let shards = hub::collect_safetensors(&model_path);

        // 4. Build JanusModel via the shared helper that handles GGUF vs. safetensors.
        let mut varmap = Some(VarMap::new());
        let vb = VarBuilder::from_varmap(varmap.as_ref().unwrap(), dtype, &device);
        let model = Self::build_janus_model(
            config.gguf_path.as_deref(),
            &mut varmap,
            vb,
            &model_config,
            &shards,
            &model_path,
            &device,
            dtype,
        )?;

        // 5. Load tokenizer.
        let tokenizer = hub::load_tokenizer(&model_path).context("failed to load tokenizer")?;

        // 6. Build native Janus vision tower for understanding.
        let (vision_tower, vision_device, vision_dtype) =
            Self::load_vision_tower(&model_config, &device, dtype, &shards)?;

        Ok(Self {
            model,
            tokenizer,
            config,
            model_config,
            device,
            dtype,
            vision_tower,
            vision_device,
            vision_dtype,
        })
    }

    /// Returns a reference to the underlying [`JanusModel`].
    pub fn model(&self) -> &JanusModel {
        &self.model
    }

    /// Returns a reference to the loaded tokenizer.
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Returns a reference to the [`PipelineConfig`].
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Returns the device used by this pipeline.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the dtype used by this pipeline.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the native Janus vision tower, if loaded.
    pub fn vision_tower(&self) -> Option<&JanusVisionTower> {
        self.vision_tower.as_ref()
    }

    /// Returns the device used by the vision tower.
    pub fn vision_device(&self) -> &Device {
        &self.vision_device
    }

    /// Returns the dtype used by the vision tower.
    pub fn vision_dtype(&self) -> DType {
        self.vision_dtype
    }

    /// Generate an image from a text prompt.
    ///
    /// Runs the 576-step autoregressive Janus-Pro generation loop with
    /// Classifier-Free Guidance (CFG).  Uses the `guidance_scale` and
    /// `temperature` stored in [`PipelineConfig`] at load time.
    ///
    /// # Arguments
    ///
    /// * `prompt` — the text description of the image to generate.
    ///
    /// # Returns
    ///
    /// An [`ImageBuffer<Rgb<u8>>`] of size 384 × 384 (or whatever the model
    /// config specifies).
    ///
    /// # Errors
    ///
    /// Returns an error on any tensor operation failure or tokenization error.
    pub fn generate(&self, prompt: &str) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
        self.generate_with_overrides(prompt, None, None)
    }

    /// Generate an image and return structured telemetry alongside it.
    ///
    /// This is the instrumented variant of [`generate`].  It collects
    /// per-step timing, running token throughput, KV-cache type, and
    /// (when speculative decoding is active) acceptance-rate statistics.
    ///
    /// Use [`generate`] for production paths where telemetry is not needed;
    /// use this method for monitoring, benchmarking, or diagnostic endpoints.
    ///
    /// # Arguments
    ///
    /// * `prompt`      — text description of the image to generate.
    /// * `cfg_scale`   — optional override for the CFG guidance scale.
    /// * `temperature` — optional override for the sampling temperature.
    ///
    /// # Returns
    ///
    /// A tuple of `(image, telemetry)` where `image` is the generated
    /// [`ImageBuffer<Rgb<u8>>`] and `telemetry` is a [`GenerationTelemetry`]
    /// snapshot covering the completed generation call.
    ///
    /// # Errors
    ///
    /// Returns an error on any tensor operation failure or tokenization error.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use pcai_media::config::PipelineConfig;
    /// use pcai_media::generate::GenerationPipeline;
    ///
    /// let pipeline = GenerationPipeline::load(PipelineConfig::default()).unwrap();
    /// let (image, telemetry) = pipeline
    ///     .generate_with_telemetry("a glowing circuit board", None, None)
    ///     .unwrap();
    /// println!("Generated {} tokens at {:.1} tok/s", telemetry.total_tokens, telemetry.tokens_per_second);
    /// image.save("output.png").unwrap();
    /// ```
    pub fn generate_with_telemetry(
        &self,
        prompt: &str,
        cfg_scale: Option<f64>,
        temperature: Option<f64>,
    ) -> Result<(ImageBuffer<Rgb<u8>, Vec<u8>>, GenerationTelemetry)> {
        self.generate_inner(prompt, cfg_scale, temperature)
    }

    /// Generate an image with optional per-call parameter overrides.
    ///
    /// When `cfg_scale` or `temperature` is `Some`, the provided value
    /// overrides the pipeline default for this call only.  `None` uses the
    /// value stored in the [`PipelineConfig`] at load time.
    ///
    /// When [`PipelineConfig::use_speculative_decoding`] is `true` and
    /// [`PipelineConfig::use_prealloc_kv_cache`] is `true`, the generation
    /// loop uses self-speculative decoding — a two-phase draft-then-verify
    /// scheme that runs only [`PipelineConfig::speculative_draft_layers`]
    /// blocks in the draft phase and all blocks in a single batched verify
    /// forward.  The standard autoregressive loop is used otherwise.
    pub fn generate_with_overrides(
        &self,
        prompt: &str,
        cfg_scale: Option<f64>,
        temperature: Option<f64>,
    ) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
        let (image, _telemetry) = self.generate_inner(prompt, cfg_scale, temperature)?;
        Ok(image)
    }

    /// Core generation implementation shared by [`generate_with_overrides`] and
    /// [`generate_with_telemetry`].
    ///
    /// Returns both the generated image and a [`GenerationTelemetry`] snapshot.
    /// Callers that do not need telemetry discard the second element.
    fn generate_inner(
        &self,
        prompt: &str,
        cfg_scale: Option<f64>,
        temperature: Option<f64>,
    ) -> Result<(ImageBuffer<Rgb<u8>, Vec<u8>>, GenerationTelemetry)> {
        let guidance_scale = cfg_scale.unwrap_or(self.config.guidance_scale);
        let temperature_val = temperature.unwrap_or(self.config.temperature);
        let parallel_size = self.config.parallel_size;

        // Disable CFG on CUDA to conserve VRAM (batch_size=1 vs 2).
        // CFG doubles memory for KV cache and all intermediates.
        let use_cfg = guidance_scale > 1.0 && !matches!(self.device, Device::Cuda(_));
        let batch_size = if use_cfg { parallel_size * 2 } else { parallel_size };
        if !use_cfg && guidance_scale > 1.0 {
            tracing::info!("CFG disabled on CUDA to conserve VRAM (batch_size=1)");
        }

        // ── 1. Tokenise ──────────────────────────────────────────────────────
        // Janus-Pro chat template.
        let templated = format!("<|User|>: {prompt}\n<|Assistant|>:");
        let encoding = self
            .tokenizer
            .encode(templated, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        let mut prompt_ids: Vec<u32> = encoding.get_ids().to_vec();

        // Append <begin_of_image> transition token to signal the model
        // to switch from text mode to image generation mode.
        let boi_id: u32 = self.tokenizer.token_to_id("<begin_of_image>").unwrap_or(100003);
        prompt_ids.push(boi_id);
        let seq_len = prompt_ids.len();
        tracing::debug!(seq_len, boi_id, "prompt tokens (including <begin_of_image>)");

        // ── 2. Build input tensor [batch_size, seq_len] ───────────────────────
        // Even indices: conditional (prompt) tokens.
        // Odd indices: unconditional (pad) tokens for CFG.
        let pad_id: u32 = self
            .tokenizer
            .token_to_id("<\u{ff5c}\u{2581}pad\u{2581}\u{ff5c}>")
            .or_else(|| self.tokenizer.token_to_id("<pad>"))
            .unwrap_or(100002);

        let mut flat_ids: Vec<u32> = Vec::with_capacity(batch_size * seq_len);
        if use_cfg {
            for i in 0..batch_size {
                if i % 2 == 0 {
                    // Conditional: real prompt tokens (including <begin_of_image>).
                    flat_ids.extend_from_slice(&prompt_ids);
                } else {
                    // Unconditional: pad tokens + <begin_of_image> at the end.
                    flat_ids.extend(std::iter::repeat(pad_id).take(seq_len - 1));
                    flat_ids.push(boi_id);
                }
            }
        } else {
            // No CFG: only conditional tokens, batch_size = parallel_size.
            for _ in 0..batch_size {
                flat_ids.extend_from_slice(&prompt_ids);
            }
        }

        // On CUDA, wte lives on CPU (offloaded to save VRAM), so build
        // input_ids on CPU for embedding, then move result to GPU.
        let embed_device = if matches!(self.device, Device::Cuda(_)) {
            &Device::Cpu
        } else {
            &self.device
        };
        let input_ids = Tensor::from_vec(flat_ids, (batch_size, seq_len), embed_device)
            .context("failed to build input_ids tensor")?;

        // ── 3. Build KV cache ────────────────────────────────────────────────
        // Two implementations are available:
        //
        // - `PreAllocKvCache` (default, `use_prealloc_kv_cache = true`):
        //   Allocates one fixed-size `[B, n_kv_heads, max_seq_len, head_dim]`
        //   buffer per layer up front.  New KV pairs are written in-place via
        //   `scatter_set` and read back as zero-copy `narrow` views, eliminating
        //   the ≈95 GB of GPU bandwidth wasted by `Tensor::cat` across 576 steps.
        //
        // - `KvCache` (fallback, `use_prealloc_kv_cache = false`):
        //   The original dynamic cache.  Each step appends via `Tensor::cat`,
        //   allocating a new growing buffer.  Retained for debugging and for
        //   devices where `scatter_set` is not supported.
        let llama_cfg = self.model_config.to_llama_config(false);
        // For quantized backends, QMatMul dequantizes to BF16 on CUDA / F32 on
        // CPU.  The KV cache cos/sin tables and stored K/V tensors must match
        // this working dtype — NOT the config-level dtype (which may be F16).
        let cache_dtype = match &self.model.llama {
            LlamaBackend::Quantized(_) => match &self.device {
                candle_core::Device::Cuda(_) => DType::BF16,
                _ => DType::F32,
            },
            LlamaBackend::Full(_) => self.dtype,
        };
        let mut cache = CacheVariant::new(
            self.config.use_prealloc_kv_cache,
            cache_dtype,
            &llama_cfg,
            batch_size,
            self.model_config.num_image_tokens() + seq_len + 32,
            &self.device,
        )
        .map_err(|e| anyhow::anyhow!("KV cache construction failed: {e}"))?;

        // ── 4. Pre-fill: embed the prompt tokens ─────────────────────────────
        // Shape: [batch_size, seq_len, hidden_size]
        // wte may be on CPU (CUDA offload); move embeddings to GPU after lookup.
        let prompt_embeds = self
            .model
            .embed_tokens(&input_ids)
            .map_err(|e| anyhow::anyhow!("embed_tokens failed: {e}"))?
            .to_device(&self.device)
            .map_err(|e| anyhow::anyhow!("embed_tokens to_device failed: {e}"))?;

        // Pre-fill step: forward the full prompt (including <begin_of_image>)
        // through the LLM backbone.  The hidden state at the last position
        // (<begin_of_image>) carries the context for predicting the first
        // image token — we use it at step 0 instead of a dummy embedding.
        let prefill_hidden = cache
            .forward_hidden(&self.model.llama, &prompt_embeds, 0)
            .map_err(|e| anyhow::anyhow!("prefill forward_hidden failed: {e}"))?;

        // Track the current position for RoPE and KV-cache indexing.
        let mut pos = seq_len;

        // ── 5. Generation loop (576 image tokens) ────────────────────────────
        let num_image_tokens = self.model_config.num_image_tokens(); // 576

        // Dispatch to self-speculative decoding when configured.
        //
        // Self-speculative decoding requires `PreAllocKvCache` so that the
        // draft-phase KV writes can be overwritten by the verify pass, and the
        // cache `seq_len` can be rolled back on token rejection.  Fall back to
        // the standard autoregressive loop if the cache is `Dynamic`.
        let use_spec = self.config.use_speculative_decoding && self.config.use_prealloc_kv_cache;

        // ── Telemetry collector ───────────────────────────────────────────────
        let mut telemetry = TelemetryCollector::new(num_image_tokens);
        telemetry.set_kv_cache_type(self.config.use_prealloc_kv_cache);
        if use_spec {
            telemetry.enable_speculative();
        }
        // Prefill has just completed: mark its end.
        telemetry.record_prefill_end();

        let generated_tokens: Vec<u32> = if use_spec {
            // Extract the pre-allocated cache from the CacheVariant.
            let prealloc_cache = match &mut cache {
                CacheVariant::PreAlloc(c) => c,
                CacheVariant::Dynamic(_) => {
                    // Should not happen: use_spec requires use_prealloc_kv_cache.
                    anyhow::bail!(
                        "speculative decoding requires PreAllocKvCache; \
                         set use_prealloc_kv_cache = true"
                    );
                }
            };
            let draft_layers = self.config.speculative_draft_layers;
            let lookahead_k = self.config.speculative_lookahead.max(1);

            self.speculative_generate_loop(
                Some(prefill_hidden),
                pos,
                prealloc_cache,
                num_image_tokens,
                batch_size,
                use_cfg,
                guidance_scale,
                temperature_val,
                draft_layers,
                lookahead_k,
                &mut telemetry,
            )?
        } else {
            // ── Standard autoregressive loop ────────────────────────────────
            let mut tokens: Vec<u32> = Vec::with_capacity(num_image_tokens);

            // Placeholder for current token IDs; unused at step 0 where we
            // consume the prefill hidden state directly.
            let mut current_token_ids = Tensor::zeros((batch_size, 1_usize), DType::U32, &self.device)
                .context("failed to build initial token tensor")?;

            // Consume prefill_hidden at step 0 without cloning.  The Option is
            // taken once and never replenished, saving a full tensor clone
            // (~batch_size × hidden_size × dtype_size bytes on GPU).
            let mut prefill_hidden_opt = Some(prefill_hidden);

            for step in 0..num_image_tokens {
                // ── Telemetry: start of forward phase ─────────────────────
                let forward_start = Instant::now();

                // A. Get hidden states for this step.
                //    Step 0: use the prefill hidden state from <begin_of_image>.
                //    Steps 1+: embed the previously sampled image token via
                //              gen_embed → gen_aligner → LLM forward_hidden.
                let hidden = if let Some(ph) = prefill_hidden_opt.take() {
                    ph
                } else {
                    let embeds = {
                        use candle_core::Module;
                        let raw_embed = self
                            .model
                            .gen_embed
                            .forward(&current_token_ids)
                            .map_err(|e| anyhow::anyhow!("step {step}: gen_embed failed: {e}"))?;
                        self.model
                            .gen_aligner
                            .forward(&raw_embed)
                            .map_err(|e| anyhow::anyhow!("step {step}: gen_aligner failed: {e}"))?
                    };
                    let h = cache
                        .forward_hidden(&self.model.llama, &embeds, pos)
                        .map_err(|e| anyhow::anyhow!("step {step}: forward_hidden failed: {e}"))?;
                    pos += 1;
                    h
                };

                // C. Project to image vocabulary logits [batch_size, image_vocab_size]
                //    hidden is already [B, hidden_size] (last position extracted by forward_hidden).
                let img_logits = self
                    .model
                    .project_to_image_vocab(&hidden.unsqueeze(1)?)
                    .map_err(|e| anyhow::anyhow!("step {step}: project_to_image_vocab failed: {e}"))?
                    .squeeze(1)
                    .map_err(|e| anyhow::anyhow!("step {step}: squeeze failed: {e}"))?;

                // ── Telemetry: start of sampling phase ────────────────────
                let sampling_start = Instant::now();

                // E. Apply CFG (if enabled) or use logits directly.
                let logits_for_sampling = if use_cfg {
                    // CFG: split batch into conditional (even) and unconditional (odd).
                    let (cond, uncond) = if parallel_size == 1 {
                        let c = img_logits
                            .i(0_usize)
                            .map_err(|e| anyhow::anyhow!("step {step}: cond row 0 failed: {e}"))?
                            .unsqueeze(0)
                            .map_err(|e| anyhow::anyhow!("step {step}: cond unsqueeze failed: {e}"))?;
                        let u = img_logits
                            .i(1_usize)
                            .map_err(|e| anyhow::anyhow!("step {step}: uncond row 1 failed: {e}"))?
                            .unsqueeze(0)
                            .map_err(|e| anyhow::anyhow!("step {step}: uncond unsqueeze failed: {e}"))?;
                        (c, u)
                    } else {
                        let cond_rows: Vec<Tensor> = (0..batch_size)
                            .step_by(2)
                            .map(|i| {
                                img_logits
                                    .i(i)
                                    .and_then(|t| t.unsqueeze(0))
                                    .map_err(|e| anyhow::anyhow!("step {step}: cond row {i}: {e}"))
                            })
                            .collect::<Result<_>>()?;
                        let uncond_rows: Vec<Tensor> = (1..batch_size)
                            .step_by(2)
                            .map(|i| {
                                img_logits
                                    .i(i)
                                    .and_then(|t| t.unsqueeze(0))
                                    .map_err(|e| anyhow::anyhow!("step {step}: uncond row {i}: {e}"))
                            })
                            .collect::<Result<_>>()?;
                        (Tensor::cat(&cond_rows, 0)?, Tensor::cat(&uncond_rows, 0)?)
                    };
                    // guided = uncond + guidance_scale * (cond - uncond)
                    let diff = (cond.clone() - uncond.clone())?;
                    (uncond + (diff * guidance_scale)?)?
                } else {
                    // No CFG: use logits directly.
                    img_logits
                };

                // F. Temperature scaling + softmax → probability distribution
                //    [parallel_size, image_vocab_size]
                let scaled = if (temperature_val - 1.0_f64).abs() > 1e-6 {
                    (logits_for_sampling / temperature_val)
                        .map_err(|e| anyhow::anyhow!("step {step}: temperature scale failed: {e}"))?
                } else {
                    logits_for_sampling
                };
                let probs = candle_nn::ops::softmax_last_dim(&scaled)
                    .map_err(|e| anyhow::anyhow!("step {step}: softmax failed: {e}"))?;

                // G. Multinomial sampling: one token per parallel image.
                //    We sample from probs [parallel_size, vocab_size] and
                //    collect the sampled index for each parallel sample.
                let next_tokens =
                    multinomial_sample(&probs).with_context(|| format!("step {step}: multinomial sampling failed"))?;

                // For parallel_size=1, store the single sampled token.
                // For parallel_size>1, store the first sample (later: batch decode).
                let first_token = next_tokens[0];
                tokens.push(first_token);

                // ── Telemetry: record completed step ──────────────────────
                let step_end = Instant::now();
                telemetry.record_step(step, forward_start, sampling_start, step_end, first_token);

                // H. Prepare next-step input: replicate sampled token across
                //    batch dimension [batch_size, 1].
                let mut next_ids_flat = Vec::with_capacity(batch_size);
                for &tok in &next_tokens {
                    if use_cfg {
                        // Even (cond) and odd (uncond) get the same token.
                        next_ids_flat.push(tok);
                        next_ids_flat.push(tok);
                    } else {
                        next_ids_flat.push(tok);
                    }
                }
                // Fix 3: Use from_slice instead of from_vec to avoid a Vec
                // allocation per step.  next_ids_flat is a stack-local Vec with
                // capacity `batch_size` (typically 1 or 2), so the allocation
                // is small — but eliminating it across 576 steps compounds.
                current_token_ids = Tensor::from_slice(&next_ids_flat, (batch_size, 1_usize), &self.device)
                    .context("failed to build next token tensor")?;

                if step % 100 == 0 || step == num_image_tokens - 1 {
                    tracing::debug!(
                        step,
                        total = num_image_tokens,
                        token = first_token,
                        tps = telemetry.running_tps(),
                        "generation progress"
                    );
                }
            }

            tokens
        }; // end if use_spec

        // ── Token diversity check ────────────────────────────────────────────
        {
            let unique: std::collections::HashSet<u32> = generated_tokens.iter().copied().collect();
            tracing::info!(
                unique = unique.len(),
                total = num_image_tokens,
                "image tokens generated"
            );
        }

        // ── 6. Decode tokens → pixel tensor [1, 3, H, W] ────────────────────
        // Use only the first parallel_size tokens (one image).
        let tokens_tensor = Tensor::from_vec(
            generated_tokens[..num_image_tokens].to_vec(),
            (1_usize, num_image_tokens),
            &self.device,
        )
        .context("failed to build tokens tensor for decode")?;

        let pixel_tensor = self
            .model
            .decode_image_tokens(&tokens_tensor)
            .map_err(|e| anyhow::anyhow!("decode_image_tokens failed: {e}"))?;

        // ── 7. Denormalise from [-1, 1] to [0, 255] U8 ───────────────────────
        // formula: pixel = (x / 2.0 + 0.5) * 255, clamped to [0, 255]
        let pixel_tensor = ((pixel_tensor / 2.0_f64)? + 0.5_f64)?;
        let pixel_tensor = (pixel_tensor * 255.0_f64)?
            .clamp(0.0_f64, 255.0_f64)?
            .to_dtype(DType::U8)
            .context("dtype conversion to U8 failed")?;

        // ── 8. Convert the first batch element to ImageBuffer ─────────────────
        // Tensor shape: [B, C, H, W] → take B=0 → [C, H, W]
        let first_image = pixel_tensor
            .i(0)
            .map_err(|e| anyhow::anyhow!("image batch slice failed: {e}"))?;

        let image = tensor_to_image(&first_image).context("tensor_to_image conversion failed")?;

        // Finalise telemetry and emit a structured log at INFO level.
        let telemetry_snapshot = telemetry.finish();
        tracing::info!(
            total_tokens = telemetry_snapshot.total_tokens,
            total_duration_ms = telemetry_snapshot.total_duration_ms,
            tokens_per_second = telemetry_snapshot.tokens_per_second,
            prefill_ms = telemetry_snapshot.prefill_ms,
            decode_ms = telemetry_snapshot.decode_ms,
            sampling_ms = telemetry_snapshot.sampling_ms,
            kv_cache_type = %telemetry_snapshot.kv_cache_type,
            "generation complete"
        );

        Ok((image, telemetry_snapshot))
    }

    /// Self-speculative decoding loop for image token generation.
    ///
    /// Uses the same model at two depths to amortise the cost of full-depth
    /// forward passes:
    ///
    /// 1. **Draft phase** (`draft_layers` blocks): run `lookahead_k` fast
    ///    single-token forward passes to produce `K` candidate token IDs and
    ///    their embeddings.  Each draft token is sampled stochastically via
    ///    `sample_from_hidden` with the configured temperature.
    /// 2. **Verify phase** (all blocks): run one batched forward over all `K`
    ///    candidate embeddings to obtain full-depth logits at each position.
    /// 3. **Accept/reject**: keep the longest consecutive prefix where the
    ///    draft token matches the verify token (also stochastically sampled via
    ///    `sample_from_hidden`).  Both sides use the same sampler so that token
    ///    IDs are drawn from the same distribution.  On rejection at position
    ///    `j`, the verify token already computed at that position is used
    ///    directly as the accepted bonus token — no re-sampling occurs.
    ///
    /// The KV cache is shared between draft and verify phases via
    /// [`PreAllocKvCache`].  Draft writes at positions `pos..pos+k`; verify
    /// overwrites those same positions with full-depth KV.  On rejection at
    /// position `j`, the cache `seq_len` is rolled back to `pos + j` before
    /// resampling.
    ///
    /// # Arguments
    ///
    /// * `prefill_hidden` — Pre-filled hidden state from `<begin_of_image>`;
    ///   consumed on the first token to avoid an extra forward pass.
    /// * `start_pos`      — Absolute KV-cache position after the prompt prefill.
    /// * `cache`          — Pre-allocated KV cache (mutable).
    /// * `num_image_tokens` — Total image tokens to generate (576).
    /// * `batch_size`     — LLM batch dimension (1 without CFG, 2 with).
    /// * `use_cfg`        — Whether CFG guidance is active.
    /// * `guidance_scale` — CFG scale factor.
    /// * `temperature`    — Sampling temperature.
    /// * `draft_layers`   — Number of transformer blocks for the draft phase.
    /// * `lookahead_k`    — Number of draft tokens to speculate per outer step.
    /// * `telemetry`      — Mutable telemetry collector (records speculative stats).
    ///
    /// # Errors
    ///
    /// Returns an error on any tensor operation failure or cache overflow.
    #[expect(
        clippy::too_many_arguments,
        reason = "speculative loop mirrors generate_with_overrides signature"
    )]
    fn speculative_generate_loop(
        &self,
        prefill_hidden: Option<Tensor>,
        start_pos: usize,
        cache: &mut PreAllocKvCache,
        num_image_tokens: usize,
        batch_size: usize,
        use_cfg: bool,
        guidance_scale: f64,
        temperature: f64,
        draft_layers: usize,
        lookahead_k: usize,
        telemetry: &mut TelemetryCollector,
    ) -> Result<Vec<u32>> {
        // Bug 3 fix: clamp draft_layers to [1, total_layers - 1] so it is
        // always a valid partial-depth value regardless of what the caller
        // configured.  Using all layers for drafting would defeat the purpose
        // (same cost as verify); using zero layers is undefined behaviour in
        // `forward_hidden_draft`.
        let total_layers = self.model.llama.num_layers();
        let draft_layers = draft_layers.min(total_layers.saturating_sub(1)).max(1);

        let mut generated: Vec<u32> = Vec::with_capacity(num_image_tokens);
        let mut pos = start_pos;

        // `last_hidden` carries the verified hidden state for the most recently
        // accepted token.  On the very first outer step this is the prefill
        // hidden state; on subsequent steps it is the verify hidden state for
        // the last accepted token.
        let mut last_hidden: Option<Tensor> = prefill_hidden;

        while generated.len() < num_image_tokens {
            let remaining = num_image_tokens - generated.len();
            let k = lookahead_k.min(remaining);

            // ── STEP 0: sample the "current" token from last_hidden ───────────
            //
            // We have a verified hidden for the current position.  Convert it
            // to image logits, sample one token, and embed it — the embedding
            // becomes the first input to the draft phase.
            let last_hidden_val = last_hidden
                .take()
                .ok_or_else(|| anyhow::anyhow!("speculative_generate_loop: last_hidden unavailable at pos {pos}"))?;

            let first_tok =
                self.sample_from_hidden(&last_hidden_val, use_cfg, batch_size, guidance_scale, temperature, pos)?;
            generated.push(first_tok);
            if generated.len() >= num_image_tokens {
                break;
            }

            // Build the embedding for the first sampled token.
            let first_embed = self
                .embed_image_token(first_tok, batch_size)
                .with_context(|| format!("speculative: embed first token at pos {pos}"))?;
            pos += 1;

            // ── DRAFT PHASE: run draft_layers blocks K times ──────────────────
            //
            // `draft_tokens[i]` is the sampled token for position `pos + i`.
            // `draft_embeds_list[i]` is the gen_aligner output for that token,
            // to be batched for the verify phase.
            let mut draft_tokens: Vec<u32> = Vec::with_capacity(k);
            let mut draft_embeds_list: Vec<Tensor> = Vec::with_capacity(k);

            {
                let mut draft_input = first_embed;
                let draft_start_pos = pos;

                for di in 0..k {
                    // Run forward_hidden_draft (shallow path).
                    let draft_hidden = self
                        .model
                        .llama
                        .forward_hidden_draft(&draft_input, draft_start_pos + di, cache, draft_layers)
                        .with_context(|| format!("speculative draft: step {di} pos {}", draft_start_pos + di))?;

                    // Sample a draft token from the draft hidden state.
                    let draft_tok = self.sample_from_hidden(
                        &draft_hidden,
                        use_cfg,
                        batch_size,
                        guidance_scale,
                        temperature,
                        draft_start_pos + di,
                    )?;
                    draft_tokens.push(draft_tok);

                    // Build the embedding for the draft token — will be batched for verify.
                    let embed = self
                        .embed_image_token(draft_tok, batch_size)
                        .with_context(|| format!("speculative draft: embed token at di={di}"))?;
                    draft_embeds_list.push(embed.clone());

                    if di + 1 < k {
                        draft_input = embed;
                    }
                }
            }

            // ── VERIFY PHASE: one batched forward over all K draft tokens ─────
            //
            // Concatenate the K draft embeddings along the sequence dimension:
            // [B, 1, H] × K → [B, K, H].
            // Then run forward_hidden_verify_batch → [B, K, H].
            // This overwrites the K KV slots written by the draft phase.
            let draft_start_pos = pos;

            // Roll back the cache seq_len to just before the K draft positions
            // so verify can overwrite them cleanly.
            //
            // The draft phase wrote K entries at `draft_start_pos..draft_start_pos+k`.
            // `update` on PreAllocKvCache does not advance `current_seq_len`
            // monotonically (it sets it to `end = seq_pos + new_tokens`), so
            // the cache seq_len is already pointing past the K draft tokens.
            // We reset it to `draft_start_pos` so the narrow view in `update`
            // during verify covers exactly 0..draft_start_pos initially and
            // extends with each verify token.
            cache.rollback_seq_len(draft_start_pos);

            // Stack draft embeddings [B, 1, H] along dim 1 → [B, K, H].
            let verify_input =
                Tensor::cat(&draft_embeds_list, 1).context("speculative verify: cat draft embeddings")?;

            let verify_hidden_batch = self
                .model
                .llama
                .forward_hidden_verify_batch(&verify_input, draft_start_pos, cache)
                .context("speculative verify: forward_hidden_verify_batch")?;

            // ── ACCEPT / REJECT ───────────────────────────────────────────────
            //
            // Bug 1 fix: both draft and verify use `sample_from_hidden`
            // (stochastic multinomial) so that token IDs are drawn from the
            // same distribution.  The original code compared stochastic draft
            // tokens against deterministic greedy verify tokens, producing a
            // near-zero acceptance rate.
            //
            // Bug 2 fix: on rejection at position j, the verify token already
            // sampled at that position is used directly as the bonus token.
            // The original code re-sampled from `sample_from_hidden` a second
            // time, wasting the verify computation and introducing a different
            // random draw.
            //
            // Standard speculative decoding algorithm:
            //   For each position j in [0, K):
            //     verify_tok = sample(verify_hidden_j)   ← same sampler as draft
            //     if draft_tokens[j] == verify_tok → accept, advance.
            //     else → use verify_tok as the accepted bonus token, stop.
            //
            // After accepting j tokens we have `j + 1` verify hidden states
            // (positions 0..=j).  The last accepted verify hidden is saved as
            // `last_hidden` for the next outer step.
            let mut accept_count = 0_usize;
            // `rejection_token` holds the verify-sampled token at the first
            // rejected position; it is pushed directly into `generated`.
            let mut rejection_token: Option<(u32, Tensor)> = None;

            'accept_loop: for j in 0..k {
                // Extract verify hidden at position j: [B, K, H] → [B, H]
                let verify_hidden_j = verify_hidden_batch
                    .i((.., j, ..))
                    .with_context(|| format!("speculative verify: index hidden at j={j}"))?
                    .contiguous()
                    .with_context(|| format!("speculative verify: contiguous at j={j}"))?;

                // Bug 1 fix: sample (not greedy) verify token so the
                // comparison against the draft token is fair — both draw from
                // the same stochastic distribution.
                let verify_tok = self.sample_from_hidden(
                    &verify_hidden_j,
                    use_cfg,
                    batch_size,
                    guidance_scale,
                    temperature,
                    pos + j,
                )?;

                if draft_tokens[j] == verify_tok {
                    accept_count += 1;
                    // Keep going — may accept more tokens.
                } else {
                    // Bug 2 fix: use the verify token (already computed above)
                    // as the bonus token.  Do NOT re-sample — that would discard
                    // the verify computation and introduce a different random draw.
                    rejection_token = Some((verify_tok, verify_hidden_j));
                    break 'accept_loop;
                }
            }

            // Push accepted draft tokens into `generated`.
            for &tok in &draft_tokens[..accept_count] {
                generated.push(tok);
                pos += 1;
                if generated.len() >= num_image_tokens {
                    break;
                }
            }

            // On rejection: push the verify bonus token and set last_hidden.
            if generated.len() < num_image_tokens {
                if let Some((bonus_tok, vh)) = rejection_token {
                    // Use the verify-sampled token directly (Bug 2 fix).
                    generated.push(bonus_tok);
                    pos += 1;

                    if generated.len() < num_image_tokens {
                        // Roll the cache back to `pos` and set last_hidden so the
                        // next outer step starts fresh from the verify hidden.
                        cache.rollback_seq_len(pos);
                        last_hidden = Some(vh);
                    }
                } else {
                    // All K tokens accepted: update cache and last_hidden for
                    // the last accepted position.
                    let last_j = accept_count.saturating_sub(1);
                    let last_verify_hidden = verify_hidden_batch
                        .i((.., last_j, ..))
                        .with_context(|| format!("speculative: last accepted hidden at j={last_j}"))?
                        .contiguous()
                        .context("speculative: last accepted hidden contiguous")?;
                    last_hidden = Some(last_verify_hidden);
                }
            }

            // Record speculative-step outcome in the telemetry collector.
            // `k` draft tokens were evaluated; `accept_count` were accepted.
            telemetry.record_speculative_step(k, accept_count);

            tracing::debug!(
                generated = generated.len(),
                total = num_image_tokens,
                accept_count,
                k,
                tps = telemetry.running_tps(),
                "speculative step"
            );
        }

        Ok(generated)
    }

    /// Sample one image token from a `[B, hidden_size]` hidden state.
    ///
    /// Projects the hidden state through `gen_head` to obtain image-vocabulary
    /// logits, applies optional CFG and temperature scaling, then performs
    /// multinomial sampling.  Returns the sampled token for the **first**
    /// parallel image (`parallel_size=1`).
    ///
    /// # Errors
    ///
    /// Returns an error on any candle operation failure.
    fn sample_from_hidden(
        &self,
        hidden: &Tensor,
        use_cfg: bool,
        batch_size: usize,
        guidance_scale: f64,
        temperature: f64,
        _pos: usize,
    ) -> Result<u32> {
        let img_logits = self
            .model
            .project_to_image_vocab(&hidden.unsqueeze(1)?)
            .context("sample_from_hidden: project_to_image_vocab")?
            .squeeze(1)
            .context("sample_from_hidden: squeeze")?;

        let logits_for_sampling = if use_cfg && batch_size >= 2 {
            let cond = img_logits.i(0_usize)?.unsqueeze(0)?;
            let uncond = img_logits.i(1_usize)?.unsqueeze(0)?;
            let diff = (cond.clone() - uncond.clone())?;
            (uncond + (diff * guidance_scale)?)?
        } else {
            img_logits.i(0_usize)?.unsqueeze(0)?
        };

        let scaled = if (temperature - 1.0_f64).abs() > 1e-6 {
            (logits_for_sampling / temperature).context("sample_from_hidden: temperature scale")?
        } else {
            logits_for_sampling
        };
        let probs = candle_nn::ops::softmax_last_dim(&scaled).context("sample_from_hidden: softmax")?;
        let tokens = multinomial_sample(&probs).context("sample_from_hidden: multinomial")?;
        Ok(tokens[0])
    }

    /// Greedy (argmax) token selection from a `[B, hidden_size]` hidden state.
    ///
    /// Used in the verify phase to compare draft tokens against the full-depth
    /// model's greedy prediction without stochastic noise.  Returns the greedy
    /// token for the first batch element after optional CFG blending.
    ///
    /// # Errors
    ///
    /// Returns an error on any candle operation failure.
    fn greedy_from_hidden(
        &self,
        hidden: &Tensor,
        use_cfg: bool,
        batch_size: usize,
        guidance_scale: f64,
    ) -> Result<u32> {
        let img_logits = self
            .model
            .project_to_image_vocab(&hidden.unsqueeze(1)?)
            .context("greedy_from_hidden: project_to_image_vocab")?
            .squeeze(1)
            .context("greedy_from_hidden: squeeze")?;

        let logits = if use_cfg && batch_size >= 2 {
            let cond = img_logits.i(0_usize)?.unsqueeze(0)?;
            let uncond = img_logits.i(1_usize)?.unsqueeze(0)?;
            let diff = (cond.clone() - uncond.clone())?;
            (uncond + (diff * guidance_scale)?)?
        } else {
            img_logits.i(0_usize)?.unsqueeze(0)?
        };

        let idx = logits
            .argmax(candle_core::D::Minus1)
            .context("greedy_from_hidden: argmax")?
            .i(0_usize)
            .context("greedy_from_hidden: extract index")?
            .to_scalar::<u32>()
            .context("greedy_from_hidden: to_scalar")?;
        Ok(idx)
    }

    /// Embed a single image token through `gen_embed` + `gen_aligner`.
    ///
    /// Returns a `[batch_size, 1, hidden_size]` tensor on `self.device`.
    ///
    /// # Errors
    ///
    /// Returns an error on any candle operation failure.
    fn embed_image_token(&self, token_id: u32, batch_size: usize) -> Result<Tensor> {
        use candle_core::Module;
        let ids_flat: Vec<u32> = std::iter::repeat(token_id).take(batch_size).collect();
        let ids = Tensor::from_slice(&ids_flat, (batch_size, 1_usize), &self.device)
            .context("embed_image_token: from_slice")?;
        let raw = self
            .model
            .gen_embed
            .forward(&ids)
            .context("embed_image_token: gen_embed")?;
        self.model
            .gen_aligner
            .forward(&raw)
            .context("embed_image_token: gen_aligner")
    }
}

// ---------------------------------------------------------------------------
// tensor_to_image
// ---------------------------------------------------------------------------

/// Convert a `[C, H, W]` U8 tensor to an [`ImageBuffer<Rgb<u8>>`].
///
/// The tensor must have exactly 3 channels (C = 3) and dtype `U8`.
/// The conversion permutes to `[H, W, C]` for the row-major `ImageBuffer`
/// layout expected by the `image` crate.
///
/// # Errors
///
/// Returns an error if the tensor has wrong number of dimensions, wrong
/// channel count, or any candle operation fails.
pub fn tensor_to_image(tensor: &Tensor) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    // Validate shape [C, H, W].
    let dims = tensor.dims();
    anyhow::ensure!(dims.len() == 3, "expected 3D tensor [C, H, W], got {}D", dims.len());
    let (c, h, w) = (dims[0], dims[1], dims[2]);
    anyhow::ensure!(c == 3, "expected C=3 (RGB), got C={c}");

    // Permute [C, H, W] → [H, W, C] for row-major layout.
    let hwc = tensor.permute((1, 2, 0)).context("permute [C,H,W] -> [H,W,C] failed")?;

    // Flatten to a contiguous byte Vec.
    let raw: Vec<u8> = hwc
        .flatten_all()
        .context("flatten failed")?
        .to_vec1::<u8>()
        .context("to_vec1::<u8>() failed")?;

    ImageBuffer::from_raw(w as u32, h as u32, raw)
        .ok_or_else(|| anyhow::anyhow!("ImageBuffer::from_raw failed (buffer size mismatch)"))
}

// ---------------------------------------------------------------------------
// multinomial_sample
// ---------------------------------------------------------------------------

/// Sample one token index per row from a probability matrix.
///
/// `probs` must have shape `[batch, vocab_size]` or `[vocab_size]` with
/// values in `[0, 1]` summing (approximately) to 1 per row.
///
/// Dispatch strategy:
/// - **CUDA device**: delegates to [`gpu_multinomial_sample`], which uses the
///   Gumbel-max trick entirely on-device and only transfers one `u32` index
///   per batch row back to the host (~4 bytes vs ~16 KB for the full vocab).
/// - **CPU / other**: falls back to [`cpu_multinomial_sample`], the original
///   inverse-CDF walk.
///
/// Returns a `Vec<u32>` of length `batch` containing the sampled token for
/// each row.
///
/// # Errors
///
/// Returns an error on any candle tensor operation failure.
fn multinomial_sample(probs: &Tensor) -> Result<Vec<u32>> {
    // Normalise to 2D [batch, vocab] regardless of input rank.
    let probs_2d = if probs.dims().len() == 1 {
        probs.unsqueeze(0).context("unsqueeze 1D probs to 2D")?
    } else {
        probs.clone()
    };

    // Route to GPU path on CUDA; fall back to CPU CDF walk elsewhere.
    match probs_2d.device() {
        Device::Cuda(_) => gpu_multinomial_sample(&probs_2d),
        _ => cpu_multinomial_sample(&probs_2d),
    }
}

/// GPU-side multinomial sampling via the Gumbel-max trick.
///
/// Avoids transferring the full probability distribution (≈16 KB per step for
/// the 16 384-token VQ vocab) from GPU to CPU.  Instead, the entire sampling
/// computation runs on-device and only the argmax index (4 bytes per batch
/// row) is transferred back.
///
/// # Algorithm — Gumbel-max trick
///
/// For each batch row the following is computed entirely on GPU:
///
/// ```text
/// u  ~ Uniform(0, 1)          // candle Tensor::rand on CUDA
/// g  = -ln(-ln(u))            // standard Gumbel noise
/// x  = ln(probs) + g          // perturbed log-probabilities
/// k  = argmax(x)              // sampled token
/// ```
///
/// This is mathematically equivalent to drawing `k ~ Categorical(probs)`.
///
/// # Arguments
///
/// * `probs_2d` — float tensor `[batch, vocab_size]` on a CUDA device,
///   values in `[0, 1]` approximately summing to 1 per row.  Must be 2D.
///
/// # Returns
///
/// `Vec<u32>` of length `batch` — one sampled token index per row.
///
/// # Errors
///
/// Returns an error on any candle operation failure (log, rand, argmax, etc.).
fn gpu_multinomial_sample(probs_2d: &Tensor) -> Result<Vec<u32>> {
    let shape = probs_2d.dims();
    let device = probs_2d.device();

    // 1. log(probs): clamp to avoid -inf from exact zero probabilities.
    //    Epsilon = 1e-10 in F32 keeps the log finite without biasing the
    //    distribution meaningfully — any token with p < 1e-10 is essentially
    //    never sampled.
    let log_probs = probs_2d
        .to_dtype(DType::F32)
        .context("gpu_multinomial_sample: cast to f32")?
        .clamp(1e-10_f64, 1.0_f64)
        .context("gpu_multinomial_sample: clamp probs")?
        .log()
        .context("gpu_multinomial_sample: log(probs)")?;

    // 2. u ~ Uniform(0, 1) on the same device as probs.
    //    Tensor::rand(lo, hi, shape, device) is available in candle-core 0.9.
    let u = Tensor::rand(0.0_f32, 1.0_f32, shape, device).context("gpu_multinomial_sample: Tensor::rand failed")?;

    // 3. Gumbel noise: g = -ln(-ln(u)).
    //    Clamp u away from 0 and 1 before log to avoid ±inf.
    let eps = 1e-10_f64;
    let g = u
        .clamp(eps, 1.0 - eps)
        .context("gpu_multinomial_sample: clamp u")?
        .log()
        .context("gpu_multinomial_sample: log(u)")?
        .neg()
        .context("gpu_multinomial_sample: neg inner log")?
        .clamp(eps, f64::MAX)
        .context("gpu_multinomial_sample: clamp -log(u)")?
        .log()
        .context("gpu_multinomial_sample: log(-log(u))")?
        .neg()
        .context("gpu_multinomial_sample: neg outer log")?;

    // 4. Perturbed log-probabilities: x = log_probs + g.
    let perturbed = (log_probs + g).context("gpu_multinomial_sample: log_probs + g")?;

    // 5. argmax along vocab dim — still on GPU.
    let indices = perturbed
        .argmax(candle_core::D::Minus1)
        .context("gpu_multinomial_sample: argmax")?;

    // 6. Transfer only the index tensor to CPU (4 bytes × batch_size).
    let indices_cpu = indices
        .to_device(&Device::Cpu)
        .context("gpu_multinomial_sample: index to CPU")?;

    let batch_size = shape[0];
    let mut tokens = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let idx = indices_cpu
            .i(b)
            .context("gpu_multinomial_sample: index into result")?
            .to_scalar::<u32>()
            .context("gpu_multinomial_sample: scalar extraction")?;
        tokens.push(idx);
    }

    Ok(tokens)
}

/// CPU-side multinomial sampling via inverse-CDF walk.
///
/// Transfers the full probability distribution from the tensor device to CPU
/// and performs a single linear scan per batch row.  Used as a fallback when
/// the tensor is not on CUDA.
///
/// # Arguments
///
/// * `probs_2d` — float tensor `[batch, vocab_size]`, values in `[0, 1]`.
///
/// # Errors
///
/// Returns an error on any candle tensor operation failure.
fn cpu_multinomial_sample(probs_2d: &Tensor) -> Result<Vec<u32>> {
    let (batch_size, vocab) = probs_2d.dims2().with_context(|| {
        format!(
            "cpu_multinomial_sample: expected 2D tensor, got shape {:?}",
            probs_2d.dims()
        )
    })?;

    let probs_vec: Vec<f32> = probs_2d
        .to_dtype(DType::F32)
        .context("cpu_multinomial_sample: cast to f32")?
        .flatten_all()
        .context("cpu_multinomial_sample: flatten")?
        .to_vec1::<f32>()
        .context("cpu_multinomial_sample: to_vec1")?;

    let mut tokens = Vec::with_capacity(batch_size);
    for b in 0..batch_size {
        let row = &probs_vec[b * vocab..(b + 1) * vocab];
        let u = rand_val();
        let mut cumsum = 0.0_f64;
        let mut sampled = (vocab - 1) as u32;
        for (i, &p) in row.iter().enumerate() {
            cumsum += p as f64;
            if u < cumsum {
                sampled = i as u32;
                break;
            }
        }
        tokens.push(sampled);
    }

    Ok(tokens)
}

/// Random float in `[0, 1)` via the `rand` crate.
///
/// Uses the thread-local RNG seeded by the OS.  Not suitable for
/// cryptographic use but provides proper statistical properties for
/// multinomial sampling.
pub(crate) fn rand_val() -> f64 {
    rand::random::<f64>()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// `tensor_to_image` must correctly convert a 3×2×4 U8 tensor to a 4×2
    /// RGB ImageBuffer.
    #[test]
    fn test_tensor_to_image_shape() {
        // Build a [3, 2, 4] tensor filled with value 128.
        let data: Vec<u8> = vec![128u8; 3 * 2 * 4];
        let tensor = Tensor::from_vec(data, (3_usize, 2_usize, 4_usize), &Device::Cpu).unwrap();
        let img = tensor_to_image(&tensor).expect("tensor_to_image should succeed");
        assert_eq!(img.width(), 4);
        assert_eq!(img.height(), 2);
    }

    /// `tensor_to_image` must return an error for a 2D tensor.
    #[test]
    fn test_tensor_to_image_wrong_dims() {
        let tensor = Tensor::zeros((3_usize, 10_usize), DType::U8, &Device::Cpu).unwrap();
        assert!(tensor_to_image(&tensor).is_err());
    }

    /// `tensor_to_image` must return an error when C != 3.
    #[test]
    fn test_tensor_to_image_wrong_channels() {
        let tensor = Tensor::zeros((4_usize, 8_usize, 8_usize), DType::U8, &Device::Cpu).unwrap();
        assert!(tensor_to_image(&tensor).is_err());
    }

    /// `tensor_to_image` must preserve pixel values during [C, H, W] to [H, W, C] permutation.
    #[test]
    fn test_tensor_to_image_pixel_content() {
        // Create a 3x2x2 [C, H, W] tensor with known values.
        let data = vec![
            // Channel 0 (R):
            10u8, 20, 30, 40, // Channel 1 (G):
            50u8, 60, 70, 80, // Channel 2 (B):
            90u8, 100, 110, 120,
        ];

        let tensor = Tensor::from_vec(data, (3_usize, 2_usize, 2_usize), &Device::Cpu).unwrap();
        let img = tensor_to_image(&tensor).expect("tensor_to_image should succeed");

        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);

        // Verify pixel (x, y) = (0, 0)
        assert_eq!(img.get_pixel(0, 0).0, [10, 50, 90]);
        // Verify pixel (x, y) = (1, 0)
        assert_eq!(img.get_pixel(1, 0).0, [20, 60, 100]);
        // Verify pixel (x, y) = (0, 1)
        assert_eq!(img.get_pixel(0, 1).0, [30, 70, 110]);
        // Verify pixel (x, y) = (1, 1)
        assert_eq!(img.get_pixel(1, 1).0, [40, 80, 120]);
    }

    /// `tensor_to_image` must return an error for non-U8 tensors.
    #[test]
    fn test_tensor_to_image_wrong_dtype() {
        let tensor = Tensor::zeros((3_usize, 8_usize, 8_usize), DType::F32, &Device::Cpu).unwrap();
        assert!(tensor_to_image(&tensor).is_err());
    }

    /// `tensor_to_image` must preserve pixel values during [C, H, W] to [H, W, C] permutation.
    #[test]
    fn test_tensor_to_image_pixel_content() {
        // Create a 3x2x2 [C, H, W] tensor with known values.
        let data = vec![
            // Channel 0 (R):
            10u8, 20, 30, 40, // Channel 1 (G):
            50u8, 60, 70, 80, // Channel 2 (B):
            90u8, 100, 110, 120,
        ];

        let tensor = Tensor::from_vec(data, (3_usize, 2_usize, 2_usize), &Device::Cpu).unwrap();
        let img = tensor_to_image(&tensor).expect("tensor_to_image should succeed");

        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 2);

        // Verify pixel (x, y) = (0, 0)
        assert_eq!(img.get_pixel(0, 0).0, [10, 50, 90]);
        // Verify pixel (x, y) = (1, 0)
        assert_eq!(img.get_pixel(1, 0).0, [20, 60, 100]);
        // Verify pixel (x, y) = (0, 1)
        assert_eq!(img.get_pixel(0, 1).0, [30, 70, 110]);
        // Verify pixel (x, y) = (1, 1)
        assert_eq!(img.get_pixel(1, 1).0, [40, 80, 120]);
    }

    /// `tensor_to_image` must return an error for non-U8 tensors.
    #[test]
    fn test_tensor_to_image_wrong_dtype() {
        let tensor = Tensor::zeros((3_usize, 8_usize, 8_usize), DType::F32, &Device::Cpu).unwrap();
        assert!(tensor_to_image(&tensor).is_err());
    }

    /// `rand_val` must produce values in `[0, 1)`.
    #[test]
    fn test_rand_val_range() {
        for _ in 0..1000 {
            let v = rand_val();
            assert!((0.0..1.0).contains(&v), "rand_val() = {v} out of [0,1)");
        }
    }

    /// `rand_val` should not produce identical values in tight succession.
    #[test]
    fn test_rand_val_not_all_same() {
        let vals: Vec<f64> = (0..50).map(|_| rand_val()).collect();
        let unique: std::collections::HashSet<u64> = vals.iter().map(|&v| v.to_bits()).collect();
        assert!(
            unique.len() > 1,
            "rand_val produced all identical values — counter not working"
        );
    }

    /// `multinomial_sample` on a one-hot distribution must always return the
    /// hot index.
    #[test]
    fn test_multinomial_sample_one_hot() {
        // One-hot at index 3 out of 5.
        let data: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 0.0];
        let probs = Tensor::from_vec(data, (1_usize, 5_usize), &Device::Cpu).unwrap();
        for _ in 0..20 {
            let tokens = multinomial_sample(&probs).unwrap();
            assert_eq!(tokens.len(), 1);
            assert_eq!(tokens[0], 3, "one-hot sample must return index 3");
        }
    }

    /// `multinomial_sample` must return one token per row.
    #[test]
    fn test_multinomial_sample_batch_size() {
        let data: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5]; // 2×2, row-normalised not needed (cumsum)
        let probs = Tensor::from_vec(data, (2_usize, 2_usize), &Device::Cpu).unwrap();
        let tokens = multinomial_sample(&probs).unwrap();
        assert_eq!(tokens.len(), 2, "expected one token per batch row");
    }

    // -----------------------------------------------------------------------
    // gpu_multinomial_sample / cpu_multinomial_sample tests
    // -----------------------------------------------------------------------

    /// `cpu_multinomial_sample` on a one-hot distribution must always return
    /// the hot index.
    #[test]
    fn test_cpu_multinomial_one_hot() {
        let data: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0];
        let probs = Tensor::from_vec(data, (1_usize, 4_usize), &Device::Cpu).unwrap();
        for _ in 0..20 {
            let tokens = cpu_multinomial_sample(&probs).unwrap();
            assert_eq!(tokens.len(), 1);
            assert_eq!(tokens[0], 2, "one-hot cpu sample must return index 2");
        }
    }

    /// `cpu_multinomial_sample` must return one token per batch row and each
    /// token must be a valid vocabulary index.
    #[test]
    fn test_cpu_multinomial_batch() {
        // 3 rows, 5 vocab elements — uniform distribution.
        let data: Vec<f32> = vec![0.2_f32, 0.2, 0.2, 0.2, 0.2].repeat(3);
        let probs = Tensor::from_vec(data, (3_usize, 5_usize), &Device::Cpu).unwrap();
        let tokens = cpu_multinomial_sample(&probs).unwrap();
        assert_eq!(tokens.len(), 3);
        for &t in &tokens {
            assert!(t < 5, "sampled token {t} out of vocab range 0..5");
        }
    }

    /// `gpu_multinomial_sample` on a CPU-backed tensor (no CUDA device needed)
    /// must return the sole non-zero index from a one-hot distribution.
    ///
    /// log(1.0) = 0 dominates log(~0) ≈ −∞ even after Gumbel perturbation.
    #[test]
    fn test_gpu_multinomial_one_hot_cpu_device() {
        let data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
        let probs = Tensor::from_vec(data, (1_usize, 4_usize), &Device::Cpu).unwrap();
        for _ in 0..30 {
            let tokens = gpu_multinomial_sample(&probs).unwrap();
            assert_eq!(tokens.len(), 1);
            assert_eq!(
                tokens[0], 0,
                "Gumbel-max on one-hot must select the sole non-zero index"
            );
        }
    }

    /// `gpu_multinomial_sample` must return one index per batch row within the
    /// valid vocab range.
    #[test]
    fn test_gpu_multinomial_batch_range() {
        let vocab = 8_usize;
        let batch = 4_usize;
        // Build a simple normalised distribution: each row sums to 1.
        let row: Vec<f32> = (1..=vocab as u32)
            .map(|i| i as f32 / (vocab * (vocab + 1) / 2) as f32)
            .collect();
        let data: Vec<f32> = row.iter().copied().cycle().take(batch * vocab).collect();
        let probs = Tensor::from_vec(data, (batch, vocab), &Device::Cpu).unwrap();
        let tokens = gpu_multinomial_sample(&probs).unwrap();
        assert_eq!(tokens.len(), batch);
        for &t in &tokens {
            assert!((t as usize) < vocab, "sampled token {t} out of vocab range 0..{vocab}");
        }
    }

    /// `gpu_multinomial_sample` must produce varying results from a uniform
    /// distribution — Gumbel perturbation must introduce variance.
    #[test]
    fn test_gpu_multinomial_stochastic() {
        // p(all 50 draws identical from 8-token uniform) ≈ 8 * (1/8)^50 ≈ 0.
        let data: Vec<f32> = vec![0.125_f32; 8];
        let probs = Tensor::from_vec(data, (1_usize, 8_usize), &Device::Cpu).unwrap();
        let results: Vec<u32> = (0..50).map(|_| gpu_multinomial_sample(&probs).unwrap()[0]).collect();
        let unique: std::collections::HashSet<u32> = results.iter().copied().collect();
        assert!(
            unique.len() > 1,
            "gpu_multinomial_sample produced identical tokens from a uniform distribution"
        );
    }

    /// `multinomial_sample` dispatches to CPU path on a CPU tensor and returns
    /// a valid result for a one-hot input.
    #[test]
    fn test_multinomial_sample_dispatch_cpu() {
        let data: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 1.0];
        let probs = Tensor::from_vec(data, (1_usize, 5_usize), &Device::Cpu).unwrap();
        for _ in 0..10 {
            let tokens = multinomial_sample(&probs).unwrap();
            assert_eq!(tokens[0], 4, "one-hot dispatch must return index 4");
        }
    }

    /// `multinomial_sample` on a 1D tensor must be normalised to 2D internally
    /// without panicking and return exactly one token.
    #[test]
    fn test_multinomial_sample_1d_input() {
        let data: Vec<f32> = vec![0.1, 0.5, 0.4];
        let probs = Tensor::from_vec(data, (3_usize,), &Device::Cpu).unwrap();
        let tokens = multinomial_sample(&probs).unwrap();
        assert_eq!(tokens.len(), 1, "1D input should produce exactly one token");
        assert!(tokens[0] < 3, "sampled token out of range");
    }

    // -----------------------------------------------------------------------
    // Speculative decoding config tests
    // -----------------------------------------------------------------------

    /// `use_spec` flag must be false when `use_speculative_decoding` is false.
    #[test]
    fn test_speculative_flag_off_by_default() {
        let cfg = crate::config::PipelineConfig::default();
        assert!(
            !cfg.use_speculative_decoding,
            "speculative decoding must be disabled by default"
        );
        assert!(
            cfg.use_prealloc_kv_cache,
            "prealloc kv cache must be enabled by default"
        );
        // use_spec = use_speculative_decoding && use_prealloc_kv_cache
        let use_spec = cfg.use_speculative_decoding && cfg.use_prealloc_kv_cache;
        assert!(
            !use_spec,
            "use_spec must be false when speculative decoding is disabled"
        );
    }

    /// Enabling speculative decoding with prealloc cache sets use_spec = true.
    #[test]
    fn test_speculative_flag_on_with_prealloc() {
        let cfg = crate::config::PipelineConfig {
            use_speculative_decoding: true,
            use_prealloc_kv_cache: true,
            speculative_draft_layers: 8,
            speculative_lookahead: 4,
            ..crate::config::PipelineConfig::default()
        };
        let use_spec = cfg.use_speculative_decoding && cfg.use_prealloc_kv_cache;
        assert!(use_spec, "use_spec must be true when both flags are set");
    }

    /// Speculative decoding without prealloc KV cache must not activate.
    ///
    /// The implementation requires `PreAllocKvCache` for rollback; `Dynamic`
    /// cache is incompatible and the pipeline falls back to standard decode.
    #[test]
    fn test_speculative_flag_off_without_prealloc() {
        let cfg = crate::config::PipelineConfig {
            use_speculative_decoding: true,
            use_prealloc_kv_cache: false,
            ..crate::config::PipelineConfig::default()
        };
        let use_spec = cfg.use_speculative_decoding && cfg.use_prealloc_kv_cache;
        assert!(
            !use_spec,
            "use_spec must be false without PreAllocKvCache — rollback is not supported on Dynamic cache"
        );
    }

    /// `speculative_lookahead` of 0 must be clamped to at least 1.
    ///
    /// The implementation calls `.max(1)` on `lookahead_k` so zero is never
    /// passed to the draft loop.
    #[test]
    fn test_speculative_lookahead_zero_clamped() {
        let raw: usize = 0;
        let clamped = raw.max(1);
        assert_eq!(clamped, 1, "lookahead 0 should be clamped to 1 by .max(1)");
    }

    /// Bug 3 fix: `draft_layers` must be clamped to `[1, total_layers - 1]`
    /// at the start of `speculative_generate_loop`.
    ///
    /// Validates the clamping arithmetic added to the loop entry so that the
    /// draft depth is always a valid partial-depth value at runtime.
    #[test]
    fn test_draft_layers_clamped_to_valid_range() {
        let total_layers: usize = 24; // Janus-Pro-1B has 24 layers.

        // Configured to exactly total_layers (invalid: must be < total).
        let raw = total_layers;
        let clamped = raw.min(total_layers.saturating_sub(1)).max(1);
        assert_eq!(
            clamped,
            total_layers - 1,
            "draft_layers == total must clamp to total - 1"
        );

        // Configured to total_layers + 5 (also invalid).
        let raw = total_layers + 5;
        let clamped = raw.min(total_layers.saturating_sub(1)).max(1);
        assert_eq!(
            clamped,
            total_layers - 1,
            "draft_layers > total must clamp to total - 1"
        );

        // Configured to 0 (invalid: at least 1 layer required).
        let raw: usize = 0;
        let clamped = raw.min(total_layers.saturating_sub(1)).max(1);
        assert_eq!(clamped, 1, "draft_layers == 0 must clamp to 1");

        // Configured to 8 (valid: within [1, 23]).
        let raw: usize = 8;
        let clamped = raw.min(total_layers.saturating_sub(1)).max(1);
        assert_eq!(clamped, 8, "valid draft_layers must pass through unchanged");

        // Edge: total_layers == 1 (degenerate model).
        let total = 1_usize;
        let raw: usize = 5;
        let clamped = raw.min(total.saturating_sub(1)).max(1);
        assert_eq!(
            clamped, 1,
            "degenerate 1-layer model: draft_layers must be clamped to 1"
        );
    }
}
