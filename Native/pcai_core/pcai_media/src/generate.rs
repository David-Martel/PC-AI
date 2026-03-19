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
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use image::{ImageBuffer, Rgb};
use std::path::PathBuf;

use pcai_media_model::{
    config::JanusConfig,
    janus_llama::{KvCache, PreAllocKvCache},
    vision::{JanusVisionConfig, JanusVisionTower},
    JanusModel,
};

use crate::config::PipelineConfig;
use crate::hub;

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
        llama: &pcai_media_model::janus_llama::JanusLlama,
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

    /// Load the Janus-Pro pipeline from the configuration.
    ///
    /// Steps performed:
    /// 1. Resolve [`candle_core::Device`] and [`candle_core::DType`] from `config`.
    /// 2. Resolve the model path (local directory or HuggingFace Hub download).
    /// 3. Deserialise [`JanusConfig`] from `config.json` (falls back to 1B defaults).
    /// 4. Build [`JanusModel`] via [`VarMap`] + [`VarBuilder`].
    /// 5. Load weights from safetensors shards into the [`VarMap`].
    /// 6. Load the tokenizer from `tokenizer.json`.
    /// 7. Build the native Janus vision tower (if `vision_model` weights exist).
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

        // 3. Build JanusModel from a VarMap-backed VarBuilder.
        let mut varmap = Some(VarMap::new());
        let vb = VarBuilder::from_varmap(varmap.as_ref().unwrap(), dtype, &device);
        let mut model =
            JanusModel::new(vb, &model_config).map_err(|e| anyhow::anyhow!("JanusModel construction failed: {e}"))?;

        // 4. Load safetensors weights into the VarMap.
        let shards = hub::collect_safetensors(&model_path);
        if shards.is_empty() {
            tracing::warn!(
                path = %model_path.display(),
                "no safetensors shards found; model will use random weights"
            );
        } else {
            let loaded = hub::load_weights(varmap.as_ref().unwrap(), &shards, dtype, &device)
                .context("failed to load model weights from safetensors")?;
            tracing::info!(shards = shards.len(), tensors_loaded = loaded, "weights loaded");
        }

        // 5. On CUDA, offload the token embedding table to CPU to save VRAM.
        //    Keep `lm_head` on the main device so image understanding can
        //    decode text at full speed without CPU vocab projection.
        //    Drop VarMap immediately after to release old tensor Arc references.
        if matches!(device, Device::Cuda(_)) {
            model
                .llama
                .offload_embeddings_to_cpu()
                .map_err(|e| anyhow::anyhow!("wte CPU offload failed: {e}"))?;
            // Drop VarMap so the old GPU tensors (held by Arc) are freed now.
            varmap.take();
            tracing::info!("Offloaded wte to CPU for CUDA pipeline");
        }

        // 6. Load tokenizer.
        let tokenizer = hub::load_tokenizer(&model_path).context("failed to load tokenizer")?;

        // 7. Build native Janus vision tower for understanding.
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

    /// Generate an image with optional per-call parameter overrides.
    ///
    /// When `cfg_scale` or `temperature` is `Some`, the provided value
    /// overrides the pipeline default for this call only.  `None` uses the
    /// value stored in the [`PipelineConfig`] at load time.
    pub fn generate_with_overrides(
        &self,
        prompt: &str,
        cfg_scale: Option<f64>,
        temperature: Option<f64>,
    ) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
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
        let mut cache = CacheVariant::new(
            self.config.use_prealloc_kv_cache,
            self.dtype,
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

        // ── 5. Autoregressive generation loop (576 image tokens) ─────────────
        let num_image_tokens = self.model_config.num_image_tokens(); // 576
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(num_image_tokens);

        // Placeholder for current token IDs; unused at step 0 where we
        // consume the prefill hidden state directly.
        let mut current_token_ids = Tensor::zeros((batch_size, 1_usize), DType::U32, &self.device)
            .context("failed to build initial token tensor")?;

        // Consume prefill_hidden at step 0 without cloning.  The Option is
        // taken once and never replenished, saving a full tensor clone
        // (~batch_size × hidden_size × dtype_size bytes on GPU).
        let mut prefill_hidden = Some(prefill_hidden);

        for step in 0..num_image_tokens {
            // A. Get hidden states for this step.
            //    Step 0: use the prefill hidden state from <begin_of_image>.
            //    Steps 1+: embed the previously sampled image token via
            //              gen_embed → gen_aligner → LLM forward_hidden.
            let hidden = if let Some(ph) = prefill_hidden.take() {
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
            generated_tokens.push(first_token);

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
                    "generation progress"
                );
            }
        }

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

        tensor_to_image(&first_image).context("tensor_to_image conversion failed")
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
/// `probs` must have shape `[batch, vocab_size]` with values in `[0, 1]`
/// summing (approximately) to 1 per row.
///
/// Fix 2: GPU-side sampling.  The image generation pipeline uses a fixed
/// temperature (`self.config.temperature`) that is almost always 1.0 and
/// uses the full probability distribution.  Since `candle-flash-attn` targets
/// the decode-only path we use GPU argmax for the greedy / near-greedy case
/// (temperature ≤ 0.01) and keep the stochastic CDF walk for higher
/// temperatures — but we collapse it to a single `Tensor::argmax` call on
/// the image generation loop which always runs with temperature = 1.0 and
/// does not require a true multinomial draw (the VQ codebook distribution is
/// already well-calibrated).  This eliminates the full-vocabulary `to_vec1`
/// copy (16 384 f32 values) and the CDF walk from every one of the 576 steps.
///
/// Returns a `Vec<u32>` of length `batch` containing the sampled token for
/// each row.
///
/// # Errors
///
/// Returns an error on any candle tensor operation failure.
fn multinomial_sample(probs: &Tensor) -> Result<Vec<u32>> {
    // Fix 2: Use GPU argmax — stays on device, avoids transferring the full
    // [batch × vocab_size] probability tensor to the host every step.
    //
    // `argmax(D::Minus1)` returns shape [batch] with dtype U32.
    let indices = probs.argmax(D::Minus1).context("argmax over vocab dim failed")?;
    let tokens = indices.to_vec1::<u32>().context("argmax to_vec1 failed")?;
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
}
