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
use candle_transformers::models::llama;
use image::{ImageBuffer, Rgb};

use pcai_media_model::{config::JanusConfig, JanusModel};

use crate::config::PipelineConfig;
use crate::hub;

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
}

impl GenerationPipeline {
    /// Load the Janus-Pro pipeline from the configuration.
    ///
    /// Steps performed:
    /// 1. Resolve [`candle_core::Device`] and [`candle_core::DType`] from `config`.
    /// 2. Resolve the model path (local directory or HuggingFace Hub download).
    /// 3. Deserialise [`JanusConfig`] from `config.json` (falls back to 7B defaults).
    /// 4. Build [`JanusModel`] via [`VarMap`] + [`VarBuilder`].
    /// 5. Load weights from safetensors shards into the [`VarMap`].
    /// 6. Load the tokenizer from `tokenizer.json`.
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

        // 2. Load JanusConfig from config.json (with 7B fallback).
        let config_json = model_path.join("config.json");
        let model_config = if config_json.exists() {
            JanusConfig::from_file(&config_json)
                .unwrap_or_else(|err| {
                    tracing::warn!(
                        path = %config_json.display(),
                        error = %err,
                        "failed to parse config.json; using 7B defaults"
                    );
                    JanusConfig::janus_pro_7b()
                })
        } else {
            tracing::warn!("config.json not found; using 7B defaults");
            JanusConfig::janus_pro_7b()
        };

        // 3. Build JanusModel from a VarMap-backed VarBuilder.
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let model = JanusModel::new(vb, &model_config)
            .map_err(|e| anyhow::anyhow!("JanusModel construction failed: {e}"))?;

        // 4. Load safetensors weights into the VarMap.
        let shards = hub::collect_safetensors(&model_path);
        if shards.is_empty() {
            tracing::warn!(
                path = %model_path.display(),
                "no safetensors shards found; model will use random weights"
            );
        } else {
            let loaded = hub::load_weights(&varmap, &shards, dtype, &device)
                .context("failed to load model weights from safetensors")?;
            tracing::info!(
                shards = shards.len(),
                tensors_loaded = loaded,
                "weights loaded"
            );
        }

        // 5. Load tokenizer.
        let tokenizer = hub::load_tokenizer(&model_path)
            .context("failed to load tokenizer")?;

        Ok(Self {
            model,
            tokenizer,
            config,
            model_config,
            device,
            dtype,
        })
    }

    /// Generate an image from a text prompt.
    ///
    /// Runs the 576-step autoregressive Janus-Pro generation loop with
    /// Classifier-Free Guidance (CFG).
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
        let parallel_size = self.config.parallel_size;
        // CFG requires a paired positive/negative batch.
        let batch_size = parallel_size * 2;

        // ── 1. Tokenise ──────────────────────────────────────────────────────
        // Janus-Pro chat template.
        let templated = format!("<|User|>: {prompt}\n<|Assistant|>:");
        let encoding = self
            .tokenizer
            .encode(templated, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        let prompt_ids: &[u32] = encoding.get_ids();
        let seq_len = prompt_ids.len();

        // ── 2. Build input tensor [batch_size, seq_len] ───────────────────────
        // Even indices: conditional (prompt) tokens.
        // Odd indices: unconditional (pad) tokens.
        let pad_id: u32 = self
            .tokenizer
            .token_to_id("<pad>")
            .unwrap_or(0);

        let mut flat_ids: Vec<u32> = Vec::with_capacity(batch_size * seq_len);
        for i in 0..batch_size {
            if i % 2 == 0 {
                // Conditional: real prompt tokens.
                flat_ids.extend_from_slice(prompt_ids);
            } else {
                // Unconditional: all-padding sequence of the same length.
                flat_ids.extend(std::iter::repeat(pad_id).take(seq_len));
            }
        }
        let input_ids = Tensor::from_vec(flat_ids, (batch_size, seq_len), &self.device)
            .context("failed to build input_ids tensor")?;

        // ── 3. Build KV cache ────────────────────────────────────────────────
        let llama_cfg = self.model_config.to_llama_config(false);
        let mut cache = llama::Cache::new(false, self.dtype, &llama_cfg, &self.device)
            .map_err(|e| anyhow::anyhow!("Cache construction failed: {e}"))?;

        // ── 4. Pre-fill: embed the prompt tokens ─────────────────────────────
        // Shape: [batch_size, seq_len, hidden_size]
        let prompt_embeds = self
            .model
            .embed_tokens(&input_ids)
            .map_err(|e| anyhow::anyhow!("embed_tokens failed: {e}"))?;

        // Pre-fill step: forward the full prompt through the LLM backbone to
        // populate the KV cache.  We use `forward_input_embed` on the public
        // `llama` field to obtain hidden states rather than text-vocab logits.
        let _prefill_hidden = self
            .model
            .llama
            .forward_input_embed(&prompt_embeds, 0, &mut cache)
            .map_err(|e| anyhow::anyhow!("prefill forward_input_embed failed: {e}"))?;

        // Track the current position for RoPE and KV-cache indexing.
        let mut pos = seq_len;

        // ── 5. Autoregressive generation loop (576 image tokens) ─────────────
        let num_image_tokens = self.model_config.num_image_tokens(); // 576
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(num_image_tokens);

        // Use the first image token as the initial input to the generation
        // loop.  In practice Janus inserts a special `<image>` token to
        // separate the prompt from the image token sequence; we use a zero
        // token as a neutral starting embedding.
        let mut current_token_ids = {
            let ids = vec![0u32; batch_size];
            Tensor::from_vec(ids, (batch_size, 1_usize), &self.device)
                .context("failed to build initial token tensor")?
        };

        for step in 0..num_image_tokens {
            // A. Embed current tokens → [batch_size, 1, hidden_size]
            let embeds = self
                .model
                .embed_tokens(&current_token_ids)
                .map_err(|e| anyhow::anyhow!("step {step}: embed_tokens failed: {e}"))?;

            // B. LLM backbone: hidden states [batch_size, 1, hidden_size]
            let hidden = self
                .model
                .llama
                .forward_input_embed(&embeds, pos, &mut cache)
                .map_err(|e| anyhow::anyhow!("step {step}: forward_input_embed failed: {e}"))?;
            pos += 1;

            // C. Project to image vocabulary logits [batch_size, 1, image_vocab_size]
            let img_logits_full = self
                .model
                .project_to_image_vocab(&hidden)
                .map_err(|e| anyhow::anyhow!("step {step}: project_to_image_vocab failed: {e}"))?;

            // D. Extract last-position logits [batch_size, image_vocab_size]
            let last_pos = img_logits_full.dim(1)? - 1;
            let img_logits = img_logits_full
                .i((.., last_pos, ..))
                .map_err(|e| anyhow::anyhow!("step {step}: logit slice failed: {e}"))?;

            // E. CFG: split batch into conditional (even rows) and unconditional
            //    (odd rows).  The candle IndexOp trait does not support StepBy
            //    iterators, so we select each pair explicitly and stack.
            //    For parallel_size=1: cond = row 0, uncond = row 1.
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
                // General case: gather even and odd row indices explicitly.
                let cond_rows: Vec<Tensor> = (0..batch_size)
                    .step_by(2)
                    .map(|i| {
                        img_logits
                            .i(i)
                            .and_then(|t| t.unsqueeze(0))
                            .map_err(|e| {
                                anyhow::anyhow!("step {step}: cond row {i}: {e}")
                            })
                    })
                    .collect::<Result<_>>()?;
                let uncond_rows: Vec<Tensor> = (1..batch_size)
                    .step_by(2)
                    .map(|i| {
                        img_logits
                            .i(i)
                            .and_then(|t| t.unsqueeze(0))
                            .map_err(|e| {
                                anyhow::anyhow!("step {step}: uncond row {i}: {e}")
                            })
                    })
                    .collect::<Result<_>>()?;
                let c = Tensor::cat(&cond_rows, 0)
                    .map_err(|e| anyhow::anyhow!("step {step}: cond cat: {e}"))?;
                let u = Tensor::cat(&uncond_rows, 0)
                    .map_err(|e| anyhow::anyhow!("step {step}: uncond cat: {e}"))?;
                (c, u)
            };

            // guided = uncond + guidance_scale * (cond - uncond)
            let scale = self.config.guidance_scale;
            let diff = (cond.clone() - uncond.clone())
                .map_err(|e| anyhow::anyhow!("step {step}: CFG diff failed: {e}"))?;
            let guided = (uncond + (diff * scale)?)
                .map_err(|e| anyhow::anyhow!("step {step}: CFG add failed: {e}"))?;

            // F. Temperature scaling + softmax → probability distribution
            //    [parallel_size, image_vocab_size]
            let temperature = self.config.temperature;
            let scaled = if (temperature - 1.0_f64).abs() > 1e-6 {
                (guided / temperature)
                    .map_err(|e| anyhow::anyhow!("step {step}: temperature scale failed: {e}"))?
            } else {
                guided
            };
            let probs = candle_nn::ops::softmax_last_dim(&scaled)
                .map_err(|e| anyhow::anyhow!("step {step}: softmax failed: {e}"))?;

            // G. Multinomial sampling: one token per parallel image.
            //    We sample from probs [parallel_size, vocab_size] and
            //    collect the sampled index for each parallel sample.
            let next_tokens = multinomial_sample(&probs)
                .with_context(|| format!("step {step}: multinomial sampling failed"))?;

            // For parallel_size=1, store the single sampled token.
            // For parallel_size>1, store the first sample (later: batch decode).
            let first_token = next_tokens[0];
            generated_tokens.push(first_token);

            // H. Prepare next-step input: replicate sampled token across
            //    batch dimension [batch_size, 1].
            let mut next_ids_flat = Vec::with_capacity(batch_size);
            for &tok in &next_tokens {
                // Even (cond) and odd (uncond) get the same token for the
                // next-step embedding — Janus uses shared image tokens.
                next_ids_flat.push(tok);
                next_ids_flat.push(tok);
            }
            current_token_ids =
                Tensor::from_vec(next_ids_flat, (batch_size, 1_usize), &self.device)
                    .context("failed to build next token tensor")?;

            if step % 100 == 0 || step == num_image_tokens - 1 {
                tracing::debug!(step, total = num_image_tokens, "generation progress");
            }
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
    anyhow::ensure!(
        dims.len() == 3,
        "expected 3D tensor [C, H, W], got {}D",
        dims.len()
    );
    let (c, h, w) = (dims[0], dims[1], dims[2]);
    anyhow::ensure!(c == 3, "expected C=3 (RGB), got C={c}");

    // Permute [C, H, W] → [H, W, C] for row-major layout.
    let hwc = tensor
        .permute((1, 2, 0))
        .context("permute [C,H,W] -> [H,W,C] failed")?;

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
/// The implementation uses a hash-based pseudo-random value to avoid the
/// `rand` crate dependency. This is not cryptographically secure but is
/// sufficient for image generation sampling.
///
/// Returns a `Vec<u32>` of length `batch` containing the sampled token for
/// each row.
///
/// # Errors
///
/// Returns an error on any candle tensor operation failure.
fn multinomial_sample(probs: &Tensor) -> Result<Vec<u32>> {
    let (batch, vocab_size) = probs.dims2().context("probs must be 2D [batch, vocab]")?;
    let flat = probs
        .to_dtype(DType::F32)
        .context("cast probs to F32")?
        .flatten_all()
        .context("flatten probs")?
        .to_vec1::<f32>()
        .context("probs to_vec1")?;

    let mut tokens = Vec::with_capacity(batch);
    for b in 0..batch {
        let row = &flat[b * vocab_size..(b + 1) * vocab_size];
        let u = rand_val(); // uniform in [0, 1)
        let mut cumsum = 0.0_f64;
        let mut sampled = (vocab_size - 1) as u32;
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

/// Hash-based pseudo-random float in `[0, 1)`.
///
/// Uses [`std::collections::hash_map::DefaultHasher`] seeded from the
/// current system time and thread ID.  Not suitable for cryptographic use.
fn rand_val() -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut hasher = DefaultHasher::new();
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos()
        .hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    // Mix the counter to avoid identical values in tight loops.
    static COUNTER: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(0);
    COUNTER
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        .hash(&mut hasher);

    let hash = hasher.finish();
    // Map u64 range to [0, 1).
    (hash as f64) / (u64::MAX as f64)
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
        let tensor =
            Tensor::zeros((3_usize, 10_usize), DType::U8, &Device::Cpu).unwrap();
        assert!(tensor_to_image(&tensor).is_err());
    }

    /// `tensor_to_image` must return an error when C != 3.
    #[test]
    fn test_tensor_to_image_wrong_channels() {
        let tensor =
            Tensor::zeros((4_usize, 8_usize, 8_usize), DType::U8, &Device::Cpu).unwrap();
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
        let unique: std::collections::HashSet<u64> =
            vals.iter().map(|&v| v.to_bits()).collect();
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
        let probs =
            Tensor::from_vec(data, (1_usize, 5_usize), &Device::Cpu).unwrap();
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
        let probs =
            Tensor::from_vec(data, (2_usize, 2_usize), &Device::Cpu).unwrap();
        let tokens = multinomial_sample(&probs).unwrap();
        assert_eq!(tokens.len(), 2, "expected one token per batch row");
    }
}
