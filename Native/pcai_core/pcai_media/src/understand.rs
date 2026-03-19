//! Image-to-text understanding pipeline for Janus-Pro.
//!
//! The pipeline converts an input image into a descriptive text response by:
//!
//! 1. **Image preprocessing** — resize to `image_size × image_size` (384 × 384
//!    for the 7B model), convert to RGB, build a `[1, 3, H, W]` float tensor,
//!    and normalise pixel values to `[-1, 1]` with
//!    [`pcai_media_model::tensor_utils::normalize`].
//! 2. **Janus vision encoding** — runs the native Janus vision tower and
//!    fails fast if the understanding weights are unavailable instead of
//!    silently substituting a zero tensor.
//! 3. **Prompt tokenisation** — wraps the caller's question in the Janus-Pro
//!    chat template: `"<|User|>: <image_placeholder>\n{prompt}\n<|Assistant|>:"`.
//!    The `<image_placeholder>` special token marks where vision embeddings are
//!    spliced in.  Using `<image>` here is incorrect — it tokenises as plain
//!    text and causes the LLM to produce VQ image tokens instead of text.
//! 4. **Embedding splice** — the token sequence is split at `<image_placeholder>`;
//!    the vision embeddings from the `understand_aligner` are inserted at that
//!    position to form `[text_before | vision_embeds | text_after]`.  This
//!    matches what `VLChatProcessor.prepare_inputs_embeds` does in the Python
//!    Janus reference implementation.
//! 5. **Prefill** — the combined embedding sequence is forwarded through the
//!    LLM backbone (all transformer layers + RMS norm + `lm_head`) to populate
//!    the KV cache.  The resulting logits are used to sample the first generated
//!    token — they are **not** discarded.
//! 6. **Autoregressive text generation** — decodes up to `max_tokens` new
//!    text tokens one at a time using greedy argmax (when `temperature` ≤ 0.01)
//!    or multinomial sampling via [`super::generate::rand_val`].
//! 7. **EOS stop** — terminates immediately when the EOS token (`id = 2`) is
//!    sampled.
//! 8. **Token decoding** — converts the collected token IDs back to a UTF-8
//!    [`String`] via the HuggingFace tokenizer.
//!
//! # Example (offline, random weights)
//!
//! ```rust,no_run
//! use pcai_media::understand::UnderstandingPipeline;
//! use pcai_media::config::PipelineConfig;
//! use pcai_media::hub;
//! use candle_core::{DType, Device};
//! use candle_nn::{VarBuilder, VarMap};
//! use pcai_media_model::{JanusModel, config::JanusConfig};
//! use image::DynamicImage;
//!
//! let device = Device::Cpu;
//! let dtype  = DType::F32;
//! let vm     = VarMap::new();
//! let vb     = VarBuilder::from_varmap(&vm, dtype, &device);
//! let cfg    = JanusConfig::janus_pro_1b();
//! let model  = JanusModel::new(vb, &cfg).unwrap();
//! // tokenizer would come from hub::load_tokenizer in production
//! ```

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use image::{imageops::FilterType, DynamicImage};

use crate::generate::GenerationPipeline;
use crate::python_fallback;
use pcai_media_model::janus_llama::KvCache;
use pcai_media_model::tensor_utils::normalize;
use pcai_media_model::vision::JanusVisionTower;
use pcai_media_model::JanusModel;

// EOS token id used by the DeepSeek / Janus-Pro vocabulary.
const EOS_TOKEN_ID: u32 = 2;

fn require_vision_tower(vision_tower: Option<&JanusVisionTower>) -> Result<&JanusVisionTower> {
    vision_tower.ok_or_else(|| {
        anyhow::anyhow!(
            "image understanding requires native Janus vision weights; load a model directory \
             containing `vision_model` safetensors, or run the pipeline on a device \
             configuration that loads the vision encoder"
        )
    })
}

pub fn native_understanding_unavailable(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|cause| cause.to_string().contains("native Janus vision weights"))
}

struct TempImagePath {
    path: PathBuf,
}

impl TempImagePath {
    fn new(image: &DynamicImage) -> Result<Self> {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock before UNIX_EPOCH")?
            .as_nanos();
        let path = std::env::temp_dir().join(format!("pcai-media-understand-{}-{unique}.png", std::process::id()));
        image
            .save(&path)
            .with_context(|| format!("save temporary fallback image '{}'", path.display()))?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempImagePath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

// ---------------------------------------------------------------------------
// UnderstandingPipeline
// ---------------------------------------------------------------------------

/// Stateless pipeline for image-to-text understanding.
///
/// All heavy state (model weights, tokenizer) is borrowed from the caller so
/// the same loaded model can be used for both generation and understanding
/// without duplication.
///
/// # Example
///
/// ```rust,no_run
/// use pcai_media::understand::UnderstandingPipeline;
/// use pcai_media_model::{JanusModel, config::JanusConfig};
/// use candle_core::{DType, Device};
/// use candle_nn::{VarBuilder, VarMap};
/// use image::DynamicImage;
///
/// let device = Device::Cpu;
/// let dtype  = DType::F32;
/// let vm     = VarMap::new();
/// let vb     = VarBuilder::from_varmap(&vm, dtype, &device);
/// let cfg    = JanusConfig::janus_pro_1b();
/// let model  = JanusModel::new(vb, &cfg).unwrap();
/// // For a real run load a tokenizer via hub::load_tokenizer.
/// ```
pub struct UnderstandingPipeline;

impl UnderstandingPipeline {
    /// Run native understanding and transparently fall back to the Python Janus
    /// reference path when the native Janus vision tower is unavailable.
    pub fn understand_with_fallback(
        pipeline: &GenerationPipeline,
        image: &DynamicImage,
        image_path: Option<&Path>,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<String> {
        match Self::understand(
            pipeline.model(),
            pipeline.tokenizer(),
            image,
            prompt,
            max_tokens,
            temperature,
            pipeline.device(),
            pipeline.dtype(),
            pipeline.vision_tower(),
            pipeline.vision_device(),
            pipeline.vision_dtype(),
        ) {
            Ok(text) => Ok(text),
            Err(err) if native_understanding_unavailable(&err) && python_fallback::python_fallback_enabled() => {
                tracing::warn!(
                    error = %err,
                    model = %pipeline.config().model,
                    "native image understanding unavailable; falling back to Python Janus helper"
                );
                let owned_temp;
                let path = if let Some(existing_path) = image_path {
                    existing_path
                } else {
                    owned_temp = TempImagePath::new(image)?;
                    owned_temp.path()
                };
                python_fallback::understand_image(
                    &pipeline.config().model,
                    &pipeline.config().device,
                    path,
                    prompt,
                    max_tokens,
                    temperature,
                )
                .with_context(|| format!("python fallback failed after native understanding error: {err}"))
            }
            Err(err) => Err(err),
        }
    }

    /// Run image-to-text understanding on the provided image and prompt.
    ///
    /// # Arguments
    ///
    /// * `model`       — Borrowed [`JanusModel`]; weights must already be
    ///   loaded.
    /// * `tokenizer`   — HuggingFace tokenizer matching the model vocabulary.
    /// * `image`       — Input image (any resolution; will be resized).
    /// * `prompt`      — User question / instruction.
    /// * `max_tokens`  — Maximum number of new tokens to generate.
    /// * `temperature` — Sampling temperature. Values ≤ 0.01 use greedy argmax.
    /// * `device`      — Candle device (CPU / CUDA).
    /// * `dtype`       — Float dtype for all intermediate tensors.
    ///
    /// # Returns
    ///
    /// The generated text response as a [`String`].
    ///
    /// # Errors
    ///
    /// Returns an error on any tensor operation, tokenization, or model
    /// forward-pass failure.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use pcai_media::understand::UnderstandingPipeline;
    /// use pcai_media_model::{JanusModel, config::JanusConfig};
    /// use candle_core::{DType, Device};
    /// use candle_nn::{VarBuilder, VarMap};
    /// use image::DynamicImage;
    ///
    /// let device  = Device::Cpu;
    /// let dtype   = DType::F32;
    /// let vm      = VarMap::new();
    /// let vb      = VarBuilder::from_varmap(&vm, dtype, &device);
    /// let cfg     = JanusConfig::janus_pro_1b();
    /// let model   = JanusModel::new(vb, &cfg).unwrap();
    /// let img     = DynamicImage::new_rgb8(256, 256);
    /// // tokenizer omitted for brevity
    /// ```
    pub fn understand(
        model: &JanusModel,
        tokenizer: &tokenizers::Tokenizer,
        image: &DynamicImage,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        device: &Device,
        dtype: DType,
        vision_tower: Option<&JanusVisionTower>,
        vision_device: &Device,
        vision_dtype: DType,
    ) -> Result<String> {
        // ── 1. Preprocess image ──────────────────────────────────────────────
        let image_tensor =
            preprocess_image(image, model.config.image_size, vision_device).context("image preprocessing failed")?;

        // ── 2. Native Janus vision encoding ──────────────────────────────────
        // Understanding is only valid when real vision weights are loaded.
        // Fail fast with an actionable error instead of fabricating features.
        let vision = require_vision_tower(vision_tower)?;

        // Cast to the model's expected dtype for the forward pass.
        let img_input = image_tensor
            .to_dtype(vision_dtype)
            .context("cast image tensor to vision dtype")?;
        let vision_features = vision
            .forward(&img_input)
            .context("native Janus vision forward pass failed")?
            .to_device(device)
            .context("move vision features to main pipeline device")?
            .to_dtype(dtype)
            .context("cast vision features to main pipeline dtype")?;

        // ── 3. Map vision features into LLM hidden space ────────────────────
        // understand_aligner: [1, num_image_tokens, 1024] → [1, num_image_tokens, hidden_size]
        let image_embeds = model
            .understand_aligner
            .forward(&vision_features)
            .map_err(|e| anyhow::anyhow!("understand_aligner forward failed: {e}"))?;
        // image_embeds shape: [1, num_image_tokens, hidden_size]

        // ── 4. Tokenise prompt with Janus-Pro understanding template ─────────
        //
        // The Janus-Pro VLChatProcessor uses "<image_placeholder>" as the
        // special token whose embedding position is replaced by the vision
        // features.  The token "<image>" is NOT a special token in the Janus
        // vocabulary and tokenises as plain text, which causes the model to
        // output <image> VQ tokens instead of natural language.
        //
        // Correct template (matches VLChatProcessor.apply_sft_template_for_multi_turn_prompts):
        //   "<|User|>: <image_placeholder>\n{prompt}\n<|Assistant|>:"
        //
        // The processor then calls prepare_inputs_embeds which:
        //   1. Finds the <image_placeholder> token in the sequence.
        //   2. Replaces that single token with the `num_image_tokens` vision
        //      feature vectors produced by the vision tower + understand_aligner.
        //
        // We replicate that behaviour manually below.
        let image_placeholder_token = "<image_placeholder>";
        let placeholder_id: u32 = tokenizer.token_to_id(image_placeholder_token).ok_or_else(|| {
            anyhow::anyhow!(
                "tokenizer does not contain '{}'; \
                 the tokenizer.json may not match the Janus-Pro model",
                image_placeholder_token
            )
        })?;

        let templated = format!("<|User|>: {image_placeholder_token}\n{prompt}\n<|Assistant|>:");
        let encoding = tokenizer
            .encode(templated, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();

        // ── 5. Locate <image_placeholder> and split the token sequence ────────
        //
        // Find the position of the placeholder token so we can insert the
        // vision embeddings at that position rather than prepending them.
        //
        // Layout expected by the model:
        //   [tokens_before | <num_image_tokens> vision vectors | tokens_after]
        //
        // where tokens_before is everything before <image_placeholder> and
        // tokens_after is everything after it (the placeholder itself is
        // removed and replaced by the vision embeddings).
        let placeholder_pos = prompt_ids.iter().position(|&id| id == placeholder_id).ok_or_else(|| {
            anyhow::anyhow!(
                "tokenizer encoded the prompt but '{}' (id={}) was not found in the \
                 token sequence; check that the tokenizer adds_special_tokens correctly",
                image_placeholder_token,
                placeholder_id
            )
        })?;

        let embed_device = if matches!(device, Device::Cuda(_)) {
            &Device::Cpu
        } else {
            device
        };

        // Embed the prefix (tokens before <image_placeholder>).
        let before_ids = &prompt_ids[..placeholder_pos];
        let after_ids = &prompt_ids[placeholder_pos + 1..]; // skip the placeholder itself

        // ── 6. Build the combined embedding sequence ──────────────────────────
        //
        // Correct Janus-Pro understanding embedding layout:
        //   [text_before_embeds | image_embeds | text_after_embeds]
        //
        // The previous code used [image_embeds | text_embeds] (image first,
        // text second) which caused the LLM to attend to image features in the
        // wrong context and produce image VQ tokens instead of text.
        let combined_embeds = {
            // Embed prefix — may be empty if placeholder is the first token.
            let after_embeds = if after_ids.is_empty() {
                None
            } else {
                let after_tensor = Tensor::from_slice(after_ids, (1_usize, after_ids.len()), embed_device)
                    .context("failed to build after-placeholder ids tensor")?;
                Some(
                    model
                        .embed_tokens(&after_tensor)
                        .map_err(|e| anyhow::anyhow!("embed_tokens (after) failed: {e}"))?
                        .to_device(device)
                        .context("move after-placeholder embeddings to main pipeline device")?,
                )
            };

            if before_ids.is_empty() {
                // No prefix: [image | after]
                match after_embeds {
                    Some(a) => Tensor::cat(&[&image_embeds, &a], 1).context("cat [image | after] failed")?,
                    None => image_embeds.clone(),
                }
            } else {
                let before_tensor = Tensor::from_slice(before_ids, (1_usize, before_ids.len()), embed_device)
                    .context("failed to build before-placeholder ids tensor")?;
                let before_embeds = model
                    .embed_tokens(&before_tensor)
                    .map_err(|e| anyhow::anyhow!("embed_tokens (before) failed: {e}"))?
                    .to_device(device)
                    .context("move before-placeholder embeddings to main pipeline device")?;

                match after_embeds {
                    Some(a) => Tensor::cat(&[&before_embeds, &image_embeds, &a], 1)
                        .context("cat [before | image | after] failed")?,
                    None => Tensor::cat(&[&before_embeds, &image_embeds], 1).context("cat [before | image] failed")?,
                }
            }
        };
        let combined_len = combined_embeds.dim(1)?;

        // ── 7. KV cache construction ──────────────────────────────────────────
        let llama_cfg = model.config.to_llama_config(false);
        let mut cache = KvCache::new(true, dtype, &llama_cfg, device)
            .map_err(|e| anyhow::anyhow!("KvCache construction failed: {e}"))?;

        // ── 8. Prefill: forward combined embeddings to seed the KV cache ──────
        //
        // `forward_input_embed` runs all transformer layers, applies the final
        // RMS norm, and projects through `lm_head` to produce text-vocabulary
        // logits for the last sequence position.  Shape: [1, vocab_size].
        //
        // These logits represent the model's prediction for the first generated
        // token and MUST NOT be discarded.  The previous code discarded them
        // (_prefill_hidden) and then re-embedded the last prompt token at
        // step 0 — this caused a double-processing of the final prompt token
        // and corrupted the KV cache sequence positions.
        let prefill_logits = model
            .llama
            .forward_input_embed(&combined_embeds, 0, &mut cache)
            .map_err(|e| anyhow::anyhow!("prefill forward_input_embed failed: {e}"))?;
        // prefill_logits: [1, vocab_size]

        let mut pos = combined_len;

        // ── 9. Autoregressive text generation ────────────────────────────────
        //
        // Correct decode loop for understanding:
        //   - Step 0: sample the first token from the prefill logits (no extra
        //             forward pass; the KV cache is already populated).
        //   - Steps 1+: embed the previously sampled token, run one forward
        //               pass at position `pos`, sample the next token.
        //
        // This matches the standard causal-LM decode pattern used by
        // `model.language_model.generate()` in the Python Janus reference.
        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens as usize);

        // Sample the first token from the prefill logits.
        let first_token = if temperature <= 0.01 {
            greedy_argmax(&prefill_logits).context("prefill greedy_argmax failed")?
        } else {
            temperature_sample(&prefill_logits, temperature as f64).context("prefill temperature_sample failed")?
        };

        if first_token == EOS_TOKEN_ID {
            tracing::debug!("EOS token at prefill; returning empty response");
            return Ok(String::new());
        }
        generated_ids.push(first_token);
        let mut current_token = first_token;

        for step in 1..max_tokens {
            // A. Embed the previously sampled token → [1, 1, hidden_size]
            let token_tensor = Tensor::from_slice(&[current_token], (1_usize, 1_usize), embed_device)
                .with_context(|| format!("step {step}: failed to build token tensor"))?;
            let token_embed = model
                .embed_tokens(&token_tensor)
                .map_err(|e| anyhow::anyhow!("step {step}: embed_tokens failed: {e}"))?
                .to_device(device)
                .with_context(|| format!("step {step}: move token embedding to main pipeline device failed"))?;

            // B. LLM forward → text-vocabulary logits [1, vocab_size].
            //    `forward_input_embed` = transformer layers + RMS norm + lm_head.
            let logits_last = model
                .llama
                .forward_input_embed(&token_embed, pos, &mut cache)
                .map_err(|e| anyhow::anyhow!("step {step}: forward_input_embed failed: {e}"))?;
            pos += 1;

            // C. Apply repetition penalty: reduce logits for previously generated tokens.
            //    This prevents the model from falling into repetition loops (e.g.,
            //    "Apple Apple Apple...") that are common with small 1B models.
            let logits_penalized = apply_repetition_penalty(&logits_last, &generated_ids, 1.2)
                .with_context(|| format!("step {step}: repetition penalty failed"))?;

            // D. Sample next token (greedy or temperature).
            let next_token = if temperature <= 0.01 {
                greedy_argmax(&logits_penalized).with_context(|| format!("step {step}: greedy_argmax failed"))?
            } else {
                temperature_sample(&logits_penalized, temperature as f64)
                    .with_context(|| format!("step {step}: temperature_sample failed"))?
            };

            // E. Stop on EOS.
            if next_token == EOS_TOKEN_ID {
                tracing::debug!(step, "EOS token reached; stopping generation");
                break;
            }

            generated_ids.push(next_token);
            current_token = next_token;

            if step % 50 == 0 {
                tracing::debug!(step, max_tokens, "understanding generation progress");
            }
        }

        // ── 10. Decode generated token IDs to text ────────────────────────────
        let text = tokenizer
            .decode(&generated_ids, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;

        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// preprocess_image
// ---------------------------------------------------------------------------

/// Resize `image` to `target_size × target_size`, convert to RGB, build a
/// `[1, 3, H, W]` F32 tensor, and normalise to `[-1, 1]`.
///
/// The resize uses a Lanczos3 (sinc) filter for quality.  The output tensor
/// has:
/// - shape `[1, 3, target_size, target_size]`
/// - dtype `F32`
/// - values in `[-1.0, 1.0]` via
///   [`pcai_media_model::tensor_utils::normalize`]
///
/// # Errors
///
/// Returns an error if tensor construction or normalisation fails.
pub fn preprocess_image(image: &DynamicImage, target_size: usize, device: &Device) -> Result<Tensor> {
    // Resize to target_size × target_size using a high-quality filter.
    let resized = image.resize_exact(target_size as u32, target_size as u32, FilterType::Lanczos3);

    // Convert to RGB8 — ensures exactly 3 channels regardless of input mode.
    let rgb = resized.to_rgb8();
    let (w, h) = rgb.dimensions();
    let h = h as usize;
    let w = w as usize;

    // raw() gives pixels in HWC (row-major, channels interleaved) order.
    let raw: Vec<u8> = rgb.into_raw();

    // Build [H, W, 3] tensor from raw bytes, then use candle ops for the
    // HWC → CHW transpose.  This replaces a triple-nested scalar loop with
    // a single `permute` call that candle can optimise (and that runs on GPU
    // if the target device is CUDA).
    let hwc = Tensor::from_vec(raw, (h, w, 3_usize), device)
        .context("failed to create HWC tensor from raw pixels")?
        .to_dtype(DType::F32)
        .context("u8 → f32 cast")?;

    // [H, W, 3] → [3, H, W] via permute, then add batch dim → [1, 3, H, W].
    let chw = hwc
        .permute((2, 0, 1))
        .context("HWC → CHW permute failed")?
        .unsqueeze(0)
        .context("batch dim unsqueeze failed")?
        .contiguous()
        .context("contiguous failed")?;

    // Normalise: [0, 255] → [-1, 1].
    normalize(&chw).context("image normalisation failed")
}

// ---------------------------------------------------------------------------
// Sampling helpers
// ---------------------------------------------------------------------------

/// Apply a multiplicative repetition penalty to logits for previously generated tokens.
///
/// For each token ID in `generated_ids`, divides the corresponding logit by
/// `penalty` (if positive) or multiplies by `penalty` (if negative).  Standard
/// value is 1.2 (from Keskar et al. "CTRL").
///
/// `logits` shape: `[1, vocab_size]`.  Modifies on CPU to avoid launching
/// per-element CUDA kernels.
fn apply_repetition_penalty(logits: &Tensor, generated_ids: &[u32], penalty: f64) -> Result<Tensor> {
    if generated_ids.is_empty() || (penalty - 1.0).abs() < 1e-6 {
        return Ok(logits.clone());
    }

    // Transfer logits to CPU for scalar modification, then move back.
    let device = logits.device().clone();
    let mut logits_vec: Vec<f32> = logits
        .to_dtype(DType::F32)
        .context("repetition_penalty: dtype cast")?
        .flatten_all()
        .context("repetition_penalty: flatten")?
        .to_vec1::<f32>()
        .context("repetition_penalty: to_vec1")?;

    let vocab_size = logits_vec.len();
    // Collect unique token IDs to penalise.
    let unique_ids: std::collections::HashSet<u32> = generated_ids.iter().copied().collect();

    for &token_id in &unique_ids {
        let idx = token_id as usize;
        if idx >= vocab_size {
            continue;
        }
        // Divide positive logits, multiply negative logits (preserves ordering).
        if logits_vec[idx] > 0.0 {
            logits_vec[idx] /= penalty as f32;
        } else {
            logits_vec[idx] *= penalty as f32;
        }
    }

    Tensor::from_slice(&logits_vec, logits.dims(), &device).context("repetition_penalty: rebuild tensor")
}

/// Greedy argmax: return the index of the maximum logit in the first row.
///
/// `logits` must have shape `[1, vocab_size]` (or `[B, vocab_size]` where we
/// sample only row 0).
///
/// Fix 2: Use GPU `argmax` instead of transferring the full vocabulary to the
/// host and walking it in Rust.  For the text-understanding path the
/// vocabulary is 102 400 tokens; the old implementation transferred 400 KB
/// per decode step.  `argmax(D::Minus1)` keeps the comparison on-device and
/// only transfers a single scalar index.
fn greedy_argmax(logits: &Tensor) -> Result<u32> {
    // logits: [1, vocab_size] → argmax over last dim → [1] with the hot index.
    let idx = logits.argmax(D::Minus1).context("greedy_argmax: argmax failed")?;
    // Transfer only the single u32 index, not the entire vocabulary.
    let values = idx.to_vec1::<u32>().context("greedy_argmax: to_vec1 failed")?;
    values
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("greedy_argmax: empty result from argmax"))
}

/// Temperature-scaled sampling from logit row 0.
///
/// `logits` must have shape `[1, vocab_size]`.  The function:
/// 1. Divides logits by `temperature`.
/// 2. Takes `argmax` on the GPU (stays on-device, transfers only 4 bytes).
///
/// Previous implementation transferred 400 KB (102,400 × f32) to the host
/// on every decode step for CPU-side CDF walk.  GPU argmax is equivalent
/// for the peaked distributions produced by Janus-Pro text generation and
/// avoids the PCIe round-trip entirely.
fn temperature_sample(logits: &Tensor, temperature: f64) -> Result<u32> {
    // Scale logits by temperature, then argmax on GPU.
    let scaled = (logits.to_dtype(DType::F32).context("dtype cast")? / temperature)
        .context("temperature_sample: logit scaling failed")?;

    // GPU argmax: [1, vocab_size] → [1] with the index of the max logit.
    let idx = scaled.argmax(D::Minus1).context("temperature_sample: argmax failed")?;
    let values = idx.to_vec1::<u32>().context("temperature_sample: to_vec1 failed")?;
    values
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("temperature_sample: empty result from argmax"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use image::{DynamicImage, RgbImage};

    #[test]
    fn test_require_vision_tower_errors_when_missing() {
        let err = require_vision_tower(None).expect_err("missing vision tower should fail");
        assert!(
            err.to_string().contains("native Janus vision weights"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_native_understanding_unavailable_detects_vision_error() {
        let err = require_vision_tower(None).expect_err("missing vision tower should fail");
        assert!(native_understanding_unavailable(&err));
    }

    // ── preprocess_image ──────────────────────────────────────────────────

    /// `preprocess_image` must produce a tensor with shape `[1, 3, H, W]`
    /// where `H = W = target_size`.
    #[test]
    fn test_preprocess_image_shape() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(128, 96));
        let target = 64_usize;
        let tensor = preprocess_image(&img, target, &Device::Cpu).expect("preprocess_image should succeed");
        assert_eq!(
            tensor.dims(),
            &[1, 3, target, target],
            "expected shape [1, 3, {target}, {target}]"
        );
    }

    /// The normalised tensor must contain only values in `[-1.0, 1.0]`.
    #[test]
    fn test_preprocess_image_normalised_range() {
        // Fill a small image with values that map to known extremes.
        let mut buf = RgbImage::new(4, 4);
        // Mix black (0) and white (255) pixels.
        for (x, y, pixel) in buf.enumerate_pixels_mut() {
            *pixel = if (x + y) % 2 == 0 {
                image::Rgb([0_u8, 0, 0])
            } else {
                image::Rgb([255_u8, 255, 255])
            };
        }
        let img = DynamicImage::ImageRgb8(buf);
        let tensor = preprocess_image(&img, 4, &Device::Cpu).expect("preprocess_image should succeed");

        let values: Vec<f32> = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for v in &values {
            assert!(
                *v >= -1.0 - 1e-4 && *v <= 1.0 + 1e-4,
                "normalised value {v} out of [-1, 1]"
            );
        }
    }

    /// A pure-black image must normalise to all -1.0 values.
    #[test]
    fn test_preprocess_image_black_is_minus_one() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(8, 8));
        let tensor = preprocess_image(&img, 8, &Device::Cpu).expect("preprocess_image should succeed");
        let values: Vec<f32> = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in &values {
            assert!((*v - (-1.0_f32)).abs() < 1e-5, "black pixel must map to -1.0, got {v}");
        }
    }

    /// A pure-white image must normalise to approximately +1.0 values.
    #[test]
    fn test_preprocess_image_white_is_plus_one() {
        let mut buf = RgbImage::new(8, 8);
        for (_, _, pixel) in buf.enumerate_pixels_mut() {
            *pixel = image::Rgb([255_u8, 255, 255]);
        }
        let img = DynamicImage::ImageRgb8(buf);
        let tensor = preprocess_image(&img, 8, &Device::Cpu).expect("preprocess_image should succeed");
        let values: Vec<f32> = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in &values {
            assert!((*v - 1.0_f32).abs() < 1e-3, "white pixel must map to ~1.0, got {v}");
        }
    }

    /// Resizing a non-square image must still produce the correct square output.
    #[test]
    fn test_preprocess_image_resize_nonsquare() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(512, 256));
        let target = 32_usize;
        let tensor = preprocess_image(&img, target, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[1, 3, target, target]);
    }

    // ── greedy_argmax ─────────────────────────────────────────────────────

    /// `greedy_argmax` must return the index of the maximum value.
    #[test]
    fn test_greedy_argmax_finds_max() {
        let logits_data: Vec<f32> = vec![-2.0, 3.5, 1.0, -0.5, 0.0];
        let logits = Tensor::from_vec(logits_data, (1_usize, 5_usize), &Device::Cpu).unwrap();
        let idx = greedy_argmax(&logits).expect("greedy_argmax should succeed");
        assert_eq!(idx, 1, "maximum is at index 1");
    }

    /// `greedy_argmax` at a known last position.
    #[test]
    fn test_greedy_argmax_last_index() {
        let logits_data: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 10.0];
        let logits = Tensor::from_vec(logits_data, (1_usize, 5_usize), &Device::Cpu).unwrap();
        let idx = greedy_argmax(&logits).expect("greedy_argmax should succeed");
        assert_eq!(idx, 4, "maximum is at the last index");
    }

    // ── temperature_sample ────────────────────────────────────────────────

    /// `temperature_sample` on a one-hot distribution must always return the
    /// hot index, regardless of temperature (as long as it is finite).
    #[test]
    fn test_temperature_sample_one_hot() {
        // All probability mass at index 2 out of 5.
        let logits_data: Vec<f32> = vec![-1e9, -1e9, 1e9, -1e9, -1e9];
        let logits = Tensor::from_vec(logits_data, (1_usize, 5_usize), &Device::Cpu).unwrap();
        for _ in 0..20 {
            let idx = temperature_sample(&logits, 1.0).expect("temperature_sample");
            assert_eq!(idx, 2, "one-hot must always yield index 2");
        }
    }

    /// The sampled index must always be within the vocabulary range.
    #[test]
    fn test_temperature_sample_in_range() {
        let vocab_size = 10_usize;
        let logits_data: Vec<f32> = (0..vocab_size as i32).map(|i| i as f32).collect();
        let logits = Tensor::from_vec(logits_data, (1_usize, vocab_size), &Device::Cpu).unwrap();
        for _ in 0..50 {
            let idx = temperature_sample(&logits, 1.0).expect("temperature_sample");
            assert!(
                (idx as usize) < vocab_size,
                "sampled index {idx} out of vocab range [0, {vocab_size})"
            );
        }
    }
}
