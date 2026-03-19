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
//!    chat template: `"<|User|>: <image>\n{prompt}\n<|Assistant|>:"`.
//! 4. **Text-embedding prefill** — embeds the token IDs via
//!    [`JanusModel::embed_tokens`] and forwards the full prompt through the
//!    LLM backbone to seed the KV cache.
//! 5. **Autoregressive text generation** — decodes up to `max_tokens` new
//!    text tokens one at a time using greedy argmax (when `temperature` ≤ 0.01)
//!    or multinomial sampling via [`super::generate::rand_val`].
//! 6. **EOS stop** — terminates immediately when the EOS token (`id = 2`) is
//!    sampled.
//! 7. **Token decoding** — converts the collected token IDs back to a UTF-8
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
        // Template: "<|User|>: <image>\n{prompt}\n<|Assistant|>:"
        let templated = format!("<|User|>: <image>\n{prompt}\n<|Assistant|>:");
        let encoding = tokenizer
            .encode(templated, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        let prompt_ids: &[u32] = encoding.get_ids();
        let seq_len = prompt_ids.len();

        // ── 5. Text embeddings via LLM embedding table ────────────────────────
        let embed_device = if matches!(device, Device::Cuda(_)) {
            &Device::Cpu
        } else {
            device
        };
        let prompt_id_tensor = Tensor::from_slice(prompt_ids, (1_usize, seq_len), embed_device)
            .context("failed to build prompt_ids tensor")?;
        let text_embeds = model
            .embed_tokens(&prompt_id_tensor)
            .map_err(|e| anyhow::anyhow!("embed_tokens failed: {e}"))?
            .to_device(device)
            .context("move prompt embeddings to main pipeline device")?;
        // text_embeds: [1, seq_len, hidden_size]

        // ── 6. Concatenate image + text embeddings ────────────────────────────
        // Layout: [image tokens | text tokens]
        // shape:  [1, num_image_tokens + seq_len, hidden_size]
        let combined_embeds = Tensor::cat(&[&image_embeds, &text_embeds], 1)
            .context("failed to concatenate image and text embeddings")?;
        let combined_len = combined_embeds.dim(1)?;

        // ── 7. KV cache construction ──────────────────────────────────────────
        let llama_cfg = model.config.to_llama_config(false);
        let mut cache = KvCache::new(true, dtype, &llama_cfg, device)
            .map_err(|e| anyhow::anyhow!("KvCache construction failed: {e}"))?;

        // ── 8. Prefill: forward combined embeddings to seed the KV cache ──────
        let _prefill_hidden = model
            .llama
            .forward_input_embed(&combined_embeds, 0, &mut cache)
            .map_err(|e| anyhow::anyhow!("prefill forward_input_embed failed: {e}"))?;

        let mut pos = combined_len;

        // ── 9. Autoregressive text generation ────────────────────────────────
        let mut generated_ids: Vec<u32> = Vec::with_capacity(max_tokens as usize);

        // Start with the last token of the prompt as the first auto-regressive
        // input so the first generated token conditions on the full context.
        let mut current_token: u32 = *prompt_ids.last().unwrap_or(&0);

        for step in 0..max_tokens {
            // A. Embed the current single token → [1, 1, hidden_size]
            let token_tensor = Tensor::from_slice(&[current_token], (1_usize, 1_usize), embed_device)
                .with_context(|| format!("step {step}: failed to build token tensor"))?;
            let token_embed = model
                .embed_tokens(&token_tensor)
                .map_err(|e| anyhow::anyhow!("step {step}: embed_tokens failed: {e}"))?
                .to_device(device)
                .with_context(|| format!("step {step}: move token embedding to main pipeline device failed"))?;

            // B. LLM forward → logits [1, vocab_size]
            //    forward_input_embed already extracts the last position.
            let logits_last = model
                .llama
                .forward_input_embed(&token_embed, pos, &mut cache)
                .map_err(|e| anyhow::anyhow!("step {step}: forward_input_embed failed: {e}"))?;
            pos += 1;

            // D. Sample next token (greedy or temperature)
            let next_token = if temperature <= 0.01 {
                greedy_argmax(&logits_last).with_context(|| format!("step {step}: greedy_argmax failed"))?
            } else {
                temperature_sample(&logits_last, temperature as f64)
                    .with_context(|| format!("step {step}: temperature_sample failed"))?
            };

            // E. Stop on EOS
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

    // raw() gives pixels in HWC (row-major, channels interleaved) order.
    // We need to build a CHW tensor for the model.
    let raw: Vec<u8> = rgb.into_raw();

    // Convert to f32 first so we can normalise.
    let hwc_f32: Vec<f32> = raw.iter().map(|&v| v as f32).collect();

    // Reshape from HWC → CHW by transposition:
    //   source index: h * W * 3 + w * 3 + c
    //   target index: c * H * W + h * W + w
    let h = h as usize;
    let w = w as usize;
    let mut chw: Vec<f32> = vec![0.0_f32; 3 * h * w];
    for row in 0..h {
        for col in 0..w {
            for c in 0..3usize {
                let src = row * w * 3 + col * 3 + c;
                let dst = c * h * w + row * w + col;
                chw[dst] = hwc_f32[src];
            }
        }
    }

    // Build the [1, 3, H, W] tensor.
    let tensor = Tensor::from_vec(chw, (1_usize, 3_usize, h, w), device)
        .context("failed to create image tensor from raw pixels")?;

    // Normalise: [0, 255] → [-1, 1].
    normalize(&tensor).context("image normalisation failed")
}

// ---------------------------------------------------------------------------
// Sampling helpers
// ---------------------------------------------------------------------------

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

/// Temperature-scaled multinomial sampling from logit row 0.
///
/// `logits` must have shape `[1, vocab_size]`.  The function:
/// 1. Divides logits by `temperature`.
/// 2. Applies softmax.
/// 3. Draws one sample via inverse CDF with [`super::generate::rand_val`].
fn temperature_sample(logits: &Tensor, temperature: f64) -> Result<u32> {
    // Scale and softmax on row 0.
    let row = logits.i(0_usize).context("temperature_sample: failed to index row 0")?;

    let scaled = (row.to_dtype(DType::F32).context("dtype cast")? / temperature)
        .context("temperature_sample: logit scaling failed")?;

    let probs = candle_nn::ops::softmax_last_dim(&scaled.unsqueeze(0).context("temperature_sample: unsqueeze failed")?)
        .context("temperature_sample: softmax failed")?;

    // probs: [1, vocab_size]
    let probs_vec: Vec<f32> = probs
        .flatten_all()
        .context("temperature_sample: flatten failed")?
        .to_vec1::<f32>()
        .context("temperature_sample: to_vec1 failed")?;

    // Inverse CDF sampling.
    let u = super::generate::rand_val();
    let mut cumsum = 0.0_f64;
    let mut sampled = (probs_vec.len() - 1) as u32;
    for (i, &p) in probs_vec.iter().enumerate() {
        cumsum += p as f64;
        if u < cumsum {
            sampled = i as u32;
            break;
        }
    }
    Ok(sampled)
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
