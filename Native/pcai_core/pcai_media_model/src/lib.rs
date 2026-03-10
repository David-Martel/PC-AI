//! Janus-Pro model architecture for the PC_AI media agent.
//!
//! This crate provides:
//! - [`config`]: [`JanusConfig`] with 1B and 7B presets, and conversion to
//!   `candle_transformers` Llama config.
//! - [`vq_vae`]: VQ-VAE decoder and discrete codebook lookup.
//! - [`generation_head`]: [`GenerationHead`] and [`MlpAligner`] for image
//!   generation mode.
//! - [`tensor_utils`]: Image tensor pre/post-processing utilities.
//! - [`JanusModel`]: Unified facade that assembles the full Janus-Pro model.
//!
//! # Architecture Summary
//!
//! ```text
//! input_ids [B, S]
//!   │
//!   ├─ Llama.embed()          → embeddings [B, S, hidden]
//!   │   ╰─ (optionally aligned via gen_aligner / understand_aligner)
//!   │
//!   ├─ Llama.forward_input_embed() → hidden states [B, S, hidden]
//!   │
//!   ├─ GenerationHead         → image logits [B, S, image_vocab]
//!   │
//!   └─ VqCodebook.decode()
//!        ╰─ VqVaeDecoder.decode() → pixels [B, 3, 384, 384]
//! ```
//!
//! # Example (offline smoke-test)
//!
//! ```rust,no_run
//! use candle_core::{Device, DType};
//! use candle_nn::VarMap;
//! use pcai_media_model::{JanusModel, config::JanusConfig};
//!
//! let dev = Device::Cpu;
//! let vm  = VarMap::new();
//! let vb  = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev);
//! let cfg = JanusConfig::janus_pro_7b();
//! let model = JanusModel::new(vb, &cfg).unwrap();
//! ```

pub mod config;
pub mod generation_head;
pub mod janus_llama;
pub mod tensor_utils;
pub mod vq_vae;

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, VarBuilder};

use config::JanusConfig;
use generation_head::{GenerationHead, MlpAligner};
use janus_llama::JanusLlama;
use vq_vae::{VqCodebook, VqVaeConfig, VqVaeDecoder};

// ---------------------------------------------------------------------------
// JanusModel
// ---------------------------------------------------------------------------

/// Unified Janus-Pro model facade.
///
/// Owns all sub-models required for both text-to-image generation and
/// image-to-text understanding:
///
/// | Field | Role |
/// |-------|------|
/// | `llama` | DeepSeek LLM backbone |
/// | `gen_head` | Projects hidden states → image-vocab logits |
/// | `gen_aligner` | Maps image-token embeddings → LLM input space |
/// | `vq_codebook` | Discrete codebook: token IDs → latent vectors |
/// | `vq_decoder` | VQ-GAN decoder: latent grid → RGB pixels |
/// | `understand_aligner` | Maps SigLIP features (1024-dim) → LLM hidden space |
/// | `config` | Cloned [`JanusConfig`] for downstream queries |
///
/// # Construction
///
/// Use [`JanusModel::new`] with a [`VarBuilder`] rooted at the safetensors
/// checkpoint and the matching [`JanusConfig`].  For inference without weights
/// (unit tests), construct a `VarMap`-backed builder which auto-initialises
/// all tensors to zero.
pub struct JanusModel {
    /// LLM backbone (DeepSeek / Llama architecture).
    /// Uses our custom [`JanusLlama`] which exposes pre-lm_head hidden states.
    pub llama: JanusLlama,
    /// Linear projection from hidden states to image vocabulary.
    pub gen_head: GenerationHead,
    /// MLP that maps image-token embeddings into LLM input space.
    pub gen_aligner: MlpAligner,
    /// Image-token embedding table for generation mode.
    /// Maps discrete image token IDs to continuous embeddings before the aligner.
    pub gen_embed: candle_nn::Embedding,
    /// VQ codebook: maps discrete token IDs to continuous latent vectors.
    pub vq_codebook: VqCodebook,
    /// 1×1 conv that maps codebook dim (8) → decoder z_channels (256).
    pub post_quant_conv: Conv2d,
    /// VQ-GAN decoder: maps latent grids to RGB pixel tensors.
    pub vq_decoder: VqVaeDecoder,
    /// MLP that maps SigLIP visual features into the LLM hidden space.
    pub understand_aligner: MlpAligner,
    /// Cloned configuration used during construction.
    pub config: JanusConfig,
}

impl JanusModel {
    /// Constructs the full Janus-Pro model from a [`VarBuilder`].
    ///
    /// # Arguments
    ///
    /// * `vb`     — [`VarBuilder`] rooted at the model checkpoint.
    /// * `config` — [`JanusConfig`] describing model dimensions.
    ///
    /// Sub-model weight paths relative to `vb`:
    ///
    /// | Component | Path |
    /// |-----------|------|
    /// | LLM backbone | `language_model` |
    /// | Generation head | `gen_head` |
    /// | Generation embedding | `gen_embed` |
    /// | Generation aligner | `gen_aligner` |
    /// | VQ codebook | `gen_vision_model.quantize.embedding` |
    /// | Post-quant conv | `gen_vision_model.post_quant_conv` |
    /// | VQ decoder | `gen_vision_model.decoder` |
    /// | Understand aligner | `aligner` |
    ///
    /// # Errors
    ///
    /// Returns a candle error if any weight tensor is missing or misshapen.
    pub fn new(vb: VarBuilder, config: &JanusConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let image_vocab_size = config.image_token_num_tokens;

        // ── LLM backbone ──────────────────────────────────────────────────
        // Uses our custom JanusLlama which exposes pre-lm_head hidden states
        // needed for the image generation head.
        let use_flash_attn = cfg!(feature = "flash-attn");
        let llama_cfg = config.to_llama_config(use_flash_attn);
        let llama =
            JanusLlama::load(vb.pp("language_model"), &llama_cfg)?;

        // ── Generation head ───────────────────────────────────────────────
        // hidden_size → image_vocab_size (no bias).
        let gen_head =
            GenerationHead::new(vb.pp("gen_head"), hidden_size, image_vocab_size)?;

        // ── Generation embedding ──────────────────────────────────────────
        // Image-token embedding table: maps discrete VQ token IDs to
        // vq_embed_dim-dimensional vectors (8 for both 1B and 7B).
        let vq_dim = config.vq_embed_dim;
        let gen_embed = candle_nn::embedding(
            image_vocab_size,
            vq_dim,
            vb.pp("gen_embed"),
        )?;

        // ── Generation aligner ────────────────────────────────────────────
        // Maps vq_embed_dim (8) → hidden_size (2048 or 4096).
        let gen_aligner =
            MlpAligner::new(vb.pp("gen_aligner"), vq_dim, hidden_size)?;

        // ── VQ codebook ───────────────────────────────────────────────────
        // 16 384-token codebook with vq_embed_dim-dimensional latent vectors.
        // Weights live at: gen_vision_model.quantize.embedding.weight
        let vq_codebook = VqCodebook::new(
            vb.pp("gen_vision_model.quantize.embedding"),
            image_vocab_size,
            vq_dim,
        )?;

        // ── Post-quantization conv ──────────────────────────────────────
        // 1×1 conv that maps codebook dim (8) → decoder z_channels (256).
        // Weight: gen_vision_model.post_quant_conv.{weight,bias}
        let decoder_z_channels = VqVaeConfig::default().z_channels; // 256
        let post_quant_conv = candle_nn::conv2d(
            vq_dim,
            decoder_z_channels,
            1,
            Conv2dConfig::default(),
            vb.pp("gen_vision_model.post_quant_conv"),
        )?;

        // ── VQ-VAE decoder ────────────────────────────────────────────────
        // Decoder takes 256-channel spatial grids (after post_quant_conv).
        let vq_decoder = VqVaeDecoder::new(
            vb.pp("gen_vision_model.decoder"),
            &VqVaeConfig::default(),
        )?;

        // ── Understanding aligner ─────────────────────────────────────────
        // SigLIP outputs understand_input_dim-dim vectors; map them to hidden_size.
        let understand_aligner =
            MlpAligner::new(vb.pp("aligner"), config.understand_input_dim, hidden_size)?;

        Ok(Self {
            llama,
            gen_head,
            gen_aligner,
            gen_embed,
            vq_codebook,
            post_quant_conv,
            vq_decoder,
            understand_aligner,
            config: config.clone(),
        })
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /// Runs one forward pass of the LLM backbone.
    ///
    /// Delegates directly to [`llama::Llama::forward`].  Returns logits for
    /// the **last** token position (shape `[B, vocab_size]`).
    ///
    /// # Arguments
    ///
    /// * `input_ids` — integer tensor of shape `[B, S]`.
    /// * `pos`       — absolute position index (used for RoPE and KV-cache).
    /// * `cache`     — mutable KV-cache; pass a fresh [`llama::Cache`] for
    ///   the first step of each sequence.
    ///
    /// # Errors
    ///
    /// Propagates candle tensor errors from the transformer forward pass.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        pos: usize,
        cache: &mut janus_llama::KvCache,
    ) -> Result<Tensor> {
        self.llama.forward(input_ids, pos, cache)
    }

    /// Embeds token IDs into the LLM's hidden space.
    ///
    /// Uses the LLM embedding table (`wte`) — equivalent to calling
    /// `llama::Llama::embed`.
    ///
    /// # Arguments
    ///
    /// * `token_ids` — integer tensor of shape `[B, S]`.
    ///
    /// # Returns
    ///
    /// Float tensor of shape `[B, S, hidden_size]`.
    ///
    /// # Errors
    ///
    /// Propagates candle tensor errors.
    pub fn embed_tokens(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.llama.embed(token_ids)
    }

    /// Projects LLM hidden states to image-vocabulary logits.
    ///
    /// # Arguments
    ///
    /// * `hidden_states` — float tensor of shape `[B, S, hidden_size]`.
    ///
    /// # Returns
    ///
    /// Float tensor of shape `[B, S, image_vocab_size]`.
    ///
    /// # Errors
    ///
    /// Propagates candle tensor errors from the linear projection.
    pub fn project_to_image_vocab(&self, hidden_states: &Tensor) -> Result<Tensor> {
        use candle_core::Module;
        self.gen_head.forward(hidden_states)
    }

    /// Decodes discrete image tokens into an RGB pixel tensor.
    ///
    /// Pipeline:
    /// 1. Codebook lookup → `[B, 576, vq_dim]` continuous latents.
    /// 2. Transpose + reshape → `[B, vq_dim, 24, 24]` spatial grid.
    /// 3. VQ-GAN decoder → `[B, 3, 384, 384]` pixels.
    ///
    /// # Arguments
    ///
    /// * `tokens` — integer tensor of shape `[B, 576]` (24 × 24 grid).
    ///
    /// # Returns
    ///
    /// Float tensor of shape `[B, 3, 384, 384]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the token sequence length is not 576, or on any
    /// candle tensor error.
    pub fn decode_image_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
        // Grid dimensions derived from config.
        let num_img_tokens = self.config.num_image_tokens(); // 576
        let grid_size = (num_img_tokens as f64).sqrt() as usize; // 24

        // 1. Codebook lookup: [B, H*W] → [B, vq_dim, H, W]
        //    VqCodebook::lookup_grid handles the reshape internally.
        let latent = self
            .vq_codebook
            .lookup_grid(tokens, grid_size, grid_size)?;
        // latent shape: [B, 8, 24, 24]

        // 2. Post-quantization conv: [B, 8, 24, 24] → [B, 256, 24, 24]
        let latent = self.post_quant_conv.forward(&latent)?;

        // 3. VQ-VAE decoder: [B, 256, 24, 24] → [B, 3, 384, 384]
        self.vq_decoder.decode(&latent)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarMap;

    fn cpu_vb(vm: &VarMap) -> VarBuilder<'_> {
        VarBuilder::from_varmap(vm, DType::F32, &Device::Cpu)
    }

    /// Construct the full JanusModel from zeroed weights and verify it does
    /// not panic.  Uses the 7B config (hidden=4096) so the Llama backbone
    /// allocates proper embedding tables.
    ///
    /// NOTE: This test is gated with `#[ignore]` because it allocates
    /// ~600 MB of parameter tensors (4096 × 30 layers) on the CPU, which is
    /// acceptable in CI with sufficient RAM but too slow for a quick `cargo
    /// test` run.  Run explicitly with:
    ///
    /// ```text
    /// cargo test -p pcai-media-model -- --ignored test_janus_model_construct
    /// ```
    #[test]
    #[ignore = "allocates ~600 MB; run explicitly with --ignored"]
    fn test_janus_model_construct() {
        let vm = VarMap::new();
        let vb = cpu_vb(&vm);
        let cfg = JanusConfig::janus_pro_7b();

        // Should not panic or return an error.
        let _model = JanusModel::new(vb, &cfg).expect("JanusModel construction failed");
    }

    /// Smoke-test construction using the 1B config to keep allocations smaller.
    ///
    /// Even the 1B config is large (~250 MB) so this is also marked ignored.
    #[test]
    #[ignore = "allocates ~250 MB; run explicitly with --ignored"]
    fn test_janus_model_construct_1b() {
        let vm = VarMap::new();
        let vb = cpu_vb(&vm);
        let cfg = JanusConfig::janus_pro_1b();

        let _model = JanusModel::new(vb, &cfg).expect("JanusModel 1B construction failed");
    }

    /// Verify that the module declarations compile and are re-exported cleanly.
    #[test]
    fn test_module_reexports() {
        // Instantiate types from each public sub-module to confirm the
        // module tree compiles without hidden dependency issues.
        let _cfg = config::JanusConfig::janus_pro_7b();
        let _vq_cfg = vq_vae::VqVaeConfig::default();
        let img_tok = _cfg.num_image_tokens();
        assert_eq!(img_tok, 576);
    }

    /// Verify decode_image_tokens shape contract with a tiny synthetic codebook.
    ///
    /// Uses a small artificial config where `image_size=64`, `patch_size=16`
    /// → num_image_tokens=16 (4×4 grid), and a matching tiny VQ config.
    ///
    /// This test overrides individual components manually because
    /// `JanusModel::new` ties all dimensions together through `JanusConfig`.
    /// We test `VqCodebook::lookup_grid` + `VqVaeDecoder::decode` directly
    /// (the two steps inside `decode_image_tokens`) with matching small dims.
    #[test]
    fn test_decode_image_tokens_shape_small() {
        use vq_vae::{VqCodebook, VqVaeConfig, VqVaeDecoder};

        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = cpu_vb(&vm);

        // 4×4 grid, z_channels=64, minimal decoder.
        let grid_size: usize = 4;
        let num_tokens = grid_size * grid_size; // 16
        let z_channels: usize = 64;
        let vocab_size: usize = 256;

        let cb = VqCodebook::new(vb.pp("cb"), vocab_size, z_channels).unwrap();
        let vq_cfg = VqVaeConfig {
            z_channels,
            base_channels: 64,
            ch_mult: vec![1, 2],
            num_res_blocks: 1,
            out_channels: 3,
        };
        let decoder = VqVaeDecoder::new(vb.pp("dec"), &vq_cfg).unwrap();

        // Tokens: [B=1, 16]
        let tokens =
            Tensor::zeros((1_usize, num_tokens), DType::U32, &dev).unwrap();

        // Lookup: [1, 16] → [1, 64, 4, 4]
        let latent = cb.lookup_grid(&tokens, grid_size, grid_size).unwrap();
        assert_eq!(latent.dims(), &[1, z_channels, grid_size, grid_size]);

        // Decode: [1, 64, 4, 4] → [1, 3, 8, 8]  (one upsample ×2 from 4→8)
        let img = decoder.decode(&latent).unwrap();
        assert_eq!(img.dims(), &[1, 3, 8, 8]);
    }

    /// Verify KvCache construction matches the Llama config API.
    #[test]
    fn test_kv_cache_construction() {
        let dev = Device::Cpu;
        let cfg = JanusConfig::janus_pro_1b();
        let llama_cfg = cfg.to_llama_config(false);

        // KvCache::new should succeed with CPU + F32.
        let _cache = janus_llama::KvCache::new(false, DType::F32, &llama_cfg, &dev)
            .expect("KvCache construction failed");
    }
}
