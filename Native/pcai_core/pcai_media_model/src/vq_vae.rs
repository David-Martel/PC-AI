//! VQ-VAE decoder for Janus-Pro image generation.
//!
//! This module implements the VQ-GAN style decoder that transforms a discrete
//! latent grid `[B, 256, 24, 24]` produced by the language model back into a
//! `[B, 3, 384, 384]` RGB image.
//!
//! # Architecture overview
//!
//! ```text
//! z [B, 256, 24, 24]
//!   │
//!   ├─ conv_in (256 → 512, 3×3)
//!   │
//!   ├─ mid
//!   │    ├─ mid.0  ResBlock(512, 512)
//!   │    ├─ mid.1  AttnBlock(512)
//!   │    └─ mid.2  ResBlock(512, 512)
//!   │
//!   ├─ conv_blocks.0  (Res+Attn)×3(512→512)  + Upsample(512)  → 48×48
//!   ├─ conv_blocks.1  ResBlock×3(512→256)    + Upsample(256)  → 96×96
//!   ├─ conv_blocks.2  ResBlock×3(256→256)    + Upsample(256)  → 192×192
//!   ├─ conv_blocks.3  ResBlock×3(256→128)    + Upsample(128)  → 384×384
//!   └─ conv_blocks.4  ResBlock×3(128→128)    (no upsample)
//!   │
//!   ├─ norm_out  GroupNorm(32, 128)
//!   ├─ SiLU
//!   └─ conv_out (128 → 3, 3×3)
//!
//! Output: [B, 3, 384, 384]
//! ```
//!
//! Weight names in safetensors match the Python attribute tree under
//! `gen_vision_model.decoder.*` (e.g. `conv_in`, `mid.0.*`,
//! `conv_blocks.0.res.0.*`, `conv_blocks.0.attn.0.*`,
//! `conv_blocks.0.upsample.conv.*`, `norm_out.*`, `conv_out.*`).
//!
//! # Example (offline / CPU smoke-test)
//!
//! ```rust,no_run
//! use candle_core::{Device, DType, Tensor};
//! use candle_nn::VarMap;
//! use pcai_media_model::vq_vae::{VqVaeConfig, VqVaeDecoder};
//!
//! let dev = Device::Cpu;
//! let vm  = VarMap::new();
//! let vb  = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev);
//! let cfg = VqVaeConfig::default();
//! let dec = VqVaeDecoder::new(vb, &cfg).unwrap();
//!
//! // Latent grid: batch=1, z_channels=256, 24×24
//! let z   = Tensor::zeros((1, 256, 24, 24), DType::F32, &dev).unwrap();
//! let img = dec.decode(&z).unwrap();   // → [1, 3, 384, 384]
//! assert_eq!(img.dims(), &[1, 3, 384, 384]);
//! ```

use candle_core::{Module, Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, GroupNorm, VarBuilder};

// ---------------------------------------------------------------------------
// VqVaeConfig
// ---------------------------------------------------------------------------

/// Configuration for the VQ-VAE decoder.
///
/// The defaults match the publicly released `deepseek-ai/Janus-Pro-7B` codec.
#[derive(Debug, Clone)]
pub struct VqVaeConfig {
    /// Dimensionality of the latent space (z channels from the codebook).
    pub z_channels: usize,

    /// Base channel width; all stage widths are multiples of this.
    pub base_channels: usize,

    /// Per-stage channel multipliers.  The decoder iterates these **in
    /// reverse**, so the first up-stage (highest index) handles the widest
    /// feature maps.
    pub ch_mult: Vec<usize>,

    /// Number of residual blocks per decoder stage (one extra is added per
    /// stage to match the reference implementation, making it
    /// `num_res_blocks + 1` total per stage).
    pub num_res_blocks: usize,

    /// Number of output channels (3 for RGB).
    pub out_channels: usize,
}

impl Default for VqVaeConfig {
    /// Returns the Janus-Pro-7B VQ-VAE codec defaults:
    /// `z=256, base=128, ch_mult=[1,1,2,2,4], num_res_blocks=2, out=3`.
    fn default() -> Self {
        Self {
            z_channels: 256,
            base_channels: 128,
            ch_mult: vec![1, 1, 2, 2, 4],
            num_res_blocks: 2,
            out_channels: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// ResBlock
// ---------------------------------------------------------------------------

/// Residual block: GroupNorm→SiLU→Conv(3×3)→GroupNorm→SiLU→Conv(3×3) + skip.
///
/// When `in_channels != out_channels` a 1×1 convolution is used for the skip
/// connection, matching the `nin_shortcut` weight in the reference weights.
pub struct ResBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    /// Present only when `in_channels != out_channels`.
    skip_conv: Option<Conv2d>,
}

impl ResBlock {
    /// Constructs a [`ResBlock`] and loads weights from `vb`.
    ///
    /// Weight paths under `vb`: `norm1`, `conv1`, `norm2`, `conv2`,
    /// and optionally `nin_shortcut` for the skip projection.
    pub fn new(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let pad_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };

        let norm1 = candle_nn::group_norm(32, in_channels, 1e-6, vb.pp("norm1"))?;
        let conv1 = candle_nn::conv2d(in_channels, out_channels, 3, pad_cfg, vb.pp("conv1"))?;

        let norm2 = candle_nn::group_norm(32, out_channels, 1e-6, vb.pp("norm2"))?;
        let conv2 = candle_nn::conv2d(out_channels, out_channels, 3, pad_cfg, vb.pp("conv2"))?;

        let skip_conv = if in_channels != out_channels {
            let skip_cfg = Conv2dConfig {
                padding: 0,
                ..Default::default()
            };
            Some(candle_nn::conv2d(
                in_channels,
                out_channels,
                1,
                skip_cfg,
                vb.pp("nin_shortcut"),
            )?)
        } else {
            None
        };

        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            skip_conv,
        })
    }
}

impl Module for ResBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Main branch: norm → silu → conv → norm → silu → conv
        let h = xs.apply(&self.norm1)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = h.apply(&self.conv1)?;

        let h = h.apply(&self.norm2)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = h.apply(&self.conv2)?;

        // Skip branch: identity or 1×1 projection
        let skip = match &self.skip_conv {
            Some(conv) => xs.apply(conv)?,
            None => xs.clone(),
        };

        h + skip
    }
}

// ---------------------------------------------------------------------------
// AttnBlock
// ---------------------------------------------------------------------------

/// Single-head self-attention block: GroupNorm + Q/K/V 1×1 convs + proj_out.
///
/// Spatial dimensions are flattened to a sequence for the dot-product
/// attention, then reshaped back.  The residual connection adds `xs` to the
/// projected output.
///
/// Weight paths under `vb`: `norm`, `q`, `k`, `v`, `proj_out`.
pub struct AttnBlock {
    norm: GroupNorm,
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    proj_out: Conv2d,
}

impl AttnBlock {
    /// Constructs an [`AttnBlock`] loading weights from `vb`.
    pub fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 0,
            ..Default::default()
        };
        Ok(Self {
            norm: candle_nn::group_norm(32, channels, 1e-6, vb.pp("norm"))?,
            q: candle_nn::conv2d(channels, channels, 1, cfg, vb.pp("q"))?,
            k: candle_nn::conv2d(channels, channels, 1, cfg, vb.pp("k"))?,
            v: candle_nn::conv2d(channels, channels, 1, cfg, vb.pp("v"))?,
            proj_out: candle_nn::conv2d(channels, channels, 1, cfg, vb.pp("proj_out"))?,
        })
    }
}

impl Module for AttnBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = xs.dims4()?;
        let xs_norm = xs.apply(&self.norm)?;

        // Project to Q, K, V — shape [B, C, H, W]
        let q = xs_norm.apply(&self.q)?;
        let k = xs_norm.apply(&self.k)?;
        let v = xs_norm.apply(&self.v)?;

        // Flatten spatial dims: [B, C, H, W] → [B, H*W, C]
        let q = q.flatten_from(2)?.transpose(1, 2)?; // [B, HW, C]
        let k = k.flatten_from(2)?.transpose(1, 2)?;
        let v = v.flatten_from(2)?.transpose(1, 2)?;

        // Scaled dot-product attention
        let scale = (c as f64).powf(-0.5);
        // attn_weights: [B, HW, HW]
        let attn_weights = (q.matmul(&k.t()?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;

        // Weighted sum of values: [B, HW, C]
        let out = attn_weights.matmul(&v)?;

        // Restore spatial layout: [B, HW, C] → [B, C, H, W]
        let out = out.transpose(1, 2)?.reshape((b, c, h, w))?;

        // Project and add residual
        let out = out.apply(&self.proj_out)?;
        xs + out
    }
}

// ---------------------------------------------------------------------------
// Upsample
// ---------------------------------------------------------------------------

/// 2× nearest-neighbour upsampling followed by a 3×3 convolution.
///
/// Weight path under `vb`: `conv`.
pub struct Upsample {
    conv: Conv2d,
}

impl Upsample {
    /// Constructs an [`Upsample`] block that preserves the channel count.
    pub fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            conv: candle_nn::conv2d(channels, channels, 3, cfg, vb.pp("conv"))?,
        })
    }
}

impl Module for Upsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.dims4()?;
        // Nearest-neighbour 2× upsample then refine with conv
        let xs = xs.upsample_nearest2d(h * 2, w * 2)?;
        xs.apply(&self.conv)
    }
}

// ---------------------------------------------------------------------------
// UpStage — one level of the decoder pyramid
// ---------------------------------------------------------------------------

/// One decoder stage: a sequence of [`ResBlock`]s (each optionally followed
/// by an [`AttnBlock`]) and an optional [`Upsample`].
///
/// Attention blocks are present only in the first decoder stage
/// (conv_blocks.0), which is the widest (512-channel) feature map.
struct UpStage {
    blocks: Vec<ResBlock>,
    /// Attention blocks interleaved after each ResBlock (only in stage 0).
    attns: Vec<AttnBlock>,
    upsample: Option<Upsample>,
}

impl UpStage {
    fn forward(&self, mut xs: Tensor) -> Result<Tensor> {
        for (i, block) in self.blocks.iter().enumerate() {
            xs = block.forward(&xs)?;
            if let Some(attn) = self.attns.get(i) {
                xs = attn.forward(&xs)?;
            }
        }
        if let Some(up) = &self.upsample {
            xs = up.forward(&xs)?;
        }
        Ok(xs)
    }
}

// ---------------------------------------------------------------------------
// VqVaeDecoder
// ---------------------------------------------------------------------------

/// Full VQ-VAE decoder for Janus-Pro.
///
/// Transforms a continuous latent grid `[B, z_channels, 24, 24]` to an RGB
/// image `[B, 3, 384, 384]`.
///
/// # Weight paths (relative to the `VarBuilder` root)
///
/// | Tensor | Path |
/// |--------|------|
/// | conv_in | `conv_in` |
/// | mid ResBlock 0 | `mid.0.*` |
/// | mid AttnBlock | `mid.1.*` |
/// | mid ResBlock 1 | `mid.2.*` |
/// | stage i, res block j | `conv_blocks.{i}.res.{j}.*` |
/// | stage 0, attn block j | `conv_blocks.0.attn.{j}.*` |
/// | stage i, upsample | `conv_blocks.{i}.upsample.*` |
/// | norm_out | `norm_out` |
/// | conv_out | `conv_out` |
pub struct VqVaeDecoder {
    conv_in: Conv2d,
    mid_block_1: ResBlock,
    mid_attn: AttnBlock,
    mid_block_2: ResBlock,
    up_stages: Vec<UpStage>,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl VqVaeDecoder {
    /// Constructs a [`VqVaeDecoder`] loading all weights via `vb`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if any weight tensor is missing or has an
    /// unexpected shape.
    pub fn new(vb: VarBuilder, cfg: &VqVaeConfig) -> Result<Self> {
        // ── conv_in ────────────────────────────────────────────────────────
        // Map from z_channels to the widest feature-map dimension.
        // With ch_mult = [1,1,2,2,4] and base = 128, top_ch = 128*4 = 512.
        let top_ch = cfg.base_channels * cfg.ch_mult.last().copied().unwrap_or(1);

        let conv_in_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = candle_nn::conv2d(cfg.z_channels, top_ch, 3, conv_in_cfg, vb.pp("conv_in"))?;

        // ── mid block ──────────────────────────────────────────────────────
        // Safetensors keys: mid.0.* (ResBlock), mid.1.* (AttnBlock), mid.2.* (ResBlock)
        let mid_vb = vb.pp("mid");
        let mid_block_1 = ResBlock::new(mid_vb.pp("0"), top_ch, top_ch)?;
        let mid_attn = AttnBlock::new(mid_vb.pp("1"), top_ch)?;
        let mid_block_2 = ResBlock::new(mid_vb.pp("2"), top_ch, top_ch)?;

        // ── up stages ──────────────────────────────────────────────────────
        // Safetensors keys use `conv_blocks.{i}.*` where i=0 is the widest
        // (512-channel) stage and i=4 is the narrowest (128-channel).
        //
        // We iterate ch_mult in reverse (wide → narrow):
        //   decoder_stage_idx 0 → ch_mult[4]=4 → conv_blocks.0 (512ch, has attn+upsample)
        //   decoder_stage_idx 1 → ch_mult[3]=2 → conv_blocks.1 (256ch, upsample)
        //   decoder_stage_idx 2 → ch_mult[2]=2 → conv_blocks.2 (256ch, upsample)
        //   decoder_stage_idx 3 → ch_mult[1]=1 → conv_blocks.3 (128ch, upsample)
        //   decoder_stage_idx 4 → ch_mult[0]=1 → conv_blocks.4 (128ch, no upsample)

        let num_stages = cfg.ch_mult.len();
        let cb_vb = vb.pp("conv_blocks");
        let mut up_stages: Vec<UpStage> = Vec::with_capacity(num_stages);

        let mut in_ch = top_ch;
        for (decoder_stage_idx, &mult) in cfg.ch_mult.iter().rev().enumerate() {
            let out_ch = cfg.base_channels * mult;

            let stage_vb = cb_vb.pp(decoder_stage_idx.to_string());
            let res_vb = stage_vb.pp("res");

            // num_res_blocks + 1 residual blocks per stage
            let mut blocks: Vec<ResBlock> = Vec::with_capacity(cfg.num_res_blocks + 1);
            let mut curr_ch = in_ch;
            for j in 0..=cfg.num_res_blocks {
                let blk_in = if j == 0 { curr_ch } else { out_ch };
                blocks.push(ResBlock::new(res_vb.pp(j.to_string()), blk_in, out_ch)?);
                curr_ch = out_ch;
            }

            // Attention blocks: only present in the first decoder stage
            // (conv_blocks.0), one per ResBlock, interleaved after each.
            let mut attns: Vec<AttnBlock> = Vec::new();
            if decoder_stage_idx == 0 {
                let attn_vb = stage_vb.pp("attn");
                for j in 0..=cfg.num_res_blocks {
                    attns.push(AttnBlock::new(attn_vb.pp(j.to_string()), out_ch)?);
                }
            }

            // Upsample exists on all stages except the last decoder stage
            // (conv_blocks.4 = narrowest features, no spatial upsampling).
            let upsample = if decoder_stage_idx < num_stages - 1 {
                Some(Upsample::new(stage_vb.pp("upsample"), out_ch)?)
            } else {
                None
            };

            up_stages.push(UpStage {
                blocks,
                attns,
                upsample,
            });
            in_ch = out_ch;
        }

        // ── output head ────────────────────────────────────────────────────
        // After all up-stages, in_ch == base_channels * ch_mult[0] = 128*1 = 128
        let final_ch = cfg.base_channels * cfg.ch_mult.first().copied().unwrap_or(1);

        let norm_out = candle_nn::group_norm(32, final_ch, 1e-6, vb.pp("norm_out"))?;

        let conv_out_cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_out = candle_nn::conv2d(final_ch, cfg.out_channels, 3, conv_out_cfg, vb.pp("conv_out"))?;

        Ok(Self {
            conv_in,
            mid_block_1,
            mid_attn,
            mid_block_2,
            up_stages,
            norm_out,
            conv_out,
        })
    }

    /// Decodes a latent grid into an RGB image.
    ///
    /// # Arguments
    ///
    /// * `z` — latent tensor of shape `[B, z_channels, H_lat, W_lat]`.
    ///   For Janus-Pro-7B this is `[B, 256, 24, 24]`.
    ///
    /// # Returns
    ///
    /// A tensor of shape `[B, out_channels, H_lat * 16, W_lat * 16]`,
    /// i.e. `[B, 3, 384, 384]` with the default config.
    ///
    /// # Errors
    ///
    /// Propagates any candle tensor operation error.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        // conv_in
        let mut h = z.apply(&self.conv_in)?;

        // mid
        h = self.mid_block_1.forward(&h)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_block_2.forward(&h)?;

        // upsampling stages
        for stage in &self.up_stages {
            h = stage.forward(h)?;
        }

        // output
        h = h.apply(&self.norm_out)?;
        h = candle_nn::ops::silu(&h)?;
        h = h.apply(&self.conv_out)?;

        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// VqCodebook
// ---------------------------------------------------------------------------

/// Discrete VQ codebook: a learned embedding table mapping token IDs to
/// latent vectors.
///
/// Wraps [`candle_nn::Embedding`] so that arbitrary index tensors can be
/// looked up and reshaped into the 2-D latent grid expected by
/// [`VqVaeDecoder::decode`].
///
/// # Example (offline smoke-test)
///
/// ```rust,no_run
/// use candle_core::{Device, DType, Tensor};
/// use candle_nn::VarMap;
/// use pcai_media_model::vq_vae::{VqCodebook, VqVaeConfig};
///
/// let dev = Device::Cpu;
/// let vm  = VarMap::new();
/// let vb  = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev);
/// let cfg = VqVaeConfig::default();
/// let cb  = VqCodebook::new(vb, 16384, cfg.z_channels).unwrap();
///
/// // 576 discrete tokens representing one 24×24 image grid
/// let tokens = Tensor::zeros((1, 576), DType::U32, &dev).unwrap();
/// let latent  = cb.lookup_grid(&tokens, 24, 24).unwrap(); // [1, 256, 24, 24]
/// assert_eq!(latent.dims(), &[1, 256, 24, 24]);
/// ```
pub struct VqCodebook {
    embedding: candle_nn::Embedding,
    /// Latent dimensionality (z_channels, typically 256).
    pub z_channels: usize,
}

impl VqCodebook {
    /// Constructs a [`VqCodebook`] with `vocab_size` entries of dimension
    /// `z_channels`.
    ///
    /// Weight path under `vb`: `weight` (the embedding weight matrix).
    /// Note: the parent VarBuilder should already be prefixed to
    /// `gen_vision_model.quantize.embedding` so the full key becomes
    /// `gen_vision_model.quantize.embedding.weight`.
    pub fn new(vb: VarBuilder, vocab_size: usize, z_channels: usize) -> Result<Self> {
        let embedding = candle_nn::embedding(vocab_size, z_channels, vb)?;
        Ok(Self { embedding, z_channels })
    }

    /// Looks up embeddings for a batch of token sequences and reshapes the
    /// result into a 2-D spatial grid suitable for the decoder.
    ///
    /// # Arguments
    ///
    /// * `tokens` — integer tensor of shape `[B, H_lat * W_lat]` (e.g.
    ///   `[B, 576]` for a 24×24 grid).
    /// * `h_lat`  — latent grid height (e.g. `24`).
    /// * `w_lat`  — latent grid width  (e.g. `24`).
    ///
    /// # Returns
    ///
    /// Tensor of shape `[B, z_channels, h_lat, w_lat]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the token sequence length does not equal
    /// `h_lat * w_lat`, or on any candle tensor error.
    pub fn lookup_grid(&self, tokens: &Tensor, h_lat: usize, w_lat: usize) -> Result<Tensor> {
        // tokens: [B, H*W] → embedding lookup → [B, H*W, z_channels]
        let embeds = self.embedding.forward(tokens)?;
        let (b, hw, z) = embeds.dims3()?;

        debug_assert_eq!(
            hw,
            h_lat * w_lat,
            "token sequence length {hw} != h_lat*w_lat {}",
            h_lat * w_lat
        );
        let _ = z; // used implicitly through self.z_channels

        // [B, H*W, z] → [B, z, H*W] → [B, z, H, W]
        embeds.transpose(1, 2)?.reshape((b, self.z_channels, h_lat, w_lat))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, IndexOp, Tensor};
    use candle_nn::VarMap;

    fn cpu_vb(vm: &VarMap) -> VarBuilder<'_> {
        candle_nn::VarBuilder::from_varmap(vm, DType::F32, &Device::Cpu)
    }

    // ── VqVaeConfig ──────────────────────────────────────────────────────

    #[test]
    fn test_config_default_top_ch() {
        let cfg = VqVaeConfig::default();
        // top_ch = base_channels * last(ch_mult) = 128 * 4 = 512
        let top_ch = cfg.base_channels * cfg.ch_mult.last().copied().unwrap_or(1);
        assert_eq!(top_ch, 512);
    }

    #[test]
    fn test_config_final_ch() {
        let cfg = VqVaeConfig::default();
        // final_ch = base_channels * first(ch_mult) = 128 * 1 = 128
        let final_ch = cfg.base_channels * cfg.ch_mult.first().copied().unwrap_or(1);
        assert_eq!(final_ch, 128);
    }

    // ── ResBlock ────────────────────────────────────────────────────────

    #[test]
    fn test_resblock_identity_channels_shape() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = cpu_vb(&vm);

        let block = ResBlock::new(vb.pp("rb"), 32, 32).unwrap();
        let x = Tensor::zeros((1_usize, 32_usize, 8_usize, 8_usize), DType::F32, &dev).unwrap();
        let y = block.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 32, 8, 8]);
    }

    #[test]
    fn test_resblock_channel_change_shape() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = cpu_vb(&vm);

        // nin_shortcut required because 64 != 32
        let block = ResBlock::new(vb.pp("rb"), 64, 32).unwrap();
        let x = Tensor::zeros((1_usize, 64_usize, 8_usize, 8_usize), DType::F32, &dev).unwrap();
        let y = block.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 32, 8, 8]);
    }

    #[test]
    fn test_resblock_skip_conv_only_when_needed() {
        let vm = VarMap::new();
        let vb = cpu_vb(&vm);

        let block_same = ResBlock::new(vb.pp("same"), 32, 32).unwrap();
        assert!(block_same.skip_conv.is_none(), "no skip conv when channels equal");

        let block_diff = ResBlock::new(vb.pp("diff"), 32, 64).unwrap();
        assert!(
            block_diff.skip_conv.is_some(),
            "skip conv required when channels differ"
        );
    }

    // ── AttnBlock ───────────────────────────────────────────────────────

    #[test]
    fn test_attnblock_shape_preserved() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = cpu_vb(&vm);

        let attn = AttnBlock::new(vb.pp("attn"), 32).unwrap();
        let x = Tensor::zeros((1_usize, 32_usize, 4_usize, 4_usize), DType::F32, &dev).unwrap();
        let y = attn.forward(&x).unwrap();
        assert_eq!(y.dims(), x.dims());
    }

    #[test]
    fn test_attnblock_batch_independence() {
        // Two independent identical inputs in a batch should produce identical
        // output rows (since there is no cross-sample interaction).
        // Channel count must be divisible by the GroupNorm group size (32).
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = cpu_vb(&vm);

        let attn = AttnBlock::new(vb.pp("attn"), 32).unwrap();
        let single = Tensor::rand(0.0_f32, 1.0_f32, (1_usize, 32_usize, 4_usize, 4_usize), &dev).unwrap();
        // Stack the same input twice to form a batch of 2.
        let batch = Tensor::cat(&[&single, &single], 0).unwrap();

        let out = attn.forward(&batch).unwrap();
        let row0 = out.i(0).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let row1 = out.i(1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in row0.iter().zip(row1.iter()) {
            assert!((a - b).abs() < 1e-5_f32, "batch rows differ: {a} vs {b}");
        }
    }

    // ── Upsample ─────────────────────────────────────────────────────────

    #[test]
    fn test_upsample_doubles_spatial() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = cpu_vb(&vm);

        let up = Upsample::new(vb.pp("up"), 32).unwrap();
        let x = Tensor::zeros((1_usize, 32_usize, 6_usize, 6_usize), DType::F32, &dev).unwrap();
        let y = up.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 32, 12, 12]);
    }

    // ── VqVaeDecoder ─────────────────────────────────────────────────────

    #[test]
    fn test_decoder_output_shape_default_config() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = cpu_vb(&vm);
        let cfg = VqVaeConfig::default();

        let decoder = VqVaeDecoder::new(vb, &cfg).unwrap();

        // [B=1, z=256, 24, 24] → [1, 3, 384, 384]
        let z = Tensor::zeros((1_usize, 256_usize, 24_usize, 24_usize), DType::F32, &dev).unwrap();
        let img = decoder.decode(&z).unwrap();
        assert_eq!(img.dims(), &[1, 3, 384, 384]);
    }

    #[test]
    fn test_decoder_output_shape_batch2() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = cpu_vb(&vm);
        let cfg = VqVaeConfig::default();

        let decoder = VqVaeDecoder::new(vb, &cfg).unwrap();

        let z = Tensor::zeros((2_usize, 256_usize, 24_usize, 24_usize), DType::F32, &dev).unwrap();
        let img = decoder.decode(&z).unwrap();
        assert_eq!(img.dims(), &[2, 3, 384, 384]);
    }

    #[test]
    fn test_decoder_custom_config() {
        // Minimal 2-level config: ch_mult=[1,2], base=64
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = cpu_vb(&vm);
        let cfg = VqVaeConfig {
            z_channels: 64,
            base_channels: 64,
            ch_mult: vec![1, 2],
            num_res_blocks: 1,
            out_channels: 3,
        };

        let decoder = VqVaeDecoder::new(vb, &cfg).unwrap();

        // top_ch = 64*2 = 128, 2 up stages, 1 upsample → spatial ×2
        // Input 8×8 → output 16×16
        let z = Tensor::zeros((1_usize, 64_usize, 8_usize, 8_usize), DType::F32, &dev).unwrap();
        let img = decoder.decode(&z).unwrap();
        assert_eq!(img.dims(), &[1, 3, 16, 16]);
    }

    // ── VqCodebook ────────────────────────────────────────────────────────

    #[test]
    fn test_codebook_lookup_grid_shape() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = cpu_vb(&vm);
        let cfg = VqVaeConfig::default();

        let cb = VqCodebook::new(vb, 16384, cfg.z_channels).unwrap();

        // 576 = 24*24 tokens per image, batch size 1
        let tokens = Tensor::zeros((1_usize, 576_usize), DType::U32, &dev).unwrap();
        let latent = cb.lookup_grid(&tokens, 24, 24).unwrap();
        assert_eq!(latent.dims(), &[1, 256, 24, 24]);
    }

    #[test]
    fn test_codebook_lookup_grid_batch() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = cpu_vb(&vm);

        let cb = VqCodebook::new(vb, 256, 64).unwrap();
        let tokens = Tensor::zeros((4_usize, 16_usize), DType::U32, &dev).unwrap();
        let latent = cb.lookup_grid(&tokens, 4, 4).unwrap();
        assert_eq!(latent.dims(), &[4, 64, 4, 4]);
    }
}
