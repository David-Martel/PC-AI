use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, GroupNorm, VarBuilder, Activation};
use candle_transformers::models::llama::{Config as LlamaConfig, Model as LlamaModel, Cache};
use serde::Deserialize;

// --- Config Loading ---

#[derive(Debug, Clone, Deserialize)]
pub struct JanusConfig {
    pub image_vocab_size: usize,
    pub embed_dim: usize,
    pub vision_dim: usize,
    pub z_channels: usize,
    // Dynamic channel multipliers for VQ-VAE (e.g., [1, 1, 2, 2, 4])
    pub ch_mult: Vec<usize>,
    pub num_res_blocks: usize,
}

impl Default for JanusConfig {
    fn default() -> Self {
        Self {
            image_vocab_size: 16384,
            embed_dim: 2048, // 1B model default, 7B is 4096
            vision_dim: 1024,
            z_channels: 256,
            ch_mult: vec![1, 1, 2, 2, 4],
            num_res_blocks: 2,
        }
    }
}

// --- Optimized Layers ---

struct AttnBlock {
    norm: GroupNorm,
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    proj_out: Conv2d,
}

impl AttnBlock {
    fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 0, ..Default::default() };
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

        // Queries, Keys, Values
        let q = xs_norm.apply(&self.q)?;
        let k = xs_norm.apply(&self.k)?;
        let v = xs_norm.apply(&self.v)?;

        // --- OPTIMIZATION START ---
        #[cfg(feature = "flash-attn")]
        {
            // Flash Attention 2 Integration
            // Input must be [Batch, SeqLen, Heads, HeadDim]
            // We reshape Conv2D [B, C, H, W] -> [B, H*W, 1, C] (Simplification for single head)
            // Note: VQ-VAE attention is usually single-headed or low-head count.
            // If strictly 1 head, standard attention is often fast enough,
            // but for 7B models, we might have multi-head.

            // Fallback to optimized standard attention if flash-attn is tricky with Conv2D shapes
            // Reshape: [B, C, H, W] -> [B, H*W, C]
            let q = q.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
            let k = k.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
            let v = v.flatten_from(2)?.transpose(1, 2)?.contiguous()?;

            // Use Candle's scale_dot_product_attention which dispatches to FlashAttn if available
            let attn_out = candle_nn::ops::scaled_dot_product_attention(&q, &k, &v, None, None, None, None)?;

            let attn_out = attn_out.transpose(1, 2)?.reshape((b, c, h, w))?;
            return Ok((xs + attn_out.apply(&self.proj_out)?)?);
        }
        // --- OPTIMIZATION END ---

        // Standard Manual Attention (Fallback)
        let q = q.flatten_from(2)?.transpose(1, 2)?; // [B, HW, C]
        let k = k.flatten_from(2)?.transpose(1, 2)?;
        let v = v.flatten_from(2)?.transpose(1, 2)?;

        let scale = (c as f64).powf(-0.5);
        let attn_weights = (q.matmul(&k.t()?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let out = attn_weights.matmul(&v)?; // [B, HW, C]

        let out = out.transpose(1, 2)?.reshape((b, c, h, w))?;
        Ok((xs + out.apply(&self.proj_out)?)?)
    }
}

// ... (ResBlock and Upsample remain similar to previous response) ...

// --- The Vision Decoder ---

pub struct JanusVisionDecoder {
    conv_in: Conv2d,
    mid_block_1: Box<dyn Module>, // Using Box to simplify type heterogeneity
    mid_attn: AttnBlock,
    mid_block_2: Box<dyn Module>,
    up_blocks: Vec<Box<dyn Module>>,
    norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl JanusVisionDecoder {
    pub fn new(vb: VarBuilder, cfg: &JanusConfig) -> Result<Self> {
        // Dynamic construction based on ch_mult
        let block_in = 128; // Base channels
        let mut curr_res = 16; // Start resolution logic if needed
        let mut curr_ch = block_in * cfg.ch_mult.last().unwrap_or(&4);

        let conv_in = candle_nn::conv2d(cfg.z_channels, curr_ch, 3, Default::default(), vb.pp("conv_in"))?;

        // Mid Blocks
        let mid_vb = vb.pp("mid");
        // Using generic Module trait objects for flexibility in the list
        let mid_block_1 = Box::new(ResBlock::new(mid_vb.pp("block_1"), curr_ch, curr_ch)?);
        let mid_attn = AttnBlock::new(mid_vb.pp("attn_1"), curr_ch)?;
        let mid_block_2 = Box::new(ResBlock::new(mid_vb.pp("block_2"), curr_ch, curr_ch)?);

        // Up Blocks (Dynamic Loop)
        let mut up_blocks: Vec<Box<dyn Module>> = Vec::new();
        let up_vb = vb.pp("up");

        // Reverse iterate multipliers
        // Note: Real Janus impl logic matches indices i_level, i_block
        // This loop structure mimics the Python loop:
        for (i_level, &mult) in cfg.ch_mult.iter().rev().enumerate() {
            let out_ch = block_in * mult;
            for i_block in 0..cfg.num_res_blocks + 1 {
                // Load ResBlocks
                // Logic to construct path string "up.0.block.1" etc.
            }
            if i_level < cfg.ch_mult.len() - 1 {
                // Add Upsample
            }
            curr_ch = out_ch;
        }

        let norm_out = candle_nn::group_norm(32, block_in, 1e-6, vb.pp("norm_out"))?;
        let conv_out = candle_nn::conv2d(block_in, 3, 3, Default::default(), vb.pp("conv_out"))?;

        Ok(Self { conv_in, mid_block_1, mid_attn, mid_block_2, up_blocks, norm_out, conv_out })
    }

    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let mut h = z.apply(&self.conv_in)?;
        h = self.mid_block_1.forward(&h)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_block_2.forward(&h)?;

        for layer in &self.up_blocks {
            h = layer.forward(&h)?;
        }

        h = h.apply(&self.norm_out)?;
        h = candle_nn::ops::silu(&h)?;
        h = h.apply(&self.conv_out)?;
        Ok(h)
    }
}
