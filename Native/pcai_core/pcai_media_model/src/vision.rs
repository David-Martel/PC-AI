//! Native Janus vision tower for image understanding.
//!
//! Janus-Pro uses a timm-style ViT checkpoint under `vision_model.vision_tower`
//! with fused QKV projections and a raw patch-token output. The understanding
//! path only needs the token-grid encoder, not the attention-pool head.

use candle_core::{IndexOp, Module, Result, Tensor};
use candle_nn::{conv2d, layer_norm, linear, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};

use crate::config::JanusConfig;

const VISION_HIDDEN_SIZE: usize = 1024;
const VISION_INTERMEDIATE_SIZE: usize = 4096;
const VISION_NUM_HEADS: usize = 16;
const VISION_NUM_HIDDEN_LAYERS: usize = 24;
const VISION_LAYER_NORM_EPS: f64 = 1e-6;

/// Configuration for the Janus understanding vision tower.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JanusVisionConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub layer_norm_eps: f64,
}

impl JanusVisionConfig {
    /// Number of patch tokens emitted by the encoder.
    pub fn num_patches(&self) -> usize {
        let grid = self.image_size / self.patch_size;
        grid * grid
    }

    /// Number of image tokens emitted by the encoder.
    pub fn num_image_tokens(&self) -> usize {
        self.num_patches()
    }

    /// Janus understanding encoder settings derived from the main model config.
    pub fn from_janus_config(config: &JanusConfig) -> Self {
        Self {
            image_size: config.image_size,
            patch_size: config.patch_size,
            hidden_size: VISION_HIDDEN_SIZE,
            intermediate_size: VISION_INTERMEDIATE_SIZE,
            num_attention_heads: VISION_NUM_HEADS,
            num_hidden_layers: VISION_NUM_HIDDEN_LAYERS,
            layer_norm_eps: VISION_LAYER_NORM_EPS,
        }
    }
}

#[derive(Debug, Clone)]
struct PatchEmbed {
    proj: Conv2d,
}

impl PatchEmbed {
    fn new(vb: VarBuilder<'_>, config: &JanusVisionConfig) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: config.patch_size,
            ..Default::default()
        };
        let proj = conv2d(3, config.hidden_size, config.patch_size, conv_cfg, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.proj.forward(xs)?.flatten_from(2)?.transpose(1, 2)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    qkv: Linear,
    proj: Linear,
    num_attention_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(vb: VarBuilder<'_>, config: &JanusVisionConfig) -> Result<Self> {
        let qkv = linear(config.hidden_size, config.hidden_size * 3, vb.pp("qkv"))?;
        let proj = linear(config.hidden_size, config.hidden_size, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            proj,
            num_attention_heads: config.num_attention_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((batch_size, seq_len, 3, self.num_attention_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.i(0)?.contiguous()?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;

        let attn = (q.matmul(&k.transpose(2, 3)?)? / (self.head_dim as f64).sqrt())?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let output = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, hidden_size))?;
        self.proj.forward(&output)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(vb: VarBuilder<'_>, config: &JanusVisionConfig) -> Result<Self> {
        let fc1 = linear(config.hidden_size, config.intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(config.intermediate_size, config.hidden_size, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden = self.fc1.forward(xs)?.gelu()?;
        self.fc2.forward(&hidden)
    }
}

#[derive(Debug, Clone)]
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    fn new(vb: VarBuilder<'_>, config: &JanusVisionConfig) -> Result<Self> {
        let norm1 = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("norm1"))?;
        let attn = Attention::new(vb.pp("attn"), config)?;
        let norm2 = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("norm2"))?;
        let mlp = Mlp::new(vb.pp("mlp"), config)?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.broadcast_add(&self.attn.forward(&self.norm1.forward(xs)?)?)?;
        xs.broadcast_add(&self.mlp.forward(&self.norm2.forward(&xs)?)?)
    }
}

/// Native Janus understanding vision tower.
///
/// Input: `[B, 3, H, W]`
/// Output: `[B, num_patches, 1024]`
#[derive(Debug, Clone)]
pub struct JanusVisionTower {
    patch_embed: PatchEmbed,
    pos_embed: Tensor,
    blocks: Vec<Block>,
    norm: LayerNorm,
    config: JanusVisionConfig,
}

impl JanusVisionTower {
    /// Build the native vision tower from `vision_model.vision_tower`.
    pub fn new(vb: VarBuilder<'_>, config: &JanusVisionConfig) -> Result<Self> {
        let patch_embed = PatchEmbed::new(vb.pp("patch_embed"), config)?;
        let pos_embed = vb.get((1, config.num_patches(), config.hidden_size), "pos_embed")?;
        let blocks = (0..config.num_hidden_layers)
            .map(|index| Block::new(vb.pp(format!("blocks.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        let norm = layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            norm,
            config: *config,
        })
    }

    /// Returns the encoder configuration.
    pub fn config(&self) -> &JanusVisionConfig {
        &self.config
    }

    /// Encode an image tensor into patch-token features.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.patch_embed.forward(xs)?.broadcast_add(&self.pos_embed)?;
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states)?;
        }
        self.norm.forward(&hidden_states)
    }
}

impl Module for JanusVisionTower {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Self::forward(self, xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarMap;

    #[test]
    fn test_janus_vision_config_matches_patch_grid() {
        let config = JanusVisionConfig::from_janus_config(&JanusConfig::janus_pro_1b());
        assert_eq!(config.num_patches(), 576);
        assert_eq!(config.hidden_size, 1024);
    }

    #[test]
    fn test_janus_vision_tower_output_shape() {
        let device = Device::Cpu;
        let vars = VarMap::new();
        let config = JanusVisionConfig::from_janus_config(&JanusConfig::janus_pro_1b());
        let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
        let tower = JanusVisionTower::new(vb, &config).expect("vision tower should construct");
        let input = Tensor::zeros(
            (1_usize, 3_usize, config.image_size, config.image_size),
            DType::F32,
            &device,
        )
        .expect("input tensor");
        let output = tower.forward(&input).expect("vision tower forward");
        assert_eq!(output.dims(), &[1, config.num_patches(), config.hidden_size]);
    }
}
