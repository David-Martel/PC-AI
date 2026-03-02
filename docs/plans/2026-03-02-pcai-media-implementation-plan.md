# pcai-media Janus-Pro Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a pure Rust media LLM agent around Janus-Pro with image generation, understanding, and upscaling — exposed via FFI and HTTP.

**Architecture:** Three-crate split (pcai_media_model, pcai_media, pcai_media_server) in the existing `Native/pcai_core/` workspace. Maximizes reuse of candle-transformers Llama/SigLIP/LLaVA/EnCodec, existing safetensors loading from functiongemma-core, and FFI/HTTP patterns from pcai-inference.

**Tech Stack:** Rust (edition 2021), candle 0.9 (CUDA+cuDNN), safetensors, hf-hub, axum, tokio, image crate. C# .NET 8 P/Invoke wrapper. PowerShell 5.1+ module.

---

## Task 1: Scaffold pcai_media_model crate

**Files:**
- Create: `Native/pcai_core/pcai_media_model/Cargo.toml`
- Create: `Native/pcai_core/pcai_media_model/src/lib.rs`
- Modify: `Native/pcai_core/Cargo.toml` (add workspace member)

**Step 1: Create Cargo.toml for pcai_media_model**

```toml
[package]
name = "pcai-media-model"
version = "0.1.0"
edition = "2021"
description = "Janus-Pro model architecture for PC_AI media agent"
license = "MIT"

[dependencies]
anyhow.workspace = true
candle-core = { version = "0.9", features = ["cuda", "cudnn"] }
candle-nn = "0.9"
candle-transformers = "0.9"
candle-flash-attn = { version = "0.9", optional = true }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[features]
default = []
flash-attn = ["dep:candle-flash-attn"]
```

**Step 2: Create minimal lib.rs**

```rust
pub mod config;
pub mod vq_vae;
pub mod generation_head;
pub mod tensor_utils;

// Re-export candle-transformers models we reuse directly
pub use candle_transformers::models::llama;
pub use candle_transformers::models::siglip;
```

**Step 3: Add workspace member**

In `Native/pcai_core/Cargo.toml`, add `"pcai_media_model"` to the `[workspace] members` array.

**Step 4: Verify it compiles**

Run: `cd Native/pcai_core && cargo check -p pcai-media-model`
Expected: Compiles (empty modules, will warn about missing files)

**Step 5: Commit**

```bash
git add Native/pcai_core/pcai_media_model/ Native/pcai_core/Cargo.toml
git commit -m "feat(media): scaffold pcai_media_model crate in workspace"
```

---

## Task 2: Implement config.rs (JanusConfig from HF config.json)

**Files:**
- Create: `Native/pcai_core/pcai_media_model/src/config.rs`

**Step 1: Write config test**

Create `Native/pcai_core/pcai_media_model/src/config.rs` with test at bottom:

```rust
use serde::Deserialize;
use candle_transformers::models::llama::Config as LlamaConfig;

/// Configuration for Janus-Pro models, parsed from HuggingFace config.json.
/// Supports both 1B and 7B variants.
#[derive(Debug, Clone, Deserialize)]
pub struct JanusConfig {
    /// Hidden size of the LLM backbone (1536 for 1B, 4096 for 7B)
    #[serde(default = "default_embed_dim")]
    pub hidden_size: usize,
    /// Number of transformer layers (24 for 1B, 30 for 7B)
    #[serde(default = "default_num_layers")]
    pub num_hidden_layers: usize,
    /// Number of attention heads
    #[serde(default = "default_num_heads")]
    pub num_attention_heads: usize,
    /// Number of KV heads (for GQA)
    #[serde(default = "default_num_kv_heads")]
    pub num_key_value_heads: usize,
    /// Vocabulary size (text + special tokens)
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    /// Intermediate MLP dimension
    #[serde(default = "default_intermediate")]
    pub intermediate_size: usize,
    /// Image vocabulary size for VQ tokenizer codebook
    #[serde(default = "default_image_vocab")]
    pub image_token_num_tokens: usize,
    /// Image generation resolution
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    /// Patch size for VQ tokenizer
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
}

fn default_embed_dim() -> usize { 4096 }
fn default_num_layers() -> usize { 30 }
fn default_num_heads() -> usize { 32 }
fn default_num_kv_heads() -> usize { 32 }
fn default_vocab_size() -> usize { 102400 }
fn default_intermediate() -> usize { 11008 }
fn default_image_vocab() -> usize { 16384 }
fn default_image_size() -> usize { 384 }
fn default_patch_size() -> usize { 16 }

impl JanusConfig {
    /// Number of image tokens in generation: (image_size / patch_size)^2
    pub fn num_image_tokens(&self) -> usize {
        let grid = self.image_size / self.patch_size;
        grid * grid
    }

    /// Build a candle LlamaConfig from this Janus config
    pub fn to_llama_config(&self) -> LlamaConfig {
        let mut cfg = LlamaConfig::config_7b_v2();
        cfg.hidden_size = self.hidden_size;
        cfg.num_hidden_layers = self.num_hidden_layers;
        cfg.num_attention_heads = self.num_attention_heads;
        cfg.num_key_value_heads = self.num_key_value_heads;
        cfg.vocab_size = self.vocab_size;
        cfg.intermediate_size = self.intermediate_size;
        cfg
    }

    /// Preset for Janus-Pro-1B
    pub fn janus_pro_1b() -> Self {
        Self {
            hidden_size: 2048,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            vocab_size: 102400,
            intermediate_size: 5504,
            image_token_num_tokens: 16384,
            image_size: 384,
            patch_size: 16,
        }
    }

    /// Preset for Janus-Pro-7B
    pub fn janus_pro_7b() -> Self {
        Self {
            hidden_size: 4096,
            num_hidden_layers: 30,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            vocab_size: 102400,
            intermediate_size: 11008,
            image_token_num_tokens: 16384,
            image_size: 384,
            patch_size: 16,
        }
    }

    /// Load from a config.json file
    pub fn from_file(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_7b_preset_image_tokens() {
        let cfg = JanusConfig::janus_pro_7b();
        assert_eq!(cfg.num_image_tokens(), 576); // 24*24
    }

    #[test]
    fn test_1b_preset_image_tokens() {
        let cfg = JanusConfig::janus_pro_1b();
        assert_eq!(cfg.num_image_tokens(), 576);
    }

    #[test]
    fn test_llama_config_conversion() {
        let cfg = JanusConfig::janus_pro_7b();
        let llama = cfg.to_llama_config();
        assert_eq!(llama.hidden_size, 4096);
        assert_eq!(llama.num_hidden_layers, 30);
        assert_eq!(llama.vocab_size, 102400);
    }

    #[test]
    fn test_serde_roundtrip() {
        let json = r#"{
            "hidden_size": 4096,
            "num_hidden_layers": 30,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "vocab_size": 102400,
            "intermediate_size": 11008,
            "image_token_num_tokens": 16384,
            "image_size": 384,
            "patch_size": 16
        }"#;
        let cfg: JanusConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_image_tokens(), 576);
    }

    #[test]
    fn test_serde_defaults() {
        let json = r#"{}"#;
        let cfg: JanusConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.vocab_size, 102400);
    }
}
```

**Step 2: Run tests**

Run: `cd Native/pcai_core && cargo test -p pcai-media-model`
Expected: 5 tests pass

**Step 3: Commit**

```bash
git add Native/pcai_core/pcai_media_model/src/config.rs
git commit -m "feat(media): add JanusConfig with serde, presets, and LlamaConfig conversion"
```

---

## Task 3: Migrate tensor_utils.rs

**Files:**
- Create: `Native/pcai_core/pcai_media_model/src/tensor_utils.rs`
- Source: `AI-Media/src/tensor_utils.rs`

**Step 1: Copy and enhance with tests**

```rust
use candle_core::{DType, Result, Tensor};

/// Normalize image tensor from [0, 255] to [-1, 1]
pub fn normalize(image: &Tensor) -> Result<Tensor> {
    let image = image.to_dtype(DType::F32)?;
    let image = (image / 127.5)?;
    let image = (image - 1.0)?;
    Ok(image)
}

/// Denormalize tensor from [-1, 1] to [0, 255] as U8
pub fn denormalize(tensor: &Tensor) -> Result<Tensor> {
    let tensor = (tensor + 1.0)?;
    let tensor = (tensor / 2.0)?;
    let tensor = (tensor * 255.0)?;
    let tensor = tensor.clamp(0f32, 255f32)?;
    let tensor = tensor.to_dtype(DType::U8)?;
    Ok(tensor)
}

/// Create an upper-triangular causal mask for autoregressive generation
pub fn create_causal_mask(size: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mask: Vec<u8> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_vec(mask, (size, size), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_normalize_range() {
        let data = vec![0f32, 127.5, 255.0];
        let t = Tensor::from_vec(data, (1, 3), &Device::Cpu).unwrap();
        let n = normalize(&t).unwrap();
        let vals: Vec<f32> = n.flatten_all().unwrap().to_vec1().unwrap();
        assert!((vals[0] - (-1.0)).abs() < 1e-5);
        assert!((vals[1] - 0.0).abs() < 1e-5);
        assert!((vals[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_denormalize_range() {
        let data = vec![-1.0f32, 0.0, 1.0];
        let t = Tensor::from_vec(data, (1, 3), &Device::Cpu).unwrap();
        let d = denormalize(&t).unwrap();
        let vals: Vec<u8> = d.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vals[0], 0);
        assert_eq!(vals[1], 127); // floor of 127.5
        assert_eq!(vals[2], 255);
    }

    #[test]
    fn test_normalize_denormalize_roundtrip() {
        let data = vec![0f32, 50.0, 100.0, 200.0, 255.0];
        let t = Tensor::from_vec(data.clone(), (1, 5), &Device::Cpu).unwrap();
        let n = normalize(&t).unwrap();
        let d = denormalize(&n).unwrap();
        let result: Vec<u8> = d.flatten_all().unwrap().to_vec1().unwrap();
        for (orig, recovered) in data.iter().zip(result.iter()) {
            assert!((*orig as i32 - *recovered as i32).abs() <= 1);
        }
    }

    #[test]
    fn test_causal_mask() {
        let mask = create_causal_mask(3, &Device::Cpu).unwrap();
        let vals: Vec<u8> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // Row 0: [0, 1, 1] - can see token 0 only
        // Row 1: [0, 0, 1] - can see tokens 0-1
        // Row 2: [0, 0, 0] - can see tokens 0-2
        assert_eq!(vals, vec![0, 1, 1, 0, 0, 1, 0, 0, 0]);
    }
}
```

**Step 2: Run tests**

Run: `cd Native/pcai_core && cargo test -p pcai-media-model -- tensor_utils`
Expected: 4 tests pass

**Step 3: Commit**

```bash
git add Native/pcai_core/pcai_media_model/src/tensor_utils.rs
git commit -m "feat(media): add tensor_utils with normalize/denormalize/causal_mask"
```

---

## Task 4: Implement VQ-VAE decoder (core generation component)

**Files:**
- Create: `Native/pcai_core/pcai_media_model/src/vq_vae.rs`
- Reference: `AI-Media/src/janus_model.rs` (ResBlock, AttnBlock, Upsample)
- Reference: `AI-Media/src/janus_model_v2.rs` (dynamic ch_mult)

This is the critical component that doesn't exist in candle-transformers. We port it from the existing skeleton code, completing the up_blocks loop.

**Step 1: Implement the VQ-VAE decoder**

```rust
use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, GroupNorm, VarBuilder};

// --- Helper Layers (from AI-Media/src/janus_model.rs, completed) ---

pub struct ResBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    skip_conv: Option<Conv2d>,
}

impl ResBlock {
    pub fn new(vb: VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let norm1 = candle_nn::group_norm(32, in_channels, 1e-6, vb.pp("norm1"))?;
        let conv1 = candle_nn::conv2d(in_channels, out_channels, 3, cfg, vb.pp("conv1"))?;
        let norm2 = candle_nn::group_norm(32, out_channels, 1e-6, vb.pp("norm2"))?;
        let conv2 = candle_nn::conv2d(out_channels, out_channels, 3, cfg, vb.pp("conv2"))?;

        let skip_conv = if in_channels != out_channels {
            let skip_cfg = Conv2dConfig { padding: 0, ..Default::default() };
            Some(candle_nn::conv2d(in_channels, out_channels, 1, skip_cfg, vb.pp("nin_shortcut"))?)
        } else {
            None
        };

        Ok(Self { norm1, conv1, norm2, conv2, skip_conv })
    }
}

impl Module for ResBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs.apply(&self.norm1)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = h.apply(&self.conv1)?;
        let h = h.apply(&self.norm2)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = h.apply(&self.conv2)?;

        let identity = match &self.skip_conv {
            Some(conv) => xs.apply(conv)?,
            None => xs.clone(),
        };
        (h + identity)
    }
}

pub struct AttnBlock {
    norm: GroupNorm,
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    proj_out: Conv2d,
}

impl AttnBlock {
    pub fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 0, ..Default::default() };
        let norm = candle_nn::group_norm(32, channels, 1e-6, vb.pp("norm"))?;
        let q = candle_nn::conv2d(channels, channels, 1, cfg, vb.pp("q"))?;
        let k = candle_nn::conv2d(channels, channels, 1, cfg, vb.pp("k"))?;
        let v = candle_nn::conv2d(channels, channels, 1, cfg, vb.pp("v"))?;
        let proj_out = candle_nn::conv2d(channels, channels, 1, cfg, vb.pp("proj_out"))?;
        Ok(Self { norm, q, k, v, proj_out })
    }
}

impl Module for AttnBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = xs.dims4()?;
        let h_norm = xs.apply(&self.norm)?;

        let q = h_norm.apply(&self.q)?.reshape((b, c, h * w))?.transpose(1, 2)?;
        let k = h_norm.apply(&self.k)?.reshape((b, c, h * w))?.transpose(1, 2)?;
        let v = h_norm.apply(&self.v)?.reshape((b, c, h * w))?.transpose(1, 2)?;

        let scale = (c as f64).powf(-0.5);
        let attn_weights = (q.matmul(&k.transpose(1, 2)?)? * scale)?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;

        let attn_out = attn_weights.matmul(&v)?;
        let attn_out = attn_out.transpose(1, 2)?.reshape((b, c, h, w))?;

        let out = attn_out.apply(&self.proj_out)?;
        (xs + out)
    }
}

pub struct Upsample {
    conv: Conv2d,
}

impl Upsample {
    pub fn new(vb: VarBuilder, channels: usize) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        let conv = candle_nn::conv2d(channels, channels, 3, cfg, vb.pp("conv"))?;
        Ok(Self { conv })
    }
}

impl Module for Upsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _c, h, w) = xs.dims4()?;
        let xs = xs.upsample_nearest2d(h * 2, w * 2)?;
        xs.apply(&self.conv)
    }
}

// --- VQ-VAE Decoder Configuration ---

/// Channel multipliers for the VQ-VAE decoder.
/// Standard LlamaGen config: [1, 1, 2, 2, 4]
/// Processed in reverse for decoder (upsampling).
#[derive(Debug, Clone)]
pub struct VqVaeConfig {
    pub z_channels: usize,
    pub base_channels: usize,
    pub ch_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub out_channels: usize,
}

impl Default for VqVaeConfig {
    fn default() -> Self {
        Self {
            z_channels: 256,
            base_channels: 128,
            ch_mult: vec![1, 1, 2, 2, 4],
            num_res_blocks: 2,
            out_channels: 3, // RGB
        }
    }
}

// --- Upsampling Stage (ResBlocks + optional Upsample) ---

struct UpStage {
    res_blocks: Vec<ResBlock>,
    upsample: Option<Upsample>,
}

impl UpStage {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = xs.clone();
        for block in &self.res_blocks {
            h = block.forward(&h)?;
        }
        if let Some(up) = &self.upsample {
            h = up.forward(&h)?;
        }
        Ok(h)
    }
}

// --- Full VQ-VAE Decoder ---

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
    pub fn new(vb: VarBuilder, cfg: &VqVaeConfig) -> Result<Self> {
        let conv_cfg = Conv2dConfig { padding: 1, ..Default::default() };

        // Top channel count = base_channels * last ch_mult entry
        let top_ch = cfg.base_channels * cfg.ch_mult.last().copied().unwrap_or(1);

        // Input conv: z_channels -> top_ch
        let conv_in = candle_nn::conv2d(cfg.z_channels, top_ch, 3, conv_cfg, vb.pp("conv_in"))?;

        // Mid block
        let mid_vb = vb.pp("mid");
        let mid_block_1 = ResBlock::new(mid_vb.pp("block_1"), top_ch, top_ch)?;
        let mid_attn = AttnBlock::new(mid_vb.pp("attn_1"), top_ch)?;
        let mid_block_2 = ResBlock::new(mid_vb.pp("block_2"), top_ch, top_ch)?;

        // Up stages: iterate ch_mult in reverse
        // Each stage has num_res_blocks ResBlocks and an optional Upsample (all but first in reversed order)
        let reversed_mult: Vec<usize> = cfg.ch_mult.iter().copied().rev().collect();
        let num_resolutions = reversed_mult.len();
        let mut up_stages = Vec::new();

        let mut in_ch = top_ch;
        for (i, &mult) in reversed_mult.iter().enumerate() {
            let out_ch = cfg.base_channels * mult;
            let stage_vb = vb.pp(format!("up.{}", num_resolutions - 1 - i));

            let mut res_blocks = Vec::new();
            for j in 0..(cfg.num_res_blocks + 1) {
                let block_in = if j == 0 { in_ch } else { out_ch };
                res_blocks.push(ResBlock::new(
                    stage_vb.pp(format!("block.{}", j)),
                    block_in,
                    out_ch,
                )?);
            }

            // Upsample for all but the last stage (first in original order)
            let upsample = if i < num_resolutions - 1 {
                Some(Upsample::new(stage_vb.pp("upsample"), out_ch)?)
            } else {
                None
            };

            up_stages.push(UpStage { res_blocks, upsample });
            in_ch = out_ch;
        }

        // Output: GroupNorm + conv to RGB
        let final_ch = cfg.base_channels * reversed_mult.last().copied().unwrap_or(1);
        let norm_out = candle_nn::group_norm(32, final_ch, 1e-6, vb.pp("norm_out"))?;
        let conv_out = candle_nn::conv2d(final_ch, cfg.out_channels, 3, conv_cfg, vb.pp("conv_out"))?;

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

    /// Decode latent tensor [batch, z_channels, h, w] -> [batch, 3, H, W]
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let mut h = z.apply(&self.conv_in)?;

        // Mid
        h = self.mid_block_1.forward(&h)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_block_2.forward(&h)?;

        // Up stages
        for stage in &self.up_stages {
            h = stage.forward(&h)?;
        }

        // Output
        h = h.apply(&self.norm_out)?;
        h = candle_nn::ops::silu(&h)?;
        h = h.apply(&self.conv_out)?;

        Ok(h)
    }
}

/// VQ Codebook: maps discrete token IDs to continuous embedding vectors.
/// Replicates the lookup portion of LlamaGen's VQ tokenizer.
pub struct VqCodebook {
    embedding: candle_nn::Embedding,
    pub num_tokens: usize,
    pub embed_dim: usize,
}

impl VqCodebook {
    pub fn new(vb: VarBuilder, num_tokens: usize, embed_dim: usize) -> Result<Self> {
        let embedding = candle_nn::embedding(num_tokens, embed_dim, vb.pp("embedding"))?;
        Ok(Self { embedding, num_tokens, embed_dim })
    }

    /// Look up embeddings for token IDs
    pub fn decode(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.embedding.forward(token_ids)
    }
}
```

**Step 2: Run compilation check**

Run: `cd Native/pcai_core && cargo check -p pcai-media-model`
Expected: Compiles without errors

**Step 3: Commit**

```bash
git add Native/pcai_core/pcai_media_model/src/vq_vae.rs
git commit -m "feat(media): implement VQ-VAE decoder with dynamic ch_mult upsampling stages"
```

---

## Task 5: Implement generation_head.rs

**Files:**
- Create: `Native/pcai_core/pcai_media_model/src/generation_head.rs`

**Step 1: Write generation head and aligner**

```rust
use candle_core::{Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

/// Projects LLM hidden states to image vocabulary logits.
/// Linear(hidden_size -> image_vocab_size)
pub struct GenerationHead {
    linear: Linear,
}

impl GenerationHead {
    pub fn new(vb: VarBuilder, hidden_size: usize, image_vocab_size: usize) -> Result<Self> {
        let linear = candle_nn::linear(hidden_size, image_vocab_size, vb)?;
        Ok(Self { linear })
    }
}

impl Module for GenerationHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear.forward(xs)
    }
}

/// MLP aligner that maps between the LLM embedding space and another space.
/// Used for both understanding (vision -> LLM) and generation (LLM -> VQ latent).
pub struct MlpAligner {
    layer1: Linear,
    layer2: Linear,
}

impl MlpAligner {
    pub fn new(vb: VarBuilder, in_dim: usize, out_dim: usize) -> Result<Self> {
        let layer1 = candle_nn::linear(in_dim, out_dim, vb.pp("0"))?;
        let layer2 = candle_nn::linear(out_dim, out_dim, vb.pp("2"))?;
        Ok(Self { layer1, layer2 })
    }
}

impl Module for MlpAligner {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.layer1.forward(xs)?;
        let h = candle_nn::ops::gelu(&h)?;
        self.layer2.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_generation_head_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let head = GenerationHead::new(vb, 4096, 16384).unwrap();
        let input = Tensor::zeros((1, 10, 4096), DType::F32, &device).unwrap();
        let output = head.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 10, 16384]);
    }

    #[test]
    fn test_mlp_aligner_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let aligner = MlpAligner::new(vb, 1024, 4096).unwrap();
        let input = Tensor::zeros((1, 576, 1024), DType::F32, &device).unwrap();
        let output = aligner.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 576, 4096]);
    }
}
```

**Step 2: Run tests**

Run: `cd Native/pcai_core && cargo test -p pcai-media-model -- generation_head`
Expected: 2 tests pass

**Step 3: Commit**

```bash
git add Native/pcai_core/pcai_media_model/src/generation_head.rs
git commit -m "feat(media): add GenerationHead and MlpAligner with shape tests"
```

---

## Task 6: Assemble JanusModel in lib.rs

**Files:**
- Modify: `Native/pcai_core/pcai_media_model/src/lib.rs`

**Step 1: Write the unified JanusModel facade**

Replace `lib.rs` content:

```rust
pub mod config;
pub mod vq_vae;
pub mod generation_head;
pub mod tensor_utils;

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};
use candle_transformers::models::llama::{self, Cache};
use candle_transformers::models::siglip;

use crate::config::JanusConfig;
use crate::generation_head::{GenerationHead, MlpAligner};
use crate::vq_vae::{VqCodebook, VqVaeConfig, VqVaeDecoder};

/// Unified Janus-Pro model supporting both understanding and generation.
pub struct JanusModel {
    /// LLM backbone (Llama architecture from DeepSeek-LLM)
    pub llama: llama::Llama,
    /// Generation head: projects LLM hidden states to image vocab logits
    pub gen_head: GenerationHead,
    /// Generation aligner: maps image token embeddings to LLM input space
    pub gen_aligner: MlpAligner,
    /// VQ codebook: discrete image token vocabulary
    pub vq_codebook: VqCodebook,
    /// VQ-VAE decoder: converts latent grid to pixel image
    pub vq_decoder: VqVaeDecoder,
    /// Understanding aligner: maps SigLIP features to LLM input space
    pub understand_aligner: MlpAligner,
    /// Model config
    pub config: JanusConfig,
}

impl JanusModel {
    pub fn new(vb: VarBuilder, config: &JanusConfig) -> Result<Self> {
        let llama_cfg = config.to_llama_config();

        // LLM backbone
        let llama = llama::Llama::load(vb.pp("language_model"), &llama_cfg)?;

        // Generation pathway
        let gen_head = GenerationHead::new(
            vb.pp("gen_head"),
            config.hidden_size,
            config.image_token_num_tokens,
        )?;
        let gen_aligner = MlpAligner::new(
            vb.pp("gen_aligner"),
            config.hidden_size,
            config.hidden_size,
        )?;
        let vq_codebook = VqCodebook::new(
            vb.pp("gen_vision_model.quantize.embedding"),
            config.image_token_num_tokens,
            256, // standard VQ latent dim
        )?;
        let vq_decoder = VqVaeDecoder::new(
            vb.pp("gen_vision_model.decoder"),
            &VqVaeConfig::default(),
        )?;

        // Understanding pathway
        let understand_aligner = MlpAligner::new(
            vb.pp("aligner"),
            1024, // SigLIP-L output dim
            config.hidden_size,
        )?;

        Ok(Self {
            llama,
            gen_head,
            gen_aligner,
            vq_codebook,
            vq_decoder,
            understand_aligner,
            config,
        })
    }

    /// LLM forward pass for autoregressive token prediction
    pub fn forward(&self, input_ids: &Tensor, pos: usize, cache: &mut Cache) -> Result<Tensor> {
        self.llama.forward(input_ids, pos, cache)
    }

    /// Get input embeddings from the LLM's embedding table
    pub fn embed_tokens(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.llama.embed_tokens(token_ids)
    }

    /// Project LLM hidden states to image vocabulary logits
    pub fn project_to_image_vocab(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.gen_head.forward(hidden_states)
    }

    /// Convert generated image token IDs to pixel image
    /// tokens shape: [batch, num_image_tokens] (e.g., [1, 576])
    pub fn decode_image_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
        // Look up VQ codebook embeddings
        let embeds = self.vq_codebook.decode(tokens)?; // [B, 576, 256]

        // Reshape to 2D grid: [B, 256, 24, 24]
        let (b, _seq, embed_dim) = embeds.dims3()?;
        let grid_size = (self.config.num_image_tokens() as f64).sqrt() as usize; // 24
        let latents = embeds.transpose(1, 2)?.reshape((b, embed_dim, grid_size, grid_size))?;

        // Decode to pixels
        self.vq_decoder.decode(&latents)
    }
}
```

**Step 2: Verify compilation**

Run: `cd Native/pcai_core && cargo check -p pcai-media-model`
Expected: Compiles (may warn about unused `siglip` import — keep it for Task 11)

**Step 3: Commit**

```bash
git add Native/pcai_core/pcai_media_model/src/lib.rs
git commit -m "feat(media): assemble JanusModel facade wiring Llama+VQ+GenHead+Aligners"
```

---

## Task 7: Scaffold pcai_media crate (pipeline + FFI)

**Files:**
- Create: `Native/pcai_core/pcai_media/Cargo.toml`
- Create: `Native/pcai_core/pcai_media/src/lib.rs`
- Create: `Native/pcai_core/pcai_media/src/config.rs`
- Modify: `Native/pcai_core/Cargo.toml` (add workspace member)

**Step 1: Create Cargo.toml**

```toml
[package]
name = "pcai-media"
version = "0.1.0"
edition = "2021"
description = "Janus-Pro media pipeline with FFI and HTTP support"
license = "MIT"

[lib]
name = "pcai_media"
crate-type = ["cdylib", "rlib"]

[dependencies]
pcai-media-model = { path = "../pcai_media_model" }
anyhow.workspace = true
thiserror.workspace = true
tokio.workspace = true
serde.workspace = true
serde_json.workspace = true
tracing.workspace = true
tracing-subscriber.workspace = true

candle-core = { version = "0.9", features = ["cuda", "cudnn"] }
candle-nn = "0.9"
candle-transformers = "0.9"
safetensors = "0.4"
tokenizers = "0.20"
image = "0.25"
hf-hub = "0.4"
memmap2 = "0.9"

[features]
default = ["ffi", "server"]
ffi = []
server = ["dep:axum", "dep:tower-http"]
flash-attn = ["pcai-media-model/flash-attn"]

[dependencies.axum]
workspace = true
optional = true

[dependencies.tower-http]
workspace = true
optional = true
```

**Step 2: Create pipeline config**

Create `src/config.rs`:

```rust
use candle_core::{DType, Device};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Pipeline configuration for Janus media inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// HuggingFace model ID or local path
    #[serde(default = "default_model")]
    pub model: String,
    /// Device: "cuda:0", "cuda:1", "cpu"
    #[serde(default = "default_device")]
    pub device: String,
    /// Data type: "bf16", "f16", "f32"
    #[serde(default = "default_dtype")]
    pub dtype: String,
    /// CFG guidance scale for image generation
    #[serde(default = "default_cfg_scale")]
    pub guidance_scale: f64,
    /// Sampling temperature
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Number of images to generate in parallel
    #[serde(default = "default_parallel")]
    pub parallel_size: usize,
    /// GPU layers to offload (-1 = all)
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: i32,
}

fn default_model() -> String { "deepseek-ai/Janus-Pro-7B".into() }
fn default_device() -> String { "cuda:0".into() }
fn default_dtype() -> String { "bf16".into() }
fn default_cfg_scale() -> f64 { 5.0 }
fn default_temperature() -> f64 { 1.0 }
fn default_parallel() -> usize { 1 }
fn default_gpu_layers() -> i32 { -1 }

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            model: default_model(),
            device: default_device(),
            dtype: default_dtype(),
            guidance_scale: default_cfg_scale(),
            temperature: default_temperature(),
            parallel_size: default_parallel(),
            gpu_layers: default_gpu_layers(),
        }
    }
}

impl PipelineConfig {
    pub fn resolve_device(&self) -> candle_core::Result<Device> {
        if self.device.starts_with("cuda") {
            let idx: usize = self.device
                .strip_prefix("cuda:")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            Device::new_cuda(idx)
        } else {
            Ok(Device::Cpu)
        }
    }

    pub fn resolve_dtype(&self) -> DType {
        match self.dtype.as_str() {
            "f16" => DType::F16,
            "f32" => DType::F32,
            "bf16" | _ => DType::BF16,
        }
    }

    pub fn from_file(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.guidance_scale, 5.0);
        assert_eq!(cfg.temperature, 1.0);
    }

    #[test]
    fn test_serde_roundtrip() {
        let cfg = PipelineConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let parsed: PipelineConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model, cfg.model);
    }
}
```

**Step 3: Create lib.rs**

```rust
pub mod config;
pub mod hub;
pub mod generate;
pub mod understand;

#[cfg(feature = "ffi")]
pub mod ffi;

pub use pcai_media_model as model;
```

**Step 4: Create stub modules**

Create empty stub files (will be filled in subsequent tasks):
- `src/hub.rs`: `// HuggingFace Hub model download + safetensors loading`
- `src/generate.rs`: `// Text-to-image generation pipeline`
- `src/understand.rs`: `// Image-to-text understanding pipeline`
- `src/ffi/mod.rs`: `// FFI exports`

**Step 5: Add workspace member and verify**

Add `"pcai_media"` to workspace members in `Native/pcai_core/Cargo.toml`.

Run: `cd Native/pcai_core && cargo check -p pcai-media`
Expected: Compiles with warnings about empty modules

**Step 6: Commit**

```bash
git add Native/pcai_core/pcai_media/ Native/pcai_core/Cargo.toml
git commit -m "feat(media): scaffold pcai_media crate with pipeline config and module stubs"
```

---

## Task 8: Implement hub.rs (model download + weight loading)

**Files:**
- Modify: `Native/pcai_core/pcai_media/src/hub.rs`
- Reference: `Deploy/rust-functiongemma-core/src/safetensors_utils.rs`

**Step 1: Implement model hub and weight loading**

Reuse safetensors loading patterns from functiongemma-core:

```rust
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use safetensors::MmapedSafetensors;

/// Resolve a model path: either local directory or HuggingFace repo ID.
/// Returns the local path where model files reside.
pub fn resolve_model_path(model_id: &str) -> Result<PathBuf> {
    let local = PathBuf::from(model_id);
    if local.is_dir() {
        return Ok(local);
    }

    // Download from HuggingFace Hub
    tracing::info!("Downloading model from HuggingFace: {}", model_id);
    let api = hf_hub::api::sync::Api::new()
        .context("Failed to initialize HuggingFace Hub API")?;
    let repo = api.model(model_id.to_string());

    // Download config.json and tokenizer.json first
    repo.get("config.json").context("Failed to download config.json")?;
    repo.get("tokenizer.json").context("Failed to download tokenizer.json")?;

    // Download all safetensors files
    let repo_info = api.model(model_id.to_string());

    // The hf-hub API caches files in ~/.cache/huggingface/hub/
    // After downloading, the model path is the repo's local cache dir
    let snapshot_path = repo_info.get("config.json")?;
    let model_dir = snapshot_path.parent()
        .context("Cannot determine model directory")?
        .to_path_buf();

    Ok(model_dir)
}

/// Collect all safetensors files for a model.
/// Handles both single-file (model.safetensors) and sharded (model-00001-of-00003.safetensors).
/// Adapted from Deploy/rust-functiongemma-core/src/safetensors_utils.rs
pub fn collect_safetensors(model_path: &Path) -> Vec<PathBuf> {
    // Check for single file
    let direct = model_path.join("model.safetensors");
    if direct.exists() {
        return vec![direct];
    }

    // Check for sharded files
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(model_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|v| v.to_str()) {
                if name.starts_with("model-") && name.ends_with(".safetensors") {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    files
}

/// Open memory-mapped safetensors files.
/// Adapted from Deploy/rust-functiongemma-core/src/safetensors_utils.rs
pub fn open_safetensors(paths: &[PathBuf]) -> Result<MmapedSafetensors> {
    anyhow::ensure!(!paths.is_empty(), "No safetensors files found");
    let st = unsafe {
        if paths.len() == 1 {
            MmapedSafetensors::new(&paths[0])?
        } else {
            MmapedSafetensors::multi(paths)?
        }
    };
    Ok(st)
}

/// Load model weights into a VarMap from safetensors files.
/// Adapted from Deploy/rust-functiongemma-core/src/safetensors_utils.rs
pub fn load_weights(
    varmap: &VarMap,
    paths: &[PathBuf],
    dtype: DType,
    device: &Device,
) -> Result<usize> {
    let st = open_safetensors(paths)?;
    let tensors = st.tensors();
    let st_names: std::collections::HashSet<String> =
        tensors.iter().map(|(n, _)| n.clone()).collect();

    tracing::info!("Safetensors: {} tensors found", st_names.len());
    if tracing::enabled!(tracing::Level::DEBUG) {
        let sample: Vec<_> = st_names.iter().take(5).collect();
        tracing::debug!("Sample keys: {:?}", sample);
    }

    let data = varmap.data().lock().expect("VarMap lock poisoned");
    let mut loaded = 0;

    for (name, var) in data.iter() {
        if let Ok(tensor) = st.load(name, device) {
            let tensor = if tensor.dtype() != dtype {
                tensor.to_dtype(dtype)?
            } else {
                tensor
            };
            var.set(&tensor)?;
            loaded += 1;
        }
    }

    tracing::info!("Loaded {}/{} tensors from safetensors", loaded, data.len());
    Ok(loaded)
}

/// Load a tokenizer from a model directory or HF repo
pub fn load_tokenizer(model_path: &Path) -> Result<tokenizers::Tokenizer> {
    let tokenizer_path = model_path.join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))
}
```

**Step 2: Verify compilation**

Run: `cd Native/pcai_core && cargo check -p pcai-media`
Expected: Compiles

**Step 3: Commit**

```bash
git add Native/pcai_core/pcai_media/src/hub.rs
git commit -m "feat(media): implement HF Hub model download and safetensors weight loading"
```

---

## Task 9: Implement generate.rs (text-to-image pipeline)

**Files:**
- Modify: `Native/pcai_core/pcai_media/src/generate.rs`
- Reference: `AI-Media/src/main.rs` (generation loop pattern)

**Step 1: Implement the generation pipeline**

```rust
use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{VarBuilder, VarMap};
use candle_transformers::models::llama::Cache;
use image::{ImageBuffer, Rgb};

use pcai_media_model::config::JanusConfig;
use pcai_media_model::tensor_utils::denormalize;
use pcai_media_model::JanusModel;

use crate::config::PipelineConfig;
use crate::hub;

/// Loaded pipeline ready for image generation.
pub struct GenerationPipeline {
    model: JanusModel,
    tokenizer: tokenizers::Tokenizer,
    config: PipelineConfig,
    model_config: JanusConfig,
    device: Device,
    dtype: DType,
}

impl GenerationPipeline {
    /// Load model from HF Hub or local path.
    pub fn load(config: PipelineConfig) -> Result<Self> {
        let device = config.resolve_device()
            .context("Failed to initialize device")?;
        let dtype = config.resolve_dtype();

        tracing::info!("Loading Janus model: {} on {:?} ({:?})", config.model, device, dtype);

        // Resolve model path
        let model_path = hub::resolve_model_path(&config.model)?;

        // Load model config
        let model_config = JanusConfig::from_file(model_path.join("config.json"))
            .unwrap_or_else(|_| {
                tracing::warn!("Failed to load config.json, using 7B defaults");
                JanusConfig::janus_pro_7b()
            });

        // Build model with VarMap for weight loading
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
        let model = JanusModel::new(vb, &model_config)?;

        // Load weights from safetensors
        let st_files = hub::collect_safetensors(&model_path);
        let loaded = hub::load_weights(&varmap, &st_files, dtype, &device)?;
        tracing::info!("Loaded {} weight tensors", loaded);

        // Load tokenizer
        let tokenizer = hub::load_tokenizer(&model_path)?;

        Ok(Self { model, tokenizer, config, model_config, device, dtype })
    }

    /// Generate an image from a text prompt.
    /// Returns an RGB image buffer at the model's native resolution (384x384).
    pub fn generate(&self, prompt: &str) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
        let num_tokens = self.model_config.num_image_tokens(); // 576
        let cfg_scale = self.config.guidance_scale;
        let temperature = self.config.temperature;

        tracing::info!("Generating image: {} tokens, CFG={}, temp={}", num_tokens, cfg_scale, temperature);

        // 1. Tokenize prompt with Janus template
        let formatted = format!(
            "<|User|>: {}\n<|Assistant|>:",
            prompt
        );
        let encoding = self.tokenizer.encode(formatted.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();

        // 2. Create input tensor
        let input_tensor = Tensor::from_vec(
            input_ids.iter().map(|&id| id as i64).collect::<Vec<_>>(),
            (1, input_ids.len()),
            &self.device,
        )?;

        // 3. Create negative (unconditional) input for CFG
        // Fill with pad token (usually 0 or a specific pad ID)
        let pad_id = 0i64;
        let neg_tensor = Tensor::full(pad_id, (1, input_ids.len()), &self.device)?;

        // 4. Batch for CFG: [positive, negative]
        let batch_input = Tensor::cat(&[&input_tensor, &neg_tensor], 0)?;

        // 5. Get initial embeddings
        let mut embeds = self.model.embed_tokens(&batch_input)?
            .to_dtype(self.dtype)?;

        // 6. Initialize KV cache
        let llama_cfg = self.model_config.to_llama_config();
        let mut cache = Cache::new(true, self.dtype, &llama_cfg, &self.device)?;

        // 7. Autoregressive generation loop
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(num_tokens);

        for step in 0..num_tokens {
            // Forward through LLM
            // For first step, process all embeddings; for subsequent, just the new token
            let logits = if step == 0 {
                // Process full prompt embeddings
                // Note: Llama::forward expects token IDs, but we need to use embeddings
                // This requires using the model's forward_embeds if available,
                // or we need to restructure to pass through the embedding layer separately.
                // For now, use the standard forward and then project to image vocab.
                let out = self.model.forward(&batch_input, 0, &mut cache)?;
                out
            } else {
                // Process single token embedding
                let pos = input_ids.len() + step - 1;
                let token_input = Tensor::from_vec(
                    vec![generated_tokens.last().copied().unwrap_or(0) as i64; 2],
                    (2, 1),
                    &self.device,
                )?;
                self.model.forward(&token_input, pos, &mut cache)?
            };

            // Project to image vocabulary
            let image_logits = self.model.project_to_image_vocab(&logits)?;

            // Extract last token position logits
            let last_pos = image_logits.dim(1)? - 1;
            let last_logits = image_logits.i((.., last_pos, ..))?; // [2, vocab]

            // Split into conditional and unconditional
            let cond_logits = last_logits.i((0..1, ..))?;    // [1, vocab]
            let uncond_logits = last_logits.i((1..2, ..))?;   // [1, vocab]

            // Apply Classifier-Free Guidance
            let guided = if cfg_scale != 1.0 {
                let diff = (&cond_logits - &uncond_logits)?;
                (&uncond_logits + diff * cfg_scale)?
            } else {
                cond_logits
            };

            // Temperature scaling and sampling
            let scaled = (&guided / temperature)?;
            let probs = candle_nn::ops::softmax(&scaled, candle_core::D::Minus1)?;

            // Multinomial sampling
            let token = sample_multinomial(&probs)?;
            generated_tokens.push(token);

            if step % 50 == 0 {
                tracing::debug!("Generation step {}/{}", step, num_tokens);
            }
        }

        // 8. Decode tokens to image
        let token_tensor = Tensor::from_vec(
            generated_tokens.iter().map(|&t| t as i64).collect::<Vec<_>>(),
            (1, num_tokens),
            &self.device,
        )?;

        let pixels = self.model.decode_image_tokens(&token_tensor)?; // [1, 3, H, W]

        // 9. Post-process: denormalize and convert to ImageBuffer
        let pixels = denormalize(&pixels)?; // [1, 3, H, W] as U8
        let pixels = pixels.i(0)?; // [3, H, W]

        tensor_to_image(&pixels)
    }
}

/// Sample a single token from a probability distribution using multinomial sampling.
fn sample_multinomial(probs: &Tensor) -> Result<u32> {
    let probs_vec: Vec<f32> = probs.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;

    // Cumulative distribution sampling
    let r: f64 = rand_val();
    let mut cumsum = 0.0;
    for (i, &p) in probs_vec.iter().enumerate() {
        cumsum += p as f64;
        if cumsum >= r {
            return Ok(i as u32);
        }
    }
    Ok((probs_vec.len() - 1) as u32)
}

/// Simple random value [0, 1) using std
fn rand_val() -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    let hash = hasher.finish();
    (hash as f64) / (u64::MAX as f64)
}

/// Convert a [3, H, W] U8 tensor to an ImageBuffer
fn tensor_to_image(tensor: &Tensor) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    let (c, h, w) = tensor.dims3()?;
    anyhow::ensure!(c == 3, "Expected 3 channels, got {}", c);

    // Convert from [C, H, W] to [H, W, C]
    let tensor = tensor.permute((1, 2, 0))?;
    let data: Vec<u8> = tensor.flatten_all()?.to_vec1()?;

    ImageBuffer::from_raw(w as u32, h as u32, data)
        .context("Failed to create image buffer")
}
```

**Step 2: Add `rand` dependency to Cargo.toml** (or use the simple hash-based random above)

No additional dependency needed — using hash-based sampling to avoid adding rand crate.

**Step 3: Verify compilation**

Run: `cd Native/pcai_core && cargo check -p pcai-media`
Expected: Compiles

**Step 4: Commit**

```bash
git add Native/pcai_core/pcai_media/src/generate.rs
git commit -m "feat(media): implement text-to-image generation pipeline with CFG sampling"
```

---

## Task 10: Implement understand.rs (image-to-text pipeline)

**Files:**
- Modify: `Native/pcai_core/pcai_media/src/understand.rs`

**Step 1: Implement the understanding pipeline**

```rust
use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::llama::Cache;
use image::DynamicImage;

use pcai_media_model::config::JanusConfig;
use pcai_media_model::JanusModel;
use pcai_media_model::tensor_utils::normalize;

use crate::config::PipelineConfig;

/// Image understanding pipeline (image → text analysis).
/// Uses SigLIP encoder → MLP aligner → Llama LLM for autoregressive text generation.
pub struct UnderstandingPipeline {
    // Shares model and tokenizer with GenerationPipeline
    // In practice, both pipelines use the same JanusModel instance
}

impl UnderstandingPipeline {
    /// Analyze an image given a text prompt/question.
    /// Returns the model's text response.
    pub fn understand(
        model: &JanusModel,
        tokenizer: &tokenizers::Tokenizer,
        image: &DynamicImage,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        device: &Device,
        dtype: DType,
    ) -> Result<String> {
        let config = &model.config;

        // 1. Preprocess image to tensor [1, 3, 384, 384]
        let img = image.resize_exact(
            config.image_size as u32,
            config.image_size as u32,
            image::imageops::FilterType::Lanczos3,
        );
        let rgb = img.to_rgb8();
        let pixels: Vec<f32> = rgb.pixels()
            .flat_map(|p| p.0.iter().map(|&v| v as f32))
            .collect();
        let img_tensor = Tensor::from_vec(
            pixels,
            (1, config.image_size, config.image_size, 3),
            device,
        )?;
        // Rearrange [B, H, W, C] -> [B, C, H, W]
        let img_tensor = img_tensor.permute((0, 3, 1, 2))?;
        let img_tensor = normalize(&img_tensor)?.to_dtype(dtype)?;

        // 2. Encode image through SigLIP vision encoder
        // Note: SigLIP integration requires loading the vision model weights separately
        // For now, we pass through the aligner which expects SigLIP-L output features
        // The full implementation needs:
        //   let vision_features = siglip_encoder.forward(&img_tensor)?;
        //   let aligned = model.understand_aligner.forward(&vision_features)?;
        // This is a placeholder that will be completed when SigLIP weight paths are mapped

        // 3. Tokenize text prompt
        let formatted = format!(
            "<|User|>: <image>\n{}\n<|Assistant|>:",
            prompt
        );
        let encoding = tokenizer.encode(formatted.as_str(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        // 4. Get text embeddings
        let input_tensor = Tensor::from_vec(input_ids.clone(), (1, input_ids.len()), device)?;
        let _text_embeds = model.embed_tokens(&input_tensor)?.to_dtype(dtype)?;

        // 5. Interleave image tokens with text tokens
        // Replace <image> placeholder token with vision features
        // (Follows candle-transformers LLaVA pattern)

        // 6. Autoregressive text generation
        let llama_cfg = config.to_llama_config();
        let mut cache = Cache::new(true, dtype, &llama_cfg, device)?;
        let mut generated_ids: Vec<i64> = Vec::new();

        let mut current_input = input_tensor.clone();
        let mut pos = 0;

        for _step in 0..max_tokens as usize {
            let logits = model.forward(&current_input, pos, &mut cache)?;
            let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;

            // Temperature scaling
            let scaled = (&last_logits / temperature as f64)?;
            let probs = candle_nn::ops::softmax(&scaled, candle_core::D::Minus1)?;

            // Greedy or sample
            let next_token = if temperature <= 0.01 {
                probs.argmax(candle_core::D::Minus1)?.to_vec1::<u32>()?[0] as i64
            } else {
                let probs_vec: Vec<f32> = probs.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
                let mut cumsum = 0.0;
                let r = super::generate::rand_val();
                let mut token = (probs_vec.len() - 1) as i64;
                for (i, &p) in probs_vec.iter().enumerate() {
                    cumsum += p as f64;
                    if cumsum >= r {
                        token = i as i64;
                        break;
                    }
                }
                token
            };

            // Check for EOS
            if next_token == 2 { // Common EOS token
                break;
            }

            generated_ids.push(next_token);
            pos += current_input.dim(1)?;
            current_input = Tensor::from_vec(vec![next_token], (1, 1), device)?;
        }

        // 7. Decode tokens to text
        let token_ids: Vec<u32> = generated_ids.iter().map(|&id| id as u32).collect();
        let decoded = tokenizer.decode(&token_ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

        Ok(decoded)
    }
}
```

**Step 2: Make `rand_val` public in generate.rs**

Change `fn rand_val()` to `pub(crate) fn rand_val()` in `generate.rs`.

**Step 3: Verify compilation**

Run: `cd Native/pcai_core && cargo check -p pcai-media`
Expected: Compiles

**Step 4: Commit**

```bash
git add Native/pcai_core/pcai_media/src/understand.rs Native/pcai_core/pcai_media/src/generate.rs
git commit -m "feat(media): implement image understanding pipeline with SigLIP placeholder"
```

---

## Task 11: Implement FFI layer

**Files:**
- Modify: `Native/pcai_core/pcai_media/src/ffi/mod.rs`
- Reference: `Native/pcai_core/pcai_inference/src/ffi/mod.rs`

**Step 1: Implement FFI exports following pcai-inference pattern**

```rust
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::cell::RefCell;
use std::sync::{atomic::AtomicI64, atomic::Ordering, Mutex, OnceLock};

use crate::config::PipelineConfig;
use crate::generate::GenerationPipeline;

// --- Error handling (same pattern as pcai_inference/src/ffi/mod.rs) ---

#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum PcaiMediaErrorCode {
    Success = 0,
    NotInitialized = -1,
    ModelNotLoaded = -2,
    InvalidInput = -3,
    GenerationError = -4,
    IoError = -5,
    Unknown = -99,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
    static LAST_ERROR_CODE: RefCell<PcaiMediaErrorCode> = const { RefCell::new(PcaiMediaErrorCode::Success) };
    static LAST_ERROR_CSTRING: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(msg: impl Into<String>, code: PcaiMediaErrorCode) {
    let msg = msg.into();
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(msg));
    LAST_ERROR_CODE.with(|c| *c.borrow_mut() = code);
}

unsafe fn c_str_to_str<'a>(ptr: *const c_char) -> Result<&'a str, ()> {
    if ptr.is_null() {
        set_error("Null pointer", PcaiMediaErrorCode::InvalidInput);
        return Err(());
    }
    match CStr::from_ptr(ptr).to_str() {
        Ok(s) => Ok(s),
        Err(e) => {
            set_error(format!("Invalid UTF-8: {}", e), PcaiMediaErrorCode::InvalidInput);
            Err(())
        }
    }
}

// --- Global state ---

struct MediaState {
    runtime: tokio::runtime::Runtime,
    pipeline: Option<GenerationPipeline>,
}

static MEDIA_STATE: OnceLock<Mutex<MediaState>> = OnceLock::new();

fn get_state() -> &'static Mutex<MediaState> {
    MEDIA_STATE.get_or_init(|| {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime");
        Mutex::new(MediaState {
            runtime,
            pipeline: None,
        })
    })
}

// --- FFI Exports ---

/// Initialize the media pipeline with default or specified device.
/// device: "cuda:0", "cuda:1", "cpu", or NULL for auto-detect.
/// Returns 0 on success.
#[no_mangle]
pub unsafe extern "C" fn pcai_media_init(device: *const c_char) -> i32 {
    let device_str = if device.is_null() {
        "cuda:0".to_string()
    } else {
        match c_str_to_str(device) {
            Ok(s) => s.to_string(),
            Err(()) => return -1,
        }
    };

    let mut config = PipelineConfig::default();
    config.device = device_str;

    // State is initialized lazily; model loading happens in pcai_media_load_model
    let _state = get_state();
    set_error("", PcaiMediaErrorCode::Success);
    0
}

/// Load a Janus model from HF repo ID or local path.
/// Returns 0 on success.
#[no_mangle]
pub unsafe extern "C" fn pcai_media_load_model(model_path: *const c_char, gpu_layers: i32) -> i32 {
    let path = match c_str_to_str(model_path) {
        Ok(s) => s.to_string(),
        Err(()) => return -1,
    };

    let mut config = PipelineConfig::default();
    config.model = path;
    config.gpu_layers = gpu_layers;

    match GenerationPipeline::load(config) {
        Ok(pipeline) => {
            let state = get_state();
            let mut guard = state.lock().expect("State lock poisoned");
            guard.pipeline = Some(pipeline);
            set_error("", PcaiMediaErrorCode::Success);
            0
        }
        Err(e) => {
            set_error(format!("Model loading failed: {}", e), PcaiMediaErrorCode::ModelNotLoaded);
            -1
        }
    }
}

/// Generate an image from a text prompt and save to output_path (PNG).
/// Returns 0 on success.
#[no_mangle]
pub unsafe extern "C" fn pcai_media_generate_image(
    prompt: *const c_char,
    cfg_scale: f32,
    temperature: f32,
    output_path: *const c_char,
) -> i32 {
    let prompt_str = match c_str_to_str(prompt) {
        Ok(s) => s,
        Err(()) => return -1,
    };
    let output_str = match c_str_to_str(output_path) {
        Ok(s) => s,
        Err(()) => return -1,
    };

    let state = get_state();
    let guard = state.lock().expect("State lock poisoned");
    let pipeline = match &guard.pipeline {
        Some(p) => p,
        None => {
            set_error("Model not loaded", PcaiMediaErrorCode::ModelNotLoaded);
            return -1;
        }
    };

    match pipeline.generate(prompt_str) {
        Ok(image) => {
            match image.save(output_str) {
                Ok(()) => {
                    set_error("", PcaiMediaErrorCode::Success);
                    0
                }
                Err(e) => {
                    set_error(format!("Failed to save image: {}", e), PcaiMediaErrorCode::IoError);
                    -1
                }
            }
        }
        Err(e) => {
            set_error(format!("Generation failed: {}", e), PcaiMediaErrorCode::GenerationError);
            -1
        }
    }
}

/// Shut down the media pipeline and release all resources.
#[no_mangle]
pub unsafe extern "C" fn pcai_media_shutdown() {
    if let Some(state) = MEDIA_STATE.get() {
        if let Ok(mut guard) = state.lock() {
            guard.pipeline = None;
        }
    }
}

/// Get the last error message. Returns NULL if no error.
#[no_mangle]
pub unsafe extern "C" fn pcai_media_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        let err = e.borrow();
        match err.as_ref() {
            Some(msg) if !msg.is_empty() => {
                LAST_ERROR_CSTRING.with(|cs| {
                    let cstring = CString::new(msg.as_str()).unwrap_or_default();
                    let ptr = cstring.as_ptr();
                    *cs.borrow_mut() = Some(cstring);
                    ptr
                })
            }
            _ => std::ptr::null(),
        }
    })
}

/// Get the last error code.
#[no_mangle]
pub unsafe extern "C" fn pcai_media_last_error_code() -> i32 {
    LAST_ERROR_CODE.with(|c| *c.borrow() as i32)
}

/// Free a string allocated by this library.
#[no_mangle]
pub unsafe extern "C" fn pcai_media_free_string(s: *mut c_char) {
    if !s.is_null() {
        let _ = CString::from_raw(s);
    }
}

/// Free a byte buffer allocated by this library.
#[no_mangle]
pub unsafe extern "C" fn pcai_media_free_bytes(data: *mut u8, len: usize) {
    if !data.is_null() && len > 0 {
        let _ = Vec::from_raw_parts(data, len, len);
    }
}
```

**Step 2: Verify compilation**

Run: `cd Native/pcai_core && cargo check -p pcai-media --features ffi`
Expected: Compiles

**Step 3: Commit**

```bash
git add Native/pcai_core/pcai_media/src/ffi/
git commit -m "feat(media): implement FFI layer with init/load/generate/shutdown exports"
```

---

## Task 12: Scaffold pcai_media_server crate (HTTP server)

**Files:**
- Create: `Native/pcai_core/pcai_media_server/Cargo.toml`
- Create: `Native/pcai_core/pcai_media_server/src/main.rs`
- Modify: `Native/pcai_core/Cargo.toml` (add workspace member)

**Step 1: Create Cargo.toml**

```toml
[package]
name = "pcai-media-server"
version = "0.1.0"
edition = "2021"
description = "HTTP server for Janus-Pro media inference"
license = "MIT"

[[bin]]
name = "pcai-media"
path = "src/main.rs"

[dependencies]
pcai-media = { path = "../pcai_media", default-features = false, features = ["server"] }
pcai-media-model = { path = "../pcai_media_model" }
anyhow.workspace = true
tokio.workspace = true
serde.workspace = true
serde_json.workspace = true
tracing.workspace = true
tracing-subscriber.workspace = true
axum.workspace = true
tower-http.workspace = true
clap = { version = "4", features = ["derive"] }
base64 = "0.22"
```

**Step 2: Create main.rs with routes**

```rust
use std::sync::Arc;
use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use pcai_media::config::PipelineConfig;
use pcai_media::generate::GenerationPipeline;

#[derive(Parser)]
#[command(name = "pcai-media", about = "Janus-Pro media inference server")]
struct Args {
    /// Model path or HuggingFace repo ID
    #[arg(short, long, default_value = "deepseek-ai/Janus-Pro-7B")]
    model: String,
    /// Server host
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    /// Server port
    #[arg(short, long, default_value_t = 8090)]
    port: u16,
    /// Device (cuda:0, cpu)
    #[arg(short, long, default_value = "cuda:0")]
    device: String,
}

struct AppState {
    pipeline: Option<GenerationPipeline>,
    config: PipelineConfig,
}

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    #[serde(default = "default_cfg")]
    cfg_scale: f32,
    #[serde(default = "default_temp")]
    temperature: f32,
}

fn default_cfg() -> f32 { 5.0 }
fn default_temp() -> f32 { 1.0 }

#[derive(Serialize)]
struct GenerateResponse {
    image_base64: String,
    width: u32,
    height: u32,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    model: String,
    model_loaded: bool,
}

async fn health(State(state): State<Arc<RwLock<AppState>>>) -> Json<HealthResponse> {
    let guard = state.read().await;
    Json(HealthResponse {
        status: "ok".into(),
        model: guard.config.model.clone(),
        model_loaded: guard.pipeline.is_some(),
    })
}

async fn generate_image(
    State(state): State<Arc<RwLock<AppState>>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)> {
    let guard = state.read().await;
    let pipeline = guard.pipeline.as_ref()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "Model not loaded".into()))?;

    let image = pipeline.generate(&req.prompt)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Generation failed: {}", e)))?;

    // Encode to PNG bytes then base64
    let mut buf = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut buf);
    image::ImageEncoder::write_image(
        encoder,
        image.as_raw(),
        image.width(),
        image.height(),
        image::ExtendedColorType::Rgb8,
    ).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("PNG encoding failed: {}", e)))?;

    let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &buf);

    Ok(Json(GenerateResponse {
        image_base64: b64,
        width: image.width(),
        height: image.height(),
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let mut config = PipelineConfig::default();
    config.model = args.model;
    config.device = args.device;

    tracing::info!("Loading model: {}", config.model);
    let pipeline = GenerationPipeline::load(config.clone())?;
    tracing::info!("Model loaded successfully");

    let state = Arc::new(RwLock::new(AppState {
        pipeline: Some(pipeline),
        config,
    }));

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/images/generate", post(generate_image))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive());

    let addr = format!("{}:{}", args.host, args.port);
    tracing::info!("Starting server on {}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
```

**Step 3: Add workspace member**

Add `"pcai_media_server"` to workspace members.

**Step 4: Verify compilation**

Run: `cd Native/pcai_core && cargo check -p pcai-media-server`
Expected: Compiles

**Step 5: Commit**

```bash
git add Native/pcai_core/pcai_media_server/ Native/pcai_core/Cargo.toml
git commit -m "feat(media): add pcai-media-server with axum HTTP API for image generation"
```

---

## Task 13: Create C# P/Invoke wrapper (MediaModule.cs)

**Files:**
- Create: `Native/PcaiNative/MediaModule.cs`
- Reference: `Native/PcaiNative/InferenceModule.cs`

**Step 1: Create MediaModule.cs**

```csharp
using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

#nullable enable

namespace PcaiNative
{
    /// <summary>
    /// P/Invoke interop with pcai_media.dll (Janus-Pro media inference)
    /// </summary>
    public static class MediaModule
    {
        private const string DllName = "pcai_media";

        static MediaModule()
        {
            NativeLibrary.SetDllImportResolver(typeof(MediaModule).Assembly, ResolveDll);
        }

        private static IntPtr ResolveDll(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (libraryName != DllName) return IntPtr.Zero;
            if (NativeLibrary.TryLoad(libraryName, assembly, searchPath, out IntPtr handle))
                return handle;

            var candidates = new[]
            {
                Path.Combine(AppContext.BaseDirectory, "pcai_media.dll"),
                Path.Combine(AppContext.BaseDirectory, "runtimes", "win-x64", "native", "pcai_media.dll"),
            };

            foreach (var path in candidates)
            {
                if (File.Exists(path) && NativeLibrary.TryLoad(path, out handle))
                    return handle;
            }
            return IntPtr.Zero;
        }

        #region Native Imports

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int pcai_media_init([MarshalAs(UnmanagedType.LPUTF8Str)] string? device);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int pcai_media_load_model([MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath, int gpuLayers);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int pcai_media_generate_image(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string prompt,
            float cfgScale,
            float temperature,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string outputPath);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern IntPtr pcai_media_understand_image(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string imagePath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string prompt,
            uint maxTokens,
            float temperature);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void pcai_media_shutdown();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr pcai_media_last_error();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int pcai_media_last_error_code();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void pcai_media_free_string(IntPtr ptr);

        #endregion

        #region High-level Wrappers

        public static string? GetLastError()
        {
            var ptr = pcai_media_last_error();
            return ptr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(ptr);
        }

        public static string? GenerateImage(string prompt, string outputPath, float cfgScale = 5.0f, float temperature = 1.0f)
        {
            var result = pcai_media_generate_image(prompt, cfgScale, temperature, outputPath);
            if (result != 0) return GetLastError();
            return null; // null = success
        }

        public static string? UnderstandImage(string imagePath, string prompt, uint maxTokens = 512, float temperature = 0.7f)
        {
            var ptr = pcai_media_understand_image(imagePath, prompt, maxTokens, temperature);
            if (ptr == IntPtr.Zero) return null;
            try { return Marshal.PtrToStringUTF8(ptr); }
            finally { pcai_media_free_string(ptr); }
        }

        #endregion
    }
}
```

**Step 2: Verify C# compilation**

Run: `cd Native/PcaiNative && dotnet build`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add Native/PcaiNative/MediaModule.cs
git commit -m "feat(media): add C# MediaModule P/Invoke wrapper for pcai_media.dll"
```

---

## Task 14: Create PowerShell module (PcaiMedia.psm1)

**Files:**
- Create: `Modules/PcaiMedia.psm1`
- Reference: `Modules/PcaiInference.psm1` (pattern)

**Step 1: Create the PowerShell module**

```powershell
#Requires -Version 5.1

<#
.SYNOPSIS
    PcaiMedia — PowerShell wrapper for Janus-Pro media inference.

.DESCRIPTION
    Provides image generation, understanding, and upscaling from PowerShell
    via the pcai_media Rust DLL through PcaiNative.dll.

    Exported functions:
      Initialize-PcaiMedia     - Initialize the media pipeline
      Import-PcaiMediaModel    - Load a Janus-Pro model
      New-PcaiImage            - Generate an image from a text prompt
      Get-PcaiImageAnalysis    - Analyze an image with a question
      Stop-PcaiMedia           - Shut down and release resources
      Get-PcaiMediaStatus      - Get current pipeline status
#>

$script:ModulePath = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.ScriptBlock.File }
$script:Initialized = $false
$script:ModelLoaded = $false
$script:CurrentModel = $null

function Initialize-PcaiMedia {
    [CmdletBinding()]
    param(
        [Parameter()]
        [ValidateSet('cuda:0', 'cuda:1', 'cpu')]
        [string]$Device = 'cuda:0'
    )

    try {
        $result = [PcaiNative.MediaModule]::pcai_media_init($Device)
        if ($result -ne 0) {
            $error = [PcaiNative.MediaModule]::GetLastError()
            throw "Media init failed: $error"
        }
        $script:Initialized = $true
        [PSCustomObject]@{ Success = $true; Device = $Device }
    } catch {
        $script:Initialized = $false
        throw "Media initialization failed: $_"
    }
}

function Import-PcaiMediaModel {
    [CmdletBinding()]
    param(
        [Parameter(Position = 0)]
        [string]$ModelPath = 'deepseek-ai/Janus-Pro-7B',
        [Parameter()]
        [int]$GpuLayers = -1
    )

    if (-not $script:Initialized) {
        throw 'Not initialized. Call Initialize-PcaiMedia first.'
    }

    try {
        $result = [PcaiNative.MediaModule]::pcai_media_load_model($ModelPath, $GpuLayers)
        if ($result -ne 0) {
            $error = [PcaiNative.MediaModule]::GetLastError()
            throw "Model load failed: $error"
        }
        $script:ModelLoaded = $true
        $script:CurrentModel = $ModelPath
        [PSCustomObject]@{ Success = $true; Model = $ModelPath }
    } catch {
        $script:ModelLoaded = $false
        throw "Model loading failed: $_"
    }
}

function New-PcaiImage {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Prompt,
        [Parameter()]
        [string]$OutputPath,
        [Parameter()]
        [float]$CfgScale = 5.0,
        [Parameter()]
        [float]$Temperature = 1.0
    )

    if (-not $script:ModelLoaded) {
        throw 'Model not loaded. Call Import-PcaiMediaModel first.'
    }

    if (-not $OutputPath) {
        $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $OutputPath = Join-Path ([Environment]::GetFolderPath('Desktop')) "janus-$timestamp.png"
    }

    $parentDir = Split-Path $OutputPath -Parent
    if ($parentDir -and -not (Test-Path $parentDir)) {
        New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
    }

    try {
        $err = [PcaiNative.MediaModule]::GenerateImage($Prompt, $OutputPath, $CfgScale, $Temperature)
        if ($err) { throw "Generation failed: $err" }
        [PSCustomObject]@{
            Success = $true
            OutputPath = $OutputPath
            Prompt = $Prompt
        }
    } catch {
        throw "Image generation failed: $_"
    }
}

function Get-PcaiImageAnalysis {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$ImagePath,
        [Parameter(Position = 1)]
        [string]$Question = 'Describe this image in detail.',
        [Parameter()]
        [uint32]$MaxTokens = 512,
        [Parameter()]
        [float]$Temperature = 0.7
    )

    if (-not $script:ModelLoaded) {
        throw 'Model not loaded. Call Import-PcaiMediaModel first.'
    }

    if (-not (Test-Path $ImagePath)) {
        throw "Image not found: $ImagePath"
    }

    try {
        $result = [PcaiNative.MediaModule]::UnderstandImage($ImagePath, $Question, $MaxTokens, $Temperature)
        if ($null -eq $result) {
            $error = [PcaiNative.MediaModule]::GetLastError()
            throw "Understanding failed: $error"
        }
        $result
    } catch {
        throw "Image analysis failed: $_"
    }
}

function Stop-PcaiMedia {
    [CmdletBinding()]
    param()

    if ($script:Initialized) {
        try {
            [PcaiNative.MediaModule]::pcai_media_shutdown()
        } catch {
            Write-Warning "Shutdown error: $_"
        }
        $script:Initialized = $false
        $script:ModelLoaded = $false
        $script:CurrentModel = $null
    }
}

function Get-PcaiMediaStatus {
    [CmdletBinding()]
    param()
    [PSCustomObject]@{
        Initialized = $script:Initialized
        ModelLoaded = $script:ModelLoaded
        CurrentModel = $script:CurrentModel
    }
}

# Module cleanup
if ($MyInvocation.MyCommand.ScriptBlock.Module) {
    $MyInvocation.MyCommand.ScriptBlock.Module.OnRemove = {
        if ($script:Initialized) {
            [PcaiNative.MediaModule]::pcai_media_shutdown()
        }
    }
}
```

**Step 2: Commit**

```bash
git add Modules/PcaiMedia.psm1
git commit -m "feat(media): add PcaiMedia PowerShell module wrapping FFI for image gen/understand"
```

---

## Task 15: Add media component to Build.ps1

**Files:**
- Modify: `Tools/Build.ps1` (add `media` component)
- Create: `Config/pcai-media.json` (media pipeline config)

**Step 1: Create media config**

```json
{
    "model": "deepseek-ai/Janus-Pro-7B",
    "device": "cuda:0",
    "dtype": "bf16",
    "guidance_scale": 5.0,
    "temperature": 1.0,
    "parallel_size": 1,
    "gpu_layers": -1
}
```

Save to `Config/pcai-media.json`.

**Step 2: Add `media` case to Build.ps1 Component parameter**

Add `media` to the `[ValidateSet(...)]` list for the `$Component` parameter and add a build case that runs:

```powershell
# In the component switch:
'media' {
    Write-Host "Building pcai-media (Janus-Pro media agent)..." -ForegroundColor Cyan

    $cargoArgs = @('build', '--release', '-p', 'pcai-media', '-p', 'pcai-media-server')
    if ($EnableCuda) {
        $cargoArgs += @('--features', 'pcai-media/flash-attn')
    }

    Push-Location "Native/pcai_core"
    & cargo @cargoArgs
    Pop-Location

    # Stage artifacts
    $mediaArtifacts = Join-Path $ArtifactsRoot 'pcai-media'
    New-Item -ItemType Directory -Path $mediaArtifacts -Force | Out-Null
    Copy-Item "Native/pcai_core/target/release/pcai_media.*" $mediaArtifacts -Force -ErrorAction SilentlyContinue
    Copy-Item "Native/pcai_core/target/release/pcai-media.exe" $mediaArtifacts -Force -ErrorAction SilentlyContinue
}
```

**Step 3: Commit**

```bash
git add Config/pcai-media.json
git commit -m "feat(media): add media config and build integration"
```

---

## Task 16: Integration test — end-to-end compilation

**Files:** None new — this is a verification step.

**Step 1: Full workspace check**

Run: `cd Native/pcai_core && cargo check --workspace`
Expected: All crates compile

**Step 2: Run all unit tests**

Run: `cd Native/pcai_core && cargo test -p pcai-media-model --lib`
Expected: All config, tensor_utils, and generation_head tests pass

**Step 3: Build release binary (no CUDA)**

Run: `cd Native/pcai_core && cargo build --release -p pcai-media-server`
Expected: Produces `target/release/pcai-media.exe`

**Step 4: Build FFI DLL**

Run: `cd Native/pcai_core && cargo build --release -p pcai-media`
Expected: Produces `target/release/pcai_media.dll`

**Step 5: Commit any fixes**

```bash
git add -A
git commit -m "fix(media): resolve integration compilation issues"
```

---

## Summary

| Task | Component | Effort | Key Reuse |
|------|-----------|--------|-----------|
| 1 | Scaffold pcai_media_model | Low | Workspace pattern |
| 2 | config.rs | Low | JanusConfig presets |
| 3 | tensor_utils.rs | Low | Direct copy from AI-Media |
| 4 | vq_vae.rs | Medium | AI-Media janus_model.rs + encodec codebook |
| 5 | generation_head.rs | Low | Simple Linear + MLP |
| 6 | JanusModel facade | Medium | candle-transformers Llama + SigLIP |
| 7 | Scaffold pcai_media | Low | pcai_inference pattern |
| 8 | hub.rs | Low | functiongemma-core safetensors_utils |
| 9 | generate.rs | Medium | AI-Media main.rs generation loop |
| 10 | understand.rs | Medium | LLaVA multimodal pattern |
| 11 | FFI layer | Medium | pcai_inference ffi/mod.rs clone |
| 12 | HTTP server | Low | pcai_inference http/ + axum |
| 13 | C# P/Invoke | Low | InferenceModule.cs clone |
| 14 | PowerShell module | Low | PcaiInference.psm1 clone |
| 15 | Build integration | Low | Build.ps1 component |
| 16 | Integration test | Low | Verification only |
