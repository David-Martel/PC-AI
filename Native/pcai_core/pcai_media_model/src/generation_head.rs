//! Generation head and MLP aligner for Janus-Pro.
//!
//! This module provides two components used when switching the Janus-Pro
//! language model into image-generation mode:
//!
//! - [`GenerationHead`]: a single linear projection from the LLM hidden
//!   dimension onto the image vocabulary (16 384 codebook tokens).
//! - [`MlpAligner`]: a two-layer MLP that bridges two representation spaces.
//!   Used both to map image-token embeddings into LLM input space (generation)
//!   and to map SigLIP visual features into LLM input space (understanding).
//!
//! # Example (offline smoke-test — no weights on disk)
//!
//! ```rust,no_run
//! use candle_core::{Device, DType, Tensor, Module};
//! use candle_nn::VarMap;
//! use pcai_media_model::generation_head::{GenerationHead, MlpAligner};
//!
//! let dev  = Device::Cpu;
//! let vm   = VarMap::new();
//! let vb   = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev);
//!
//! // GenerationHead: [B, S, 4096] → [B, S, 16384]
//! let head = GenerationHead::new(vb.pp("gen_head"), 4096, 16384).unwrap();
//! let h    = Tensor::zeros((1_usize, 10_usize, 4096_usize), DType::F32, &dev).unwrap();
//! let logits = head.forward(&h).unwrap();
//! assert_eq!(logits.dims(), &[1, 10, 16384]);
//!
//! // MlpAligner: [B, S, 1024] → [B, S, 4096]
//! let aligner = MlpAligner::new(vb.pp("aligner"), 1024, 4096).unwrap();
//! let feat    = Tensor::zeros((1_usize, 576_usize, 1024_usize), DType::F32, &dev).unwrap();
//! let proj    = aligner.forward(&feat).unwrap();
//! assert_eq!(proj.dims(), &[1, 576, 4096]);
//! ```

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

// ---------------------------------------------------------------------------
// GenerationHead
// ---------------------------------------------------------------------------

/// Linear projection from LLM hidden states to image-vocabulary logits.
///
/// Applied at every generation step to convert the last hidden state
/// `[B, S, hidden_size]` into logits over the VQ codebook
/// `[B, S, image_vocab_size]`.  No bias term (consistent with the reference
/// implementation).
///
/// # Weight paths (relative to the `VarBuilder` root)
///
/// | Parameter | Path |
/// |-----------|------|
/// | weight matrix | `weight` |
/// | bias (absent) | — |
///
/// # Example
///
/// ```rust,no_run
/// use candle_core::{Device, DType, Tensor, Module};
/// use candle_nn::VarMap;
/// use pcai_media_model::generation_head::GenerationHead;
///
/// let dev  = Device::Cpu;
/// let vm   = VarMap::new();
/// let vb   = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev);
/// let head = GenerationHead::new(vb.pp("h"), 4096, 16384).unwrap();
/// let x    = Tensor::zeros((1_usize, 5_usize, 4096_usize), DType::F32, &dev).unwrap();
/// assert_eq!(head.forward(&x).unwrap().dims(), &[1, 5, 16384]);
/// ```
pub struct GenerationHead {
    proj: Linear,
}

impl GenerationHead {
    /// Constructs a [`GenerationHead`] that maps `hidden_size` → `image_vocab_size`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the weight tensor cannot be created or loaded.
    pub fn new(vb: VarBuilder, hidden_size: usize, image_vocab_size: usize) -> Result<Self> {
        // linear_no_bias variant: weight only, no bias.
        let proj = candle_nn::linear_no_bias(hidden_size, image_vocab_size, vb)?;
        Ok(Self { proj })
    }
}

impl Module for GenerationHead {
    /// Projects hidden states to image-vocabulary logits.
    ///
    /// # Arguments
    ///
    /// * `xs` — `[B, S, hidden_size]`
    ///
    /// # Returns
    ///
    /// `[B, S, image_vocab_size]`
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.proj.forward(xs)
    }
}

// ---------------------------------------------------------------------------
// MlpAligner
// ---------------------------------------------------------------------------

/// Two-layer MLP aligner: `Linear → GELU → Linear`.
///
/// Maps between two representation spaces.  Janus-Pro uses this in two roles:
///
/// 1. **`gen_aligner`**: maps image-token embeddings (same `hidden_size`) into
///    the LLM input space before the transformer blocks.
/// 2. **`understand_aligner`** (also called `aligner` in the weight file):
///    maps SigLIP visual features (1 024-dim) into the LLM hidden space.
///
/// Weight indexing follows Python's `nn.Sequential` convention where the GELU
/// activation occupies index 1 and therefore has no weight:
///
/// | Index | Component | Weight path |
/// |-------|-----------|-------------|
/// | 0 | `Linear(in, out)` | `0` |
/// | 1 | `GELU` | (no weight) |
/// | 2 | `Linear(out, out)` | `2` |
///
/// # Example
///
/// ```rust,no_run
/// use candle_core::{Device, DType, Tensor, Module};
/// use candle_nn::VarMap;
/// use pcai_media_model::generation_head::MlpAligner;
///
/// let dev     = Device::Cpu;
/// let vm      = VarMap::new();
/// let vb      = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &dev);
/// let aligner = MlpAligner::new(vb.pp("a"), 1024, 4096).unwrap();
/// let x       = Tensor::zeros((1_usize, 576_usize, 1024_usize), DType::F32, &dev).unwrap();
/// assert_eq!(aligner.forward(&x).unwrap().dims(), &[1, 576, 4096]);
/// ```
pub struct MlpAligner {
    /// First linear: `in_dim → out_dim`.  Weight path: `"0"`.
    fc1: Linear,
    /// Second linear: `out_dim → out_dim`.  Weight path: `"2"`.
    fc2: Linear,
}

impl MlpAligner {
    /// Constructs an [`MlpAligner`] mapping `in_dim` → `out_dim`.
    ///
    /// Both linear layers include bias terms (matching the reference weights).
    ///
    /// # Errors
    ///
    /// Returns a candle error if any weight tensor cannot be created or loaded.
    pub fn new(vb: VarBuilder, in_dim: usize, out_dim: usize) -> Result<Self> {
        let fc1 = linear(in_dim, out_dim, vb.pp("0"))?;
        let fc2 = linear(out_dim, out_dim, vb.pp("2"))?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for MlpAligner {
    /// Applies `Linear(in_dim, out_dim) → GELU → Linear(out_dim, out_dim)`.
    ///
    /// # Arguments
    ///
    /// * `xs` — input tensor of shape `[B, S, in_dim]`
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[B, S, out_dim]`
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(xs)?;
        let h = h.gelu()?;
        self.fc2.forward(&h)
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
        candle_nn::VarBuilder::from_varmap(vm, DType::F32, &Device::Cpu)
    }

    /// GenerationHead: [1, 10, 4096] → [1, 10, 16384]
    #[test]
    fn test_generation_head_shape() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = cpu_vb(&vm);

        let head = GenerationHead::new(vb.pp("head"), 4096, 16384).unwrap();
        let x = Tensor::zeros((1_usize, 10_usize, 4096_usize), DType::F32, &dev).unwrap();
        let out = head.forward(&x).unwrap();
        assert_eq!(
            out.dims(),
            &[1, 10, 16384],
            "GenerationHead output shape mismatch"
        );
    }

    /// MlpAligner: [1, 576, 1024] → [1, 576, 4096]
    #[test]
    fn test_mlp_aligner_shape() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = cpu_vb(&vm);

        let aligner = MlpAligner::new(vb.pp("aligner"), 1024, 4096).unwrap();
        let x = Tensor::zeros((1_usize, 576_usize, 1024_usize), DType::F32, &dev).unwrap();
        let out = aligner.forward(&x).unwrap();
        assert_eq!(
            out.dims(),
            &[1, 576, 4096],
            "MlpAligner output shape mismatch"
        );
    }

    /// Verify that aligner with same in/out dimension works (used by gen_aligner).
    #[test]
    fn test_mlp_aligner_identity_dims() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = cpu_vb(&vm);

        let aligner = MlpAligner::new(vb.pp("a"), 4096, 4096).unwrap();
        let x = Tensor::zeros((2_usize, 8_usize, 4096_usize), DType::F32, &dev).unwrap();
        let out = aligner.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 8, 4096]);
    }

    /// GenerationHead with no-bias should produce finite (non-NaN) output on random input.
    #[test]
    fn test_generation_head_no_nan() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let vb = cpu_vb(&vm);

        let head = GenerationHead::new(vb.pp("h"), 64, 128).unwrap();
        let x = Tensor::rand(0.0_f32, 1.0_f32, (1_usize, 4_usize, 64_usize), &dev).unwrap();
        let out = head.forward(&x).unwrap();
        let flat = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in flat {
            assert!(v.is_finite(), "NaN/Inf in GenerationHead output");
        }
    }
}
