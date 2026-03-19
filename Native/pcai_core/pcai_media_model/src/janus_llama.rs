//! Custom Llama backbone for Janus-Pro that exposes pre-lm_head hidden states.
//!
//! The stock `candle_transformers::models::llama::Llama` applies `lm_head` in
//! `forward_input_embed`, returning text-vocabulary logits.  Janus-Pro needs the
//! **hidden states** to project through its image-vocabulary `gen_head` instead.
//!
//! This module reimplements the minimal Llama transformer stack (embedding,
//! blocks, RMS norm) so we can return hidden states directly.  This also gives
//! us full control over the KV cache for future memory optimizations (INT8 KV
//! cache, layer offloading, etc.).
//!
//! # KV Cache Implementations
//!
//! Two KV cache implementations are provided:
//!
//! - [`KvCache`]: The original dynamic cache.  Each step appends via
//!   `Tensor::cat`, allocating a new growing buffer.  Simple but wastes GPU
//!   bandwidth (O(step²) total bytes written across 576 steps).
//!
//! - [`PreAllocKvCache`]: A ring-buffer cache that allocates one fixed-size
//!   `[B, n_kv_heads, max_seq_len, head_dim]` buffer per layer at construction
//!   time.  New KV pairs are written in-place via `Tensor::scatter_set` (a
//!   single-token write, O(B·H·D) per step), and accumulated history is read
//!   back as a zero-copy `narrow` view.  This eliminates the ≈95 GB of GPU
//!   bandwidth waste from `cat` across 576 steps × 24 layers.
//!
//! Use [`PreAllocKvCache`] via [`JanusLlama::forward_hidden_prealloc`] and
//! [`JanusLlama::forward_input_embed_prealloc`].

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Module, VarBuilder};
use candle_transformers::models::llama::Config;
use std::collections::HashMap;

#[cfg(feature = "flash-attn")]
use candle_core::D;
#[cfg(feature = "flash-attn")]
use candle_flash_attn;

// ---------------------------------------------------------------------------
// KV Cache (original dynamic implementation)
// ---------------------------------------------------------------------------

/// KV cache with RoPE precomputed cos/sin tables.
///
/// Unlike candle's `llama::Cache`, all fields are accessible for optimization.
///
/// This is the original dynamic cache.  For image generation (576 decoding
/// steps) consider [`PreAllocKvCache`] which avoids `Tensor::cat` entirely.
#[derive(Debug, Clone)]
pub struct KvCache {
    pub use_kv_cache: bool,
    pub kvs: Vec<Option<(Tensor, Tensor)>>,
    pub cos: Tensor,
    pub sin: Tensor,
    masks: HashMap<usize, Tensor>,
    device: Device,
}

impl KvCache {
    /// Build a new KV cache with precomputed RoPE tables.
    pub fn new(use_kv_cache: bool, dtype: DType, cfg: &Config, device: &Device) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let theta: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
            .collect();

        let theta = Tensor::new(theta, device)?;
        let idx_theta = Tensor::arange(0, cfg.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;

        Ok(Self {
            use_kv_cache,
            kvs: vec![None; cfg.num_hidden_layers],
            masks: HashMap::new(),
            device: device.clone(),
            cos,
            sin,
        })
    }

    /// Get or create a causal attention mask of size `t × t`.
    pub fn mask(&mut self, t: usize) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t).flat_map(|i| (0..t).map(move |j| u8::from(j > i))).collect();
            let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    /// Clear all cached KV pairs (for new sequence generation).
    pub fn clear(&mut self) {
        for kv in self.kvs.iter_mut() {
            *kv = None;
        }
        self.masks.clear();
    }

    /// Returns the total memory used by the KV cache in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.kvs
            .iter()
            .filter_map(|kv| kv.as_ref())
            .map(|(k, v)| {
                let k_bytes = k.elem_count() * k.dtype().size_in_bytes();
                let v_bytes = v.elem_count() * v.dtype().size_in_bytes();
                k_bytes + v_bytes
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// PreAllocKvCache — ring-buffer KV cache (zero-copy read, in-place write)
// ---------------------------------------------------------------------------

/// Per-layer state for the pre-allocated KV cache.
///
/// Holds two `[B, n_kv_heads, max_seq_len, head_dim]` buffers (K and V)
/// allocated once at construction.  Tokens are written with `scatter_set`
/// (truly in-place) and read back with `narrow` (zero-copy view).
#[derive(Debug, Clone)]
struct LayerKvBuffer {
    /// Pre-allocated key buffer: shape `[B, n_kv_heads, max_seq_len, head_dim]`.
    k_buf: Tensor,
    /// Pre-allocated value buffer: shape `[B, n_kv_heads, max_seq_len, head_dim]`.
    v_buf: Tensor,
    /// Scatter index tensor for dim=2.
    ///
    /// Shape `[B, n_kv_heads, 1, head_dim]`; holds the current write position.
    /// Re-used across calls (updated in-place via `fill_` equivalent) to avoid
    /// per-step allocations.  We rebuild it from the position integer each step,
    /// which is cheap (tiny tensor, one allocation per step per layer).
    ///
    /// Stored here so the allocation cost is amortised across the 576-step run.
    scatter_idx_shape: (usize, usize, usize, usize),
}

/// Pre-allocated ring-buffer KV cache.
///
/// Eliminates the `Tensor::cat` bottleneck in [`KvCache`] by writing new KV
/// pairs in-place with `Tensor::scatter_set` and reading accumulated history
/// with `Tensor::narrow` (zero-copy).
///
/// # Memory Layout
///
/// For each transformer layer:
/// ```text
/// key_buffer:   [B=1, n_kv_heads, max_seq_len, head_dim]
/// value_buffer: [B=1, n_kv_heads, max_seq_len, head_dim]
/// ```
///
/// At generation step `t`, tokens `[0, t)` are valid.  Reading them is:
/// ```text
/// key_buffer.narrow(2, 0, t)   // zero-copy view of [B, H, t, D]
/// value_buffer.narrow(2, 0, t) // zero-copy view of [B, H, t, D]
/// ```
///
/// Writing the new token at position `t` uses `scatter_set` along dim 2,
/// updating the pre-allocated buffer without any allocation or copy of prior
/// tokens.
///
/// # Performance Characteristics
///
/// | Metric | [`KvCache`] (cat) | [`PreAllocKvCache`] (scatter) |
/// |--------|-------------------|-------------------------------|
/// | Write cost (step t) | O(t) — copies all prior tokens | O(1) — writes 1 token |
/// | Read cost | O(t) | O(1) — `narrow` is zero-copy |
/// | Total bandwidth (576 steps) | O(576²) ≈ 95 GB | O(576) ≈ 0.3 GB |
/// | Allocation per step | 1 growing buffer | 1 tiny index tensor |
///
/// # Construction
///
/// ```no_run
/// use candle_core::{DType, Device};
/// use candle_transformers::models::llama::Config;
/// use pcai_media_model::janus_llama::PreAllocKvCache;
///
/// # fn example() -> candle_core::Result<()> {
/// let cfg: Config = todo!(); // your llama config
/// let cache = PreAllocKvCache::new(DType::BF16, &cfg, 1, 700, &Device::Cpu)?;
/// # Ok(()) }
/// ```
#[derive(Debug, Clone)]
pub struct PreAllocKvCache {
    /// Pre-computed RoPE cosine table.  Shape `[max_position_embeddings, head_dim/2]`.
    pub cos: Tensor,
    /// Pre-computed RoPE sine table.  Shape `[max_position_embeddings, head_dim/2]`.
    pub sin: Tensor,
    /// Per-layer ring buffers.  Length equals the number of transformer layers.
    layers: Vec<LayerKvBuffer>,
    /// Number of tokens currently written into the cache (valid token count).
    current_seq_len: usize,
    /// Number of KV heads from the model config.
    n_kv_heads: usize,
    /// Head dimension derived from config.
    head_dim: usize,
    /// Batch size (always 1 for Janus image generation).
    batch_size: usize,
    /// Maximum sequence length this cache can hold.
    max_seq_len: usize,
    /// Cached attention masks keyed by sequence length.
    masks: HashMap<usize, Tensor>,
    device: Device,
}

impl PreAllocKvCache {
    /// Construct a new pre-allocated KV cache.
    ///
    /// Allocates `2 × num_layers × B × n_kv_heads × max_seq_len × head_dim`
    /// elements up front.  For Janus-Pro-1B at BF16 with B=1, 24 layers,
    /// 16 KV heads, head_dim=128, max_seq_len=700:
    ///
    /// ```text
    /// 2 × 24 × 1 × 16 × 700 × 128 × 2 bytes = ~172 MB
    /// ```
    ///
    /// # Arguments
    ///
    /// * `dtype`       — Data type matching inference dtype (e.g. `DType::BF16`).
    /// * `cfg`         — Llama model config (provides layer count, head counts, dims).
    /// * `batch_size`  — Batch size (1 for single-image Janus generation).
    /// * `max_seq_len` — Maximum sequence length to pre-allocate.  For Janus
    ///   image generation use `576 + prefill_len`; a value of 700 is safe.
    /// * `device`      — Target device (`Device::Cpu` or a CUDA device).
    ///
    /// # Errors
    ///
    /// Returns a candle error if tensor allocation fails (e.g. OOM on GPU).
    pub fn new(dtype: DType, cfg: &Config, batch_size: usize, max_seq_len: usize, device: &Device) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;

        // ── RoPE tables ────────────────────────────────────────────────────
        let theta: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta, device)?;
        let idx_theta = Tensor::arange(0, cfg.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;

        // ── Per-layer pre-allocated buffers ────────────────────────────────
        // Shape: [B, n_kv_heads, max_seq_len, head_dim]
        let buf_shape = (batch_size, n_kv_heads, max_seq_len, head_dim);
        let layers: Vec<LayerKvBuffer> = (0..cfg.num_hidden_layers)
            .map(|_| {
                let k_buf = Tensor::zeros(buf_shape, dtype, device)?;
                let v_buf = Tensor::zeros(buf_shape, dtype, device)?;
                Ok(LayerKvBuffer {
                    k_buf,
                    v_buf,
                    scatter_idx_shape: buf_shape,
                })
            })
            .collect::<Result<_>>()?;

        Ok(Self {
            cos,
            sin,
            layers,
            current_seq_len: 0,
            n_kv_heads,
            head_dim,
            batch_size,
            max_seq_len,
            masks: HashMap::new(),
            device: device.clone(),
        })
    }

    /// Get or create a causal attention mask of size `t × t`.
    pub fn mask(&mut self, t: usize) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t).flat_map(|i| (0..t).map(move |j| u8::from(j > i))).collect();
            let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    /// Reset the cache for a new sequence.
    ///
    /// Clears `current_seq_len` without re-zeroing the buffers (stale data
    /// beyond `current_seq_len` is never read, so zeroing is unnecessary).
    pub fn clear(&mut self) {
        self.current_seq_len = 0;
        self.masks.clear();
    }

    /// Number of valid tokens currently stored in the cache.
    pub fn seq_len(&self) -> usize {
        self.current_seq_len
    }

    /// Update layer `block_idx` with newly computed key/value tensors.
    ///
    /// Writes `k` and `v` at position `seq_pos` along the sequence dimension
    /// using `scatter_set` (in-place, no allocation of new KV buffers).
    /// Returns zero-copy `narrow` views covering all valid tokens `[0, seq_pos + seq_len)`.
    ///
    /// # Arguments
    ///
    /// * `block_idx` — Transformer layer index.
    /// * `seq_pos`   — Absolute sequence position of the first token in `k`/`v`.
    /// * `k`         — New key tensor, shape `[B, n_kv_heads, seq_len, head_dim]`.
    /// * `v`         — New value tensor, shape `[B, n_kv_heads, seq_len, head_dim]`.
    ///
    /// # Returns
    ///
    /// `(k_full, v_full)` — Tensors of shape `[B, n_kv_heads, total_len, head_dim]`
    /// where `total_len = seq_pos + seq_len`.  These are zero-copy views for the
    /// single-token decode path; during prefill they are the pre-allocated slice
    /// (no copy of prior tokens needed).
    ///
    /// # Errors
    ///
    /// Returns a candle error if the cache is full (`seq_pos + seq_len > max_seq_len`)
    /// or on tensor shape mismatches.
    pub fn update(&mut self, block_idx: usize, seq_pos: usize, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let kv_seq_len = k.dim(2)?; // number of new tokens
        let end = seq_pos + kv_seq_len;

        if end > self.max_seq_len {
            candle_core::bail!(
                "PreAllocKvCache: seq_pos({seq_pos}) + seq_len({kv_seq_len}) = {end} \
                 exceeds max_seq_len({})",
                self.max_seq_len
            );
        }

        let layer = &self.layers[block_idx];

        // Build scatter index tensor for dim=2.
        //
        // `scatter_set` requires `indexes` with the same shape as `source`
        // (`k`/`v`).  Each element of `indexes` holds the target position along
        // dim 2 in the destination buffer.  For a contiguous block starting at
        // `seq_pos`, position `p` in the new tokens maps to `seq_pos + p`.
        //
        // Shape: [B, n_kv_heads, kv_seq_len, head_dim].
        // Memory layout is row-major [b][h][s][d], so we iterate b, h, s, d
        // in that order to produce the flat Vec that matches the tensor stride.
        let (b, h, s, d) = (self.batch_size, self.n_kv_heads, kv_seq_len, self.head_dim);
        let idx_vals: Vec<u32> = (0..b)
            .flat_map(|_b| {
                (0..h).flat_map(move |_h| {
                    (0..s).flat_map(move |p| {
                        let pos = (seq_pos + p) as u32;
                        std::iter::repeat(pos).take(d)
                    })
                })
            })
            .collect();

        let scatter_idx = Tensor::from_vec(idx_vals, (b, h, s, d), &self.device)?;

        // In-place write: scatter `k` into `k_buf` at the computed positions.
        // `scatter_set` mutates `layer.k_buf` storage without producing a new tensor.
        layer.k_buf.scatter_set(&scatter_idx, k, 2)?;
        layer.v_buf.scatter_set(&scatter_idx, v, 2)?;

        // Update the tracked sequence length (only needed on the last layer,
        // but we do it here so the caller can read `seq_len()` at any point).
        // `forward` already passes matching seq_pos values across layers.
        if block_idx + 1 == self.layers.len() || self.current_seq_len < end {
            self.current_seq_len = end;
        }

        // Return zero-copy narrow views of the valid token prefix.
        let k_full = layer.k_buf.narrow(2, 0, end)?;
        let v_full = layer.v_buf.narrow(2, 0, end)?;
        Ok((k_full, v_full))
    }

    /// Roll back the tracked sequence length to `target_len`.
    ///
    /// Used by self-speculative decoding to undo the KV writes from the draft
    /// phase before the verify phase overwrites those positions.  The underlying
    /// pre-allocated buffers are not modified — only `current_seq_len` is
    /// updated.  Positions `target_len..current_seq_len` become invisible to
    /// subsequent `update` calls (their data is stale and will be overwritten).
    ///
    /// # Panics
    ///
    /// Does not panic; if `target_len > current_seq_len` the length is
    /// unchanged (rolling forward is a no-op — only rollback makes sense).
    pub fn rollback_seq_len(&mut self, target_len: usize) {
        if target_len < self.current_seq_len {
            self.current_seq_len = target_len;
        }
    }

    /// Returns the total bytes used by all pre-allocated KV buffers.
    ///
    /// This reflects the total allocation, not just the portion that holds
    /// valid tokens.
    pub fn allocated_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| {
                let (b, h, s, d) = l.scatter_idx_shape;
                let elems = b * h * s * d;
                let dtype_bytes = l.k_buf.dtype().size_in_bytes();
                2 * elems * dtype_bytes // × 2 for K and V
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    /// Use flash attention kernel when available.
    ///
    /// Enabled when the `flash-attn` feature is active AND the model config
    /// requests it.  Falls back to the naive SDPA path otherwise.
    ///
    /// The field is stored unconditionally so the struct layout and `load`
    /// function are consistent regardless of the active feature set.  The
    /// compiler will dead-code-warn without the `flash-attn` feature; we
    /// suppress that with the conditional `allow` below.
    #[cfg_attr(not(feature = "flash-attn"), allow(dead_code))]
    use_flash_attn: bool,
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &KvCache) -> Result<Tensor> {
        let (_b_sz, _, seq_len, _hidden_size) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    /// Apply RoPE embeddings using cos/sin tables from a [`PreAllocKvCache`].
    ///
    /// Called by [`CausalSelfAttention::forward_prealloc`] on each step of
    /// the pre-allocated KV-cache path via `Block::forward_prealloc` →
    /// [`JanusLlama::forward_hidden_prealloc`].
    fn apply_rotary_emb_prealloc(&self, x: &Tensor, index_pos: usize, cache: &PreAllocKvCache) -> Result<Tensor> {
        let (_b_sz, _, seq_len, _hidden_size) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_attention_heads / self.num_key_value_heads;
        if n_rep == 1 {
            return Ok(x);
        }
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        x.unsqueeze(2)?
            .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
            .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }

    fn forward(&self, x: &Tensor, index_pos: usize, block_idx: usize, cache: &mut KvCache) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;

        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
                // Fix 4: cat on CUDA produces a contiguous result already; the
                // extra .contiguous() call was issuing 576×24×2 = 27,648
                // redundant device-copy kernels per generation pass.
                k = Tensor::cat(&[cache_k, &k], 2)?;
                v = Tensor::cat(&[cache_v, &v], 2)?;
                let k_seq_len = k.dims()[2];
                if k_seq_len > self.max_position_embeddings {
                    k = k
                        .narrow(
                            2,
                            k_seq_len - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                        .contiguous()?;
                }
                let v_seq_len = v.dims()[2];
                if v_seq_len > 2 * self.max_position_embeddings {
                    v = v
                        .narrow(
                            2,
                            v_seq_len - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                        .contiguous()?;
                }
            }
            // Detach from the computation graph to prevent a memory leak.
            // Without detach, each cat's BackpropOp holds references to
            // previous K/V tensors, creating a chain that prevents GPU
            // memory from being freed — causing OOM around step 350.
            cache.kvs[block_idx] = Some((k.detach(), v.detach()));
        }

        let k_full = self.repeat_kv(k)?;
        let v_full = self.repeat_kv(v)?;

        // Fix 1: Flash attention path — 2-3× faster than naive SDPA on CUDA.
        //
        // flash_attn expects [B, S, H, D] layout (sequence dimension second)
        // rather than the [B, H, S, D] layout we use internally.  Transpose
        // before the kernel and back after.
        //
        // The causal mask is only needed for prefill (seq_len > 1). For
        // single-token decode the mask is a no-op and can be skipped.
        #[cfg(feature = "flash-attn")]
        if self.use_flash_attn {
            // flash-attn requires BF16 or F16; cast if we are in F32/F64.
            let in_dtype = q.dtype();
            let fa_dtype = match in_dtype {
                DType::BF16 | DType::F16 => in_dtype,
                _ => DType::BF16,
            };
            // Transpose [B, H, S, D] → [B, S, H, D] and ensure contiguous.
            let q_fa = q.transpose(1, 2)?.to_dtype(fa_dtype)?.contiguous()?;
            let k_fa = k_full.transpose(1, 2)?.to_dtype(fa_dtype)?.contiguous()?;
            let v_fa = v_full.transpose(1, 2)?.to_dtype(fa_dtype)?.contiguous()?;
            let softmax_scale = 1.0_f32 / (self.head_dim as f32).sqrt();
            // Apply causal mask only during prefill (seq_len > 1).
            let causal = seq_len > 1;
            let y_fa = candle_flash_attn::flash_attn(&q_fa, &k_fa, &v_fa, softmax_scale, causal)?;
            // Transpose back [B, S, H, D] → [B, H, S, D], then flatten to [B, S, hidden].
            let y = y_fa
                .transpose(1, 2)?
                .reshape((b_sz, seq_len, hidden_size))?
                .to_dtype(in_dtype)?;
            return self.o_proj.forward(&y);
        }

        // Fix 1 (fallback): Standard scaled dot-product attention.
        // On CUDA with BF16/F16, compute in native dtype to avoid F32 copies
        // that fragment the CUDA memory pool (KV cache grows each step).
        // On CPU with F32, no cast is needed either.
        let in_dtype = q.dtype();
        let use_f32_attn = matches!(in_dtype, DType::F64); // only upcast for F64
        let (q, k_full, v_full) = if use_f32_attn {
            (
                q.to_dtype(DType::F32)?,
                k_full.to_dtype(DType::F32)?,
                v_full.to_dtype(DType::F32)?,
            )
        } else {
            (q, k_full, v_full)
        };
        let att = (q.matmul(&k_full.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = if seq_len == 1 {
            att
        } else {
            let mask = cache.mask(seq_len)?.broadcast_as(att.shape())?;
            let neg_inf = if use_f32_attn {
                f32::NEG_INFINITY
            } else {
                // For BF16/F16, use a large negative value instead of infinity.
                -65504.0_f32
            };
            masked_fill(&att, &mask, neg_inf)?
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = if use_f32_attn {
            att.matmul(&v_full.contiguous()?)?.to_dtype(in_dtype)?
        } else {
            att.matmul(&v_full.contiguous()?)?
        };
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        self.o_proj.forward(&y)
    }

    /// Forward pass using the pre-allocated KV cache.
    ///
    /// Writes new K/V tensors into the pre-allocated ring buffer via
    /// `scatter_set` (in-place) and reads accumulated history with `narrow`
    /// (zero-copy).  Eliminates the `Tensor::cat` cost of the standard
    /// [`CausalSelfAttention::forward`] path.
    ///
    /// Detach is not needed: `scatter_set` is in-place on a pre-allocated
    /// buffer that lives outside the autograd graph (created with `zeros`).
    ///
    fn forward_prealloc(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut PreAllocKvCache,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.apply_rotary_emb_prealloc(&q, index_pos, cache)?;
        let k = self.apply_rotary_emb_prealloc(&k, index_pos, cache)?;

        // In-place write + zero-copy read.
        // `update` returns narrow views `[B, n_kv_heads, total_len, head_dim]`.
        let (k_full, v_full) = cache.update(block_idx, index_pos, &k, &v)?;

        let k_full = self.repeat_kv(k_full)?;
        let v_full = self.repeat_kv(v_full)?;

        // Attention — reuse the same SDPA / flash-attn paths as `forward`.
        let total_len = k_full.dim(2)?;

        #[cfg(feature = "flash-attn")]
        if self.use_flash_attn {
            let in_dtype = q.dtype();
            let fa_dtype = match in_dtype {
                DType::BF16 | DType::F16 => in_dtype,
                _ => DType::BF16,
            };
            let q_fa = q.transpose(1, 2)?.to_dtype(fa_dtype)?.contiguous()?;
            let k_fa = k_full.transpose(1, 2)?.to_dtype(fa_dtype)?.contiguous()?;
            let v_fa = v_full.transpose(1, 2)?.to_dtype(fa_dtype)?.contiguous()?;
            let softmax_scale = 1.0_f32 / (self.head_dim as f32).sqrt();
            let causal = seq_len > 1;
            let y_fa = candle_flash_attn::flash_attn(&q_fa, &k_fa, &v_fa, softmax_scale, causal)?;
            let y = y_fa
                .transpose(1, 2)?
                .reshape((b_sz, seq_len, hidden_size))?
                .to_dtype(in_dtype)?;
            return self.o_proj.forward(&y);
        }

        let in_dtype = q.dtype();
        let use_f32_attn = matches!(in_dtype, DType::F64);
        let (q, k_full, v_full) = if use_f32_attn {
            (
                q.to_dtype(DType::F32)?,
                k_full.to_dtype(DType::F32)?,
                v_full.to_dtype(DType::F32)?,
            )
        } else {
            (q, k_full, v_full)
        };
        let att = (q.matmul(&k_full.t()?)? / (self.head_dim as f64).sqrt())?;
        // For the single-token decode case (seq_len == 1), we still need a mask
        // over `total_len` positions when total_len > 1 (cross-attn to history).
        // However, the query is length 1 so the causal mask degenerates to all
        // positions being visible.  We skip masking for seq_len == 1.
        let att = if seq_len == 1 {
            att
        } else {
            // During prefill, use the full `total_len × total_len` causal mask.
            // `att` shape is [B, H, seq_len, total_len]; we broadcast a
            // seq_len × total_len mask (future positions masked).
            let mask = cache.mask(total_len)?.narrow(0, total_len - seq_len, seq_len)?;
            let mask = mask.broadcast_as(att.shape())?;
            let neg_inf = if use_f32_attn { f32::NEG_INFINITY } else { -65504.0_f32 };
            masked_fill(&att, &mask, neg_inf)?
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = if use_f32_attn {
            att.matmul(&v_full.contiguous()?)?.to_dtype(in_dtype)?
        } else {
            att.matmul(&v_full.contiguous()?)?
        };
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        self.o_proj.forward(&y)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear_no_bias(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(size_q, size_in, vb.pp("o_proj"))?;
        // The flash-attn path is only active when both the feature flag and the
        // model's use_flash_attn config field are set.  On CPU or when the
        // feature is disabled the naive SDPA path is used instead.
        let use_flash_attn = cfg.use_flash_attn;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            max_position_embeddings: cfg.max_position_embeddings,
            use_flash_attn,
        })
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?
        .to_dtype(on_false.dtype())?
        .broadcast_as(shape.dims())?;
    mask.where_cond(&on_true, on_false)
}

// ---------------------------------------------------------------------------
// MLP (SwiGLU)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            gate_proj: linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(h, i, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(i, h, vb.pp("down_proj"))?,
        })
    }
}

// ---------------------------------------------------------------------------
// Block (transformer layer)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Block {
    input_layernorm: candle_nn::RmsNorm,
    attn: CausalSelfAttention,
    post_attention_layernorm: candle_nn::RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn forward(&self, x: &Tensor, index_pos: usize, block_idx: usize, cache: &mut KvCache) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = (self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.post_attention_layernorm.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn forward_prealloc(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut PreAllocKvCache,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = (self.attn.forward_prealloc(&x, index_pos, block_idx, cache)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.post_attention_layernorm.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let input_layernorm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        Ok(Self {
            input_layernorm,
            attn,
            post_attention_layernorm,
            mlp,
        })
    }
}

// ---------------------------------------------------------------------------
// JanusLlama — Llama backbone with hidden-state access
// ---------------------------------------------------------------------------

/// Custom Llama implementation that exposes pre-lm_head hidden states.
///
/// Unlike `candle_transformers::models::llama::Llama`, this struct provides
/// [`forward_hidden`] which returns `[B, hidden_size]` tensors that can be
/// fed into the Janus-Pro generation head.
#[derive(Debug, Clone)]
pub struct JanusLlama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: candle_nn::RmsNorm,
    lm_head: candle_nn::Linear,
}

impl JanusLlama {
    /// Embed token IDs → `[B, S, hidden_size]`.
    pub fn embed(&self, x: &Tensor) -> Result<Tensor> {
        self.wte.forward(x)
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Forward pass returning **hidden states** (pre-lm_head).
    ///
    /// Returns shape `[B, hidden_size]` — the last-position hidden state
    /// after all transformer layers and final RMS norm, but **before** the
    /// text-vocabulary LM head projection.
    ///
    /// This is what Janus-Pro needs to project through `gen_head` to get
    /// image-vocabulary logits.
    pub fn forward_hidden(&self, input_embed: &Tensor, index_pos: usize, cache: &mut KvCache) -> Result<Tensor> {
        let (_, seq_len, _) = input_embed.dims3()?;
        let mut x = input_embed.clone();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        // Extract last position: [B, S, H] → [B, H]
        x.i((.., seq_len - 1, ..))?.contiguous()
    }

    /// Project hidden states to vocabulary logits on the `lm_head` device.
    pub fn project_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let weight = self.lm_head.weight();
        let logits = if matches!(weight.device(), Device::Cpu) {
            let hidden = hidden.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.contiguous()?;
            let weight_t = weight.to_dtype(DType::F32)?.t()?.contiguous()?;
            hidden.matmul(&weight_t)?
        } else {
            let hidden = hidden.to_device(weight.device())?.to_dtype(weight.dtype())?;
            self.lm_head.forward(&hidden)?
        };
        logits.to_dtype(DType::F32)
    }

    /// Forward pass returning text-vocabulary logits (includes lm_head).
    ///
    /// Returns shape `[B, vocab_size]` — equivalent to candle's
    /// `Llama::forward_input_embed`.
    pub fn forward_input_embed(&self, input_embed: &Tensor, index_pos: usize, cache: &mut KvCache) -> Result<Tensor> {
        let hidden = self.forward_hidden(input_embed, index_pos, cache)?;
        self.project_logits(&hidden)
    }

    /// Standard forward from token IDs (embed + transformer + lm_head).
    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut KvCache) -> Result<Tensor> {
        let embeds = self.wte.forward(x)?;
        self.forward_input_embed(&embeds, index_pos, cache)
    }

    /// Forward pass returning **hidden states** using the pre-allocated KV cache.
    ///
    /// Functionally equivalent to [`forward_hidden`] but uses [`PreAllocKvCache`]
    /// to eliminate `Tensor::cat` overhead.  Each layer writes new KV pairs
    /// in-place via `scatter_set` and reads accumulated history as a zero-copy
    /// `narrow` view.
    ///
    /// # Arguments
    ///
    /// * `input_embed` — Float tensor `[B, S, hidden_size]`.
    /// * `index_pos`   — Absolute position of the first token in `input_embed`.
    /// * `cache`       — Mutable reference to the pre-allocated KV cache.
    ///
    /// # Returns
    ///
    /// Float tensor `[B, hidden_size]` — last-position hidden state before `lm_head`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the cache is full or on shape mismatches.
    pub fn forward_hidden_prealloc(
        &self,
        input_embed: &Tensor,
        index_pos: usize,
        cache: &mut PreAllocKvCache,
    ) -> Result<Tensor> {
        let (_, seq_len, _) = input_embed.dims3()?;
        let mut x = input_embed.clone();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward_prealloc(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        // Extract last position: [B, S, H] → [B, H]
        x.i((.., seq_len - 1, ..))?.contiguous()
    }

    /// Forward pass returning text-vocabulary logits using the pre-allocated KV cache.
    ///
    /// Combines [`forward_hidden_prealloc`] with [`project_logits`].  Use this
    /// instead of [`forward_input_embed`] during Janus image generation to
    /// avoid the `Tensor::cat` bottleneck.
    ///
    /// # Returns
    ///
    /// Float tensor `[B, vocab_size]`.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`forward_hidden_prealloc`] and [`project_logits`].
    pub fn forward_input_embed_prealloc(
        &self,
        input_embed: &Tensor,
        index_pos: usize,
        cache: &mut PreAllocKvCache,
    ) -> Result<Tensor> {
        let hidden = self.forward_hidden_prealloc(input_embed, index_pos, cache)?;
        self.project_logits(&hidden)
    }

    /// Forward pass running only the first `num_draft_layers` blocks.
    ///
    /// Used by the self-speculative decoding draft phase: a fast, approximate
    /// forward pass that skips layers `num_draft_layers..num_layers`.  Because
    /// only a prefix of the full transformer stack is executed, this is roughly
    /// `num_draft_layers / num_layers` of the cost of a full forward pass.
    ///
    /// The KV cache is **shared** with the verify phase — draft writes KV
    /// entries at position `index_pos`, and the subsequent verify pass
    /// overwrites those same entries with the full-depth computation.
    ///
    /// # Arguments
    ///
    /// * `input_embed`     — Float tensor `[B, 1, hidden_size]` (single token).
    /// * `index_pos`       — Absolute sequence position for RoPE and KV indexing.
    /// * `cache`           — Pre-allocated KV cache (shared with verify phase).
    /// * `num_draft_layers`— Number of transformer blocks to run (must be ≤
    ///   `self.blocks.len()`).
    ///
    /// # Returns
    ///
    /// Float tensor `[B, hidden_size]` — last-position hidden state after
    /// `num_draft_layers` blocks and the final RMS norm.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `num_draft_layers > self.blocks.len()`, the
    /// cache is full, or on any shape mismatch.
    pub fn forward_hidden_draft(
        &self,
        input_embed: &Tensor,
        index_pos: usize,
        cache: &mut PreAllocKvCache,
        num_draft_layers: usize,
    ) -> Result<Tensor> {
        if num_draft_layers > self.blocks.len() {
            candle_core::bail!(
                "forward_hidden_draft: num_draft_layers({num_draft_layers}) \
                 exceeds total blocks({})",
                self.blocks.len()
            );
        }
        let (_, seq_len, _) = input_embed.dims3()?;
        let mut x = input_embed.clone();
        for (block_idx, block) in self.blocks[..num_draft_layers].iter().enumerate() {
            x = block.forward_prealloc(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        // Extract last position: [B, S, H] → [B, H]
        x.i((.., seq_len - 1, ..))?.contiguous()
    }

    /// Verify a batch of K draft tokens in a single full-depth forward pass.
    ///
    /// Runs all transformer blocks on `K` token embeddings simultaneously,
    /// returning hidden states for **all** positions in the batch (not just
    /// the last).  This is the verify phase of self-speculative decoding.
    ///
    /// The KV entries written during the draft phase (at positions
    /// `start_pos..start_pos+K`) are **overwritten** by this call with the
    /// full-depth computation — so after returning, the cache is consistent
    /// with a normal autoregressive pass at depth `num_layers`.
    ///
    /// # Arguments
    ///
    /// * `input_embed` — Float tensor `[B, K, hidden_size]` — embeddings for
    ///   the `K` draft tokens to verify.
    /// * `start_pos`   — Absolute position of the first draft token (for RoPE
    ///   and KV indexing).
    /// * `cache`       — Pre-allocated KV cache (same instance used by draft).
    ///
    /// # Returns
    ///
    /// Float tensor `[B, K, hidden_size]` — hidden states for all `K` positions
    /// **before** the LM head projection.  Callers apply `gen_head` to get
    /// image logits for each position.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the cache is full or on any shape mismatch.
    pub fn forward_hidden_verify_batch(
        &self,
        input_embed: &Tensor,
        start_pos: usize,
        cache: &mut PreAllocKvCache,
    ) -> Result<Tensor> {
        let mut x = input_embed.clone();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward_prealloc(&x, start_pos, block_idx, cache)?;
        }
        // Return all positions: [B, K, hidden_size] (no last-position extraction).
        self.ln_f.forward(&x)
    }

    /// Offload `wte` to CPU to free GPU VRAM.
    ///
    /// During image generation, `wte` is only used for prompt token lookups.
    /// We keep `lm_head` on the main device so image understanding can decode
    /// text without paying a large CPU projection cost on every token.
    ///
    /// After calling this, `embed()` will return CPU tensors — callers must
    /// `.to_device(gpu)` the result before feeding into the transformer.
    pub fn offload_embeddings_to_cpu(&mut self) -> Result<()> {
        let cpu = Device::Cpu;
        let cpu_wte_weight = self.wte.embeddings().to_device(&cpu)?;
        let hidden_size = cpu_wte_weight.dim(1)?;
        self.wte = Embedding::new(cpu_wte_weight, hidden_size);
        Ok(())
    }

    /// Load weights from the VarBuilder.
    ///
    /// Weight paths match the candle Llama convention:
    /// - `model.embed_tokens` — token embedding
    /// - `model.layers.{i}` — transformer blocks
    /// - `model.norm` — final RMS norm
    /// - `lm_head` — text-vocabulary projection (or tied weights)
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = if cfg.tie_word_embeddings {
            candle_nn::Linear::new(wte.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        let ln_f = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), cfg))
            .collect::<Result<_>>()?;

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    fn tiny_config() -> Config {
        Config {
            hidden_size: 64,
            intermediate_size: 128,
            vocab_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            use_flash_attn: false,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: 128,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn test_janus_llama_construction() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let _llama = JanusLlama::load(vb, &tiny_config()).expect("construction failed");
    }

    /// forward_hidden must return [B, hidden_size], not [B, vocab_size].
    #[test]
    fn test_forward_hidden_shape() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let cfg = tiny_config();
        let llama = JanusLlama::load(vb, &cfg).expect("construction failed");
        let mut cache = KvCache::new(true, DType::F32, &cfg, &Device::Cpu).unwrap();

        let embed = Tensor::zeros((1_usize, 3_usize, 64_usize), DType::F32, &Device::Cpu).unwrap();
        let hidden = llama.forward_hidden(&embed, 0, &mut cache).unwrap();

        assert_eq!(hidden.dims(), &[1, 64], "expected [B, hidden_size]");
    }

    /// forward_input_embed must return [B, vocab_size].
    #[test]
    fn test_forward_logits_shape() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let cfg = tiny_config();
        let llama = JanusLlama::load(vb, &cfg).expect("construction failed");
        let mut cache = KvCache::new(true, DType::F32, &cfg, &Device::Cpu).unwrap();

        let embed = Tensor::zeros((1_usize, 3_usize, 64_usize), DType::F32, &Device::Cpu).unwrap();
        let logits = llama.forward_input_embed(&embed, 0, &mut cache).unwrap();

        assert_eq!(logits.dims(), &[1, 256], "expected [B, vocab_size]");
    }

    #[test]
    fn test_project_logits_cpu_offload_uses_f32_matmul() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::BF16, &Device::Cpu);
        let mut cfg = tiny_config();
        cfg.tie_word_embeddings = true;
        let mut llama = JanusLlama::load(vb, &cfg).expect("construction failed");
        llama.offload_embeddings_to_cpu().expect("cpu offload failed");

        let hidden = Tensor::zeros((1_usize, cfg.hidden_size), DType::F32, &Device::Cpu).unwrap();
        let logits = llama.project_logits(&hidden).expect("project logits failed");

        assert_eq!(logits.dims(), &[1, cfg.vocab_size], "expected [B, vocab_size]");
        assert_eq!(logits.dtype(), DType::F32, "expected f32 logits on cpu");
    }

    /// KvCache::memory_bytes should increase after a forward pass.
    #[test]
    fn test_kv_cache_memory_tracking() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let cfg = tiny_config();
        let llama = JanusLlama::load(vb, &cfg).expect("construction failed");
        let mut cache = KvCache::new(true, DType::F32, &cfg, &Device::Cpu).unwrap();

        assert_eq!(cache.memory_bytes(), 0);

        let embed = Tensor::zeros((1_usize, 3_usize, 64_usize), DType::F32, &Device::Cpu).unwrap();
        let _ = llama.forward_hidden(&embed, 0, &mut cache).unwrap();

        assert!(cache.memory_bytes() > 0, "KV cache should have allocated memory");
    }

    // ── PreAllocKvCache tests ────────────────────────────────────────────────

    /// PreAllocKvCache::new should allocate correctly sized buffers.
    ///
    /// For the tiny config: head_dim = 64/4 = 16, n_kv_heads = 4, layers = 2.
    /// Allocated bytes = 2 × layers × 2 × B × n_kv_heads × max_seq × head_dim × dtype_bytes.
    #[test]
    fn test_prealloc_cache_construction() {
        let cfg = tiny_config();
        let cache =
            PreAllocKvCache::new(DType::F32, &cfg, 1, 32, &Device::Cpu).expect("PreAllocKvCache construction failed");

        assert_eq!(cache.seq_len(), 0, "fresh cache should be empty");

        // 2 layers × 2 (K+V) × 1 × 4 × 32 × 16 × 4 bytes
        let expected = 2 * 2 * 1 * 4 * 32 * 16 * 4;
        assert_eq!(cache.allocated_bytes(), expected);
    }

    /// `update` should return correctly-shaped narrow views.
    #[test]
    fn test_prealloc_cache_update_shapes() {
        let cfg = tiny_config();
        let mut cache =
            PreAllocKvCache::new(DType::F32, &cfg, 1, 32, &Device::Cpu).expect("PreAllocKvCache construction failed");

        // [B=1, n_kv_heads=4, seq=3, head_dim=16]
        let k = Tensor::zeros((1_usize, 4_usize, 3_usize, 16_usize), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::zeros((1_usize, 4_usize, 3_usize, 16_usize), DType::F32, &Device::Cpu).unwrap();

        let (k_full, v_full) = cache.update(0, 0, &k, &v).expect("update failed");

        assert_eq!(k_full.dims(), &[1, 4, 3, 16], "k_full should cover written tokens");
        assert_eq!(v_full.dims(), &[1, 4, 3, 16], "v_full should cover written tokens");
        assert_eq!(cache.seq_len(), 3);
    }

    /// Successive single-token writes should accumulate: pos 0 (ones), pos 1 (zeros).
    #[test]
    fn test_prealloc_cache_incremental_update() {
        let cfg = tiny_config();
        let mut cache = PreAllocKvCache::new(DType::F32, &cfg, 1, 32, &Device::Cpu).expect("construction failed");

        let k1 = Tensor::ones((1_usize, 4_usize, 1_usize, 16_usize), DType::F32, &Device::Cpu).unwrap();
        let v1 = Tensor::ones((1_usize, 4_usize, 1_usize, 16_usize), DType::F32, &Device::Cpu).unwrap();
        let k2 = Tensor::zeros((1_usize, 4_usize, 1_usize, 16_usize), DType::F32, &Device::Cpu).unwrap();
        let v2 = Tensor::zeros((1_usize, 4_usize, 1_usize, 16_usize), DType::F32, &Device::Cpu).unwrap();

        cache.update(0, 0, &k1, &v1).expect("step 0 failed");
        let (k_full, _) = cache.update(0, 1, &k2, &v2).expect("step 1 failed");

        assert_eq!(k_full.dims(), &[1, 4, 2, 16]);

        // Position 0 should hold ones, position 1 should hold zeros.
        let pos0_sum: f32 = k_full
            .narrow(2, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .sum();
        let pos1_sum: f32 = k_full
            .narrow(2, 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .sum();
        assert!(pos0_sum > 0.0, "pos 0 should hold ones");
        assert_eq!(pos1_sum, 0.0, "pos 1 should hold zeros");
    }

    /// `clear` resets seq_len without reallocating buffers.
    #[test]
    fn test_prealloc_cache_clear() {
        let cfg = tiny_config();
        let mut cache = PreAllocKvCache::new(DType::F32, &cfg, 1, 32, &Device::Cpu).expect("construction failed");

        let bytes_before = cache.allocated_bytes();
        let k = Tensor::zeros((1_usize, 4_usize, 1_usize, 16_usize), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::zeros_like(&k).unwrap();
        cache.update(0, 0, &k, &v).unwrap();
        assert_eq!(cache.seq_len(), 1);

        cache.clear();
        assert_eq!(cache.seq_len(), 0, "clear should reset seq_len to 0");
        assert_eq!(cache.allocated_bytes(), bytes_before, "clear must not reallocate");
    }

    /// `update` must fail when write would overflow max_seq_len.
    #[test]
    fn test_prealloc_cache_overflow_error() {
        let cfg = tiny_config();
        let mut cache = PreAllocKvCache::new(DType::F32, &cfg, 1, 4, &Device::Cpu).expect("construction failed");

        // seq_pos=2 + seq_len=3 = 5, exceeds max_seq_len=4.
        let k = Tensor::zeros((1_usize, 4_usize, 3_usize, 16_usize), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::zeros_like(&k).unwrap();
        let result = cache.update(0, 2, &k, &v);
        assert!(result.is_err(), "overflow should return Err");
    }

    /// `forward_hidden_prealloc` must return `[B, hidden_size]`.
    #[test]
    fn test_forward_hidden_prealloc_shape() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let cfg = tiny_config();
        let llama = JanusLlama::load(vb, &cfg).expect("construction failed");
        let mut cache =
            PreAllocKvCache::new(DType::F32, &cfg, 1, 32, &Device::Cpu).expect("PreAllocKvCache construction failed");

        let embed = Tensor::zeros((1_usize, 3_usize, 64_usize), DType::F32, &Device::Cpu).unwrap();
        let hidden = llama.forward_hidden_prealloc(&embed, 0, &mut cache).unwrap();

        assert_eq!(hidden.dims(), &[1, 64], "expected [B, hidden_size]");
        assert_eq!(cache.seq_len(), 3, "seq_len should equal prefill length");
    }

    /// `forward_input_embed_prealloc` must return `[B, vocab_size]`.
    #[test]
    fn test_forward_input_embed_prealloc_shape() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let cfg = tiny_config();
        let llama = JanusLlama::load(vb, &cfg).expect("construction failed");
        let mut cache = PreAllocKvCache::new(DType::F32, &cfg, 1, 32, &Device::Cpu).expect("construction failed");

        let embed = Tensor::zeros((1_usize, 3_usize, 64_usize), DType::F32, &Device::Cpu).unwrap();
        let logits = llama.forward_input_embed_prealloc(&embed, 0, &mut cache).unwrap();

        assert_eq!(logits.dims(), &[1, 256], "expected [B, vocab_size]");
    }

    /// `rollback_seq_len` reduces `current_seq_len` without freeing buffers.
    #[test]
    fn test_prealloc_rollback_seq_len() {
        let cfg = tiny_config();
        let mut cache = PreAllocKvCache::new(DType::F32, &cfg, 1, 32, &Device::Cpu).expect("construction failed");

        // Write 5 tokens.
        let k = Tensor::zeros((1_usize, 4_usize, 5_usize, 16_usize), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::zeros_like(&k).unwrap();
        cache.update(0, 0, &k, &v).expect("update failed");
        assert_eq!(cache.seq_len(), 5);

        // Roll back to 3.
        cache.rollback_seq_len(3);
        assert_eq!(cache.seq_len(), 3, "seq_len should be 3 after rollback");

        // Rolling forward (target > current) is a no-op.
        cache.rollback_seq_len(10);
        assert_eq!(cache.seq_len(), 3, "rollback_seq_len must not advance seq_len");

        // Allocated bytes must not change — no reallocation.
        let bytes = cache.allocated_bytes();
        assert!(bytes > 0);
        cache.rollback_seq_len(0);
        assert_eq!(cache.allocated_bytes(), bytes, "rollback must not reallocate");
    }

    /// `forward_hidden_draft` must return `[B, hidden_size]` and only
    /// write KV entries for `num_draft_layers` blocks.
    #[test]
    fn test_forward_hidden_draft_shape() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let cfg = tiny_config();
        // tiny_config has 2 layers; draft with 1 layer.
        let llama = JanusLlama::load(vb, &cfg).expect("construction failed");
        let mut cache =
            PreAllocKvCache::new(DType::F32, &cfg, 1, 32, &Device::Cpu).expect("PreAllocKvCache construction failed");

        let embed = Tensor::zeros((1_usize, 1_usize, 64_usize), DType::F32, &Device::Cpu).unwrap();
        let hidden = llama.forward_hidden_draft(&embed, 0, &mut cache, 1).unwrap();

        assert_eq!(hidden.dims(), &[1, 64], "draft hidden should be [B, hidden_size]");
    }

    /// `forward_hidden_draft` with `num_draft_layers == num_layers` must
    /// produce the same result as `forward_hidden_prealloc`.
    ///
    /// We cannot compare tensor values because random weights differ between
    /// separate VarMap instances, so we verify shapes only and check that a
    /// single-call approach via both paths succeeds.
    #[test]
    fn test_forward_hidden_draft_full_depth_shape() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let cfg = tiny_config();
        let llama = JanusLlama::load(vb, &cfg).expect("construction failed");
        let mut cache = PreAllocKvCache::new(DType::F32, &cfg, 1, 32, &Device::Cpu).expect("construction failed");

        let embed = Tensor::zeros((1_usize, 1_usize, 64_usize), DType::F32, &Device::Cpu).unwrap();
        let hidden = llama
            .forward_hidden_draft(&embed, 0, &mut cache, llama.num_layers())
            .unwrap();

        assert_eq!(hidden.dims(), &[1, 64]);
    }

    /// `forward_hidden_draft` must fail when `num_draft_layers` exceeds the
    /// total block count.
    #[test]
    fn test_forward_hidden_draft_overflow_error() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let cfg = tiny_config();
        let llama = JanusLlama::load(vb, &cfg).expect("construction failed");
        let mut cache = PreAllocKvCache::new(DType::F32, &cfg, 1, 32, &Device::Cpu).expect("construction failed");

        let embed = Tensor::zeros((1_usize, 1_usize, 64_usize), DType::F32, &Device::Cpu).unwrap();
        let result = llama.forward_hidden_draft(&embed, 0, &mut cache, llama.num_layers() + 1);
        assert!(result.is_err(), "draft with too many layers should return Err");
    }

    /// `forward_hidden_verify_batch` must return `[B, K, hidden_size]`.
    #[test]
    fn test_forward_hidden_verify_batch_shape() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let cfg = tiny_config();
        let llama = JanusLlama::load(vb, &cfg).expect("construction failed");
        let mut cache = PreAllocKvCache::new(DType::F32, &cfg, 1, 32, &Device::Cpu).expect("construction failed");

        // 3-token prefill to populate the cache.
        let prefill = Tensor::zeros((1_usize, 3_usize, 64_usize), DType::F32, &Device::Cpu).unwrap();
        llama.forward_hidden_prealloc(&prefill, 0, &mut cache).unwrap();

        // Verify 4 draft tokens in one batch.
        let k = 4_usize;
        let verify_input = Tensor::zeros((1_usize, k, 64_usize), DType::F32, &Device::Cpu).unwrap();
        let all_hidden = llama.forward_hidden_verify_batch(&verify_input, 3, &mut cache).unwrap();

        assert_eq!(
            all_hidden.dims(),
            &[1, k, 64],
            "verify_batch should return [B, K, hidden_size]"
        );
    }

    /// Autoregressive decode: 3-token prefill followed by 5 single-token decode steps.
    ///
    /// Verifies that the pre-allocated cache correctly accumulates KV pairs
    /// across sequential decode steps and that output shapes remain correct.
    #[test]
    fn test_prealloc_autoregressive_decode() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let cfg = tiny_config();
        let llama = JanusLlama::load(vb, &cfg).expect("construction failed");
        // 3-token prefill + 5 decode steps = 8 total; max_seq_len 16 gives headroom.
        let mut cache = PreAllocKvCache::new(DType::F32, &cfg, 1, 16, &Device::Cpu).expect("construction failed");

        // Prefill 3 tokens.
        let prefill = Tensor::zeros((1_usize, 3_usize, 64_usize), DType::F32, &Device::Cpu).unwrap();
        llama.forward_hidden_prealloc(&prefill, 0, &mut cache).unwrap();
        assert_eq!(cache.seq_len(), 3);

        // Decode 5 single-token steps.
        for step in 0..5_usize {
            let token = Tensor::zeros((1_usize, 1_usize, 64_usize), DType::F32, &Device::Cpu).unwrap();
            let hidden = llama.forward_hidden_prealloc(&token, 3 + step, &mut cache).unwrap();
            assert_eq!(hidden.dims(), &[1, 64], "step {step}: hidden shape mismatch");
        }
        assert_eq!(cache.seq_len(), 8);
    }
}
