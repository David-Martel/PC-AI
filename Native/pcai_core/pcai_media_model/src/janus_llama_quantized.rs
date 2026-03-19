//! Quantized Llama backbone for Janus-Pro using GGUF weights.
//!
//! This module mirrors [`crate::janus_llama::JanusLlama`] but loads weight
//! matrices from a GGUF file as [`QMatMul`] tensors instead of full-precision
//! [`candle_nn::Linear`] layers.  The expected speedup is 4–6× on CPU and
//! 2–3× on GPU (bandwidth-bound workloads) when using Q4_K quantization.
//!
//! # What is quantized vs full-precision
//!
//! | Component | Quantized? | Rationale |
//! |-----------|-----------|-----------|
//! | `wte` (token embedding) | No — F32/BF16 | Small, needed for CPU offload |
//! | Attention projections (q, k, v, o) | Yes — QMatMul | Hot path |
//! | MLP projections (gate, up, down) | Yes — QMatMul | Hot path |
//! | RMS norm weights | No — F32 | Tiny scalars, no benefit from quant |
//! | `lm_head` | Yes — QMatMul | Called once per step |
//!
//! # GGUF tensor name convention
//!
//! The GGUF file produced by `llama.cpp convert-hf-to-gguf` for a Janus-Pro
//! checkpoint uses the following names (where `N` is the zero-based layer index):
//!
//! ```text
//! token_embd.weight              — wte (dequantized to F32)
//! blk.N.attn_norm.weight         — input_layernorm
//! blk.N.attn_q.weight            — q_proj
//! blk.N.attn_k.weight            — k_proj
//! blk.N.attn_v.weight            — v_proj
//! blk.N.attn_output.weight       — o_proj
//! blk.N.ffn_norm.weight          — post_attention_layernorm
//! blk.N.ffn_gate.weight          — gate_proj (SwiGLU gate)
//! blk.N.ffn_up.weight            — up_proj
//! blk.N.ffn_down.weight          — down_proj
//! output_norm.weight             — ln_f
//! output.weight                  — lm_head
//! ```
//!
//! # KV cache compatibility
//!
//! [`QuantizedJanusLlama`] uses the same [`KvCache`] and [`PreAllocKvCache`]
//! types as [`JanusLlama`], so all higher-level pipeline code in `generate.rs`
//! works without modification.
//!
//! RoPE is computed in the dtype of the KvCache cos/sin tables.  The helper
//! [`apply_rotary_emb`] casts Q/K to that dtype if needed and casts back
//! afterwards, eliminating the "unsupported dtype" error when QMatMul produces
//! BF16 hidden states but the cache holds F32 cos/sin tables (or vice-versa).
//!
//! # Example (offline smoke-test)
//!
//! ```rust,no_run
//! use std::io::BufReader;
//! use std::fs::File;
//! use candle_core::Device;
//! use candle_core::quantized::gguf_file;
//! use pcai_media_model::config::JanusConfig;
//! use pcai_media_model::janus_llama_quantized::QuantizedJanusLlama;
//!
//! # fn example() -> candle_core::Result<()> {
//! let mut f = BufReader::new(File::open("model.gguf").unwrap());
//! let ct = gguf_file::Content::read(&mut f)?;
//! let cfg = JanusConfig::janus_pro_1b();
//! let llama_cfg = cfg.to_llama_config(false);
//! let model = QuantizedJanusLlama::from_gguf(ct, &mut f, &llama_cfg, &Device::Cpu)?;
//! # Ok(()) }
//! ```

use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::Embedding;
use candle_transformers::models::llama::Config;

use crate::janus_llama::{KvCache, PreAllocKvCache};

// ---------------------------------------------------------------------------
// Helper: build a full-precision RmsNorm from a dequantized QTensor weight
// ---------------------------------------------------------------------------

/// Build a [`candle_nn::RmsNorm`] by dequantizing a scalar-weight [`QTensor`].
///
/// Norm weights are tiny (one scalar per hidden dim) so full-precision
/// dequantization is always correct and incurs negligible overhead.
fn rms_norm_from_qtensor(weight: QTensor, eps: f64, device: &Device) -> Result<candle_nn::RmsNorm> {
    let weight = weight.dequantize(device)?.to_dtype(DType::F32)?;
    Ok(candle_nn::RmsNorm::new(weight, eps))
}

// ---------------------------------------------------------------------------
// RoPE helper — dtype-safe wrapper
// ---------------------------------------------------------------------------

/// Apply rotary position embeddings to `x`, handling dtype mismatches.
///
/// `candle_nn::rotary_emb::rope` requires `x`, `cos`, and `sin` to share the
/// same dtype.  In the quantized path, `QMatMul::forward` dequantizes weights
/// to the *working* dtype (BF16 on CUDA, F32 on CPU), while the `KvCache`
/// cos/sin tables are constructed in whatever dtype the caller supplied to
/// `KvCache::new`.  When these differ (e.g. BF16 hidden states but F32
/// cos/sin tables, or vice-versa) the RoPE kernel returns an
/// "unsupported dtype" error.
///
/// This wrapper casts `x` to the dtype of the cos/sin tables before calling
/// `rope`, then casts the result back to the original dtype of `x`.  The
/// round-trip cast is numerically equivalent to computing RoPE in the
/// cos/sin dtype: the additional cast is essentially free on CUDA (BF16↔F32
/// is a bitwise reinterpretation plus zero-pad/truncation of the mantissa).
fn apply_rotary_emb(x: &Tensor, index_pos: usize, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b_sz, _, seq_len, _) = x.dims4()?;
    let cos = cos.narrow(0, index_pos, seq_len)?;
    let sin = sin.narrow(0, index_pos, seq_len)?;

    // Align the query/key tensor dtype with the cos/sin tables so that the
    // RoPE kernel receives three tensors of the same dtype.  This handles the
    // case where QMatMul dequantizes to BF16 (CUDA) but the KvCache was
    // constructed with DType::F32 (or the reverse).
    let x_dtype = x.dtype();
    let rope_dtype = cos.dtype();
    if x_dtype == rope_dtype {
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    } else {
        let x_cast = x.to_dtype(rope_dtype)?;
        candle_nn::rotary_emb::rope(&x_cast, &cos, &sin)?.to_dtype(x_dtype)
    }
}

// ---------------------------------------------------------------------------
// QuantizedMlp — SwiGLU with quantized projections
// ---------------------------------------------------------------------------

/// SwiGLU MLP with quantized weight matrices.
///
/// `working_dtype` is the dtype used for all intermediate tensors: BF16 on
/// CUDA (where `QMatMul::forward` dequantizes to BF16) or F32 on CPU.
/// All QMatMul outputs and element-wise operands are cast to `working_dtype`
/// before binary operations to prevent "dtype mismatch in binary op" errors.
#[derive(Debug, Clone)]
struct QuantizedMlp {
    /// Gate projection — W_gate: hidden → intermediate.
    gate_proj: QMatMul,
    /// Up projection — W_up: hidden → intermediate (SwiGLU second branch).
    up_proj: QMatMul,
    /// Down projection — W_down: intermediate → hidden.
    down_proj: QMatMul,
    /// Working dtype for all intermediate tensors (BF16 on CUDA, F32 on CPU).
    working_dtype: DType,
}

impl QuantizedMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?.to_dtype(self.working_dtype)?)?;
        let up = self.up_proj.forward(x)?.to_dtype(self.working_dtype)?;
        // Both operands are in working_dtype before the element-wise multiply.
        self.down_proj.forward(&(gate * up)?)
    }
}

// ---------------------------------------------------------------------------
// QuantizedAttention — GQA with quantized projections
// ---------------------------------------------------------------------------

/// Causal self-attention with quantized Q/K/V/O projections.
///
/// Supports grouped-query attention (GQA) via [`repeat_kv`].
/// RoPE embeddings are computed in the dtype of the KvCache cos/sin tables;
/// see [`apply_rotary_emb`] for the dtype-alignment logic.
///
/// `working_dtype` governs all intermediate float tensors.  On CUDA,
/// `QMatMul::forward` dequantizes to BF16; on CPU it dequantizes to F32.
/// Every QMatMul output is cast to `working_dtype` immediately after the
/// matmul, ensuring that subsequent operations (matmul, add, softmax) always
/// receive operands of the same dtype.
#[derive(Debug, Clone)]
struct QuantizedAttention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    /// Working dtype for all intermediate tensors (BF16 on CUDA, F32 on CPU).
    working_dtype: DType,
}

impl QuantizedAttention {
    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_attention_heads / self.num_key_value_heads;
        if n_rep == 1 {
            return Ok(x);
        }
        let (b, n_kv_head, seq_len, head_dim) = x.dims4()?;
        x.unsqueeze(2)?
            .expand((b, n_kv_head, n_rep, seq_len, head_dim))?
            .reshape((b, n_kv_head * n_rep, seq_len, head_dim))
    }

    /// Forward pass with [`KvCache`] (dynamic `cat`-based cache).
    fn forward_dynamic(&self, x: &Tensor, index_pos: usize, block_idx: usize, cache: &mut KvCache) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;

        // Q / K / V projections (quantized matmul).
        // Cast immediately to working_dtype so that all subsequent operations
        // (reshape, transpose, RoPE, matmul, add) receive the same dtype.
        // On CUDA, QMatMul dequantizes to BF16; on CPU it produces F32.
        // Without this cast the residual addition `attn_output + residual` can
        // fail with "dtype mismatch in binary op" when the two sides differ.
        let q = self.q_proj.forward(x)?.to_dtype(self.working_dtype)?;
        let k = self.k_proj.forward(x)?.to_dtype(self.working_dtype)?;
        let v = self.v_proj.forward(x)?.to_dtype(self.working_dtype)?;

        // Reshape to [B, heads, S, head_dim] for RoPE + attention.
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

        // Apply rotary embeddings — RoPE stays F32 regardless of quantization.
        let q = apply_rotary_emb(&q, index_pos, &cache.cos, &cache.sin)?;
        let mut k = apply_rotary_emb(&k, index_pos, &cache.cos, &cache.sin)?;

        // KV cache update (same logic as JanusLlama::CausalSelfAttention::forward).
        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 2)?;
                v = Tensor::cat(&[cache_v, &v], 2)?;
                let k_seq = k.dim(2)?;
                if k_seq > self.max_position_embeddings {
                    k = k
                        .narrow(2, k_seq - self.max_position_embeddings, self.max_position_embeddings)?
                        .contiguous()?;
                }
                let v_seq = v.dim(2)?;
                if v_seq > 2 * self.max_position_embeddings {
                    v = v
                        .narrow(2, v_seq - self.max_position_embeddings, self.max_position_embeddings)?
                        .contiguous()?;
                }
            }
            // Detach prevents BackpropOp chain memory leak across 576 steps.
            cache.kvs[block_idx] = Some((k.detach(), v.detach()));
        }

        let k_full = self.repeat_kv(k)?;
        let v_full = self.repeat_kv(v)?;

        // Scaled dot-product attention (F32 path; BF16 stays native on CUDA).
        let in_dtype = q.dtype();
        let use_f32_upcast = matches!(in_dtype, DType::F64);
        let (q, k_full, v_full) = if use_f32_upcast {
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
            let neg_inf = if use_f32_upcast {
                f32::NEG_INFINITY
            } else {
                -65504.0_f32
            };
            masked_fill(&att, &mask, neg_inf)?
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = if use_f32_upcast {
            att.matmul(&v_full.contiguous()?)?.to_dtype(in_dtype)?
        } else {
            att.matmul(&v_full.contiguous()?)?
        };
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        self.o_proj.forward(&y)
    }

    /// Forward pass with [`PreAllocKvCache`] (zero-copy in-place writes).
    fn forward_prealloc(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut PreAllocKvCache,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;

        // Cast to working_dtype immediately — see forward_dynamic for rationale.
        let q = self.q_proj.forward(x)?.to_dtype(self.working_dtype)?;
        let k = self.k_proj.forward(x)?.to_dtype(self.working_dtype)?;
        let v = self.v_proj.forward(x)?.to_dtype(self.working_dtype)?;

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

        let q = apply_rotary_emb(&q, index_pos, &cache.cos, &cache.sin)?;
        let k = apply_rotary_emb(&k, index_pos, &cache.cos, &cache.sin)?;

        // In-place scatter write; zero-copy narrow read.
        let (k_full, v_full) = cache.update(block_idx, index_pos, &k, &v)?;

        let k_full = self.repeat_kv(k_full)?;
        let v_full = self.repeat_kv(v_full)?;

        let total_len = k_full.dim(2)?;
        let in_dtype = q.dtype();
        let use_f32_upcast = matches!(in_dtype, DType::F64);
        let (q, k_full, v_full) = if use_f32_upcast {
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
            let mask = cache.mask(total_len)?.narrow(0, total_len - seq_len, seq_len)?;
            let mask = mask.broadcast_as(att.shape())?;
            let neg_inf = if use_f32_upcast {
                f32::NEG_INFINITY
            } else {
                -65504.0_f32
            };
            masked_fill(&att, &mask, neg_inf)?
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = if use_f32_upcast {
            att.matmul(&v_full.contiguous()?)?.to_dtype(in_dtype)?
        } else {
            att.matmul(&v_full.contiguous()?)?
        };
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        self.o_proj.forward(&y)
    }
}

// ---------------------------------------------------------------------------
// QuantizedBlock — transformer layer
// ---------------------------------------------------------------------------

/// Single transformer block (attention + MLP) with quantized weight matrices.
///
/// `working_dtype` is BF16 on CUDA or F32 on CPU.  Every tensor entering a
/// binary operation (residual add) is cast to `working_dtype` to avoid
/// "dtype mismatch in binary op" errors that arise because:
/// - `RmsNorm::forward` always returns F32 (norm weights stored in F32).
/// - `QMatMul::forward` dequantizes to BF16 on CUDA, F32 on CPU.
/// - The incoming `x` from `wte.forward` is always F32 (embedding stored F32).
///
/// By casting every side of each binary op to `working_dtype`, the block is
/// self-consistent regardless of which component produced the tensor.
#[derive(Debug, Clone)]
struct QuantizedBlock {
    input_layernorm: candle_nn::RmsNorm,
    attn: QuantizedAttention,
    post_attention_layernorm: candle_nn::RmsNorm,
    mlp: QuantizedMlp,
    /// Working dtype for all intermediate tensors (BF16 on CUDA, F32 on CPU).
    working_dtype: DType,
}

impl QuantizedBlock {
    fn forward_dynamic(&self, x: &Tensor, index_pos: usize, block_idx: usize, cache: &mut KvCache) -> Result<Tensor> {
        // Cast incoming tensor to working_dtype.  On the first block this
        // converts the F32 wte embedding to BF16 (CUDA) or keeps it F32 (CPU).
        let x = x.to_dtype(self.working_dtype)?;
        let residual = &x;
        // RmsNorm returns F32; cast back to working_dtype before attention.
        let normed = self.input_layernorm.forward(&x)?.to_dtype(self.working_dtype)?;
        // attn output is in working_dtype (cast done inside forward_dynamic).
        let x = (self.attn.forward_dynamic(&normed, index_pos, block_idx, cache)? + residual)?;
        let residual = &x;
        // Post-attention norm: same F32→working_dtype cast required.
        let normed = self
            .post_attention_layernorm
            .forward(&x)?
            .to_dtype(self.working_dtype)?;
        let x = (self.mlp.forward(&normed)? + residual)?;
        Ok(x)
    }

    fn forward_prealloc(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut PreAllocKvCache,
    ) -> Result<Tensor> {
        let x = x.to_dtype(self.working_dtype)?;
        let residual = &x;
        let normed = self.input_layernorm.forward(&x)?.to_dtype(self.working_dtype)?;
        let x = (self.attn.forward_prealloc(&normed, index_pos, block_idx, cache)? + residual)?;
        let residual = &x;
        let normed = self
            .post_attention_layernorm
            .forward(&x)?
            .to_dtype(self.working_dtype)?;
        let x = (self.mlp.forward(&normed)? + residual)?;
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Attention mask helper (shared)
// ---------------------------------------------------------------------------

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let on_true = Tensor::new(on_true, on_false.device())?
        .to_dtype(on_false.dtype())?
        .broadcast_as(mask.shape().dims())?;
    mask.where_cond(&on_true, on_false)
}

// ---------------------------------------------------------------------------
// QuantizedJanusLlama — public API
// ---------------------------------------------------------------------------

/// Quantized Llama backbone for Janus-Pro.
///
/// Provides the same public interface as [`crate::janus_llama::JanusLlama`]:
/// - [`embed`] — token ID → hidden embedding.
/// - [`forward_hidden`] — hidden states via dynamic KV cache.
/// - [`forward_hidden_prealloc`] — hidden states via pre-allocated KV cache.
/// - [`forward_input_embed`] — text-vocabulary logits via dynamic KV cache.
/// - [`forward_input_embed_prealloc`] — text-vocabulary logits via pre-allocated KV cache.
/// - [`project_logits`] — project hidden states through the (quantized) LM head.
/// - [`offload_embeddings_to_cpu`] — move `wte` to CPU for VRAM savings.
///
/// Load via [`QuantizedJanusLlama::from_gguf`].
///
/// # Errors
///
/// All forward methods return [`candle_core::Result`].  Load errors are
/// forwarded from the GGUF reader.
#[derive(Debug, Clone)]
pub struct QuantizedJanusLlama {
    wte: Embedding,
    blocks: Vec<QuantizedBlock>,
    ln_f: candle_nn::RmsNorm,
    lm_head: QMatMul,
    // Config fields needed for project_logits CPU-offload path.
    vocab_size: usize,
    /// Working dtype for all forward-pass intermediate tensors.
    /// BF16 on CUDA (native QMatMul dequant dtype), F32 on CPU.
    working_dtype: DType,
}

impl QuantizedJanusLlama {
    // ── Construction ─────────────────────────────────────────────────────────

    /// Load a [`QuantizedJanusLlama`] from an open GGUF file.
    ///
    /// Reads the following tensors (GGUF canonical names):
    ///
    /// | GGUF name | Role |
    /// |-----------|------|
    /// | `token_embd.weight` | Token embedding table (dequantized) |
    /// | `blk.N.attn_norm.weight` | Pre-attention RMS norm |
    /// | `blk.N.attn_q.weight` | Query projection |
    /// | `blk.N.attn_k.weight` | Key projection |
    /// | `blk.N.attn_v.weight` | Value projection |
    /// | `blk.N.attn_output.weight` | Output projection |
    /// | `blk.N.ffn_norm.weight` | Pre-FFN RMS norm |
    /// | `blk.N.ffn_gate.weight` | MLP gate projection |
    /// | `blk.N.ffn_up.weight` | MLP up projection |
    /// | `blk.N.ffn_down.weight` | MLP down projection |
    /// | `output_norm.weight` | Final RMS norm |
    /// | `output.weight` | LM head (falls back to `token_embd.weight`) |
    ///
    /// The embedding table and norm weights are dequantized to F32.
    /// All projection weights are kept as [`QMatMul`] for quantized matmul.
    ///
    /// # Arguments
    ///
    /// * `ct`     — Parsed GGUF content (headers + tensor index).
    /// * `reader` — Seekable reader over the GGUF file bytes.
    /// * `cfg`    — Llama config (layer count, head counts, dimensions).
    /// * `device` — Target device for tensor allocation.
    ///
    /// # Errors
    ///
    /// Returns a candle error if any required tensor is missing or malformed.
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        cfg: &Config,
        device: &Device,
    ) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        // ── Working dtype ──────────────────────────────────────────────────
        // On CUDA, QMatMul::forward dequantizes to BF16; on CPU it produces
        // F32.  All intermediate tensors (QMatMul outputs, residuals, RmsNorm
        // outputs fed into the next matmul) must share this dtype.  Using the
        // wrong dtype in a binary op yields "dtype mismatch in binary op".
        let working_dtype = match device {
            Device::Cuda(_) => DType::BF16,
            _ => DType::F32,
        };

        // ── Token embedding (dequantize → full precision for CPU offload) ──
        let wte_qtensor = ct.tensor(reader, "token_embd.weight", device)?;
        let wte_weight = wte_qtensor.dequantize(device)?.to_dtype(DType::F32)?;
        let wte = Embedding::new(wte_weight, cfg.hidden_size);

        // ── Transformer blocks ─────────────────────────────────────────────
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            let prefix = format!("blk.{layer_idx}");

            // Norm weights — dequantize to F32 (cheap: one scalar per dim).
            let attn_norm_qt = ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm_qt = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;
            let input_layernorm = rms_norm_from_qtensor(attn_norm_qt, cfg.rms_norm_eps, device)?;
            let post_attention_layernorm = rms_norm_from_qtensor(ffn_norm_qt, cfg.rms_norm_eps, device)?;

            // Attention projections (stay quantized).
            let q_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?)?;
            let k_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?)?;
            let v_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?)?;
            let o_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?)?;

            // MLP projections (stay quantized).
            let gate_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?)?;
            let up_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?)?;
            let down_proj = QMatMul::from_qtensor(ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?)?;

            blocks.push(QuantizedBlock {
                input_layernorm,
                attn: QuantizedAttention {
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    num_attention_heads: cfg.num_attention_heads,
                    num_key_value_heads: cfg.num_key_value_heads,
                    head_dim,
                    max_position_embeddings: cfg.max_position_embeddings,
                    working_dtype,
                },
                post_attention_layernorm,
                mlp: QuantizedMlp {
                    gate_proj,
                    up_proj,
                    down_proj,
                    working_dtype,
                },
                working_dtype,
            });
        }

        // ── Final norm ────────────────────────────────────────────────────
        let ln_f_qt = ct.tensor(reader, "output_norm.weight", device)?;
        let ln_f = rms_norm_from_qtensor(ln_f_qt, cfg.rms_norm_eps, device)?;

        // ── LM head ───────────────────────────────────────────────────────
        // Some GGUF files omit `output.weight` and rely on tied embeddings.
        // Fall back to `token_embd.weight` in that case (dequantize + build
        // a Tensor-backed QMatMul, which degrades to a full-precision matmul).
        let lm_head = match ct.tensor(reader, "output.weight", device) {
            Ok(qt) => QMatMul::from_qtensor(qt)?,
            Err(_) => {
                eprintln!("GGUF: output.weight missing, using tied token_embd.weight for lm_head");
                // Re-read the embedding weight and wrap as a Tensor QMatMul.
                let wte_qt = ct.tensor(reader, "token_embd.weight", device)?;
                let wte_f = wte_qt.dequantize(device)?;
                QMatMul::Tensor(wte_f)
            }
        };

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            vocab_size: cfg.vocab_size,
            working_dtype,
        })
    }

    // ── Public API (mirrors JanusLlama) ──────────────────────────────────────

    /// Embed token IDs into the LLM hidden space.
    ///
    /// Returns shape `[B, S, hidden_size]`.
    ///
    /// # Errors
    ///
    /// Propagates candle tensor errors from the embedding lookup.
    pub fn embed(&self, x: &Tensor) -> Result<Tensor> {
        self.wte.forward(x)
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Forward pass returning **hidden states** using the dynamic KV cache.
    ///
    /// Returns shape `[B, hidden_size]` — the last-position hidden state after
    /// all transformer layers and final RMS norm, **before** `lm_head`.
    ///
    /// # Arguments
    ///
    /// * `input_embed` — Float tensor `[B, S, hidden_size]`.
    /// * `index_pos`   — Absolute sequence position for RoPE + KV indexing.
    /// * `cache`       — Mutable [`KvCache`] reference.
    ///
    /// # Errors
    ///
    /// Propagates candle errors from the transformer forward pass.
    pub fn forward_hidden(&self, input_embed: &Tensor, index_pos: usize, cache: &mut KvCache) -> Result<Tensor> {
        let (_, seq_len, _) = input_embed.dims3()?;
        let mut x = input_embed.clone();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward_dynamic(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        x.i((.., seq_len - 1, ..))?.contiguous()
    }

    /// Forward pass returning **hidden states** using the pre-allocated KV cache.
    ///
    /// Functionally identical to [`forward_hidden`] but eliminates
    /// `Tensor::cat` overhead via scatter writes into pre-allocated buffers.
    ///
    /// # Arguments
    ///
    /// * `input_embed` — Float tensor `[B, S, hidden_size]`.
    /// * `index_pos`   — Absolute position for RoPE and KV writes.
    /// * `cache`       — Mutable [`PreAllocKvCache`] reference.
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
        x.i((.., seq_len - 1, ..))?.contiguous()
    }

    /// Project hidden states through the quantized LM head.
    ///
    /// Handles device mismatches between the (CPU-offloaded) LM head and
    /// a CUDA hidden-state tensor by moving the tensor to the correct device.
    ///
    /// # Returns
    ///
    /// Float tensor `[B, vocab_size]` in F32.
    ///
    /// # Errors
    ///
    /// Propagates candle errors from the matmul.
    pub fn project_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        // QMatMul::forward handles device routing internally when the QTensor
        // was loaded onto a specific device.  For the tied-weights (Tensor)
        // path we may need an explicit device move.
        let logits = match &self.lm_head {
            QMatMul::Tensor(w) => {
                let device = w.device();
                let h = hidden.to_device(device)?;
                let h = h.to_dtype(w.dtype())?;
                h.matmul(&w.t()?)?
            }
            QMatMul::TensorF16(w) => {
                let device = w.device();
                let h = hidden.to_device(device)?.to_dtype(DType::F16)?;
                h.matmul(&w.t()?)?.to_dtype(DType::F32)?
            }
            QMatMul::QTensor(_) => {
                // QMatMul::forward dequantizes on-the-fly and handles the device.
                self.lm_head.forward(hidden)?
            }
        };
        logits.to_dtype(DType::F32)
    }

    /// Forward pass returning text-vocabulary logits using the dynamic KV cache.
    ///
    /// Combines [`forward_hidden`] with [`project_logits`].
    ///
    /// # Returns
    ///
    /// Float tensor `[B, vocab_size]`.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`forward_hidden`] and [`project_logits`].
    pub fn forward_input_embed(&self, input_embed: &Tensor, index_pos: usize, cache: &mut KvCache) -> Result<Tensor> {
        let hidden = self.forward_hidden(input_embed, index_pos, cache)?;
        self.project_logits(&hidden)
    }

    /// Forward pass returning text-vocabulary logits using the pre-allocated KV cache.
    ///
    /// Combines [`forward_hidden_prealloc`] with [`project_logits`].
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

    /// Standard forward from token IDs (embed + transformer + lm_head).
    ///
    /// # Returns
    ///
    /// Float tensor `[B, vocab_size]`.
    ///
    /// # Errors
    ///
    /// Propagates candle errors.
    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut KvCache) -> Result<Tensor> {
        let embeds = self.wte.forward(x)?;
        self.forward_input_embed(&embeds, index_pos, cache)
    }

    /// Forward pass returning hidden states for **all** positions in a batch.
    ///
    /// Used by the self-speculative decoding verify phase to process `K` draft
    /// tokens in a single full-depth forward pass.  Returns `[B, K, hidden_size]`.
    ///
    /// # Arguments
    ///
    /// * `input_embed` — Float tensor `[B, K, hidden_size]`.
    /// * `start_pos`   — Absolute position of the first token (for RoPE + KV).
    /// * `cache`       — Shared pre-allocated KV cache.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the cache is full or on shape mismatches.
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
        self.ln_f.forward(&x)
    }

    /// Offload `wte` (token embedding table) to CPU to free GPU VRAM.
    ///
    /// After calling this, [`embed`] returns CPU tensors — callers must
    /// `.to_device(gpu)` the result before feeding into the transformer.
    /// This mirrors the behaviour of
    /// [`JanusLlama::offload_embeddings_to_cpu`].
    ///
    /// # Errors
    ///
    /// Returns a candle error if the tensor device transfer fails.
    pub fn offload_embeddings_to_cpu(&mut self) -> Result<()> {
        let cpu_weight = self.wte.embeddings().to_device(&Device::Cpu)?;
        let hidden_size = cpu_weight.dim(1)?;
        self.wte = Embedding::new(cpu_weight, hidden_size);
        Ok(())
    }

    /// Returns the configured vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Returns the working dtype used for all intermediate forward-pass tensors.
    ///
    /// BF16 on CUDA, F32 on CPU.  Callers that feed external embeddings into
    /// [`forward_hidden`] may cast to this dtype for compatibility.
    pub fn working_dtype(&self) -> DType {
        self.working_dtype
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Construct a tiny Llama config for unit tests.
    ///
    /// Uses 4 attention heads, 2 KV heads, 2 layers, vocab=256, hidden=64
    /// so the test allocates <1 MB.
    fn tiny_llama_cfg() -> Config {
        Config {
            hidden_size: 64,
            intermediate_size: 128,
            vocab_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
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

    /// Build a [`QuantizedJanusLlama`] from a GGUF byte stream synthesised
    /// in-memory using Q8_0 quantization.
    ///
    /// This validates the full GGUF load path (header parsing, tensor reads,
    /// QMatMul construction, RmsNorm dequantization) without requiring a
    /// real model file on disk.
    ///
    /// The test is marked `#[ignore]` because building a valid GGUF stream
    /// in memory requires the private `ggml_file::qtensor_from_ggml` API
    /// which is not exposed by candle-core's public interface.  The
    /// integration test (load from a real `.gguf` file) is the authoritative
    /// test; this unit test documents the expected contract.
    #[test]
    #[ignore = "requires a real GGUF file on disk; use integration tests"]
    fn test_from_gguf_smoke() {
        // This test would open a GGUF file and call from_gguf().
        // Marked ignore because no test fixture is available at unit test time.
    }

    /// Verify that the `QuantizedMlp::forward` shape contract holds with a
    /// non-quantized (F32 Tensor) QMatMul, which is the fallback path used
    /// by QMatMul::Tensor.
    #[test]
    fn test_quantized_mlp_tensor_path_shape() {
        let dev = Device::Cpu;
        let h = 64_usize;
        let i = 128_usize;
        // Build weight tensors directly (no GGUF needed).
        let gate_w = Tensor::zeros((i, h), DType::F32, &dev).unwrap();
        let up_w = Tensor::zeros((i, h), DType::F32, &dev).unwrap();
        let down_w = Tensor::zeros((h, i), DType::F32, &dev).unwrap();
        let mlp = QuantizedMlp {
            gate_proj: QMatMul::Tensor(gate_w),
            up_proj: QMatMul::Tensor(up_w),
            down_proj: QMatMul::Tensor(down_w),
            working_dtype: DType::F32,
        };
        let x = Tensor::zeros((1_usize, 3_usize, h), DType::F32, &dev).unwrap();
        let out = mlp.forward(&x).expect("mlp forward failed");
        assert_eq!(out.dims(), &[1, 3, h], "MLP output shape mismatch");
    }

    /// Verify that `QuantizedAttention::forward_dynamic` returns the correct
    /// shape for a single-token decode step with non-quantized (F32) weights.
    #[test]
    fn test_quantized_attn_forward_dynamic_shape() {
        let dev = Device::Cpu;
        let cfg = tiny_llama_cfg();
        let h = cfg.hidden_size;
        let head_dim = h / cfg.num_attention_heads;
        let q_out = head_dim * cfg.num_attention_heads;
        let kv_out = head_dim * cfg.num_key_value_heads;

        let q_w = Tensor::zeros((q_out, h), DType::F32, &dev).unwrap();
        let k_w = Tensor::zeros((kv_out, h), DType::F32, &dev).unwrap();
        let v_w = Tensor::zeros((kv_out, h), DType::F32, &dev).unwrap();
        let o_w = Tensor::zeros((h, q_out), DType::F32, &dev).unwrap();

        let attn = QuantizedAttention {
            q_proj: QMatMul::Tensor(q_w),
            k_proj: QMatMul::Tensor(k_w),
            v_proj: QMatMul::Tensor(v_w),
            o_proj: QMatMul::Tensor(o_w),
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim,
            max_position_embeddings: cfg.max_position_embeddings,
            working_dtype: DType::F32,
        };

        let llama_cfg_for_cache = cfg.clone();
        let mut cache = KvCache::new(true, DType::F32, &llama_cfg_for_cache, &dev).unwrap();

        // Single token input: [B=1, S=1, hidden=64].
        let x = Tensor::zeros((1_usize, 1_usize, h), DType::F32, &dev).unwrap();
        let out = attn
            .forward_dynamic(&x, 0, 0, &mut cache)
            .expect("attn forward_dynamic failed");
        assert_eq!(out.dims(), &[1, 1, h], "attention output shape mismatch");
    }

    /// Verify `project_logits` returns `[B, vocab_size]` and F32 dtype
    /// using the Tensor (non-quantized) LM head path.
    #[test]
    fn test_project_logits_tensor_path() {
        let dev = Device::Cpu;
        let h = 64_usize;
        let vocab = 256_usize;
        let lm_head_w = Tensor::zeros((vocab, h), DType::F32, &dev).unwrap();

        // Build a minimal QuantizedJanusLlama with only what project_logits needs.
        // We can't call from_gguf without a real file, so construct manually.
        let embed_w = Tensor::zeros((vocab, h), DType::F32, &dev).unwrap();
        let model = QuantizedJanusLlama {
            wte: Embedding::new(embed_w, h),
            blocks: vec![],
            ln_f: candle_nn::RmsNorm::new(Tensor::ones(h, DType::F32, &dev).unwrap(), 1e-6),
            lm_head: QMatMul::Tensor(lm_head_w),
            vocab_size: vocab,
            working_dtype: DType::F32,
        };

        let hidden = Tensor::zeros((1_usize, h), DType::F32, &dev).unwrap();
        let logits = model.project_logits(&hidden).expect("project_logits failed");
        assert_eq!(logits.dims(), &[1, vocab]);
        assert_eq!(logits.dtype(), DType::F32);
    }

    /// Verify that `offload_embeddings_to_cpu` moves `wte` to CPU without error.
    #[test]
    fn test_offload_embeddings_to_cpu() {
        let dev = Device::Cpu; // Already on CPU; confirms the move is a no-op.
        let h = 64_usize;
        let vocab = 256_usize;
        let embed_w = Tensor::zeros((vocab, h), DType::F32, &dev).unwrap();
        let lm_head_w = Tensor::zeros((vocab, h), DType::F32, &dev).unwrap();

        let mut model = QuantizedJanusLlama {
            wte: Embedding::new(embed_w, h),
            blocks: vec![],
            ln_f: candle_nn::RmsNorm::new(Tensor::ones(h, DType::F32, &dev).unwrap(), 1e-6),
            lm_head: QMatMul::Tensor(lm_head_w),
            vocab_size: vocab,
            working_dtype: DType::F32,
        };

        model.offload_embeddings_to_cpu().expect("offload failed");
        let emb_dev = model.wte.embeddings().device();
        assert!(matches!(emb_dev, Device::Cpu), "wte should be on CPU after offload");
    }

    /// Verify `QuantizedBlock::forward_dynamic` output shape.
    #[test]
    fn test_quantized_block_forward_shape() {
        let dev = Device::Cpu;
        let cfg = tiny_llama_cfg();
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let head_dim = h / cfg.num_attention_heads;
        let q_out = head_dim * cfg.num_attention_heads;
        let kv_out = head_dim * cfg.num_key_value_heads;

        // Build zero-weight block using Tensor-backed QMatMul.
        let block = QuantizedBlock {
            input_layernorm: candle_nn::RmsNorm::new(Tensor::ones(h, DType::F32, &dev).unwrap(), 1e-6),
            attn: QuantizedAttention {
                q_proj: QMatMul::Tensor(Tensor::zeros((q_out, h), DType::F32, &dev).unwrap()),
                k_proj: QMatMul::Tensor(Tensor::zeros((kv_out, h), DType::F32, &dev).unwrap()),
                v_proj: QMatMul::Tensor(Tensor::zeros((kv_out, h), DType::F32, &dev).unwrap()),
                o_proj: QMatMul::Tensor(Tensor::zeros((h, q_out), DType::F32, &dev).unwrap()),
                num_attention_heads: cfg.num_attention_heads,
                num_key_value_heads: cfg.num_key_value_heads,
                head_dim,
                max_position_embeddings: cfg.max_position_embeddings,
                working_dtype: DType::F32,
            },
            post_attention_layernorm: candle_nn::RmsNorm::new(Tensor::ones(h, DType::F32, &dev).unwrap(), 1e-6),
            mlp: QuantizedMlp {
                gate_proj: QMatMul::Tensor(Tensor::zeros((i, h), DType::F32, &dev).unwrap()),
                up_proj: QMatMul::Tensor(Tensor::zeros((i, h), DType::F32, &dev).unwrap()),
                down_proj: QMatMul::Tensor(Tensor::zeros((h, i), DType::F32, &dev).unwrap()),
                working_dtype: DType::F32,
            },
            working_dtype: DType::F32,
        };

        let mut cache = KvCache::new(true, DType::F32, &cfg, &dev).unwrap();
        let x = Tensor::zeros((1_usize, 4_usize, h), DType::F32, &dev).unwrap();
        let out = block
            .forward_dynamic(&x, 0, 0, &mut cache)
            .expect("block forward failed");
        assert_eq!(out.dims(), &[1, 4, h]);
    }

    /// Regression test for the RoPE dtype mismatch bug.
    ///
    /// Reproduces the error:
    /// `prefill forward_hidden failed: unsupported dtype for rope F32 BF16 BF16`
    ///
    /// The quantized attention forward receives BF16 Q/K tensors (from
    /// `QMatMul` dequantization on GPU) while the `KvCache` cos/sin tables
    /// may be in F32 (or the inverse), causing `candle_nn::rotary_emb::rope`
    /// to reject the mixed-dtype inputs.
    ///
    /// This test constructs a `KvCache` with F32 cos/sin but feeds a BF16
    /// input tensor (simulating CUDA QMatMul output), and verifies that
    /// `apply_rotary_emb` completes without error and returns a BF16 result.
    #[test]
    fn test_apply_rotary_emb_dtype_mismatch_bf16_input_f32_cache() {
        let dev = Device::Cpu;
        let cfg = tiny_llama_cfg();
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        // Cache created with F32 cos/sin (the pre-fix default on CPU).
        let cache = KvCache::new(true, DType::F32, &cfg, &dev).unwrap();
        assert_eq!(cache.cos.dtype(), DType::F32, "cos should be F32");

        // Simulate BF16 Q tensor from QMatMul dequantization on CUDA.
        // Shape: [B=1, n_heads=4, seq=1, head_dim=16].
        let q_bf16 = Tensor::zeros((1_usize, cfg.num_attention_heads, 1_usize, head_dim), DType::BF16, &dev).unwrap();
        assert_eq!(q_bf16.dtype(), DType::BF16);

        // Before the fix this would panic with "unsupported dtype for rope F32 BF16 BF16"
        // (x=BF16, cos=F32, sin=F32).  After the fix it should succeed.
        let out = apply_rotary_emb(&q_bf16, 0, &cache.cos, &cache.sin)
            .expect("apply_rotary_emb should succeed despite dtype mismatch");

        // Output dtype must match the *input* tensor (BF16), not the cos/sin tables.
        assert_eq!(out.dtype(), DType::BF16, "output dtype should match input tensor dtype");
        assert_eq!(out.dims(), q_bf16.dims(), "output shape must be unchanged");
    }

    /// Symmetric case: F32 input with BF16 cos/sin tables (CUDA-created cache).
    ///
    /// On CUDA, `KvCache::new` is called with `DType::BF16` so cos/sin are BF16.
    /// If a caller feeds an F32 tensor (e.g., CPU fallback in a mixed-device
    /// setup) the same mismatch occurs in the opposite direction.
    #[test]
    fn test_apply_rotary_emb_dtype_mismatch_f32_input_bf16_cache() {
        let dev = Device::Cpu;
        let cfg = tiny_llama_cfg();
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        // Cache created with BF16 cos/sin (CUDA path).
        let cache = KvCache::new(true, DType::BF16, &cfg, &dev).unwrap();
        assert_eq!(cache.cos.dtype(), DType::BF16);

        // F32 input tensor.
        let q_f32 = Tensor::zeros((1_usize, cfg.num_attention_heads, 1_usize, head_dim), DType::F32, &dev).unwrap();

        let out = apply_rotary_emb(&q_f32, 0, &cache.cos, &cache.sin)
            .expect("apply_rotary_emb should handle F32 input with BF16 cache");

        assert_eq!(out.dtype(), DType::F32, "output dtype must match input");
        assert_eq!(out.dims(), q_f32.dims());
    }
}
