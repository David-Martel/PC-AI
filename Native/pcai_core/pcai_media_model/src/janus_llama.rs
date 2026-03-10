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

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, Module, VarBuilder};
use candle_transformers::models::llama::Config;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// KV Cache
// ---------------------------------------------------------------------------

/// KV cache with RoPE precomputed cos/sin tables.
///
/// Unlike candle's `llama::Cache`, all fields are accessible for optimization.
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
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
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
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &KvCache) -> Result<Tensor> {
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

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut KvCache,
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
        let mut v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;

        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
                let k_seq_len = k.dims()[1];
                if k_seq_len > self.max_position_embeddings {
                    k = k
                        .narrow(
                            D::Minus1,
                            k_seq_len - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                        .contiguous()?;
                }
                let v_seq_len = v.dims()[1];
                if v_seq_len > 2 * self.max_position_embeddings {
                    v = v
                        .narrow(
                            D::Minus1,
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

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        // Standard scaled dot-product attention.
        // On CUDA with BF16/F16, compute in native dtype to avoid F32 copies
        // that fragment the CUDA memory pool (KV cache grows each step).
        // On CPU with F32, no cast is needed either.
        let in_dtype = q.dtype();
        let use_f32_attn = matches!(in_dtype, DType::F64); // only upcast for F64
        let (q, k, v) = if use_f32_attn {
            (q.to_dtype(DType::F32)?, k.to_dtype(DType::F32)?, v.to_dtype(DType::F32)?)
        } else {
            (q, k, v)
        };
        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = if seq_len == 1 {
            att
        } else {
            let mask = cache.mask(seq_len)?.broadcast_as(att.shape())?;
            let neg_inf = if use_f32_attn { f32::NEG_INFINITY } else {
                // For BF16/F16, use a large negative value instead of infinity
                -65504.0_f32
            };
            masked_fill(&att, &mask, neg_inf)?
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = if use_f32_attn {
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        } else {
            att.matmul(&v.contiguous()?)?
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
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            max_position_embeddings: cfg.max_position_embeddings,
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
    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = (self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.post_attention_layernorm.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = Mlp::load(vb.pp("mlp"), cfg)?;
        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
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
    pub fn forward_hidden(
        &self,
        input_embed: &Tensor,
        index_pos: usize,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (_, seq_len, _) = input_embed.dims3()?;
        let mut x = input_embed.clone();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        // Extract last position: [B, S, H] → [B, H]
        x.i((.., seq_len - 1, ..))?.contiguous()
    }

    /// Forward pass returning text-vocabulary logits (includes lm_head).
    ///
    /// Returns shape `[B, vocab_size]` — equivalent to candle's
    /// `Llama::forward_input_embed`.
    pub fn forward_input_embed(
        &self,
        input_embed: &Tensor,
        index_pos: usize,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let hidden = self.forward_hidden(input_embed, index_pos, cache)?;
        let logits = self.lm_head.forward(&hidden)?;
        logits.to_dtype(DType::F32)
    }

    /// Standard forward from token IDs (embed + transformer + lm_head).
    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut KvCache) -> Result<Tensor> {
        let embeds = self.wte.forward(x)?;
        self.forward_input_embed(&embeds, index_pos, cache)
    }

    /// Offload `wte` and `lm_head` to CPU to free GPU VRAM.
    ///
    /// During image generation, `wte` is only used once (prefill) and `lm_head`
    /// is not used at all (we use `gen_head` instead).  Moving them to CPU
    /// saves ~800 MB VRAM on the 1B model.
    ///
    /// After calling this, `embed()` will return CPU tensors — callers must
    /// `.to_device(gpu)` the result before feeding into the transformer.
    pub fn offload_embeddings_to_cpu(&mut self) -> Result<()> {
        let cpu = Device::Cpu;
        let cpu_wte_weight = self.wte.embeddings().to_device(&cpu)?;
        let hidden_size = cpu_wte_weight.dim(1)?;
        self.wte = Embedding::new(cpu_wte_weight, hidden_size);

        let cpu_lm_weight = self.lm_head.weight().to_device(&cpu)?;
        self.lm_head = candle_nn::Linear::new(cpu_lm_weight, None);
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
        let ln_f =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
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

        let embed =
            Tensor::zeros((1_usize, 3_usize, 64_usize), DType::F32, &Device::Cpu).unwrap();
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

        let embed =
            Tensor::zeros((1_usize, 3_usize, 64_usize), DType::F32, &Device::Cpu).unwrap();
        let logits = llama.forward_input_embed(&embed, 0, &mut cache).unwrap();

        assert_eq!(logits.dims(), &[1, 256], "expected [B, vocab_size]");
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

        let embed =
            Tensor::zeros((1_usize, 3_usize, 64_usize), DType::F32, &Device::Cpu).unwrap();
        let _ = llama.forward_hidden(&embed, 0, &mut cache).unwrap();

        assert!(cache.memory_bytes() > 0, "KV cache should have allocated memory");
    }
}
