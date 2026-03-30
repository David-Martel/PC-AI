use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Linear, VarBuilder};
use qlora_rs::{dequantize_nf4, QLoraConfig, QLoraLayer, QuantizedLinear};
use serde::Deserialize;
use std::cell::Cell;

#[cfg(feature = "flash-attn")]
use candle_flash_attn as flash_attn_backend;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: Option<usize>,
    pub layer_types: Option<Vec<String>>,
}

#[derive(Debug)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    fn from_weight(weight: Tensor, eps: f64) -> Result<Self> {
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}

pub struct LoraLinear {
    base: Option<Linear>,
    lora_a: Option<Tensor>,
    lora_b: Option<Tensor>,
    scale: f64,
    dropout: f64,
    training: Cell<bool>,
    qlora: Option<QuantizedLinear>,
    qmatmul: Option<QMatMul>,
}

impl std::fmt::Debug for LoraLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoraLinear")
            .field("has_base", &self.base.is_some())
            .field("has_lora", &(self.lora_a.is_some() && self.lora_b.is_some()))
            .field("has_qlora", &self.qlora.is_some())
            .field("has_qmatmul", &self.qmatmul.is_some())
            .field("scale", &self.scale)
            .field("dropout", &self.dropout)
            .field("training", &self.training.get())
            .finish()
    }
}

impl LoraLinear {
    pub fn new(in_dim: usize, out_dim: usize, settings: LoraSettings, vb: VarBuilder) -> Result<Self> {
        // Match the base model's naming schema (e.g., q_proj.weight)
        let base_weight = vb.get((out_dim, in_dim), "weight")?;
        Self::new_with_base(base_weight, settings, vb)
    }

    pub fn new_with_base(base_weight: Tensor, settings: LoraSettings, vb: VarBuilder) -> Result<Self> {
        let base_weight = if settings.use_4bit && base_weight.dtype() != DType::F32 {
            base_weight.to_dtype(DType::F32)?
        } else {
            base_weight
        };
        let (out_dim, in_dim) = base_weight.dims2()?;
        if settings.use_4bit {
            let qlora_cfg = settings.qlora_config()?;
            let qlora = QuantizedLinear::from_weight_with_varbuilder(&base_weight, None, &qlora_cfg, vb)
                .map_err(|err| candle_core::Error::msg(err.to_string()))?;
            return Ok(Self {
                base: None,
                lora_a: None,
                lora_b: None,
                scale: qlora_cfg.scale(),
                dropout: settings.dropout,
                training: Cell::new(false),
                qlora: Some(qlora),
                qmatmul: None,
            });
        }

        let qmatmul = if settings.use_candle_qmatmul {
            let dtype = settings.candle_qmatmul_dtype.unwrap_or(GgmlDType::Q4_0);
            let qtensor = QTensor::quantize(&base_weight, dtype)?;
            Some(QMatMul::from_qtensor(qtensor)?)
        } else {
            None
        };
        let base = if qmatmul.is_some() {
            None
        } else {
            Some(candle_nn::Linear::new(base_weight, None))
        };

        let lora_a = if settings.r > 0 {
            Some(vb.pp("lora_a").get((settings.r, in_dim), "weight")?)
        } else {
            None
        };
        let lora_b = if settings.r > 0 {
            Some(vb.pp("lora_b").get((out_dim, settings.r), "weight")?)
        } else {
            None
        };
        Ok(Self {
            base,
            lora_a,
            lora_b,
            scale: settings.scale(),
            dropout: settings.dropout,
            training: Cell::new(false),
            qlora: None,
            qmatmul,
        })
    }

    pub fn merge(&mut self) -> Result<()> {
        // QLoRA merge: dequantize NF4 base, add LoRA delta, store as plain Linear.
        // LoRA weights live inside the QuantizedLinear, not in self.lora_a/b.
        if let Some(qlora) = self.qlora.take() {
            let ql = QLoraLayer::new(qlora);
            let base_f32 = dequantize_nf4(ql.quantized_weight(), ql.device())
                .map_err(|e| candle_core::Error::msg(e.to_string()))?;
            let (lora_a, lora_b) = ql.lora_weights();
            let scale = ql.lora_scale();
            let delta = lora_b.matmul(lora_a)?.affine(scale, 0.0)?;
            let merged = base_f32.add(&delta)?;
            self.base = Some(candle_nn::Linear::new(merged, None));
            self.lora_a = None;
            self.lora_b = None;
            return Ok(());
        }

        // Candle QMatMul merge: dequantize via f16 path then cast to f32,
        // add LoRA delta (stored in self.lora_a/b), store as plain Linear.
        if let Some(qmatmul) = self.qmatmul.take() {
            let base_f32 = qmatmul.dequantize_f16()?.to_dtype(candle_core::DType::F32)?;
            let merged = if let (Some(a), Some(b)) = (&self.lora_a, &self.lora_b) {
                let delta = b.matmul(a)?.affine(self.scale, 0.0)?;
                base_f32.add(&delta)?
            } else {
                base_f32
            };
            self.base = Some(candle_nn::Linear::new(merged, None));
            self.lora_a = None;
            self.lora_b = None;
            return Ok(());
        }

        // Standard (full-precision) LoRA merge into the existing base weight.
        if let (Some(a), Some(b)) = (&self.lora_a, &self.lora_b) {
            let base = self
                .base
                .as_ref()
                .ok_or_else(|| candle_core::Error::msg("Base linear missing"))?;
            let delta = b.matmul(a)?.affine(self.scale, 0.0)?;
            let new_weight = base.weight().add(&delta)?;
            self.base = Some(candle_nn::Linear::new(new_weight, None));
            self.lora_a = None;
            self.lora_b = None;
        }
        Ok(())
    }

    /// Set training mode. When `true`, LoRA dropout is applied during the
    /// forward pass.  Uses interior mutability (`Cell`) so this works through
    /// shared references, which is required because `Module::forward` takes
    /// `&self`.
    pub fn set_training(&self, training: bool) {
        self.training.set(training);
    }
}

impl Module for LoraLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(qlora) = &self.qlora {
            let x_fp32 = if x.dtype() != DType::F32 {
                x.to_dtype(DType::F32)?
            } else {
                x.clone()
            };
            let out = qlora
                .forward(&x_fp32)
                .map_err(|err| candle_core::Error::msg(err.to_string()))?;
            if out.dtype() != x.dtype() {
                return out.to_dtype(x.dtype());
            }
            return Ok(out);
        }
        let base_out = if let Some(qmatmul) = &self.qmatmul {
            let in_dtype = x.dtype();
            let x_f32 = if in_dtype != DType::F32 {
                x.to_dtype(DType::F32)?
            } else {
                x.clone()
            };
            let out = qmatmul.forward(&x_f32)?;
            if out.dtype() != in_dtype {
                out.to_dtype(in_dtype)?
            } else {
                out
            }
        } else {
            let base = self
                .base
                .as_ref()
                .ok_or_else(|| candle_core::Error::msg("Base linear missing"))?;
            base.forward(x)?
        };
        if let (Some(a), Some(b)) = (&self.lora_a, &self.lora_b) {
            let (b_sz, seq_len, hidden_dim) = x.dims3()?;
            let x_flat = x.reshape((b_sz * seq_len, hidden_dim))?;
            let lora_out = x_flat.matmul(&a.t()?)?.matmul(&b.t()?)?;
            let lora_out = lora_out.reshape((b_sz, seq_len, ()))?;
            let lora_out = lora_out.affine(self.scale, 0.0)?;
            // Apply dropout to the LoRA output when in training mode
            let lora_out = if self.dropout > 0.0 && self.training.get() {
                candle_nn::ops::dropout(&lora_out, self.dropout as f32)?
            } else {
                lora_out
            };
            Ok(base_out.add(&lora_out)?)
        } else {
            Ok(base_out)
        }
    }
}

#[derive(Debug)]
struct Mlp {
    gate_proj: LoraLinear,
    up_proj: LoraLinear,
    down_proj: LoraLinear,
    act: Activation,
}

impl Mlp {
    fn new(cfg: &Config, lora: LoraSettings, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;

        let gate_proj = LoraLinear::new(hidden_size, intermediate_size, lora, vb.pp("gate_proj"))?;
        let up_proj = LoraLinear::new(hidden_size, intermediate_size, lora, vb.pp("up_proj"))?;
        let down_proj = LoraLinear::new(intermediate_size, hidden_size, lora, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act: Activation::Gelu,
        })
    }

    fn new_with_base(
        _cfg: &Config,
        lora: LoraSettings,
        vb: VarBuilder,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
    ) -> Result<Self> {
        let gate_proj = LoraLinear::new_with_base(gate_weight, lora, vb.pp("gate_proj"))?;
        let up_proj = LoraLinear::new_with_base(up_weight, lora, vb.pp("up_proj"))?;
        let down_proj = LoraLinear::new_with_base(down_weight, lora, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act: Activation::Gelu,
        })
    }

    fn set_training(&self, training: bool) {
        self.gate_proj.set_training(training);
        self.up_proj.set_training(training);
        self.down_proj.set_training(training);
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = self.act.forward(&gate)?;
        let up = self.up_proj.forward(x)?;
        let x = (gate * up)?;
        self.down_proj.forward(&x)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    head_dim: usize,
    theta: f32,
    cache: std::sync::Arc<std::sync::Mutex<Option<(usize, Tensor, Tensor)>>>,
}

impl RotaryEmbedding {
    fn new(theta: f32, head_dim: usize) -> Self {
        Self {
            head_dim,
            theta,
            cache: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }

    fn forward(&self, x: &Tensor, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|e| candle_core::Error::msg(format!("RoPE cache mutex poisoned: {e}")))?;
        if let Some((cached_len, cos, sin)) = cache.as_ref() {
            if *cached_len >= seq_len && cos.device().is_cuda() == x.device().is_cuda() {
                return Ok((cos.narrow(0, 0, seq_len)?, sin.narrow(0, 0, seq_len)?));
            }
        }

        let device = x.device();
        let dim = self.head_dim;

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / self.theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::new(&inv_freq[..], &device)?.to_dtype(DType::F32)?;

        let t = Tensor::arange(0u32, seq_len as u32, &device)?.to_dtype(DType::F32)?;
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;

        let emb = Tensor::cat(&[&freqs, &freqs], 1)?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;

        *cache = Some((seq_len, cos.clone(), sin.clone()));
        Ok((cos, sin))
    }
}

fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, _seq_len, head_dim) = x.dims4()?;
    let x1 = x.narrow(3, 0, head_dim / 2)?;
    let x2 = x.narrow(3, head_dim / 2, head_dim / 2)?;

    let rotate_x = Tensor::cat(&[&x2.neg()?, &x1], 3)?;

    let cos = cos.to_dtype(x.dtype())?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.to_dtype(x.dtype())?.unsqueeze(0)?.unsqueeze(0)?;

    x.broadcast_mul(&cos)? + rotate_x.broadcast_mul(&sin)?
}

#[derive(Debug)]
struct Attention {
    q_proj: LoraLinear,
    k_proj: LoraLinear,
    v_proj: LoraLinear,
    o_proj: LoraLinear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: RotaryEmbedding,
    use_flash_attn: bool,
    #[expect(
        dead_code,
        reason = "sliding window attention config field reserved for Mistral/Gemma variants; not yet wired into the attention kernel"
    )]
    sliding_window: Option<usize>,
}

#[cfg(feature = "flash-attn")]
fn flash_attn(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
    flash_attn_backend::flash_attn(q, k, v, scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    Err(candle_core::Error::msg("flash-attn feature not enabled"))
}

#[derive(Debug, Clone, Copy)]
pub struct LoraSettings {
    pub r: usize,
    pub alpha: f64,
    pub dropout: f64,
    pub use_4bit: bool,
    pub use_candle_qmatmul: bool,
    pub candle_qmatmul_dtype: Option<GgmlDType>,
    pub qlora_block_size: usize,
    pub qlora_double_quant: bool,
    pub qlora_cache_dequantized: bool,
    pub qlora_qv_only: bool,
    pub use_flash_attn: bool,
}

impl LoraSettings {
    pub fn new(r: usize, alpha: f64, dropout: f64, use_4bit: bool) -> Self {
        Self {
            r,
            alpha,
            dropout,
            use_4bit,
            use_candle_qmatmul: false,
            candle_qmatmul_dtype: None,
            qlora_block_size: 64,
            qlora_double_quant: true,
            qlora_cache_dequantized: false,
            qlora_qv_only: false,
            use_flash_attn: false,
        }
    }

    pub fn enable_candle_qmatmul(&mut self, dtype: GgmlDType) {
        self.use_candle_qmatmul = true;
        self.candle_qmatmul_dtype = Some(dtype);
    }

    pub fn enable_flash_attn(&mut self) {
        self.use_flash_attn = true;
    }

    fn scale(&self) -> f64 {
        if self.r > 0 {
            self.alpha / self.r as f64
        } else {
            1.0
        }
    }

    fn qlora_alpha_usize(&self) -> usize {
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            1
        } else {
            self.alpha.round().max(1.0) as usize
        }
    }

    fn qlora_config(&self) -> Result<QLoraConfig> {
        if self.r == 0 {
            return Err(candle_core::Error::msg("QLoRA requires lora_r > 0"));
        }
        let mut cfg = if self.qlora_qv_only {
            QLoraConfig::preset_qv_bf16(self.r, self.qlora_alpha_usize())
        } else {
            QLoraConfig::preset_all_bf16(self.r, self.qlora_alpha_usize())
        };
        cfg.lora.dropout = self.dropout;
        cfg.quantization.block_size = self.qlora_block_size;
        cfg.quantization.double_quant = self.qlora_double_quant;
        cfg.cache_dequantized = self.qlora_cache_dequantized;
        Ok(cfg)
    }
}

fn load_weight(st: &MmapedSafetensors, name: &str, device: &Device, dtype: DType) -> Result<Tensor> {
    let mut tensor = st.load(name, device)?;
    if tensor.dtype() != dtype {
        tensor = tensor.to_dtype(dtype)?;
    }
    Ok(tensor)
}

impl Attention {
    fn new(cfg: &Config, lora: LoraSettings, sliding: bool, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj = LoraLinear::new(dim, num_heads * head_dim, lora, vb.pp("q_proj"))?;
        let k_proj = LoraLinear::new(dim, num_kv_heads * head_dim, lora, vb.pp("k_proj"))?;
        let v_proj = LoraLinear::new(dim, num_kv_heads * head_dim, lora, vb.pp("v_proj"))?;
        let o_proj = LoraLinear::new(num_heads * head_dim, dim, lora, vb.pp("o_proj"))?;
        let q_norm = Some(RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?);
        let k_norm = Some(RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?);

        let rotary_emb = RotaryEmbedding::new(cfg.rope_theta as f32, head_dim);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            use_flash_attn: lora.use_flash_attn,
            sliding_window: if sliding { cfg.sliding_window } else { None },
        })
    }

    fn new_with_base(
        cfg: &Config,
        lora: LoraSettings,
        sliding: bool,
        vb: VarBuilder,
        q_weight: Tensor,
        k_weight: Tensor,
        v_weight: Tensor,
        o_weight: Tensor,
        q_norm_weight: Option<Tensor>,
        k_norm_weight: Option<Tensor>,
    ) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_proj = LoraLinear::new_with_base(q_weight, lora, vb.pp("q_proj"))?;
        let k_proj = LoraLinear::new_with_base(k_weight, lora, vb.pp("k_proj"))?;
        let v_proj = LoraLinear::new_with_base(v_weight, lora, vb.pp("v_proj"))?;
        let o_proj = LoraLinear::new_with_base(o_weight, lora, vb.pp("o_proj"))?;
        let q_norm = match q_norm_weight {
            Some(weight) => Some(RmsNorm::from_weight(weight, cfg.rms_norm_eps)?),
            None => Some(RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?),
        };
        let k_norm = match k_norm_weight {
            Some(weight) => Some(RmsNorm::from_weight(weight, cfg.rms_norm_eps)?),
            None => Some(RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?),
        };

        let rotary_emb = RotaryEmbedding::new(cfg.rope_theta as f32, head_dim);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            use_flash_attn: lora.use_flash_attn,
            sliding_window: if sliding { cfg.sliding_window } else { None },
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden_size) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = if let Some(norm) = &self.q_norm {
            norm.forward(&q)?
        } else {
            q
        };
        let k = if let Some(norm) = &self.k_norm {
            norm.forward(&k)?
        } else {
            k
        };

        let (cos, sin) = self.rotary_emb.forward(&q, seq_len)?;

        let q = apply_rotary_emb(&q, &cos, &sin)?;
        let k = apply_rotary_emb(&k, &cos, &sin)?;

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_output = if self.use_flash_attn && x.device().is_cuda() {
            let q_flash = q.transpose(1, 2)?.contiguous()?;
            let k_flash = k.transpose(1, 2)?.contiguous()?;
            let v_flash = v.transpose(1, 2)?.contiguous()?;
            let out = flash_attn(&q_flash, &k_flash, &v_flash, scale as f32, true)?;
            out.transpose(1, 2)?
        } else {
            let attn_weights = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
            // Apply causal mask: prevent attending to future tokens.
            // tril2 produces a lower-triangular matrix of ones; positions above the
            // diagonal are zero.  We use where_cond to replace those positions with
            // -inf so that softmax drives them to zero probability.
            let attn_weights = if seq_len > 1 {
                let mask = Tensor::tril2(seq_len, attn_weights.dtype(), attn_weights.device())?
                    .broadcast_as(attn_weights.shape())?;
                let neg_inf = Tensor::new(f32::NEG_INFINITY, attn_weights.device())?
                    .to_dtype(attn_weights.dtype())?
                    .broadcast_as(attn_weights.shape())?;
                mask.where_cond(&attn_weights, &neg_inf)?
            } else {
                attn_weights
            };
            let attn_weights = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
            attn_weights.matmul(&v)?
        };

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }

    fn forward_with_cache(
        &self,
        x: &Tensor,
        cache: &mut Option<KvCacheEntry>,
        past_len: usize,
        kv_quant: KvCacheQuant,
        kv_max_len: Option<usize>,
        kv_store_on_cpu: bool,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden_size) = x.dims3()?;
        if seq_len != 1 {
            return self.forward(x);
        }

        let device = x.device();
        let cpu_device = Device::Cpu;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = if let Some(norm) = &self.q_norm {
            norm.forward(&q)?
        } else {
            q
        };
        let k = if let Some(norm) = &self.k_norm {
            norm.forward(&k)?
        } else {
            k
        };

        let (cos, sin) = self.rotary_emb.forward(&q, past_len + 1)?;
        let cos = cos.narrow(0, past_len, 1)?;
        let sin = sin.narrow(0, past_len, 1)?;

        let q = apply_rotary_emb(&q, &cos, &sin)?;
        let k = apply_rotary_emb(&k, &cos, &sin)?;

        let dtype = k.dtype();
        let (mut k, mut v) = if let Some(entry) = cache {
            let (cached_k, cached_v) = match entry {
                KvCacheEntry::Full { k, v } => (k.to_device(device)?, v.to_device(device)?),
                KvCacheEntry::Int8 {
                    k,
                    v,
                    k_scale,
                    v_scale,
                    dtype,
                } => {
                    let k_dev = k.to_device(device)?;
                    let v_dev = v.to_device(device)?;
                    (
                        dequantize_tensor_int8(&k_dev, *k_scale, *dtype)?,
                        dequantize_tensor_int8(&v_dev, *v_scale, *dtype)?,
                    )
                }
            };
            // O(seq_len) copy per decode step. Acceptable for FunctionGemma's short
            // routing sequences (typically < 100 tokens); no ring-buffer needed here.
            let k = Tensor::cat(&[&cached_k, &k], 2)?;
            let v = Tensor::cat(&[&cached_v, &v], 2)?;
            (k, v)
        } else {
            (k, v)
        };

        if let Some(max_len) = kv_max_len {
            let (_, _, seq_len, _) = k.dims4()?;
            if seq_len > max_len {
                let start = seq_len - max_len;
                k = k.narrow(2, start, max_len)?;
                v = v.narrow(2, start, max_len)?;
            }
        }
        let store_on_cpu = kv_store_on_cpu && !device.is_cpu();
        *cache = Some(match kv_quant {
            KvCacheQuant::Int8 => {
                let (k_q, k_scale) = quantize_tensor_int8(&k)?;
                let (v_q, v_scale) = quantize_tensor_int8(&v)?;
                let k_q = if store_on_cpu { k_q.to_device(&cpu_device)? } else { k_q };
                let v_q = if store_on_cpu { v_q.to_device(&cpu_device)? } else { v_q };
                KvCacheEntry::Int8 {
                    k: k_q,
                    v: v_q,
                    k_scale,
                    v_scale,
                    dtype,
                }
            }
            KvCacheQuant::None => {
                let k_store = if store_on_cpu {
                    k.to_device(&cpu_device)?
                } else {
                    k.clone()
                };
                let v_store = if store_on_cpu {
                    v.to_device(&cpu_device)?
                } else {
                    v.clone()
                };
                KvCacheEntry::Full { k: k_store, v: v_store }
            }
        });

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_output = if self.use_flash_attn && x.device().is_cuda() {
            let q_flash = q.transpose(1, 2)?.contiguous()?;
            let k_flash = k.transpose(1, 2)?.contiguous()?;
            let v_flash = v.transpose(1, 2)?.contiguous()?;
            let out = flash_attn(&q_flash, &k_flash, &v_flash, scale as f32, true)?;
            out.transpose(1, 2)?
        } else {
            let attn_weights = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
            let attn_weights = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
            attn_weights.matmul(&v)?
        };

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }

    /// Attention forward pass using a pre-allocated ring-buffer KV cache.
    ///
    /// Instead of `Tensor::cat` to grow the cache, this writes the new K/V
    /// directly into the pre-allocated buffer via `slice_set` (zero-alloc,
    /// zero-copy of existing cache).
    fn forward_with_prealloc_cache(
        &self,
        x: &Tensor,
        cache: &mut PreAllocKvCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _hidden_size) = x.dims3()?;
        if seq_len != 1 {
            return self.forward(x);
        }

        let device = x.device();
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [batch, heads, 1, head_dim]
        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = if let Some(norm) = &self.q_norm {
            norm.forward(&q)?
        } else {
            q
        };
        let k = if let Some(norm) = &self.k_norm {
            norm.forward(&k)?
        } else {
            k
        };

        // RoPE uses absolute position (not ring position)
        let rope_offset = cache.rope_offset();
        let (cos, sin) = self.rotary_emb.forward(&q, rope_offset + 1)?;
        let cos = cos.narrow(0, rope_offset, 1)?;
        let sin = sin.narrow(0, rope_offset, 1)?;

        let q = apply_rotary_emb(&q, &cos, &sin)?;
        let k = apply_rotary_emb(&k, &cos, &sin)?;

        // Ensure contiguous and matching dtype before slice_set (required by candle).
        // The pre-allocated buffer may be BF16 while projections output F32.
        let buf_dtype = cache.layers[layer_idx].k.dtype();
        let k = k.to_dtype(buf_dtype)?.contiguous()?;
        let v = v.to_dtype(buf_dtype)?.contiguous()?;

        // Write into pre-allocated buffer — zero allocation, zero copy
        cache.write_and_advance(layer_idx, &k, &v)?;

        // Read back the valid portion of the cache for attention.
        // Cast back to the query dtype for the matmul.
        let q_dtype = q.dtype();
        let k = cache.k_valid(layer_idx)?.to_dtype(q_dtype)?.to_device(device)?;
        let v = cache.v_valid(layer_idx)?.to_dtype(q_dtype)?.to_device(device)?;

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_output = if self.use_flash_attn && x.device().is_cuda() {
            let q_flash = q.transpose(1, 2)?.contiguous()?;
            let k_flash = k.transpose(1, 2)?.contiguous()?;
            let v_flash = v.transpose(1, 2)?.contiguous()?;
            let out = flash_attn(&q_flash, &k_flash, &v_flash, scale as f32, true)?;
            out.transpose(1, 2)?
        } else {
            let attn_weights = q.matmul(&k.transpose(2, 3)?)?.affine(scale, 0.0)?;
            let attn_weights = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
            attn_weights.matmul(&v)?
        };

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b, n_kv_head, seq_len, head_dim) = x.dims4()?;
            let x = x.unsqueeze(2)?.expand((b, n_kv_head, n_rep, seq_len, head_dim))?;
            x.reshape((b, n_kv_head * n_rep, seq_len, head_dim))
        }
    }

    fn set_training(&self, training: bool) {
        self.q_proj.set_training(training);
        self.k_proj.set_training(training);
        self.v_proj.set_training(training);
        self.o_proj.set_training(training);
    }
}

#[derive(Debug)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, lora: LoraSettings, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let is_sliding = if let Some(types) = &cfg.layer_types {
            types.get(layer_idx).map(|s| s == "sliding_attention").unwrap_or(false)
        } else {
            false
        };

        let self_attn = Attention::new(cfg, lora, is_sliding, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, lora, vb.pp("mlp"))?;
        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let pre_feedforward_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("pre_feedforward_layernorm"))?;
        let post_attention_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        let post_feedforward_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_feedforward_layernorm"))?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            pre_feedforward_layernorm,
            post_attention_layernorm,
            post_feedforward_layernorm,
        })
    }

    fn new_with_base(
        cfg: &Config,
        lora: LoraSettings,
        layer_idx: usize,
        vb: VarBuilder,
        st: &MmapedSafetensors,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let is_sliding = if let Some(types) = &cfg.layer_types {
            types.get(layer_idx).map(|s| s == "sliding_attention").unwrap_or(false)
        } else {
            false
        };

        let prefix = format!("model.layers.{}", layer_idx);
        let attn_prefix = format!("{prefix}.self_attn");
        let mlp_prefix = format!("{prefix}.mlp");

        let q_weight = load_weight(st, &format!("{attn_prefix}.q_proj.weight"), device, dtype)?;
        let k_weight = load_weight(st, &format!("{attn_prefix}.k_proj.weight"), device, dtype)?;
        let v_weight = load_weight(st, &format!("{attn_prefix}.v_proj.weight"), device, dtype)?;
        let o_weight = load_weight(st, &format!("{attn_prefix}.o_proj.weight"), device, dtype)?;
        let q_norm_weight = load_weight(st, &format!("{attn_prefix}.q_norm.weight"), device, dtype)?;
        let k_norm_weight = load_weight(st, &format!("{attn_prefix}.k_norm.weight"), device, dtype)?;

        let gate_weight = load_weight(st, &format!("{mlp_prefix}.gate_proj.weight"), device, dtype)?;
        let up_weight = load_weight(st, &format!("{mlp_prefix}.up_proj.weight"), device, dtype)?;
        let down_weight = load_weight(st, &format!("{mlp_prefix}.down_proj.weight"), device, dtype)?;

        let input_ln_weight = load_weight(st, &format!("{prefix}.input_layernorm.weight"), device, dtype)?;
        let pre_ff_ln_weight = load_weight(st, &format!("{prefix}.pre_feedforward_layernorm.weight"), device, dtype)?;
        let post_attn_ln_weight = load_weight(st, &format!("{prefix}.post_attention_layernorm.weight"), device, dtype)?;
        let post_ff_ln_weight = load_weight(
            st,
            &format!("{prefix}.post_feedforward_layernorm.weight"),
            device,
            dtype,
        )?;

        let self_attn = Attention::new_with_base(
            cfg,
            lora,
            is_sliding,
            vb.pp("self_attn"),
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            Some(q_norm_weight),
            Some(k_norm_weight),
        )?;
        let mlp = Mlp::new_with_base(cfg, lora, vb.pp("mlp"), gate_weight, up_weight, down_weight)?;
        let input_layernorm = RmsNorm::from_weight(input_ln_weight, cfg.rms_norm_eps)?;
        let pre_feedforward_layernorm = RmsNorm::from_weight(pre_ff_ln_weight, cfg.rms_norm_eps)?;
        let post_attention_layernorm = RmsNorm::from_weight(post_attn_ln_weight, cfg.rms_norm_eps)?;
        let post_feedforward_layernorm = RmsNorm::from_weight(post_ff_ln_weight, cfg.rms_norm_eps)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            pre_feedforward_layernorm,
            post_attention_layernorm,
            post_feedforward_layernorm,
        })
    }

    fn forward_with_cache(
        &self,
        x: &Tensor,
        cache: &mut Option<KvCacheEntry>,
        past_len: usize,
        kv_quant: KvCacheQuant,
        kv_max_len: Option<usize>,
        kv_store_on_cpu: bool,
    ) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self
            .self_attn
            .forward_with_cache(&x, cache, past_len, kv_quant, kv_max_len, kv_store_on_cpu)?;
        let x = (x + residual)?;
        let x = self.post_attention_layernorm.forward(&x)?;

        let residual = x.clone();
        let x = self.pre_feedforward_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = (x + residual)?;
        self.post_feedforward_layernorm.forward(&x)
    }

    fn forward_with_prealloc_cache(
        &self,
        x: &Tensor,
        cache: &mut PreAllocKvCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self
            .self_attn
            .forward_with_prealloc_cache(&x, cache, layer_idx)?;
        let x = (x + residual)?;
        let x = self.post_attention_layernorm.forward(&x)?;

        let residual = x.clone();
        let x = self.pre_feedforward_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = (x + residual)?;
        self.post_feedforward_layernorm.forward(&x)
    }

    fn set_training(&self, training: bool) {
        self.self_attn.set_training(training);
        self.mlp.set_training(training);
    }
}

impl Module for DecoderLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x)?;
        let x = (x + residual)?;
        let x = self.post_attention_layernorm.forward(&x)?;

        let residual = x.clone();
        let x = self.pre_feedforward_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = (x + residual)?;
        self.post_feedforward_layernorm.forward(&x)
    }
}

#[derive(Debug)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
}

#[derive(Debug, Clone, Copy)]
pub enum KvCacheQuant {
    None,
    Int8,
}

impl KvCacheQuant {
    pub fn from_str(value: Option<&str>) -> Self {
        match value.unwrap_or("none").to_ascii_lowercase().as_str() {
            "int8" | "i8" => KvCacheQuant::Int8,
            _ => KvCacheQuant::None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum KvCacheEntry {
    Full {
        k: Tensor,
        v: Tensor,
    },
    Int8 {
        k: Tensor,
        v: Tensor,
        k_scale: f32,
        v_scale: f32,
        dtype: DType,
    },
}

#[derive(Debug)]
pub struct KvCache {
    pub past_len: usize,
    pub layers: Vec<Option<KvCacheEntry>>,
    pub quant: KvCacheQuant,
    pub max_len: Option<usize>,
    pub store_on_cpu: bool,
}

impl KvCache {
    pub fn new(num_layers: usize, quant: KvCacheQuant, max_len: Option<usize>, store_on_cpu: bool) -> Self {
        Self {
            past_len: 0,
            layers: vec![None; num_layers],
            quant,
            max_len,
            store_on_cpu,
        }
    }
}

// ---------------------------------------------------------------------------
// Pre-allocated ring-buffer KV cache
// ---------------------------------------------------------------------------

/// Per-layer pre-allocated K and V tensors with a write cursor.
///
/// Instead of growing via `Tensor::cat` each decode step, we pre-allocate the
/// full `[batch, num_kv_heads, max_seq_len, head_dim]` buffer once and write
/// into it via `Tensor::slice_set` (in-place, zero-allocation).
#[derive(Debug)]
pub struct PreAllocKvLayer {
    /// Shape: `[batch, num_kv_heads, max_seq_len, head_dim]`
    pub k: Tensor,
    /// Shape: `[batch, num_kv_heads, max_seq_len, head_dim]`
    pub v: Tensor,
}

/// A pre-allocated, fixed-size KV cache that eliminates per-token allocation.
///
/// The cache pre-allocates K and V tensors for every layer at construction
/// time.  During generation each new K/V slice is written directly into the
/// pre-allocated buffer via `Tensor::slice_set` on the sequence dimension
/// (dim 2).  A `write_cursor` tracks the next write position and wraps
/// around when `max_seq_len` is reached (ring-buffer semantics).
///
/// # Benefits over `KvCache`
///
/// * **Zero allocation per token** — no `Tensor::cat` means no new buffer
///   allocation on every decode step.
/// * **Zero copy of existing cache** — `cat` copies the entire accumulated
///   cache; `slice_set` only writes the new slice.
/// * **No VRAM fragmentation** — a single contiguous allocation per layer.
/// * **Deterministic memory usage** — VRAM is bounded at init time.
///
/// # Limitations
///
/// * Does **not** support int8 quantised storage (the existing `KvCacheQuant::Int8`
///   path requires per-step re-quantisation with a changing scale factor that is
///   incompatible with a fixed pre-allocated buffer).  Falls back to the original
///   `KvCache` if int8 is requested.
/// * Requires contiguous tensors (enforced by `slice_set`).
/// * The ring-buffer wrap-around means that once `max_seq_len` is exceeded,
///   the oldest positions are silently overwritten.  For FunctionGemma routing
///   (typically < 100 tokens), this never triggers.
#[derive(Debug)]
pub struct PreAllocKvCache {
    /// Per-layer pre-allocated buffers.
    pub layers: Vec<PreAllocKvLayer>,
    /// Next write position in the sequence dimension (dim 2).
    pub write_cursor: usize,
    /// Number of valid (written) positions.  May be less than `write_cursor`
    /// only on the first pass before wrap-around.
    pub valid_len: usize,
    /// Maximum sequence length (the size of the pre-allocated seq dimension).
    pub max_seq_len: usize,
}

impl PreAllocKvCache {
    /// Pre-allocate K and V buffers for all layers.
    ///
    /// * `num_layers` — number of transformer layers
    /// * `batch` — batch size (typically 1)
    /// * `num_kv_heads` — number of key/value heads (from `Config`)
    /// * `head_dim` — per-head dimension (from `Config`)
    /// * `max_seq_len` — maximum sequence length to pre-allocate
    /// * `dtype` — element type (e.g. `DType::BF16` or `DType::F32`)
    /// * `device` — target device (CPU or CUDA)
    pub fn new(
        num_layers: usize,
        batch: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let k = Tensor::zeros((batch, num_kv_heads, max_seq_len, head_dim), dtype, device)?;
            let v = Tensor::zeros((batch, num_kv_heads, max_seq_len, head_dim), dtype, device)?;
            layers.push(PreAllocKvLayer { k, v });
        }
        Ok(Self {
            layers,
            write_cursor: 0,
            valid_len: 0,
            max_seq_len,
        })
    }

    /// Reset the cache for a new request without deallocating.
    ///
    /// The underlying tensors keep their device allocation; only the cursors
    /// are zeroed.  This is O(1).
    pub fn reset(&mut self) {
        self.write_cursor = 0;
        self.valid_len = 0;
    }

    /// Write a new K/V pair into layer `layer_idx` at the current cursor and
    /// advance.  Returns the valid sequence length after the write (for
    /// constructing the attention mask / RoPE offset).
    ///
    /// `new_k` and `new_v` have shape `[batch, num_kv_heads, 1, head_dim]`
    /// (a single new token).
    pub fn write_and_advance(
        &mut self,
        layer_idx: usize,
        new_k: &Tensor,
        new_v: &Tensor,
    ) -> Result<usize> {
        let pos = self.write_cursor % self.max_seq_len;
        let layer = &self.layers[layer_idx];

        // slice_set writes into dim=2 (sequence dimension) at offset `pos`.
        // Both self.k and new_k must be contiguous — guaranteed by our
        // construction (zeros) and the caller (.contiguous() after projection).
        layer.k.slice_set(new_k, 2, pos)?;
        layer.v.slice_set(new_v, 2, pos)?;

        // Only advance cursor after last layer to keep layers in sync.
        // The caller (Model::forward_with_prealloc_cache) advances once
        // after iterating all layers by calling `advance_cursor()`.
        Ok(self.valid_len.min(self.max_seq_len))
    }

    /// Advance the write cursor by one position.  Call once per decode step
    /// after all layers have been written.
    pub fn advance_cursor(&mut self) {
        self.write_cursor += 1;
        self.valid_len = self.write_cursor.min(self.max_seq_len);
    }

    /// Return the K tensor for a layer, narrowed to the valid region.
    ///
    /// If the buffer has not wrapped (`valid_len < max_seq_len`), returns
    /// `k[:, :, 0..valid_len, :]`.  After wrap-around, returns the full
    /// buffer (all positions are valid).
    pub fn k_valid(&self, layer_idx: usize) -> Result<Tensor> {
        let layer = &self.layers[layer_idx];
        if self.valid_len < self.max_seq_len {
            layer.k.narrow(2, 0, self.valid_len)
        } else {
            Ok(layer.k.clone())
        }
    }

    /// Return the V tensor for a layer, narrowed to the valid region.
    pub fn v_valid(&self, layer_idx: usize) -> Result<Tensor> {
        let layer = &self.layers[layer_idx];
        if self.valid_len < self.max_seq_len {
            layer.v.narrow(2, 0, self.valid_len)
        } else {
            Ok(layer.v.clone())
        }
    }

    /// The RoPE position offset for the next token (used for positional
    /// embeddings).  Equals the total number of tokens written so far,
    /// even after wrap-around — RoPE uses absolute position, not ring
    /// position.
    pub fn rope_offset(&self) -> usize {
        self.write_cursor
    }
}

fn quantize_tensor_int8(tensor: &Tensor) -> Result<(Tensor, f32)> {
    let f32 = tensor.to_dtype(DType::F32)?;
    let max_sq = f32.sqr()?.max_all()?.to_scalar::<f32>()?;
    let max = max_sq.sqrt().max(1e-6);
    let scale = max / 127.0;
    let scaled = f32.affine(1.0 / scale as f64, 0.0)?.round()?.clamp(-127f64, 127f64)?;
    let shifted = scaled.affine(1.0, 128.0)?;
    let q = shifted.to_dtype(DType::U8)?;
    Ok((q, scale))
}

fn dequantize_tensor_int8(tensor: &Tensor, scale: f32, dtype: DType) -> Result<Tensor> {
    let f32 = tensor.to_dtype(DType::F32)?;
    let shifted = f32.affine(1.0, -128.0)?;
    let deq = shifted.affine(scale as f64, 0.0)?;
    deq.to_dtype(dtype)
}

impl Model {
    pub fn new(cfg: &Config, lora: LoraSettings, vb: VarBuilder, tie_embeddings: bool) -> Result<Self> {
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, lora, i, vb.pp(format!("model.layers.{}", i)))?);
        }

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;

        let lm_head = if tie_embeddings {
            candle_nn::Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn new_qlora(
        cfg: &Config,
        lora: LoraSettings,
        vb: VarBuilder,
        tie_embeddings: bool,
        st: &MmapedSafetensors,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let embed_weight = load_weight(st, "model.embed_tokens.weight", device, dtype)?;
        let embed_tokens = candle_nn::Embedding::new(embed_weight, cfg.hidden_size);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new_with_base(
                cfg,
                lora,
                i,
                vb.pp(format!("model.layers.{}", i)),
                st,
                device,
                dtype,
            )?);
        }

        let norm_weight = load_weight(st, "model.norm.weight", device, dtype)?;
        let norm = RmsNorm::from_weight(norm_weight, cfg.rms_norm_eps)?;

        let lm_head = if tie_embeddings {
            candle_nn::Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            let lm_weight = load_weight(st, "lm_head.weight", device, dtype)?;
            candle_nn::Linear::new(lm_weight, None)
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut x = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x)
    }

    pub fn forward_with_cache(&self, input_ids: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        if seq_len != 1 {
            return self.forward(input_ids);
        }

        let mut x = self.embed_tokens.forward(input_ids)?;
        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward_with_cache(
                &x,
                &mut cache.layers[idx],
                cache.past_len,
                cache.quant,
                cache.max_len,
                cache.store_on_cpu,
            )?;
        }
        let x = self.norm.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        cache.past_len += 1;
        Ok(logits)
    }

    pub fn generate_with_cache(
        &self,
        input_ids: &Tensor,
        max_len: usize,
        device: &Device,
        kv_quant: KvCacheQuant,
        kv_max_len: Option<usize>,
        kv_store_on_cpu: bool,
    ) -> Result<Vec<u32>> {
        let mut generated = Vec::new();
        let mut cache = KvCache::new(self.layers.len(), kv_quant, kv_max_len, kv_store_on_cpu);

        let (_b, seq_len) = input_ids.dims2()?;
        let mut last_logits = None;
        for idx in 0..seq_len {
            let token = input_ids.narrow(1, idx, 1)?;
            let logits = self.forward_with_cache(&token, &mut cache)?;
            last_logits = Some(logits);
        }

        for _ in 0..max_len {
            let logits = match &last_logits {
                Some(t) => t.clone(),
                None => self.forward(input_ids)?,
            };
            let last_logits_tensor = logits.squeeze(0)?.squeeze(0)?;
            // Use to_vec1 + index instead of to_scalar to avoid a GPU→CPU scalar
            // transfer that stalls the pipeline; to_vec1 batches the copy.
            let next_id = last_logits_tensor.argmax(0)?.to_vec1::<u32>()?[0];
            generated.push(next_id);

            if next_id == 1 || next_id == 107 || next_id == 106 {
                break;
            }

            let next_tensor = Tensor::from_slice(&[next_id], (1, 1), device)?;
            let logits = self.forward_with_cache(&next_tensor, &mut cache)?;
            last_logits = Some(logits);
        }
        Ok(generated)
    }

    /// Forward pass using the pre-allocated ring-buffer KV cache.
    ///
    /// Like `forward_with_cache`, this only processes single-token inputs
    /// (seq_len == 1).  For the prompt prefill (seq_len > 1), falls back to
    /// `forward()`.
    pub fn forward_with_prealloc_cache(
        &self,
        input_ids: &Tensor,
        cache: &mut PreAllocKvCache,
    ) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        if seq_len != 1 {
            return self.forward(input_ids);
        }

        let mut x = self.embed_tokens.forward(input_ids)?;
        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward_with_prealloc_cache(&x, cache, idx)?;
        }
        // Advance cursor once after all layers have written their K/V
        cache.advance_cursor();

        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x)
    }

    /// Generate tokens using the pre-allocated ring-buffer KV cache.
    ///
    /// This is the zero-allocation alternative to `generate_with_cache`.
    /// The KV cache is pre-allocated once and reused across all decode steps.
    ///
    /// # Arguments
    ///
    /// * `input_ids` — prompt token IDs, shape `[1, seq_len]`
    /// * `max_len` — maximum number of tokens to generate
    /// * `device` — target device
    /// * `cfg` — model config (needed for `num_kv_heads`, `head_dim`)
    /// * `kv_max_len` — pre-allocation size for the ring buffer
    pub fn generate_with_prealloc_cache(
        &self,
        input_ids: &Tensor,
        max_len: usize,
        device: &Device,
        cfg: &Config,
        kv_max_len: usize,
    ) -> Result<Vec<u32>> {
        // Use the embedding weight dtype as the cache dtype — this matches
        // what the K/V projections will produce.  Falls back to F32 on CPU.
        let dtype = self.embed_tokens.embeddings().dtype();
        let mut cache = PreAllocKvCache::new(
            self.layers.len(),
            1, // batch = 1
            cfg.num_key_value_heads,
            cfg.head_dim,
            kv_max_len,
            dtype,
            device,
        )?;

        let mut generated = Vec::new();
        let (_b, seq_len) = input_ids.dims2()?;

        // Prefill: feed each prompt token through the model one at a time
        // to populate the KV cache.
        let mut last_logits = None;
        for idx in 0..seq_len {
            let token = input_ids.narrow(1, idx, 1)?;
            let logits = self.forward_with_prealloc_cache(&token, &mut cache)?;
            last_logits = Some(logits);
        }

        // Decode: generate tokens autoregressively.
        for _ in 0..max_len {
            let logits = match &last_logits {
                Some(t) => t.clone(),
                None => self.forward(input_ids)?,
            };
            let last_logits_tensor = logits.squeeze(0)?.squeeze(0)?;
            let next_id = last_logits_tensor.argmax(0)?.to_vec1::<u32>()?[0];
            generated.push(next_id);

            if next_id == 1 || next_id == 107 || next_id == 106 {
                break;
            }

            let next_tensor = Tensor::from_slice(&[next_id], (1, 1), device)?;
            let logits = self.forward_with_prealloc_cache(&next_tensor, &mut cache)?;
            last_logits = Some(logits);
        }
        Ok(generated)
    }

    pub fn generate(&self, input_ids: &Tensor, max_len: usize, device: &Device) -> Result<Vec<u32>> {
        let mut generated = Vec::new();
        let mut current_ids = input_ids.clone();

        for _ in 0..max_len {
            let logits = self.forward(&current_ids)?;
            let (_b, s, _v) = logits.dims3()?;
            let last_logits = logits.narrow(1, s - 1, 1)?.squeeze(0)?.squeeze(0)?;

            // Greedy sampling — use to_vec1 + index instead of to_scalar so a
            // single batched copy services the GPU→CPU transfer.
            let next_id = last_logits.argmax(0)?.to_vec1::<u32>()?[0];
            generated.push(next_id);

            // Check for EOS (special ID for Gemma) - typically 1 or 107
            if next_id == 1 || next_id == 107 || next_id == 106 {
                break;
            }

            let next_tensor = Tensor::from_slice(&[next_id], (1, 1), device)?;
            current_ids = Tensor::cat(&[&current_ids, &next_tensor], 1)?;
        }
        Ok(generated)
    }

    /// Enable or disable training mode for all LoRA layers in the model.
    ///
    /// When training mode is active, LoRA dropout is applied during the forward
    /// pass.  Call `set_training(false)` before evaluation or inference to
    /// disable dropout.
    pub fn set_training(&self, training: bool) {
        for layer in &self.layers {
            layer.set_training(training);
        }
    }

    pub fn merge_adapters(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.self_attn.q_proj.merge()?;
            layer.self_attn.k_proj.merge()?;
            layer.self_attn.v_proj.merge()?;
            layer.self_attn.o_proj.merge()?;

            layer.mlp.gate_proj.merge()?;
            layer.mlp.up_proj.merge()?;
            layer.mlp.down_proj.merge()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Result, Tensor};
    use candle_nn::VarMap;

    #[test]
    fn test_rmsnorm() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let norm = RmsNorm::new(64, 1e-5, vb)?;
        let x = Tensor::ones((1, 10, 64), DType::F32, &device)?;
        let y = norm.forward(&x)?;
        assert_eq!(y.dims(), &[1, 10, 64]);
        Ok(())
    }

    #[test]
    fn test_prealloc_kv_cache_new() -> Result<()> {
        let device = Device::Cpu;
        let cache = PreAllocKvCache::new(4, 1, 2, 64, 128, DType::F32, &device)?;
        assert_eq!(cache.layers.len(), 4);
        assert_eq!(cache.write_cursor, 0);
        assert_eq!(cache.valid_len, 0);
        assert_eq!(cache.max_seq_len, 128);
        // Verify pre-allocated shapes
        for layer in &cache.layers {
            assert_eq!(layer.k.dims(), &[1, 2, 128, 64]);
            assert_eq!(layer.v.dims(), &[1, 2, 128, 64]);
        }
        Ok(())
    }

    #[test]
    fn test_prealloc_kv_cache_write_and_read() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PreAllocKvCache::new(2, 1, 2, 4, 8, DType::F32, &device)?;

        // Write 3 tokens into layer 0
        for i in 0..3 {
            let val = (i + 1) as f32;
            let k = Tensor::full(val, (1, 2, 1, 4), &device)?;
            let v = Tensor::full(val * 10.0, (1, 2, 1, 4), &device)?;
            cache.write_and_advance(0, &k, &v)?;
            cache.write_and_advance(1, &k, &v)?;
            cache.advance_cursor();
        }

        assert_eq!(cache.write_cursor, 3);
        assert_eq!(cache.valid_len, 3);

        // Read back valid K for layer 0
        let k_valid = cache.k_valid(0)?;
        assert_eq!(k_valid.dims(), &[1, 2, 3, 4]);

        // Verify the first position has value 1.0
        let k_first = k_valid.narrow(2, 0, 1)?.squeeze(2)?;
        let vals: Vec<f32> = k_first.flatten_all()?.to_vec1()?;
        assert!((vals[0] - 1.0).abs() < 1e-6);

        // Verify the third position has value 3.0
        let k_third = k_valid.narrow(2, 2, 1)?.squeeze(2)?;
        let vals: Vec<f32> = k_third.flatten_all()?.to_vec1()?;
        assert!((vals[0] - 3.0).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_prealloc_kv_cache_reset() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PreAllocKvCache::new(2, 1, 2, 4, 8, DType::F32, &device)?;

        // Write a token
        let k = Tensor::ones((1, 2, 1, 4), DType::F32, &device)?;
        let v = Tensor::ones((1, 2, 1, 4), DType::F32, &device)?;
        cache.write_and_advance(0, &k, &v)?;
        cache.write_and_advance(1, &k, &v)?;
        cache.advance_cursor();
        assert_eq!(cache.valid_len, 1);

        // Reset
        cache.reset();
        assert_eq!(cache.write_cursor, 0);
        assert_eq!(cache.valid_len, 0);

        // Buffer still exists (no deallocation)
        assert_eq!(cache.layers[0].k.dims(), &[1, 2, 8, 4]);
        Ok(())
    }

    #[test]
    fn test_prealloc_kv_cache_wrap_around() -> Result<()> {
        let device = Device::Cpu;
        // Small max_seq_len of 4 to test wrap-around
        let mut cache = PreAllocKvCache::new(1, 1, 1, 2, 4, DType::F32, &device)?;

        // Write 6 tokens (exceeding max_seq_len of 4)
        for i in 0..6 {
            let val = (i + 1) as f32;
            let k = Tensor::full(val, (1, 1, 1, 2), &device)?;
            let v = Tensor::full(val, (1, 1, 1, 2), &device)?;
            cache.write_and_advance(0, &k, &v)?;
            cache.advance_cursor();
        }

        // After 6 writes into a buffer of 4, valid_len should be 4
        assert_eq!(cache.valid_len, 4);
        assert_eq!(cache.write_cursor, 6);

        // The full buffer should be returned (all 4 positions valid)
        let k_valid = cache.k_valid(0)?;
        assert_eq!(k_valid.dims(), &[1, 1, 4, 2]);

        // Position 0 should have token 5 (wrap: 4 % 4 = 0, value 5.0)
        // Position 1 should have token 6 (wrap: 5 % 4 = 1, value 6.0)
        // Position 2 should have token 3 (not overwritten, value 3.0)
        // Position 3 should have token 4 (not overwritten, value 4.0)
        let vals: Vec<f32> = k_valid.flatten_all()?.to_vec1()?;
        assert!((vals[0] - 5.0).abs() < 1e-6, "pos 0: expected 5.0, got {}", vals[0]);
        assert!((vals[2] - 6.0).abs() < 1e-6, "pos 1: expected 6.0, got {}", vals[2]);
        assert!((vals[4] - 3.0).abs() < 1e-6, "pos 2: expected 3.0, got {}", vals[4]);
        assert!((vals[6] - 4.0).abs() < 1e-6, "pos 3: expected 4.0, got {}", vals[6]);

        Ok(())
    }

    #[test]
    fn test_prealloc_kv_cache_rope_offset() -> Result<()> {
        let device = Device::Cpu;
        let mut cache = PreAllocKvCache::new(1, 1, 1, 2, 4, DType::F32, &device)?;

        assert_eq!(cache.rope_offset(), 0);

        let k = Tensor::ones((1, 1, 1, 2), DType::F32, &device)?;
        let v = Tensor::ones((1, 1, 1, 2), DType::F32, &device)?;
        cache.write_and_advance(0, &k, &v)?;
        cache.advance_cursor();
        assert_eq!(cache.rope_offset(), 1);

        // After wrap-around, rope_offset continues to increase
        for _ in 0..5 {
            cache.write_and_advance(0, &k, &v)?;
            cache.advance_cursor();
        }
        assert_eq!(cache.rope_offset(), 6);

        Ok(())
    }
}
