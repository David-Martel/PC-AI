//! The opt-in, published Gemma3 checkpoint semantics.

use crate::config::RuntimeConfig;
use anyhow::{bail, ensure, Context};
use candle_core::{safetensors::MmapedSafetensors, DType, Device, Tensor};
use candle_nn::VarBuilder;
use rust_functiongemma_core::{
    detect_tie_embeddings,
    hf_gemma3::{Config, Model},
};
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;

pub(super) const PROFILE: &str = "hf_gemma3_4_57_3";
const MAX_CONTEXT: usize = 4096;

pub(super) struct Backend {
    model: Model,
    eos: Vec<u32>,
    context: usize,
}

#[derive(Deserialize)]
struct GenerationConfig {
    eos_token_id: Eos,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Eos {
    Single(u32),
    Multiple(Vec<u32>),
}

pub(super) fn validate_options(runtime: &RuntimeConfig) -> anyhow::Result<()> {
    ensure!(
        runtime.router_lora_path.trim().is_empty(),
        "{PROFILE} does not support LoRA"
    );
    ensure!(
        runtime.router_kv_cache
            && runtime
                .router_kv_cache_quant
                .as_deref()
                .is_none_or(|value| value.eq_ignore_ascii_case("none"))
            && matches!(
                runtime.router_kv_cache_store.to_ascii_lowercase().as_str(),
                "gpu" | "device"
            ),
        "{PROFILE} requires an unquantized, device-local KV cache"
    );
    ensure!(
        !runtime.router_flash_attn,
        "{PROFILE} does not support the custom FlashAttention setting"
    );
    ensure!(
        !runtime.router_candle_qmatmul,
        "{PROFILE} does not support custom qmatmul"
    );
    Ok(())
}

fn checkpoint_config(mut raw: Value) -> anyhow::Result<Config> {
    ensure!(
        raw["model_type"].as_str() == Some("gemma3_text"),
        "{PROFILE} requires a Gemma3 text checkpoint"
    );
    if let Some(pattern) = raw.get("_sliding_window_pattern").cloned() {
        ensure!(
            raw.get("sliding_window_pattern")
                .is_none_or(|existing| *existing == pattern),
            "conflicting Gemma3 sliding-window patterns"
        );
        raw["sliding_window_pattern"] = pattern;
    }
    let cfg: Config = serde_json::from_value(raw.clone()).context("invalid Gemma3 checkpoint configuration")?;
    ensure!(
        cfg.num_hidden_layers > 0
            && cfg.num_key_value_heads > 0
            && cfg.num_attention_heads >= cfg.num_key_value_heads
            && cfg.num_attention_heads % cfg.num_key_value_heads == 0
            && cfg.head_dim > 0
            && cfg.head_dim % 2 == 0
            && cfg.sliding_window > 0
            && cfg.sliding_window_pattern > 0
            && cfg.query_pre_attn_scalar > 0
            && cfg.vocab_size > 0,
        "invalid Gemma3 dimensions or attention schedule"
    );
    if let Some(types) = raw.get("layer_types") {
        let types = types.as_array().context("Gemma3 layer_types must be an array")?;
        ensure!(
            types.len() == cfg.num_hidden_layers
                && types.iter().enumerate().all(|(idx, value)| {
                    let expected = if (idx + 1) % cfg.sliding_window_pattern == 0 {
                        "full_attention"
                    } else {
                        "sliding_attention"
                    };
                    value.as_str() == Some(expected)
                }),
            "Gemma3 layer_types do not match the checkpoint's periodic attention schedule"
        );
    }
    Ok(cfg)
}

fn context_limit(cfg: &Config, runtime: &RuntimeConfig) -> anyhow::Result<usize> {
    let limit = cfg
        .max_position_embeddings
        .min(MAX_CONTEXT)
        .min(runtime.router_max_seq_len.unwrap_or(MAX_CONTEXT))
        .min(runtime.router_kv_cache_max_len.unwrap_or(MAX_CONTEXT));
    ensure!(limit > 1, "{PROFILE} requires a context limit of at least two tokens");
    Ok(limit)
}

impl Backend {
    pub(super) fn load(
        path: &Path,
        raw_config: Value,
        weights: MmapedSafetensors,
        runtime: &RuntimeConfig,
        dtype: DType,
        device: &Device,
    ) -> anyhow::Result<Self> {
        ensure!(detect_tie_embeddings(&weights), "{PROFILE} requires tied embeddings");
        let mut cfg = checkpoint_config(raw_config)?;
        let context = context_limit(&cfg, runtime)?;
        cfg.max_position_embeddings = context;
        let generation: GenerationConfig = serde_json::from_slice(
            &std::fs::read(path.join("generation_config.json")).context("read Gemma3 generation configuration")?,
        )
        .context("invalid Gemma3 generation configuration")?;
        let eos = match generation.eos_token_id {
            Eos::Single(id) => vec![id],
            Eos::Multiple(ids) => ids,
        };
        ensure!(
            !eos.is_empty() && eos.iter().all(|&id| (id as usize) < cfg.vocab_size),
            "Gemma3 EOS IDs must be nonempty and within the vocabulary"
        );
        let builder = VarBuilder::from_backend(Box::new(weights), dtype, device.clone());
        Ok(Self {
            model: Model::new(false, &cfg, builder)?,
            eos,
            context,
        })
    }

    pub(super) fn generate(&mut self, input: &Tensor, max_new: usize) -> anyhow::Result<Vec<u32>> {
        self.model.clear_kv_cache();
        let result = self.generate_request(input, max_new);
        self.model.clear_kv_cache();
        result
    }

    fn generate_request(&mut self, input: &Tensor, max_new: usize) -> anyhow::Result<Vec<u32>> {
        let (batch, prompt_len) = input.dims2()?;
        if batch != 1
            || prompt_len == 0
            || max_new == 0
            || prompt_len.checked_add(max_new).is_none_or(|len| len > self.context)
        {
            bail!(
                "{PROFILE} requires a nonempty batch-one prompt and decode within {} tokens",
                self.context
            );
        }
        let mut logits = self.model.forward(input, 0)?;
        let mut tokens = Vec::with_capacity(max_new);
        for step in 0..max_new {
            let next = logits.flatten_all()?.argmax(0)?.to_scalar::<u32>()?;
            tokens.push(next);
            if self.eos.contains(&next) || step + 1 == max_new {
                break;
            }
            let input = Tensor::from_slice(&[next], (1, 1), input.device())?;
            logits = self.model.forward(&input, prompt_len + step)?;
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Protects: the opt-in backend never silently substitutes unsupported inference options.
    // Detects: accepting LoRA, custom quantization, offloaded KV or disabled cache settings.
    // Needs: only the runtime config defaults.
    // Breadcrumb: hf_gemma3::validate_options.
    #[test]
    fn rejects_options_that_the_publisher_backend_cannot_honor() {
        assert!(validate_options(&RuntimeConfig::default()).is_ok());
        for options in [
            RuntimeConfig {
                router_lora_path: "adapter.safetensors".into(),
                ..RuntimeConfig::default()
            },
            RuntimeConfig {
                router_kv_cache_quant: Some("int8".into()),
                ..RuntimeConfig::default()
            },
            RuntimeConfig {
                router_kv_cache_store: "cpu".into(),
                ..RuntimeConfig::default()
            },
            RuntimeConfig {
                router_kv_cache: false,
                ..RuntimeConfig::default()
            },
            RuntimeConfig {
                router_flash_attn: true,
                ..RuntimeConfig::default()
            },
            RuntimeConfig {
                router_candle_qmatmul: true,
                ..RuntimeConfig::default()
            },
        ] {
            assert!(validate_options(&options).is_err());
        }
    }

    // Protects: the checkpoint's periodic layer schedule and distinct context caps are honored.
    // Detects: alias drift, impossible layer patterns, and context values that could overrun the model cache.
    // Needs: a small in-memory checkpoint configuration.
    // Breadcrumb: hf_gemma3::checkpoint_config and hf_gemma3::context_limit.
    #[test]
    fn validates_publisher_attention_schedule_and_context_caps() -> anyhow::Result<()> {
        let raw = json!({
            "model_type": "gemma3_text",
            "attention_bias": false,
            "head_dim": 2,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 0.000001,
            "rope_theta": 1_000_000.0,
            "rope_local_base_freq": 10_000.0,
            "vocab_size": 8,
            "query_pre_attn_scalar": 2,
            "sliding_window": 4,
            "_sliding_window_pattern": 2,
            "layer_types": ["sliding_attention", "full_attention"],
            "max_position_embeddings": 32_768
        });
        let cfg = checkpoint_config(raw.clone())?;
        assert_eq!(cfg.sliding_window_pattern, 2);
        assert_eq!(context_limit(&cfg, &RuntimeConfig::default())?, MAX_CONTEXT);
        let limited = RuntimeConfig {
            router_max_seq_len: Some(16),
            router_kv_cache_max_len: Some(12),
            ..RuntimeConfig::default()
        };
        assert_eq!(context_limit(&cfg, &limited)?, 12);
        let mut invalid = raw.clone();
        invalid["layer_types"] = json!(["full_attention", "sliding_attention"]);
        assert!(checkpoint_config(invalid).is_err());
        let mut invalid = raw;
        invalid["sliding_window_pattern"] = json!(3);
        assert!(checkpoint_config(invalid).is_err());
        Ok(())
    }
}
