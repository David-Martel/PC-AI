//! Janus-Pro model configuration.
//!
//! [`JanusConfig`] describes the LLM backbone, the vision tokeniser, and the
//! image codec dimensions used by the Janus-Pro architecture.  It can be
//! deserialised from a `config.json` file and converted into the
//! `candle_transformers` Llama [`Config`](candle_transformers::models::llama::Config)
//! needed to build the underlying language model.
//!
//! # Example
//!
//! ```rust
//! use pcai_media_model::config::JanusConfig;
//!
//! let cfg = JanusConfig::janus_pro_7b();
//! assert_eq!(cfg.num_image_tokens(), 576);
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// JanusConfig
// ---------------------------------------------------------------------------

/// Configuration for the Janus-Pro model architecture.
///
/// The 7B defaults match the publicly released `config.json` for
/// `deepseek-ai/Janus-Pro-7B`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JanusConfig {
    /// Dimensionality of each hidden layer.
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    /// Number of transformer layers.
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,

    /// Number of query attention heads.
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,

    /// Number of key-value heads (grouped-query attention).
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,

    /// Vocabulary size of the language model.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    /// Dimensionality of the feed-forward intermediate layer.
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,

    /// Number of discrete image tokens in the VQ codebook (image vocabulary).
    #[serde(default = "default_image_token_num_tokens")]
    pub image_token_num_tokens: usize,

    /// Resolution of the image in pixels (square assumed).
    #[serde(default = "default_image_size")]
    pub image_size: usize,

    /// Patch size used by the vision encoder.
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
}

// ---------------------------------------------------------------------------
// serde default functions
// ---------------------------------------------------------------------------

fn default_hidden_size() -> usize {
    4096
}
fn default_num_hidden_layers() -> usize {
    30
}
fn default_num_attention_heads() -> usize {
    32
}
fn default_num_key_value_heads() -> usize {
    32
}
fn default_vocab_size() -> usize {
    102400
}
fn default_intermediate_size() -> usize {
    11008
}
fn default_image_token_num_tokens() -> usize {
    16384
}
fn default_image_size() -> usize {
    384
}
fn default_patch_size() -> usize {
    16
}

// ---------------------------------------------------------------------------
// impl JanusConfig
// ---------------------------------------------------------------------------

impl Default for JanusConfig {
    /// Returns the 7B defaults.
    fn default() -> Self {
        Self::janus_pro_7b()
    }
}

impl JanusConfig {
    /// Returns the number of image tokens produced by the vision encoder.
    ///
    /// This equals `(image_size / patch_size)²`.  For the 7B model that is
    /// `(384 / 16)² = 576`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use pcai_media_model::config::JanusConfig;
    /// assert_eq!(JanusConfig::janus_pro_7b().num_image_tokens(), 576);
    /// ```
    pub fn num_image_tokens(&self) -> usize {
        let patches = self.image_size / self.patch_size;
        patches * patches
    }

    /// Converts this config into a `candle_transformers` Llama
    /// [`Config`](candle_transformers::models::llama::Config) suitable for
    /// constructing the LLM backbone.
    ///
    /// The `use_flash_attn` flag is forwarded as-is; all other fields are
    /// derived from `self`.
    pub fn to_llama_config(
        &self,
        use_flash_attn: bool,
    ) -> candle_transformers::models::llama::Config {
        candle_transformers::models::llama::Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            use_flash_attn,
            // DeepSeek models use 1e-6 RMS norm epsilon.
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: candle_transformers::models::llama::DEFAULT_MAX_SEQ_LEN,
            tie_word_embeddings: false,
        }
    }

    /// Preset for **Janus-Pro-1B**.
    ///
    /// Smaller model suitable for experimentation on consumer hardware.
    pub fn janus_pro_1b() -> Self {
        Self {
            hidden_size: 2048,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            vocab_size: default_vocab_size(),
            intermediate_size: 5504,
            image_token_num_tokens: default_image_token_num_tokens(),
            image_size: default_image_size(),
            patch_size: default_patch_size(),
        }
    }

    /// Preset for **Janus-Pro-7B** (default).
    ///
    /// Matches the publicly released `deepseek-ai/Janus-Pro-7B` configuration.
    pub fn janus_pro_7b() -> Self {
        Self {
            hidden_size: default_hidden_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            num_key_value_heads: default_num_key_value_heads(),
            vocab_size: default_vocab_size(),
            intermediate_size: default_intermediate_size(),
            image_token_num_tokens: default_image_token_num_tokens(),
            image_size: default_image_size(),
            patch_size: default_patch_size(),
        }
    }

    /// Deserialises a [`JanusConfig`] from a `config.json` file on disk.
    ///
    /// Fields absent from the file are filled with 7B defaults via serde's
    /// `#[serde(default)]` annotations.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the JSON is malformed.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let cfg = serde_json::from_str(&contents)?;
        Ok(cfg)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// 7B preset should yield 576 image tokens: (384/16)^2 = 576.
    #[test]
    fn test_7b_preset_image_tokens() {
        let cfg = JanusConfig::janus_pro_7b();
        assert_eq!(cfg.num_image_tokens(), 576);
    }

    /// 1B preset uses the same image encoder, so tokens are identical.
    #[test]
    fn test_1b_preset_image_tokens() {
        let cfg = JanusConfig::janus_pro_1b();
        // Same image_size / patch_size as 7B.
        assert_eq!(cfg.num_image_tokens(), 576);
        // But a smaller hidden dimension.
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 24);
    }

    /// Converted Llama config should mirror JanusConfig backbone fields.
    #[test]
    fn test_llama_config_conversion() {
        let janus = JanusConfig::janus_pro_7b();
        let llama = janus.to_llama_config(false);
        assert_eq!(llama.hidden_size, janus.hidden_size);
        assert_eq!(llama.intermediate_size, janus.intermediate_size);
        assert_eq!(llama.vocab_size, janus.vocab_size);
        assert_eq!(llama.num_hidden_layers, janus.num_hidden_layers);
        assert_eq!(llama.num_attention_heads, janus.num_attention_heads);
        assert_eq!(llama.num_key_value_heads, janus.num_key_value_heads);
        assert!(!llama.use_flash_attn);
    }

    /// Round-trip through JSON must preserve all fields.
    #[test]
    fn test_serde_roundtrip() {
        let original = JanusConfig::janus_pro_7b();
        let json = serde_json::to_string(&original).expect("serialise");
        let decoded: JanusConfig = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(decoded.hidden_size, original.hidden_size);
        assert_eq!(decoded.num_hidden_layers, original.num_hidden_layers);
        assert_eq!(decoded.num_attention_heads, original.num_attention_heads);
        assert_eq!(decoded.num_key_value_heads, original.num_key_value_heads);
        assert_eq!(decoded.vocab_size, original.vocab_size);
        assert_eq!(decoded.intermediate_size, original.intermediate_size);
        assert_eq!(decoded.image_token_num_tokens, original.image_token_num_tokens);
        assert_eq!(decoded.image_size, original.image_size);
        assert_eq!(decoded.patch_size, original.patch_size);
    }

    /// An empty JSON object `{}` should deserialise to the 7B defaults.
    #[test]
    fn test_serde_defaults() {
        let cfg: JanusConfig = serde_json::from_str("{}").expect("empty object deserialise");
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 30);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 32);
        assert_eq!(cfg.vocab_size, 102400);
        assert_eq!(cfg.intermediate_size, 11008);
        assert_eq!(cfg.image_token_num_tokens, 16384);
        assert_eq!(cfg.image_size, 384);
        assert_eq!(cfg.patch_size, 16);
        assert_eq!(cfg.num_image_tokens(), 576);
    }
}
