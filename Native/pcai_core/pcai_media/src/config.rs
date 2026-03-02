//! Pipeline configuration for the Janus-Pro media pipeline.
//!
//! [`PipelineConfig`] controls which model to load, which hardware device to
//! target, what numeric dtype to use, and the generation hyper-parameters.
//!
//! All fields carry serde defaults so a minimal JSON file such as `{}` is
//! valid and produces reasonable settings.
//!
//! # Example
//!
//! ```rust
//! use pcai_media::config::PipelineConfig;
//!
//! // Default configuration — CPU, F32, 7B model.
//! let cfg = PipelineConfig::default();
//! assert_eq!(cfg.model, "deepseek-ai/Janus-Pro-7B");
//!
//! let dev = cfg.resolve_device().unwrap();
//! let dtype = cfg.resolve_dtype();
//! ```

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// PipelineConfig
// ---------------------------------------------------------------------------

/// Configuration for the Janus-Pro generation pipeline.
///
/// Serialize / deserialise via [`serde_json`].  All fields have serde defaults
/// matching the documented 7B model settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// HuggingFace model ID **or** absolute path to a local model directory.
    ///
    /// Defaults to `"deepseek-ai/Janus-Pro-7B"`.
    #[serde(default = "default_model")]
    pub model: String,

    /// Target compute device.
    ///
    /// Accepted values: `"cpu"`, `"cuda"`, `"cuda:0"`, `"cuda:1"`, …
    /// Defaults to `"cpu"`.
    #[serde(default = "default_device")]
    pub device: String,

    /// Floating-point dtype for inference.
    ///
    /// Accepted values: `"f32"`, `"f16"`, `"bf16"`.
    /// Defaults to `"f32"`.
    #[serde(default = "default_dtype")]
    pub dtype: String,

    /// Classifier-Free Guidance scale.
    ///
    /// Higher values push the model to follow the prompt more strictly.
    /// Defaults to `5.0`.
    #[serde(default = "default_guidance_scale")]
    pub guidance_scale: f64,

    /// Sampling temperature applied to image-token logits.
    ///
    /// `1.0` is neutral; lower values sharpen the distribution.
    /// Defaults to `1.0`.
    #[serde(default = "default_temperature")]
    pub temperature: f64,

    /// Number of images to generate in parallel.
    ///
    /// Internally the batch is doubled (positive + negative for CFG).
    /// Defaults to `1`.
    #[serde(default = "default_parallel_size")]
    pub parallel_size: usize,

    /// Number of model layers to offload to the GPU (`0` = CPU only).
    ///
    /// Currently reserved for future use in mixed CPU/GPU execution.
    /// Defaults to `0`.
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: i32,
}

// ---------------------------------------------------------------------------
// serde default helpers
// ---------------------------------------------------------------------------

fn default_model() -> String {
    "deepseek-ai/Janus-Pro-7B".to_string()
}
fn default_device() -> String {
    "cpu".to_string()
}
fn default_dtype() -> String {
    "f32".to_string()
}
fn default_guidance_scale() -> f64 {
    5.0
}
fn default_temperature() -> f64 {
    1.0
}
fn default_parallel_size() -> usize {
    1
}
fn default_gpu_layers() -> i32 {
    0
}

// ---------------------------------------------------------------------------
// Default impl
// ---------------------------------------------------------------------------

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            model: default_model(),
            device: default_device(),
            dtype: default_dtype(),
            guidance_scale: default_guidance_scale(),
            temperature: default_temperature(),
            parallel_size: default_parallel_size(),
            gpu_layers: default_gpu_layers(),
        }
    }
}

// ---------------------------------------------------------------------------
// Methods
// ---------------------------------------------------------------------------

impl PipelineConfig {
    /// Resolve the `device` string to a [`candle_core::Device`].
    ///
    /// Supported strings:
    /// - `"cpu"` — returns [`Device::Cpu`].
    /// - `"cuda"` or `"cuda:0"` through `"cuda:N"` — returns a CUDA device.
    ///
    /// # Errors
    ///
    /// Returns an error if a CUDA device is requested but CUDA is unavailable,
    /// or if the device string cannot be parsed.
    pub fn resolve_device(&self) -> Result<Device> {
        let s = self.device.trim().to_lowercase();
        if s == "cpu" {
            return Ok(Device::Cpu);
        }
        // Accept "cuda" or "cuda:N".
        let ordinal: usize = if s == "cuda" {
            0
        } else if let Some(rest) = s.strip_prefix("cuda:") {
            rest.parse::<usize>()
                .with_context(|| format!("invalid CUDA ordinal in device string '{}'", self.device))?
        } else {
            anyhow::bail!("unrecognised device '{}'; expected 'cpu' or 'cuda[:N]'", self.device);
        };
        Device::new_cuda(ordinal)
            .with_context(|| format!("failed to open CUDA device {ordinal}"))
    }

    /// Resolve the `dtype` string to a [`candle_core::DType`].
    ///
    /// Supported strings (case-insensitive): `"f32"`, `"f16"`, `"bf16"`.
    /// Any unrecognised value falls back to `DType::F32` with a warning log.
    pub fn resolve_dtype(&self) -> DType {
        match self.dtype.trim().to_lowercase().as_str() {
            "f16" => DType::F16,
            "bf16" => DType::BF16,
            "f32" => DType::F32,
            other => {
                tracing::warn!(
                    dtype = other,
                    "unrecognised dtype '{}'; falling back to F32",
                    other
                );
                DType::F32
            }
        }
    }

    /// Load a [`PipelineConfig`] from a JSON file on disk.
    ///
    /// Missing fields in the file are filled with defaults.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the JSON is malformed.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read pipeline config from '{}'", path.display()))?;
        let cfg = serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse pipeline config from '{}'", path.display()))?;
        Ok(cfg)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Default config should carry the documented 7B model ID.
    #[test]
    fn test_default_model_id() {
        let cfg = PipelineConfig::default();
        assert_eq!(cfg.model, "deepseek-ai/Janus-Pro-7B");
        assert_eq!(cfg.device, "cpu");
        assert_eq!(cfg.dtype, "f32");
        assert!((cfg.guidance_scale - 5.0).abs() < f64::EPSILON);
        assert!((cfg.temperature - 1.0).abs() < f64::EPSILON);
        assert_eq!(cfg.parallel_size, 1);
        assert_eq!(cfg.gpu_layers, 0);
    }

    /// An empty JSON object `{}` must deserialise to the same defaults.
    #[test]
    fn test_serde_empty_object() {
        let cfg: PipelineConfig = serde_json::from_str("{}").expect("empty object should be valid");
        assert_eq!(cfg.model, "deepseek-ai/Janus-Pro-7B");
        assert_eq!(cfg.device, "cpu");
        assert_eq!(cfg.dtype, "f32");
        assert!((cfg.guidance_scale - 5.0).abs() < f64::EPSILON);
        assert!((cfg.temperature - 1.0).abs() < f64::EPSILON);
        assert_eq!(cfg.parallel_size, 1);
        assert_eq!(cfg.gpu_layers, 0);
    }

    /// Round-trip through JSON must preserve all fields.
    #[test]
    fn test_serde_roundtrip() {
        let original = PipelineConfig {
            model: "deepseek-ai/Janus-Pro-1B".to_string(),
            device: "cpu".to_string(),
            dtype: "f16".to_string(),
            guidance_scale: 7.5,
            temperature: 0.8,
            parallel_size: 2,
            gpu_layers: 20,
        };
        let json = serde_json::to_string(&original).expect("serialise failed");
        let decoded: PipelineConfig = serde_json::from_str(&json).expect("deserialise failed");
        assert_eq!(decoded.model, original.model);
        assert_eq!(decoded.device, original.device);
        assert_eq!(decoded.dtype, original.dtype);
        assert!((decoded.guidance_scale - original.guidance_scale).abs() < f64::EPSILON);
        assert!((decoded.temperature - original.temperature).abs() < f64::EPSILON);
        assert_eq!(decoded.parallel_size, original.parallel_size);
        assert_eq!(decoded.gpu_layers, original.gpu_layers);
    }

    /// `resolve_device` on `"cpu"` must return `Device::Cpu`.
    #[test]
    fn test_resolve_device_cpu() {
        let cfg = PipelineConfig::default(); // device = "cpu"
        let dev = cfg.resolve_device().expect("cpu device should always succeed");
        assert!(matches!(dev, Device::Cpu));
    }

    /// Unrecognised device string must return an error.
    #[test]
    fn test_resolve_device_unknown() {
        let cfg = PipelineConfig {
            device: "tpu".to_string(),
            ..PipelineConfig::default()
        };
        assert!(cfg.resolve_device().is_err());
    }

    /// `resolve_dtype` must map recognised strings correctly.
    #[test]
    fn test_resolve_dtype_variants() {
        let cases = [
            ("f32", DType::F32),
            ("f16", DType::F16),
            ("bf16", DType::BF16),
            ("F32", DType::F32),
            ("BF16", DType::BF16),
        ];
        for (input, expected) in cases {
            let cfg = PipelineConfig {
                dtype: input.to_string(),
                ..PipelineConfig::default()
            };
            assert_eq!(cfg.resolve_dtype(), expected, "failed for dtype='{input}'");
        }
    }

    /// An unrecognised dtype string must fall back to F32 without panicking.
    #[test]
    fn test_resolve_dtype_unknown_falls_back() {
        let cfg = PipelineConfig {
            dtype: "fp8".to_string(),
            ..PipelineConfig::default()
        };
        assert_eq!(cfg.resolve_dtype(), DType::F32);
    }

    /// `from_file` must round-trip a written config.
    #[test]
    fn test_from_file_roundtrip() {
        let original = PipelineConfig {
            model: "local/model".to_string(),
            device: "cpu".to_string(),
            dtype: "bf16".to_string(),
            guidance_scale: 3.0,
            temperature: 0.5,
            parallel_size: 1,
            gpu_layers: 0,
        };
        let tmp = std::env::temp_dir().join("pcai_media_config_test.json");
        let json = serde_json::to_string_pretty(&original).expect("serialise");
        std::fs::write(&tmp, &json).expect("write temp file");
        let loaded = PipelineConfig::from_file(&tmp).expect("from_file");
        assert_eq!(loaded.model, original.model);
        assert_eq!(loaded.dtype, original.dtype);
        std::fs::remove_file(&tmp).ok();
    }
}
