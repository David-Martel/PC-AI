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
//! // Default configuration — CPU, BF16, 1B model.
//! let cfg = PipelineConfig::default();
//! assert_eq!(cfg.model, "deepseek-ai/Janus-Pro-1B");
//!
//! let dev = cfg.resolve_device().unwrap();
//! let dtype = cfg.resolve_dtype();
//! ```

use std::{collections::HashSet, path::Path, process::Command, sync::Mutex};

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use serde::{Deserialize, Serialize};

/// Serialises all `std::env::set_var` / `remove_var` mutations in this module.
///
/// `set_var` and `remove_var` are deprecated since Rust 1.83 and have
/// undefined behaviour when called concurrently from multiple threads
/// (the C `setenv(3)` they delegate to is not thread-safe).  By holding this
/// lock for the entire duration of any env-var mutation + CUDA device open
/// sequence, we ensure only one thread modifies the environment at a time.
static CUDA_ENV_MUTEX: Mutex<()> = Mutex::new(());

// ---------------------------------------------------------------------------
// PipelineConfig
// ---------------------------------------------------------------------------

/// Configuration for the Janus-Pro generation pipeline.
///
/// Serialize / deserialise via [`serde_json`].  All fields have serde defaults
/// matching the documented 1B model settings used by the repo's default path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// HuggingFace model ID **or** absolute path to a local model directory.
    ///
    /// Defaults to `"deepseek-ai/Janus-Pro-1B"`.
    #[serde(default = "default_model")]
    pub model: String,

    /// Target compute device.
    ///
    /// Accepted values: `"cpu"`, `"cuda"`, `"cuda:auto"`, `"cuda:0"`, `"cuda:1"`, …
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

    /// Use the pre-allocated ring-buffer KV cache for image generation.
    ///
    /// When `true` (the default), [`PreAllocKvCache`] is used instead of the
    /// dynamic [`KvCache`].  The pre-allocated cache eliminates the ≈95 GB of
    /// GPU bandwidth waste from `Tensor::cat` across the 576 image-generation
    /// steps by writing each new KV pair in-place via `scatter_set` and reading
    /// accumulated history with zero-copy `narrow` views.
    ///
    /// Set to `false` to fall back to the original dynamic cache (useful for
    /// debugging or for devices where `scatter_set` is not supported).
    ///
    /// [`PreAllocKvCache`]: pcai_media_model::janus_llama::PreAllocKvCache
    /// [`KvCache`]: pcai_media_model::janus_llama::KvCache
    #[serde(default = "default_use_prealloc_kv_cache")]
    pub use_prealloc_kv_cache: bool,

    /// Enable self-speculative decoding for image token generation.
    ///
    /// When `true`, the generation loop uses a two-phase draft-then-verify
    /// scheme instead of the standard autoregressive loop:
    ///
    /// 1. **Draft phase**: run only the first [`speculative_draft_layers`]
    ///    transformer blocks to cheaply predict `K` candidate tokens.
    /// 2. **Verify phase**: run all layers on the `K` candidates in a single
    ///    batched forward pass and accept the longest consecutive prefix where
    ///    draft and verify agree.
    ///
    /// Requires [`use_prealloc_kv_cache`] to be `true` (the [`PreAllocKvCache`]
    /// provides the `seq_len` rollback needed on rejection).
    ///
    /// Expected speedup: ~1.5× at 75% acceptance rate, ~1.6× at 80%.
    ///
    /// Defaults to `false`.
    #[serde(default = "default_use_speculative_decoding")]
    pub use_speculative_decoding: bool,

    /// Number of transformer layers used in the speculative draft phase.
    ///
    /// Only relevant when [`use_speculative_decoding`] is `true`.  Must be
    /// strictly less than the total number of transformer blocks in the model
    /// (24 for Janus-Pro-1B; 36 for the 7B variant).
    ///
    /// A good starting point is one-third of the total layer count:
    /// `8` for the 1B model, `12` for the 7B model.  Fewer draft layers ⟹
    /// faster draft phase but lower acceptance rate; more draft layers ⟹
    /// slower draft phase but higher acceptance rate.
    ///
    /// Defaults to `8` (one-third of 24 for Janus-Pro-1B).
    #[serde(default = "default_speculative_draft_layers")]
    pub speculative_draft_layers: usize,

    /// Number of candidate tokens generated per draft phase (`K`).
    ///
    /// Only relevant when [`use_speculative_decoding`] is `true`.  Each
    /// speculative step produces at most `K` accepted tokens for the cost of
    /// `K` draft forwards plus one verify forward (all `K` in parallel).
    ///
    /// Higher `K` ⟹ better amortisation of verify cost, but lower per-token
    /// acceptance probability (`acceptance_rate^K` drops steeply).  Values of
    /// 3–5 are typical; `4` is the default.
    ///
    /// Defaults to `4`.
    #[serde(default = "default_speculative_lookahead")]
    pub speculative_lookahead: usize,
}

// ---------------------------------------------------------------------------
// serde default helpers
// ---------------------------------------------------------------------------

fn default_model() -> String {
    "deepseek-ai/Janus-Pro-1B".to_string()
}
fn default_device() -> String {
    "cpu".to_string()
}
fn default_dtype() -> String {
    "bf16".to_string()
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
fn default_use_prealloc_kv_cache() -> bool {
    true
}
fn default_use_speculative_decoding() -> bool {
    false
}
fn default_speculative_draft_layers() -> usize {
    8
}
fn default_speculative_lookahead() -> usize {
    4
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
            use_prealloc_kv_cache: default_use_prealloc_kv_cache(),
            use_speculative_decoding: default_use_speculative_decoding(),
            speculative_draft_layers: default_speculative_draft_layers(),
            speculative_lookahead: default_speculative_lookahead(),
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
    /// - `"cuda:auto"` — prefers the highest-memory GPU reported by `nvidia-smi`.
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
        if s == "cuda:auto" {
            return Self::resolve_auto_cuda_device();
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
        // Acquire the env-var mutex before calling into the _locked helpers.
        let _guard = CUDA_ENV_MUTEX.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        Self::resolve_cuda_ordinal_locked(ordinal).with_context(|| format!("failed to open CUDA device {ordinal}"))
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
                tracing::warn!(dtype = other, "unrecognised dtype '{}'; falling back to F32", other);
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

    fn resolve_auto_cuda_device() -> Result<Device> {
        // Hold the env-var mutex for the entire auto-selection routine.
        // Every branch that calls set_var / remove_var (directly or via
        // resolve_cuda_candidate / resolve_cuda_with_visible_mapping /
        // ensure_cuda_device_order) is covered by this single lock acquisition.
        let _guard = CUDA_ENV_MUTEX.lock().unwrap_or_else(std::sync::PoisonError::into_inner);

        // Fast path: when NVML is compiled in, ask it for the best GPU and try
        // to open it immediately.  NVML avoids spawning a subprocess and gives
        // us live free-memory information (rather than total-memory from
        // nvidia-smi), so we prefer it.  If NVML fails for any reason — driver
        // not installed, NVML not available in a container, device open error —
        // we fall through to the full nvidia-smi candidate loop below.
        #[cfg(any(feature = "cuda", feature = "nvml"))]
        {
            match Self::best_nvml_gpu() {
                Ok(Some(best)) => {
                    tracing::info!(
                        gpu_index = best.index,
                        gpu_name = %best.name,
                        memory_total_mb = best.memory_total_mb,
                        memory_used_mb = best.memory_used_mb,
                        "NVML: selected best available GPU by free memory"
                    );
                    let candidate = Self::candidate_from_nvml_gpu(&best);
                    match Self::resolve_cuda_candidate_locked(&candidate) {
                        Ok(device) => {
                            tracing::info!(
                                gpu_index = best.index,
                                gpu_name = %best.name,
                                "NVML: successfully opened CUDA device"
                            );
                            return Ok(device);
                        }
                        Err(err) => {
                            tracing::warn!(
                                gpu_index = best.index,
                                gpu_name = %best.name,
                                error = %err,
                                "NVML: CUDA device open failed; falling back to nvidia-smi inventory"
                            );
                        }
                    }
                }
                Ok(None) => {
                    tracing::debug!("NVML: no GPUs found; falling back to nvidia-smi inventory");
                }
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "NVML: GPU query failed; falling back to nvidia-smi inventory"
                    );
                }
            }
        }

        Self::ensure_cuda_device_order_locked();
        // If CUDA_VISIBLE_DEVICES is pre-set, honour it — but fall through to
        // the GPU inventory on failure instead of aborting immediately.
        if let Ok(visible_devices) = std::env::var("CUDA_VISIBLE_DEVICES") {
            if !visible_devices.trim().is_empty() {
                tracing::info!(
                    visible_devices = %visible_devices,
                    "honouring preconfigured CUDA_VISIBLE_DEVICES for cuda:auto"
                );
                match Device::new_cuda(0) {
                    Ok(dev) => return Ok(dev),
                    Err(err) => {
                        tracing::warn!(
                            error = %err,
                            visible_devices = %visible_devices,
                            "preconfigured CUDA_VISIBLE_DEVICES failed; clearing and trying GPU inventory"
                        );
                        // SAFETY: CUDA_ENV_MUTEX is held; no other thread is
                        // reading or writing the process environment concurrently
                        // inside this crate.
                        unsafe { std::env::remove_var("CUDA_VISIBLE_DEVICES") };
                    }
                }
            }
        }

        let gpus = Self::all_cuda_candidates()?;
        if gpus.is_empty() {
            tracing::warn!("GPU inventory unavailable; falling back to CUDA device 0");
            return Self::resolve_cuda_ordinal_locked(0);
        }

        // Try each GPU in descending memory order.  This ensures that if the
        // highest-VRAM device cannot initialise (e.g. driver/toolkit mismatch,
        // unsupported compute capability) we gracefully fall back to the next.
        let mut last_error = None;
        for gpu in &gpus {
            tracing::info!(
                physical_ordinal = gpu.index,
                gpu_name = %gpu.name,
                memory_mb = gpu.memory_mb,
                "attempting CUDA device"
            );
            // Clear any prior CUDA_VISIBLE_DEVICES mapping from a failed attempt.
            // SAFETY: CUDA_ENV_MUTEX is held for the duration of this loop.
            unsafe { std::env::remove_var("CUDA_VISIBLE_DEVICES") };
            match Self::resolve_cuda_candidate_locked(gpu) {
                Ok(device) => {
                    tracing::info!(
                        physical_ordinal = gpu.index,
                        gpu_name = %gpu.name,
                        gpu_uuid = gpu.uuid.as_deref().unwrap_or(""),
                        "successfully opened CUDA device"
                    );
                    return Ok(device);
                }
                Err(err) => {
                    tracing::warn!(
                        physical_ordinal = gpu.index,
                        gpu_name = %gpu.name,
                        error = %err,
                        "CUDA device open failed; trying next GPU"
                    );
                    last_error = Some(err);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("no CUDA devices available")))
    }

    /// Requires `CUDA_ENV_MUTEX` to be held by the caller.
    fn resolve_cuda_candidate_locked(gpu: &NvidiaSmiGpu) -> Result<Device> {
        Self::ensure_cuda_device_order_locked();
        match Device::new_cuda(gpu.index) {
            Ok(device) => Ok(device),
            Err(original_error) => {
                if let Some(uuid) = gpu.uuid.as_deref() {
                    tracing::warn!(
                        physical_ordinal = gpu.index,
                        gpu_name = %gpu.name,
                        gpu_uuid = uuid,
                        error = %original_error,
                        "direct CUDA ordinal open failed; retrying with UUID-based CUDA_VISIBLE_DEVICES remap"
                    );
                    Self::resolve_cuda_with_visible_mapping_locked(uuid, &format!("GPU UUID {uuid}"))
                } else {
                    tracing::warn!(
                        physical_ordinal = gpu.index,
                        gpu_name = %gpu.name,
                        error = %original_error,
                        "GPU UUID unavailable; retrying with ordinal CUDA_VISIBLE_DEVICES remap"
                    );
                    Self::resolve_cuda_with_visible_mapping_locked(
                        &gpu.index.to_string(),
                        &format!("physical GPU {}", gpu.index),
                    )
                }
            }
        }
    }

    /// Requires `CUDA_ENV_MUTEX` to be held by the caller.
    fn resolve_cuda_ordinal_locked(ordinal: usize) -> Result<Device> {
        Self::ensure_cuda_device_order_locked();
        match Device::new_cuda(ordinal) {
            Ok(device) => Ok(device),
            Err(original_error) if ordinal > 0 => {
                tracing::warn!(
                    physical_ordinal = ordinal,
                    error = %original_error,
                    "direct CUDA ordinal open failed; retrying with CUDA_VISIBLE_DEVICES remap"
                );
                Self::resolve_cuda_with_visible_mapping_locked(&ordinal.to_string(), &format!("physical GPU {ordinal}"))
            }
            Err(error) => Err(error.into()),
        }
    }

    /// Set `CUDA_VISIBLE_DEVICES` to `visible_device` and attempt to open
    /// CUDA device 0.
    ///
    /// Requires `CUDA_ENV_MUTEX` to be held by the caller.
    fn resolve_cuda_with_visible_mapping_locked(visible_device: &str, description: &str) -> Result<Device> {
        Self::ensure_cuda_device_order_locked();
        // SAFETY: CUDA_ENV_MUTEX is held by the caller; no other thread in
        // this crate reads or writes the process environment concurrently.
        unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", visible_device) };
        Device::new_cuda(0).with_context(|| {
            format!(
                "failed to open CUDA device 0 after mapping {description} via \
                 CUDA_VISIBLE_DEVICES={visible_device}"
            )
        })
    }

    /// Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` if it is not already configured.
    ///
    /// Requires `CUDA_ENV_MUTEX` to be held by the caller.
    fn ensure_cuda_device_order_locked() {
        if std::env::var("CUDA_DEVICE_ORDER")
            .map(|value| value.trim().is_empty())
            .unwrap_or(true)
        {
            // SAFETY: CUDA_ENV_MUTEX is held by the caller; no other thread in
            // this crate reads or writes the process environment concurrently.
            unsafe { std::env::set_var("CUDA_DEVICE_ORDER", "PCI_BUS_ID") };
        }
    }

    /// Return all GPU candidates for `cuda:auto`, preferring NVML inventory when
    /// available and falling back to `nvidia-smi`.
    fn all_cuda_candidates() -> Result<Vec<NvidiaSmiGpu>> {
        let mut seen = HashSet::new();
        let mut gpus = Vec::new();

        #[cfg(any(feature = "cuda", feature = "nvml"))]
        {
            if let Ok(Some(best_gpu)) = Self::best_nvml_gpu() {
                let candidate = Self::candidate_from_nvml_gpu(&best_gpu);
                let dedupe_key = Self::candidate_dedupe_key(&candidate);
                seen.insert(dedupe_key);
                gpus.push(candidate);
            }
        }

        for gpu in Self::all_nvidia_smi_gpus()? {
            let dedupe_key = Self::candidate_dedupe_key(&gpu);
            if seen.insert(dedupe_key) {
                gpus.push(gpu);
            }
        }

        Ok(gpus)
    }

    fn candidate_dedupe_key(gpu: &NvidiaSmiGpu) -> String {
        gpu.uuid.clone().unwrap_or_else(|| format!("index:{}", gpu.index))
    }

    /// Return all GPUs reported by `nvidia-smi`, sorted by memory descending
    /// (highest-VRAM first).  Returns an empty `Vec` when `nvidia-smi` is
    /// unavailable or reports no devices.
    fn all_nvidia_smi_gpus() -> Result<Vec<NvidiaSmiGpu>> {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=index,uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output();

        let output = match output {
            Ok(output) if output.status.success() => output,
            Ok(output) => {
                tracing::warn!(status = ?output.status.code(), "nvidia-smi query failed");
                return Ok(Vec::new());
            }
            Err(error) => {
                tracing::warn!(error = %error, "unable to run nvidia-smi for GPU inventory");
                return Ok(Vec::new());
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut gpus: Vec<NvidiaSmiGpu> = stdout.lines().filter_map(NvidiaSmiGpu::parse).collect();
        // Sort by memory descending, then by index ascending (stable tiebreak).
        gpus.sort_by_key(|gpu| (std::cmp::Reverse(gpu.memory_mb), gpu.index));
        Ok(gpus)
    }

    #[cfg(any(feature = "cuda", feature = "nvml"))]
    fn best_nvml_gpu() -> Result<Option<pcai_core_lib::gpu::GpuInfo>> {
        pcai_core_lib::gpu::best_available_gpu()
    }

    #[cfg(any(feature = "cuda", feature = "nvml"))]
    fn candidate_from_nvml_gpu(gpu: &pcai_core_lib::gpu::GpuInfo) -> NvidiaSmiGpu {
        NvidiaSmiGpu {
            index: gpu.index as usize,
            uuid: Some(gpu.uuid.clone()),
            name: gpu.name.clone(),
            memory_mb: gpu.memory_total_mb as usize,
        }
    }
}

#[derive(Debug, Clone)]
struct NvidiaSmiGpu {
    index: usize,
    uuid: Option<String>,
    name: String,
    memory_mb: usize,
}

impl NvidiaSmiGpu {
    fn parse(line: &str) -> Option<Self> {
        let mut parts = line.split(',').map(str::trim);
        let index = parts.next()?.parse().ok()?;
        let uuid = parts.next().map(str::to_string).filter(|value| !value.is_empty());
        let name = parts.next()?.to_string();
        let memory_mb = parts.next()?.parse().ok()?;
        Some(Self {
            index,
            uuid,
            name,
            memory_mb,
        })
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
        assert_eq!(cfg.model, "deepseek-ai/Janus-Pro-1B");
        assert_eq!(cfg.device, "cpu");
        assert_eq!(cfg.dtype, "bf16");
        assert!((cfg.guidance_scale - 5.0).abs() < f64::EPSILON);
        assert!((cfg.temperature - 1.0).abs() < f64::EPSILON);
        assert_eq!(cfg.parallel_size, 1);
        assert_eq!(cfg.gpu_layers, 0);
        assert!(
            cfg.use_prealloc_kv_cache,
            "prealloc KV cache should be enabled by default"
        );
        assert!(
            !cfg.use_speculative_decoding,
            "speculative decoding should be disabled by default"
        );
        assert_eq!(cfg.speculative_draft_layers, 8, "default draft layers should be 8");
        assert_eq!(cfg.speculative_lookahead, 4, "default lookahead K should be 4");
    }

    /// An empty JSON object `{}` must deserialise to the same defaults.
    #[test]
    fn test_serde_empty_object() {
        let cfg: PipelineConfig = serde_json::from_str("{}").expect("empty object should be valid");
        assert_eq!(cfg.model, "deepseek-ai/Janus-Pro-1B");
        assert_eq!(cfg.device, "cpu");
        assert_eq!(cfg.dtype, "bf16");
        assert!((cfg.guidance_scale - 5.0).abs() < f64::EPSILON);
        assert!((cfg.temperature - 1.0).abs() < f64::EPSILON);
        assert_eq!(cfg.parallel_size, 1);
        assert_eq!(cfg.gpu_layers, 0);
        assert!(cfg.use_prealloc_kv_cache, "prealloc KV cache should be the default");
        assert!(
            !cfg.use_speculative_decoding,
            "speculative decoding should be off by default"
        );
        assert_eq!(cfg.speculative_draft_layers, 8);
        assert_eq!(cfg.speculative_lookahead, 4);
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
            use_prealloc_kv_cache: false,
            use_speculative_decoding: true,
            speculative_draft_layers: 8,
            speculative_lookahead: 4,
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
        assert_eq!(decoded.use_prealloc_kv_cache, original.use_prealloc_kv_cache);
        assert_eq!(decoded.use_speculative_decoding, original.use_speculative_decoding);
        assert_eq!(decoded.speculative_draft_layers, original.speculative_draft_layers);
        assert_eq!(decoded.speculative_lookahead, original.speculative_lookahead);
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

    #[test]
    fn test_nvidia_smi_gpu_parse() {
        let parsed =
            NvidiaSmiGpu::parse("1, GPU-12345678-90ab-cdef-1234-567890abcdef, NVIDIA GeForce RTX 5060 Ti, 16311")
                .expect("parse gpu line");
        assert_eq!(parsed.index, 1);
        assert_eq!(parsed.uuid.as_deref(), Some("GPU-12345678-90ab-cdef-1234-567890abcdef"));
        assert_eq!(parsed.name, "NVIDIA GeForce RTX 5060 Ti");
        assert_eq!(parsed.memory_mb, 16_311);
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
            use_prealloc_kv_cache: true,
            use_speculative_decoding: true,
            speculative_draft_layers: 8,
            speculative_lookahead: 4,
        };
        let tmp = std::env::temp_dir().join("pcai_media_config_test.json");
        let json = serde_json::to_string_pretty(&original).expect("serialise");
        std::fs::write(&tmp, &json).expect("write temp file");
        let loaded = PipelineConfig::from_file(&tmp).expect("from_file");
        assert_eq!(loaded.model, original.model);
        assert_eq!(loaded.dtype, original.dtype);
        assert_eq!(loaded.use_prealloc_kv_cache, original.use_prealloc_kv_cache);
        assert_eq!(loaded.use_speculative_decoding, original.use_speculative_decoding);
        assert_eq!(loaded.speculative_draft_layers, original.speculative_draft_layers);
        assert_eq!(loaded.speculative_lookahead, original.speculative_lookahead);
        std::fs::remove_file(&tmp).ok();
    }

    /// `use_speculative_decoding = true` with valid draft layers must round-trip correctly.
    #[test]
    fn test_speculative_decoding_serde() {
        let json = r#"{
            "use_speculative_decoding": true,
            "speculative_draft_layers": 12,
            "speculative_lookahead": 3
        }"#;
        let cfg: PipelineConfig = serde_json::from_str(json).expect("deserialise speculative config");
        assert!(cfg.use_speculative_decoding);
        assert_eq!(cfg.speculative_draft_layers, 12);
        assert_eq!(cfg.speculative_lookahead, 3);
        // Other fields should still carry defaults.
        assert_eq!(cfg.model, "deepseek-ai/Janus-Pro-1B");
        assert!(cfg.use_prealloc_kv_cache);
    }
}
