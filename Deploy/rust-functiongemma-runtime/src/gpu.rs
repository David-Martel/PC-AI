#![cfg(feature = "model")]

use crate::config::{runtime_config, RuntimeConfig};
use candle_core::Device;
use rust_functiongemma_core::gpu::{
    configure_and_log_cuda_mem_pool, log_cuda_snapshot, resolve_device_with_index, CudaMemPoolConfig,
    DeviceSelectionParams,
};
use rust_functiongemma_core::{normalize_device_label, GpuInfo};
use std::sync::OnceLock;

pub(crate) use rust_functiongemma_core::default_dtype;

pub(crate) static CUDA_MEMPOOL_CONFIGURED: OnceLock<()> = OnceLock::new();

/// Cached nvidia-smi query results. Queried once at first use and reused for
/// all subsequent device resolution, label, and auto-index calls.
static NVIDIA_SMI_CACHE: OnceLock<Vec<GpuInfo>> = OnceLock::new();

/// Return cached nvidia-smi GPU list. The subprocess is executed at most once
/// per process lifetime; subsequent calls return the cached result.
pub(crate) fn query_nvidia_smi() -> Vec<GpuInfo> {
    NVIDIA_SMI_CACHE
        .get_or_init(|| {
            let visible = parse_visible_devices().unwrap_or_default();
            rust_functiongemma_core::gpu::query_nvidia_smi(runtime_config().min_vram_mb, &visible)
        })
        .clone()
}

pub(crate) fn auto_cuda_index() -> Option<usize> {
    let visible = parse_visible_devices().unwrap_or_default();
    rust_functiongemma_core::gpu::auto_cuda_index(runtime_config().min_vram_mb, &visible)
}

pub(crate) fn parse_visible_devices() -> Option<Vec<usize>> {
    let list = runtime_config().cuda_visible_devices.clone();
    if list.is_empty() {
        return None;
    }
    Some(list)
}

pub(crate) fn resolve_device(cfg: &RuntimeConfig) -> (Device, Option<usize>) {
    let params = DeviceSelectionParams {
        device_label: &cfg.router_device,
        gpu_index: cfg.router_gpu,
        force_cpu: !cfg!(feature = "cuda"),
        min_vram_mb: cfg.min_vram_mb,
        cuda_visible_devices: &cfg.cuda_visible_devices,
    };
    resolve_device_with_index(&params)
}

pub(crate) fn device_label(cfg: &RuntimeConfig) -> String {
    if !cfg!(feature = "cuda") {
        return "cpu".to_string();
    }
    let configured_device = cfg.router_device.trim().to_string();
    if !configured_device.is_empty() && configured_device.to_lowercase() != "auto" {
        return normalize_device_label(&configured_device);
    }
    if let Some(gpu) = cfg.router_gpu {
        return format!("cuda:{}", gpu);
    }
    if let Some(idx) = auto_cuda_index() {
        if let Some(info) = query_nvidia_smi().into_iter().find(|g| g.runtime_index == idx) {
            return format!("cuda:{} ({} {} MB)", idx, info.name, info.memory_mb);
        }
        return format!("cuda:{}", idx);
    }
    "cpu".to_string()
}

pub(crate) fn maybe_configure_cuda_mem_pool(device_index: Option<usize>) {
    if !runtime_config().router_cuda_mem_pool {
        return;
    }
    let idx = match device_index {
        Some(v) => v,
        None => return,
    };
    if CUDA_MEMPOOL_CONFIGURED.get().is_some() {
        return;
    }
    let cfg = CudaMemPoolConfig {
        enable: true,
        release_threshold_mb: runtime_config().router_cuda_mem_pool_release_threshold_mb,
        reuse_follow_event_dependencies: true,
        reuse_allow_opportunistic: true,
        reuse_allow_internal_dependencies: true,
        trim_to_mb: runtime_config().router_cuda_mem_pool_trim_mb,
    };
    configure_and_log_cuda_mem_pool(idx, cfg);
    let _ = CUDA_MEMPOOL_CONFIGURED.set(());
}

pub(crate) fn maybe_log_cuda_snapshot(tag: &str, device_index: Option<usize>) {
    if !runtime_config().router_cuda_mem_snapshot {
        return;
    }
    log_cuda_snapshot(tag, device_index);
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::*;

    // Protects: the CPU model build uses and reports CPU even with CUDA options present.
    // Detects: implicit GPU selection, a misleading health label, or a missing CPU tensor backend.
    // Needs: the model feature, no CUDA feature, and host CPU only.
    // Breadcrumb: resolve_device, device_label and FunctionGemma core's force_cpu selection.
    #[test]
    fn cpu_model_uses_cpu_even_with_cuda_preferences() -> anyhow::Result<()> {
        for label in ["auto", "cpu", "cuda:1"] {
            let cfg = RuntimeConfig {
                router_device: label.to_string(),
                router_gpu: Some(1),
                cuda_visible_devices: vec![1],
                ..RuntimeConfig::default()
            };

            let (device, cuda_index) = resolve_device(&cfg);
            assert!(matches!(device, Device::Cpu));
            assert_eq!(cuda_index, None);
            assert_eq!(device_label(&cfg), "cpu");
            let values = candle_core::Tensor::new(&[2.0f32, 3.0], &device)?;
            assert_eq!(values.sqr()?.to_vec1::<f32>()?, [4.0, 9.0]);
        }
        Ok(())
    }
}
