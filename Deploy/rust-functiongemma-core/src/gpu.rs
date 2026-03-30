use anyhow::{Context, Result};
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device};
use std::ffi::c_void;
use std::process::Command;
#[cfg(feature = "nvml")]
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// NVML singleton (feature-gated)
// ---------------------------------------------------------------------------

/// Process-wide NVML singleton.
///
/// NVML must not be initialised and torn down on every call — the library
/// loads the kernel driver on `Nvml::init()` and unloads it on `Drop`.
/// Repeated init/drop cycles waste time and can cause driver-level reference
/// counting issues on some platforms.
///
/// `OnceLock` guarantees that `Nvml::init()` is called exactly once across all
/// threads; subsequent calls return a reference to the already-initialised
/// instance.  The `Option` encodes permanent initialisation failure so callers
/// can fall back without retrying.
#[cfg(feature = "nvml")]
static NVML: OnceLock<Option<nvml_wrapper::Nvml>> = OnceLock::new();

/// Return a reference to the process-wide NVML singleton, initialising it on
/// the first call.
///
/// Returns `None` (after logging via tracing) if NVML is unavailable — e.g. no
/// NVIDIA driver is installed or the library is not present.
#[cfg(feature = "nvml")]
fn get_nvml() -> Option<&'static nvml_wrapper::Nvml> {
    NVML.get_or_init(|| match nvml_wrapper::Nvml::init() {
        Ok(nvml) => Some(nvml),
        Err(err) => {
            tracing::warn!(
                "NVML initialisation failed (no NVIDIA driver or NVML not \
                 available): {err}"
            );
            None
        }
    })
    .as_ref()
}

// ---------------------------------------------------------------------------
// GPU scoring heuristic (shared between NVML and nvidia-smi paths)
// ---------------------------------------------------------------------------

/// Score a GPU by VRAM size with bonuses for known high-performance models.
/// Higher score = preferred device for `auto_cuda_index`.
fn score_gpu(name: &str, memory_mb: u64) -> i64 {
    let mut score = memory_mb as i64;
    let name_lower = name.to_lowercase();
    if name_lower.contains("5060") {
        score += 1_000_000;
    }
    if name_lower.contains("rtx") && name_lower.contains("2000") {
        score += 500_000;
    }
    if name_lower.contains("ada") {
        score += 50_000;
    }
    score
}

// ---------------------------------------------------------------------------
// NVML GPU discovery (~1ms vs ~800ms for nvidia-smi)
// ---------------------------------------------------------------------------

/// Query GPUs via NVML, applying the same filtering and scoring as
/// [`query_nvidia_smi`].  Returns an empty `Vec` if NVML is unavailable.
///
/// This is ~800x faster than spawning an nvidia-smi subprocess because NVML
/// uses direct driver ioctls rather than process creation + CSV parsing.
#[cfg(feature = "nvml")]
pub fn query_gpus_nvml(min_vram_mb: Option<u64>, visible_devices: &[usize]) -> Vec<GpuInfo> {
    let nvml = match get_nvml() {
        Some(n) => n,
        None => return Vec::new(),
    };

    let count = match nvml.device_count() {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut gpus = Vec::new();
    for raw_idx in 0..count {
        let idx = raw_idx as usize;

        if !visible_devices.is_empty() && !visible_devices.contains(&idx) {
            continue;
        }

        let device = match nvml.device_by_index(raw_idx) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let name = device.name().unwrap_or_else(|_| format!("GPU {idx}"));

        let memory_mb = device.memory_info().map(|mi| mi.total / (1024 * 1024)).unwrap_or(0);

        if let Some(min) = min_vram_mb {
            if memory_mb < min {
                continue;
            }
        }

        let score = score_gpu(&name, memory_mb);

        let runtime_index = if !visible_devices.is_empty() {
            visible_devices.iter().position(|v| *v == idx).unwrap_or(0)
        } else {
            idx
        };

        gpus.push(GpuInfo {
            physical_index: idx,
            runtime_index,
            name,
            memory_mb,
            score,
        });
    }

    gpus
}

/// Query CUDA memory usage via NVML instead of cudarc.
///
/// This is useful for preflight-style diagnostics where cudarc may not be
/// initialised yet.  Returns `None` if NVML is unavailable or the device
/// index is out of range.
#[cfg(feature = "nvml")]
pub fn cuda_mem_snapshot_nvml(device_index: u32) -> Option<CudaMemSnapshot> {
    let nvml = get_nvml()?;
    let device = nvml.device_by_index(device_index).ok()?;
    let mem = device.memory_info().ok()?;
    let used = mem.total.saturating_sub(mem.free);
    Some(CudaMemSnapshot {
        free_bytes: mem.free,
        total_bytes: mem.total,
        used_bytes: used,
    })
}

// ---------------------------------------------------------------------------
// GPU discovery (nvidia-smi)
// ---------------------------------------------------------------------------

/// Metadata for a single GPU discovered via nvidia-smi.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub physical_index: usize,
    pub runtime_index: usize,
    pub name: String,
    pub memory_mb: u64,
    pub score: i64,
}

/// Query nvidia-smi for available GPUs, filtered by `min_vram_mb` and
/// `visible_devices`.  Pass an empty slice for `visible_devices` to consider
/// all GPUs.
///
/// When the `nvml` feature is enabled, this function first attempts a direct
/// NVML query (~1 ms) and only falls back to spawning the nvidia-smi
/// subprocess (~800 ms) when NVML is unavailable.
pub fn query_nvidia_smi(min_vram_mb: Option<u64>, visible_devices: &[usize]) -> Vec<GpuInfo> {
    // Fast path: try NVML first when available.
    #[cfg(feature = "nvml")]
    {
        let result = query_gpus_nvml(min_vram_mb, visible_devices);
        if !result.is_empty() {
            return result;
        }
    }

    // Slow path: spawn nvidia-smi subprocess.
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"])
        .output();
    let output = match output {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 3 {
            continue;
        }
        let idx = match parts[0].parse::<usize>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if !visible_devices.is_empty() && !visible_devices.contains(&idx) {
            continue;
        }
        let name = parts[1].to_string();
        let memory_mb = parts[2].parse::<u64>().unwrap_or(0);
        if let Some(min) = min_vram_mb {
            if memory_mb < min {
                continue;
            }
        }
        let score = score_gpu(&name, memory_mb);
        let runtime_index = if !visible_devices.is_empty() {
            visible_devices.iter().position(|v| *v == idx).unwrap_or(0)
        } else {
            idx
        };
        gpus.push(GpuInfo {
            physical_index: idx,
            runtime_index,
            name,
            memory_mb,
            score,
        });
    }
    gpus
}

/// Select the best CUDA device index by score, falling back to probing.
/// Pass an empty slice for `visible_devices` to consider all GPUs.
pub fn auto_cuda_index(min_vram_mb: Option<u64>, visible_devices: &[usize]) -> Option<usize> {
    let mut gpus = query_nvidia_smi(min_vram_mb, visible_devices);
    if !gpus.is_empty() {
        gpus.sort_by_key(|g| -(g.score));
        return Some(gpus[0].runtime_index);
    }
    let max_idx = if visible_devices.is_empty() {
        4
    } else {
        visible_devices.len()
    };
    for idx in 0..max_idx {
        if Device::new_cuda(idx).is_ok() {
            return Some(idx);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Device label parsing
// ---------------------------------------------------------------------------

/// Parse a device label like `"cuda:0"`, `"gpu:1"`, `"cpu"` into a CUDA
/// index.  Returns `None` for CPU or unparseable labels.
pub fn parse_cuda_index(label: &str) -> Option<usize> {
    let lower = label.trim().to_lowercase();
    if lower == "cpu" {
        return None;
    }
    if let Some(rest) = lower.strip_prefix("cuda:") {
        return rest.trim().parse::<usize>().ok();
    }
    if let Some(rest) = lower.strip_prefix("gpu:") {
        return rest.trim().parse::<usize>().ok();
    }
    lower.parse::<usize>().ok()
}

/// Normalize a raw device string to a canonical form (`"cpu"`,
/// `"cuda:0"`, etc.).
pub fn normalize_device_label(raw: &str) -> String {
    let lower = raw.trim().to_lowercase();
    if lower == "cpu" {
        return "cpu".to_string();
    }
    if lower == "cuda" || lower == "gpu" {
        return "cuda:0".to_string();
    }
    if let Some(rest) = lower.strip_prefix("cuda:") {
        return format!("cuda:{}", rest.trim());
    }
    if let Some(rest) = lower.strip_prefix("gpu:") {
        return format!("cuda:{}", rest.trim());
    }
    lower
}

// ---------------------------------------------------------------------------
// Parameterized device resolution
// ---------------------------------------------------------------------------

/// Parameters for [`resolve_device_with_index`].
///
/// Both the runtime and training crates populate this from their own
/// config structs and then delegate to the shared resolution logic.
pub struct DeviceSelectionParams<'a> {
    /// Device label from config (e.g. `"auto"`, `"cpu"`, `"cuda:0"`).
    pub device_label: &'a str,
    /// Explicit CUDA device index override.
    pub gpu_index: Option<usize>,
    /// Force CPU regardless of other settings.
    pub force_cpu: bool,
    /// Minimum VRAM (MiB) required to accept a GPU.
    pub min_vram_mb: Option<u64>,
    /// CUDA device indices to consider (empty = all).
    pub cuda_visible_devices: &'a [usize],
}

/// Resolve a candle [`Device`] and its optional CUDA index from explicit
/// configuration parameters.
///
/// Resolution order:
/// 1. `force_cpu` → CPU
/// 2. Explicit `device_label` (non-empty, not `"auto"`) → parse as `cpu` or
///    `cuda:N`
/// 3. Explicit `gpu_index` → try `Device::new_cuda`
/// 4. [`auto_cuda_index`] (nvidia-smi + probe fallback) → best available GPU
/// 5. Fallback → CPU
pub fn resolve_device_with_index(params: &DeviceSelectionParams) -> (Device, Option<usize>) {
    if params.force_cpu {
        return (Device::Cpu, None);
    }

    let label = params.device_label.trim();
    if !label.is_empty() && !label.eq_ignore_ascii_case("auto") {
        let normalized = normalize_device_label(label);
        if normalized == "cpu" {
            return (Device::Cpu, None);
        }
        if let Some(idx) = parse_cuda_index(&normalized) {
            if let Ok(dev) = Device::new_cuda(idx) {
                return (dev, Some(idx));
            }
        }
    }

    if let Some(gpu) = params.gpu_index {
        if let Ok(dev) = Device::new_cuda(gpu) {
            return (dev, Some(gpu));
        }
    }

    if let Some(idx) = auto_cuda_index(params.min_vram_mb, params.cuda_visible_devices) {
        if let Ok(dev) = Device::new_cuda(idx) {
            return (dev, Some(idx));
        }
    }

    (Device::Cpu, None)
}

/// Return the preferred dtype for a device: BF16 on CUDA, F32 on CPU.
pub fn default_dtype(device: &Device) -> DType {
    if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    }
}

/// Parse a GGML dtype string into a [`GgmlDType`].
///
/// Accepts common aliases such as `"q4"`, `"q4_0"`, `"q4-0"`, `"bf16"`, etc.
/// Returns `None` for unrecognised or empty input.
pub fn parse_ggml_dtype(value: Option<&str>) -> Option<GgmlDType> {
    let raw = value?.trim().to_ascii_lowercase();
    match raw.as_str() {
        "f32" => Some(GgmlDType::F32),
        "f16" => Some(GgmlDType::F16),
        "bf16" => Some(GgmlDType::BF16),
        "q4" | "q4_0" | "q4-0" => Some(GgmlDType::Q4_0),
        "q4_1" | "q4-1" => Some(GgmlDType::Q4_1),
        "q5_0" | "q5-0" => Some(GgmlDType::Q5_0),
        "q5_1" | "q5-1" => Some(GgmlDType::Q5_1),
        "q8_0" | "q8-0" => Some(GgmlDType::Q8_0),
        "q8_1" | "q8-1" => Some(GgmlDType::Q8_1),
        "q2k" | "q2_k" | "q2-k" => Some(GgmlDType::Q2K),
        "q3k" | "q3_k" | "q3-k" => Some(GgmlDType::Q3K),
        "q4k" | "q4_k" | "q4-k" => Some(GgmlDType::Q4K),
        "q5k" | "q5_k" | "q5-k" => Some(GgmlDType::Q5K),
        "q6k" | "q6_k" | "q6-k" => Some(GgmlDType::Q6K),
        "q8k" | "q8_k" | "q8-k" => Some(GgmlDType::Q8K),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// CUDA memory management
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct CudaMemSnapshot {
    pub free_bytes: u64,
    pub total_bytes: u64,
    pub used_bytes: u64,
}

impl CudaMemSnapshot {
    pub fn free_mb(self) -> u64 {
        self.free_bytes / 1024 / 1024
    }

    pub fn total_mb(self) -> u64 {
        self.total_bytes / 1024 / 1024
    }

    pub fn used_mb(self) -> u64 {
        self.used_bytes / 1024 / 1024
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CudaMemPoolConfig {
    pub enable: bool,
    pub release_threshold_mb: Option<u64>,
    pub reuse_follow_event_dependencies: bool,
    pub reuse_allow_opportunistic: bool,
    pub reuse_allow_internal_dependencies: bool,
    pub trim_to_mb: Option<u64>,
}

impl Default for CudaMemPoolConfig {
    fn default() -> Self {
        Self {
            enable: false,
            release_threshold_mb: None,
            reuse_follow_event_dependencies: true,
            reuse_allow_opportunistic: true,
            reuse_allow_internal_dependencies: true,
            trim_to_mb: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CudaMemPoolStatus {
    pub release_threshold_bytes: Option<u64>,
    pub reserved_current_bytes: Option<u64>,
    pub used_current_bytes: Option<u64>,
}

/// Print a CUDA memory snapshot as a JSON log line.
///
/// Callers should gate this behind their own config check (e.g.
/// `runtime_config().router_cuda_mem_snapshot` or
/// `train_config().cuda_mem_snapshot`).  This function always prints when
/// a snapshot is available.
pub fn log_cuda_snapshot(tag: &str, device_index: Option<usize>) {
    let snapshot = cuda_mem_snapshot(device_index);
    if let Some(snapshot) = snapshot {
        println!(
            "{{\"event\":\"cuda_mem_snapshot\",\"tag\":\"{}\",\"device\":{},\"free_mb\":{},\"used_mb\":{},\"total_mb\":{}}}",
            tag,
            device_index.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string()),
            snapshot.free_mb(),
            snapshot.used_mb(),
            snapshot.total_mb()
        );
    }
}

/// Configure CUDA memory pool and log the result as a JSON line.
///
/// Combines [`configure_cuda_mem_pool`] with standardised JSON logging.
/// Callers should gate this behind their own config check and pass the
/// fully-populated [`CudaMemPoolConfig`].
pub fn configure_and_log_cuda_mem_pool(device_index: usize, cfg: CudaMemPoolConfig) {
    match configure_cuda_mem_pool(device_index, cfg) {
        Ok(Some(status)) => {
            println!(
                "{{\"event\":\"cuda_mem_pool_configured\",\"device\":{},\"release_threshold_mb\":{},\"reserved_mb\":{},\"used_mb\":{}}}",
                device_index,
                status
                    .release_threshold_bytes
                    .map(|v| v / 1024 / 1024)
                    .unwrap_or(0),
                status
                    .reserved_current_bytes
                    .map(|v| v / 1024 / 1024)
                    .unwrap_or(0),
                status
                    .used_current_bytes
                    .map(|v| v / 1024 / 1024)
                    .unwrap_or(0)
            );
        }
        Ok(None) => println!(
            "{{\"event\":\"cuda_mem_pool_unavailable\",\"device\":{}}}",
            device_index
        ),
        Err(err) => println!(
            "{{\"event\":\"cuda_mem_pool_error\",\"device\":{},\"error\":\"{}\"}}",
            device_index, err
        ),
    }
}

pub fn cuda_mem_snapshot(device_index: Option<usize>) -> Option<CudaMemSnapshot> {
    use cudarc::runtime::{result, sys};
    if let Some(idx) = device_index {
        unsafe {
            if sys::cudaSetDevice(idx as i32).result().is_err() {
                return None;
            }
        }
    }
    let (free, total) = result::get_mem_info().ok()?;
    let used = total.saturating_sub(free);
    Some(CudaMemSnapshot {
        free_bytes: free as u64,
        total_bytes: total as u64,
        used_bytes: used as u64,
    })
}

pub fn configure_cuda_mem_pool(device_index: usize, cfg: CudaMemPoolConfig) -> Result<Option<CudaMemPoolStatus>> {
    use cudarc::runtime::sys;

    if !cfg.enable {
        return Ok(None);
    }

    unsafe {
        sys::cudaSetDevice(device_index as i32)
            .result()
            .context("cudaSetDevice failed")?;
    }

    let mut pools_supported: i32 = 0;
    unsafe {
        sys::cudaDeviceGetAttribute(
            &mut pools_supported as *mut i32,
            sys::cudaDeviceAttr::cudaDevAttrMemoryPoolsSupported,
            device_index as i32,
        )
        .result()
        .context("cudaDeviceGetAttribute(cudaDevAttrMemoryPoolsSupported) failed")?;
    }
    if pools_supported == 0 {
        return Ok(None);
    }

    let mut pool: sys::cudaMemPool_t = std::ptr::null_mut();
    unsafe {
        sys::cudaDeviceGetDefaultMemPool(&mut pool as *mut _, device_index as i32)
            .result()
            .context("cudaDeviceGetDefaultMemPool failed")?;
    }

    unsafe {
        let follow = if cfg.reuse_follow_event_dependencies {
            1u32
        } else {
            0u32
        };
        sys::cudaMemPoolSetAttribute(
            pool,
            sys::cudaMemPoolAttr::cudaMemPoolReuseFollowEventDependencies,
            &follow as *const _ as *mut c_void,
        )
        .result()
        .context("cudaMemPoolSetAttribute(reuse_follow_event_dependencies) failed")?;

        let opportunistic = if cfg.reuse_allow_opportunistic { 1u32 } else { 0u32 };
        sys::cudaMemPoolSetAttribute(
            pool,
            sys::cudaMemPoolAttr::cudaMemPoolReuseAllowOpportunistic,
            &opportunistic as *const _ as *mut c_void,
        )
        .result()
        .context("cudaMemPoolSetAttribute(reuse_allow_opportunistic) failed")?;

        let internal = if cfg.reuse_allow_internal_dependencies {
            1u32
        } else {
            0u32
        };
        sys::cudaMemPoolSetAttribute(
            pool,
            sys::cudaMemPoolAttr::cudaMemPoolReuseAllowInternalDependencies,
            &internal as *const _ as *mut c_void,
        )
        .result()
        .context("cudaMemPoolSetAttribute(reuse_allow_internal_dependencies) failed")?;
    }

    if let Some(threshold_mb) = cfg.release_threshold_mb {
        let threshold_bytes = threshold_mb.saturating_mul(1024 * 1024);
        unsafe {
            sys::cudaMemPoolSetAttribute(
                pool,
                sys::cudaMemPoolAttr::cudaMemPoolAttrReleaseThreshold,
                &threshold_bytes as *const _ as *mut c_void,
            )
            .result()
            .context("cudaMemPoolSetAttribute(release_threshold) failed")?;
        }
    }

    if let Some(trim_mb) = cfg.trim_to_mb {
        let trim_bytes = trim_mb.saturating_mul(1024 * 1024);
        let trim_bytes = usize::try_from(trim_bytes).unwrap_or(usize::MAX);
        unsafe {
            sys::cudaMemPoolTrimTo(pool, trim_bytes)
                .result()
                .context("cudaMemPoolTrimTo failed")?;
        }
    }

    let mut release_threshold: u64 = 0;
    let mut reserved_current: u64 = 0;
    let mut used_current: u64 = 0;
    unsafe {
        let _ = sys::cudaMemPoolGetAttribute(
            pool,
            sys::cudaMemPoolAttr::cudaMemPoolAttrReleaseThreshold,
            &mut release_threshold as *mut _ as *mut c_void,
        );
        let _ = sys::cudaMemPoolGetAttribute(
            pool,
            sys::cudaMemPoolAttr::cudaMemPoolAttrReservedMemCurrent,
            &mut reserved_current as *mut _ as *mut c_void,
        );
        let _ = sys::cudaMemPoolGetAttribute(
            pool,
            sys::cudaMemPoolAttr::cudaMemPoolAttrUsedMemCurrent,
            &mut used_current as *mut _ as *mut c_void,
        );
    }

    Ok(Some(CudaMemPoolStatus {
        release_threshold_bytes: Some(release_threshold),
        reserved_current_bytes: Some(reserved_current),
        used_current_bytes: Some(used_current),
    }))
}
