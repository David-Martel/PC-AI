//! GPU monitoring via NVIDIA Management Library (NVML).
//!
//! Provides direct hardware access to GPU metrics without spawning nvidia-smi
//! subprocesses.  Falls back gracefully when NVML is unavailable.
//!
//! # Feature gate
//!
//! This module is compiled only when the `nvml` Cargo feature is enabled.
//! Add `--features nvml` to any `cargo` invocation that needs GPU metrics.
//!
//! # Examples
//!
//! ```no_run
//! # #[cfg(feature = "nvml")]
//! # {
//! use pcai_core_lib::gpu;
//!
//! // List every NVIDIA GPU present in the system.
//! let gpus = gpu::gpu_inventory().unwrap_or_default();
//! for g in &gpus {
//!     println!("{}: {} MB free", g.name,
//!              g.memory_total_mb.saturating_sub(g.memory_used_mb));
//! }
//!
//! // Pick the GPU with the most free VRAM for a cuda:auto placement.
//! if let Ok(Some(best)) = gpu::best_available_gpu() {
//!     println!("Using GPU {} (index {})", best.name, best.index);
//! }
//! # }
//! ```

use std::sync::OnceLock;

use anyhow::{Context, Result};
use log::warn;
use serde::Serialize;

// ── Public data types ─────────────────────────────────────────────────────────

/// Static properties of a single NVIDIA GPU.
///
/// Returned by [`gpu_inventory`] and [`best_available_gpu`].
#[derive(Debug, Clone, Serialize)]
pub struct GpuInfo {
    /// Zero-based device index as enumerated by NVML.
    pub index: u32,
    /// Globally-unique device identifier string (e.g. `GPU-xxxxxxxx-...`).
    pub uuid: String,
    /// Human-readable product name (e.g. `NVIDIA GeForce RTX 5060 Ti`).
    pub name: String,
    /// Installed display driver version string (e.g. `561.09`).
    pub driver_version: String,
    /// Compute capability formatted as `"major.minor"` (e.g. `"8.9"`).
    pub compute_capability: String,
    /// Total installed framebuffer memory in mebibytes.
    pub memory_total_mb: u64,
    /// Currently allocated framebuffer memory in mebibytes.
    pub memory_used_mb: u64,
    /// GPU die temperature in degrees Celsius.
    pub temperature_c: u32,
    /// Instantaneous board power draw in watts.
    pub power_draw_watts: f64,
    /// Fan speed as a percentage of maximum (0–100). Returns 0 when fans are
    /// not present or not query-able.
    pub fan_speed_pct: u32,
    /// PCIe link generation currently negotiated (e.g. `4` for PCIe 4.0).
    pub pcie_gen: u32,
    /// PCIe link width currently negotiated (number of lanes, e.g. `16`).
    pub pcie_width: u32,
}

/// Real-time utilization snapshot for a single NVIDIA GPU.
///
/// Returned by [`gpu_utilization`].
#[derive(Debug, Clone, Serialize)]
pub struct GpuUtilization {
    /// Fraction of time the GPU compute engine was active (0–100 %).
    pub gpu_util_pct: u32,
    /// Fraction of time the GPU memory interface was active (0–100 %).
    pub mem_util_pct: u32,
    /// Encoder engine utilization (0–100 %).
    pub encoder_util_pct: u32,
    /// Decoder engine utilization (0–100 %).
    pub decoder_util_pct: u32,
    /// Framebuffer memory currently in use, in mebibytes.
    pub memory_used_mb: u64,
    /// Framebuffer memory currently free, in mebibytes.
    pub memory_free_mb: u64,
    /// GPU die temperature in degrees Celsius.
    pub temperature_c: u32,
    /// Instantaneous board power draw in watts.
    pub power_draw_watts: f64,
    /// Current SM (shader multiprocessor) clock in MHz.
    pub clock_sm_mhz: u32,
    /// Current memory clock in MHz.
    pub clock_mem_mhz: u32,
}

// ── NVML singleton ────────────────────────────────────────────────────────────

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
static NVML: OnceLock<Option<nvml_wrapper::Nvml>> = OnceLock::new();

/// Return a reference to the process-wide NVML singleton, initialising it on
/// the first call.
///
/// Returns `None` (after logging a warning) if NVML is unavailable — e.g. no
/// NVIDIA driver is installed or the library is not present in the container.
fn get_nvml() -> Option<&'static nvml_wrapper::Nvml> {
    NVML.get_or_init(|| match nvml_wrapper::Nvml::init() {
        Ok(nvml) => Some(nvml),
        Err(err) => {
            warn!(
                "NVML initialisation failed (no NVIDIA driver or NVML not \
                     available): {err}"
            );
            None
        }
    })
    .as_ref()
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Return the number of NVIDIA GPUs detected by NVML.
///
/// Returns `Ok(0)` when NVML is unavailable.  This is a cheap call that only
/// asks NVML for the device count; it does not query per-device properties.
///
/// # Errors
///
/// Returns an error when NVML initialises successfully but the device-count
/// query itself fails.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "nvml")]
/// let n = pcai_core_lib::gpu::gpu_count().unwrap_or(0);
/// println!("{n} NVIDIA GPU(s) present");
/// ```
pub fn gpu_count() -> Result<u32> {
    let nvml = match get_nvml() {
        Some(n) => n,
        None => return Ok(0),
    };

    nvml.device_count().context("Failed to query NVML device count")
}

/// Return inventory information for every NVIDIA GPU in the system.
///
/// Initialises NVML on the first call.  If NVML is not available (no NVIDIA
/// driver installed, or running inside a container without NVML passthrough)
/// the function logs a warning and returns an empty [`Vec`] instead of an
/// error, allowing callers to degrade gracefully.
///
/// # Errors
///
/// Returns an error only when NVML initialises successfully but a subsequent
/// per-device query fails unexpectedly.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "nvml")]
/// let inventory = pcai_core_lib::gpu::gpu_inventory().unwrap_or_default();
/// ```
pub fn gpu_inventory() -> Result<Vec<GpuInfo>> {
    let nvml = match get_nvml() {
        Some(n) => n,
        None => return Ok(Vec::new()),
    };

    let driver = nvml.sys_driver_version().unwrap_or_else(|_| String::from("unknown"));

    let count = nvml.device_count().context("Failed to query NVML device count")?;

    let mut gpus = Vec::with_capacity(count as usize);

    for idx in 0..count {
        match nvml.device_by_index(idx) {
            Ok(device) => match query_gpu_info(&device, idx, &driver) {
                Ok(info) => gpus.push(info),
                Err(err) => warn!("Failed to query info for GPU at index {idx}: {err}"),
            },
            Err(err) => warn!("Failed to open NVML device at index {idx}: {err}"),
        }
    }

    Ok(gpus)
}

/// Return real-time utilization metrics for the GPU at `device_index`.
///
/// # Errors
///
/// Returns an error when NVML is unavailable or when the specified device
/// index is out of range.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "nvml")]
/// let util = pcai_core_lib::gpu::gpu_utilization(0)?;
/// println!("GPU load: {}%", util.gpu_util_pct);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn gpu_utilization(device_index: u32) -> Result<GpuUtilization> {
    let nvml = get_nvml().context("NVML unavailable — no NVIDIA driver detected")?;

    let device = nvml
        .device_by_index(device_index)
        .with_context(|| format!("Failed to open NVML device at index {device_index}"))?;

    query_gpu_utilization(&device)
}

/// Return the GPU with the most free framebuffer memory.
///
/// Returns `Ok(None)` when no NVIDIA GPUs are present.  This is intended for
/// `cuda:auto` placement logic that needs to pick the least-loaded device.
///
/// # Errors
///
/// Returns an error only when NVML initialises but a device query fails.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "nvml")]
/// if let Ok(Some(gpu)) = pcai_core_lib::gpu::best_available_gpu() {
///     println!("Best GPU: {} (index {})", gpu.name, gpu.index);
/// }
/// ```
pub fn best_available_gpu() -> Result<Option<GpuInfo>> {
    let gpus = gpu_inventory()?;

    let best = gpus
        .into_iter()
        .max_by_key(|g| g.memory_total_mb.saturating_sub(g.memory_used_mb));

    Ok(best)
}

/// Return the installed NVIDIA display driver version string via NVML.
///
/// This replaces the subprocess invocation
/// `nvidia-smi --query-gpu=driver_version --format=csv,noheader`.
///
/// # Errors
///
/// Returns an error when NVML is unavailable or the driver version cannot be
/// queried.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "nvml")]
/// let ver = pcai_core_lib::gpu::driver_version()?;
/// println!("Driver: {ver}");
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn driver_version() -> Result<String> {
    let nvml = get_nvml().context("NVML unavailable — no NVIDIA driver detected")?;

    nvml.sys_driver_version().context("Failed to query NVML driver version")
}

/// Return the CUDA driver version as `(major, minor)`.
///
/// The CUDA driver version reported by NVML reflects the maximum CUDA version
/// supported by the installed display driver, not the CUDA toolkit version.
///
/// # Errors
///
/// Returns an error when NVML is unavailable or the version cannot be queried.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "nvml")]
/// let (major, minor) = pcai_core_lib::gpu::cuda_driver_version()?;
/// println!("CUDA driver: {major}.{minor}");
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn cuda_driver_version() -> Result<(i32, i32)> {
    let nvml = get_nvml().context("NVML unavailable — no NVIDIA driver detected")?;

    let version = nvml
        .sys_cuda_driver_version()
        .context("Failed to query NVML CUDA driver version")?;

    // nvml-wrapper encodes the version as a single integer:
    // version = major * 1000 + minor * 10
    let major = version / 1000;
    let minor = (version % 1000) / 10;
    Ok((major, minor))
}

/// Query static properties from an already-opened NVML `Device`.
fn query_gpu_info(device: &nvml_wrapper::Device<'_>, index: u32, driver_version: &str) -> Result<GpuInfo> {
    use nvml_wrapper::enum_wrappers::device::TemperatureSensor;

    let uuid = device.uuid().unwrap_or_else(|_| format!("GPU-{index}"));

    let name = device.name().unwrap_or_else(|_| format!("Unknown GPU {index}"));

    let (cc_major, cc_minor) = device
        .cuda_compute_capability()
        .map(|cc| (cc.major, cc.minor))
        .unwrap_or((0, 0));

    let memory_info = device.memory_info().context("Failed to query NVML memory info")?;

    let temperature_c = device.temperature(TemperatureSensor::Gpu).unwrap_or(0);

    // Power is in milliwatts from NVML; convert to watts.
    let power_draw_watts = device.power_usage().map(|mw| f64::from(mw) / 1000.0).unwrap_or(0.0);

    let fan_speed_pct = device.fan_speed(0).unwrap_or(0);

    let pcie_gen = device.current_pcie_link_gen().unwrap_or(0);

    let pcie_width = device.current_pcie_link_width().unwrap_or(0);

    Ok(GpuInfo {
        index,
        uuid,
        name,
        driver_version: driver_version.to_owned(),
        compute_capability: format!("{cc_major}.{cc_minor}"),
        memory_total_mb: memory_info.total / (1024 * 1024),
        memory_used_mb: memory_info.used / (1024 * 1024),
        temperature_c,
        power_draw_watts,
        fan_speed_pct,
        pcie_gen,
        pcie_width,
    })
}

/// Query real-time utilisation from an already-opened NVML `Device`.
fn query_gpu_utilization(device: &nvml_wrapper::Device<'_>) -> Result<GpuUtilization> {
    use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};

    let utilization = device
        .utilization_rates()
        .context("Failed to query NVML utilization rates")?;

    let (encoder_util, _) = device
        .encoder_utilization()
        .map(|e| (e.utilization, e.sampling_period))
        .unwrap_or((0, 0));

    let (decoder_util, _) = device
        .decoder_utilization()
        .map(|d| (d.utilization, d.sampling_period))
        .unwrap_or((0, 0));

    let memory_info = device.memory_info().context("Failed to query NVML memory info")?;

    let temperature_c = device.temperature(TemperatureSensor::Gpu).unwrap_or(0);

    let power_draw_watts = device.power_usage().map(|mw| f64::from(mw) / 1000.0).unwrap_or(0.0);

    let clock_sm_mhz = device
        .clock(Clock::SM, nvml_wrapper::enum_wrappers::device::ClockId::Current)
        .unwrap_or(0);

    let clock_mem_mhz = device
        .clock(Clock::Memory, nvml_wrapper::enum_wrappers::device::ClockId::Current)
        .unwrap_or(0);

    Ok(GpuUtilization {
        gpu_util_pct: utilization.gpu,
        mem_util_pct: utilization.memory,
        encoder_util_pct: encoder_util,
        decoder_util_pct: decoder_util,
        memory_used_mb: memory_info.used / (1024 * 1024),
        memory_free_mb: memory_info.free / (1024 * 1024),
        temperature_c,
        power_draw_watts,
        clock_sm_mhz,
        clock_mem_mhz,
    })
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// GpuInfo must serialise to valid JSON with all expected fields.
    #[test]
    fn gpu_info_serializes_to_json() {
        let info = GpuInfo {
            index: 0,
            uuid: "GPU-test-uuid".to_owned(),
            name: "Test GPU".to_owned(),
            driver_version: "560.00".to_owned(),
            compute_capability: "8.9".to_owned(),
            memory_total_mb: 8192,
            memory_used_mb: 1024,
            temperature_c: 65,
            power_draw_watts: 75.5,
            fan_speed_pct: 40,
            pcie_gen: 4,
            pcie_width: 16,
        };

        let json = serde_json::to_string(&info).expect("GpuInfo should serialise");
        assert!(json.contains("\"index\":0"));
        assert!(json.contains("\"uuid\":\"GPU-test-uuid\""));
        assert!(json.contains("\"name\":\"Test GPU\""));
        assert!(json.contains("\"compute_capability\":\"8.9\""));
        assert!(json.contains("\"memory_total_mb\":8192"));
        assert!(json.contains("\"pcie_gen\":4"));
        assert!(json.contains("\"pcie_width\":16"));
    }

    /// GpuUtilization must serialise to valid JSON with all expected fields.
    #[test]
    fn gpu_utilization_serializes_to_json() {
        let util = GpuUtilization {
            gpu_util_pct: 95,
            mem_util_pct: 80,
            encoder_util_pct: 0,
            decoder_util_pct: 0,
            memory_used_mb: 6000,
            memory_free_mb: 2192,
            temperature_c: 82,
            power_draw_watts: 150.0,
            clock_sm_mhz: 2400,
            clock_mem_mhz: 9001,
        };

        let json = serde_json::to_string(&util).expect("GpuUtilization should serialise");
        assert!(json.contains("\"gpu_util_pct\":95"));
        assert!(json.contains("\"encoder_util_pct\":0"));
        assert!(json.contains("\"clock_sm_mhz\":2400"));
        assert!(json.contains("\"clock_mem_mhz\":9001"));
    }

    /// Vec<GpuInfo> serialises to a JSON array, matching the FFI contract.
    #[test]
    fn gpu_info_vec_serializes_to_json_array() {
        let inventory: Vec<GpuInfo> = Vec::new();
        let json = serde_json::to_string(&inventory).expect("empty Vec should serialise");
        assert_eq!(json, "[]");
    }

    /// gpu_inventory returns Ok (possibly empty) even when NVML is not present.
    ///
    /// This test always passes because `gpu_inventory` is designed to return an
    /// empty Vec rather than an error when NVML fails to initialise.
    #[test]
    fn gpu_inventory_graceful_when_nvml_absent() {
        // This will succeed on machines without NVIDIA drivers (returns empty vec)
        // and succeed on machines with NVIDIA drivers (returns populated vec).
        let result = gpu_inventory();
        assert!(result.is_ok(), "gpu_inventory should never fail on init error");
    }

    /// best_available_gpu returns Ok(None) when no GPUs are present.
    ///
    /// Same graceful-degradation guarantee as gpu_inventory.
    #[test]
    fn best_available_gpu_ok_when_no_gpus() {
        // On a machine without NVIDIA, this returns Ok(None).
        // On a machine with NVIDIA, this returns Ok(Some(...)).
        let result = best_available_gpu();
        assert!(result.is_ok(), "best_available_gpu should never fail on init error");
    }

    /// Compute capability string format is `"major.minor"`.
    #[test]
    fn compute_capability_format() {
        let info = GpuInfo {
            index: 0,
            uuid: String::new(),
            name: String::new(),
            driver_version: String::new(),
            compute_capability: "8.9".to_owned(),
            memory_total_mb: 0,
            memory_used_mb: 0,
            temperature_c: 0,
            power_draw_watts: 0.0,
            fan_speed_pct: 0,
            pcie_gen: 0,
            pcie_width: 0,
        };

        let parts: Vec<&str> = info.compute_capability.split('.').collect();
        assert_eq!(parts.len(), 2, "compute_capability must be 'major.minor'");
        assert!(parts[0].parse::<u32>().is_ok(), "major must be numeric");
        assert!(parts[1].parse::<u32>().is_ok(), "minor must be numeric");
    }
}
