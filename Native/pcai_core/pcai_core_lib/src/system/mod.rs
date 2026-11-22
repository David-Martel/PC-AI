//! PCAI System Module - Native Telemetry and Interrogation
//!
//! Provides optimized access to hardware, OS, and process information.

use serde::Serialize;
use std::sync::{Mutex, PoisonError};
use std::time::Instant;
use sysinfo::{Components, Disks, Networks, System};

use crate::string::PcaiStringBuffer;
use crate::PcaiStatus;

pub mod logs;
pub mod path;

use std::ffi::CStr;
use std::os::raw::c_char;

#[derive(Serialize)]
pub struct SystemSummary {
    os_name: String,
    os_version: String,
    hostname: String,
    cpu_count: usize,
    cpu_brand: String,
    memory_total_gb: f64,
    memory_available_gb: f64,
    disks: Vec<DiskInfo>,
    networks: Vec<NetworkInfo>,
    temperatures: Vec<ComponentTemp>,
}

#[derive(Serialize)]
pub struct DiskInfo {
    pub name: String,
    pub mount_point: String,
    pub total_gb: f64,
    pub available_gb: f64,
    pub is_removable: bool,
}

#[derive(Serialize)]
pub struct NetworkInfo {
    pub interface: String,
    pub received_kb: u64,
    pub transmitted_kb: u64,
}

#[derive(Serialize)]
pub struct ComponentTemp {
    pub label: String,
    pub celsius: Option<f32>,
}

struct CpuSampler {
    system: System,
    refreshed_at: Instant,
    valid_delta: bool,
}

#[derive(Serialize)]
struct CpuReading {
    cpu_usage: Vec<Option<f32>>,
    avg_load: Option<f32>,
    sample_status: &'static str,
    sample_age_milliseconds: u128,
}

impl CpuSampler {
    fn new() -> Result<Self, PcaiStatus> {
        let mut system = System::new()?;
        system.refresh_cpu_all();
        Ok(Self {
            system,
            refreshed_at: Instant::now(),
            valid_delta: false,
        })
    }

    fn sample(&mut self) -> CpuReading {
        if self.refreshed_at.elapsed() >= sysinfo::MINIMUM_CPU_UPDATE_INTERVAL {
            self.system.refresh_cpu_usage();
            self.refreshed_at = Instant::now();
            self.valid_delta = true;
        }
        CpuReading {
            cpu_usage: self
                .system
                .cpus()
                .iter()
                .map(|cpu| self.valid_delta.then(|| cpu.usage()))
                .collect(),
            avg_load: self.valid_delta.then(|| self.system.global_cpu_usage()),
            sample_status: if self.valid_delta { "Sampled" } else { "WarmingUp" },
            sample_age_milliseconds: self.refreshed_at.elapsed().as_millis(),
        }
    }
}

/// Samples CPU usage from the process-wide sampler.
///
/// The sampler is created lazily inside the mutex rather than in a `OnceLock`, so a
/// failed `System::new` is returned to the caller and retried on the next call
/// instead of being cached forever.
fn read_cpu_usage() -> Result<CpuReading, PcaiStatus> {
    static SAMPLER: Mutex<Option<CpuSampler>> = Mutex::new(None);
    read_cpu_usage_from(&SAMPLER, CpuSampler::new)
}

/// [`read_cpu_usage`] over an explicit slot and constructor, so a test can inject a failing
/// sampler constructor without touching the process-wide sampler.
fn read_cpu_usage_from(
    slot: &Mutex<Option<CpuSampler>>,
    make: fn() -> Result<CpuSampler, PcaiStatus>,
) -> Result<CpuReading, PcaiStatus> {
    let mut slot = slot.lock().unwrap_or_else(PoisonError::into_inner);
    let sampler = crate::lazy_slot::get_or_try_init(&mut slot, make)?;
    Ok(sampler.sample())
}

/// Retrieves a comprehensive summary of system status.
///
/// # Errors
///
/// Returns the [`PcaiStatus`] mapped from the sysinfo error when the system, disk,
/// network or component probe cannot be created (all fallible since sysinfo 0.40).
pub fn get_system_summary() -> Result<SystemSummary, PcaiStatus> {
    let mut sys = System::new_all()?;
    sys.refresh_all();

    let disks_list = Disks::new_with_refreshed_list()?;
    let disks = disks_list
        .iter()
        .map(|d| DiskInfo {
            name: d.name().to_string_lossy().into_owned(),
            mount_point: d.mount_point().to_string_lossy().into_owned(),
            total_gb: d.total_space() as f64 / 1_000_000_000.0,
            available_gb: d.available_space() as f64 / 1_000_000_000.0,
            is_removable: d.is_removable(),
        })
        .collect();

    let networks_list = Networks::new_with_refreshed_list()?;
    let networks = networks_list
        .iter()
        .map(|(name, data)| NetworkInfo {
            interface: name.to_string_lossy().into_owned(),
            received_kb: data.received() / 1024,
            transmitted_kb: data.transmitted() / 1024,
        })
        .collect();

    let components_list = Components::new_with_refreshed_list()?;
    let temperatures = components_list
        .iter()
        .map(|c| ComponentTemp {
            label: c.label().to_string(),
            celsius: c.temperature().filter(|value| value.is_finite()),
        })
        .collect();

    Ok(SystemSummary {
        os_name: System::name().unwrap_or_default(),
        os_version: System::os_version().unwrap_or_default(),
        hostname: System::host_name().unwrap_or_default(),
        cpu_count: sys.cpus().len(),
        cpu_brand: sys.cpus().first().map(|c| c.brand().to_string()).unwrap_or_default(),
        memory_total_gb: sys.total_memory() as f64 / 1_000_000_000.0,
        memory_available_gb: sys.available_memory() as f64 / 1_000_000_000.0,
        disks,
        networks,
        temperatures,
    })
}

// ============================================================================
// FFI Implementation
// ============================================================================

/// Returns a JSON report of the system status.
#[no_mangle]
pub extern "C" fn pcai_query_system_info() -> PcaiStringBuffer {
    match get_system_summary() {
        Ok(summary) => crate::string::json_to_buffer(&summary),
        Err(status) => PcaiStringBuffer::error(status),
    }
}

/// Specialized query for hardware metrics (temps, CPU usage).
#[no_mangle]
pub extern "C" fn pcai_query_hardware_metrics() -> PcaiStringBuffer {
    match collect_hardware_metrics() {
        Ok(buffer) => buffer,
        Err(status) => PcaiStringBuffer::error(status),
    }
}

fn collect_hardware_metrics() -> Result<PcaiStringBuffer, PcaiStatus> {
    let cpu = read_cpu_usage()?;

    #[derive(Serialize)]
    struct Metrics {
        cpu_usage: Vec<Option<f32>>,
        avg_load: Option<f32>,
        cpu_sample_status: &'static str,
        cpu_sample_age_milliseconds: u128,
        temps: Vec<ComponentTemp>,
    }

    let components_list = Components::new_with_refreshed_list()?;
    let temps = components_list
        .iter()
        .map(|c| ComponentTemp {
            label: c.label().to_string(),
            celsius: c.temperature().filter(|value| value.is_finite()),
        })
        .collect();

    Ok(crate::string::json_to_buffer(&Metrics {
        cpu_usage: cpu.cpu_usage,
        avg_load: cpu.avg_load,
        cpu_sample_status: cpu.sample_status,
        cpu_sample_age_milliseconds: cpu.sample_age_milliseconds,
        temps,
    }))
}

// ============================================================================
// FFI Entry Points (Migrated from pcai_system)
// ============================================================================

/// Analyze the PATH environment variable for issues
#[no_mangle]
pub extern "C" fn pcai_analyze_path() -> path::PathAnalysisStats {
    let (stats, _) = path::analyze_path();
    stats
}

/// Analyze PATH and return detailed JSON report
#[no_mangle]
pub extern "C" fn pcai_analyze_path_json() -> PcaiStringBuffer {
    let (_, json) = path::analyze_path();

    match serde_json::to_string_pretty(&json) {
        Ok(s) => PcaiStringBuffer::from_string(&s),
        Err(e) => {
            let error_json = format!(r#"{{"status":"Error","error":"{}"}}"#, e);
            PcaiStringBuffer::from_string(&error_json)
        }
    }
}

/// Search log files for a pattern
#[no_mangle]
pub extern "C" fn pcai_search_logs(
    root_path: *const c_char,
    pattern: *const c_char,
    file_pattern: *const c_char,
    case_sensitive: bool,
    context_lines: u32,
    max_matches: u32,
) -> logs::LogSearchStats {
    // Validate pointers
    if root_path.is_null() || pattern.is_null() {
        return logs::LogSearchStats::error(PcaiStatus::NullPointer);
    }

    // Convert C strings to Rust strings
    let root_path_str = unsafe {
        match CStr::from_ptr(root_path).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return logs::LogSearchStats::error(PcaiStatus::InvalidArgument),
        }
    };

    let pattern_str = unsafe {
        match CStr::from_ptr(pattern).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return logs::LogSearchStats::error(PcaiStatus::InvalidArgument),
        }
    };

    let file_pattern_str = if file_pattern.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(file_pattern).to_str() {
                Ok(s) => Some(s.to_string()),
                Err(_) => None,
            }
        }
    };

    let options = logs::LogSearchOptions {
        pattern: pattern_str,
        root_path: root_path_str,
        file_pattern: file_pattern_str,
        case_sensitive,
        context_lines: context_lines as usize,
        max_matches: max_matches as usize,
        max_files: 1000,
    };

    let (stats, _) = logs::search_logs(&options);
    stats
}

/// Search log files and return JSON results
#[no_mangle]
pub extern "C" fn pcai_search_logs_json(
    root_path: *const c_char,
    pattern: *const c_char,
    file_pattern: *const c_char,
    case_sensitive: bool,
    context_lines: u32,
    max_matches: u32,
) -> PcaiStringBuffer {
    // Validate pointers
    if root_path.is_null() || pattern.is_null() {
        let error_json = r#"{"status":"Error","error":"Null pointer provided"}"#;
        return PcaiStringBuffer::from_string(error_json);
    }

    // Convert C strings to Rust strings
    let root_path_str = unsafe {
        match CStr::from_ptr(root_path).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                let error_json = r#"{"status":"Error","error":"Invalid UTF-8 in root_path"}"#;
                return PcaiStringBuffer::from_string(error_json);
            }
        }
    };

    let pattern_str = unsafe {
        match CStr::from_ptr(pattern).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                let error_json = r#"{"status":"Error","error":"Invalid UTF-8 in pattern"}"#;
                return PcaiStringBuffer::from_string(error_json);
            }
        }
    };

    let file_pattern_str = if file_pattern.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(file_pattern).to_str() {
                Ok(s) => Some(s.to_string()),
                Err(_) => None,
            }
        }
    };

    let options = logs::LogSearchOptions {
        pattern: pattern_str,
        root_path: root_path_str,
        file_pattern: file_pattern_str,
        case_sensitive,
        context_lines: context_lines as usize,
        max_matches: max_matches as usize,
        max_files: 1000,
    };

    let (_, json) = logs::search_logs(&options);

    match serde_json::to_string_pretty(&json) {
        Ok(s) => PcaiStringBuffer::from_string(&s),
        Err(e) => {
            let error_json = format!(r#"{{"status":"Error","error":"{}"}}"#, e);
            PcaiStringBuffer::from_string(&error_json)
        }
    }
}

#[cfg(test)]
mod measurement_tests {
    use super::*;

    #[test]
    fn unavailable_temperature_serializes_as_null() {
        let temperature = ComponentTemp {
            label: "Unavailable".into(),
            celsius: None,
        };
        let value = serde_json::to_value(temperature).expect("Temperature serialization");
        assert!(value["celsius"].is_null());
    }

    /// Protects: a failed CPU sampler construction reaches the caller as the mapped
    /// `PcaiStatus`, leaves the slot empty, is retried, and a filled slot is reused.
    /// Detects: caching the failure, swallowing it into a zeroed reading, or rebuilding the
    /// sampler on every call (which would never leave the warm-up state).
    /// Needs: injected constructors and a local slot; the process-wide sampler is untouched.
    #[test]
    fn failed_cpu_sampler_init_propagates_status_then_retries_and_reuses() {
        fn unsupported() -> Result<CpuSampler, PcaiStatus> {
            Err(PcaiStatus::NotImplemented)
        }
        let slot = Mutex::new(None);

        let failed = read_cpu_usage_from(&slot, unsupported);
        assert_eq!(failed.err(), Some(PcaiStatus::NotImplemented));
        assert!(slot.lock().unwrap_or_else(PoisonError::into_inner).is_none());

        let first = read_cpu_usage_from(&slot, CpuSampler::new).expect("healthy constructor succeeds");
        assert_eq!(first.sample_status, "WarmingUp");
        assert!(slot.lock().unwrap_or_else(PoisonError::into_inner).is_some());

        // The failing constructor is never consulted once the slot is filled.
        let second = read_cpu_usage_from(&slot, unsupported).expect("filled slot is reused");
        assert!(second.sample_age_milliseconds >= first.sample_age_milliseconds);
    }

    /// Protects: the summary still builds from the now-fallible system, disk, network and
    /// component constructors.
    /// Detects: a healthy host failing after the sysinfo 0.40 port (live call, not a failure
    /// control; there is no seam for the four sysinfo constructors).
    /// Needs: a supported host with at least one CPU and real memory.
    #[test]
    fn system_summary_builds_on_a_supported_host() {
        let summary = get_system_summary().expect("summary builds on a supported host");
        assert!(summary.cpu_count > 0);
        assert!(summary.memory_total_gb > 0.0);
    }

    /// Protects: CPU usage is `None` until a real sampling interval has elapsed.
    /// Detects: the 0.40 `Cpu::usage` rename wiring the wrong value, or reporting a first
    /// reading as a measurement.
    /// Needs: a supported host and one real `MINIMUM_CPU_UPDATE_INTERVAL` of sleep.
    #[test]
    fn cpu_usage_requires_a_real_sampling_interval() {
        let mut sampler = CpuSampler::new().expect("sampler starts on a supported host");
        let first = sampler.sample();
        assert_eq!(first.sample_status, "WarmingUp");
        assert!(first.avg_load.is_none());
        assert!(first.cpu_usage.iter().all(Option::is_none));
        std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
        let second = sampler.sample();
        assert_eq!(second.sample_status, "Sampled");
        assert!(second.avg_load.is_some_and(|value| (0.0..=100.0).contains(&value)));
        assert!(second.cpu_usage.iter().all(Option::is_some));
    }
}
