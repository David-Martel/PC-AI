pub mod device_codes;
#[cfg(windows)]
pub mod disk_health;
#[cfg(windows)]
pub mod event_log;
#[cfg(windows)]
pub mod network;
#[cfg(windows)]
pub mod pnp;
#[cfg(windows)]
pub mod process;
#[cfg(windows)]
pub mod usb;

use std::sync::{Mutex, PoisonError};
use sysinfo::System;

use crate::PcaiStatus;

/// Shared sampler, created lazily inside the mutex rather than in a `OnceLock`, so a
/// failed `System::new_all` (fallible since sysinfo 0.40) is returned to the caller
/// and retried on the next call instead of being cached forever.
static SYS: Mutex<Option<System>> = Mutex::new(None);

#[derive(Debug, serde::Serialize)]
pub struct ProcessMetrics {
    pub pid: u32,
    pub name: String,
    pub cpu_usage: f32,
    pub memory_mb: u64,
}

#[derive(Debug, serde::Serialize)]
pub struct SystemTelemetry {
    pub cpu_usage: f32,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
    pub gpu_utilization: Option<f32>,
    pub ps_server_healthy: bool,
    pub ps_processes: Vec<ProcessMetrics>,
    pub timestamp: u64,
}

/// Collects high-fidelity system telemetry.
///
/// # Errors
///
/// Returns the [`PcaiStatus`] mapped from the sysinfo error when the shared
/// `System` cannot be created.
pub fn collect_telemetry() -> Result<SystemTelemetry, PcaiStatus> {
    let mut slot = SYS.lock().unwrap_or_else(PoisonError::into_inner);
    let sys = match slot.as_mut() {
        Some(sys) => sys,
        None => slot.insert(System::new_all()?),
    };

    sys.refresh_all();

    let cpu_usage = sys.global_cpu_usage();
    let memory_used = sys.used_memory() / 1024 / 1024;
    let memory_total = sys.total_memory() / 1024 / 1024;

    let mut ps_processes = Vec::new();
    let mut ps_server_healthy = false;

    for (pid, process) in sys.processes() {
        let name_os = process.name();
        let name = name_os.to_string_lossy();

        let is_ps = if cfg!(windows) {
            name.ends_with("pwsh.exe") || name.ends_with("powershell.exe")
        } else {
            name == "pwsh" || name == "powershell"
        };

        if is_ps {
            ps_processes.push(ProcessMetrics {
                pid: pid.as_u32(),
                name: name.into_owned(),
                cpu_usage: process.cpu_usage(),
                memory_mb: process.memory() / 1024 / 1024,
            });
            ps_server_healthy = true;
        }
    }

    Ok(SystemTelemetry {
        cpu_usage,
        memory_used_mb: memory_used,
        memory_total_mb: memory_total,
        gpu_utilization: None,
        ps_server_healthy,
        ps_processes,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    })
}

/// Reports whether resource use is within safe limits.
///
/// # Errors
///
/// Returns the [`PcaiStatus`] when telemetry cannot be collected, so "unknown" is
/// not mistaken for "safe".
pub fn check_resource_safety(gpu_limit: f32) -> Result<bool, PcaiStatus> {
    let telemetry = collect_telemetry()?;
    if let Some(gpu) = telemetry.gpu_utilization {
        if gpu > (gpu_limit * 100.0) {
            return Ok(false);
        }
    }
    // A NaN reading is not over the limit, as before: only a CPU above 90 % is unsafe.
    Ok(telemetry.cpu_usage <= 90.0 || telemetry.cpu_usage.is_nan())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sysinfo 0.40 port made the shared `System` fallible and moved it behind a
    /// lazily initialised mutex; two consecutive calls prove both the first-call
    /// initialisation and the reuse path return real figures.
    #[test]
    fn collect_telemetry_initialises_then_reuses_the_shared_system() {
        let first = collect_telemetry().expect("telemetry collects on a supported host");
        let second = collect_telemetry().expect("shared system is reused");
        assert!(first.memory_total_mb > 0);
        assert_eq!(first.memory_total_mb, second.memory_total_mb);
    }
}
