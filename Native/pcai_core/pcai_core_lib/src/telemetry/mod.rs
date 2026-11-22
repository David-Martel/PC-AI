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
    collect_telemetry_from(&SYS, System::new_all)
}

/// [`collect_telemetry`] over an explicit slot and constructor, so a test can inject a
/// failing `System::new_all` and observe the retry contract without global state.
fn collect_telemetry_from(
    slot: &Mutex<Option<System>>,
    make: fn() -> Result<System, sysinfo::Error>,
) -> Result<SystemTelemetry, PcaiStatus> {
    let mut slot = slot.lock().unwrap_or_else(PoisonError::into_inner);
    let sys = crate::lazy_slot::get_or_try_init(&mut slot, || Ok(make()?))?;

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

    fn unsupported() -> Result<System, sysinfo::Error> {
        Err(sysinfo::Error::Unsupported)
    }

    /// Protects: a failed `System::new_all` reaches the caller as the mapped `PcaiStatus`,
    /// leaves the shared slot empty, and is retried on the next call; a filled slot is reused.
    /// Detects: caching the failure (the `OnceLock` shape), swallowing the error into a
    /// default `SystemTelemetry`, or rebuilding the `System` on every call.
    /// Needs: no real sysinfo failure; an injected constructor, and a local slot so the
    /// process-wide `SYS` is untouched. The final call injects a failing constructor to prove
    /// the filled slot is reused.
    #[test]
    fn failed_init_propagates_status_keeps_slot_empty_then_retries_and_reuses() {
        let slot = Mutex::new(None);

        let failed = collect_telemetry_from(&slot, unsupported);
        assert_eq!(failed.err(), Some(PcaiStatus::NotImplemented));
        assert!(slot.lock().unwrap_or_else(PoisonError::into_inner).is_none());

        let built = collect_telemetry_from(&slot, System::new_all).expect("healthy constructor succeeds");
        assert!(built.memory_total_mb > 0);
        assert!(slot.lock().unwrap_or_else(PoisonError::into_inner).is_some());

        let reused = collect_telemetry_from(&slot, unsupported).expect("filled slot is reused");
        assert_eq!(reused.memory_total_mb, built.memory_total_mb);
    }

    /// Protects: the public entry point is wired to the shared slot and a real constructor.
    /// Detects: a healthy host failing after the sysinfo 0.40 port (live call, not a failure
    /// control; the injected test above covers propagation and retry).
    /// Needs: a supported host with real memory.
    #[test]
    fn collect_telemetry_reports_real_memory_on_a_supported_host() {
        let telemetry = collect_telemetry().expect("telemetry collects on a supported host");
        assert!(telemetry.memory_total_mb > 0);
    }
}
