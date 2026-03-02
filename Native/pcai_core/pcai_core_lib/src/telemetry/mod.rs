pub mod device_codes;
pub mod disk_health;
pub mod event_log;
pub mod network;
pub mod pnp;
pub mod process;
pub mod usb;

use std::sync::Mutex;
use std::sync::OnceLock;
use sysinfo::System;

static SYS: OnceLock<Mutex<System>> = OnceLock::new();

fn get_system() -> &'static Mutex<System> {
    SYS.get_or_init(|| Mutex::new(System::new_all()))
}

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
pub fn collect_telemetry() -> SystemTelemetry {
    let mut sys = match get_system().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
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

        if name.ends_with("pwsh.exe") || name.ends_with("powershell.exe") || name == "pwsh" || name == "powershell" {
            ps_processes.push(ProcessMetrics {
                pid: pid.as_u32(),
                name: name.into_owned(),
                cpu_usage: process.cpu_usage(),
                memory_mb: process.memory() / 1024 / 1024,
            });
            ps_server_healthy = true;
        }
    }

    SystemTelemetry {
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
    }
}

pub fn check_resource_safety(gpu_limit: f32) -> bool {
    let telemetry = collect_telemetry();
    if let Some(gpu) = telemetry.gpu_utilization {
        if gpu > (gpu_limit * 100.0) {
            return false;
        }
    }
    if telemetry.cpu_usage > 90.0 {
        return false;
    }
    true
}
