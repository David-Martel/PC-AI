//! Process Monitoring
//!
//! Process enumeration and statistics gathering using sysinfo crate.

use crate::PcaiStatus;
use serde::Serialize;
use sysinfo::{Pid, ProcessesToUpdate, System};

/// FFI-safe process statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ProcessStats {
    pub status: PcaiStatus,
    pub total_processes: u32,
    pub total_threads: u32,
    pub system_cpu_usage: f32,
    pub system_memory_used_bytes: u64,
    pub system_memory_total_bytes: u64,
    pub elapsed_ms: u64,
}

impl Default for ProcessStats {
    fn default() -> Self {
        Self {
            status: PcaiStatus::Success,
            total_processes: 0,
            total_threads: 0,
            system_cpu_usage: 0.0,
            system_memory_used_bytes: 0,
            system_memory_total_bytes: 0,
            elapsed_ms: 0,
        }
    }
}

/// Individual process information
#[derive(Debug, Clone, Serialize)]
pub struct ProcessInfo {
    pub pid: u32,
    pub name: String,
    pub cpu_usage: f32,
    pub memory_bytes: u64,
    pub memory_formatted: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exe_path: Option<String>,
}

/// JSON output structure for process list
#[derive(Debug, Serialize)]
pub struct ProcessListJson {
    pub status: String,
    pub total_processes: u32,
    pub total_threads: u32,
    pub system_cpu_usage: f32,
    pub system_memory_used_bytes: u64,
    pub system_memory_total_bytes: u64,
    pub elapsed_ms: u64,
    pub sort_by: String,
    pub processes: Vec<ProcessInfo>,
}

/// Format bytes as human-readable string
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Get system-wide process statistics
pub fn get_process_stats() -> ProcessStats {
    let mut sys = System::new_all();
    sys.refresh_all();

    // Need a second refresh for accurate CPU usage
    std::thread::sleep(std::time::Duration::from_millis(100));
    sys.refresh_all();

    let mut total_threads = 0u32;
    for _process in sys.processes().values() {
        // Count threads (sysinfo doesn't directly expose thread count on all platforms)
        // Using 1 as minimum since every process has at least one thread
        total_threads += 1;
    }

    ProcessStats {
        status: PcaiStatus::Success,
        total_processes: sys.processes().len() as u32,
        total_threads,
        system_cpu_usage: sys.global_cpu_usage(),
        system_memory_used_bytes: sys.used_memory(),
        system_memory_total_bytes: sys.total_memory(),
        elapsed_ms: 0,
    }
}

/// Get top N processes sorted by memory or CPU
pub fn get_top_processes(top_n: usize, sort_by: &str) -> (ProcessStats, Vec<ProcessInfo>) {
    let mut sys = System::new_all();
    sys.refresh_all();

    // Second refresh for accurate CPU readings
    std::thread::sleep(std::time::Duration::from_millis(100));
    sys.refresh_all();

    let stats = ProcessStats {
        status: PcaiStatus::Success,
        total_processes: sys.processes().len() as u32,
        total_threads: sys.processes().len() as u32, // Approximation
        system_cpu_usage: sys.global_cpu_usage(),
        system_memory_used_bytes: sys.used_memory(),
        system_memory_total_bytes: sys.total_memory(),
        elapsed_ms: 0,
    };

    // Collect process info
    let mut processes: Vec<ProcessInfo> = sys
        .processes()
        .iter()
        .map(|(pid, process)| ProcessInfo {
            pid: pid.as_u32(),
            name: process.name().to_string_lossy().to_string(),
            cpu_usage: process.cpu_usage(),
            memory_bytes: process.memory(),
            memory_formatted: format_bytes(process.memory()),
            status: format!("{:?}", process.status()),
            exe_path: process.exe().map(|p| p.to_string_lossy().to_string()),
        })
        .collect();

    // Sort by specified criteria
    match sort_by.to_lowercase().as_str() {
        "cpu" => processes.sort_by(|a, b| {
            b.cpu_usage
                .partial_cmp(&a.cpu_usage)
                .unwrap_or(std::cmp::Ordering::Equal)
        }),
        _ => processes.sort_by(|a, b| b.memory_bytes.cmp(&a.memory_bytes)),
    }

    processes.truncate(top_n);

    (stats, processes)
}

/// Get process by PID
pub fn get_process_by_pid(pid: u32) -> Option<ProcessInfo> {
    let mut sys = System::new();
    sys.refresh_processes(ProcessesToUpdate::Some(&[Pid::from_u32(pid)]), true);

    sys.process(Pid::from_u32(pid)).map(|process| ProcessInfo {
        pid,
        name: process.name().to_string_lossy().to_string(),
        cpu_usage: process.cpu_usage(),
        memory_bytes: process.memory(),
        memory_formatted: format_bytes(process.memory()),
        status: format!("{:?}", process.status()),
        exe_path: process.exe().map(|p| p.to_string_lossy().to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_process_stats() {
        let stats = get_process_stats();
        assert_eq!(stats.status, PcaiStatus::Success);
        assert!(stats.total_processes > 0);
        assert!(stats.system_memory_total_bytes > 0);
    }

    #[test]
    fn test_get_top_processes_by_memory() {
        let (stats, processes) = get_top_processes(10, "memory");
        assert_eq!(stats.status, PcaiStatus::Success);
        assert!(!processes.is_empty());

        // Verify sorted by memory (descending)
        for i in 1..processes.len() {
            assert!(processes[i - 1].memory_bytes >= processes[i].memory_bytes);
        }
    }

    #[test]
    fn test_get_top_processes_by_cpu() {
        let (stats, processes) = get_top_processes(10, "cpu");
        assert_eq!(stats.status, PcaiStatus::Success);
        assert!(!processes.is_empty());

        // Verify sorted by CPU (descending)
        for i in 1..processes.len() {
            assert!(processes[i - 1].cpu_usage >= processes[i].cpu_usage);
        }
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_get_current_process() {
        let pid = std::process::id();
        let process = get_process_by_pid(pid);
        assert!(process.is_some());

        let p = process.expect("TODO: Verify unwrap");
        assert_eq!(p.pid, pid);
        assert!(!p.name.is_empty());
    }
}
