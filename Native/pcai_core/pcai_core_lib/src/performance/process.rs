//! Process Monitoring
//!
//! Process enumeration and statistics gathering using sysinfo crate.

use crate::string::{bytes_to_buffer, PcaiByteBuffer};
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

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessListCompactHeader {
    pub status: PcaiStatus,
    pub reserved: u32,
    pub total_processes: u32,
    pub total_threads: u32,
    pub system_cpu_usage: f32,
    pub _cpu_padding: [u8; 4],
    pub system_memory_used_bytes: u64,
    pub system_memory_total_bytes: u64,
    pub elapsed_ms: u64,
    pub sort_by_offset: u32,
    pub sort_by_length: u32,
    pub entry_count: u64,
    pub string_bytes: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessListCompactEntry {
    pub pid: u32,
    pub name_offset: u32,
    pub name_length: u32,
    pub status_offset: u32,
    pub status_length: u32,
    pub exe_path_offset: u32,
    pub exe_path_length: u32,
    pub cpu_usage: f32,
    pub _cpu_padding: [u8; 4],
    pub memory_bytes: u64,
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

fn append_pod<T: Copy>(target: &mut Vec<u8>, value: &T) {
    let bytes = unsafe {
        std::slice::from_raw_parts((value as *const T) as *const u8, std::mem::size_of::<T>())
    };
    target.extend_from_slice(bytes);
}

fn append_pod_slice<T: Copy>(target: &mut Vec<u8>, values: &[T]) {
    if values.is_empty() {
        return;
    }

    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    };
    target.extend_from_slice(bytes);
}

fn usize_to_u32(value: usize) -> Result<u32, PcaiStatus> {
    u32::try_from(value).map_err(|_| PcaiStatus::OutOfMemory)
}

fn append_string(string_data: &mut Vec<u8>, value: &str) -> Result<(u32, u32), PcaiStatus> {
    let bytes = value.as_bytes();
    let offset = usize_to_u32(string_data.len())?;
    string_data.extend_from_slice(bytes);
    let length = usize_to_u32(bytes.len())?;
    Ok((offset, length))
}

pub fn pack_top_processes_compact(
    stats: &ProcessStats,
    sort_by: &str,
    processes: &[ProcessInfo],
) -> Result<PcaiByteBuffer, PcaiStatus> {
    let mut string_data = Vec::new();
    let (sort_by_offset, sort_by_length) = append_string(&mut string_data, sort_by)?;
    let mut entries = Vec::with_capacity(processes.len());

    for process in processes {
        let (name_offset, name_length) = append_string(&mut string_data, &process.name)?;
        let (status_offset, status_length) = append_string(&mut string_data, &process.status)?;
        let (exe_path_offset, exe_path_length) = match process.exe_path.as_deref() {
            Some(path) => append_string(&mut string_data, path)?,
            None => (0, 0),
        };

        entries.push(ProcessListCompactEntry {
            pid: process.pid,
            name_offset,
            name_length,
            status_offset,
            status_length,
            exe_path_offset,
            exe_path_length,
            cpu_usage: process.cpu_usage,
            _cpu_padding: [0; 4],
            memory_bytes: process.memory_bytes,
        });
    }

    let header = ProcessListCompactHeader {
        status: stats.status,
        reserved: 0,
        total_processes: stats.total_processes,
        total_threads: stats.total_threads,
        system_cpu_usage: stats.system_cpu_usage,
        _cpu_padding: [0; 4],
        system_memory_used_bytes: stats.system_memory_used_bytes,
        system_memory_total_bytes: stats.system_memory_total_bytes,
        elapsed_ms: stats.elapsed_ms,
        sort_by_offset,
        sort_by_length,
        entry_count: entries.len() as u64,
        string_bytes: string_data.len() as u64,
    };

    let mut packed = Vec::with_capacity(
        std::mem::size_of::<ProcessListCompactHeader>()
            + (entries.len() * std::mem::size_of::<ProcessListCompactEntry>())
            + string_data.len(),
    );
    append_pod(&mut packed, &header);
    append_pod_slice(&mut packed, &entries);
    packed.extend_from_slice(&string_data);
    Ok(bytes_to_buffer(packed))
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

        let p = process.expect("current process should be visible to sysinfo");
        assert_eq!(p.pid, pid);
        assert!(!p.name.is_empty());
    }
}
