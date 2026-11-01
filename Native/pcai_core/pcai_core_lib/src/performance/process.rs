//! Process Monitoring
//!
//! Process enumeration and statistics gathering using sysinfo crate.

use std::time::{Duration, Instant};

use crate::string::{bytes_to_buffer, PcaiByteBuffer};
use crate::PcaiStatus;
use serde::Serialize;
use sysinfo::{
    MemoryRefreshKind, Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind,
    MINIMUM_CPU_UPDATE_INTERVAL,
};

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

fn process_refresh_kind(include_cpu: bool, include_exe: bool) -> ProcessRefreshKind {
    let mut refresh_kind = ProcessRefreshKind::nothing()
        .with_memory()
        .without_tasks();

    if include_cpu {
        refresh_kind = refresh_kind.with_cpu();
    }

    if include_exe {
        refresh_kind = refresh_kind.with_exe(UpdateKind::OnlyIfNotSet);
    }

    refresh_kind
}

pub struct ProcessSampler {
    system: System,
    cpu_ready: bool,
    last_cpu_refresh: Option<Instant>,
}

impl Default for ProcessSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessSampler {
    pub fn new() -> Self {
        Self {
            system: System::new(),
            cpu_ready: false,
            last_cpu_refresh: None,
        }
    }

    pub fn get_process_stats(&mut self) -> ProcessStats {
        let start = Instant::now();
        self.refresh_snapshot(false);
        self.build_stats(start)
    }

    pub fn get_top_processes(&mut self, top_n: usize, sort_by: &str) -> (ProcessStats, Vec<ProcessInfo>) {
        let start = Instant::now();
        let sort_key = sort_by.to_ascii_lowercase();
        let include_cpu = sort_key == "cpu";
        self.refresh_snapshot(include_cpu);

        let stats = self.build_stats(start);

        let mut processes: Vec<ProcessInfo> = self
            .system
            .processes()
            .iter()
            .map(|(pid, process)| ProcessInfo {
                pid: pid.as_u32(),
                name: process.name().to_string_lossy().to_string(),
                cpu_usage: process.cpu_usage(),
                memory_bytes: process.memory(),
                memory_formatted: format_bytes(process.memory()),
                status: format!("{:?}", process.status()),
                exe_path: None,
            })
            .collect();

        match sort_key.as_str() {
            "cpu" => processes.sort_by(|a, b| {
                b.cpu_usage
                    .partial_cmp(&a.cpu_usage)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
            _ => processes.sort_by(|a, b| b.memory_bytes.cmp(&a.memory_bytes)),
        }

        processes.truncate(top_n);

        if !processes.is_empty() {
            let top_pids: Vec<Pid> = processes.iter().map(|process| Pid::from_u32(process.pid)).collect();
            self.system.refresh_processes_specifics(
                ProcessesToUpdate::Some(&top_pids),
                false,
                process_refresh_kind(false, true),
            );

            for process in &mut processes {
                process.exe_path = self
                    .system
                    .process(Pid::from_u32(process.pid))
                    .and_then(|entry| entry.exe().map(|path| path.to_string_lossy().to_string()));
            }
        }

        (stats, processes)
    }

    fn build_stats(&self, start: Instant) -> ProcessStats {
        ProcessStats {
            status: PcaiStatus::Success,
            total_processes: self.system.processes().len() as u32,
            total_threads: self.system.processes().len() as u32,
            system_cpu_usage: self.system.global_cpu_usage(),
            system_memory_used_bytes: self.system.used_memory(),
            system_memory_total_bytes: self.system.total_memory(),
            elapsed_ms: start.elapsed().as_millis() as u64,
        }
    }

    fn refresh_snapshot(&mut self, include_cpu: bool) {
        let refresh_kind = process_refresh_kind(include_cpu, false);
        self.system
            .refresh_memory_specifics(MemoryRefreshKind::nothing().with_ram());

        if include_cpu {
            self.refresh_cpu_snapshot(refresh_kind);
        } else {
            self.system
                .refresh_processes_specifics(ProcessesToUpdate::All, true, refresh_kind);
            self.system.refresh_cpu_usage();
        }
    }

    fn refresh_cpu_snapshot(&mut self, refresh_kind: ProcessRefreshKind) {
        if self.cpu_ready {
            let sleep_for = self
                .last_cpu_refresh
                .map(|last_refresh| MINIMUM_CPU_UPDATE_INTERVAL.saturating_sub(last_refresh.elapsed()))
                .unwrap_or(Duration::ZERO);
            if !sleep_for.is_zero() {
                std::thread::sleep(sleep_for);
            }

            self.system.refresh_cpu_usage();
            self.system
                .refresh_processes_specifics(ProcessesToUpdate::All, true, refresh_kind);
            self.last_cpu_refresh = Some(Instant::now());
            return;
        }

        self.system.refresh_cpu_usage();
        self.system
            .refresh_processes_specifics(ProcessesToUpdate::All, true, refresh_kind);
        std::thread::sleep(MINIMUM_CPU_UPDATE_INTERVAL);
        self.system.refresh_cpu_usage();
        self.system
            .refresh_processes_specifics(ProcessesToUpdate::All, true, refresh_kind);
        self.cpu_ready = true;
        self.last_cpu_refresh = Some(Instant::now());
    }
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
    let bytes = unsafe { std::slice::from_raw_parts((value as *const T) as *const u8, std::mem::size_of::<T>()) };
    target.extend_from_slice(bytes);
}

fn append_pod_slice<T: Copy>(target: &mut Vec<u8>, values: &[T]) {
    if values.is_empty() {
        return;
    }

    let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values)) };
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
    ProcessSampler::new().get_process_stats()
}

/// Get top N processes sorted by memory or CPU
pub fn get_top_processes(top_n: usize, sort_by: &str) -> (ProcessStats, Vec<ProcessInfo>) {
    ProcessSampler::new().get_top_processes(top_n, sort_by)
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
