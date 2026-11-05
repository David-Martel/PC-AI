//! Process Monitoring
//!
//! Process enumeration and statistics gathering using sysinfo crate.

use crate::string::{bytes_to_buffer, PcaiByteBuffer};
use crate::PcaiStatus;
use serde::Serialize;
use std::time::Instant;
#[cfg(windows)]
use sysinfo::CpuRefreshKind;
use sysinfo::{
    MemoryRefreshKind, Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind, MINIMUM_CPU_UPDATE_INTERVAL,
};

#[cfg(windows)]
fn windows_cpu_times() -> Option<(u64, u64, u64)> {
    use windows::Win32::Foundation::FILETIME;
    use windows::Win32::System::Threading::{GetActiveProcessorGroupCount, GetSystemTimes};
    // GetSystemTimes covers only the calling group on hosts with multiple groups.
    // Preserve an unknown global value instead of presenting a group as the host.
    if unsafe { GetActiveProcessorGroupCount() } != 1 {
        return None;
    }
    let mut idle = FILETIME::default();
    let mut kernel = FILETIME::default();
    let mut user = FILETIME::default();
    // SAFETY: all three pointers reference distinct, live FILETIME outputs.
    unsafe { GetSystemTimes(Some(&raw mut idle), Some(&raw mut kernel), Some(&raw mut user)) }.ok()?;
    let ticks = |time: FILETIME| (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime);
    Some((ticks(idle), ticks(kernel), ticks(user)))
}

#[cfg(windows)]
fn windows_cpu_percent(previous: Option<(u64, u64, u64)>, current: Option<(u64, u64, u64)>) -> f32 {
    let Some(((old_idle, old_kernel, old_user), (idle, kernel, user))) = previous.zip(current) else {
        return f32::NAN;
    };
    let Some((idle_delta, total_delta)) = idle.checked_sub(old_idle).zip(
        kernel
            .checked_sub(old_kernel)
            .and_then(|delta| delta.checked_add(user.checked_sub(old_user)?)),
    ) else {
        return f32::NAN;
    };
    // Kernel includes idle. Subtract idle once from kernel + user.
    if total_delta == 0 || idle_delta > total_delta {
        return f32::NAN;
    }
    ((total_delta - idle_delta) as f64 * 100.0 / total_delta as f64) as f32
}

/// Reuses process state and qualified CPU baselines across snapshots.
///
/// Windows system CPU uses GetSystemTimes deltas, avoiding PDH startup. A missing
/// baseline, API failure or multiple processor groups yields NaN (JSON null),
/// because GetSystemTimes cannot establish whole-host CPU across groups.
#[derive(Debug)]
pub struct ProcessSampler {
    system: System,
    last_cpu_refresh: Option<Instant>,
    #[cfg(windows)]
    system_cpu_times: Option<(u64, u64, u64)>,
    #[cfg(windows)]
    system_cpu_usage: f32,
}

impl Default for ProcessSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessSampler {
    /// Creates an uninitialized sampler without enumerating processes.
    pub fn new() -> Self {
        #[cfg(windows)]
        let system = {
            let mut system = System::new();
            // Establish process CPU normalization without opening PDH counters.
            system.refresh_cpu_list(CpuRefreshKind::nothing());
            system
        };
        #[cfg(not(windows))]
        let system = System::new();
        Self {
            system,
            last_cpu_refresh: None,
            #[cfg(windows)]
            system_cpu_times: None,
            #[cfg(windows)]
            system_cpu_usage: f32::NAN,
        }
    }

    /// Returns system statistics with a qualified CPU baseline.
    pub fn get_process_stats(&mut self) -> ProcessStats {
        let start = Instant::now();
        self.refresh_snapshot(false);
        self.build_stats(start)
    }

    /// Returns sorted rows with fresh memory and qualified, possibly cached CPU.
    ///
    /// CPU sorting waits for the minimum CPU sampling interval. Memory sorting
    /// reuses the previous CPU reading within that interval. Newly discovered
    /// processes can report zero CPU until their next qualified refresh.
    pub fn get_top_processes(&mut self, top_n: usize, sort_by: &str) -> (ProcessStats, Vec<ProcessInfo>) {
        let start = Instant::now();
        self.refresh_snapshot(sort_by.eq_ignore_ascii_case("cpu"));
        let mut processes: Vec<ProcessInfo> = self
            .system
            .processes()
            .iter()
            .map(|(pid, process)| ProcessInfo {
                pid: pid.as_u32(),
                name: process.name().to_string_lossy().into_owned(),
                cpu_usage: process.cpu_usage(),
                memory_bytes: process.memory(),
                memory_formatted: format_bytes(process.memory()),
                status: format!("{:?}", process.status()),
                exe_path: None,
            })
            .collect();
        if sort_by.eq_ignore_ascii_case("cpu") {
            processes.sort_by(|a, b| b.cpu_usage.total_cmp(&a.cpu_usage));
        } else {
            processes.sort_by_key(|row| std::cmp::Reverse(row.memory_bytes));
        }
        processes.truncate(top_n);
        if !processes.is_empty() {
            let pids: Vec<Pid> = processes.iter().map(|row| Pid::from_u32(row.pid)).collect();
            self.system.refresh_processes_specifics(
                ProcessesToUpdate::Some(&pids),
                false,
                ProcessRefreshKind::nothing()
                    .without_tasks()
                    .with_exe(UpdateKind::OnlyIfNotSet),
            );
            for row in &mut processes {
                row.exe_path = self
                    .system
                    .process(Pid::from_u32(row.pid))
                    .and_then(|process| process.exe().map(|path| path.to_string_lossy().into_owned()));
            }
        }
        (self.build_stats(start), processes)
    }

    fn refresh_snapshot(&mut self, require_fresh_cpu: bool) {
        self.system
            .refresh_memory_specifics(MemoryRefreshKind::nothing().with_ram());
        if self.last_cpu_refresh.is_none() {
            self.refresh_processes(true);
            self.last_cpu_refresh = Some(Instant::now());
            std::thread::sleep(MINIMUM_CPU_UPDATE_INTERVAL);
            self.refresh_processes(true);
        } else {
            let remaining = MINIMUM_CPU_UPDATE_INTERVAL.saturating_sub(
                self.last_cpu_refresh
                    .map_or(MINIMUM_CPU_UPDATE_INTERVAL, |last| last.elapsed()),
            );
            if require_fresh_cpu && !remaining.is_zero() {
                std::thread::sleep(remaining);
            }
            self.refresh_processes(require_fresh_cpu || remaining.is_zero());
        }
    }

    fn refresh_processes(&mut self, include_cpu: bool) {
        let mut kind = ProcessRefreshKind::nothing().without_tasks().with_memory();
        if include_cpu {
            kind = kind.with_cpu();
            #[cfg(not(windows))]
            self.system.refresh_cpu_usage();
            #[cfg(windows)]
            {
                let current = windows_cpu_times();
                self.system_cpu_usage = windows_cpu_percent(self.system_cpu_times, current);
                self.system_cpu_times = current;
            }
        }
        self.system
            .refresh_processes_specifics(ProcessesToUpdate::All, true, kind);
        if include_cpu {
            self.last_cpu_refresh = Some(Instant::now());
        }
    }

    fn build_stats(&self, start: Instant) -> ProcessStats {
        ProcessStats {
            status: PcaiStatus::Success,
            total_processes: self.system.processes().len() as u32,
            // Existing ABI reports a minimum approximation, not actual thread counts.
            total_threads: self.system.processes().len() as u32,
            #[cfg(not(windows))]
            system_cpu_usage: self.system.global_cpu_usage(),
            #[cfg(windows)]
            system_cpu_usage: self.system_cpu_usage,
            system_memory_used_bytes: self.system.used_memory(),
            system_memory_total_bytes: self.system.total_memory(),
            elapsed_ms: start.elapsed().as_millis() as u64,
        }
    }
}

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
    let bytes = unsafe { std::slice::from_raw_parts((value as *const T) as *const u8, std::mem::size_of::<T>()) };
    target.extend_from_slice(bytes);
}

fn append_compact_process_entry(target: &mut Vec<u8>, entry: &ProcessListCompactEntry) {
    // The repr(C) entry has implicit alignment padding at bytes 36..40 on the
    // supported 64-bit ABI. Reading its whole allocation would read uninitialized
    // padding. Encode fields instead, retaining the existing 48-byte wire layout.
    for value in [
        entry.pid,
        entry.name_offset,
        entry.name_length,
        entry.status_offset,
        entry.status_length,
        entry.exe_path_offset,
        entry.exe_path_length,
    ] {
        target.extend_from_slice(&value.to_ne_bytes());
    }
    target.extend_from_slice(&entry.cpu_usage.to_ne_bytes());
    target.extend_from_slice(&entry._cpu_padding);
    target.extend_from_slice(&[0_u8; 4]);
    target.extend_from_slice(&entry.memory_bytes.to_ne_bytes());
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
    for entry in &entries {
        append_compact_process_entry(&mut packed, entry);
    }
    packed.extend_from_slice(&string_data);
    Ok(bytes_to_buffer(packed))
}

/// Get system-wide process statistics.
pub fn get_process_stats() -> ProcessStats {
    ProcessSampler::new().get_process_stats()
}

/// Get top N processes sorted by memory or CPU.
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
    #[ignore = "bounded subprocess fixture; invoked by the CPU sampling test"]
    fn cpu_busy_child_fixture() {
        let Some(mode) = std::env::var_os("PCAI_SAMPLER_CHILD_MODE") else {
            return;
        };
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pipe_stop = std::sync::Arc::clone(&stop);
        std::thread::spawn(move || {
            use std::io::Read;
            let _ = std::io::stdin().read(&mut [0_u8]);
            pipe_stop.store(true, std::sync::atomic::Ordering::Release);
        });
        let start = Instant::now();
        let mut value = 0_u64;
        // Parent closes stdin when sampling is complete. The existing ten-second
        // hard guard also stops the fixture if its parent disappears or stalls.
        while !stop.load(std::sync::atomic::Ordering::Acquire) && start.elapsed() < std::time::Duration::from_secs(10) {
            if mode == "busy" {
                for _ in 0..10_000 {
                    value = std::hint::black_box(value.wrapping_mul(17).wrapping_add(3));
                }
            } else {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
        std::hint::black_box(value);
    }

    #[test]
    fn qualified_interval_observes_positive_cpu_from_a_real_busy_child() {
        let mut sampler = ProcessSampler::new();
        sampler.refresh_snapshot(true);
        let mut child = std::process::Command::new(std::env::current_exe().expect("test executable should resolve"))
            .args([
                "--ignored",
                "--exact",
                "performance::process::tests::cpu_busy_child_fixture",
            ])
            .env("PCAI_SAMPLER_CHILD_MODE", "busy")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .spawn()
            .expect("owned busy fixture should start");
        let mut positive_cpu = false;
        for _ in 0..3 {
            sampler.refresh_snapshot(true);
            positive_cpu = sampler
                .system
                .process(Pid::from_u32(child.id()))
                .is_some_and(|process| process.cpu_usage() > 0.0);
            if positive_cpu {
                break;
            }
        }
        drop(child.stdin.take());
        child.wait().expect("owned busy fixture should be reaped");
        assert!(
            positive_cpu,
            "qualified sampling must observe real CPU work, not merely sorted zero rows"
        );
    }

    #[test]
    fn persistent_sampler_qualifies_cpu_and_refreshes_exited_processes() {
        let mut sampler = ProcessSampler::new();
        assert!(sampler.last_cpu_refresh.is_none());
        sampler.refresh_snapshot(true);
        let mut child = std::process::Command::new(std::env::current_exe().expect("test executable should resolve"))
            .args([
                "--ignored",
                "--exact",
                "performance::process::tests::cpu_busy_child_fixture",
            ])
            .env("PCAI_SAMPLER_CHILD_MODE", "idle")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .spawn()
            .expect("fixture child should start");
        let child_pid = child.id();
        sampler.refresh_snapshot(false);
        let child_was_present = sampler.system.process(Pid::from_u32(child_pid)).is_some();
        drop(child.stdin.take());
        child.wait().expect("owned fixture should be reaped");
        assert!(child_was_present);
        sampler.refresh_snapshot(false);
        assert!(sampler.system.process(Pid::from_u32(child_pid)).is_none());
        assert!(sampler.system.process(Pid::from_u32(std::process::id())).is_some());
    }

    #[cfg(windows)]
    #[test]
    fn windows_cpu_delta_excludes_idle_once_and_rejects_unknown_samples() {
        assert_eq!(windows_cpu_percent(Some((20, 50, 10)), Some((50, 90, 30))), 50.0);
        assert!(windows_cpu_percent(None, Some((1, 2, 3))).is_nan());
        assert!(windows_cpu_percent(Some((1, 2, 3)), None).is_nan());
        assert!(windows_cpu_percent(Some((1, 2, 3)), Some((1, 2, 3))).is_nan());
        assert!(windows_cpu_percent(Some((1, 2, 3)), Some((0, 2, 4))).is_nan());
        assert!(windows_cpu_percent(Some((1, 2, 3)), Some((10, 3, 4))).is_nan());
    }

    #[cfg(windows)]
    #[test]
    fn windows_cpu_reports_only_qualified_whole_host_samples() {
        let groups = unsafe { windows::Win32::System::Threading::GetActiveProcessorGroupCount() };
        println!("Windows processor groups: {groups}");
        let mut sampler = ProcessSampler::new();
        assert!(!sampler.system.cpus().is_empty());
        let stats = sampler.get_process_stats();
        if groups == 1 {
            assert!((0.0..=100.0).contains(&stats.system_cpu_usage));
        } else {
            assert!(stats.system_cpu_usage.is_nan());
        }
    }

    #[test]
    fn cpu_sort_uses_qualified_refresh_and_elapsed_includes_sampling() {
        let mut sampler = ProcessSampler::new();
        let (stats, rows) = sampler.get_top_processes(15, "cpu");
        assert!(stats.elapsed_ms >= MINIMUM_CPU_UPDATE_INTERVAL.as_millis() as u64);
        assert!(rows.windows(2).all(|rows| rows[0].cpu_usage >= rows[1].cpu_usage));
        let last = sampler.last_cpu_refresh.expect("CPU baseline should be established");
        sampler.get_top_processes(0, "cpu");
        assert!(
            sampler
                .last_cpu_refresh
                .expect("CPU refresh should persist")
                .duration_since(last)
                >= MINIMUM_CPU_UPDATE_INTERVAL
        );
    }

    #[test]
    fn compact_layout_remains_unchanged() {
        assert_eq!(std::mem::size_of::<ProcessListCompactHeader>(), 72);
        assert_eq!(std::mem::size_of::<ProcessListCompactEntry>(), 48);
        assert_eq!(std::mem::offset_of!(ProcessListCompactEntry, memory_bytes), 40);
    }

    #[test]
    fn compact_entry_encodes_fields_and_zeroes_alignment_padding() {
        let entry = ProcessListCompactEntry {
            pid: 17,
            name_offset: 3,
            name_length: 5,
            status_offset: 8,
            status_length: 7,
            exe_path_offset: 15,
            exe_path_length: 15,
            cpu_usage: 37.5,
            _cpu_padding: [0; 4],
            memory_bytes: 0x1122_3344_5566_7788,
        };
        let mut bytes = Vec::new();
        append_compact_process_entry(&mut bytes, &entry);
        assert_eq!(bytes.len(), 48);
        assert_eq!(&bytes[..4], &17_u32.to_ne_bytes());
        assert_eq!(&bytes[28..32], &37.5_f32.to_ne_bytes());
        assert_eq!(&bytes[32..40], &[0_u8; 8]);
        assert_eq!(&bytes[40..48], &0x1122_3344_5566_7788_u64.to_ne_bytes());
        let mut repeat = Vec::new();
        append_compact_process_entry(&mut repeat, &entry);
        assert_eq!(bytes, repeat);
    }

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
