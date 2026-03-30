use serde::Serialize;
use windows_sys::Win32::Foundation::*;
use windows_sys::Win32::System::ProcessStatus::*;
use windows_sys::Win32::System::Threading::*;

#[derive(Serialize)]
pub struct ProcessHistory {
    pub pid: u32,
    pub name: String,
    pub path: String,
    pub working_set_kb: u64,
    pub peak_working_set_kb: u64,
    pub page_file_usage_kb: u64,
    pub memory_mb: f64,
    pub cpu_percent: f64,
    pub threads: u32,
    pub handles: u32,
    pub io_read_bytes: u64,
    pub io_write_bytes: u64,
}

pub fn collect_process_telemetry() -> Vec<ProcessHistory> {
    let mut results = Vec::new();
    let mut pids = [0u32; 2048]; // Increased buffer
    let mut bytes_returned = 0;

    unsafe {
        if EnumProcesses(
            pids.as_mut_ptr(),
            std::mem::size_of_val(&pids) as u32,
            &mut bytes_returned,
        ) != 0
        {
            let count = bytes_returned as usize / std::mem::size_of::<u32>();
            for &pid in &pids[..count] {
                if pid == 0 {
                    continue;
                }

                let handle = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, 0, pid);
                if !handle.is_null() {
                    let mut path_buf = [0u16; 512];
                    let mut full_path = String::from("N/A");
                    let mut process_name = String::from("Unknown");

                    // Try to get full path
                    let mut size = path_buf.len() as u32;
                    if QueryFullProcessImageNameW(handle, 0, path_buf.as_mut_ptr(), &mut size) != 0 {
                        full_path = String::from_utf16_lossy(&path_buf[..size as usize]);
                        process_name = std::path::Path::new(&full_path)
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("Unknown")
                            .to_string();
                    }

                    let mut counters = PROCESS_MEMORY_COUNTERS {
                        cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
                        PageFaultCount: 0,
                        PeakWorkingSetSize: 0,
                        WorkingSetSize: 0,
                        QuotaPeakPagedPoolUsage: 0,
                        QuotaPagedPoolUsage: 0,
                        QuotaPeakNonPagedPoolUsage: 0,
                        QuotaNonPagedPoolUsage: 0,
                        PagefileUsage: 0,
                        PeakPagefileUsage: 0,
                    };

                    let mut io_counters = IO_COUNTERS {
                        ReadOperationCount: 0,
                        WriteOperationCount: 0,
                        OtherOperationCount: 0,
                        ReadTransferCount: 0,
                        WriteTransferCount: 0,
                        OtherTransferCount: 0,
                    };

                    let mut handle_count = 0;
                    GetProcessHandleCount(handle, &mut handle_count);

                    // Note: cpu_percent would ideally require two samples.
                    // For now, we return 0 or a very rough estimate if we had session state.
                    // To keep it high-perf, we'll stick to raw counters.

                    if GetProcessMemoryInfo(
                        handle,
                        &mut counters,
                        std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
                    ) != 0
                        && GetProcessIoCounters(handle, &mut io_counters) != 0
                    {
                        results.push(ProcessHistory {
                            pid,
                            name: process_name,
                            path: full_path,
                            working_set_kb: (counters.WorkingSetSize / 1024) as u64,
                            peak_working_set_kb: (counters.PeakWorkingSetSize / 1024) as u64,
                            page_file_usage_kb: (counters.PagefileUsage / 1024) as u64,
                            memory_mb: (counters.WorkingSetSize as f64) / 1024.0 / 1024.0,
                            cpu_percent: 0.0,
                            threads: 0, // requires toolhelp32 or similar enumeration
                            handles: handle_count,
                            io_read_bytes: io_counters.ReadTransferCount,
                            io_write_bytes: io_counters.WriteTransferCount,
                        });
                    }
                    CloseHandle(handle);
                }
            }
        }
    }

    results
}
