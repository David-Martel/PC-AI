//! PCAI Performance Module
//!
//! Consolidated performance monitoring logic.

pub mod disk;
pub mod memory;
pub mod process;

use crate::string::PcaiStringBuffer;
use crate::PcaiStatus;
use std::ffi::{c_char, CStr};
use std::time::Instant;

// Re-export types
pub use disk::DiskUsageStats;
pub use memory::MemoryStats;
pub use process::ProcessStats;

#[no_mangle]
pub unsafe extern "C" fn pcai_get_disk_usage(root_path: *const c_char, top_n: u32) -> DiskUsageStats {
    if root_path.is_null() {
        return DiskUsageStats::error(PcaiStatus::NullPointer);
    }

    let path_str = match CStr::from_ptr(root_path).to_str() {
        Ok(s) => s,
        Err(_) => return DiskUsageStats::error(PcaiStatus::InvalidUtf8),
    };

    match disk::get_disk_usage(path_str, top_n as usize) {
        Ok((stats, _)) => stats,
        Err(_) => DiskUsageStats::error(PcaiStatus::IoError),
    }
}

#[no_mangle]
pub unsafe extern "C" fn pcai_get_disk_usage_json(root_path: *const c_char, top_n: u32) -> PcaiStringBuffer {
    if root_path.is_null() {
        return PcaiStringBuffer::error(PcaiStatus::NullPointer);
    }

    let path_str = match CStr::from_ptr(root_path).to_str() {
        Ok(s) => s,
        Err(_) => return PcaiStringBuffer::error(PcaiStatus::InvalidUtf8),
    };

    match disk::get_disk_usage(path_str, top_n as usize) {
        Ok((stats, entries)) => {
            let json_output = disk::DiskUsageJson {
                status: "Success".to_string(),
                root_path: path_str.to_string(),
                total_size_bytes: stats.total_size_bytes,
                total_files: stats.total_files,
                total_dirs: stats.total_dirs,
                elapsed_ms: 0, // Calculated inside get_disk_usage usually
                top_entries: entries,
            };
            crate::string::json_to_buffer(&json_output)
        }
        Err(_) => PcaiStringBuffer::error(PcaiStatus::IoError),
    }
}

#[no_mangle]
pub extern "C" fn pcai_get_process_stats() -> ProcessStats {
    process::get_process_stats()
}

#[no_mangle]
pub unsafe extern "C" fn pcai_get_top_processes_json(top_n: u32, sort_by: *const c_char) -> PcaiStringBuffer {
    let sort_key = if sort_by.is_null() {
        "memory"
    } else {
        CStr::from_ptr(sort_by).to_str().unwrap_or("memory")
    };

    let start = Instant::now();
    let (stats, processes) = process::get_top_processes(top_n as usize, sort_key);

    let json_output = process::ProcessListJson {
        status: "Success".to_string(),
        total_processes: stats.total_processes,
        total_threads: stats.total_threads,
        system_cpu_usage: stats.system_cpu_usage,
        system_memory_used_bytes: stats.system_memory_used_bytes,
        system_memory_total_bytes: stats.system_memory_total_bytes,
        elapsed_ms: start.elapsed().as_millis() as u64,
        sort_by: sort_key.to_string(),
        processes,
    };

    crate::string::json_to_buffer(&json_output)
}

#[no_mangle]
pub extern "C" fn pcai_get_memory_stats() -> MemoryStats {
    memory::get_memory_stats()
}

#[no_mangle]
pub extern "C" fn pcai_get_memory_stats_json() -> PcaiStringBuffer {
    let stats = memory::get_memory_stats();

    let json_output = memory::MemoryJson {
        status: "Success".to_string(),
        total_memory_bytes: stats.total_memory_bytes,
        used_memory_bytes: stats.used_memory_bytes,
        available_memory_bytes: stats.available_memory_bytes,
        total_swap_bytes: stats.total_swap_bytes,
        used_swap_bytes: stats.used_swap_bytes,
        memory_usage_percent: if stats.total_memory_bytes > 0 {
            (stats.used_memory_bytes as f64 / stats.total_memory_bytes as f64) * 100.0
        } else {
            0.0
        },
        swap_usage_percent: if stats.total_swap_bytes > 0 {
            (stats.used_swap_bytes as f64 / stats.total_swap_bytes as f64) * 100.0
        } else {
            0.0
        },
        elapsed_ms: 0,
    };

    crate::string::json_to_buffer(&json_output)
}
