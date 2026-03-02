//! Memory Analysis
//!
//! System memory and swap usage monitoring.

use crate::PcaiStatus;
use serde::Serialize;
use sysinfo::System;

/// FFI-safe memory statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    pub status: PcaiStatus,
    pub total_memory_bytes: u64,
    pub used_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub total_swap_bytes: u64,
    pub used_swap_bytes: u64,
    pub elapsed_ms: u64,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            status: PcaiStatus::Success,
            total_memory_bytes: 0,
            used_memory_bytes: 0,
            available_memory_bytes: 0,
            total_swap_bytes: 0,
            used_swap_bytes: 0,
            elapsed_ms: 0,
        }
    }
}

/// Memory usage information for display
#[derive(Debug, Clone, Serialize)]
pub struct MemoryUsageInfo {
    pub total_memory: String,
    pub used_memory: String,
    pub available_memory: String,
    pub memory_percent: f64,
    pub total_swap: String,
    pub used_swap: String,
    pub swap_percent: f64,
}

/// JSON output structure for memory stats
#[derive(Debug, Serialize)]
pub struct MemoryJson {
    pub status: String,
    pub total_memory_bytes: u64,
    pub used_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub total_swap_bytes: u64,
    pub used_swap_bytes: u64,
    pub memory_usage_percent: f64,
    pub swap_usage_percent: f64,
    pub elapsed_ms: u64,
}

/// Format bytes as human-readable string
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Get system memory statistics
pub fn get_memory_stats() -> MemoryStats {
    let mut sys = System::new_all();
    sys.refresh_memory();

    MemoryStats {
        status: PcaiStatus::Success,
        total_memory_bytes: sys.total_memory(),
        used_memory_bytes: sys.used_memory(),
        available_memory_bytes: sys.available_memory(),
        total_swap_bytes: sys.total_swap(),
        used_swap_bytes: sys.used_swap(),
        elapsed_ms: 0,
    }
}

/// Get formatted memory usage information
pub fn get_memory_usage_info() -> MemoryUsageInfo {
    let stats = get_memory_stats();

    let memory_percent = if stats.total_memory_bytes > 0 {
        (stats.used_memory_bytes as f64 / stats.total_memory_bytes as f64) * 100.0
    } else {
        0.0
    };

    let swap_percent = if stats.total_swap_bytes > 0 {
        (stats.used_swap_bytes as f64 / stats.total_swap_bytes as f64) * 100.0
    } else {
        0.0
    };

    MemoryUsageInfo {
        total_memory: format_bytes(stats.total_memory_bytes),
        used_memory: format_bytes(stats.used_memory_bytes),
        available_memory: format_bytes(stats.available_memory_bytes),
        memory_percent,
        total_swap: format_bytes(stats.total_swap_bytes),
        used_swap: format_bytes(stats.used_swap_bytes),
        swap_percent,
    }
}

/// Check if system is under memory pressure
pub fn is_memory_pressure() -> bool {
    let stats = get_memory_stats();
    if stats.total_memory_bytes == 0 {
        return false;
    }

    let usage_percent = (stats.used_memory_bytes as f64 / stats.total_memory_bytes as f64) * 100.0;

    // Consider memory pressure if usage > 90%
    usage_percent > 90.0
}

/// Get processes consuming the most memory
pub fn get_top_memory_consumers(top_n: usize) -> Vec<super::process::ProcessInfo> {
    let (_, processes) = super::process::get_top_processes(top_n, "memory");
    processes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_memory_stats() {
        let stats = get_memory_stats();
        assert_eq!(stats.status, PcaiStatus::Success);
        assert!(stats.total_memory_bytes > 0);
        // Used + available should be close to total
        assert!(stats.used_memory_bytes <= stats.total_memory_bytes);
    }

    #[test]
    fn test_get_memory_usage_info() {
        let info = get_memory_usage_info();
        assert!(!info.total_memory.is_empty());
        assert!(info.memory_percent >= 0.0 && info.memory_percent <= 100.0);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
        assert_eq!(format_bytes(1099511627776), "1.00 TB");
    }

    #[test]
    fn test_is_memory_pressure() {
        // This test just verifies the function runs
        let _ = is_memory_pressure();
    }

    #[test]
    fn test_get_top_memory_consumers() {
        let consumers = get_top_memory_consumers(5);
        assert!(!consumers.is_empty());
        // Should be sorted by memory descending
        for i in 1..consumers.len() {
            assert!(consumers[i - 1].memory_bytes >= consumers[i].memory_bytes);
        }
    }
}
