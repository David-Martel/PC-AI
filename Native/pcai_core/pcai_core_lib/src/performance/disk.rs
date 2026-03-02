//! Disk Usage Analysis
//!
//! Parallel directory traversal for calculating disk usage with top-N breakdown.

use crate::PcaiStatus;
use parking_lot::Mutex;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use walkdir::WalkDir;

/// FFI-safe disk usage statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DiskUsageStats {
    pub status: PcaiStatus,
    pub total_size_bytes: u64,
    pub total_files: u64,
    pub total_dirs: u64,
    pub elapsed_ms: u64,
}

impl DiskUsageStats {
    pub fn error(status: PcaiStatus) -> Self {
        Self {
            status,
            total_size_bytes: 0,
            total_files: 0,
            total_dirs: 0,
            elapsed_ms: 0,
        }
    }
}

impl Default for DiskUsageStats {
    fn default() -> Self {
        Self {
            status: PcaiStatus::Success,
            total_size_bytes: 0,
            total_files: 0,
            total_dirs: 0,
            elapsed_ms: 0,
        }
    }
}

/// Directory usage entry for top-N breakdown
#[derive(Debug, Clone, Serialize)]
pub struct DirUsageEntry {
    pub path: String,
    pub size_bytes: u64,
    pub file_count: u64,
    pub size_formatted: String,
}

/// JSON output structure for disk usage
#[derive(Debug, Serialize)]
pub struct DiskUsageJson {
    pub status: String,
    pub root_path: String,
    pub total_size_bytes: u64,
    pub total_files: u64,
    pub total_dirs: u64,
    pub elapsed_ms: u64,
    pub top_entries: Vec<DirUsageEntry>,
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

/// Calculate directory size recursively (single directory)
fn calculate_dir_size(path: &Path) -> (u64, u64) {
    let mut size = 0u64;
    let mut count = 0u64;

    for entry in WalkDir::new(path)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            if let Ok(metadata) = entry.metadata() {
                size += metadata.len();
                count += 1;
            }
        }
    }

    (size, count)
}

/// Get disk usage statistics with top-N largest directories
pub fn get_disk_usage(root_path: &str, top_n: usize) -> io::Result<(DiskUsageStats, Vec<DirUsageEntry>)> {
    let root = Path::new(root_path);
    if !root.exists() {
        return Err(io::Error::new(io::ErrorKind::NotFound, "path not found"));
    }

    if !root.is_dir() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "path is not a directory"));
    }

    // Collect immediate subdirectories
    let subdirs: Vec<_> = fs::read_dir(root)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.path())
        .collect();

    // Atomic counters for totals
    let total_size = AtomicU64::new(0);
    let total_files = AtomicU64::new(0);
    let total_dirs = AtomicU64::new(subdirs.len() as u64);

    // Thread-safe map for directory sizes
    let dir_sizes: Arc<Mutex<HashMap<String, (u64, u64)>>> = Arc::new(Mutex::new(HashMap::new()));

    // Process subdirectories in parallel
    subdirs.par_iter().for_each(|subdir| {
        let (size, count) = calculate_dir_size(subdir);
        total_size.fetch_add(size, Ordering::Relaxed);
        total_files.fetch_add(count, Ordering::Relaxed);

        let mut sizes = dir_sizes.lock();
        sizes.insert(subdir.to_string_lossy().to_string(), (size, count));
    });

    // Also count files directly in root
    let root_files: Vec<_> = fs::read_dir(root)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .collect();

    for entry in &root_files {
        if let Ok(metadata) = entry.metadata() {
            total_size.fetch_add(metadata.len(), Ordering::Relaxed);
            total_files.fetch_add(1, Ordering::Relaxed);
        }
    }

    // Build stats
    let stats = DiskUsageStats {
        status: PcaiStatus::Success,
        total_size_bytes: total_size.load(Ordering::Relaxed),
        total_files: total_files.load(Ordering::Relaxed),
        total_dirs: total_dirs.load(Ordering::Relaxed),
        elapsed_ms: 0,
    };

    // Get top-N entries sorted by size
    let sizes = dir_sizes.lock();
    let mut entries: Vec<_> = sizes
        .iter()
        .map(|(path, (size, count))| DirUsageEntry {
            path: path.clone(),
            size_bytes: *size,
            file_count: *count,
            size_formatted: format_bytes(*size),
        })
        .collect();

    entries.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
    entries.truncate(top_n);

    Ok((stats, entries))
}

/// Get disk space information for all drives
pub fn get_disk_space() -> Vec<DriveInfo> {
    use sysinfo::Disks;

    let disks = Disks::new_with_refreshed_list();
    disks
        .iter()
        .map(|disk| DriveInfo {
            name: disk.name().to_string_lossy().to_string(),
            mount_point: disk.mount_point().to_string_lossy().to_string(),
            file_system: disk.file_system().to_string_lossy().to_string(),
            total_bytes: disk.total_space(),
            available_bytes: disk.available_space(),
            used_bytes: disk.total_space().saturating_sub(disk.available_space()),
            is_removable: disk.is_removable(),
        })
        .collect()
}

/// Drive information structure
#[derive(Debug, Clone, Serialize)]
pub struct DriveInfo {
    pub name: String,
    pub mount_point: String,
    pub file_system: String,
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub used_bytes: u64,
    pub is_removable: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_disk_usage_basic() {
        let temp_dir = TempDir::new().expect("TODO: Verify unwrap");
        let temp_path = temp_dir.path();

        // Create some test files
        let mut file1 = File::create(temp_path.join("file1.txt")).expect("TODO: Verify unwrap");
        file1.write_all(b"Hello, World!").expect("TODO: Verify unwrap");

        let mut file2 = File::create(temp_path.join("file2.txt")).expect("TODO: Verify unwrap");
        file2.write_all(b"Test content here").expect("TODO: Verify unwrap");

        // Create a subdirectory with a file
        let subdir = temp_path.join("subdir");
        fs::create_dir(&subdir).expect("TODO: Verify unwrap");
        let mut file3 = File::create(subdir.join("file3.txt")).expect("TODO: Verify unwrap");
        file3.write_all(b"Subdirectory content").expect("TODO: Verify unwrap");

        let result = get_disk_usage(temp_path.to_str().expect("TODO: Verify unwrap"), 10);
        assert!(result.is_ok());

        let (stats, entries) = result.expect("TODO: Verify unwrap");
        assert_eq!(stats.status, PcaiStatus::Success);
        assert!(stats.total_files >= 3);
        assert!(stats.total_size_bytes > 0);
        assert_eq!(entries.len(), 1); // Only one subdir
    }

    #[test]
    fn test_disk_usage_nonexistent() {
        let result = get_disk_usage("C:\\nonexistent\\path\\xyz", 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_disk_space() {
        let drives = get_disk_space();
        assert!(!drives.is_empty());

        for drive in &drives {
            assert!(drive.total_bytes >= drive.available_bytes);
            assert_eq!(
                drive.used_bytes,
                drive.total_bytes.saturating_sub(drive.available_bytes)
            );
        }
    }
}
