//! Disk Usage Analysis
//!
//! Parallel directory traversal for calculating disk usage with top-N breakdown.

use crate::string::{bytes_to_buffer, PcaiByteBuffer};
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

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DiskUsageCompactHeader {
    pub status: PcaiStatus,
    pub reserved: u32,
    pub total_size_bytes: u64,
    pub total_files: u64,
    pub total_dirs: u64,
    pub elapsed_ms: u64,
    pub root_path_offset: u32,
    pub root_path_length: u32,
    pub entry_count: u64,
    pub string_bytes: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DiskUsageCompactEntry {
    pub path_offset: u32,
    pub path_length: u32,
    pub size_bytes: u64,
    pub file_count: u64,
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

pub fn pack_disk_usage_compact(
    root_path: &str,
    stats: &DiskUsageStats,
    entries: &[DirUsageEntry],
) -> Result<PcaiByteBuffer, PcaiStatus> {
    let mut string_data = Vec::new();
    let (root_path_offset, root_path_length) = append_string(&mut string_data, root_path)?;
    let mut compact_entries = Vec::with_capacity(entries.len());

    for entry in entries {
        let (path_offset, path_length) = append_string(&mut string_data, &entry.path)?;
        compact_entries.push(DiskUsageCompactEntry {
            path_offset,
            path_length,
            size_bytes: entry.size_bytes,
            file_count: entry.file_count,
        });
    }

    let header = DiskUsageCompactHeader {
        status: stats.status,
        reserved: 0,
        total_size_bytes: stats.total_size_bytes,
        total_files: stats.total_files,
        total_dirs: stats.total_dirs,
        elapsed_ms: stats.elapsed_ms,
        root_path_offset,
        root_path_length,
        entry_count: compact_entries.len() as u64,
        string_bytes: string_data.len() as u64,
    };

    let mut packed = Vec::with_capacity(
        std::mem::size_of::<DiskUsageCompactHeader>()
            + (compact_entries.len() * std::mem::size_of::<DiskUsageCompactEntry>())
            + string_data.len(),
    );
    append_pod(&mut packed, &header);
    append_pod_slice(&mut packed, &compact_entries);
    packed.extend_from_slice(&string_data);
    Ok(bytes_to_buffer(packed))
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
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let temp_path = temp_dir.path();

        // Create some test files
        let mut file1 = File::create(temp_path.join("file1.txt")).expect("failed to create file1.txt");
        file1.write_all(b"Hello, World!").expect("failed to write file1.txt");

        let mut file2 = File::create(temp_path.join("file2.txt")).expect("failed to create file2.txt");
        file2.write_all(b"Test content here").expect("failed to write file2.txt");

        // Create a subdirectory with a file
        let subdir = temp_path.join("subdir");
        fs::create_dir(&subdir).expect("failed to create subdir");
        let mut file3 = File::create(subdir.join("file3.txt")).expect("failed to create file3.txt");
        file3.write_all(b"Subdirectory content").expect("failed to write file3.txt");

        let result = get_disk_usage(temp_path.to_str().expect("temp path is valid UTF-8"), 10);
        assert!(result.is_ok());

        let (stats, entries) = result.expect("get_disk_usage succeeded");
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
