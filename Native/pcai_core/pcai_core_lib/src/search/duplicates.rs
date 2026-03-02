//! Parallel duplicate file detection using SHA-256 hashing.
//!
//! Uses the `ignore` crate (ripgrep's file walker) for fast parallel directory
//! traversal and `rayon` for parallel hash computation.

use std::collections::HashMap;
use std::ffi::CStr;
use std::fs::File;
use std::io::{BufReader, Read};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use globset::{Glob, GlobMatcher};
use ignore::WalkBuilder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::path::parse_path_ffi;
use crate::string::{json_to_buffer, PcaiStringBuffer};
use crate::PcaiStatus;

/// Statistics returned by duplicate detection operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DuplicateStats {
    /// Operation status
    pub status: PcaiStatus,
    /// Total files scanned
    pub files_scanned: u64,
    /// Number of duplicate groups found
    pub duplicate_groups: u64,
    /// Total number of duplicate files (excluding originals)
    pub duplicate_files: u64,
    /// Total bytes wasted by duplicates
    pub wasted_bytes: u64,
    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
}

impl DuplicateStats {
    fn error(status: PcaiStatus) -> Self {
        Self {
            status,
            ..Default::default()
        }
    }
}

/// A group of duplicate files sharing the same hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroup {
    /// SHA-256 hash of the file contents
    pub hash: String,
    /// Size of each file in bytes
    pub size: u64,
    /// Paths to all files with this hash
    pub paths: Vec<String>,
}

/// Complete result of a duplicate detection operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateResult {
    /// Operation status (as string for JSON)
    pub status: String,
    /// Total files scanned
    pub files_scanned: u64,
    /// Number of duplicate groups
    pub duplicate_groups: u64,
    /// Total duplicate files (excluding originals)
    pub duplicate_files: u64,
    /// Total bytes wasted
    pub wasted_bytes: u64,
    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
    /// Groups of duplicate files
    pub groups: Vec<DuplicateGroup>,
}

/// Configuration for duplicate detection.
struct DuplicateConfig {
    root_path: PathBuf,
    min_size: u64,
    include_matcher: Option<GlobMatcher>,
    exclude_matcher: Option<GlobMatcher>,
}

impl DuplicateConfig {
    fn from_ffi(
        root_path: *const c_char,
        min_size: u64,
        include_pattern: *const c_char,
        exclude_pattern: *const c_char,
    ) -> Result<Self, PcaiStatus> {
        // Parse root path with cross-platform normalization
        let root = parse_path_ffi(root_path)?;

        if !root.exists() {
            return Err(PcaiStatus::PathNotFound);
        }

        // Parse include pattern
        let include_matcher = if !include_pattern.is_null() {
            let c_str = unsafe { CStr::from_ptr(include_pattern) };
            let pattern = c_str.to_str().map_err(|_| PcaiStatus::InvalidUtf8)?;
            if !pattern.is_empty() {
                let glob = Glob::new(pattern).map_err(|_| PcaiStatus::InvalidArgument)?;
                Some(glob.compile_matcher())
            } else {
                None
            }
        } else {
            None
        };

        // Parse exclude pattern
        let exclude_matcher = if !exclude_pattern.is_null() {
            let c_str = unsafe { CStr::from_ptr(exclude_pattern) };
            let pattern = c_str.to_str().map_err(|_| PcaiStatus::InvalidUtf8)?;
            if !pattern.is_empty() {
                let glob = Glob::new(pattern).map_err(|_| PcaiStatus::InvalidArgument)?;
                Some(glob.compile_matcher())
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            root_path: root,
            min_size,
            include_matcher,
            exclude_matcher,
        })
    }

    fn should_include(&self, path: &Path, size: u64) -> bool {
        // Check minimum size
        if size < self.min_size {
            return false;
        }

        // Check include pattern
        if let Some(ref matcher) = self.include_matcher {
            if !matcher.is_match(path) {
                return false;
            }
        }

        // Check exclude pattern
        if let Some(ref matcher) = self.exclude_matcher {
            if matcher.is_match(path) {
                return false;
            }
        }

        true
    }
}

/// Computes SHA-256 hash of a file.
fn hash_file(path: &Path) -> std::io::Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file); // 1MB buffer
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536]; // 64KB read chunks

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let hash = hasher.finalize();
    Ok(format!("{:x}", hash))
}

/// File info collected during scanning.
#[derive(Debug)]
struct FileInfo {
    path: PathBuf,
    size: u64,
}

/// Finds duplicate files and returns full results with file groups.
fn find_duplicates_impl(config: &DuplicateConfig) -> DuplicateResult {
    let start = Instant::now();
    let files_scanned = AtomicU64::new(0);

    // Phase 1: Collect all files that pass filters
    let mut files_by_size: HashMap<u64, Vec<PathBuf>> = HashMap::new();

    let walker = WalkBuilder::new(&config.root_path)
        .hidden(false) // Include hidden files
        .git_ignore(false) // Don't use .gitignore
        .threads(num_cpus::get())
        .build();

    for entry in walker.flatten() {
        if let Ok(metadata) = entry.metadata() {
            if metadata.is_file() {
                let size = metadata.len();
                let path = entry.path().to_path_buf();

                if config.should_include(&path, size) {
                    files_scanned.fetch_add(1, Ordering::Relaxed);
                    files_by_size.entry(size).or_default().push(path);
                }
            }
        }
    }

    // Phase 2: Only hash files that have size duplicates (optimization)
    let files_to_hash: Vec<FileInfo> = files_by_size
        .into_iter()
        .filter(|(_, paths)| paths.len() > 1) // Only sizes with potential duplicates
        .flat_map(|(size, paths)| {
            paths.into_iter().map(move |path| FileInfo { path, size })
        })
        .collect();

    // Phase 3: Parallel hashing
    let hash_map: Mutex<HashMap<String, Vec<(PathBuf, u64)>>> = Mutex::new(HashMap::new());

    files_to_hash.par_iter().for_each(|file_info| {
        if let Ok(hash) = hash_file(&file_info.path) {
            let mut map = hash_map.lock().expect("TODO: Verify unwrap");
            map.entry(hash)
                .or_default()
                .push((file_info.path.clone(), file_info.size));
        }
    });

    // Phase 4: Build result groups
    // Recover data even if mutex was poisoned (thread panicked)
    let hash_map = hash_map.into_inner().unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut groups: Vec<DuplicateGroup> = Vec::new();
    let mut total_duplicate_files = 0u64;
    let mut total_wasted_bytes = 0u64;

    for (hash, files) in hash_map {
        if files.len() > 1 {
            let size = files[0].1;
            let paths: Vec<String> = files.iter().map(|(p, _)| p.to_string_lossy().into_owned()).collect();

            let duplicates = files.len() as u64 - 1; // Exclude "original"
            total_duplicate_files += duplicates;
            total_wasted_bytes += duplicates * size;

            groups.push(DuplicateGroup { hash, size, paths });
        }
    }

    // Sort groups by wasted space (largest first)
    groups.sort_by(|a, b| {
        let waste_a = (a.paths.len() as u64 - 1) * a.size;
        let waste_b = (b.paths.len() as u64 - 1) * b.size;
        waste_b.cmp(&waste_a)
    });

    let elapsed = start.elapsed();

    DuplicateResult {
        status: "Success".to_string(),
        files_scanned: files_scanned.load(Ordering::Relaxed),
        duplicate_groups: groups.len() as u64,
        duplicate_files: total_duplicate_files,
        wasted_bytes: total_wasted_bytes,
        elapsed_ms: elapsed.as_millis() as u64,
        groups,
    }
}

/// Finds duplicate files and returns only statistics (no file list).
fn find_duplicates_stats_impl(config: &DuplicateConfig) -> DuplicateStats {
    let result = find_duplicates_impl(config);

    DuplicateStats {
        status: PcaiStatus::Success,
        files_scanned: result.files_scanned,
        duplicate_groups: result.duplicate_groups,
        duplicate_files: result.duplicate_files,
        wasted_bytes: result.wasted_bytes,
        elapsed_ms: result.elapsed_ms,
    }
}

// ============================================================================
// FFI Entry Points
// ============================================================================

/// FFI entry point for finding duplicates with full JSON result.
pub fn find_duplicates_ffi(
    root_path: *const c_char,
    min_size: u64,
    include_pattern: *const c_char,
    exclude_pattern: *const c_char,
) -> PcaiStringBuffer {
    match DuplicateConfig::from_ffi(root_path, min_size, include_pattern, exclude_pattern) {
        Ok(config) => {
            let result = find_duplicates_impl(&config);
            json_to_buffer(&result)
        }
        Err(status) => PcaiStringBuffer::error(status),
    }
}

/// FFI entry point for finding duplicates with stats only.
pub fn find_duplicates_stats_ffi(
    root_path: *const c_char,
    min_size: u64,
    include_pattern: *const c_char,
    exclude_pattern: *const c_char,
) -> DuplicateStats {
    match DuplicateConfig::from_ffi(root_path, min_size, include_pattern, exclude_pattern) {
        Ok(config) => find_duplicates_stats_impl(&config),
        Err(status) => DuplicateStats::error(status),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_dir() -> TempDir {
        let dir = TempDir::new().expect("TODO: Verify unwrap");

        // Create some duplicate files
        fs::write(dir.path().join("file1.txt"), "duplicate content").expect("TODO: Verify unwrap");
        fs::write(dir.path().join("file2.txt"), "duplicate content").expect("TODO: Verify unwrap");
        fs::write(dir.path().join("unique.txt"), "unique content").expect("TODO: Verify unwrap");

        // Create a subdirectory with more files
        let subdir = dir.path().join("subdir");
        fs::create_dir(&subdir).expect("TODO: Verify unwrap");
        fs::write(subdir.join("file3.txt"), "duplicate content").expect("TODO: Verify unwrap");
        fs::write(subdir.join("other.txt"), "other content").expect("TODO: Verify unwrap");

        dir
    }

    #[test]
    fn test_find_duplicates_basic() {
        let dir = create_test_dir();

        let config = DuplicateConfig {
            root_path: dir.path().to_path_buf(),
            min_size: 0,
            include_matcher: None,
            exclude_matcher: None,
        };

        let result = find_duplicates_impl(&config);

        assert_eq!(result.status, "Success");
        assert!(result.files_scanned >= 5);
        assert_eq!(result.duplicate_groups, 1); // One group of 3 duplicates
        assert_eq!(result.duplicate_files, 2); // 3 files - 1 original = 2 duplicates
    }

    #[test]
    fn test_find_duplicates_with_min_size() {
        let dir = create_test_dir();

        // Create a small file
        fs::write(dir.path().join("tiny.txt"), "x").expect("TODO: Verify unwrap");
        fs::write(dir.path().join("tiny2.txt"), "x").expect("TODO: Verify unwrap");

        let config = DuplicateConfig {
            root_path: dir.path().to_path_buf(),
            min_size: 10, // Skip files smaller than 10 bytes
            include_matcher: None,
            exclude_matcher: None,
        };

        let result = find_duplicates_impl(&config);

        // tiny.txt and tiny2.txt should be excluded
        assert_eq!(result.duplicate_groups, 1);
    }

    #[test]
    fn test_hash_file() {
        let dir = TempDir::new().expect("TODO: Verify unwrap");
        let path = dir.path().join("test.txt");
        fs::write(&path, "hello world").expect("TODO: Verify unwrap");

        let hash = hash_file(&path).expect("TODO: Verify unwrap");

        // Known SHA-256 of "hello world"
        assert_eq!(hash, "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
    }

    #[test]
    fn test_duplicate_stats() {
        let dir = create_test_dir();

        let config = DuplicateConfig {
            root_path: dir.path().to_path_buf(),
            min_size: 0,
            include_matcher: None,
            exclude_matcher: None,
        };

        let stats = find_duplicates_stats_impl(&config);

        assert!(stats.status.is_success());
        assert!(stats.files_scanned >= 5);
        assert_eq!(stats.duplicate_groups, 1);
    }
}
