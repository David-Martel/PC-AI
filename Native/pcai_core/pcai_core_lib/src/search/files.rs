//! Fast file search with glob pattern matching.
//!
//! Uses `ignore` crate for parallel directory walking and `globset` for
//! efficient pattern matching.

use std::ffi::CStr;
use std::os::raw::c_char;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use globset::{Glob, GlobMatcher};
use serde::{Deserialize, Serialize};

use crate::search::walker::{run_walker, WalkerConfig};

use crate::path::parse_path_ffi;
use crate::string::{json_to_buffer, PcaiStringBuffer};
use crate::PcaiStatus;

/// Statistics returned by file search operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FileSearchStats {
    /// Operation status
    pub status: PcaiStatus,
    /// Total files scanned
    pub files_scanned: u64,
    /// Number of files matched
    pub files_matched: u64,
    /// Total size of matched files in bytes
    pub total_size: u64,
    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
}

impl FileSearchStats {
    fn error(status: PcaiStatus) -> Self {
        Self {
            status,
            ..Default::default()
        }
    }
}

/// Information about a found file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoundFile {
    /// Full path to the file
    pub path: String,
    /// File size in bytes
    pub size: u64,
    /// Last modified timestamp (Unix epoch seconds)
    pub modified: u64,
    /// Whether the file is read-only
    pub readonly: bool,
}

/// Complete result of a file search operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSearchResult {
    /// Operation status (as string for JSON)
    pub status: String,
    /// Pattern used for search
    pub pattern: String,
    /// Total files scanned
    pub files_scanned: u64,
    /// Number of files matched
    pub files_matched: u64,
    /// Total size of matched files
    pub total_size: u64,
    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
    /// Matched files (may be truncated by max_results)
    pub files: Vec<FoundFile>,
    /// Whether results were truncated
    pub truncated: bool,
}

/// Configuration for file search.
struct FileSearchConfig {
    root_path: PathBuf,
    pattern: String,
    matcher: GlobMatcher,
    max_results: u64,
}

impl FileSearchConfig {
    fn from_ffi(
        root_path: *const c_char,
        pattern: *const c_char,
        max_results: u64,
    ) -> Result<Self, PcaiStatus> {
        // Parse root path with cross-platform normalization
        let root = parse_path_ffi(root_path)?;

        if !root.exists() {
            return Err(PcaiStatus::PathNotFound);
        }

        // Parse pattern (required)
        if pattern.is_null() {
            return Err(PcaiStatus::NullPointer);
        }

        let c_str = unsafe { CStr::from_ptr(pattern) };
        let pattern_str = c_str.to_str().map_err(|_| PcaiStatus::InvalidUtf8)?;

        if pattern_str.is_empty() {
            return Err(PcaiStatus::InvalidArgument);
        }

        let glob = Glob::new(pattern_str).map_err(|_| PcaiStatus::InvalidArgument)?;

        Ok(Self {
            root_path: root,
            pattern: pattern_str.to_string(),
            matcher: glob.compile_matcher(),
            max_results,
        })
    }
}

/// Searches for files matching the pattern.
fn find_files_impl(config: &FileSearchConfig) -> FileSearchResult {
    let start = Instant::now();

    // Wrap shared state in Arc for thread-safe cloning
    let files_matched = Arc::new(AtomicU64::new(0));
    let total_size = Arc::new(AtomicU64::new(0));
    let found_files = Arc::new(Mutex::new(Vec::new()));
    let truncated = Arc::new(AtomicBool::new(false));

    let walker_config = WalkerConfig {
        root_path: &config.root_path,
        include_patterns: vec![],
        exclude_patterns: vec![],
        git_ignore: false,
        hidden: false,
    };

    // Clone Arcs for closure
    let files_matched_clone = files_matched.clone();
    let total_size_clone = total_size.clone();
    let found_files_clone = found_files.clone();
    let truncated_clone = truncated.clone();

    // Clone config fields for closure (must be 'static / owned)
    let matcher = config.matcher.clone();
    let max_results = config.max_results;

    let stats = run_walker(walker_config, move |entry: &ignore::DirEntry| {
        if let Ok(metadata) = entry.metadata() {
             if metadata.is_file() {
                 let path = entry.path();
                 if matcher.is_match(path) || matcher.is_match(path.file_name().unwrap_or_default()) {
                      let current = files_matched_clone.fetch_add(1, Ordering::Relaxed);
                      if max_results > 0 && current >= max_results {
                          truncated_clone.store(true, Ordering::Relaxed);
                          return ignore::WalkState::Quit;
                      }

                      let size = metadata.len();
                      total_size_clone.fetch_add(size, Ordering::Relaxed);

                      let modified = metadata.modified().ok().and_then(|t: std::time::SystemTime| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d: std::time::Duration| d.as_secs()).unwrap_or(0);
                      let readonly = metadata.permissions().readonly();

                      let file_info = FoundFile {
                          path: path.to_string_lossy().into_owned(),
                          size,
                          modified,
                          readonly
                      };
                      found_files_clone.lock().expect("TODO: Verify unwrap").push(file_info);
                 }
             }
        }
        ignore::WalkState::Continue
    });

    let elapsed = start.elapsed();
    let mut files = std::mem::take(&mut *found_files.lock().expect("TODO: Verify unwrap"));
    files.sort_by(|a, b| a.path.cmp(&b.path));

    FileSearchResult {
        status: "Success".to_string(),
        pattern: config.pattern.clone(),
        files_scanned: stats.files_scanned.load(Ordering::Relaxed),
        files_matched: files_matched.load(Ordering::Relaxed),
        total_size: total_size.load(Ordering::Relaxed),
        elapsed_ms: elapsed.as_millis() as u64,
        files,
        truncated: truncated.load(Ordering::Relaxed),
    }
}

/// Returns only statistics without the file list.
fn find_files_stats_impl(config: &FileSearchConfig) -> FileSearchStats {
    let start = Instant::now();

    // Wrap shared state in Arc
    let files_matched = Arc::new(AtomicU64::new(0));
    let total_size = Arc::new(AtomicU64::new(0));

    let walker_config = WalkerConfig {
        root_path: &config.root_path,
        include_patterns: vec![],
        exclude_patterns: vec![],
        git_ignore: false,
        hidden: false,
    };

    // Clone Arcs for closure
    let files_matched_clone = files_matched.clone();
    let total_size_clone = total_size.clone();

    // Clone config fields
    let matcher = config.matcher.clone();
    // max_results unused? No, needed for logic?
    // find_files_stats usually runs until completion (stats), but can optimize if we have limit?
    // Original NukeNul stats didn't limit?
    // But files.rs logic for stats (lines 211 in previous) didn't use max_results check.
    // So I can omit max_results.

    let stats = run_walker(walker_config, move |entry: &ignore::DirEntry| {
        if let Ok(metadata) = entry.metadata() {
            if metadata.is_file() {
                 let path = entry.path();
                 if matcher.is_match(path) || matcher.is_match(path.file_name().unwrap_or_default()) {
                     files_matched_clone.fetch_add(1, Ordering::Relaxed);
                     total_size_clone.fetch_add(metadata.len(), Ordering::Relaxed);
                 }
            }
        }
        ignore::WalkState::Continue
    });

    let elapsed = start.elapsed();

    FileSearchStats {
        status: PcaiStatus::Success,
        files_scanned: stats.files_scanned.load(Ordering::Relaxed),
        files_matched: files_matched.load(Ordering::Relaxed),
        total_size: total_size.load(Ordering::Relaxed),
        elapsed_ms: elapsed.as_millis() as u64,
    }
}

// ============================================================================
// FFI Entry Points
// ============================================================================

/// FFI entry point for file search with full JSON result.
pub fn find_files_ffi(
    root_path: *const c_char,
    pattern: *const c_char,
    max_results: u64,
) -> PcaiStringBuffer {
    match FileSearchConfig::from_ffi(root_path, pattern, max_results) {
        Ok(config) => {
            let result = find_files_impl(&config);
            json_to_buffer(&result)
        }
        Err(status) => PcaiStringBuffer::error(status),
    }
}

/// FFI entry point for file search with stats only.
pub fn find_files_stats_ffi(
    root_path: *const c_char,
    pattern: *const c_char,
    max_results: u64,
) -> FileSearchStats {
    match FileSearchConfig::from_ffi(root_path, pattern, max_results) {
        Ok(config) => find_files_stats_impl(&config),
        Err(status) => FileSearchStats::error(status),
    }
}

pub mod tests {
    // Tests omitted
}
