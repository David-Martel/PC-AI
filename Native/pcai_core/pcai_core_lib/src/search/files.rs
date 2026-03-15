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
use crate::string::{bytes_to_buffer, json_to_buffer, PcaiByteBuffer, PcaiStringBuffer};
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

/// Statistics returned by directory manifest operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DirectoryManifestStats {
    pub status: PcaiStatus,
    pub entries_returned: u64,
    pub file_count: u64,
    pub directory_count: u64,
    pub total_size: u64,
    pub elapsed_ms: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FileSearchCompactHeader {
    pub status: PcaiStatus,
    pub reserved: u32,
    pub files_scanned: u64,
    pub files_matched: u64,
    pub total_size: u64,
    pub elapsed_ms: u64,
    pub entry_count: u64,
    pub string_bytes: u64,
    pub truncated: u8,
    pub _padding: [u8; 7],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct FileSearchCompactEntry {
    pub path_offset: u32,
    pub path_length: u32,
    pub size: u64,
    pub modified: u64,
    pub readonly: u8,
    pub _padding: [u8; 7],
}

impl DirectoryManifestStats {
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

/// Information about a manifest entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryManifestEntry {
    pub path: String,
    pub relative_path: String,
    pub entry_type: String,
    pub extension: String,
    pub depth: u32,
    pub size: u64,
    pub modified: u64,
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

/// Complete result of a directory manifest operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryManifestResult {
    pub status: String,
    pub root_path: String,
    pub max_depth: u32,
    pub entries_returned: u64,
    pub file_count: u64,
    pub directory_count: u64,
    pub total_size: u64,
    pub elapsed_ms: u64,
    pub entries: Vec<DirectoryManifestEntry>,
    pub truncated: bool,
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

fn pack_file_search_result(result: &FileSearchResult) -> Result<PcaiByteBuffer, PcaiStatus> {
    let mut string_data = Vec::new();
    let mut entries = Vec::with_capacity(result.files.len());

    for file in &result.files {
        let path_bytes = file.path.as_bytes();
        let path_offset = usize_to_u32(string_data.len())?;
        string_data.extend_from_slice(path_bytes);
        let path_length = usize_to_u32(path_bytes.len())?;

        entries.push(FileSearchCompactEntry {
            path_offset,
            path_length,
            size: file.size,
            modified: file.modified,
            readonly: if file.readonly { 1 } else { 0 },
            _padding: [0; 7],
        });
    }

    let header = FileSearchCompactHeader {
        status: PcaiStatus::Success,
        reserved: 0,
        files_scanned: result.files_scanned,
        files_matched: result.files_matched,
        total_size: result.total_size,
        elapsed_ms: result.elapsed_ms,
        entry_count: entries.len() as u64,
        string_bytes: string_data.len() as u64,
        truncated: if result.truncated { 1 } else { 0 },
        _padding: [0; 7],
    };

    let mut packed = Vec::with_capacity(
        std::mem::size_of::<FileSearchCompactHeader>()
            + (entries.len() * std::mem::size_of::<FileSearchCompactEntry>())
            + string_data.len(),
    );
    append_pod(&mut packed, &header);
    append_pod_slice(&mut packed, &entries);
    packed.extend_from_slice(&string_data);
    Ok(bytes_to_buffer(packed))
}

/// Configuration for file search.
struct FileSearchConfig {
    root_path: PathBuf,
    pattern: String,
    matcher: GlobMatcher,
    max_results: u64,
}

struct DirectoryManifestConfig {
    root_path: PathBuf,
    max_depth: u32,
    max_results: u64,
}

impl DirectoryManifestConfig {
    fn from_ffi(root_path: *const c_char, max_depth: u32, max_results: u64) -> Result<Self, PcaiStatus> {
        let root = parse_path_ffi(root_path)?;
        if !root.exists() {
            return Err(PcaiStatus::PathNotFound);
        }

        Ok(Self {
            root_path: root,
            max_depth,
            max_results,
        })
    }
}

impl FileSearchConfig {
    fn from_ffi(root_path: *const c_char, pattern: *const c_char, max_results: u64) -> Result<Self, PcaiStatus> {
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
        let path = entry.path();
        if !(matcher.is_match(path) || matcher.is_match(path.file_name().unwrap_or_default())) {
            return ignore::WalkState::Continue;
        }

        if let Ok(metadata) = entry.metadata() {
            if metadata.is_file() {
                let current = files_matched_clone.fetch_add(1, Ordering::Relaxed);
                if max_results > 0 && current >= max_results {
                    truncated_clone.store(true, Ordering::Relaxed);
                    return ignore::WalkState::Quit;
                }

                let size = metadata.len();
                total_size_clone.fetch_add(size, Ordering::Relaxed);

                let modified = metadata
                    .modified()
                    .ok()
                    .and_then(|t: std::time::SystemTime| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d: std::time::Duration| d.as_secs())
                    .unwrap_or(0);
                let readonly = metadata.permissions().readonly();

                let file_info = FoundFile {
                    path: path.to_string_lossy().into_owned(),
                    size,
                    modified,
                    readonly,
                };
                found_files_clone
                    .lock()
                    .expect("found files mutex poisoned")
                    .push(file_info);
            }
        }
        ignore::WalkState::Continue
    });

    let elapsed = start.elapsed();
    let mut files = std::mem::take(&mut *found_files.lock().expect("found files mutex poisoned"));
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

    let stats = run_walker(walker_config, move |entry: &ignore::DirEntry| {
        let path = entry.path();
        if !(matcher.is_match(path) || matcher.is_match(path.file_name().unwrap_or_default())) {
            return ignore::WalkState::Continue;
        }

        if let Ok(metadata) = entry.metadata() {
            if metadata.is_file() {
                files_matched_clone.fetch_add(1, Ordering::Relaxed);
                total_size_clone.fetch_add(metadata.len(), Ordering::Relaxed);
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

fn collect_directory_manifest_impl(config: &DirectoryManifestConfig) -> DirectoryManifestResult {
    let start = Instant::now();

    let file_count = Arc::new(AtomicU64::new(0));
    let directory_count = Arc::new(AtomicU64::new(0));
    let total_size = Arc::new(AtomicU64::new(0));
    let truncated = Arc::new(AtomicBool::new(false));
    let entries = Arc::new(Mutex::new(Vec::new()));

    let walker_config = WalkerConfig {
        root_path: &config.root_path,
        include_patterns: vec![],
        exclude_patterns: vec![],
        git_ignore: false,
        hidden: false,
    };

    let root_path = config.root_path.clone();
    let max_depth = config.max_depth;
    let max_results = config.max_results;
    let file_count_clone = file_count.clone();
    let directory_count_clone = directory_count.clone();
    let total_size_clone = total_size.clone();
    let truncated_clone = truncated.clone();
    let entries_clone = entries.clone();

    run_walker(walker_config, move |entry: &ignore::DirEntry| {
        let depth = entry.depth() as u32;
        if depth == 0 {
            return ignore::WalkState::Continue;
        }
        if max_depth > 0 && depth > max_depth {
            return ignore::WalkState::Continue;
        }

        if let Ok(metadata) = entry.metadata() {
            let is_dir = metadata.is_dir();
            let current = {
                let guard = entries_clone.lock().expect("directory manifest lock poisoned");
                guard.len() as u64
            };

            if max_results > 0 && current >= max_results {
                truncated_clone.store(true, Ordering::Relaxed);
                return ignore::WalkState::Quit;
            }

            if is_dir {
                directory_count_clone.fetch_add(1, Ordering::Relaxed);
            } else if metadata.is_file() {
                file_count_clone.fetch_add(1, Ordering::Relaxed);
                total_size_clone.fetch_add(metadata.len(), Ordering::Relaxed);
            }

            let relative_path = entry
                .path()
                .strip_prefix(&root_path)
                .ok()
                .map(|path| path.to_string_lossy().replace('\\', "/"))
                .unwrap_or_else(|| entry.path().to_string_lossy().replace('\\', "/"));

            let extension = entry
                .path()
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or_default()
                .to_string();

            let modified = metadata
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);

            let manifest_entry = DirectoryManifestEntry {
                path: entry.path().to_string_lossy().into_owned(),
                relative_path,
                entry_type: if is_dir {
                    "directory".to_string()
                } else {
                    "file".to_string()
                },
                extension,
                depth,
                size: if metadata.is_file() { metadata.len() } else { 0 },
                modified,
                readonly: metadata.permissions().readonly(),
            };

            entries_clone
                .lock()
                .expect("directory manifest lock poisoned")
                .push(manifest_entry);
        }

        ignore::WalkState::Continue
    });

    let elapsed = start.elapsed();
    let mut manifest_entries = std::mem::take(&mut *entries.lock().expect("directory manifest lock poisoned"));
    manifest_entries.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));

    DirectoryManifestResult {
        status: "Success".to_string(),
        root_path: config.root_path.to_string_lossy().into_owned(),
        max_depth: config.max_depth,
        entries_returned: manifest_entries.len() as u64,
        file_count: file_count.load(Ordering::Relaxed),
        directory_count: directory_count.load(Ordering::Relaxed),
        total_size: total_size.load(Ordering::Relaxed),
        elapsed_ms: elapsed.as_millis() as u64,
        entries: manifest_entries,
        truncated: truncated.load(Ordering::Relaxed),
    }
}

fn collect_directory_manifest_stats_impl(config: &DirectoryManifestConfig) -> DirectoryManifestStats {
    let start = Instant::now();
    let file_count = Arc::new(AtomicU64::new(0));
    let directory_count = Arc::new(AtomicU64::new(0));
    let total_size = Arc::new(AtomicU64::new(0));
    let entries_returned = Arc::new(AtomicU64::new(0));

    let walker_config = WalkerConfig {
        root_path: &config.root_path,
        include_patterns: vec![],
        exclude_patterns: vec![],
        git_ignore: false,
        hidden: false,
    };

    let max_depth = config.max_depth;
    let max_results = config.max_results;
    let file_count_clone = file_count.clone();
    let directory_count_clone = directory_count.clone();
    let total_size_clone = total_size.clone();
    let entries_returned_clone = entries_returned.clone();

    run_walker(walker_config, move |entry: &ignore::DirEntry| {
        let depth = entry.depth() as u32;
        if depth == 0 {
            return ignore::WalkState::Continue;
        }
        if max_depth > 0 && depth > max_depth {
            return ignore::WalkState::Continue;
        }

        let current = entries_returned_clone.load(Ordering::Relaxed);
        if max_results > 0 && current >= max_results {
            return ignore::WalkState::Quit;
        }

        if let Ok(metadata) = entry.metadata() {
            entries_returned_clone.fetch_add(1, Ordering::Relaxed);
            if metadata.is_dir() {
                directory_count_clone.fetch_add(1, Ordering::Relaxed);
            } else if metadata.is_file() {
                file_count_clone.fetch_add(1, Ordering::Relaxed);
                total_size_clone.fetch_add(metadata.len(), Ordering::Relaxed);
            }
        }

        ignore::WalkState::Continue
    });

    DirectoryManifestStats {
        status: PcaiStatus::Success,
        entries_returned: entries_returned.load(Ordering::Relaxed),
        file_count: file_count.load(Ordering::Relaxed),
        directory_count: directory_count.load(Ordering::Relaxed),
        total_size: total_size.load(Ordering::Relaxed),
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

// ============================================================================
// FFI Entry Points
// ============================================================================

/// FFI entry point for file search with full JSON result.
pub fn find_files_ffi(root_path: *const c_char, pattern: *const c_char, max_results: u64) -> PcaiStringBuffer {
    match FileSearchConfig::from_ffi(root_path, pattern, max_results) {
        Ok(config) => {
            let result = find_files_impl(&config);
            json_to_buffer(&result)
        }
        Err(status) => PcaiStringBuffer::error(status),
    }
}

/// FFI entry point for file search with stats only.
pub fn find_files_stats_ffi(root_path: *const c_char, pattern: *const c_char, max_results: u64) -> FileSearchStats {
    match FileSearchConfig::from_ffi(root_path, pattern, max_results) {
        Ok(config) => find_files_stats_impl(&config),
        Err(status) => FileSearchStats::error(status),
    }
}

/// FFI entry point for file search with compact binary result.
pub fn find_files_compact_ffi(root_path: *const c_char, pattern: *const c_char, max_results: u64) -> PcaiByteBuffer {
    match FileSearchConfig::from_ffi(root_path, pattern, max_results) {
        Ok(config) => {
            let result = find_files_impl(&config);
            match pack_file_search_result(&result) {
                Ok(buffer) => buffer,
                Err(status) => PcaiByteBuffer::error(status),
            }
        }
        Err(status) => PcaiByteBuffer::error(status),
    }
}

/// FFI entry point for directory manifest collection with full JSON result.
pub fn collect_directory_manifest_ffi(root_path: *const c_char, max_depth: u32, max_results: u64) -> PcaiStringBuffer {
    match DirectoryManifestConfig::from_ffi(root_path, max_depth, max_results) {
        Ok(config) => {
            let result = collect_directory_manifest_impl(&config);
            json_to_buffer(&result)
        }
        Err(status) => PcaiStringBuffer::error(status),
    }
}

/// FFI entry point for directory manifest collection with stats only.
pub fn collect_directory_manifest_stats_ffi(
    root_path: *const c_char,
    max_depth: u32,
    max_results: u64,
) -> DirectoryManifestStats {
    match DirectoryManifestConfig::from_ffi(root_path, max_depth, max_results) {
        Ok(config) => collect_directory_manifest_stats_impl(&config),
        Err(status) => DirectoryManifestStats::error(status),
    }
}

pub mod tests {
    // Tests omitted
}
