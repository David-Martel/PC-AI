//! Parallel content search with regex support.
//!
//! Uses `ignore` crate for fast parallel file walking and `regex` for
//! pattern matching within file contents.

use std::ffi::CStr;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use globset::{Glob, GlobMatcher};
use ignore::{DirEntry, ParallelVisitor, ParallelVisitorBuilder, WalkBuilder, WalkState};
use regex::{bytes::Regex as BytesRegex, Regex};
use serde::{Deserialize, Serialize};

use crate::path::parse_path_ffi;
use crate::string::{bytes_to_buffer, json_to_buffer, PcaiByteBuffer, PcaiStringBuffer};
use crate::PcaiStatus;

/// Statistics returned by content search operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ContentSearchStats {
    /// Operation status
    pub status: PcaiStatus,
    /// Total files scanned
    pub files_scanned: u64,
    /// Files with matches
    pub files_matched: u64,
    /// Total number of matches
    pub total_matches: u64,
    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ContentSearchCompactHeader {
    pub status: PcaiStatus,
    pub reserved: u32,
    pub files_scanned: u64,
    pub files_matched: u64,
    pub total_matches: u64,
    pub elapsed_ms: u64,
    pub entry_count: u64,
    pub string_bytes: u64,
    pub truncated: u8,
    pub _padding: [u8; 7],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ContentSearchCompactEntry {
    pub path_offset: u32,
    pub path_length: u32,
    pub line_offset: u32,
    pub line_length: u32,
    pub line_number: u64,
}

impl ContentSearchStats {
    fn error(status: PcaiStatus) -> Self {
        Self {
            status,
            ..Default::default()
        }
    }
}

/// A single match within a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentMatch {
    /// Path to the file containing the match
    pub path: String,
    /// Line number (1-indexed)
    pub line_number: u64,
    /// The matched line content
    pub line: String,
    /// Context lines before the match (if requested)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub before: Vec<String>,
    /// Context lines after the match (if requested)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub after: Vec<String>,
}

/// Complete result of a content search operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSearchResult {
    /// Operation status (as string for JSON)
    pub status: String,
    /// Regex pattern used
    pub pattern: String,
    /// File pattern used (if any)
    pub file_pattern: Option<String>,
    /// Total files scanned
    pub files_scanned: u64,
    /// Files with matches
    pub files_matched: u64,
    /// Total number of matches
    pub total_matches: u64,
    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
    /// All matches found
    pub matches: Vec<ContentMatch>,
    /// Whether results were truncated
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

fn pack_content_search_result(result: &ContentSearchResult) -> Result<PcaiByteBuffer, PcaiStatus> {
    let mut string_data = Vec::new();
    let mut entries = Vec::with_capacity(result.matches.len());

    for entry in &result.matches {
        let path_bytes = entry.path.as_bytes();
        let path_offset = usize_to_u32(string_data.len())?;
        string_data.extend_from_slice(path_bytes);
        let path_length = usize_to_u32(path_bytes.len())?;

        let line_bytes = entry.line.as_bytes();
        let line_offset = usize_to_u32(string_data.len())?;
        string_data.extend_from_slice(line_bytes);
        let line_length = usize_to_u32(line_bytes.len())?;

        entries.push(ContentSearchCompactEntry {
            path_offset,
            path_length,
            line_offset,
            line_length,
            line_number: entry.line_number,
        });
    }

    let header = ContentSearchCompactHeader {
        status: PcaiStatus::Success,
        reserved: 0,
        files_scanned: result.files_scanned,
        files_matched: result.files_matched,
        total_matches: result.total_matches,
        elapsed_ms: result.elapsed_ms,
        entry_count: entries.len() as u64,
        string_bytes: string_data.len() as u64,
        truncated: if result.truncated { 1 } else { 0 },
        _padding: [0; 7],
    };

    let mut packed = Vec::with_capacity(
        std::mem::size_of::<ContentSearchCompactHeader>()
            + (entries.len() * std::mem::size_of::<ContentSearchCompactEntry>())
            + string_data.len(),
    );
    append_pod(&mut packed, &header);
    append_pod_slice(&mut packed, &entries);
    packed.extend_from_slice(&string_data);
    Ok(bytes_to_buffer(packed))
}

/// Configuration for content search.
struct ContentSearchConfig {
    root_path: PathBuf,
    regex: Regex,
    bytes_regex: BytesRegex,
    pattern_str: String,
    file_matcher: Option<GlobMatcher>,
    file_pattern_str: Option<String>,
    max_results: u64,
    context_lines: u32,
}

impl ContentSearchConfig {
    fn from_ffi(
        root_path: *const c_char,
        pattern: *const c_char,
        file_pattern: *const c_char,
        max_results: u64,
        context_lines: u32,
    ) -> Result<Self, PcaiStatus> {
        // Parse root path with cross-platform normalization
        let root = parse_path_ffi(root_path)?;

        if !root.exists() {
            return Err(PcaiStatus::PathNotFound);
        }

        // Parse regex pattern (required)
        if pattern.is_null() {
            return Err(PcaiStatus::NullPointer);
        }

        let c_str = unsafe { CStr::from_ptr(pattern) };
        let pattern_str = c_str.to_str().map_err(|_| PcaiStatus::InvalidUtf8)?;

        if pattern_str.is_empty() {
            return Err(PcaiStatus::InvalidArgument);
        }

        let regex = Regex::new(pattern_str).map_err(|_| PcaiStatus::InvalidArgument)?;
        let bytes_regex = BytesRegex::new(pattern_str).map_err(|_| PcaiStatus::InvalidArgument)?;

        // Parse file pattern (optional)
        let (file_matcher, file_pattern_str) = if !file_pattern.is_null() {
            let c_str = unsafe { CStr::from_ptr(file_pattern) };
            let pattern = c_str.to_str().map_err(|_| PcaiStatus::InvalidUtf8)?;
            if !pattern.is_empty() {
                let glob = Glob::new(pattern).map_err(|_| PcaiStatus::InvalidArgument)?;
                (Some(glob.compile_matcher()), Some(pattern.to_string()))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        Ok(Self {
            root_path: root,
            regex,
            bytes_regex,
            pattern_str: pattern_str.to_string(),
            file_matcher,
            file_pattern_str,
            max_results,
            context_lines,
        })
    }
}

const DEFAULT_TEXT_EXTENSIONS: &[&str] = &[
    "txt", "log", "md", "json", "xml", "yaml", "yml", "toml", "ini", "cfg", "conf", "config", "ps1", "psm1", "psd1",
    "bat", "cmd", "sh", "bash", "py", "rs", "js", "ts", "jsx", "tsx", "cs", "cpp", "c", "h", "hpp", "java", "go", "rb",
    "php", "html", "htm", "css", "scss", "sass", "less", "sql", "graphql", "proto",
];

fn should_search_file_path(path: &Path, matcher: Option<&GlobMatcher>) -> bool {
    if let Some(matcher) = matcher {
        return matcher.is_match(path) || matcher.is_match(path.file_name().unwrap_or_default());
    }

    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            DEFAULT_TEXT_EXTENSIONS
                .iter()
                .any(|candidate| ext.eq_ignore_ascii_case(candidate))
        })
        .unwrap_or(false)
}

#[derive(Clone)]
struct ContentSearchWorkerConfig {
    regex: Regex,
    bytes_regex: BytesRegex,
    file_matcher: Option<GlobMatcher>,
    max_results: u64,
    context_lines: u32,
    collect_matches: bool,
}

impl ContentSearchWorkerConfig {
    fn from_config(config: &ContentSearchConfig, collect_matches: bool) -> Self {
        Self {
            regex: config.regex.clone(),
            bytes_regex: config.bytes_regex.clone(),
            file_matcher: config.file_matcher.clone(),
            max_results: config.max_results,
            context_lines: config.context_lines,
            collect_matches,
        }
    }
}

#[derive(Default)]
struct ContentSearchShared {
    files_scanned: AtomicU64,
    files_matched: AtomicU64,
    total_matches: AtomicU64,
    stored_matches: AtomicU64,
    matches: Mutex<Vec<ContentMatch>>,
    truncated: AtomicBool,
}

struct ContentSearchVisitorBuilder {
    shared: Arc<ContentSearchShared>,
    config: ContentSearchWorkerConfig,
}

impl<'s> ParallelVisitorBuilder<'s> for ContentSearchVisitorBuilder {
    fn build(&mut self) -> Box<dyn ParallelVisitor + 's> {
        Box::new(ContentSearchVisitor {
            shared: self.shared.clone(),
            config: self.config.clone(),
            local_files_scanned: 0,
            local_files_matched: 0,
            local_total_matches: 0,
            local_matches: Vec::new(),
        })
    }
}

struct ContentSearchVisitor {
    shared: Arc<ContentSearchShared>,
    config: ContentSearchWorkerConfig,
    local_files_scanned: u64,
    local_files_matched: u64,
    local_total_matches: u64,
    local_matches: Vec<ContentMatch>,
}

impl ContentSearchVisitor {
    fn reserve_result_slots(&self, requested: usize) -> usize {
        if !self.config.collect_matches || requested == 0 {
            return 0;
        }

        if self.config.max_results == 0 {
            return requested;
        }

        loop {
            let current = self.shared.stored_matches.load(Ordering::Relaxed);
            if current >= self.config.max_results {
                return 0;
            }

            let remaining = (self.config.max_results - current) as usize;
            let granted = remaining.min(requested);
            if self
                .shared
                .stored_matches
                .compare_exchange_weak(current, current + granted as u64, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return granted;
            }
        }
    }

    fn flush_local_matches(&mut self) {
        if self.local_matches.is_empty() {
            return;
        }

        let mut shared_matches = self
            .shared
            .matches
            .lock()
            .expect("content search matches mutex poisoned");
        shared_matches.append(&mut self.local_matches);
    }
}

impl Drop for ContentSearchVisitor {
    fn drop(&mut self) {
        self.shared
            .files_scanned
            .fetch_add(self.local_files_scanned, Ordering::Relaxed);
        self.shared
            .files_matched
            .fetch_add(self.local_files_matched, Ordering::Relaxed);
        self.shared
            .total_matches
            .fetch_add(self.local_total_matches, Ordering::Relaxed);
        self.flush_local_matches();
    }
}

impl ParallelVisitor for ContentSearchVisitor {
    fn visit(&mut self, result: Result<DirEntry, ignore::Error>) -> WalkState {
        if self.shared.truncated.load(Ordering::Relaxed) {
            return WalkState::Quit;
        }

        let entry = match result {
            Ok(entry) => entry,
            Err(_) => return WalkState::Continue,
        };

        let file_type = match entry.file_type() {
            Some(file_type) if file_type.is_file() => file_type,
            _ => return WalkState::Continue,
        };
        let _ = file_type;

        let path = entry.path();
        if !should_search_file_path(path, self.config.file_matcher.as_ref()) {
            return WalkState::Continue;
        }

        self.local_files_scanned += 1;

        let outcome = match search_file_streaming(path, &self.config) {
            Ok(outcome) => outcome,
            Err(_) => return WalkState::Continue,
        };
        if outcome.total_matches == 0 {
            return WalkState::Continue;
        }

        self.local_files_matched += 1;
        self.local_total_matches += outcome.total_matches;

        if self.config.collect_matches {
            let collected_count = outcome.matches.len();
            let allowed = self.reserve_result_slots(collected_count);
            if allowed > 0 {
                self.local_matches.extend(outcome.matches.into_iter().take(allowed));
            }

            if allowed < collected_count {
                self.shared.truncated.store(true, Ordering::Relaxed);
                self.flush_local_matches();
                return WalkState::Quit;
            }

            if self.local_matches.len() >= 256 {
                self.flush_local_matches();
            }
        }

        WalkState::Continue
    }
}

struct SearchFileOutcome {
    total_matches: u64,
    matches: Vec<ContentMatch>,
}

struct ContentSearchExecution {
    files_scanned: u64,
    files_matched: u64,
    total_matches: u64,
    matches: Vec<ContentMatch>,
    truncated: bool,
    elapsed_ms: u64,
}

fn execute_content_search(config: &ContentSearchConfig, collect_matches: bool) -> ContentSearchExecution {
    let start = Instant::now();
    let shared = Arc::new(ContentSearchShared::default());

    let mut builder = WalkBuilder::new(&config.root_path);
    builder
        .hidden(false)
        .git_ignore(false)
        .git_global(false)
        .git_exclude(false);

    let mut visitor_builder = ContentSearchVisitorBuilder {
        shared: shared.clone(),
        config: ContentSearchWorkerConfig::from_config(config, collect_matches),
    };
    builder.build_parallel().visit(&mut visitor_builder);

    let mut matches = std::mem::take(&mut *shared.matches.lock().unwrap_or_else(|poisoned| poisoned.into_inner()));
    matches.sort_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then(left.line_number.cmp(&right.line_number))
            .then(left.line.cmp(&right.line))
    });

    ContentSearchExecution {
        files_scanned: shared.files_scanned.load(Ordering::Relaxed),
        files_matched: shared.files_matched.load(Ordering::Relaxed),
        total_matches: shared.total_matches.load(Ordering::Relaxed),
        matches,
        truncated: shared.truncated.load(Ordering::Relaxed),
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

/// Searches file contents for matches using parallel processing.
fn search_content_impl(config: &ContentSearchConfig) -> ContentSearchResult {
    let execution = execute_content_search(config, true);

    ContentSearchResult {
        status: "Success".to_string(),
        pattern: config.pattern_str.clone(),
        file_pattern: config.file_pattern_str.clone(),
        files_scanned: execution.files_scanned,
        files_matched: execution.files_matched,
        total_matches: execution.total_matches,
        elapsed_ms: execution.elapsed_ms,
        matches: execution.matches,
        truncated: execution.truncated,
    }
}

/// Searches a single file for matches using streaming to avoid memory explosion.
/// Uses a rolling buffer for context lines to minimize memory usage.
fn trim_line_endings(buffer: &mut Vec<u8>) {
    while matches!(buffer.last(), Some(b'\n' | b'\r')) {
        buffer.pop();
    }
}

fn search_file_streaming(path: &Path, config: &ContentSearchWorkerConfig) -> std::io::Result<SearchFileOutcome> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(64 * 1024, file); // 64KB buffer
    let mut matches = Vec::new();
    let mut total_matches = 0u64;

    let context_size = config.context_lines as usize;
    let path_str = path.to_string_lossy().into_owned();

    if context_size == 0 {
        // Fast path: reuse a single byte buffer and only decode matched lines.
        let mut line_number = 0u64;
        let mut line_buffer = Vec::with_capacity(1024);
        loop {
            line_buffer.clear();
            let bytes_read = reader.read_until(b'\n', &mut line_buffer)?;
            if bytes_read == 0 {
                break;
            }

            line_number += 1;
            trim_line_endings(&mut line_buffer);
            if config.bytes_regex.is_match(&line_buffer) {
                total_matches += 1;
                if config.collect_matches {
                    let line = match String::from_utf8(line_buffer.clone()) {
                        Ok(line) => line,
                        Err(_) => String::from_utf8_lossy(&line_buffer).into_owned(),
                    };
                    matches.push(ContentMatch {
                        path: path_str.clone(),
                        line_number,
                        line,
                        before: Vec::new(),
                        after: Vec::new(),
                    });
                }
            }
        }
    } else {
        // Context path: use rolling buffer for before context
        // Read all lines since we need forward context (after)
        let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;

        for (idx, line) in lines.iter().enumerate() {
            if config.regex.is_match(line) {
                total_matches += 1;
                if config.collect_matches {
                    let before: Vec<String> = {
                        let start = idx.saturating_sub(context_size);
                        lines[start..idx].to_vec()
                    };

                    let after: Vec<String> = {
                        let end = (idx + 1 + context_size).min(lines.len());
                        lines[idx + 1..end].to_vec()
                    };

                    matches.push(ContentMatch {
                        path: path_str.clone(),
                        line_number: idx as u64 + 1,
                        line: line.clone(),
                        before,
                        after,
                    });
                }
            }
        }
    }

    Ok(SearchFileOutcome { total_matches, matches })
}

/// Legacy function for backward compatibility in tests.
#[expect(
    dead_code,
    reason = "legacy compatibility shim retained for test callers; not part of the public API"
)]
fn search_file(path: &Path, config: &ContentSearchConfig) -> std::io::Result<Vec<ContentMatch>> {
    let worker_config = ContentSearchWorkerConfig::from_config(config, true);
    Ok(search_file_streaming(path, &worker_config)?.matches)
}

/// Returns only statistics without the match list.
fn search_content_stats_impl(config: &ContentSearchConfig) -> ContentSearchStats {
    let result = execute_content_search(config, false);

    ContentSearchStats {
        status: PcaiStatus::Success,
        files_scanned: result.files_scanned,
        files_matched: result.files_matched,
        total_matches: result.total_matches,
        elapsed_ms: result.elapsed_ms,
    }
}

// ============================================================================
// FFI Entry Points
// ============================================================================

/// FFI entry point for content search with full JSON result.
pub fn search_content_ffi(
    root_path: *const c_char,
    pattern: *const c_char,
    file_pattern: *const c_char,
    max_results: u64,
    context_lines: u32,
) -> PcaiStringBuffer {
    match ContentSearchConfig::from_ffi(root_path, pattern, file_pattern, max_results, context_lines) {
        Ok(config) => {
            let result = search_content_impl(&config);
            json_to_buffer(&result)
        }
        Err(status) => PcaiStringBuffer::error(status),
    }
}

/// FFI entry point for content search with stats only.
pub fn search_content_stats_ffi(
    root_path: *const c_char,
    pattern: *const c_char,
    file_pattern: *const c_char,
    max_results: u64,
) -> ContentSearchStats {
    match ContentSearchConfig::from_ffi(root_path, pattern, file_pattern, max_results, 0) {
        Ok(config) => search_content_stats_impl(&config),
        Err(status) => ContentSearchStats::error(status),
    }
}

/// FFI entry point for content search with compact binary result.
pub fn search_content_compact_ffi(
    root_path: *const c_char,
    pattern: *const c_char,
    file_pattern: *const c_char,
    max_results: u64,
) -> PcaiByteBuffer {
    match ContentSearchConfig::from_ffi(root_path, pattern, file_pattern, max_results, 0) {
        Ok(config) => {
            let result = search_content_impl(&config);
            match pack_content_search_result(&result) {
                Ok(buffer) => buffer,
                Err(status) => PcaiByteBuffer::error(status),
            }
        }
        Err(status) => PcaiByteBuffer::error(status),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_dir() -> TempDir {
        let dir = TempDir::new().expect("failed to create temp dir");

        fs::write(dir.path().join("file1.txt"), "Hello world\nThis is a test\nHello again")
            .expect("failed to write file1.txt");
        fs::write(
            dir.path().join("file2.log"),
            "Error: something failed\nWarning: check this\nError: another failure",
        )
        .expect("failed to write file2.log");
        fs::write(dir.path().join("data.json"), r#"{"key": "value"}"#).expect("failed to write data.json");

        let subdir = dir.path().join("subdir");
        fs::create_dir(&subdir).expect("failed to create subdir");
        fs::write(subdir.join("nested.txt"), "Hello from nested").expect("failed to write nested.txt");

        dir
    }

    #[test]
    fn test_search_content_basic() {
        let dir = create_test_dir();

        let regex = Regex::new("Hello").expect("valid regex");
        let config = ContentSearchConfig {
            root_path: dir.path().to_path_buf(),
            regex,
            bytes_regex: BytesRegex::new("Hello").expect("valid bytes regex"),
            pattern_str: "Hello".to_string(),
            file_matcher: None,
            file_pattern_str: None,
            max_results: 0,
            context_lines: 0,
        };

        let result = search_content_impl(&config);

        assert_eq!(result.status, "Success");
        assert_eq!(result.total_matches, 3); // 2 in file1.txt + 1 in nested.txt
        assert_eq!(result.files_matched, 2);
    }

    #[test]
    fn test_search_content_with_file_pattern() {
        let dir = create_test_dir();

        let regex = Regex::new("Error").expect("valid regex");
        let glob = Glob::new("*.log").expect("valid glob pattern");
        let config = ContentSearchConfig {
            root_path: dir.path().to_path_buf(),
            regex,
            bytes_regex: BytesRegex::new("Error").expect("valid bytes regex"),
            pattern_str: "Error".to_string(),
            file_matcher: Some(glob.compile_matcher()),
            file_pattern_str: Some("*.log".to_string()),
            max_results: 0,
            context_lines: 0,
        };

        let result = search_content_impl(&config);

        assert_eq!(result.status, "Success");
        assert_eq!(result.total_matches, 2); // Only from .log file
        assert_eq!(result.files_matched, 1);
    }

    #[test]
    fn test_search_content_with_context() {
        let dir = create_test_dir();

        let regex = Regex::new("test").expect("valid regex");
        let config = ContentSearchConfig {
            root_path: dir.path().to_path_buf(),
            regex,
            bytes_regex: BytesRegex::new("test").expect("valid bytes regex"),
            pattern_str: "test".to_string(),
            file_matcher: None,
            file_pattern_str: None,
            max_results: 0,
            context_lines: 1,
        };

        let result = search_content_impl(&config);

        assert_eq!(result.status, "Success");
        assert_eq!(result.total_matches, 1);

        let m = &result.matches[0];
        assert_eq!(m.line, "This is a test");
        assert_eq!(m.before.len(), 1); // "Hello world"
        assert_eq!(m.after.len(), 1); // "Hello again"
    }

    #[test]
    fn test_search_content_max_results() {
        let dir = create_test_dir();

        let regex = Regex::new("Hello|Error").expect("valid regex");
        let config = ContentSearchConfig {
            root_path: dir.path().to_path_buf(),
            regex,
            bytes_regex: BytesRegex::new("Hello|Error").expect("valid bytes regex"),
            pattern_str: "Hello|Error".to_string(),
            file_matcher: None,
            file_pattern_str: None,
            max_results: 2,
            context_lines: 0,
        };

        let result = search_content_impl(&config);

        assert_eq!(result.status, "Success");
        assert!(result.matches.len() <= 2);
        assert!(result.truncated);
    }

    #[test]
    fn test_search_content_stats() {
        let dir = create_test_dir();

        let regex = Regex::new("Hello").expect("valid regex");
        let config = ContentSearchConfig {
            root_path: dir.path().to_path_buf(),
            regex,
            bytes_regex: BytesRegex::new("Hello").expect("valid bytes regex"),
            pattern_str: "Hello".to_string(),
            file_matcher: None,
            file_pattern_str: None,
            max_results: 0,
            context_lines: 0,
        };

        let stats = search_content_stats_impl(&config);

        assert!(stats.status.is_success());
        assert_eq!(stats.total_matches, 3);
        assert_eq!(stats.files_matched, 2);
    }
}
