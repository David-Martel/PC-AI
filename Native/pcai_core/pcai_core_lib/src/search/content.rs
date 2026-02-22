//! Parallel content search with regex support.
//!
//! Uses `ignore` crate for fast parallel file walking and `regex` for
//! pattern matching within file contents.

use std::ffi::CStr;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use globset::{Glob, GlobMatcher};
use ignore::WalkBuilder;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::path::parse_path_ffi;
use crate::string::{json_to_buffer, PcaiStringBuffer};
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

/// Configuration for content search.
struct ContentSearchConfig {
    root_path: PathBuf,
    regex: Regex,
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
            pattern_str: pattern_str.to_string(),
            file_matcher,
            file_pattern_str,
            max_results,
            context_lines,
        })
    }

    /// Checks if a file should be searched based on the file pattern.
    fn should_search_file(&self, path: &Path) -> bool {
        if let Some(ref matcher) = self.file_matcher {
            matcher.is_match(path)
                || matcher.is_match(path.file_name().unwrap_or_default())
        } else {
            // Default: search common text file extensions
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase();

            matches!(
                ext.as_str(),
                "txt" | "log" | "md" | "json" | "xml" | "yaml" | "yml"
                    | "toml" | "ini" | "cfg" | "conf" | "config"
                    | "ps1" | "psm1" | "psd1" | "bat" | "cmd" | "sh" | "bash"
                    | "py" | "rs" | "js" | "ts" | "jsx" | "tsx" | "cs" | "cpp"
                    | "c" | "h" | "hpp" | "java" | "go" | "rb" | "php"
                    | "html" | "htm" | "css" | "scss" | "sass" | "less"
                    | "sql" | "graphql" | "proto"
            )
        }
    }
}

/// Detects if a file is likely binary by checking the first 8KB for null bytes.
fn is_binary_file(path: &Path) -> bool {
    const SAMPLE_SIZE: usize = 8192;

    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return true, // Treat unreadable files as binary
    };

    let mut reader = BufReader::new(file);
    let mut buffer = [0u8; SAMPLE_SIZE];

    let bytes_read = match reader.read(&mut buffer) {
        Ok(n) => n,
        Err(_) => return true,
    };

    // Check for null bytes (common in binary files)
    buffer[..bytes_read].contains(&0)
}

/// Searches file contents for matches using parallel processing.
fn search_content_impl(config: &ContentSearchConfig) -> ContentSearchResult {
    let start = Instant::now();
    let files_scanned = AtomicU64::new(0);
    let files_matched = AtomicU64::new(0);
    let total_matches = AtomicU64::new(0);
    let all_matches: Mutex<Vec<ContentMatch>> = Mutex::new(Vec::new());
    let truncated = AtomicBool::new(false);

    // Collect files to search first (single-threaded for consistent ordering)
    let mut files_to_search: Vec<PathBuf> = Vec::new();

    let walker = WalkBuilder::new(&config.root_path)
        .hidden(false)
        .git_ignore(false)
        .build();

    for entry in walker.flatten() {
        if let Ok(metadata) = entry.metadata() {
            if metadata.is_file() {
                let path = entry.path();
                if config.should_search_file(path) {
                    files_to_search.push(path.to_path_buf());
                }
            }
        }
    }

    // Parallel file search with rayon
    files_to_search.par_iter().for_each(|file_path| {
        // Early exit if truncated
        if truncated.load(Ordering::Relaxed) {
            return;
        }

        // Skip binary files
        if is_binary_file(file_path) {
            return;
        }

        files_scanned.fetch_add(1, Ordering::Relaxed);

        if let Ok(matches) = search_file_streaming(file_path, config) {
            if !matches.is_empty() {
                files_matched.fetch_add(1, Ordering::Relaxed);
                let match_count = matches.len() as u64;
                total_matches.fetch_add(match_count, Ordering::Relaxed);

                // Lock only when we have matches to add
                let mut all = all_matches.lock().expect("TODO: Verify unwrap");
                for m in matches {
                    if config.max_results > 0 && all.len() as u64 >= config.max_results {
                        truncated.store(true, Ordering::Relaxed);
                        break;
                    }
                    all.push(m);
                }
            }
        }
    });

    let elapsed = start.elapsed();
    // Recover data even if mutex was poisoned (thread panicked)
    let matches = all_matches
        .into_inner()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    ContentSearchResult {
        status: "Success".to_string(),
        pattern: config.pattern_str.clone(),
        file_pattern: config.file_pattern_str.clone(),
        files_scanned: files_scanned.load(Ordering::Relaxed),
        files_matched: files_matched.load(Ordering::Relaxed),
        total_matches: total_matches.load(Ordering::Relaxed),
        elapsed_ms: elapsed.as_millis() as u64,
        matches,
        truncated: truncated.load(Ordering::Relaxed),
    }
}

/// Searches a single file for matches using streaming to avoid memory explosion.
/// Uses a rolling buffer for context lines to minimize memory usage.
fn search_file_streaming(
    path: &Path,
    config: &ContentSearchConfig,
) -> std::io::Result<Vec<ContentMatch>> {
    let file = File::open(path)?;
    let reader = BufReader::with_capacity(64 * 1024, file); // 64KB buffer
    let mut matches = Vec::new();

    let context_size = config.context_lines as usize;
    let path_str = path.to_string_lossy().into_owned();

    if context_size == 0 {
        // Fast path: no context needed, pure streaming
        for (idx, line_result) in reader.lines().enumerate() {
            let line = line_result?;
            if config.regex.is_match(&line) {
                matches.push(ContentMatch {
                    path: path_str.clone(),
                    line_number: idx as u64 + 1,
                    line,
                    before: Vec::new(),
                    after: Vec::new(),
                });
            }
        }
    } else {
        // Context path: use rolling buffer for before context
        // Read all lines since we need forward context (after)
        let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;

        for (idx, line) in lines.iter().enumerate() {
            if config.regex.is_match(line) {
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

    Ok(matches)
}

/// Legacy function for backward compatibility in tests.
#[allow(dead_code)]
fn search_file(path: &Path, config: &ContentSearchConfig) -> std::io::Result<Vec<ContentMatch>> {
    search_file_streaming(path, config)
}

/// Returns only statistics without the match list.
fn search_content_stats_impl(config: &ContentSearchConfig) -> ContentSearchStats {
    let result = search_content_impl(config);

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
    match ContentSearchConfig::from_ffi(root_path, pattern, file_pattern, max_results, context_lines)
    {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_dir() -> TempDir {
        let dir = TempDir::new().expect("TODO: Verify unwrap");

        fs::write(
            dir.path().join("file1.txt"),
            "Hello world\nThis is a test\nHello again",
        ).expect("TODO: Verify unwrap");
        fs::write(
            dir.path().join("file2.log"),
            "Error: something failed\nWarning: check this\nError: another failure",
        ).expect("TODO: Verify unwrap");
        fs::write(dir.path().join("data.json"), r#"{"key": "value"}"#).expect("TODO: Verify unwrap");

        let subdir = dir.path().join("subdir");
        fs::create_dir(&subdir).expect("TODO: Verify unwrap");
        fs::write(subdir.join("nested.txt"), "Hello from nested").expect("TODO: Verify unwrap");

        dir
    }

    #[test]
    fn test_search_content_basic() {
        let dir = create_test_dir();

        let regex = Regex::new("Hello").expect("TODO: Verify unwrap");
        let config = ContentSearchConfig {
            root_path: dir.path().to_path_buf(),
            regex,
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

        let regex = Regex::new("Error").expect("TODO: Verify unwrap");
        let glob = Glob::new("*.log").expect("TODO: Verify unwrap");
        let config = ContentSearchConfig {
            root_path: dir.path().to_path_buf(),
            regex,
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

        let regex = Regex::new("test").expect("TODO: Verify unwrap");
        let config = ContentSearchConfig {
            root_path: dir.path().to_path_buf(),
            regex,
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

        let regex = Regex::new("Hello|Error").expect("TODO: Verify unwrap");
        let config = ContentSearchConfig {
            root_path: dir.path().to_path_buf(),
            regex,
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

        let regex = Regex::new("Hello").expect("TODO: Verify unwrap");
        let config = ContentSearchConfig {
            root_path: dir.path().to_path_buf(),
            regex,
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
