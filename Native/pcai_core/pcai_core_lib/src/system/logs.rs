//! Log File Search
//!
//! Parallel log file searching with regex support, context lines, and
//! structured JSON output optimized for LLM consumption.

use crate::PcaiStatus;
use ignore::WalkBuilder;
use rayon::prelude::*;
use regex::Regex;
use serde::Serialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// FFI-safe log search statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LogSearchStats {
    pub status: PcaiStatus,
    pub files_searched: u64,
    pub files_with_matches: u64,
    pub total_matches: u64,
    pub bytes_searched: u64,
    pub elapsed_ms: u64,
}

impl Default for LogSearchStats {
    fn default() -> Self {
        Self {
            status: PcaiStatus::Success,
            files_searched: 0,
            files_with_matches: 0,
            total_matches: 0,
            bytes_searched: 0,
            elapsed_ms: 0,
        }
    }
}

impl LogSearchStats {
    pub fn error(status: PcaiStatus) -> Self {
        Self {
            status,
            ..Default::default()
        }
    }
}

/// A single match in a log file
#[derive(Debug, Clone, Serialize)]
pub struct LogMatch {
    pub file_path: String,
    pub line_number: u64,
    pub line_content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub context_before: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub context_after: Vec<String>,
}

/// Matches grouped by file
#[derive(Debug, Clone, Serialize)]
pub struct FileMatches {
    pub file_path: String,
    pub file_size_bytes: u64,
    pub match_count: u64,
    pub matches: Vec<LogMatch>,
}

/// JSON output structure for log search
#[derive(Debug, Serialize)]
pub struct LogSearchJson {
    pub status: String,
    pub pattern: String,
    pub root_path: String,
    pub files_searched: u64,
    pub files_with_matches: u64,
    pub total_matches: u64,
    pub bytes_searched: u64,
    pub elapsed_ms: u64,
    pub file_pattern: Option<String>,
    pub case_sensitive: bool,
    pub results: Vec<FileMatches>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncated: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation_message: Option<String>,
}

/// Options for log search
pub struct LogSearchOptions {
    pub pattern: String,
    pub root_path: String,
    pub file_pattern: Option<String>,
    pub case_sensitive: bool,
    pub context_lines: usize,
    pub max_matches: usize,
    pub max_files: usize,
}

impl Default for LogSearchOptions {
    fn default() -> Self {
        Self {
            pattern: String::new(),
            root_path: String::new(),
            file_pattern: Some("*.log".to_string()),
            case_sensitive: false,
            context_lines: 2,
            max_matches: 1000,
            max_files: 100,
        }
    }
}

/// Search log files for a pattern
pub fn search_logs(options: &LogSearchOptions) -> (LogSearchStats, LogSearchJson) {
    let start = std::time::Instant::now();

    // Validate inputs
    if options.pattern.is_empty() {
        return (
            LogSearchStats::error(PcaiStatus::InvalidArgument),
            LogSearchJson {
                status: "Error".to_string(),
                pattern: options.pattern.clone(),
                root_path: options.root_path.clone(),
                files_searched: 0,
                files_with_matches: 0,
                total_matches: 0,
                bytes_searched: 0,
                elapsed_ms: 0,
                file_pattern: options.file_pattern.clone(),
                case_sensitive: options.case_sensitive,
                results: Vec::new(),
                truncated: None,
                truncation_message: Some("Empty search pattern".to_string()),
            },
        );
    }

    let root_path = Path::new(&options.root_path);
    if !root_path.exists() {
        return (
            LogSearchStats::error(PcaiStatus::IoError),
            LogSearchJson {
                status: "Error".to_string(),
                pattern: options.pattern.clone(),
                root_path: options.root_path.clone(),
                files_searched: 0,
                files_with_matches: 0,
                total_matches: 0,
                bytes_searched: 0,
                elapsed_ms: 0,
                file_pattern: options.file_pattern.clone(),
                case_sensitive: options.case_sensitive,
                results: Vec::new(),
                truncated: None,
                truncation_message: Some(format!("Path does not exist: {}", options.root_path)),
            },
        );
    }

    // Build regex
    let regex_pattern = if options.case_sensitive {
        options.pattern.clone()
    } else {
        format!("(?i){}", options.pattern)
    };

    let regex = match Regex::new(&regex_pattern) {
        Ok(r) => r,
        Err(e) => {
            return (
                LogSearchStats::error(PcaiStatus::InvalidArgument),
                LogSearchJson {
                    status: "Error".to_string(),
                    pattern: options.pattern.clone(),
                    root_path: options.root_path.clone(),
                    files_searched: 0,
                    files_with_matches: 0,
                    total_matches: 0,
                    bytes_searched: 0,
                    elapsed_ms: 0,
                    file_pattern: options.file_pattern.clone(),
                    case_sensitive: options.case_sensitive,
                    results: Vec::new(),
                    truncated: None,
                    truncation_message: Some(format!("Invalid regex pattern: {}", e)),
                },
            );
        }
    };

    // Build file glob pattern
    let file_glob = options
        .file_pattern
        .as_ref()
        .and_then(|p| globset::Glob::new(p).ok())
        .map(|g| g.compile_matcher());

    // Collect files to search
    let files_to_search: Vec<_> = WalkBuilder::new(&options.root_path)
        .hidden(false)
        .git_ignore(false)
        .build()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|ft| ft.is_file()).unwrap_or(false))
        .filter(|e| {
            if let Some(ref glob) = file_glob {
                glob.is_match(e.path())
            } else {
                // Default to common log extensions
                let ext = e.path().extension().and_then(|s| s.to_str()).unwrap_or("");
                matches!(ext.to_lowercase().as_str(), "log" | "txt" | "json")
            }
        })
        .take(options.max_files)
        .collect();

    // Atomic counters for parallel processing
    let files_searched = Arc::new(AtomicU64::new(0));
    let files_with_matches = Arc::new(AtomicU64::new(0));
    let total_matches = Arc::new(AtomicU64::new(0));
    let bytes_searched = Arc::new(AtomicU64::new(0));

    // Search files in parallel
    let results: Vec<FileMatches> = files_to_search
        .par_iter()
        .filter_map(|entry| {
            let path = entry.path();
            files_searched.fetch_add(1, Ordering::Relaxed);

            // Get file size
            let file_size = path.metadata().map(|m| m.len()).unwrap_or(0);
            bytes_searched.fetch_add(file_size, Ordering::Relaxed);

            // Open and search file
            let file = match File::open(path) {
                Ok(f) => f,
                Err(_) => return None,
            };

            let reader = BufReader::new(file);
            let lines: Vec<String> = reader.lines().map_while(Result::ok).collect();

            let mut matches = Vec::new();
            let context = options.context_lines;

            for (idx, line) in lines.iter().enumerate() {
                if regex.is_match(line) {
                    // Check if we've hit max matches
                    if total_matches.load(Ordering::Relaxed) >= options.max_matches as u64 {
                        break;
                    }

                    total_matches.fetch_add(1, Ordering::Relaxed);

                    // Get context lines
                    let context_before: Vec<String> = if idx >= context {
                        lines[idx - context..idx].to_vec()
                    } else {
                        lines[0..idx].to_vec()
                    };

                    let context_after: Vec<String> = if idx + context < lines.len() {
                        lines[idx + 1..=idx + context].to_vec()
                    } else if idx + 1 < lines.len() {
                        lines[idx + 1..].to_vec()
                    } else {
                        Vec::new()
                    };

                    matches.push(LogMatch {
                        file_path: path.to_string_lossy().to_string(),
                        line_number: (idx + 1) as u64,
                        line_content: line.clone(),
                        context_before,
                        context_after,
                    });
                }
            }

            if !matches.is_empty() {
                files_with_matches.fetch_add(1, Ordering::Relaxed);
                Some(FileMatches {
                    file_path: path.to_string_lossy().to_string(),
                    file_size_bytes: file_size,
                    match_count: matches.len() as u64,
                    matches,
                })
            } else {
                None
            }
        })
        .collect();

    let elapsed_ms = start.elapsed().as_millis() as u64;
    let total_match_count = total_matches.load(Ordering::Relaxed);
    let truncated = total_match_count >= options.max_matches as u64;

    let stats = LogSearchStats {
        status: PcaiStatus::Success,
        files_searched: files_searched.load(Ordering::Relaxed),
        files_with_matches: files_with_matches.load(Ordering::Relaxed),
        total_matches: total_match_count,
        bytes_searched: bytes_searched.load(Ordering::Relaxed),
        elapsed_ms,
    };

    let json = LogSearchJson {
        status: "Success".to_string(),
        pattern: options.pattern.clone(),
        root_path: options.root_path.clone(),
        files_searched: stats.files_searched,
        files_with_matches: stats.files_with_matches,
        total_matches: stats.total_matches,
        bytes_searched: stats.bytes_searched,
        elapsed_ms,
        file_pattern: options.file_pattern.clone(),
        case_sensitive: options.case_sensitive,
        results,
        truncated: if truncated { Some(true) } else { None },
        truncation_message: if truncated {
            Some(format!("Results truncated at {} matches", options.max_matches))
        } else {
            None
        },
    };

    (stats, json)
}

/// Format bytes as human-readable string
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_search_logs_empty_pattern() {
        let options = LogSearchOptions {
            pattern: "".to_string(),
            root_path: ".".to_string(),
            ..Default::default()
        };

        let (stats, json) = search_logs(&options);
        assert_eq!(stats.status, PcaiStatus::InvalidArgument);
        assert_eq!(json.status, "Error");
    }

    #[test]
    fn test_search_logs_invalid_path() {
        let options = LogSearchOptions {
            pattern: "test".to_string(),
            root_path: "/nonexistent/path/xyz123".to_string(),
            ..Default::default()
        };

        let (stats, json) = search_logs(&options);
        assert_eq!(stats.status, PcaiStatus::IoError);
        assert_eq!(json.status, "Error");
    }

    #[test]
    fn test_search_logs_with_matches() {
        // Create a temp directory with test log file
        let dir = tempdir().expect("failed to create temp dir");
        let log_path = dir.path().join("test.log");
        let mut file = File::create(&log_path).expect("failed to create test.log");
        writeln!(file, "Line 1: normal content").expect("failed to write line 1");
        writeln!(file, "Line 2: ERROR something failed").expect("failed to write line 2");
        writeln!(file, "Line 3: more normal content").expect("failed to write line 3");
        writeln!(file, "Line 4: ERROR another failure").expect("failed to write line 4");
        writeln!(file, "Line 5: final line").expect("failed to write line 5");

        let options = LogSearchOptions {
            pattern: "ERROR".to_string(),
            root_path: dir.path().to_string_lossy().to_string(),
            file_pattern: Some("*.log".to_string()),
            case_sensitive: true,
            context_lines: 1,
            max_matches: 100,
            max_files: 10,
        };

        let (stats, json) = search_logs(&options);
        assert_eq!(stats.status, PcaiStatus::Success);
        assert_eq!(stats.files_searched, 1);
        assert_eq!(stats.files_with_matches, 1);
        assert_eq!(stats.total_matches, 2);
        assert_eq!(json.results.len(), 1);
        assert_eq!(json.results[0].match_count, 2);
    }

    #[test]
    fn test_search_logs_case_insensitive() {
        let dir = tempdir().expect("failed to create temp dir");
        let log_path = dir.path().join("test.log");
        let mut file = File::create(&log_path).expect("failed to create test.log");
        writeln!(file, "error lowercase").expect("failed to write error lowercase line");
        writeln!(file, "ERROR uppercase").expect("failed to write ERROR uppercase line");
        writeln!(file, "Error mixed").expect("failed to write Error mixed line");

        let options = LogSearchOptions {
            pattern: "error".to_string(),
            root_path: dir.path().to_string_lossy().to_string(),
            file_pattern: Some("*.log".to_string()),
            case_sensitive: false,
            context_lines: 0,
            max_matches: 100,
            max_files: 10,
        };

        let (stats, _json) = search_logs(&options);
        assert_eq!(stats.status, PcaiStatus::Success);
        assert_eq!(stats.total_matches, 3);
    }
}
