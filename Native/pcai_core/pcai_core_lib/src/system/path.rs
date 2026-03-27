//! PATH Environment Variable Analysis
//!
//! Analyzes system and user PATH environment variables for issues including
//! duplicates, non-existent directories, empty entries, and trailing slashes.

use crate::PcaiStatus;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// FFI-safe PATH analysis statistics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PathAnalysisStats {
    pub status: PcaiStatus,
    pub total_entries: u32,
    pub unique_entries: u32,
    pub duplicate_count: u32,
    pub non_existent_count: u32,
    pub empty_count: u32,
    pub trailing_slash_count: u32,
    pub cross_duplicate_count: u32,
    pub elapsed_ms: u64,
}

impl Default for PathAnalysisStats {
    fn default() -> Self {
        Self {
            status: PcaiStatus::Success,
            total_entries: 0,
            unique_entries: 0,
            duplicate_count: 0,
            non_existent_count: 0,
            empty_count: 0,
            trailing_slash_count: 0,
            cross_duplicate_count: 0,
            elapsed_ms: 0,
        }
    }
}

impl PathAnalysisStats {
    pub fn error(status: PcaiStatus) -> Self {
        Self {
            status,
            ..Default::default()
        }
    }
}

/// Individual PATH entry with issue flags
#[derive(Debug, Clone, Serialize)]
pub struct PathEntry {
    pub path: String,
    pub source: String, // "User" or "Machine"
    pub exists: bool,
    pub is_empty: bool,
    pub has_trailing_slash: bool,
    pub is_duplicate: bool,
    pub duplicate_of: Option<String>,
    pub is_cross_duplicate: bool,
}

/// Duplicate group information
#[derive(Debug, Clone, Serialize)]
pub struct DuplicateGroup {
    pub normalized_path: String,
    pub entries: Vec<PathEntryRef>,
}

/// Reference to a PATH entry
#[derive(Debug, Clone, Serialize)]
pub struct PathEntryRef {
    pub path: String,
    pub source: String,
    pub index: usize,
}

/// JSON output structure for PATH analysis
#[derive(Debug, Serialize)]
pub struct PathAnalysisJson {
    pub status: String,
    pub total_entries: u32,
    pub unique_entries: u32,
    pub machine_total_entries: u32,
    pub user_total_entries: u32,
    pub duplicate_count: u32,
    pub non_existent_count: u32,
    pub empty_count: u32,
    pub trailing_slash_count: u32,
    pub cross_duplicate_count: u32,
    pub elapsed_ms: u64,
    pub health_status: String,
    pub issues: Vec<PathIssue>,
    pub duplicates: Vec<DuplicateGroup>,
    pub non_existent: Vec<PathEntryRef>,
    pub machine_entries: Vec<PathEntryRef>,
    pub user_entries: Vec<PathEntryRef>,
    pub recommendations: Vec<String>,
}

/// Individual issue in PATH
#[derive(Debug, Clone, Serialize)]
pub struct PathIssue {
    pub severity: String,
    pub issue_type: String,
    pub path: String,
    pub source: String,
    pub description: String,
}

/// Normalize a path for comparison (lowercase on Windows, resolve trailing slashes)
fn normalize_path(path: &str) -> String {
    let mut normalized = path.trim().to_string();

    // On Windows, paths are case-insensitive
    #[cfg(windows)]
    {
        normalized = normalized.to_lowercase();
    }

    // Remove trailing slashes/backslashes (except for root paths like C:\)
    while normalized.len() > 3 && (normalized.ends_with('\\') || normalized.ends_with('/')) {
        normalized.pop();
    }

    // Normalize path separators on Windows
    #[cfg(windows)]
    {
        normalized = normalized.replace('/', "\\");
    }

    normalized
}

/// Get User and Machine PATH separately on Windows
#[cfg(windows)]
fn get_user_and_machine_paths() -> (Vec<String>, Vec<String>) {
    use winreg::enums::{HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE};
    use winreg::RegKey;

    let mut user_paths = Vec::new();
    let mut machine_paths = Vec::new();

    // Try to read User PATH from registry
    if let Ok(hkcu) = RegKey::predef(HKEY_CURRENT_USER).open_subkey("Environment") {
        if let Ok(path) = hkcu.get_value::<String, _>("Path") {
            user_paths = path.split(';').map(|s| s.to_string()).collect();
        }
    }

    // Try to read Machine PATH from registry
    if let Ok(hklm) =
        RegKey::predef(HKEY_LOCAL_MACHINE).open_subkey(r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
    {
        if let Ok(path) = hklm.get_value::<String, _>("Path") {
            machine_paths = path.split(';').map(|s| s.to_string()).collect();
        }
    }

    (user_paths, machine_paths)
}

#[cfg(not(windows))]
fn get_user_and_machine_paths() -> (Vec<String>, Vec<String>) {
    let path = std::env::var("PATH").unwrap_or_default();
    let entries: Vec<String> = path.split(':').map(|s| s.to_string()).collect();
    // On Unix, we don't have a clear User/Machine distinction
    (entries, Vec::new())
}

/// Analyze PATH environment variable
pub fn analyze_path() -> (PathAnalysisStats, PathAnalysisJson) {
    let start = std::time::Instant::now();

    // Get combined PATH
    let path_str = std::env::var("PATH").unwrap_or_default();

    #[cfg(windows)]
    let separator = ';';
    #[cfg(not(windows))]
    let separator = ':';

    let entries: Vec<&str> = path_str.split(separator).collect();

    // Get User and Machine paths for cross-duplicate detection
    let (user_paths, machine_paths) = get_user_and_machine_paths();

    // Create sets for User and Machine normalized paths
    let user_normalized: HashSet<String> = user_paths.iter().map(|p| normalize_path(p)).collect();
    let machine_normalized: HashSet<String> = machine_paths.iter().map(|p| normalize_path(p)).collect();

    // Track normalized paths for duplicate detection
    let mut seen: HashMap<String, Vec<(usize, String, String)>> = HashMap::new();

    let mut stats = PathAnalysisStats::default();
    let mut all_entries = Vec::new();
    let mut issues = Vec::new();
    let mut non_existent_entries = Vec::new();

    // First pass: collect all entries and identify basic issues
    for (idx, entry) in entries.iter().enumerate() {
        let entry_str = entry.to_string();
        let normalized = normalize_path(&entry_str);

        // Determine source (User or Machine)
        let source = if user_normalized.contains(&normalized) {
            "User"
        } else if machine_normalized.contains(&normalized) {
            "Machine"
        } else {
            "Unknown"
        };

        // Track for duplicate detection — move `normalized` into the map key (no clone needed
        // here because `normalized` is not used after this point in the loop body).
        seen.entry(normalized)
            .or_default()
            .push((idx, entry_str.clone(), source.to_string()));

        all_entries.push(PathEntryRef {
            path: entry_str.clone(),
            source: source.to_string(),
            index: idx,
        });

        // Check for empty entries
        if entry.trim().is_empty() {
            stats.empty_count += 1;
            issues.push(PathIssue {
                severity: "Low".to_string(),
                issue_type: "Empty".to_string(),
                path: entry_str.clone(),
                source: source.to_string(),
                description: "Empty PATH entry".to_string(),
            });
        }

        // Check for trailing slashes
        if entry.len() > 3 && (entry.ends_with('\\') || entry.ends_with('/')) && !entry.ends_with(":\\") {
            stats.trailing_slash_count += 1;
            issues.push(PathIssue {
                severity: "Low".to_string(),
                issue_type: "TrailingSlash".to_string(),
                path: entry_str.clone(),
                source: source.to_string(),
                description: "Path has trailing slash".to_string(),
            });
        }

        // Check if path exists (skip empty entries)
        if !entry.trim().is_empty() && !Path::new(entry).exists() {
            stats.non_existent_count += 1;
            non_existent_entries.push(PathEntryRef {
                path: entry_str.clone(),
                source: source.to_string(),
                index: idx,
            });
            issues.push(PathIssue {
                severity: "Medium".to_string(),
                issue_type: "NonExistent".to_string(),
                path: entry_str.clone(),
                source: source.to_string(),
                description: "Directory does not exist".to_string(),
            });
        }
    }

    // Second pass: identify duplicates
    let mut duplicate_groups = Vec::new();
    for (normalized, occurrences) in &seen {
        if occurrences.len() > 1 {
            // Count duplicates (occurrences - 1 since first is original)
            stats.duplicate_count += (occurrences.len() - 1) as u32;

            // Check for cross-duplicates (same path in both User and Machine)
            let has_user = occurrences.iter().any(|(_, _, src)| src == "User");
            let has_machine = occurrences.iter().any(|(_, _, src)| src == "Machine");

            if has_user && has_machine {
                stats.cross_duplicate_count += 1;
            }

            let entries: Vec<PathEntryRef> = occurrences
                .iter()
                .map(|(idx, path, src)| PathEntryRef {
                    path: path.clone(),
                    source: src.clone(),
                    index: *idx,
                })
                .collect();

            // Add duplicate issues
            for (i, entry) in entries.iter().enumerate() {
                if i > 0 {
                    issues.push(PathIssue {
                        severity: if has_user && has_machine { "High" } else { "Medium" }.to_string(),
                        issue_type: if has_user && has_machine {
                            "CrossDuplicate"
                        } else {
                            "Duplicate"
                        }
                        .to_string(),
                        path: entry.path.clone(),
                        source: entry.source.clone(),
                        description: format!("Duplicate of entry at index {}", entries[0].index),
                    });
                }
            }

            duplicate_groups.push(DuplicateGroup {
                normalized_path: normalized.clone(),
                entries,
            });
        }
    }

    stats.total_entries = entries.len() as u32;
    stats.unique_entries = seen.len() as u32;
    stats.elapsed_ms = start.elapsed().as_millis() as u64;

    // Determine health status
    let health_status = if stats.duplicate_count == 0 && stats.non_existent_count == 0 && stats.empty_count == 0 {
        "Healthy"
    } else if stats.cross_duplicate_count > 0 || stats.non_existent_count > 5 {
        "NeedsAttention"
    } else {
        "Minor Issues"
    };

    // Generate recommendations
    let mut recommendations = Vec::new();

    if stats.duplicate_count > 0 {
        recommendations.push(format!(
            "Remove {} duplicate PATH entries to improve lookup performance",
            stats.duplicate_count
        ));
    }

    if stats.cross_duplicate_count > 0 {
        recommendations.push(format!(
            "Found {} paths in both User and Machine PATH - consolidate to one location",
            stats.cross_duplicate_count
        ));
    }

    if stats.non_existent_count > 0 {
        recommendations.push(format!(
            "Remove {} non-existent directories from PATH",
            stats.non_existent_count
        ));
    }

    if stats.empty_count > 0 {
        recommendations.push(format!("Remove {} empty PATH entries", stats.empty_count));
    }

    if stats.trailing_slash_count > 0 {
        recommendations.push(format!(
            "Remove trailing slashes from {} PATH entries",
            stats.trailing_slash_count
        ));
    }

    let machine_entries: Vec<PathEntryRef> = all_entries.iter().filter(|e| e.source == "Machine").cloned().collect();
    let user_entries: Vec<PathEntryRef> = all_entries.iter().filter(|e| e.source == "User").cloned().collect();
    let machine_total_entries = machine_entries.len() as u32;
    let user_total_entries = user_entries.len() as u32;

    let json = PathAnalysisJson {
        status: "Success".to_string(),
        total_entries: stats.total_entries,
        unique_entries: stats.unique_entries,
        machine_total_entries,
        user_total_entries,
        duplicate_count: stats.duplicate_count,
        non_existent_count: stats.non_existent_count,
        empty_count: stats.empty_count,
        trailing_slash_count: stats.trailing_slash_count,
        cross_duplicate_count: stats.cross_duplicate_count,
        elapsed_ms: stats.elapsed_ms,
        health_status: health_status.to_string(),
        issues,
        duplicates: duplicate_groups,
        non_existent: non_existent_entries,
        machine_entries,
        user_entries,
        recommendations,
    };

    (stats, json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_path() {
        // Test trailing slash removal
        assert_eq!(
            normalize_path("C:\\Windows\\System32\\"),
            normalize_path("C:\\Windows\\System32")
        );

        // Test forward slash conversion on Windows
        #[cfg(windows)]
        assert_eq!(
            normalize_path("C:/Windows/System32"),
            normalize_path("C:\\Windows\\System32")
        );

        // Test case insensitivity on Windows
        #[cfg(windows)]
        assert_eq!(normalize_path("C:\\WINDOWS"), normalize_path("C:\\windows"));
    }

    #[test]
    fn test_analyze_path() {
        let (stats, json) = analyze_path();
        assert_eq!(stats.status, PcaiStatus::Success);
        assert!(stats.total_entries > 0);
        assert!(!json.health_status.is_empty());
    }

    #[test]
    fn test_path_entry_detection() {
        // Create a test case with known values
        let test_path = "C:\\Windows\\System32";
        let normalized = normalize_path(test_path);
        assert!(!normalized.is_empty());
        assert!(!normalized.ends_with('\\'));
    }
}
