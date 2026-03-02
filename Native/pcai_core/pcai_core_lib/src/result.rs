//! Result structures for FFI operations.
//!
//! Provides standardized result structures that include status codes,
//! statistics, and timing information.

use crate::error::PcaiStatus;

/// Generic result structure for operations that process items.
///
/// This structure is used as a base for more specialized results.
///
/// # C# Mapping
/// ```csharp
/// [StructLayout(LayoutKind.Sequential)]
/// public struct PcaiResult {
///     public PcaiStatus Status;
///     public ulong Processed;
///     public ulong Matched;
///     public ulong Errors;
///     public ulong ElapsedMs;
/// }
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PcaiResult {
    /// Operation status code
    pub status: PcaiStatus,

    /// Number of items processed
    pub processed: u64,

    /// Number of items that matched the criteria
    pub matched: u64,

    /// Number of errors encountered
    pub errors: u64,

    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
}

impl PcaiResult {
    /// Creates a new successful result with the given statistics.
    pub fn success(processed: u64, matched: u64, errors: u64, elapsed_ms: u64) -> Self {
        Self {
            status: PcaiStatus::Success,
            processed,
            matched,
            errors,
            elapsed_ms,
        }
    }

    /// Creates an error result with the given status.
    pub fn error(status: PcaiStatus) -> Self {
        Self {
            status,
            processed: 0,
            matched: 0,
            errors: 1,
            elapsed_ms: 0,
        }
    }

    /// Creates a result for a null pointer error.
    pub fn null_pointer() -> Self {
        Self::error(PcaiStatus::NullPointer)
    }

    /// Creates a result for an invalid argument error.
    pub fn invalid_argument() -> Self {
        Self::error(PcaiStatus::InvalidArgument)
    }

    /// Creates a result for a path not found error.
    pub fn path_not_found() -> Self {
        Self::error(PcaiStatus::PathNotFound)
    }

    /// Returns true if the operation was successful.
    pub fn is_success(&self) -> bool {
        self.status.is_success()
    }
}

/// Result structure for duplicate file detection operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DuplicateResult {
    /// Operation status code
    pub status: PcaiStatus,

    /// Total number of files scanned
    pub files_scanned: u64,

    /// Number of files successfully hashed
    pub files_hashed: u64,

    /// Number of duplicate groups found
    pub duplicate_groups: u64,

    /// Total number of duplicate files (excluding originals)
    pub duplicate_files: u64,

    /// Total bytes wasted by duplicates
    pub wasted_bytes: u64,

    /// Number of errors encountered
    pub errors: u64,

    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
}

impl DuplicateResult {
    /// Creates a new result with the given statistics.
    pub fn new(
        files_scanned: u64,
        files_hashed: u64,
        duplicate_groups: u64,
        duplicate_files: u64,
        wasted_bytes: u64,
        errors: u64,
        elapsed_ms: u64,
    ) -> Self {
        Self {
            status: PcaiStatus::Success,
            files_scanned,
            files_hashed,
            duplicate_groups,
            duplicate_files,
            wasted_bytes,
            errors,
            elapsed_ms,
        }
    }

    /// Creates an error result with the given status.
    pub fn error(status: PcaiStatus) -> Self {
        Self {
            status,
            errors: 1,
            ..Default::default()
        }
    }
}

/// Result structure for disk usage analysis.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DiskUsageResult {
    /// Operation status code
    pub status: PcaiStatus,

    /// Total number of files counted
    pub file_count: u64,

    /// Total number of directories counted
    pub dir_count: u64,

    /// Total size in bytes
    pub total_bytes: u64,

    /// Number of errors encountered
    pub errors: u64,

    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
}

impl DiskUsageResult {
    /// Creates a new result with the given statistics.
    pub fn new(file_count: u64, dir_count: u64, total_bytes: u64, errors: u64, elapsed_ms: u64) -> Self {
        Self {
            status: PcaiStatus::Success,
            file_count,
            dir_count,
            total_bytes,
            errors,
            elapsed_ms,
        }
    }

    /// Creates an error result with the given status.
    pub fn error(status: PcaiStatus) -> Self {
        Self {
            status,
            errors: 1,
            ..Default::default()
        }
    }
}

/// Result structure for file search operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchResult {
    /// Operation status code
    pub status: PcaiStatus,

    /// Number of files examined
    pub files_examined: u64,

    /// Number of files that matched the criteria
    pub files_matched: u64,

    /// Number of lines matched (for content search)
    pub lines_matched: u64,

    /// Number of errors encountered
    pub errors: u64,

    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
}

impl SearchResult {
    /// Creates a new result with the given statistics.
    pub fn new(files_examined: u64, files_matched: u64, lines_matched: u64, errors: u64, elapsed_ms: u64) -> Self {
        Self {
            status: PcaiStatus::Success,
            files_examined,
            files_matched,
            lines_matched,
            errors,
            elapsed_ms,
        }
    }

    /// Creates an error result with the given status.
    pub fn error(status: PcaiStatus) -> Self {
        Self {
            status,
            errors: 1,
            ..Default::default()
        }
    }
}

/// Result structure for PATH analysis.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PathAnalysisResult {
    /// Operation status code
    pub status: PcaiStatus,

    /// Total number of PATH entries
    pub total_entries: u64,

    /// Number of duplicate entries
    pub duplicate_entries: u64,

    /// Number of non-existent paths
    pub nonexistent_entries: u64,

    /// Number of empty entries
    pub empty_entries: u64,

    /// Number of valid, unique entries
    pub valid_entries: u64,

    /// Elapsed time in milliseconds
    pub elapsed_ms: u64,
}

impl PathAnalysisResult {
    /// Creates a new result with the given statistics.
    pub fn new(
        total_entries: u64,
        duplicate_entries: u64,
        nonexistent_entries: u64,
        empty_entries: u64,
        valid_entries: u64,
        elapsed_ms: u64,
    ) -> Self {
        Self {
            status: PcaiStatus::Success,
            total_entries,
            duplicate_entries,
            nonexistent_entries,
            empty_entries,
            valid_entries,
            elapsed_ms,
        }
    }

    /// Creates an error result with the given status.
    pub fn error(status: PcaiStatus) -> Self {
        Self {
            status,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcai_result_success() {
        let result = PcaiResult::success(100, 50, 2, 150);
        assert!(result.is_success());
        assert_eq!(result.processed, 100);
        assert_eq!(result.matched, 50);
        assert_eq!(result.errors, 2);
        assert_eq!(result.elapsed_ms, 150);
    }

    #[test]
    fn test_pcai_result_error() {
        let result = PcaiResult::error(PcaiStatus::PathNotFound);
        assert!(!result.is_success());
        assert_eq!(result.status, PcaiStatus::PathNotFound);
        assert_eq!(result.errors, 1);
    }

    #[test]
    fn test_duplicate_result_default() {
        let result = DuplicateResult::default();
        assert_eq!(result.status, PcaiStatus::Success);
        assert_eq!(result.files_scanned, 0);
    }

    #[test]
    fn test_disk_usage_result_new() {
        let result = DiskUsageResult::new(1000, 100, 1024 * 1024 * 1024, 5, 500);
        assert!(result.status.is_success());
        assert_eq!(result.file_count, 1000);
        assert_eq!(result.dir_count, 100);
        assert_eq!(result.total_bytes, 1024 * 1024 * 1024);
    }
}
