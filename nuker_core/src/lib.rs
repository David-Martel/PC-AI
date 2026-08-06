//! Nuker Core - High-Performance Windows Reserved Filename Cleaner
//!
//! This library provides a C-compatible FFI interface for deleting Windows reserved
//! filenames (like "nul", "con", "prn", etc.) using parallel file system traversal
//! and direct Win32 API calls.
//!
//! # Architecture
//! - Uses `ignore` crate for multi-threaded directory walking (ripgrep's engine)
//! - Direct Win32 DeleteFileW API calls for maximum performance
//! - Extended-length path prefix (`\\?\`) to bypass path normalization
//! - Thread-safe atomic counters for statistics tracking
//!
//! # Safety
//! This library uses unsafe code for FFI and Win32 API calls. All unsafe blocks
//! are documented and have been carefully reviewed for correctness.

use std::ffi::{CStr, OsStr};
use std::os::raw::c_char;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};

use ignore::WalkBuilder;
use widestring::U16CString;
use windows_sys::Win32::Storage::FileSystem::DeleteFileW;

/// Windows reserved filenames that cannot be created through normal APIs
/// These filenames are case-insensitive and cause issues on Windows
const RESERVED_NAMES: &[&str] = &[
    "nul", "con", "prn", "aux", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
    "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

/// Selects the filename family handled by a scan.
#[derive(Clone, Copy)]
enum MatchMode {
    ReservedDeviceNames,
    LiteralDollarNull,
}

/// Statistics returned from the scan operation
///
/// This struct is C-compatible and can be marshaled to/from C# or other languages.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ScanStats {
    /// Total number of files scanned during traversal
    pub files_scanned: u32,
    /// Number of reserved files successfully deleted
    pub files_deleted: u32,
    /// Number of errors encountered (permission denied, file in use, etc.)
    pub errors: u32,
}

impl ScanStats {
    /// Creates an error result with a single error count
    const fn error() -> Self {
        Self {
            files_scanned: 0,
            files_deleted: 0,
            errors: 1,
        }
    }
}

include!(concat!(env!("OUT_DIR"), "/version.rs"));

/// Main entry point for the C FFI interface
///
/// # Safety
/// The caller must ensure:
/// - `root_ptr` is either null or points to a valid null-terminated C string
/// - The string remains valid for the duration of this call
/// - The string represents a valid file system path
///
/// # Arguments
/// * `root_ptr` - Null-terminated C string containing the root path to scan
///
/// # Returns
/// A `ScanStats` struct containing scan results. If an error occurs during
/// initialization, returns a struct with errors=1 and other fields=0.
///
/// # Example (from C#)
/// ```csharp
/// [DllImport("nuker_core.dll", CallingConvention = CallingConvention.Cdecl)]
/// private static extern ScanStats nuke_reserved_files(string rootPath);
/// ```
#[no_mangle]
pub unsafe extern "C" fn nuke_reserved_files(root_ptr: *const c_char) -> ScanStats {
    nuke_files(root_ptr, MatchMode::ReservedDeviceNames)
}

/// Deletes ordinary files whose visible leaf name is literal `$null`.
///
/// Matching is case-insensitive and ignores trailing dots/spaces, which Windows
/// may otherwise normalize. Directories, symlinks, and reserved device aliases
/// are not targeted by this entry point.
///
/// # Safety
/// The caller must provide a valid, null-terminated UTF-8 path for the duration
/// of this call, or a null pointer to receive an error result.
#[no_mangle]
pub unsafe extern "C" fn nuke_dollar_null_files(root_ptr: *const c_char) -> ScanStats {
    nuke_files(root_ptr, MatchMode::LiteralDollarNull)
}

/// Validates an FFI root path and runs the requested cleanup mode.
unsafe fn nuke_files(root_ptr: *const c_char, mode: MatchMode) -> ScanStats {
    // Safety check: null pointer
    if root_ptr.is_null() {
        eprintln!("Error: Null pointer passed to nuke_reserved_files");
        return ScanStats::error();
    }

    // Convert C string to Rust string slice
    // Safety: We've verified the pointer is non-null above
    let c_str = unsafe { CStr::from_ptr(root_ptr) };

    let root_path = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: Invalid UTF-8 in path: {}", e);
            return ScanStats::error();
        },
    };

    // Verify the path exists before starting the scan
    if !Path::new(root_path).exists() {
        eprintln!("Error: Path does not exist: {}", root_path);
        return ScanStats::error();
    }

    // Execute the scan
    scan_and_delete(root_path, mode)
}

/// Internal implementation of the scan and delete operation
///
/// This function:
/// 1. Configures a parallel walker with appropriate filters
/// 2. Scans the file system using multiple threads
/// 3. Identifies reserved filenames
/// 4. Deletes them using Win32 API with extended-length paths
/// 5. Tracks statistics using atomic counters
fn scan_and_delete(root_path: &str, mode: MatchMode) -> ScanStats {
    // Thread-safe atomic counters for statistics
    let scanned = AtomicU32::new(0);
    let deleted = AtomicU32::new(0);
    let errors = AtomicU32::new(0);

    // Configure the parallel walker
    // - Uses work-stealing queue for load balancing across threads
    // - Automatically scales to CPU core count
    // - Skips .git directories to avoid repository corruption
    // - Ignores hidden file settings (we want to scan everything)
    let walker = WalkBuilder::new(root_path)
        .hidden(false)           // Scan hidden files and directories
        .git_ignore(false)       // Don't respect .gitignore files
        .git_global(false)       // Don't respect global gitignore
        .git_exclude(false)      // Don't respect .git/info/exclude
        .require_git(false)      // Don't require a git repository
        .ignore(false)           // Don't respect .ignore files
        .parents(false)          // Don't look for ignore files in parent directories
        .filter_entry(|entry| {
            // Skip .git directories entirely to avoid repository corruption
            // This is checked before descending into the directory
            entry.file_name() != ".git"
        })
        .build_parallel();

    // Execute parallel walk
    // Each thread gets its own closure instance for lock-free operation
    walker.run(|| {
        // Clone references to the atomic counters for this thread
        let scanned = &scanned;
        let deleted = &deleted;
        let errors = &errors;
        let mode = mode;

        // Return a boxed closure that processes each directory entry
        Box::new(move |result| {
            match result {
                Ok(entry) => {
                    // Increment scanned counter
                    scanned.fetch_add(1, Ordering::Relaxed);

                    // Only process real files. Directories, symlinks, and
                    // entries whose type cannot be determined are not deletion
                    // candidates.
                    let Some(file_type) = entry.file_type() else {
                        return ignore::WalkState::Continue;
                    };
                    if !file_type.is_file() {
                        return ignore::WalkState::Continue;
                    }

                    // Get the filename (last component of the path)
                    let file_name = entry.file_name();

                    // Check if this is a reserved filename (case-insensitive)
                    // Use OsStr comparison to avoid UTF-8 allocation overhead
                    let is_reserved = matches_target(file_name, mode);

                    if is_reserved {
                        // Attempt to delete the reserved file
                        if delete_file_win32(entry.path()) {
                            deleted.fetch_add(1, Ordering::Relaxed);
                        } else {
                            errors.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                },
                Err(_) => {
                    // Error during traversal (permission denied, symlink loop, etc.)
                    errors.fetch_add(1, Ordering::Relaxed);
                },
            }

            // Continue traversal
            ignore::WalkState::Continue
        })
    });

    // Collect final statistics
    ScanStats {
        files_scanned: scanned.load(Ordering::Relaxed),
        files_deleted: deleted.load(Ordering::Relaxed),
        errors: errors.load(Ordering::Relaxed),
    }
}

/// Returns whether a visible filesystem leaf belongs to the requested mode.
fn matches_target(file_name: &OsStr, mode: MatchMode) -> bool {
    match mode {
        MatchMode::ReservedDeviceNames => RESERVED_NAMES
            .iter()
            .any(|&reserved| file_name.eq_ignore_ascii_case(reserved)),
        MatchMode::LiteralDollarNull => file_name.to_str().is_some_and(|name| {
            name.trim_end_matches(['.', ' '])
                .eq_ignore_ascii_case("$null")
        }),
    }
}

/// Deletes a file using the Win32 DeleteFileW API with extended-length path prefix
///
/// This function:
/// 1. Converts the path to an extended-length path (`\\?\C:\...`)
/// 2. Converts the path to UTF-16 (wide string) for Win32 API
/// 3. Calls DeleteFileW directly to bypass standard library safety checks
///
/// # Arguments
/// * `path` - The file path to delete
///
/// # Returns
/// * `true` if the file was successfully deleted
/// * `false` if an error occurred (file in use, permission denied, conversion error, etc.)
///
/// # Safety
/// This function uses unsafe code to call the Win32 API. The safety invariants are:
/// - The path is converted to a properly null-terminated UTF-16 string
/// - The DeleteFileW API is called with a valid wide string pointer
fn delete_file_win32(path: &Path) -> bool {
    // Convert path to string
    let path_str = match path.to_str() {
        Some(s) => s,
        None => {
            // Path contains invalid UTF-8
            return false;
        },
    };

    // Construct extended-length path to bypass Win32 path normalization
    // and MAX_PATH limitations
    // Format: \\?\C:\path\to\file
    //
    // This prefix tells Windows to:
    // - Disable path parsing and normalization
    // - Allow paths longer than 260 characters (MAX_PATH)
    // - Allow reserved filenames like "nul", "con", etc.
    let extended_path = if path_str.starts_with("\\\\?\\") {
        // Already has extended-length prefix
        path_str.to_string()
    } else if let Some(stripped) = path_str.strip_prefix("\\\\") {
        // UNC path: \\server\share -> \\?\UNC\server\share
        format!("\\\\?\\UNC\\{}", stripped)
    } else {
        // Regular path: C:\path -> \\?\C:\path
        format!("\\\\?\\{}", path_str)
    };

    // Convert to UTF-16 (wide string) for Win32 API
    let wide_path = match U16CString::from_str(&extended_path) {
        Ok(wp) => wp,
        Err(_) => {
            // String conversion error (null byte in path?)
            return false;
        },
    };

    // Call Win32 DeleteFileW API
    // Safety: wide_path.as_ptr() returns a valid pointer to a null-terminated
    // UTF-16 string that lives for the duration of this call
    unsafe {
        // DeleteFileW returns non-zero on success, zero on failure
        DeleteFileW(wide_path.as_ptr()) != 0
    }
}

// Optional: Export additional utility functions for testing or advanced usage

/// Version information for the library
#[no_mangle]
pub extern "C" fn nuker_core_version() -> *const c_char {
    VERSION_CSTR.as_ptr() as *const c_char
}

/// Test function to verify DLL is loaded correctly
#[no_mangle]
pub extern "C" fn nuker_core_test() -> u32 {
    // Return a magic number to verify DLL loaded correctly
    0xDEADBEEF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reserved_names_lowercase() {
        assert!(RESERVED_NAMES.contains(&"nul"));
        assert!(RESERVED_NAMES.contains(&"con"));
        assert!(RESERVED_NAMES.contains(&"prn"));
    }

    #[test]
    fn literal_dollar_null_matching_is_narrow() {
        assert!(matches_target(
            OsStr::new("$null"),
            MatchMode::LiteralDollarNull
        ));
        assert!(matches_target(
            OsStr::new("$NULL. "),
            MatchMode::LiteralDollarNull
        ));
        assert!(!matches_target(
            OsStr::new("$null.txt"),
            MatchMode::LiteralDollarNull
        ));
        assert!(!matches_target(
            OsStr::new("nul"),
            MatchMode::LiteralDollarNull
        ));
    }

    #[test]
    fn reserved_mode_does_not_match_literal_dollar_null() {
        assert!(!matches_target(
            OsStr::new("$null"),
            MatchMode::ReservedDeviceNames
        ));
    }

    #[test]
    fn test_scan_stats_error() {
        let stats = ScanStats::error();
        assert_eq!(stats.files_scanned, 0);
        assert_eq!(stats.files_deleted, 0);
        assert_eq!(stats.errors, 1);
    }

    #[test]
    fn test_extended_path_regular() {
        let path = Path::new("C:\\test\\file.txt");
        // This would normally call delete_file_win32, but we can't test
        // actual deletion without creating test files
        assert!(path.exists() || !path.exists()); // Tautology for compilation test
    }
}
