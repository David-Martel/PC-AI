//! Nuker Core - High-Performance Windows Problematic-Filename Cleaner
//!
//! This library provides a C-compatible FFI interface for deleting Windows filenames
//! that standard tooling cannot remove (reserved device aliases, stray literal `$null`
//! artifacts, and shell path-mangling artifacts), using parallel file system traversal
//! and direct Win32 API calls.
//!
//! # Architecture
//! - Uses `ignore` crate for multi-threaded directory walking (ripgrep's engine)
//! - Direct Win32 `DeleteFileW`/`RemoveDirectoryW` calls for maximum performance
//! - Extended-length path prefix (`\\?\`) to bypass path normalization
//! - Thread-safe collection of per-entry results (deleted / would-delete / skipped)
//!
//! # Match families
//! A scan can combine any of three independent name families:
//! - **Reserved device names**: `nul`, `con`, `prn`, `aux`, `com1-9`, `lpt1-9`
//! - **Literal `$null`**: real files/dirs whose visible leaf is exactly `$null`
//!   (case-insensitive, trailing dots/spaces ignored). Files are only deleted when
//!   zero-byte unless `allow_nonempty` is set (see [`NukeOptions`]).
//! - **Path-mangle artifacts**: leaf names ending in `;<letter>` or `;<letter>:`,
//!   produced by shells that mis-concatenate a path.
//!
//! Directories are never touched unless `include_dirs` is set, and even then only an
//! *empty* directory is ever removed (`RemoveDirectoryW` itself refuses non-empty
//! directories - we never recurse to empty one out).
//!
//! # Safety
//! This library uses unsafe code for FFI and Win32 API calls. All unsafe blocks
//! are documented and have been carefully reviewed for correctness.

use std::ffi::{CStr, CString, OsStr};
use std::os::raw::c_char;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, PoisonError};

use ignore::WalkBuilder;
use widestring::U16CString;
use windows_sys::Win32::Foundation::GetLastError;
use windows_sys::Win32::Storage::FileSystem::{DeleteFileW, RemoveDirectoryW};

/// Windows reserved filenames that cannot be created through normal APIs
/// These filenames are case-insensitive and cause issues on Windows
const RESERVED_NAMES: &[&str] = &[
    "nul", "con", "prn", "aux", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
    "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

/// How many bytes of a non-empty `$null` file are captured for operator review.
const CONTENT_PREVIEW_MAX_BYTES: usize = 200;

/// `RemoveDirectoryW` failure code meaning "directory has children" (not an error,
/// just a rejected candidate - see [`DirDeleteOutcome::NotEmpty`]).
const ERROR_DIR_NOT_EMPTY: u32 = 145;

/// Statistics returned from the legacy (pre-`nuke_files_ex`) C FFI functions.
///
/// This struct is C-compatible and can be marshaled to/from C# or other languages.
/// Retained for ABI compatibility with existing callers of `nuke_reserved_files`
/// and `nuke_dollar_null_files`; new integrations should prefer [`nuke_files_ex`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ScanStats {
    /// Total number of files scanned during traversal
    pub files_scanned: u32,
    /// Number of matching entries successfully deleted
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

/// Options controlling a [`nuke_files_ex`] scan. Every field is a C-ABI-safe `u8`
/// boolean (`0` = false, any other value = true) to avoid platform-specific `BOOL`
/// marshaling ambiguity.
///
/// # Fields
/// - `match_reserved` - include the reserved-device-name family
/// - `match_dollar_null` - include the literal `$null` family
/// - `match_path_mangle` - include the path-mangle-artifact family
/// - `include_dirs` - also consider empty directories as deletion candidates
/// - `dry_run` - walk and match, but never delete anything
/// - `allow_nonempty` - allow deleting `$null` files larger than zero bytes
///   (only meaningful when `match_dollar_null` is set)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NukeOptions {
    pub match_reserved: u8,
    pub match_dollar_null: u8,
    pub match_path_mangle: u8,
    pub include_dirs: u8,
    pub dry_run: u8,
    pub allow_nonempty: u8,
}

include!(concat!(env!("OUT_DIR"), "/version.rs"));

// ---------------------------------------------------------------------------
// Legacy FFI entry points (stable ABI, preserved for existing callers)
// ---------------------------------------------------------------------------

/// Deletes reserved-device-name files (`nul`, `con`, `prn`, ...) under `root_ptr`.
///
/// # Safety
/// The caller must ensure:
/// - `root_ptr` is either null or points to a valid null-terminated C string
/// - The string remains valid for the duration of this call
/// - The string represents a valid file system path
#[no_mangle]
pub unsafe extern "C" fn nuke_reserved_files(root_ptr: *const c_char) -> ScanStats {
    legacy_scan(
        root_ptr,
        NukeOptions {
            match_reserved: 1,
            match_dollar_null: 0,
            match_path_mangle: 0,
            include_dirs: 0,
            dry_run: 0,
            allow_nonempty: 0,
        },
    )
}

/// Deletes zero-byte files whose visible leaf name is literal `$null`.
///
/// Matching is case-insensitive and ignores trailing dots/spaces, which Windows
/// may otherwise normalize. Directories, symlinks, and reserved device aliases
/// are not targeted by this entry point.
///
/// # Behavior change
/// As of the `--dollar-null-only` zero-byte safety gate, this legacy entry point
/// now only deletes **zero-byte** `$null` files (matching the documented workspace
/// policy: "Discovery hooks may delete only zero-byte matches"). Non-empty `$null`
/// files are left untouched. Use [`nuke_files_ex`] with `allow_nonempty = 1` to
/// opt into deleting larger files.
///
/// # Safety
/// The caller must provide a valid, null-terminated UTF-8 path for the duration
/// of this call, or a null pointer to receive an error result.
#[no_mangle]
pub unsafe extern "C" fn nuke_dollar_null_files(root_ptr: *const c_char) -> ScanStats {
    legacy_scan(
        root_ptr,
        NukeOptions {
            match_reserved: 0,
            match_dollar_null: 1,
            match_path_mangle: 0,
            include_dirs: 0,
            dry_run: 0,
            allow_nonempty: 0,
        },
    )
}

/// Shared implementation for the legacy `ScanStats`-returning entry points.
unsafe fn legacy_scan(root_ptr: *const c_char, options: NukeOptions) -> ScanStats {
    match validate_and_run(root_ptr, options) {
        Ok(output) => ScanStats {
            files_scanned: output.counts.scanned,
            #[allow(clippy::cast_possible_truncation)]
            files_deleted: output.counts.deleted as u32,
            errors: output.counts.errors,
        },
        Err(message) => {
            eprintln!("Error: {message}");
            ScanStats::error()
        }
    }
}

// ---------------------------------------------------------------------------
// Primary FFI entry point: composable families, dry-run, auditable JSON result
// ---------------------------------------------------------------------------

/// Runs a scan/delete pass with the requested [`NukeOptions`] and returns a
/// heap-allocated, null-terminated UTF-8 JSON string describing the result.
///
/// On success the JSON has the shape:
/// ```json
/// {
///   "status": "success",
///   "dry_run": false,
///   "counts": {"scanned": 0, "deleted": 0, "would_delete": 0, "skipped": 0, "errors": 0},
///   "deleted": [{"path": "...", "family": "reserved", "kind": "file"}],
///   "would_delete": [],
///   "skipped": [{"path": "...", "family": "dollar_null", "kind": "file",
///                "reason": "...", "size": 877, "content_preview": "..."}]
/// }
/// ```
/// On failure (invalid input, path does not exist, ...) it has the shape:
/// ```json
/// {"status": "error", "message": "..."}
/// ```
///
/// The caller MUST pass the returned pointer to [`nuke_free_string`] exactly once
/// to release it. This function never returns a null pointer.
///
/// # Safety
/// The caller must ensure `root_ptr` is either null or a valid null-terminated
/// UTF-8 C string that remains valid for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn nuke_files_ex(
    root_ptr: *const c_char,
    options: NukeOptions,
) -> *mut c_char {
    let json = match validate_and_run(root_ptr, options) {
        Ok(output) => serde_json::to_string(&output).unwrap_or_else(|e| {
            format!("{{\"status\":\"error\",\"message\":\"serialization failed: {e}\"}}")
        }),
        Err(message) => {
            let err = ErrorOutput {
                status: "error",
                message,
            };
            serde_json::to_string(&err).unwrap_or_else(|_| {
                "{\"status\":\"error\",\"message\":\"unknown error\"}".to_string()
            })
        }
    };
    to_c_string(json)
}

/// Frees a string previously returned by [`nuke_files_ex`].
///
/// # Safety
/// `ptr` must be either null (a no-op) or a pointer previously returned by
/// `nuke_files_ex` that has not already been freed.
#[no_mangle]
pub unsafe extern "C" fn nuke_free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    drop(unsafe { CString::from_raw(ptr) });
}

/// Validates an FFI root path and runs the requested scan.
unsafe fn validate_and_run(
    root_ptr: *const c_char,
    options: NukeOptions,
) -> Result<EngineOutput, String> {
    if root_ptr.is_null() {
        return Err("null pointer passed for root path".to_string());
    }

    // Safety: caller contract guarantees a valid, null-terminated string.
    let c_str = unsafe { CStr::from_ptr(root_ptr) };

    let root_path = c_str
        .to_str()
        .map_err(|e| format!("invalid UTF-8 in path: {e}"))?;

    if !Path::new(root_path).exists() {
        return Err(format!("path does not exist: {root_path}"));
    }

    run_engine(root_path, options)
}

/// Converts an owned JSON `String` into a heap `*mut c_char` for return across FFI.
///
/// `serde_json` escapes control characters (including NUL) in string values, so
/// `CString::new` should never observe an embedded NUL here; the fallback exists
/// purely as a defensive measure so this function can never panic.
fn to_c_string(json: String) -> *mut c_char {
    match CString::new(json) {
        Ok(cs) => cs.into_raw(),
        Err(_) => CString::new(
            "{\"status\":\"error\",\"message\":\"internal: JSON contained an embedded NUL byte\"}",
        )
        .unwrap_or_default()
        .into_raw(),
    }
}

// ---------------------------------------------------------------------------
// Match families
// ---------------------------------------------------------------------------

/// Identifies which family a matched entry belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatchFamily {
    Reserved,
    DollarNull,
    PathMangle,
}

impl MatchFamily {
    const fn as_str(self) -> &'static str {
        match self {
            MatchFamily::Reserved => "reserved",
            MatchFamily::DollarNull => "dollar_null",
            MatchFamily::PathMangle => "path_mangle",
        }
    }
}

/// Which match families are active for a scan (resolved from [`NukeOptions`]).
#[derive(Debug, Clone, Copy)]
struct ActiveFamilies {
    reserved: bool,
    dollar_null: bool,
    path_mangle: bool,
}

impl ActiveFamilies {
    fn any(self) -> bool {
        self.reserved || self.dollar_null || self.path_mangle
    }
}

fn resolve_families(options: &NukeOptions) -> ActiveFamilies {
    ActiveFamilies {
        reserved: options.match_reserved != 0,
        dollar_null: options.match_dollar_null != 0,
        path_mangle: options.match_path_mangle != 0,
    }
}

/// Returns the family a leaf filename belongs to, checking only families enabled
/// in `families`. Families are checked in a fixed order so a name matching more
/// than one narrow rule reports the first (this cannot currently happen given the
/// three families are mutually exclusive by construction, but a fixed order keeps
/// results deterministic if that ever changes).
fn classify(file_name: &OsStr, families: ActiveFamilies) -> Option<MatchFamily> {
    if families.reserved && matches_reserved(file_name) {
        return Some(MatchFamily::Reserved);
    }
    if families.dollar_null && matches_dollar_null(file_name) {
        return Some(MatchFamily::DollarNull);
    }
    if families.path_mangle && matches_path_mangle(file_name) {
        return Some(MatchFamily::PathMangle);
    }
    None
}

/// Returns whether `file_name` is a reserved Windows device name (case-insensitive).
fn matches_reserved(file_name: &OsStr) -> bool {
    RESERVED_NAMES
        .iter()
        .any(|&reserved| file_name.eq_ignore_ascii_case(reserved))
}

/// Returns whether `file_name` is literal `$null` (case-insensitive, trailing
/// dots/spaces ignored - Windows may otherwise normalize them away).
fn matches_dollar_null(file_name: &OsStr) -> bool {
    file_name.to_str().is_some_and(|name| {
        name.trim_end_matches(['.', ' '])
            .eq_ignore_ascii_case("$null")
    })
}

/// Returns whether `file_name` looks like a shell path-mangling artifact: a leaf
/// name ending in `;<letter>` or `;<letter>:` (e.g. `foo;C` or `foo;C:`), produced
/// when a shell mis-concatenates a path and a stray drive-letter fragment is
/// appended after a semicolon. Deliberately narrow: `;` is a legal filename
/// character and a broader rule would be dangerous.
fn matches_path_mangle(file_name: &OsStr) -> bool {
    let Some(name) = file_name.to_str() else {
        return false;
    };
    let core = name.strip_suffix(':').unwrap_or(name);
    let bytes = core.as_bytes();
    if bytes.len() < 2 {
        return false;
    }
    let last = bytes[bytes.len() - 1];
    let semicolon = bytes[bytes.len() - 2];
    semicolon == b';' && last.is_ascii_alphabetic()
}

/// Returns a skip reason when a `$null` file of the given `size` must NOT be
/// deleted under the current options (the zero-byte safety gate). Returns `None`
/// when the file is eligible for deletion.
fn dollar_null_skip_reason(size: u64, allow_nonempty: bool) -> Option<String> {
    if size == 0 || allow_nonempty {
        None
    } else {
        Some(format!(
            "non-empty $null file ({size} bytes); rerun with --allow-nonempty to delete"
        ))
    }
}

// ---------------------------------------------------------------------------
// Win32 deletion helpers
// ---------------------------------------------------------------------------

/// Converts a path to an extended-length path (`\\?\C:\...` / `\\?\UNC\...`) to
/// bypass Win32 path normalization and `MAX_PATH` limitations. Returns `None` if
/// the path is not valid UTF-8.
fn to_extended_path(path: &Path) -> Option<String> {
    let path_str = path.to_str()?;
    if path_str.starts_with(r"\\?\") {
        Some(path_str.to_string())
    } else if let Some(stripped) = path_str.strip_prefix(r"\\") {
        Some(format!(r"\\?\UNC\{stripped}"))
    } else {
        Some(format!(r"\\?\{path_str}"))
    }
}

/// Maps a Win32 error code to a short human-readable description for the common
/// cases this tool encounters; falls back to the raw numeric code otherwise.
fn describe_win32_error(code: u32) -> String {
    match code {
        2 => "file not found (ERROR_FILE_NOT_FOUND)".to_string(),
        5 => "access denied (ERROR_ACCESS_DENIED)".to_string(),
        19 => "write-protected media (ERROR_WRITE_PROTECT)".to_string(),
        32 => "sharing violation, file in use (ERROR_SHARING_VIOLATION)".to_string(),
        145 => "directory not empty (ERROR_DIR_NOT_EMPTY)".to_string(),
        other => format!("Win32 error {other}"),
    }
}

/// Deletes a file using the Win32 `DeleteFileW` API with an extended-length path
/// prefix, bypassing standard library path normalization (required for reserved
/// device names like `nul`).
fn delete_file_win32(path: &Path) -> Result<(), String> {
    let Some(extended_path) = to_extended_path(path) else {
        return Err("path contains invalid UTF-8".to_string());
    };
    let wide_path = match U16CString::from_str(&extended_path) {
        Ok(wp) => wp,
        Err(_) => return Err("path contains an embedded NUL byte".to_string()),
    };
    // Safety: wide_path is a valid, null-terminated UTF-16 string that lives for
    // the duration of this call.
    unsafe {
        if DeleteFileW(wide_path.as_ptr()) != 0 {
            Ok(())
        } else {
            Err(describe_win32_error(GetLastError()))
        }
    }
}

/// Outcome of a failed directory deletion attempt.
enum DirDeleteOutcome {
    /// `RemoveDirectoryW` refused because the directory has children. This is the
    /// deliberate safety mechanism for `--include-dirs`: we never recurse to
    /// empty a directory out, we only remove ones that are already empty.
    NotEmpty,
    /// Any other failure (permission denied, in use, etc.).
    Failed(String),
}

/// Deletes an EMPTY directory using the Win32 `RemoveDirectoryW` API. Never
/// recurses or removes directory contents - a non-empty directory is reported as
/// [`DirDeleteOutcome::NotEmpty`], not attempted.
fn delete_dir_win32(path: &Path) -> Result<(), DirDeleteOutcome> {
    let Some(extended_path) = to_extended_path(path) else {
        return Err(DirDeleteOutcome::Failed(
            "path contains invalid UTF-8".to_string(),
        ));
    };
    let wide_path = match U16CString::from_str(&extended_path) {
        Ok(wp) => wp,
        Err(_) => {
            return Err(DirDeleteOutcome::Failed(
                "path contains an embedded NUL byte".to_string(),
            ))
        }
    };
    // Safety: wide_path is a valid, null-terminated UTF-16 string that lives for
    // the duration of this call.
    unsafe {
        if RemoveDirectoryW(wide_path.as_ptr()) != 0 {
            Ok(())
        } else {
            let code = GetLastError();
            if code == ERROR_DIR_NOT_EMPTY {
                Err(DirDeleteOutcome::NotEmpty)
            } else {
                Err(DirDeleteOutcome::Failed(describe_win32_error(code)))
            }
        }
    }
}

/// Reads up to [`CONTENT_PREVIEW_MAX_BYTES`] of `path` for operator review,
/// lossily decoded as UTF-8. Returns `None` if the file cannot be opened or read
/// (this is best-effort auditing, not a correctness requirement).
fn read_content_preview(path: &Path) -> Option<String> {
    let extended = to_extended_path(path)?;
    let mut file = std::fs::File::open(extended).ok()?;
    let mut buf = vec![0u8; CONTENT_PREVIEW_MAX_BYTES];
    let n = std::io::Read::read(&mut file, &mut buf).ok()?;
    buf.truncate(n);
    Some(String::from_utf8_lossy(&buf).into_owned())
}

// ---------------------------------------------------------------------------
// JSON result types
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Serialize)]
struct Counts {
    scanned: u32,
    deleted: usize,
    would_delete: usize,
    skipped: usize,
    errors: u32,
}

#[derive(Debug, serde::Serialize)]
struct ActionEntry {
    path: String,
    family: &'static str,
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content_preview: Option<String>,
}

#[derive(Debug, serde::Serialize)]
struct EngineOutput {
    status: &'static str,
    dry_run: bool,
    counts: Counts,
    deleted: Vec<ActionEntry>,
    would_delete: Vec<ActionEntry>,
    skipped: Vec<ActionEntry>,
}

#[derive(Debug, serde::Serialize)]
struct ErrorOutput {
    status: &'static str,
    message: String,
}

// ---------------------------------------------------------------------------
// Scan engine
// ---------------------------------------------------------------------------

/// Thread-safe accumulator for scan results. Matches are rare relative to the
/// total number of scanned entries, so a plain `Mutex<Vec<_>>` per bucket is
/// simple and does not become a contention bottleneck.
struct Collector {
    scanned: AtomicU32,
    errors: AtomicU32,
    deleted: Mutex<Vec<ActionEntry>>,
    would_delete: Mutex<Vec<ActionEntry>>,
    skipped: Mutex<Vec<ActionEntry>>,
}

impl Collector {
    fn new() -> Self {
        Self {
            scanned: AtomicU32::new(0),
            errors: AtomicU32::new(0),
            deleted: Mutex::new(Vec::new()),
            would_delete: Mutex::new(Vec::new()),
            skipped: Mutex::new(Vec::new()),
        }
    }

    fn push_deleted(&self, entry: ActionEntry) {
        self.deleted
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(entry);
    }

    fn push_would_delete(&self, entry: ActionEntry) {
        self.would_delete
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(entry);
    }

    fn push_skipped(&self, entry: ActionEntry) {
        self.skipped
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(entry);
    }

    fn into_output(self, dry_run: bool) -> EngineOutput {
        let deleted = self
            .deleted
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner);
        let would_delete = self
            .would_delete
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner);
        let skipped = self
            .skipped
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner);

        let counts = Counts {
            scanned: self.scanned.load(Ordering::Relaxed),
            deleted: deleted.len(),
            would_delete: would_delete.len(),
            skipped: skipped.len(),
            errors: self.errors.load(Ordering::Relaxed),
        };

        EngineOutput {
            status: "success",
            dry_run,
            counts,
            deleted,
            would_delete,
            skipped,
        }
    }
}

/// Internal implementation of the scan (and, unless `dry_run`, delete) operation.
///
/// This function:
/// 1. Configures a parallel walker with all ignore rules disabled (junk hides in
///    gitignored trees) except a `.git` directory skip
/// 2. Scans the file system using multiple threads
/// 3. Classifies each entry against the active match families
/// 4. Applies the zero-byte safety gate to `$null` file matches
/// 5. Deletes (or, in dry-run, records) matching entries via Win32 APIs
fn run_engine(root_path: &str, options: NukeOptions) -> Result<EngineOutput, String> {
    let families = resolve_families(&options);
    if !families.any() {
        return Err(
            "no match family selected: enable at least one of reserved, dollar-null, or path-mangle"
                .to_string(),
        );
    }

    let collector = Collector::new();

    // Configure the parallel walker
    // - Uses work-stealing queue for load balancing across threads
    // - Automatically scales to CPU core count
    // - Skips .git directories to avoid repository corruption
    // - Ignores hidden file settings and all .gitignore/.ignore rules (we want to
    //   scan everything - junk hides in gitignored trees)
    let walker = WalkBuilder::new(root_path)
        .hidden(false) // Scan hidden files and directories
        .git_ignore(false) // Don't respect .gitignore files
        .git_global(false) // Don't respect global gitignore
        .git_exclude(false) // Don't respect .git/info/exclude
        .require_git(false) // Don't require a git repository
        .ignore(false) // Don't respect .ignore files
        .parents(false) // Don't look for ignore files in parent directories
        .filter_entry(|entry| {
            // Skip .git directories entirely to avoid repository corruption
            entry.file_name() != ".git"
        })
        .build_parallel();

    // Execute parallel walk. Each thread gets its own closure instance for
    // lock-free scanning; matches are pushed into the shared Collector.
    walker.run(|| {
        let collector = &collector;

        Box::new(move |result| {
            match result {
                Ok(entry) => {
                    collector.scanned.fetch_add(1, Ordering::Relaxed);
                    process_entry(&entry, families, options, collector);
                }
                Err(_) => {
                    // Error during traversal (permission denied, symlink loop, etc.)
                    collector.errors.fetch_add(1, Ordering::Relaxed);
                }
            }
            ignore::WalkState::Continue
        })
    });

    Ok(collector.into_output(options.dry_run != 0))
}

/// Evaluates a single walk entry against the active match families and, if it
/// matches, applies the zero-byte gate (for `$null` files) and either records a
/// would-delete candidate (dry-run) or attempts the Win32 delete.
fn process_entry(
    entry: &ignore::DirEntry,
    families: ActiveFamilies,
    options: NukeOptions,
    collector: &Collector,
) {
    let Some(file_type) = entry.file_type() else {
        return;
    };
    let is_dir = file_type.is_dir();
    let is_file = file_type.is_file();

    // Only real files are candidates by default; directories only when opted in.
    // Symlinks, reparse points, and unknown types are never candidates.
    let dirs_enabled = options.include_dirs != 0;
    if !(is_file || is_dir && dirs_enabled) {
        return;
    }

    // Never touch the scan root itself (depth 0), even if its name happens to
    // match a family - it is the directory being scanned, not a candidate.
    if is_dir && entry.depth() == 0 {
        return;
    }

    let Some(family) = classify(entry.file_name(), families) else {
        return;
    };

    let kind = if is_dir { "dir" } else { "file" };
    let path_display = entry.path().display().to_string();

    // Zero-byte safety gate: only applies to real (non-directory) $null files.
    // Directories get an equivalent safety net for free: RemoveDirectoryW simply
    // refuses to remove a non-empty directory.
    let mut size: Option<u64> = None;
    let mut content_preview: Option<String> = None;

    if family == MatchFamily::DollarNull && is_file {
        match entry.metadata() {
            Ok(meta) => {
                let len = meta.len();
                size = Some(len);
                if len > 0 {
                    content_preview = read_content_preview(entry.path());
                }
                if let Some(reason) = dollar_null_skip_reason(len, options.allow_nonempty != 0) {
                    collector.push_skipped(ActionEntry {
                        path: path_display,
                        family: family.as_str(),
                        kind,
                        reason: Some(reason),
                        size,
                        content_preview,
                    });
                    return;
                }
            }
            Err(e) => {
                collector.errors.fetch_add(1, Ordering::Relaxed);
                collector.push_skipped(ActionEntry {
                    path: path_display,
                    family: family.as_str(),
                    kind,
                    reason: Some(format!("failed to read metadata: {e}")),
                    size: None,
                    content_preview: None,
                });
                return;
            }
        }
    }

    if options.dry_run != 0 {
        collector.push_would_delete(ActionEntry {
            path: path_display,
            family: family.as_str(),
            kind,
            reason: None,
            size,
            content_preview,
        });
        return;
    }

    if is_dir {
        match delete_dir_win32(entry.path()) {
            Ok(()) => collector.push_deleted(ActionEntry {
                path: path_display,
                family: family.as_str(),
                kind,
                reason: None,
                size,
                content_preview,
            }),
            Err(DirDeleteOutcome::NotEmpty) => collector.push_skipped(ActionEntry {
                path: path_display,
                family: family.as_str(),
                kind,
                reason: Some("directory not empty (skipped, not deleted)".to_string()),
                size,
                content_preview,
            }),
            Err(DirDeleteOutcome::Failed(reason)) => {
                collector.errors.fetch_add(1, Ordering::Relaxed);
                collector.push_skipped(ActionEntry {
                    path: path_display,
                    family: family.as_str(),
                    kind,
                    reason: Some(reason),
                    size,
                    content_preview,
                });
            }
        }
    } else {
        match delete_file_win32(entry.path()) {
            Ok(()) => collector.push_deleted(ActionEntry {
                path: path_display,
                family: family.as_str(),
                kind,
                reason: None,
                size,
                content_preview,
            }),
            Err(reason) => {
                collector.errors.fetch_add(1, Ordering::Relaxed);
                collector.push_skipped(ActionEntry {
                    path: path_display,
                    family: family.as_str(),
                    kind,
                    reason: Some(reason),
                    size,
                    content_preview,
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Misc exports
// ---------------------------------------------------------------------------

/// Version information for the library
#[no_mangle]
pub extern "C" fn nuker_core_version() -> *const c_char {
    VERSION_CSTR.as_ptr() as *const c_char
}

/// Test function to verify DLL is loaded correctly
#[no_mangle]
pub extern "C" fn nuker_core_test() -> u32 {
    // Return a magic number to verify DLL loaded correctly
    0xDEAD_BEEF
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
    fn matches_reserved_is_case_insensitive() {
        assert!(matches_reserved(OsStr::new("nul")));
        assert!(matches_reserved(OsStr::new("NUL")));
        assert!(matches_reserved(OsStr::new("Nul")));
        assert!(!matches_reserved(OsStr::new("$null")));
        assert!(!matches_reserved(OsStr::new("nul.txt")));
    }

    #[test]
    fn literal_dollar_null_matching_is_narrow() {
        assert!(matches_dollar_null(OsStr::new("$null")));
        assert!(matches_dollar_null(OsStr::new("$NULL. ")));
        assert!(!matches_dollar_null(OsStr::new("$null.txt")));
        assert!(!matches_dollar_null(OsStr::new("nul")));
    }

    #[test]
    fn path_mangle_matches_narrow_pattern() {
        assert!(matches_path_mangle(OsStr::new("foo;C")));
        assert!(matches_path_mangle(OsStr::new("bar;C:")));
        assert!(matches_path_mangle(OsStr::new("headscale-ops;C")));
        assert!(matches_path_mangle(OsStr::new(";Z")));
    }

    #[test]
    fn path_mangle_rejects_lookalikes() {
        assert!(!matches_path_mangle(OsStr::new("a;b.txt")));
        assert!(!matches_path_mangle(OsStr::new("semi;colon")));
        assert!(!matches_path_mangle(OsStr::new("x;CD")));
        assert!(!matches_path_mangle(OsStr::new("plainfile.txt")));
        assert!(!matches_path_mangle(OsStr::new("C")));
    }

    #[test]
    fn classify_respects_active_families() {
        let all = ActiveFamilies {
            reserved: true,
            dollar_null: true,
            path_mangle: true,
        };
        assert_eq!(
            classify(OsStr::new("nul"), all),
            Some(MatchFamily::Reserved)
        );
        assert_eq!(
            classify(OsStr::new("$null"), all),
            Some(MatchFamily::DollarNull)
        );
        assert_eq!(
            classify(OsStr::new("foo;C"), all),
            Some(MatchFamily::PathMangle)
        );
        assert_eq!(classify(OsStr::new("readme.txt"), all), None);

        let none = ActiveFamilies {
            reserved: false,
            dollar_null: false,
            path_mangle: false,
        };
        assert_eq!(classify(OsStr::new("nul"), none), None);
        assert!(!none.any());
        assert!(all.any());
    }

    #[test]
    fn classify_families_are_independently_toggleable() {
        let only_path_mangle = ActiveFamilies {
            reserved: false,
            dollar_null: false,
            path_mangle: true,
        };
        assert_eq!(classify(OsStr::new("$null"), only_path_mangle), None);
        assert_eq!(
            classify(OsStr::new("foo;C"), only_path_mangle),
            Some(MatchFamily::PathMangle)
        );
    }

    #[test]
    fn zero_byte_gate_allows_empty_files() {
        assert!(dollar_null_skip_reason(0, false).is_none());
        assert!(dollar_null_skip_reason(0, true).is_none());
    }

    #[test]
    fn zero_byte_gate_blocks_nonempty_by_default() {
        let reason = dollar_null_skip_reason(877, false);
        assert!(reason.is_some());
        assert!(reason.unwrap().contains("877 bytes"));
    }

    #[test]
    fn zero_byte_gate_allow_nonempty_overrides() {
        assert!(dollar_null_skip_reason(877, true).is_none());
    }

    #[test]
    fn test_scan_stats_error() {
        let stats = ScanStats::error();
        assert_eq!(stats.files_scanned, 0);
        assert_eq!(stats.files_deleted, 0);
        assert_eq!(stats.errors, 1);
    }

    #[test]
    fn extended_path_regular() {
        let path = Path::new(r"C:\test\file.txt");
        assert_eq!(
            to_extended_path(path).as_deref(),
            Some(r"\\?\C:\test\file.txt")
        );
    }

    #[test]
    fn extended_path_unc() {
        let path = Path::new(r"\\server\share\file.txt");
        assert_eq!(
            to_extended_path(path).as_deref(),
            Some(r"\\?\UNC\server\share\file.txt")
        );
    }

    #[test]
    fn extended_path_already_extended_is_unchanged() {
        let path = Path::new(r"\\?\C:\already\extended");
        assert_eq!(
            to_extended_path(path).as_deref(),
            Some(r"\\?\C:\already\extended")
        );
    }

    #[test]
    fn describe_win32_error_maps_known_codes() {
        assert!(describe_win32_error(5).contains("access denied"));
        assert!(describe_win32_error(145).contains("not empty"));
        assert!(describe_win32_error(999_999).contains("999999"));
    }

    #[test]
    fn resolve_families_maps_options() {
        let options = NukeOptions {
            match_reserved: 1,
            match_dollar_null: 0,
            match_path_mangle: 1,
            include_dirs: 0,
            dry_run: 0,
            allow_nonempty: 0,
        };
        let families = resolve_families(&options);
        assert!(families.reserved);
        assert!(!families.dollar_null);
        assert!(families.path_mangle);
    }
}
