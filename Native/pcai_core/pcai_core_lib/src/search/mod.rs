//! PCAI Search Module
//!
//! Consolidated search functionality including files, content, and duplicates.

pub mod content;
pub mod duplicates;
pub mod files;
pub mod walker;

use crate::string::PcaiStringBuffer;
use std::os::raw::c_char;

// Re-export FFI from submodules with unified naming if needed,
// but submodules have their own ffi functions usually named _ffi.
// We'll expose the standard pcai_* names here.

#[no_mangle]
pub extern "C" fn pcai_find_files(
    root_path: *const c_char,
    pattern: *const c_char,
    max_results: u64,
) -> PcaiStringBuffer {
    files::find_files_ffi(root_path, pattern, max_results)
}

#[no_mangle]
pub extern "C" fn pcai_find_files_stats(
    root_path: *const c_char,
    pattern: *const c_char,
    max_results: u64,
) -> files::FileSearchStats {
    files::find_files_stats_ffi(root_path, pattern, max_results)
}

#[no_mangle]
pub extern "C" fn pcai_search_content(
    root_path: *const c_char,
    pattern: *const c_char,
    file_pattern: *const c_char,
    max_results: u64,
    context_lines: u32,
) -> PcaiStringBuffer {
    content::search_content_ffi(root_path, pattern, file_pattern, max_results, context_lines)
}

#[no_mangle]
pub extern "C" fn pcai_search_content_stats(
    root_path: *const c_char,
    pattern: *const c_char,
    file_pattern: *const c_char,
    max_results: u64,
) -> content::ContentSearchStats {
    content::search_content_stats_ffi(root_path, pattern, file_pattern, max_results)
}

#[no_mangle]
pub extern "C" fn pcai_find_duplicates(
    root_path: *const c_char,
    min_size: u64,
    include_pattern: *const c_char,
    exclude_pattern: *const c_char,
) -> PcaiStringBuffer {
    duplicates::find_duplicates_ffi(root_path, min_size, include_pattern, exclude_pattern)
}

#[no_mangle]
pub extern "C" fn pcai_find_duplicates_stats(
    root_path: *const c_char,
    min_size: u64,
    include_pattern: *const c_char,
    exclude_pattern: *const c_char,
) -> duplicates::DuplicateStats {
    duplicates::find_duplicates_stats_ffi(root_path, min_size, include_pattern, exclude_pattern)
}
