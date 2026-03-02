//! Filesystem operations with FFI exports.
//!
//! Consolidated from the legacy pcai_fs crate into pcai_core_lib.

use crate::error::PcaiStatus;
use crate::string::PcaiStringBuffer;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::ffi::CStr;
use std::fs;
use std::os::raw::c_char;
use std::path::Path;
use std::time::Instant;
use walkdir::WalkDir;

pub mod ops;

#[derive(Debug, Serialize, Deserialize)]
struct ReplaceResult {
    status: String,
    files_scanned: usize,
    files_changed: usize,
    matches_replaced: usize,
    elapsed_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl ReplaceResult {
    fn success(files_scanned: usize, files_changed: usize, matches_replaced: usize, elapsed_ms: u64) -> Self {
        Self {
            status: "success".to_string(),
            files_scanned,
            files_changed,
            matches_replaced,
            elapsed_ms,
            error: None,
        }
    }

    fn error(msg: String) -> Self {
        Self {
            status: "error".to_string(),
            files_scanned: 0,
            files_changed: 0,
            matches_replaced: 0,
            elapsed_ms: 0,
            error: Some(msg),
        }
    }
}

unsafe fn c_str_to_str<'a>(ptr: *const c_char) -> Result<&'a str, PcaiStatus> {
    if ptr.is_null() {
        return Err(PcaiStatus::NullPointer);
    }
    CStr::from_ptr(ptr).to_str().map_err(|_| PcaiStatus::InvalidUtf8)
}

fn create_backup(path: &Path) -> std::io::Result<()> {
    let backup_path = path.with_extension(format!(
        "{}.bak",
        path.extension().and_then(|s| s.to_str()).unwrap_or("")
    ));
    fs::copy(path, backup_path)?;
    Ok(())
}

fn replace_in_file_impl(
    file_path: &Path,
    pattern: &str,
    replacement: &str,
    is_regex: bool,
    backup: bool,
) -> Result<usize, String> {
    let content = fs::read_to_string(file_path).map_err(|e| format!("Failed to read file: {}", e))?;

    let (new_content, count) = if is_regex {
        let re = Regex::new(pattern).map_err(|e| format!("Invalid regex pattern: {}", e))?;
        let new = re.replace_all(&content, replacement).to_string();
        let matches = re.find_iter(&content).count();
        (new, matches)
    } else {
        let matches = content.matches(pattern).count();
        let new = content.replace(pattern, replacement);
        (new, matches)
    };

    if count > 0 {
        if backup {
            create_backup(file_path).map_err(|e| format!("Failed to create backup: {}", e))?;
        }

        fs::write(file_path, new_content).map_err(|e| format!("Failed to write file: {}", e))?;
    }

    Ok(count)
}

fn matches_pattern(file_name: &str, pattern: &str) -> bool {
    if pattern == "*" || pattern == "*.*" {
        return true;
    }

    if pattern == file_name {
        return true;
    }

    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.len() == 2 {
            let prefix = parts[0];
            let suffix = parts[1];
            return file_name.starts_with(prefix) && file_name.ends_with(suffix);
        }
    }

    false
}

#[no_mangle]
pub extern "C" fn pcai_fs_version() -> u32 {
    1
}

#[no_mangle]
pub unsafe extern "C" fn pcai_delete_fs_item(path: *const c_char, recursive: bool) -> PcaiStatus {
    let path_str = match c_str_to_str(path) {
        Ok(s) => s,
        Err(e) => return e,
    };
    ops::delete_item(path_str, recursive)
}

#[no_mangle]
pub unsafe extern "C" fn pcai_replace_in_file(
    file_path: *const c_char,
    pattern: *const c_char,
    replacement: *const c_char,
    is_regex: bool,
    backup: bool,
) -> PcaiStatus {
    let file_path_str = match c_str_to_str(file_path) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let pattern_str = match c_str_to_str(pattern) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let replacement_str = match c_str_to_str(replacement) {
        Ok(s) => s,
        Err(e) => return e,
    };

    let path = Path::new(file_path_str);
    if !path.exists() {
        return PcaiStatus::PathNotFound;
    }

    match replace_in_file_impl(path, pattern_str, replacement_str, is_regex, backup) {
        Ok(_) => PcaiStatus::Success,
        Err(_) => PcaiStatus::IoError,
    }
}

#[no_mangle]
pub unsafe extern "C" fn pcai_replace_in_files(
    root_path: *const c_char,
    file_pattern: *const c_char,
    content_pattern: *const c_char,
    replacement: *const c_char,
    is_regex: bool,
    backup: bool,
) -> PcaiStringBuffer {
    let start = Instant::now();

    let root_str = match c_str_to_str(root_path) {
        Ok(s) => s,
        Err(_) => {
            let result = ReplaceResult::error("Invalid root path pointer".to_string());
            let json = serde_json::to_string(&result).unwrap_or_default();
            return PcaiStringBuffer::from_string(&json);
        }
    };
    let file_pat = match c_str_to_str(file_pattern) {
        Ok(s) => s,
        Err(_) => {
            let result = ReplaceResult::error("Invalid file pattern pointer".to_string());
            let json = serde_json::to_string(&result).unwrap_or_default();
            return PcaiStringBuffer::from_string(&json);
        }
    };
    let content_pat = match c_str_to_str(content_pattern) {
        Ok(s) => s,
        Err(_) => {
            let result = ReplaceResult::error("Invalid content pattern pointer".to_string());
            let json = serde_json::to_string(&result).unwrap_or_default();
            return PcaiStringBuffer::from_string(&json);
        }
    };
    let repl = match c_str_to_str(replacement) {
        Ok(s) => s,
        Err(_) => {
            let result = ReplaceResult::error("Invalid replacement pointer".to_string());
            let json = serde_json::to_string(&result).unwrap_or_default();
            return PcaiStringBuffer::from_string(&json);
        }
    };

    let mut files = Vec::new();
    for entry in WalkDir::new(root_str)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            if let Some(name) = entry.file_name().to_str() {
                if matches_pattern(name, file_pat) {
                    files.push(entry.path().to_path_buf());
                }
            }
        }
    }

    let files_scanned = files.len();
    let results: Vec<(bool, usize)> = files
        .par_iter()
        .map(
            |path| match replace_in_file_impl(path, content_pat, repl, is_regex, backup) {
                Ok(count) => (count > 0, count),
                Err(_) => (false, 0),
            },
        )
        .collect();

    let files_changed = results.iter().filter(|(changed, _)| *changed).count();
    let matches_replaced = results.iter().map(|(_, count)| count).sum();
    let elapsed_ms = start.elapsed().as_millis() as u64;

    let result = ReplaceResult::success(files_scanned, files_changed, matches_replaced, elapsed_ms);

    let json = serde_json::to_string(&result)
        .unwrap_or_else(|_| r#"{"status":"error","error":"JSON serialization failed"}"#.to_string());

    PcaiStringBuffer::from_string(&json)
}
