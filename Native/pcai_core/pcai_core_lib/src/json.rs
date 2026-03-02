use crate::string::rust_str_to_c;
use std::ffi::CStr;
use std::os::raw::c_char;

/// Extracts JSON from a string that might contain markdown backsticks.
pub fn extract_json_from_markdown(input: &str) -> Option<String> {
    if let Some(start) = input.find("```json") {
        let after_start = &input[start + 7..];
        if let Some(end) = after_start.find("```") {
            return Some(after_start[..end].trim().to_string());
        }
    }

    let trimmed = input.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        return Some(trimmed.to_string());
    }

    None
}

/// FFI export for JSON extraction.
#[no_mangle]
pub extern "C" fn pcai_extract_json(input: *const c_char) -> *mut c_char {
    if input.is_null() {
        return std::ptr::null_mut();
    }
    let c_str = unsafe { CStr::from_ptr(input) };
    let text = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match extract_json_from_markdown(text) {
        Some(json) => rust_str_to_c(&json),
        None => std::ptr::null_mut(),
    }
}

/// Validates that a string is valid JSON.
#[no_mangle]
pub extern "C" fn pcai_is_valid_json(input: *const c_char) -> bool {
    if input.is_null() {
        return false;
    }
    let c_str = unsafe { CStr::from_ptr(input) };
    let text = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return false,
    };

    serde_json::from_str::<serde_json::Value>(text).is_ok()
}
