//! PCAI Prompt Engine - Native Template Assembly
//!
//! Provides high-performance string interpolation and prompt assembly
//! to minimize PowerShell managed overhead.

use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize)]
pub struct PromptTemplate {
    pub name: String,
    pub template: String,
}

/// Simple high-performance placeholder replacement.
/// Replaces {{key}} with value.
pub fn assemble_prompt(template: &str, variables: &HashMap<String, String>) -> String {
    let mut result = template.to_string();
    for (key, value) in variables {
        let placeholder = format!("{{{{{}}}}}", key);
        result = result.replace(&placeholder, value);
    }
    result
}

// ============================================================================
// FFI Implementation
// ============================================================================

use crate::string::PcaiStringBuffer;
use std::ffi::CStr;
use std::os::raw::c_char;

/// Assembles a prompt from a JSON-formatted variable map.
/// Input variables must be a JSON string: {"key": "value"}
#[no_mangle]
pub extern "C" fn pcai_assemble_prompt(template: *const c_char, json_vars: *const c_char) -> PcaiStringBuffer {
    let template = unsafe {
        if template.is_null() {
            return crate::string::error_buffer(crate::error::PcaiStatus::InvalidArgument);
        }
        CStr::from_ptr(template).to_string_lossy()
    };

    let json_vars = unsafe {
        if json_vars.is_null() {
            return crate::string::error_buffer(crate::error::PcaiStatus::InvalidArgument);
        }
        CStr::from_ptr(json_vars).to_string_lossy()
    };

    let vars: HashMap<String, String> = match serde_json::from_str(&json_vars) {
        Ok(v) => v,
        Err(_) => return crate::string::error_buffer(crate::error::PcaiStatus::JsonError),
    };

    let assembled = assemble_prompt(&template, &vars);
    crate::string::rust_str_to_buffer(&assembled)
}
