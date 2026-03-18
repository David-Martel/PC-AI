//! PCAI Core Library - FFI Utilities and Shared Types
//!
//! This crate provides FFI entry points for C# P/Invoke interop.
//! Raw pointer arguments are validated with null checks before dereference.

// Allow raw pointer dereference in non-unsafe FFI functions - this is intentional
// as all FFI entry points perform null checks before dereferencing.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

pub mod error;
pub mod fs;
#[cfg(feature = "functiongemma")]
pub mod functiongemma;
#[cfg(feature = "nvml")]
pub mod gpu;
pub mod hash;
pub mod json;
pub mod path;
pub mod performance;
pub mod prompt_engine;
pub mod result;
pub mod search;
pub mod string;
pub mod system;
pub mod telemetry;
pub mod tokenizer;
pub mod vmm_health;

pub use error::PcaiStatus;
pub use json::{extract_json_from_markdown, pcai_extract_json, pcai_is_valid_json};
pub use path::{normalize_path, parse_path_ffi, PathStyle};
pub use result::PcaiResult;
pub use string::{c_str_to_rust, json_to_buffer_pretty, rust_str_to_c, PcaiStringBuffer};

include!(concat!(env!("OUT_DIR"), "/version.rs"));

/// Magic number for DLL verification
pub const MAGIC_NUMBER: u32 = 0x5043_4149;

#[no_mangle]
pub extern "C" fn pcai_core_version() -> *const c_char {
    VERSION_CSTR.as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn pcai_core_test() -> u32 {
    MAGIC_NUMBER
}

#[no_mangle]
pub extern "C" fn pcai_search_version() -> *const c_char {
    pcai_core_version()
}

#[no_mangle]
pub extern "C" fn pcai_free_string(buffer: *mut c_char) {
    if !buffer.is_null() {
        unsafe {
            let _ = CString::from_raw(buffer);
        }
    }
}

#[no_mangle]
pub extern "C" fn pcai_string_copy(input: *const c_char) -> *mut c_char {
    if input.is_null() {
        return std::ptr::null_mut();
    }
    let c_str = unsafe { CStr::from_ptr(input) };
    match c_str.to_str() {
        Ok(s) => rust_str_to_c(s),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn pcai_cpu_count() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

#[no_mangle]
pub extern "C" fn pcai_estimate_tokens(text: *const c_char) -> usize {
    let text = unsafe {
        if text.is_null() {
            return 0;
        }
        std::ffi::CStr::from_ptr(text)
    };

    match text.to_str() {
        Ok(s) => tokenizer::estimate_tokens(s),
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn pcai_check_resource_safety(gpu_limit: f32) -> i32 {
    if telemetry::check_resource_safety(gpu_limit) {
        1
    } else {
        0
    }
}

#[no_mangle]
pub extern "C" fn pcai_get_system_telemetry_json() -> *mut c_char {
    let tel = telemetry::collect_telemetry();
    match serde_json::to_string(&tel) {
        Ok(json) => rust_str_to_c(&json),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn pcai_get_vmm_health_json() -> *mut c_char {
    let health = vmm_health::check_vmm_health();
    match serde_json::to_string(&health) {
        Ok(json) => rust_str_to_c(&json),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn pcai_query_full_context_json() -> *mut c_char {
    // Aggregates everything for the LLM context
    #[derive(serde::Serialize)]
    struct FullContext {
        system: system::SystemSummary,
        telemetry: telemetry::SystemTelemetry,
        vmm: vmm_health::VmmHealth,
    }

    let context = FullContext {
        system: system::get_system_summary(),
        telemetry: telemetry::collect_telemetry(),
        vmm: vmm_health::check_vmm_health(),
    };

    match serde_json::to_string(&context) {
        Ok(json) => rust_str_to_c(&json),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn pcai_get_usb_deep_diagnostics_json() -> *mut c_char {
    let devices = telemetry::usb::collect_usb_diagnostics();
    match serde_json::to_string(&devices) {
        Ok(json) => rust_str_to_c(&json),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn pcai_get_pnp_devices_json(class_filter: *const c_char) -> *mut c_char {
    let filter = if class_filter.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(class_filter).to_str().ok() }
    };

    let devices = telemetry::pnp::collect_pnp_devices(filter);
    match serde_json::to_string(&devices) {
        Ok(json) => rust_str_to_c(&json),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn pcai_get_disk_health_json() -> *mut c_char {
    let health = telemetry::disk_health::collect_disk_health();
    match serde_json::to_string(&health) {
        Ok(json) => rust_str_to_c(&json),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn pcai_sample_hardware_events_json(days: u32, max_events: u32) -> *mut c_char {
    let events = telemetry::event_log::sample_hardware_events(days, max_events);
    match serde_json::to_string(&events) {
        Ok(json) => rust_str_to_c(&json),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn pcai_get_network_throughput_json() -> *mut c_char {
    let interfaces = telemetry::network::collect_network_diagnostics();
    match serde_json::to_string(&interfaces) {
        Ok(json) => rust_str_to_c(&json),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn pcai_get_process_history_json() -> *mut c_char {
    let history = telemetry::process::collect_process_telemetry();
    match serde_json::to_string(&history) {
        Ok(json) => rust_str_to_c(&json),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn pcai_query_prompt_assembly(template: *const c_char, json_vars: *const c_char) -> PcaiStringBuffer {
    prompt_engine::pcai_assemble_prompt(template, json_vars)
}

#[no_mangle]
pub extern "C" fn pcai_get_usb_problem_info(code: u32) -> *mut c_char {
    pcai_get_pnp_problem_info(code)
}

#[no_mangle]
pub extern "C" fn pcai_get_pnp_problem_info(code: u32) -> *mut c_char {
    match telemetry::device_codes::get_problem_info(code) {
        Some(info) => {
            let json = serde_json::json!({
                "code": info.code,
                "short_description": info.short_description,
                "help_summary": info.help_summary,
                "help_url": info.help_url
            });
            match serde_json::to_string(&json) {
                Ok(s) => rust_str_to_c(&s),
                Err(_) => std::ptr::null_mut(),
            }
        }
        None => std::ptr::null_mut(),
    }
}

// ── GPU FFI exports (feature = "nvml") ───────────────────────────────────────

/// Return the number of NVIDIA GPUs detected by NVML.
///
/// Returns `0` when NVML is unavailable (no driver installed).
/// Returns `-1` on an unexpected NVML error after successful initialisation.
///
/// # Safety
///
/// This function is safe to call from any thread; it does not access
/// raw pointer arguments.
#[cfg(feature = "nvml")]
#[no_mangle]
pub extern "C" fn pcai_gpu_count() -> i32 {
    match gpu::gpu_count() {
        Ok(n) => n as i32,
        Err(_) => -1,
    }
}

/// Return a JSON array describing every NVIDIA GPU in the system.
///
/// Each element is a serialised [`gpu::GpuInfo`] object.  Returns an empty
/// JSON array (`"[]"`) when NVML is unavailable.  Returns a null pointer
/// when JSON serialisation fails (should never happen in practice).
///
/// The caller must free the returned string with [`pcai_free_string`].
///
/// # Safety
///
/// This function is safe to call from any thread.
#[cfg(feature = "nvml")]
#[no_mangle]
pub extern "C" fn pcai_gpu_info_json() -> *mut c_char {
    let gpus = gpu::gpu_inventory().unwrap_or_default();
    match serde_json::to_string(&gpus) {
        Ok(json) => rust_str_to_c(&json),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Return the installed NVIDIA display driver version as a null-terminated
/// C string (e.g. `"561.09"`).
///
/// Returns a null pointer when NVML is unavailable or the query fails.
/// The caller must free the returned string with [`pcai_free_string`].
///
/// # Safety
///
/// This function is safe to call from any thread.
#[cfg(feature = "nvml")]
#[no_mangle]
pub extern "C" fn pcai_driver_version() -> *mut c_char {
    match gpu::driver_version() {
        Ok(ver) => rust_str_to_c(&ver),
        Err(_) => std::ptr::null_mut(),
    }
}
