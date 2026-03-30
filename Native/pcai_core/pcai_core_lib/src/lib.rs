//! PCAI Core Library - FFI Utilities and Shared Types
//!
//! This crate provides FFI entry points for C# P/Invoke interop.
//! Raw pointer arguments are validated with null checks before dereference.

#![expect(
    clippy::not_unsafe_ptr_arg_deref,
    reason = "FFI entry points receive raw pointers from C# P/Invoke; each \
              function performs an explicit null check before any dereference, \
              making the dereference safe even though the function is not marked \
              `unsafe` (matching the `extern \"C\"` calling convention requirement)"
)]

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
#[cfg(feature = "nvml")]
pub mod preflight;
pub mod process_lasso;
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

/// Return real-time utilization metrics for the GPU at `device_index` as JSON.
///
/// Returns a null pointer when NVML is unavailable or the index is out of range.
/// The caller must free the returned string with [`pcai_free_string`].
///
/// # Safety
///
/// This function is safe to call from any thread.
#[cfg(feature = "nvml")]
#[no_mangle]
pub extern "C" fn pcai_gpu_utilization_json(device_index: u32) -> *mut c_char {
    match gpu::gpu_utilization(device_index) {
        Ok(util) => match serde_json::to_string(&util) {
            Ok(json) => rust_str_to_c(&json),
            Err(_) => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Return the CUDA driver version as a JSON object `{"major": X, "minor": Y}`.
///
/// Returns a null pointer when NVML is unavailable.
/// The caller must free the returned string with [`pcai_free_string`].
///
/// # Safety
///
/// This function is safe to call from any thread.
#[cfg(feature = "nvml")]
#[no_mangle]
pub extern "C" fn pcai_cuda_driver_version_json() -> *mut c_char {
    match gpu::cuda_driver_version() {
        Ok((major, minor)) => {
            let json = serde_json::json!({
                "major": major,
                "minor": minor
            });
            match serde_json::to_string(&json) {
                Ok(s) => rust_str_to_c(&s),
                Err(_) => std::ptr::null_mut(),
            }
        }
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
/// Return a Process Lasso snapshot as JSON.
///
/// # Safety
///
/// `config_path` and `log_path`, when non-null, must be valid null-terminated UTF-8 strings.
pub unsafe extern "C" fn pcai_get_process_lasso_snapshot_json(
    config_path: *const c_char,
    log_path: *const c_char,
    lookback_minutes: u32,
) -> *mut c_char {
    let config_path =
        unsafe { string::c_str_to_rust(config_path) }.unwrap_or("C:\\ProgramData\\ProcessLasso\\config\\prolasso.ini");
    let log_path =
        unsafe { string::c_str_to_rust(log_path) }.unwrap_or("C:\\ProgramData\\ProcessLasso\\logs\\processlasso.log");

    match process_lasso::collect_snapshot(config_path, log_path, lookback_minutes) {
        Ok(snapshot) => match serde_json::to_string(&snapshot) {
            Ok(json) => rust_str_to_c(&json),
            Err(_) => std::ptr::null_mut(),
        },
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
    #[cfg(windows)]
    {
        let devices = telemetry::usb::collect_usb_diagnostics();
        match serde_json::to_string(&devices) {
            Ok(json) => rust_str_to_c(&json),
            Err(_) => std::ptr::null_mut(),
        }
    }
    #[cfg(not(windows))]
    rust_str_to_c("[]")
}

#[no_mangle]
/// Return a JSON array of PNP devices.
///
/// # Safety
///
/// `class_filter` must be a valid, null-terminated C string if not null.
pub unsafe extern "C" fn pcai_get_pnp_devices_json(class_filter: *const c_char) -> *mut c_char {
    let _ = class_filter;
    #[cfg(windows)]
    {
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
    #[cfg(not(windows))]
    rust_str_to_c("[]")
}

#[no_mangle]
pub extern "C" fn pcai_get_disk_health_json() -> *mut c_char {
    #[cfg(windows)]
    {
        let health = telemetry::disk_health::collect_disk_health();
        match serde_json::to_string(&health) {
            Ok(json) => rust_str_to_c(&json),
            Err(_) => std::ptr::null_mut(),
        }
    }
    #[cfg(not(windows))]
    rust_str_to_c("[]")
}

#[no_mangle]
pub extern "C" fn pcai_sample_hardware_events_json(days: u32, max_events: u32) -> *mut c_char {
    let _ = days;
    let _ = max_events;
    #[cfg(windows)]
    {
        let events = telemetry::event_log::sample_hardware_events(days, max_events);
        match serde_json::to_string(&events) {
            Ok(json) => rust_str_to_c(&json),
            Err(_) => std::ptr::null_mut(),
        }
    }
    #[cfg(not(windows))]
    rust_str_to_c("[]")
}

#[no_mangle]
pub extern "C" fn pcai_get_network_throughput_json() -> *mut c_char {
    #[cfg(windows)]
    {
        let interfaces = telemetry::network::collect_network_diagnostics();
        match serde_json::to_string(&interfaces) {
            Ok(json) => rust_str_to_c(&json),
            Err(_) => std::ptr::null_mut(),
        }
    }
    #[cfg(not(windows))]
    rust_str_to_c("[]")
}

#[no_mangle]
pub extern "C" fn pcai_get_process_history_json() -> *mut c_char {
    #[cfg(windows)]
    {
        let history = telemetry::process::collect_process_telemetry();
        match serde_json::to_string(&history) {
            Ok(json) => rust_str_to_c(&json),
            Err(_) => std::ptr::null_mut(),
        }
    }
    #[cfg(not(windows))]
    rust_str_to_c("[]")
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

/// Run a GPU preflight readiness check and return the result as a JSON string.
///
/// # Parameters
///
/// * `model_path` -- Path to a GGUF model file (null for VRAM-only inventory).
/// * `context_length` -- Context length override (0 = use model default).
/// * `required_mb` -- Minimum VRAM required in MB (used when `model_path` is null).
///
/// # Returns
///
/// A heap-allocated JSON string.  Caller must free with [`pcai_free_string`].
/// Returns null only if JSON serialisation fails (should never happen).
///
/// JSON schema:
/// ```json
/// {
///   "verdict": "go | warn | fail",
///   "reason": "human-readable explanation",
///   "model_estimate_mb": 5800,
///   "best_gpu_index": 1,
///   "gpus": [{ "index": 0, "name": "...", "total_mb": 8192, ... }]
/// }
/// ```
///
/// # Safety
///
/// `model_path` must be null or a valid null-terminated UTF-8 C string.
#[cfg(feature = "nvml")]
#[no_mangle]
pub extern "C" fn pcai_gpu_preflight_json(
    model_path: *const c_char,
    context_length: u64,
    required_mb: u64,
) -> *mut c_char {
    let result = if model_path.is_null() {
        preflight::check_vram_state(required_mb)
    } else {
        match unsafe { CStr::from_ptr(model_path) }.to_str() {
            Ok(path_str) => preflight::check_readiness(path_str, context_length),
            Err(_) => Ok(preflight::PreflightResult {
                verdict: preflight::Verdict::Fail,
                reason: "Invalid UTF-8 in model path".to_owned(),
                model_estimate_mb: 0,
                best_gpu_index: None,
                gpus: Vec::new(),
            }),
        }
    };

    let preflight_result = match result {
        Ok(r) => r,
        Err(e) => preflight::PreflightResult {
            verdict: preflight::Verdict::Fail,
            reason: format!("Preflight error: {e}"),
            model_estimate_mb: 0,
            best_gpu_index: None,
            gpus: Vec::new(),
        },
    };

    match serde_json::to_string(&preflight_result) {
        Ok(json) => rust_str_to_c(&json),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Run a roofline performance analysis for all detected GPUs and return a
/// JSON array of [`gpu::roofline::RooflineAnalysis`] results.
///
/// # Parameters
///
/// * `model_params_billions` -- Model size in billions of parameters (e.g. 7.0).
/// * `quant_bits` -- Quantization bit-width per parameter (e.g. 4.5 for Q4_K_M).
/// * `context_length` -- Context length for the analysis (e.g. 4096).
/// * `actual_toks` -- Measured decode tok/s.  Pass `0.0` or negative for
///   "no measurement" (efficiency fields will be null in JSON).
///
/// # Returns
///
/// A heap-allocated JSON string.  Caller must free with [`pcai_free_string`].
/// Returns null only if JSON serialisation fails or no GPUs have known specs.
///
/// # Safety
///
/// This function is safe to call from any thread; it does not accept pointer
/// arguments.
#[cfg(feature = "nvml")]
#[no_mangle]
pub extern "C" fn pcai_gpu_roofline_json(
    model_params_billions: f64,
    quant_bits: f64,
    context_length: u64,
    actual_toks: f64,
) -> *mut c_char {
    let gpus = gpu::gpu_inventory().unwrap_or_default();
    let actual = if actual_toks > 0.0 { Some(actual_toks) } else { None };

    let mut analyses = Vec::new();
    for g in &gpus {
        if let Some(specs) = gpu::roofline::GpuSpecs::from_compute_capability(&g.compute_capability, &g.name) {
            analyses.push(gpu::roofline::analyze_roofline(
                &specs,
                model_params_billions,
                quant_bits,
                context_length,
                actual,
            ));
        }
    }

    match serde_json::to_string(&analyses) {
        Ok(json) => rust_str_to_c(&json),
        Err(_) => std::ptr::null_mut(),
    }
}
