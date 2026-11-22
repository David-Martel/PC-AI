//! C ABI exports for pcai_media.
//!
//! This module provides a C-compatible FFI interface for calling the
//! pcai-media library from PowerShell via P/Invoke, or from any C/C#
//! caller.
//!
//! # Safety
//!
//! All FFI functions accept raw pointers from C callers. The safety
//! requirements are documented on each function. This module allows
//! `clippy::not_unsafe_ptr_arg_deref` because marking FFI functions as
//! `unsafe` does not help C/C#/PowerShell callers who cannot see Rust's
//! `unsafe` keyword.
//!
//! ## Thread Safety
//!
//! - Global state is protected by `Mutex`
//! - The Tokio runtime is created lazily via `OnceLock` and reused
//! - Errors are stored in thread-local storage
//!
//! [Worker acquisition controls](../../../../../Tests/Portable/media-worker-lock/README.md)
//! exercise poisoned-lock refusal without a model or device.
//!
//! ## Usage from PowerShell
//!
//! ```powershell
//! # Load the DLL
//! Add-Type -Path "pcai_media.dll" -Namespace PCAI -Name Media
//!
//! # Initialize (defaults to cuda:0)
//! [PCAI.Media]::pcai_media_init("cuda:0")
//!
//! # Load model (HF repo ID or local path)
//! [PCAI.Media]::pcai_media_load_model("deepseek-ai/Janus-Pro-7B", -1)
//!
//! # Generate image
//! [PCAI.Media]::pcai_media_generate_image(
//!     "a glowing circuit board",
//!     5.0,   # cfg_scale
//!     1.0,   # temperature
//!     "C:\output\image.png"
//! )
//! ```

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use tokio::runtime::Runtime;

use crate::config::PipelineConfig;
use crate::generate::GenerationPipeline;
use crate::understand::UnderstandingPipeline;

// ============================================================================
// Error Codes
// ============================================================================

/// Error codes returned by pcai_media FFI functions.
///
/// All exported functions return 0 on success and a negative code on failure.
/// Use [`pcai_media_last_error`] to retrieve a human-readable message and
/// [`pcai_media_last_error_code`] to retrieve this numeric code.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcaiMediaErrorCode {
    /// Operation completed successfully.
    Success = 0,
    /// Library not initialised — call `pcai_media_init` first.
    NotInitialized = -1,
    /// Model not loaded — call `pcai_media_load_model` first.
    ModelNotLoaded = -2,
    /// Invalid input parameter (null pointer, invalid UTF-8, etc.).
    InvalidInput = -3,
    /// Generation or pipeline operation failed.
    GenerationError = -4,
    /// I/O error (file not found, permission denied, etc.).
    IoError = -5,
    /// Unknown or unclassified error.
    Unknown = -99,
}

impl PcaiMediaErrorCode {
    /// Convert an `i32` to the corresponding [`PcaiMediaErrorCode`].
    ///
    /// Any unrecognised value maps to [`PcaiMediaErrorCode::Unknown`].
    pub fn from_i32(code: i32) -> Self {
        match code {
            0 => Self::Success,
            -1 => Self::NotInitialized,
            -2 => Self::ModelNotLoaded,
            -3 => Self::InvalidInput,
            -4 => Self::GenerationError,
            -5 => Self::IoError,
            _ => Self::Unknown,
        }
    }
}

// ============================================================================
// Async Request Tracking
// ============================================================================

/// Monotonically increasing request ID counter.
static NEXT_REQUEST_ID: AtomicI64 = AtomicI64::new(1);

/// Status of an async image generation request.
enum MediaRequestStatus {
    /// Request accepted, not yet picked up by worker thread.
    Pending,
    /// Worker thread is actively running generation.
    Running,
    /// The worker won the Running -> Publishing transition and is moving the
    /// finished file into place.  Polls report it as running; a cancel now
    /// loses (returns -1) because the publish is already committed.
    Publishing,
    /// Generation finished successfully; path is the output file.
    Complete(String),
    /// Generation failed; text is the error message.
    Failed(String),
    /// Request was cancelled before or during generation.
    Cancelled,
}

/// Result of polling an async media request.
///
/// Returned by [`pcai_media_poll_result`].  When `status` is 2 (complete) or
/// 3 (failed) the `text` pointer is non-null and **must** be freed by the
/// caller with [`pcai_media_free_string`].
#[repr(C)]
pub struct PcaiMediaAsyncResult {
    /// 0 = pending, 1 = running, 2 = complete, 3 = failed, 4 = cancelled, -1 = unknown id
    pub status: i32,
    /// Result text (output path on success, error message on failure).
    /// Only valid when `status` is 2 or 3.  Caller must free with
    /// [`pcai_media_free_string`].
    pub text: *mut c_char,
}

// ============================================================================
// Thread-local Error Storage
// ============================================================================

thread_local! {
    /// Human-readable last error message for the current thread.
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
    /// Numeric last error code for the current thread.
    static LAST_ERROR_CODE: RefCell<PcaiMediaErrorCode> =
        const { RefCell::new(PcaiMediaErrorCode::Success) };
    /// CString buffer that keeps the pointer returned by `pcai_media_last_error`
    /// alive until the next call on the same thread.
    static LAST_ERROR_CSTRING: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Set the last error message and code for the current thread.
fn set_error(msg: impl Into<String>, code: PcaiMediaErrorCode) {
    let msg = msg.into();
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(msg));
    LAST_ERROR_CODE.with(|c| *c.borrow_mut() = code);
}

/// Render every operation and cause in a generation failure.
fn generation_error_message(error: &anyhow::Error) -> String {
    format!("{error:#}")
}

/// Store the complete generation diagnostic at synchronous FFI boundaries.
fn report_generation_error(error: &anyhow::Error) -> i32 {
    set_error(
        format!("Image generation failed: {}", generation_error_message(error)),
        PcaiMediaErrorCode::GenerationError,
    );
    PcaiMediaErrorCode::GenerationError as i32
}

/// Preserve the complete diagnostic in an asynchronous terminal status.
fn generation_failure_status(error: &anyhow::Error) -> MediaRequestStatus {
    MediaRequestStatus::Failed(generation_error_message(error))
}

/// Clear the last error for the current thread (called at the start of each
/// exported function).
fn clear_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
    LAST_ERROR_CODE.with(|c| *c.borrow_mut() = PcaiMediaErrorCode::Success);
}

// ============================================================================
// Pointer Helpers
// ============================================================================

/// Convert a raw C string pointer to a `&str`, validating null and UTF-8.
///
/// # Safety
///
/// `ptr` must be either null or a valid null-terminated C string for the
/// duration of this call.
unsafe fn c_str_to_str<'a>(ptr: *const c_char) -> Result<&'a str, PcaiMediaErrorCode> {
    if ptr.is_null() {
        set_error("Null pointer received", PcaiMediaErrorCode::InvalidInput);
        return Err(PcaiMediaErrorCode::InvalidInput);
    }
    // SAFETY: caller guarantees ptr is a valid null-terminated C string.
    CStr::from_ptr(ptr).to_str().map_err(|e| {
        set_error(
            format!("Invalid UTF-8 in C string: {e}"),
            PcaiMediaErrorCode::InvalidInput,
        );
        PcaiMediaErrorCode::InvalidInput
    })
}

// ============================================================================
// Global State
// ============================================================================

/// Global media pipeline state.
struct MediaState {
    /// Tokio multi-thread runtime shared across all calls.
    ///
    /// Retained for future async hub operations (e.g., background downloads).
    /// Not currently used by the synchronous load/generate paths.
    #[expect(
        dead_code,
        reason = "Tokio runtime retained for future async hub download operations; not yet exercised by synchronous FFI paths"
    )]
    runtime: Runtime,
    /// Device string set by `pcai_media_init`, propagated to `load_model`.
    device: String,
    /// Async request status map (request ID → status).
    requests: HashMap<i64, MediaRequestStatus>,
}

/// The global singleton protected by a `Mutex`.
static MEDIA_STATE: OnceLock<Mutex<MediaState>> = OnceLock::new();

/// Lazily initialise and return the global [`MediaState`] mutex.
///
/// The runtime is created with [`tokio::runtime::Builder::new_multi_thread`]
/// so that async operations like HuggingFace Hub downloads work correctly.
fn get_state() -> &'static Mutex<MediaState> {
    MEDIA_STATE.get_or_init(|| {
        let runtime = Runtime::new().expect("failed to create Tokio runtime for pcai_media");
        Mutex::new(MediaState {
            runtime,
            device: "cuda:0".to_string(),
            requests: HashMap::new(),
        })
    })
}

/// Serializes model load/replace, generation and understanding.
///
/// One generation at a time bounds model/KV-cache memory.  Lock order is
/// always this gate first, then the state mutex, and the state mutex is only
/// held for short bookkeeping, never across inference or model loading.  The
/// loaded model lives in [`MODEL_SLOT`] under its own short-lived mutex.
/// `pcai_media_poll_result`, `pcai_media_cancel`, `pcai_media_init` and
/// `pcai_media_shutdown` take only the state mutex, so they stay responsive
/// while a generation is running.
static INFERENCE_GATE: Mutex<()> = Mutex::new(());

/// A loaded-model handle guarded by a lifecycle epoch.
///
/// `epoch` increments on every shutdown.  A load captures the epoch when it
/// is admitted and only installs its model if the epoch is unchanged, so a
/// shutdown that runs while a (long, lock-free) load is in flight cannot be
/// undone by that load finishing afterwards.
struct ModelSlot<T> {
    epoch: u64,
    model: Option<Arc<T>>,
}

impl<T> ModelSlot<T> {
    const fn new() -> Self {
        Self { epoch: 0, model: None }
    }
}

/// The loaded generation pipeline.  Its mutex is only ever held for a field
/// read or write, and models are always dropped after it is released.
static MODEL_SLOT: Mutex<ModelSlot<GenerationPipeline>> = Mutex::new(ModelSlot::new());

fn lock_slot<T>(slot: &Mutex<ModelSlot<T>>) -> MutexGuard<'_, ModelSlot<T>> {
    // The slot is two plain fields, so it stays consistent after a panic.
    slot.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Clone the current model handle, if any.
fn slot_current<T>(slot: &Mutex<ModelSlot<T>>) -> Option<Arc<T>> {
    lock_slot(slot).model.clone()
}

/// Capture the lifecycle epoch at load admission.
fn slot_begin_load<T>(slot: &Mutex<ModelSlot<T>>) -> u64 {
    lock_slot(slot).epoch
}

/// Install `model` if no shutdown happened since `epoch` was captured.
///
/// Returns whether it was installed.  A stale model and any model it
/// replaces are dropped here, after the slot lock is released, because
/// freeing a large model can be slow.
fn slot_install<T>(slot: &Mutex<ModelSlot<T>>, epoch: u64, model: T) -> bool {
    let (installed, displaced) = {
        let mut guard = lock_slot(slot);
        if guard.epoch == epoch {
            (true, guard.model.replace(Arc::new(model)))
        } else {
            (false, None)
        }
    };
    drop(displaced);
    installed
}

/// Release the model and invalidate every in-flight load.
fn slot_shutdown<T>(slot: &Mutex<ModelSlot<T>>) {
    let released = {
        let mut guard = lock_slot(slot);
        guard.epoch = guard.epoch.wrapping_add(1);
        guard.model.take()
    };
    drop(released);
}

/// Run `load` outside every lock and install its result if the slot was not
/// shut down meanwhile.  `Ok(false)` means the load completed but was
/// discarded because of a shutdown.
fn load_into_slot<T>(slot: &Mutex<ModelSlot<T>>, load: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<bool> {
    let epoch = slot_begin_load(slot);
    let model = load()?;
    Ok(slot_install(slot, epoch, model))
}

/// Take the inference gate, recovering from poison.
///
/// The gate guards no data, so a panic in an earlier holder must not wedge
/// every later inference call.
fn acquire_inference_gate() -> MutexGuard<'static, ()> {
    INFERENCE_GATE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Lock the state mutex briefly for request-map bookkeeping, recovering from
/// poison (the request map stays structurally valid after a panic).
fn lock_state_recovering() -> MutexGuard<'static, MediaState> {
    get_state().lock().unwrap_or_else(PoisonError::into_inner)
}

/// Clone the loaded pipeline handle without holding any lock afterwards.
///
/// Returns the status code and message the FFI boundary should report.
fn loaded_pipeline() -> Result<Arc<GenerationPipeline>, (PcaiMediaErrorCode, String)> {
    slot_current(&MODEL_SLOT).ok_or_else(|| {
        (
            PcaiMediaErrorCode::ModelNotLoaded,
            "No model loaded — call pcai_media_load_model first".to_string(),
        )
    })
}

// ============================================================================
// Exported Functions
// ============================================================================

/// Initialise the pcai_media library.
///
/// This call is idempotent — calling it a second time is harmless and returns
/// `Success` immediately.  The global Tokio runtime is created lazily the
/// first time this (or any other) exported function is called.
///
/// # Arguments
///
/// * `device` — Target compute device string, e.g. `"cpu"`, `"cuda"`, or
///   `"cuda:0"`.  Pass `NULL` to default to `"cuda:0"`.
///
/// # Returns
///
/// `0` on success, negative error code on failure.
///
/// # Safety
///
/// `device` must be a valid null-terminated C string or `NULL`.
#[no_mangle]
pub extern "C" fn pcai_media_init(device: *const c_char) -> i32 {
    clear_error();

    let device_str: String = if device.is_null() {
        "cuda:0".to_string()
    } else {
        match unsafe { c_str_to_str(device) } {
            Ok(s) => s.to_string(),
            Err(code) => return code as i32,
        }
    };

    // Trigger lazy initialisation of the global state.  The device string is
    // stored nowhere here — it is applied when load_model creates the config.
    // We just validate it looks sane.
    let valid_device = device_str == "cpu" || device_str == "cuda" || device_str.starts_with("cuda:");

    if !valid_device {
        set_error(
            format!("Unrecognised device '{device_str}'; expected 'cpu' or 'cuda[:N]'"),
            PcaiMediaErrorCode::InvalidInput,
        );
        return PcaiMediaErrorCode::InvalidInput as i32;
    }

    // Ensure the global state (and its runtime) exists, then store the device.
    let state = get_state();
    if let Ok(mut guard) = state.lock() {
        guard.device = device_str.clone();
    }

    tracing::info!(device = %device_str, "pcai_media_init");
    PcaiMediaErrorCode::Success as i32
}

/// Load the Janus-Pro model into the pipeline.
///
/// The `model_path` argument accepts either a HuggingFace Hub model ID
/// (e.g. `"deepseek-ai/Janus-Pro-7B"`) or an absolute path to a local model
/// directory containing `config.json`, `*.safetensors`, and `tokenizer.json`.
///
/// # Arguments
///
/// * `model_path` — HF repo ID or local path (null-terminated C string).
/// * `gpu_layers` — Number of layers to offload to GPU.  Pass `0` for
///   CPU-only inference.  This value is stored in [`PipelineConfig::gpu_layers`].
///
/// # Returns
///
/// `0` on success, negative error code on failure.
///
/// # Safety
///
/// `model_path` must be a valid null-terminated C string.
#[no_mangle]
pub extern "C" fn pcai_media_load_model(model_path: *const c_char, gpu_layers: i32) -> i32 {
    clear_error();

    let path_str = match unsafe { c_str_to_str(model_path) } {
        Ok(s) => s.to_string(),
        Err(code) => return code as i32,
    };

    // No generation may run while the model is loaded or replaced, and the
    // state mutex must not be held for the (long) load.
    let _gate = acquire_inference_gate();
    let device = match get_state().lock() {
        Ok(guard) => guard.device.clone(),
        Err(e) => {
            set_error(
                format!("Failed to acquire state lock: {e}"),
                PcaiMediaErrorCode::GenerationError,
            );
            return PcaiMediaErrorCode::GenerationError as i32;
        }
    };

    let config = PipelineConfig {
        model: path_str.clone(),
        gpu_layers,
        device,
        ..PipelineConfig::default()
    };

    // `GenerationPipeline::load` is synchronous (uses blocking I/O internally
    // through hf-hub and safetensors).  Call it directly -- no async executor
    // is required for the load path.  Neither the state mutex nor the model
    // slot is held while it runs; a shutdown that arrives meanwhile bumps the
    // slot epoch and the finished model is discarded instead of installed.
    match load_into_slot(&MODEL_SLOT, || GenerationPipeline::load(config)) {
        Ok(true) => {
            tracing::info!(model = %path_str, gpu_layers, "model loaded");
            PcaiMediaErrorCode::Success as i32
        }
        Ok(false) => {
            set_error(
                format!("Model '{path_str}' finished loading after pcai_media_shutdown and was discarded"),
                PcaiMediaErrorCode::ModelNotLoaded,
            );
            PcaiMediaErrorCode::ModelNotLoaded as i32
        }
        Err(e) => {
            let msg = format!("Failed to load model '{path_str}': {e:#}");
            let code = if e.to_string().contains("not found")
                || e.to_string().contains("No such file")
                || e.to_string().contains("cannot open")
            {
                PcaiMediaErrorCode::IoError
            } else {
                PcaiMediaErrorCode::GenerationError
            };
            set_error(msg, code);
            code as i32
        }
    }
}

/// Generate an image from a text prompt and save it as a PNG file.
///
/// # Arguments
///
/// * `prompt` — Text description of the image to generate.
/// * `cfg_scale` — Classifier-Free Guidance scale (higher = more
///   prompt-faithful; typical range 1.0 – 10.0).  Pass `0.0` to use the
///   default of `5.0`.
/// * `temperature` — Sampling temperature (1.0 = neutral; lower = sharper).
///   Pass `0.0` to use the default of `1.0`.
/// * `output_path` — Absolute file path where the PNG will be written.
///
/// # Returns
///
/// `0` on success, negative error code on failure.
///
/// # Safety
///
/// All pointer arguments must be valid null-terminated C strings.
#[no_mangle]
pub extern "C" fn pcai_media_generate_image(
    prompt: *const c_char,
    cfg_scale: f32,
    temperature: f32,
    output_path: *const c_char,
) -> i32 {
    clear_error();

    let prompt_str = match unsafe { c_str_to_str(prompt) } {
        Ok(s) => s.to_string(),
        Err(code) => return code as i32,
    };
    let out_str = match unsafe { c_str_to_str(output_path) } {
        Ok(s) => s.to_string(),
        Err(code) => return code as i32,
    };

    // Serialize inference; the state mutex is only held to clone the handle.
    let _gate = acquire_inference_gate();
    let pipeline = match loaded_pipeline() {
        Ok(p) => p,
        Err((code, message)) => {
            set_error(message, code);
            return code as i32;
        }
    };

    // Override config fields when non-zero caller values are supplied.
    // `generate_with_overrides` applies the per-call overrides without
    // mutating the pipeline's stored config.
    let override_cfg = if cfg_scale > 0.0 { Some(cfg_scale as f64) } else { None };
    let override_temp = if temperature > 0.0 {
        Some(temperature as f64)
    } else {
        None
    };

    let generate_result = pipeline.generate_with_overrides(&prompt_str, override_cfg, override_temp);

    let image = match generate_result {
        Ok(img) => img,
        Err(e) => {
            return report_generation_error(&e);
        }
    };

    // Save to PNG.
    if let Err(e) = write_image_atomically(&image, Path::new(&out_str)) {
        set_error(
            format!("Failed to save image to '{out_str}': {e}"),
            PcaiMediaErrorCode::IoError,
        );
        return PcaiMediaErrorCode::IoError as i32;
    }

    tracing::info!(
        prompt = %prompt_str,
        output = %out_str,
        "image saved"
    );
    PcaiMediaErrorCode::Success as i32
}

/// Run image-to-text understanding on an existing image file.
///
/// Returns a heap-allocated UTF-8 string containing the model's text
/// response.  The caller **must** free the returned pointer with
/// [`pcai_media_free_string`].
///
/// # Arguments
///
/// * `image_path`  — Absolute file path to the input image (PNG, JPEG, BMP).
/// * `prompt`      — Question or instruction for the model.
/// * `max_tokens`  — Maximum number of text tokens to generate.
/// * `temperature` — Sampling temperature (`0.0` = greedy, `1.0` = creative).
///
/// # Returns
///
/// A non-null heap-allocated C string on success, or `NULL` on failure.
/// Inspect [`pcai_media_last_error`] for failure details.
///
/// # Safety
///
/// `image_path` and `prompt` must be valid null-terminated C strings.
#[no_mangle]
pub extern "C" fn pcai_media_understand_image(
    image_path: *const c_char,
    prompt: *const c_char,
    max_tokens: u32,
    temperature: f32,
) -> *mut c_char {
    clear_error();

    let path_str = match unsafe { c_str_to_str(image_path) } {
        Ok(s) => s.to_string(),
        Err(_) => return std::ptr::null_mut(),
    };
    let prompt_str = match unsafe { c_str_to_str(prompt) } {
        Ok(s) => s.to_string(),
        Err(_) => return std::ptr::null_mut(),
    };

    let _gate = acquire_inference_gate();
    let pipeline = match loaded_pipeline() {
        Ok(p) => p,
        Err((code, message)) => {
            set_error(message, code);
            return std::ptr::null_mut();
        }
    };

    // Load image from file.
    let dyn_image = match image::open(&path_str) {
        Ok(img) => img,
        Err(e) => {
            set_error(
                format!("Failed to open image '{path_str}': {e}"),
                PcaiMediaErrorCode::IoError,
            );
            return std::ptr::null_mut();
        }
    };

    // Run understanding pipeline (borrows from GenerationPipeline).
    let result = UnderstandingPipeline::understand_with_fallback(
        &pipeline,
        &dyn_image,
        Some(std::path::Path::new(&path_str)),
        &prompt_str,
        max_tokens,
        temperature,
    );

    match result {
        Ok(text) => match CString::new(text) {
            Ok(cstr) => cstr.into_raw(),
            Err(e) => {
                set_error(
                    format!("Response text contains null byte: {e}"),
                    PcaiMediaErrorCode::GenerationError,
                );
                std::ptr::null_mut()
            }
        },
        Err(e) => {
            set_error(
                format!("Understanding failed: {e}"),
                PcaiMediaErrorCode::GenerationError,
            );
            std::ptr::null_mut()
        }
    }
}

/// Generate an image and return raw PNG bytes instead of writing to a file.
///
/// On success, `*out_data` is set to a heap-allocated byte buffer and
/// `*out_len` is set to the buffer length.  The caller **must** free the
/// buffer with [`pcai_media_free_bytes`].
///
/// # Arguments
///
/// * `prompt`      — Text prompt.
/// * `cfg_scale`   — CFG scale (pass `0.0` for default).
/// * `temperature` — Sampling temperature (pass `0.0` for default).
/// * `out_data`    — Output pointer to the PNG byte buffer.
/// * `out_len`     — Output pointer to the byte count.
///
/// # Returns
///
/// `0` on success, negative error code on failure.
///
/// # Safety
///
/// `prompt` must be a valid null-terminated C string.  `out_data` and
/// `out_len` must be valid, non-null, writable pointers.
#[no_mangle]
pub extern "C" fn pcai_media_generate_image_bytes(
    prompt: *const c_char,
    cfg_scale: f32,
    temperature: f32,
    out_data: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    clear_error();

    if out_data.is_null() || out_len.is_null() {
        set_error(
            "out_data and out_len must be non-null",
            PcaiMediaErrorCode::InvalidInput,
        );
        return PcaiMediaErrorCode::InvalidInput as i32;
    }

    let prompt_str = match unsafe { c_str_to_str(prompt) } {
        Ok(s) => s.to_string(),
        Err(code) => return code as i32,
    };

    // Serialize inference; the state mutex is only held to clone the handle.
    let _gate = acquire_inference_gate();
    let pipeline = match loaded_pipeline() {
        Ok(p) => p,
        Err((code, message)) => {
            set_error(message, code);
            return code as i32;
        }
    };

    let override_cfg = if cfg_scale > 0.0 { Some(cfg_scale as f64) } else { None };
    let override_temp = if temperature > 0.0 {
        Some(temperature as f64)
    } else {
        None
    };

    let image = match pipeline.generate_with_overrides(&prompt_str, override_cfg, override_temp) {
        Ok(img) => img,
        Err(e) => {
            return report_generation_error(&e);
        }
    };

    // Encode to PNG in memory.
    let mut png_buf: Vec<u8> = Vec::new();
    {
        use image::codecs::png::PngEncoder;
        use image::ImageEncoder;
        let encoder = PngEncoder::new(&mut png_buf);
        if let Err(e) = encoder.write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            image::ExtendedColorType::Rgb8,
        ) {
            set_error(format!("PNG encode failed: {e}"), PcaiMediaErrorCode::GenerationError);
            return PcaiMediaErrorCode::GenerationError as i32;
        }
    }

    // Convert to boxed slice to guarantee len == capacity for safe deallocation.
    let boxed = png_buf.into_boxed_slice();
    let len = boxed.len();
    let ptr = Box::into_raw(boxed) as *mut u8;

    // SAFETY: out_data and out_len were checked for null above.
    unsafe {
        *out_data = ptr;
        *out_len = len;
    }

    tracing::info!(
        prompt = %prompt_str,
        png_bytes = len,
        "image generated (bytes)"
    );
    PcaiMediaErrorCode::Success as i32
}

// ----------------------------------------------------------------------------
// Staged output files
// ----------------------------------------------------------------------------
//
// A generated image is never written straight to the caller's destination.
// It is encoded into a staging file next to the destination, and only
// published (renamed over the destination) if the request is still live.
// That way a cancelled request can never create or overwrite the caller's
// file, and a failed write never leaves a partial file behind.

/// Short staging path in `dest`'s parent directory: `.pcai-<16 hex>.partial`.
///
/// The name is independent of the destination's, so a valid destination near
/// the 255-character component limit still gets a legal staging name.
fn staging_path(dest: &Path) -> PathBuf {
    let name = format!(".pcai-{:016x}.partial", rand::random::<u64>());
    match dest.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.join(name),
        _ => PathBuf::from(name),
    }
}

/// Encoder for `dest`: the image format follows the destination extension,
/// as `RgbImage::save` did.
fn encode_for(dest: &Path) -> impl FnOnce(&image::RgbImage, &mut File) -> io::Result<()> {
    let format = image::ImageFormat::from_path(dest);
    move |image, file| {
        let format = format.map_err(io::Error::other)?;
        let mut writer = BufWriter::new(file);
        image.write_to(&mut writer, format).map_err(io::Error::other)?;
        writer.flush()
    }
}

/// Exclusively create a staging file, retrying with a new name on collision.
fn create_staging_file(dest: &Path) -> io::Result<(PathBuf, File)> {
    for _ in 0..STAGING_ATTEMPTS {
        let staged = staging_path(dest);
        match OpenOptions::new().write(true).create_new(true).open(&staged) {
            Ok(file) => return Ok((staged, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "no unused staging file name after several attempts",
    ))
}

/// Attempts to find an unused staging name before giving up.
const STAGING_ATTEMPTS: u32 = 8;

/// Encode `image` into a fresh, exclusively created staging file next to
/// `dest` and flush it to disk.  Any failure removes the staging file.
fn stage_image(
    image: &image::RgbImage,
    dest: &Path,
    encode: impl FnOnce(&image::RgbImage, &mut File) -> io::Result<()>,
) -> io::Result<PathBuf> {
    let (staged, mut file) = create_staging_file(dest)?;
    let written = encode(image, &mut file).and_then(|()| file.sync_all());
    drop(file);
    match written {
        Ok(()) => Ok(staged),
        Err(error) => {
            let _ = fs::remove_file(&staged);
            Err(error)
        }
    }
}

/// Replace `dest` with the staged file.
///
/// `std::fs::rename` replaces an existing file on Windows
/// (`MOVEFILE_REPLACE_EXISTING`) and on Unix; it fails if `dest` is a
/// directory or is held open without delete sharing.  Callers remove the
/// staging file on error.
fn publish_staged(staged: &Path, dest: &Path) -> io::Result<()> {
    fs::rename(staged, dest)
}

/// Stage and publish in one step, for the synchronous paths that cannot be
/// cancelled but must still never leave a partial destination.
fn write_image_atomically(image: &image::RgbImage, dest: &Path) -> io::Result<()> {
    let staged = stage_image(image, dest, encode_for(dest))?;
    publish_staged(&staged, dest).inspect_err(|_| {
        let _ = fs::remove_file(&staged);
    })
}

/// Run one queued async generation request to a terminal status.
///
/// The request stays `Pending` while it waits for the inference gate, so
/// `pcai_media_cancel` can drop it before any work starts and
/// `pcai_media_poll_result` reports it truthfully.  The global state mutex is
/// taken only for short status flips: never while waiting for the gate, never
/// across `generate`, encoding, the rename or file cleanup, so a slow or
/// stalled destination cannot block poll/cancel of any request.
///
/// Per-request exclusion between cancel and publish is a small state machine
/// (`Pending -> Running -> Publishing -> Complete | Failed`, or `Cancelled`
/// from `Pending`/`Running`): the worker flips `Running -> Publishing` under
/// the lock, then renames with no lock held.  A cancel that lands first wins
/// (the staging file is removed, the destination untouched); a cancel that
/// finds `Publishing` loses and returns -1, and the request completes.
fn execute_async_request(id: i64, output_path: String, generate: impl FnOnce() -> anyhow::Result<image::RgbImage>) {
    let encode = encode_for(Path::new(&output_path));
    execute_async_request_with(id, output_path, generate, encode);
}

/// [`execute_async_request`] with an injectable encoder, so tests can use a
/// slow or failing one.
fn execute_async_request_with(
    id: i64,
    output_path: String,
    generate: impl FnOnce() -> anyhow::Result<image::RgbImage>,
    encode: impl FnOnce(&image::RgbImage, &mut File) -> io::Result<()>,
) {
    execute_async_request_staged(id, output_path, generate, encode, publish_staged);
}

/// [`execute_async_request_with`] with an injectable publish step, so tests
/// can stall the rename.
fn execute_async_request_staged(
    id: i64,
    output_path: String,
    generate: impl FnOnce() -> anyhow::Result<image::RgbImage>,
    encode: impl FnOnce(&image::RgbImage, &mut File) -> io::Result<()>,
    publish: impl FnOnce(&Path, &Path) -> io::Result<()>,
) {
    let gate = acquire_inference_gate();

    if !begin_async_request(id) {
        return;
    }

    let result = generate();
    drop(gate);

    let image = match result {
        Ok(image) => image,
        Err(e) => {
            record_terminal_status(id, generation_failure_status(&e));
            return;
        }
    };

    // A cancel during generation: do not even create the staging file.
    if !request_is_running(id) {
        return;
    }

    let dest = Path::new(&output_path);
    let staged = match stage_image(&image, dest, encode) {
        Ok(staged) => staged,
        Err(e) => {
            record_terminal_status(id, MediaRequestStatus::Failed(format!("Save failed: {e}")));
            return;
        }
    };

    // Commit to publishing under the lock, then rename without it.
    if !begin_publish(id) {
        let _ = fs::remove_file(&staged);
        return;
    }
    let terminal = match publish(&staged, dest) {
        Ok(()) => MediaRequestStatus::Complete(output_path),
        Err(e) => {
            let _ = fs::remove_file(&staged);
            MediaRequestStatus::Failed(format!("Save failed: {e}"))
        }
    };
    finish_publish(id, terminal);
}

/// Pending -> Running, unless the request was cancelled while queued.
///
/// Returns whether the worker may run.  The state lock is released before
/// this returns, so generation never runs under it.
fn begin_async_request(id: i64) -> bool {
    let mut guard = lock_state_recovering();
    if !matches!(guard.requests.get(&id), Some(MediaRequestStatus::Pending)) {
        return false;
    }
    guard.requests.insert(id, MediaRequestStatus::Running);
    true
}

/// Running -> Publishing.  `false` means a cancel got there first.
fn begin_publish(id: i64) -> bool {
    let mut guard = lock_state_recovering();
    if !matches!(guard.requests.get(&id), Some(MediaRequestStatus::Running)) {
        return false;
    }
    guard.requests.insert(id, MediaRequestStatus::Publishing);
    true
}

/// Publishing -> Complete | Failed.  Nothing can cancel a publishing request,
/// so the transition is unconditional for that state.
fn finish_publish(id: i64, terminal: MediaRequestStatus) {
    let mut guard = lock_state_recovering();
    if matches!(guard.requests.get(&id), Some(MediaRequestStatus::Publishing)) {
        guard.requests.insert(id, terminal);
    }
}

/// Whether the request is still `Running` (not cancelled or already polled away).
fn request_is_running(id: i64) -> bool {
    matches!(
        lock_state_recovering().requests.get(&id),
        Some(MediaRequestStatus::Running)
    )
}

/// Store a terminal status unless the request was cancelled mid-flight.
fn record_terminal_status(id: i64, status: MediaRequestStatus) {
    let mut guard = lock_state_recovering();
    if matches!(guard.requests.get(&id), Some(MediaRequestStatus::Running)) {
        guard.requests.insert(id, status);
    }
}

/// Submit an asynchronous image generation request.
///
/// Returns a request ID (`> 0`) immediately.  The caller should poll for
/// results with [`pcai_media_poll_result`] and can cancel with
/// [`pcai_media_cancel`].
///
/// # Arguments
///
/// * `prompt`      — Text prompt.
/// * `cfg_scale`   — CFG scale (pass `0.0` for default).
/// * `temperature` — Sampling temperature (pass `0.0` for default).
/// * `output_path` — File path where the PNG will be written on success.
///
/// # Returns
///
/// Request ID `> 0` on success, `-1` on error.
///
/// # Safety
///
/// All pointer arguments must be valid null-terminated C strings.
#[no_mangle]
pub extern "C" fn pcai_media_generate_image_async(
    prompt: *const c_char,
    cfg_scale: f32,
    temperature: f32,
    output_path: *const c_char,
) -> i64 {
    clear_error();

    let prompt_str = match unsafe { c_str_to_str(prompt) } {
        Ok(s) => s.to_string(),
        Err(_) => return -1,
    };
    let out_str = match unsafe { c_str_to_str(output_path) } {
        Ok(s) => s.to_string(),
        Err(_) => return -1,
    };

    // Validate that a model is loaded before allocating an ID.
    if slot_current(&MODEL_SLOT).is_none() {
        set_error(
            "No model loaded — call pcai_media_load_model first",
            PcaiMediaErrorCode::ModelNotLoaded,
        );
        return -1;
    }

    // Allocate a unique request ID and register it as Pending.
    let id = NEXT_REQUEST_ID.fetch_add(1, AtomicOrdering::SeqCst);
    {
        let state = get_state();
        if let Ok(mut g) = state.lock() {
            g.requests.insert(id, MediaRequestStatus::Pending);
        }
    }

    // Spawn a plain OS thread (not Tokio) to avoid nesting block_on.
    let override_cfg = if cfg_scale > 0.0 { Some(cfg_scale as f64) } else { None };
    let override_temp = if temperature > 0.0 {
        Some(temperature as f64)
    } else {
        None
    };
    std::thread::spawn(move || {
        execute_async_request(id, out_str, move || {
            let pipeline = loaded_pipeline().map_err(|(_, message)| anyhow::anyhow!(message))?;
            pipeline.generate_with_overrides(&prompt_str, override_cfg, override_temp)
        });
    });

    id
}

/// Poll the status of an async media request.
///
/// When `status` is 2 (complete) or 3 (failed), the request is removed from
/// the internal map and the `text` pointer is non-null.  The caller **must**
/// free it with [`pcai_media_free_string`].
///
/// Status codes:
/// * 0 = pending
/// * 1 = running
/// * 2 = complete (`text` = output file path)
/// * 3 = failed (`text` = error message)
/// * 4 = cancelled
/// * -1 = unknown request ID
///
/// # Arguments
///
/// * `request_id` — ID returned by [`pcai_media_generate_image_async`].
#[no_mangle]
pub extern "C" fn pcai_media_poll_result(request_id: i64) -> PcaiMediaAsyncResult {
    let state = get_state();
    let mut guard = match state.lock() {
        Ok(g) => g,
        Err(_) => {
            return PcaiMediaAsyncResult {
                status: -1,
                text: std::ptr::null_mut(),
            };
        }
    };

    match guard.requests.get(&request_id) {
        Some(MediaRequestStatus::Pending) => PcaiMediaAsyncResult {
            status: 0,
            text: std::ptr::null_mut(),
        },
        Some(MediaRequestStatus::Running | MediaRequestStatus::Publishing) => PcaiMediaAsyncResult {
            status: 1,
            text: std::ptr::null_mut(),
        },
        Some(MediaRequestStatus::Complete(_)) => {
            if let Some(MediaRequestStatus::Complete(text)) = guard.requests.remove(&request_id) {
                let c_str = CString::new(text.replace('\0', "\u{FFFD}")).unwrap_or_default();
                PcaiMediaAsyncResult {
                    status: 2,
                    text: c_str.into_raw(),
                }
            } else {
                PcaiMediaAsyncResult {
                    status: -1,
                    text: std::ptr::null_mut(),
                }
            }
        }
        Some(MediaRequestStatus::Failed(_)) => {
            if let Some(MediaRequestStatus::Failed(text)) = guard.requests.remove(&request_id) {
                let c_str = CString::new(text.replace('\0', "\u{FFFD}")).unwrap_or_default();
                PcaiMediaAsyncResult {
                    status: 3,
                    text: c_str.into_raw(),
                }
            } else {
                PcaiMediaAsyncResult {
                    status: -1,
                    text: std::ptr::null_mut(),
                }
            }
        }
        Some(MediaRequestStatus::Cancelled) => {
            guard.requests.remove(&request_id);
            PcaiMediaAsyncResult {
                status: 4,
                text: std::ptr::null_mut(),
            }
        }
        None => PcaiMediaAsyncResult {
            status: -1,
            text: std::ptr::null_mut(),
        },
    }
}

/// Cancel an async media request.
///
/// Marks a pending or running request as cancelled.  The cancellation is
/// cooperative — the worker thread checks for it at defined points.
///
/// # Arguments
///
/// * `request_id` — ID returned by [`pcai_media_generate_image_async`].
///
/// # Returns
///
/// `0` if the request was successfully cancelled, `-1` if the ID was not
/// found or the request had already finished.  A request whose output is
/// already being moved into place (it still polls as running) can no longer be
/// cancelled: this returns `-1` and the request completes.
#[no_mangle]
pub extern "C" fn pcai_media_cancel(request_id: i64) -> i32 {
    let state = get_state();
    let mut guard = match state.lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };

    match guard.requests.get_mut(&request_id) {
        Some(status @ MediaRequestStatus::Pending | status @ MediaRequestStatus::Running) => {
            *status = MediaRequestStatus::Cancelled;
            0
        }
        _ => -1,
    }
}

// ============================================================================
// Upscale (optional — requires `upscale` feature)
// ============================================================================

/// Upscale an image 4x using RealESRGAN.
///
/// Loads the ONNX model from `model_path`, reads the image from `input_path`,
/// performs 4x super-resolution, and saves the result to `output_path`.
///
/// # Returns
///
/// `0` on success, negative error code on failure.
#[cfg(feature = "upscale")]
#[no_mangle]
pub extern "C" fn pcai_media_upscale_image(
    model_path: *const c_char,
    input_path: *const c_char,
    output_path: *const c_char,
) -> i32 {
    clear_error();

    let model = match unsafe { c_str_to_str(model_path) } {
        Ok(s) => s,
        Err(code) => return code as i32,
    };
    let input = match unsafe { c_str_to_str(input_path) } {
        Ok(s) => s,
        Err(code) => return code as i32,
    };
    let output = match unsafe { c_str_to_str(output_path) } {
        Ok(s) => s,
        Err(code) => return code as i32,
    };

    let mut pipeline = match crate::upscale::UpscalePipeline::load(model) {
        Ok(p) => p,
        Err(e) => {
            set_error(
                format!("failed to load upscale model: {e:#}"),
                PcaiMediaErrorCode::IoError,
            );
            return PcaiMediaErrorCode::IoError as i32;
        }
    };

    let img = match image::open(input) {
        Ok(i) => i,
        Err(e) => {
            set_error(format!("failed to open input image: {e}"), PcaiMediaErrorCode::IoError);
            return PcaiMediaErrorCode::IoError as i32;
        }
    };

    let upscaled = match pipeline.upscale(&img) {
        Ok(u) => u,
        Err(e) => {
            set_error(format!("upscale failed: {e:#}"), PcaiMediaErrorCode::GenerationError);
            return PcaiMediaErrorCode::GenerationError as i32;
        }
    };

    if let Err(e) = upscaled.save(output) {
        set_error(
            format!("failed to save upscaled image: {e}"),
            PcaiMediaErrorCode::IoError,
        );
        return PcaiMediaErrorCode::IoError as i32;
    }

    tracing::info!(input = input, output = output, "image upscaled 4x");
    PcaiMediaErrorCode::Success as i32
}

/// Release the loaded pipeline and free associated memory.
///
/// After this call [`pcai_media_generate_image`] will fail until
/// [`pcai_media_load_model`] is called again, including when a load was in
/// flight: that load finishes, is discarded, and reports `ModelNotLoaded`.
/// A generation already running finishes on its own handle.  Calling this
/// function before [`pcai_media_init`] is safe and is a no-op.
#[no_mangle]
pub extern "C" fn pcai_media_shutdown() {
    clear_error();

    slot_shutdown(&MODEL_SLOT);
    tracing::info!("pcai_media_shutdown: pipeline released");
}

/// Return a pointer to the last error message for the current thread.
///
/// The returned pointer is valid until the next call to **any** pcai_media
/// function on the **same thread**.  Do **not** call
/// [`pcai_media_free_string`] on this pointer.
///
/// # Returns
///
/// A null-terminated C string pointer, or `NULL` if there is no pending error.
#[no_mangle]
pub extern "C" fn pcai_media_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        let borrow = e.borrow();
        match borrow.as_ref() {
            Some(msg) => match CString::new(msg.as_str()) {
                Ok(cstr) => {
                    let ptr = cstr.as_ptr();
                    // Store the CString in the thread-local so the pointer
                    // remains valid until the next call on this thread.
                    LAST_ERROR_CSTRING.with(|buf| {
                        *buf.borrow_mut() = Some(cstr);
                    });
                    ptr
                }
                Err(_) => std::ptr::null(),
            },
            None => std::ptr::null(),
        }
    })
}

/// Return the numeric error code for the last error on the current thread.
///
/// Returns `0` ([`PcaiMediaErrorCode::Success`]) if there is no pending error.
#[no_mangle]
pub extern "C" fn pcai_media_last_error_code() -> i32 {
    LAST_ERROR_CODE.with(|c| *c.borrow() as i32)
}

/// Free a string that was allocated by this library and returned to the caller.
///
/// Only call this for strings explicitly documented as requiring a free call.
/// Do **not** call this on the pointer returned by [`pcai_media_last_error`].
///
/// # Safety
///
/// `s` must be a pointer previously returned by a pcai_media function that
/// transfers ownership, or `NULL`.  Must not be called twice on the same
/// pointer.
#[no_mangle]
pub extern "C" fn pcai_media_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    // SAFETY: `s` is non-null and was produced by `CString::into_raw` inside
    // this library. Ownership is transferred back here so the allocation is
    // freed exactly once.
    unsafe {
        let _ = CString::from_raw(s);
    }
}

/// Free a byte buffer that was allocated by this library.
///
/// # Arguments
///
/// * `data` — Pointer to the start of the buffer.
/// * `len` — Number of bytes in the buffer.
///
/// # Safety
///
/// `data` must be a pointer previously returned by a pcai_media function that
/// transfers ownership, or `NULL`.  Must not be called twice on the same
/// pointer.
#[no_mangle]
pub extern "C" fn pcai_media_free_bytes(data: *mut u8, len: usize) {
    if data.is_null() {
        return;
    }
    // SAFETY: `data` was allocated via `Box::into_raw` on a `Box<[u8]>` with
    // exactly `len` bytes.  Reconstructing the `Box<[u8]>` and dropping it
    // returns the memory to the allocator correctly.
    unsafe {
        let slice = std::slice::from_raw_parts_mut(data, len);
        let _ = Box::from_raw(slice);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn chained_generation_error() -> anyhow::Error {
        anyhow::Error::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid image token",
        ))
        .context("step 7: projection failed")
        .context("generate_inner failed")
    }

    #[test]
    fn test_sync_generation_error_preserves_complete_chain() {
        let error = chained_generation_error();
        assert_eq!(
            report_generation_error(&error),
            PcaiMediaErrorCode::GenerationError as i32
        );
        assert_eq!(pcai_media_last_error_code(), PcaiMediaErrorCode::GenerationError as i32);
        let pointer = pcai_media_last_error();
        assert!(!pointer.is_null(), "generation failure must provide a diagnostic");
        // SAFETY: the non-null thread-local CString stays live until the next FFI call.
        let message = unsafe { CStr::from_ptr(pointer) }
            .to_str()
            .expect("diagnostic should be UTF-8");
        assert_eq!(
            message,
            "Image generation failed: generate_inner failed: step 7: projection failed: invalid image token"
        );
        assert_eq!(
            error
                .downcast_ref::<std::io::Error>()
                .expect("typed cause must be retained")
                .kind(),
            std::io::ErrorKind::InvalidData
        );
        clear_error();
    }

    #[test]
    fn test_async_generation_failure_preserves_complete_chain() {
        let error = chained_generation_error();
        match generation_failure_status(&error) {
            MediaRequestStatus::Failed(message) => assert_eq!(
                message,
                "generate_inner failed: step 7: projection failed: invalid image token"
            ),
            _ => panic!("generation error must produce a failed terminal status"),
        }
        assert_eq!(
            error
                .downcast_ref::<std::io::Error>()
                .expect("typed cause must be retained")
                .kind(),
            std::io::ErrorKind::InvalidData
        );
    }
    use std::ffi::CString;

    // ── Error code enum values ────────────────────────────────────────────

    #[test]
    fn test_error_code_discriminants() {
        assert_eq!(PcaiMediaErrorCode::Success as i32, 0);
        assert_eq!(PcaiMediaErrorCode::NotInitialized as i32, -1);
        assert_eq!(PcaiMediaErrorCode::ModelNotLoaded as i32, -2);
        assert_eq!(PcaiMediaErrorCode::InvalidInput as i32, -3);
        assert_eq!(PcaiMediaErrorCode::GenerationError as i32, -4);
        assert_eq!(PcaiMediaErrorCode::IoError as i32, -5);
        assert_eq!(PcaiMediaErrorCode::Unknown as i32, -99);
    }

    #[test]
    fn test_error_code_from_i32_roundtrip() {
        for &(value, expected) in &[
            (0_i32, PcaiMediaErrorCode::Success),
            (-1, PcaiMediaErrorCode::NotInitialized),
            (-2, PcaiMediaErrorCode::ModelNotLoaded),
            (-3, PcaiMediaErrorCode::InvalidInput),
            (-4, PcaiMediaErrorCode::GenerationError),
            (-5, PcaiMediaErrorCode::IoError),
            (-99, PcaiMediaErrorCode::Unknown),
            (-42, PcaiMediaErrorCode::Unknown), // unrecognised maps to Unknown
        ] {
            assert_eq!(
                PcaiMediaErrorCode::from_i32(value),
                expected,
                "from_i32({value}) should be {expected:?}"
            );
        }
    }

    // ── set_error / last_error round-trip ─────────────────────────────────

    #[test]
    fn test_set_error_and_last_error_roundtrip() {
        // Set an error and verify it is readable through the FFI surface.
        set_error("test error message", PcaiMediaErrorCode::GenerationError);

        let ptr = pcai_media_last_error();
        assert!(!ptr.is_null(), "last_error should be non-null after set_error");

        let retrieved = unsafe { CStr::from_ptr(ptr).to_str().expect("valid UTF-8") };
        assert_eq!(retrieved, "test error message");

        let code = pcai_media_last_error_code();
        assert_eq!(code, PcaiMediaErrorCode::GenerationError as i32);

        // Clean up for other tests running on the same thread.
        clear_error();
    }

    #[test]
    fn test_no_error_returns_null() {
        clear_error();
        let ptr = pcai_media_last_error();
        assert!(ptr.is_null(), "last_error should be null when no error is set");
        assert_eq!(pcai_media_last_error_code(), 0);
    }

    #[test]
    fn test_clear_error_resets_code() {
        set_error("temporary", PcaiMediaErrorCode::IoError);
        assert_eq!(pcai_media_last_error_code(), PcaiMediaErrorCode::IoError as i32);
        clear_error();
        assert_eq!(pcai_media_last_error_code(), PcaiMediaErrorCode::Success as i32);
    }

    // ── c_str_to_str ─────────────────────────────────────────────────────

    #[test]
    fn test_c_str_to_str_null_pointer() {
        let result = unsafe { c_str_to_str(std::ptr::null()) };
        assert!(result.is_err(), "c_str_to_str should return Err for a null pointer");
        assert_eq!(result.unwrap_err(), PcaiMediaErrorCode::InvalidInput);
    }

    #[test]
    fn test_c_str_to_str_valid_string() {
        let cstring = CString::new("hello pcai_media").expect("valid C string");
        let result = unsafe { c_str_to_str(cstring.as_ptr()) };
        assert!(result.is_ok(), "c_str_to_str should succeed for valid UTF-8");
        assert_eq!(result.unwrap(), "hello pcai_media");
    }

    #[test]
    fn test_c_str_to_str_empty_string() {
        let cstring = CString::new("").expect("empty C string");
        let result = unsafe { c_str_to_str(cstring.as_ptr()) };
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "");
    }

    // ── double-init doesn't panic ─────────────────────────────────────────

    #[test]
    fn test_double_init_no_panic() {
        let device = CString::new("cpu").expect("valid C string");
        let first = pcai_media_init(device.as_ptr());
        assert_eq!(first, PcaiMediaErrorCode::Success as i32);
        let second = pcai_media_init(device.as_ptr());
        assert_eq!(second, PcaiMediaErrorCode::Success as i32);
    }

    #[test]
    fn test_init_null_device_defaults_to_cuda() {
        // NULL device should succeed (defaults to "cuda:0" but validation does
        // not actually open the device, so it is always Success here).
        let result = pcai_media_init(std::ptr::null());
        assert_eq!(result, PcaiMediaErrorCode::Success as i32);
    }

    #[test]
    fn test_init_invalid_device_returns_error() {
        let bad = CString::new("tpu:0").expect("valid C string");
        let result = pcai_media_init(bad.as_ptr());
        assert_eq!(result, PcaiMediaErrorCode::InvalidInput as i32);
    }

    // ── shutdown without init doesn't panic ──────────────────────────────

    #[test]
    fn test_shutdown_without_init_no_panic() {
        // This test must be robust to other tests having already initialised
        // the global state on this test-runner process.  The important thing
        // is that it does not panic.
        pcai_media_shutdown();
    }

    #[test]
    fn test_shutdown_twice_no_panic() {
        let device = CString::new("cpu").expect("valid C string");
        pcai_media_init(device.as_ptr());
        pcai_media_shutdown();
        pcai_media_shutdown(); // second call must be a no-op
    }

    // ── free helpers ─────────────────────────────────────────────────────

    #[test]
    fn test_free_string_null_no_panic() {
        pcai_media_free_string(std::ptr::null_mut());
    }

    #[test]
    fn test_free_bytes_null_no_panic() {
        pcai_media_free_bytes(std::ptr::null_mut(), 0);
    }

    #[test]
    fn test_free_bytes_valid_buffer() {
        // Allocate a buffer the same way the library would and free it.
        let buf: Vec<u8> = vec![1u8, 2, 3, 4];
        let len = buf.len();
        let ptr = {
            let mut b = buf;
            let p = b.as_mut_ptr();
            std::mem::forget(b);
            p
        };
        // This must not panic or produce UB under Miri / Valgrind.
        pcai_media_free_bytes(ptr, len);
    }

    // ── understand_image null safety ────────────────────────────────────

    #[test]
    fn test_understand_image_null_image_path() {
        let prompt = CString::new("describe this").expect("valid");
        let result = pcai_media_understand_image(std::ptr::null(), prompt.as_ptr(), 128, 0.7);
        assert!(result.is_null(), "null image_path should return null");
        assert_eq!(pcai_media_last_error_code(), PcaiMediaErrorCode::InvalidInput as i32,);
    }

    #[test]
    fn test_understand_image_null_prompt() {
        let path = CString::new("test.png").expect("valid");
        let result = pcai_media_understand_image(path.as_ptr(), std::ptr::null(), 128, 0.7);
        assert!(result.is_null(), "null prompt should return null");
        assert_eq!(pcai_media_last_error_code(), PcaiMediaErrorCode::InvalidInput as i32,);
    }

    #[test]
    fn test_understand_image_no_model_loaded() {
        // Ensure init has run (other tests may have already done this).
        let device = CString::new("cpu").expect("valid");
        pcai_media_init(device.as_ptr());
        pcai_media_shutdown(); // clear any loaded model

        let path = CString::new("test.png").expect("valid");
        let prompt = CString::new("describe").expect("valid");
        let result = pcai_media_understand_image(path.as_ptr(), prompt.as_ptr(), 128, 0.7);
        assert!(result.is_null(), "should fail without loaded model");
        // Could be ModelNotLoaded or IoError depending on execution order.
        assert_ne!(pcai_media_last_error_code(), 0);
    }

    // ── generate_image_bytes null safety ────────────────────────────────

    #[test]
    fn test_generate_image_bytes_null_out_data() {
        let prompt = CString::new("test").expect("valid");
        let mut len: usize = 0;
        let result =
            pcai_media_generate_image_bytes(prompt.as_ptr(), 0.0, 0.0, std::ptr::null_mut(), &mut len as *mut usize);
        assert_eq!(result, PcaiMediaErrorCode::InvalidInput as i32);
    }

    #[test]
    fn test_generate_image_bytes_null_out_len() {
        let prompt = CString::new("test").expect("valid");
        let mut data: *mut u8 = std::ptr::null_mut();
        let result = pcai_media_generate_image_bytes(
            prompt.as_ptr(),
            0.0,
            0.0,
            &mut data as *mut *mut u8,
            std::ptr::null_mut(),
        );
        assert_eq!(result, PcaiMediaErrorCode::InvalidInput as i32);
    }

    #[test]
    fn test_generate_image_bytes_null_prompt() {
        let mut data: *mut u8 = std::ptr::null_mut();
        let mut len: usize = 0;
        let result = pcai_media_generate_image_bytes(
            std::ptr::null(),
            0.0,
            0.0,
            &mut data as *mut *mut u8,
            &mut len as *mut usize,
        );
        assert_eq!(result, PcaiMediaErrorCode::InvalidInput as i32);
    }

    // ── async FFI tests ──────────────────────────────────────────────────

    #[test]
    fn test_async_generate_null_prompt() {
        let out = CString::new("out.png").expect("valid");
        let id = pcai_media_generate_image_async(std::ptr::null(), 0.0, 0.0, out.as_ptr());
        assert_eq!(id, -1, "null prompt should return -1");
        assert_eq!(pcai_media_last_error_code(), PcaiMediaErrorCode::InvalidInput as i32,);
    }

    #[test]
    fn test_async_generate_null_output_path() {
        let prompt = CString::new("test").expect("valid");
        let id = pcai_media_generate_image_async(prompt.as_ptr(), 0.0, 0.0, std::ptr::null());
        assert_eq!(id, -1, "null output_path should return -1");
        assert_eq!(pcai_media_last_error_code(), PcaiMediaErrorCode::InvalidInput as i32,);
    }

    #[test]
    fn test_async_generate_no_model_loaded() {
        let device = CString::new("cpu").expect("valid");
        pcai_media_init(device.as_ptr());
        pcai_media_shutdown(); // ensure no model

        let prompt = CString::new("test").expect("valid");
        let out = CString::new("out.png").expect("valid");
        let id = pcai_media_generate_image_async(prompt.as_ptr(), 5.0, 1.0, out.as_ptr());
        assert_eq!(id, -1, "should fail without loaded model");
        assert_ne!(pcai_media_last_error_code(), 0);
    }

    #[test]
    fn test_poll_result_unknown_id() {
        let result = pcai_media_poll_result(999_999);
        assert_eq!(result.status, -1, "unknown ID should return status -1");
        assert!(result.text.is_null());
    }

    #[test]
    fn test_cancel_unknown_id() {
        let rc = pcai_media_cancel(999_999);
        assert_eq!(rc, -1, "cancelling an unknown ID should return -1");
    }

    #[test]
    fn test_async_result_repr_c() {
        // Verify PcaiMediaAsyncResult has the expected layout for FFI callers.
        let expected_size = if std::mem::size_of::<*mut c_char>() == 8 {
            16 // 64-bit: i32 (4) + 4 padding + pointer (8)
        } else {
            8 // 32-bit: i32 (4) + pointer (4)
        };
        assert_eq!(
            std::mem::size_of::<PcaiMediaAsyncResult>(),
            expected_size,
            "PcaiMediaAsyncResult layout must match expected C ABI size"
        );
    }

    // ── Inference gate and lock scope ─────────────────────────────────────
    //
    // The global state mutex must never be held across generation or model
    // load, otherwise pcai_media_poll_result and pcai_media_cancel block for
    // the whole run (the bug fixed for pcai_inference in #374).  These tests
    // use a mock generate closure through `execute_async_request`, so no
    // model, GPU or download is needed.

    /// Register a fresh async request id in the global request map.
    fn register_request(status: MediaRequestStatus) -> i64 {
        let id = NEXT_REQUEST_ID.fetch_add(1, AtomicOrdering::SeqCst);
        lock_state_recovering().requests.insert(id, status);
        id
    }

    /// Run `call` on a helper thread; fail instead of hanging if it blocks.
    fn finishes_within<T: Send + 'static>(what: &str, call: impl FnOnce() -> T + Send + 'static) -> T {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(call());
        });
        match rx.recv_timeout(std::time::Duration::from_secs(10)) {
            Ok(value) => value,
            Err(_) => panic!("{what} blocked while inference was in flight"),
        }
    }

    /// Poll a request through the C ABI, freeing any returned text.
    fn poll_status(id: i64) -> i32 {
        let result = pcai_media_poll_result(id);
        pcai_media_free_string(result.text);
        result.status
    }

    fn mock_image() -> anyhow::Result<image::RgbImage> {
        Ok(image::RgbImage::new(1, 1))
    }

    // Protects: one generation at a time, so model and KV-cache memory stay bounded.
    // Detects: removing the inference gate lets queued async requests overlap.
    // Needs: no model or GPU; a temp directory for the PNG outputs.
    #[test]
    fn test_async_requests_are_serialized_and_all_complete() {
        use std::sync::atomic::AtomicUsize;
        let tmp = tempfile::tempdir().expect("tempdir");
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mut workers = Vec::new();
        let mut ids = Vec::new();
        for index in 0..4 {
            let id = register_request(MediaRequestStatus::Pending);
            ids.push(id);
            let output = tmp
                .path()
                .join(format!("out{index}.png"))
                .to_string_lossy()
                .into_owned();
            let (active, peak) = (active.clone(), peak.clone());
            workers.push(std::thread::spawn(move || {
                execute_async_request(id, output, move || {
                    let now = active.fetch_add(1, AtomicOrdering::SeqCst) + 1;
                    peak.fetch_max(now, AtomicOrdering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(25));
                    active.fetch_sub(1, AtomicOrdering::SeqCst);
                    mock_image()
                });
            }));
        }
        for worker in workers {
            worker.join().expect("worker must not panic");
        }
        assert_eq!(peak.load(AtomicOrdering::SeqCst), 1, "generations must not overlap");
        for id in ids {
            assert_eq!(poll_status(id), 2, "request {id} must complete");
        }
    }

    // Protects: poll, cancel, init and shutdown stay responsive during a generation.
    // Detects: holding the state mutex across generation (poll would time out here),
    //   a queued request leaving Pending before it owns the gate, a cancelled queued
    //   request still running.
    // Needs: no model or GPU; the in-flight generation is a mock blocked on a channel.
    #[test]
    fn test_poll_and_cancel_respond_during_generation_and_cancel_skips_queued_work() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let running = register_request(MediaRequestStatus::Pending);
        let queued = register_request(MediaRequestStatus::Pending);
        let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let ran_queued = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let running_out = tmp.path().join("running.png").to_string_lossy().into_owned();
        let first = std::thread::spawn(move || {
            execute_async_request(running, running_out, move || {
                entered_tx.send(()).expect("test is listening");
                release_rx
                    .recv_timeout(std::time::Duration::from_secs(10))
                    .map_err(|_| anyhow::anyhow!("test never released the generation"))?;
                mock_image()
            });
        });
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("first generation must start");

        let queued_out = tmp.path().join("queued.png").to_string_lossy().into_owned();
        let flag = ran_queued.clone();
        let second = std::thread::spawn(move || {
            execute_async_request(queued, queued_out, move || {
                flag.store(true, AtomicOrdering::SeqCst);
                mock_image()
            });
        });

        // The first generation still owns the gate; every status path must answer.
        assert_eq!(finishes_within("poll", move || poll_status(running)), 1);
        assert_eq!(
            finishes_within("poll", move || poll_status(queued)),
            0,
            "queued stays Pending"
        );
        assert_eq!(finishes_within("cancel", move || pcai_media_cancel(queued)), 0);
        assert_eq!(finishes_within("poll", move || poll_status(queued)), 4);
        assert_eq!(finishes_within("cancel", || pcai_media_cancel(999_998)), -1);
        assert_eq!(
            finishes_within("init", || pcai_media_init(c"cpu".as_ptr())),
            PcaiMediaErrorCode::Success as i32
        );
        finishes_within("shutdown", || pcai_media_shutdown());

        release_tx.send(()).expect("generation is waiting");
        first.join().expect("first worker must not panic");
        second.join().expect("second worker must not panic");
        assert!(
            !ran_queued.load(AtomicOrdering::SeqCst),
            "a request cancelled while queued must never generate"
        );
        assert_eq!(poll_status(running), 2);
    }

    // Protects: every synchronous inference entry point (generate, bytes, understand,
    //   load) queues behind the gate, so model replacement cannot race a generation.
    // Detects: an entry point that skips acquire_inference_gate, or one that holds the
    //   state mutex while waiting for it (the status calls below would then hang).
    // Needs: no model loaded in this process; the gate is held by the test as a stand-in
    //   for a running generation.
    #[test]
    fn test_sync_entry_points_queue_behind_gate_without_blocking_status() {
        let model_dir = tempfile::tempdir().expect("tempdir");
        let model_dir = CString::new(model_dir.path().to_string_lossy().into_owned()).unwrap();
        let (tx, rx) = std::sync::mpsc::channel::<(&'static str, i64)>();
        let gate = acquire_inference_gate();

        let mut workers = Vec::new();
        let sender = tx.clone();
        workers.push(std::thread::spawn(move || {
            let rc = pcai_media_generate_image(c"a dog".as_ptr(), 0.0, 0.0, c"x.png".as_ptr());
            sender.send(("generate", i64::from(rc))).unwrap();
        }));
        let sender = tx.clone();
        workers.push(std::thread::spawn(move || {
            let mut data: *mut u8 = std::ptr::null_mut();
            let mut len = 0usize;
            let rc = pcai_media_generate_image_bytes(c"a dog".as_ptr(), 0.0, 0.0, &mut data, &mut len);
            sender.send(("bytes", i64::from(rc))).unwrap();
        }));
        let sender = tx.clone();
        workers.push(std::thread::spawn(move || {
            let text = pcai_media_understand_image(c"x.png".as_ptr(), c"describe".as_ptr(), 8, 0.0);
            sender.send(("understand", i64::from(text.is_null()))).unwrap();
        }));
        let sender = tx.clone();
        workers.push(std::thread::spawn(move || {
            let rc = pcai_media_load_model(model_dir.as_ptr(), 0);
            sender.send(("load", i64::from(rc))).unwrap();
        }));
        drop(tx);

        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(300)).is_err(),
            "inference entry points must wait for the gate"
        );
        assert_eq!(finishes_within("poll", || poll_status(999_997)), -1);
        assert_eq!(finishes_within("cancel", || pcai_media_cancel(999_997)), -1);
        drop(gate);

        let mut outcomes = std::collections::HashMap::new();
        for _ in 0..4 {
            let (name, value) = rx
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("entry points must finish after the gate is released");
            outcomes.insert(name, value);
        }
        for worker in workers {
            worker.join().expect("entry point must not panic");
        }
        assert_eq!(outcomes["generate"], PcaiMediaErrorCode::ModelNotLoaded as i64);
        assert_eq!(outcomes["bytes"], PcaiMediaErrorCode::ModelNotLoaded as i64);
        assert_eq!(outcomes["understand"], 1, "understand returns NULL without a model");
        assert_ne!(outcomes["load"], 0, "an empty model directory must not load");
    }

    // Protects: an incomplete checkpoint is refused at the FFI boundary with an existing
    //   status code and a diagnostic naming the missing tensors, and leaves no model loaded.
    // Detects: a weight-completeness check that is bypassed or whose typed error is
    //   flattened into a success; a half-loaded pipeline left behind after failure.
    // Needs: a temp directory only; the 7B constructor metadata uses scalar-backed views,
    //   so nothing large is allocated and no GPU or download is involved.
    #[test]
    fn test_load_model_rejects_incomplete_checkpoint_and_stays_unloaded() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let config = pcai_media_model::config::JanusConfig::janus_pro_7b();
        std::fs::write(tmp.path().join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
        let unrelated = candle_core::Tensor::zeros(1, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        candle_core::safetensors::save(
            &HashMap::from([("unrelated.weight", unrelated)]),
            tmp.path().join("model.safetensors"),
        )
        .unwrap();
        let path = CString::new(tmp.path().to_string_lossy().into_owned()).unwrap();

        assert_eq!(pcai_media_init(c"cpu".as_ptr()), PcaiMediaErrorCode::Success as i32);
        let rc = pcai_media_load_model(path.as_ptr(), 0);
        assert_eq!(rc, PcaiMediaErrorCode::GenerationError as i32);
        let message = unsafe { CStr::from_ptr(pcai_media_last_error()) }
            .to_str()
            .expect("diagnostic should be UTF-8")
            .to_string();
        assert!(message.contains("incomplete safetensors weights"), "{message}");
        assert!(message.contains("required tensors"), "{message}");

        let rc = pcai_media_generate_image(c"a dog".as_ptr(), 0.0, 0.0, c"x.png".as_ptr());
        assert_eq!(rc, PcaiMediaErrorCode::ModelNotLoaded as i32);
    }

    // ── Staged output: cancel must never touch the destination ────────────

    /// Sorted file names in `dir`, to assert no staging file is left behind.
    fn dir_listing(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .expect("read_dir")
            .map(|entry| entry.expect("dir entry").file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    /// Poll a request and return its status with any text.
    fn poll_full(id: i64) -> (i32, Option<String>) {
        let result = pcai_media_poll_result(id);
        let text = if result.text.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(result.text) }.to_string_lossy().into_owned())
        };
        pcai_media_free_string(result.text);
        (result.status, text)
    }

    const ORIGINAL: &[u8] = b"original destination bytes";

    // Protects: a request cancelled while generating never creates or overwrites
    //   the caller's destination file, and leaves no staging file.
    // Detects: encoding straight to the destination before the cancel check.
    // Needs: a temp directory; the cancel is issued from inside the mock generate.
    #[test]
    fn test_cancel_during_generation_leaves_destination_untouched() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let existing = tmp.path().join("existing.png");
        fs::write(&existing, ORIGINAL).unwrap();
        let fresh = tmp.path().join("fresh.png");
        for dest in [&existing, &fresh] {
            let id = register_request(MediaRequestStatus::Pending);
            execute_async_request(id, dest.to_string_lossy().into_owned(), move || {
                assert_eq!(pcai_media_cancel(id), 0, "request is Running while generating");
                mock_image()
            });
            assert_eq!(poll_status(id), 4, "poll reports the cancellation");
        }
        assert_eq!(
            fs::read(&existing).unwrap(),
            ORIGINAL,
            "existing destination keeps its bytes"
        );
        assert!(!fresh.exists(), "cancelled request must not create the destination");
        assert_eq!(dir_listing(tmp.path()), ["existing.png"], "no staging file left");
    }

    // Protects: a cancel that lands after encoding but before publish still leaves the
    //   destination untouched and removes the staging file.
    // Detects: moving the cancellation check after the publish step (the destination
    //   would be replaced and Complete reported).
    // Needs: a temp directory; the cancel is issued from inside the mock encoder.
    #[test]
    fn test_cancel_after_encode_before_publish_discards_staged_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let existing = tmp.path().join("existing.png");
        fs::write(&existing, ORIGINAL).unwrap();
        let fresh = tmp.path().join("fresh.png");
        for dest in [&existing, &fresh] {
            let id = register_request(MediaRequestStatus::Pending);
            let dest_path = dest.to_string_lossy().into_owned();
            let encode = encode_for(Path::new(&dest_path));
            execute_async_request_with(id, dest_path, mock_image, move |image, file| {
                encode(image, file)?;
                assert_eq!(pcai_media_cancel(id), 0, "request is still Running after encode");
                Ok(())
            });
            assert_eq!(poll_status(id), 4);
        }
        assert_eq!(fs::read(&existing).unwrap(), ORIGINAL);
        assert!(!fresh.exists());
        assert_eq!(dir_listing(tmp.path()), ["existing.png"], "staging file removed");
    }

    // Protects: a normal completion publishes the whole file and removes the staging file.
    // Detects: a publish that leaves the `.partial` name behind or writes a truncated file.
    // Needs: a temp directory only.
    #[test]
    fn test_completion_publishes_full_file_and_leaves_no_partial() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dest = tmp.path().join("out.png");
        fs::write(&dest, ORIGINAL).unwrap();
        let id = register_request(MediaRequestStatus::Pending);
        let dest_path = dest.to_string_lossy().into_owned();
        execute_async_request(id, dest_path.clone(), mock_image);
        let (status, text) = poll_full(id);
        assert_eq!(status, 2);
        assert_eq!(text.as_deref(), Some(dest_path.as_str()));
        let decoded = image::open(&dest).expect("published file is a complete PNG");
        assert_eq!(decoded.width(), 1);
        assert_eq!(dir_listing(tmp.path()), ["out.png"]);
    }

    // Protects: a failed write reports Failed, leaves no partial file and never touches
    //   an existing destination; a failed publish does the same.
    // Detects: encoding in place, or a staging file leaked on the error path.
    // Needs: a temp directory; the encoder writes some bytes then fails, and a
    //   directory squatting on the destination makes the rename fail.
    #[test]
    fn test_write_and_publish_failures_leave_no_partial_and_keep_destination() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let existing = tmp.path().join("existing.png");
        fs::write(&existing, ORIGINAL).unwrap();

        let id = register_request(MediaRequestStatus::Pending);
        execute_async_request_with(id, existing.to_string_lossy().into_owned(), mock_image, |_, file| {
            file.write_all(b"half an image")?;
            Err(io::Error::other("disk full"))
        });
        let (status, text) = poll_full(id);
        assert_eq!(status, 3);
        let text = text.expect("failure carries a message");
        assert!(text.contains("Save failed") && text.contains("disk full"), "{text}");
        assert_eq!(fs::read(&existing).unwrap(), ORIGINAL);
        assert_eq!(
            dir_listing(tmp.path()),
            ["existing.png"],
            "no partial after write failure"
        );

        let blocked = tmp.path().join("blocked.png");
        fs::create_dir(&blocked).unwrap();
        let id = register_request(MediaRequestStatus::Pending);
        execute_async_request(id, blocked.to_string_lossy().into_owned(), mock_image);
        let (status, text) = poll_full(id);
        assert_eq!(status, 3, "publishing over a directory must fail");
        assert!(text.expect("message").contains("Save failed"));
        assert!(blocked.is_dir());
        assert_eq!(
            dir_listing(tmp.path()),
            ["blocked.png", "existing.png"],
            "no partial after publish failure"
        );
    }

    // Protects: the synchronous save path is atomic too: a failure keeps the old file.
    // Detects: `image.save` writing in place and leaving a truncated destination.
    // Needs: a temp directory; the destination name has no usable image extension.
    #[test]
    fn test_sync_write_failure_keeps_existing_destination_and_leaves_no_partial() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let existing = tmp.path().join("existing.unknownext");
        fs::write(&existing, ORIGINAL).unwrap();
        assert!(write_image_atomically(&image::RgbImage::new(1, 1), &existing).is_err());
        assert_eq!(fs::read(&existing).unwrap(), ORIGINAL);
        assert_eq!(dir_listing(tmp.path()), ["existing.unknownext"]);

        let good = tmp.path().join("good.png");
        write_image_atomically(&image::RgbImage::new(2, 2), &good).expect("png write");
        assert_eq!(image::open(&good).unwrap().width(), 2);
        assert_eq!(dir_listing(tmp.path()), ["existing.unknownext", "good.png"]);
    }

    // Protects: a destination near the 255-character component limit still publishes,
    //   on both the asynchronous and synchronous paths.
    // Detects: a staging name derived from the destination basename (248 + suffix
    //   exceeds 255 and the exclusive create fails with an invalid-name error).
    // Needs: a temp directory on a filesystem with a 255-character component limit.
    #[test]
    fn test_long_valid_destination_name_publishes_on_both_paths() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let name = format!("{}.png", "a".repeat(244));
        assert_eq!(name.len(), 248);
        let dest = tmp.path().join(&name);

        let id = register_request(MediaRequestStatus::Pending);
        execute_async_request(id, dest.to_string_lossy().into_owned(), mock_image);
        let (status, text) = poll_full(id);
        assert_eq!(status, 2, "async publish failed: {text:?}");
        assert_eq!(dir_listing(tmp.path()), std::slice::from_ref(&name));
        fs::remove_file(&dest).unwrap();

        write_image_atomically(&image::RgbImage::new(2, 2), &dest).expect("sync publish");
        assert_eq!(image::open(&dest).unwrap().width(), 2);
        assert_eq!(dir_listing(tmp.path()), [name]);
    }

    // Protects: staging names are short, independent of the destination, unique per
    //   call and live next to the destination.
    // Detects: a name that embeds the destination basename or repeats between calls.
    // Needs: nothing on disk.
    #[test]
    fn test_staging_path_is_short_unique_and_beside_destination() {
        let dest = Path::new("some_dir").join("x".repeat(240));
        let first = staging_path(&dest);
        let second = staging_path(&dest);
        assert_ne!(first, second);
        assert_eq!(first.parent(), dest.parent());
        let leaf = first.file_name().unwrap().to_string_lossy().into_owned();
        assert!(leaf.starts_with(".pcai-") && leaf.ends_with(".partial"), "{leaf}");
        assert!(leaf.len() < 40, "{leaf}");
        assert_eq!(staging_path(Path::new("bare.png")).parent(), Some(Path::new("")));
    }

    // ── Model slot lifecycle: shutdown must win over an in-flight load ────

    /// Model stand-in that records when it is dropped and whether the slot
    /// lock was held at that moment.
    struct Probe {
        slot: Arc<Mutex<ModelSlot<Probe>>>,
        drops: Arc<std::sync::atomic::AtomicUsize>,
        dropped_under_lock: Arc<std::sync::atomic::AtomicBool>,
    }

    impl Drop for Probe {
        fn drop(&mut self) {
            if self.slot.try_lock().is_err() {
                self.dropped_under_lock.store(true, AtomicOrdering::SeqCst);
            }
            self.drops.fetch_add(1, AtomicOrdering::SeqCst);
        }
    }

    struct ProbeRig {
        slot: Arc<Mutex<ModelSlot<Probe>>>,
        drops: Arc<std::sync::atomic::AtomicUsize>,
        dropped_under_lock: Arc<std::sync::atomic::AtomicBool>,
    }

    impl ProbeRig {
        fn new() -> Self {
            Self {
                slot: Arc::new(Mutex::new(ModelSlot::new())),
                drops: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                dropped_under_lock: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            }
        }
        fn probe(&self) -> Probe {
            Probe {
                slot: self.slot.clone(),
                drops: self.drops.clone(),
                dropped_under_lock: self.dropped_under_lock.clone(),
            }
        }
        fn drops(&self) -> usize {
            self.drops.load(AtomicOrdering::SeqCst)
        }
        fn dropped_under_lock(&self) -> bool {
            self.dropped_under_lock.load(AtomicOrdering::SeqCst)
        }
    }

    // Protects: shutdown during an in-flight load is final: the finished model is
    //   discarded (outside the slot lock) and nothing is loaded afterwards.
    // Detects: an unconditional install after load (the pre-fix behaviour), or a stale
    //   model freed while the slot lock is held.
    // Needs: no model; the paused load is a closure blocked on a channel.
    #[test]
    fn test_shutdown_during_load_discards_the_stale_model() {
        let rig = ProbeRig::new();
        let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let slot = rig.slot.clone();
        let probe = rig.probe();
        let loader = std::thread::spawn(move || {
            load_into_slot(&slot, move || {
                started_tx.send(()).unwrap();
                release_rx
                    .recv_timeout(std::time::Duration::from_secs(10))
                    .map_err(|_| anyhow::anyhow!("test never released the load"))?;
                Ok(probe)
            })
        });
        started_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("load starts");

        let slot = rig.slot.clone();
        finishes_within("slot shutdown", move || slot_shutdown(&slot));

        release_tx.send(()).unwrap();
        let installed = loader
            .join()
            .expect("loader must not panic")
            .expect("load itself succeeds");
        assert!(!installed, "a load that finished after shutdown must not install");
        assert!(slot_current(&rig.slot).is_none(), "slot stays unloaded");
        assert_eq!(rig.drops(), 1, "the stale model is dropped");
        assert!(
            !rig.dropped_under_lock(),
            "the stale model is dropped outside the slot lock"
        );
    }

    // Protects: the reverse ordering still works: shutdown, then a new load, installs
    //   normally, and replacing a model frees the old one outside the lock.
    // Detects: an epoch check that rejects every load after the first shutdown.
    // Needs: no model.
    #[test]
    fn test_load_after_shutdown_installs_and_replacement_drops_outside_lock() {
        let rig = ProbeRig::new();
        slot_shutdown(&rig.slot);
        let first = rig.probe();
        assert!(load_into_slot(&rig.slot, move || Ok(first)).unwrap());
        assert!(slot_current(&rig.slot).is_some());
        assert_eq!(rig.drops(), 0);

        let second = rig.probe();
        assert!(load_into_slot(&rig.slot, move || Ok(second)).unwrap());
        assert_eq!(rig.drops(), 1, "the replaced model is dropped");
        assert!(!rig.dropped_under_lock());

        slot_shutdown(&rig.slot);
        assert!(slot_current(&rig.slot).is_none());
        assert_eq!(rig.drops(), 2, "shutdown releases the model");
        assert!(!rig.dropped_under_lock());
    }

    // Protects: a failed load leaves the slot, and the epoch, alone.
    // Detects: an error path that clears a previously loaded model.
    // Needs: no model.
    #[test]
    fn test_failed_load_keeps_existing_model() {
        let rig = ProbeRig::new();
        let kept = rig.probe();
        assert!(load_into_slot(&rig.slot, move || Ok(kept)).unwrap());
        let error = load_into_slot(&rig.slot, || -> anyhow::Result<Probe> {
            anyhow::bail!("bad checkpoint")
        })
        .expect_err("load error propagates");
        assert!(error.to_string().contains("bad checkpoint"));
        assert!(slot_current(&rig.slot).is_some());
        assert_eq!(rig.drops(), 0);
    }

    // Protects: the exported pcai_media_shutdown invalidates in-flight loads of the
    //   real global slot (not just the generic helper).
    // Detects: shutdown that stops touching MODEL_SLOT, or only clears the handle
    //   without bumping the epoch.
    // Needs: no model.
    #[test]
    fn test_exported_shutdown_bumps_the_global_epoch() {
        let before = slot_begin_load(&MODEL_SLOT);
        pcai_media_shutdown();
        assert_ne!(slot_begin_load(&MODEL_SLOT), before);
        assert!(slot_current(&MODEL_SLOT).is_none());
    }

    // ── Slow publish must not block other requests ──────────────────────

    /// Run `call` on a helper thread and require it to answer within 100 ms.
    fn answers_within_100ms<T: Send + 'static>(what: &str, call: impl FnOnce() -> T + Send + 'static) -> T {
        let (value, elapsed) = finishes_within(what, move || {
            let started = std::time::Instant::now();
            let value = call();
            (value, started.elapsed())
        });
        assert!(
            elapsed < std::time::Duration::from_millis(100),
            "{what} took {elapsed:?} while another request was publishing"
        );
        value
    }

    // Protects: a publish on slow or stalled storage blocks nobody else: poll and cancel
    //   of other requests answer within 100 ms while one request sits in its rename.
    // Detects: the rename (or any file IO) running under the global request-table lock.
    // Needs: no model; the rename is a mock publish step blocked on a channel.
    #[test]
    fn test_slow_publish_does_not_block_status_of_other_requests() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let publishing = register_request(MediaRequestStatus::Pending);
        let other_running = register_request(MediaRequestStatus::Running);
        let other_queued = register_request(MediaRequestStatus::Pending);
        let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();

        let dest = tmp.path().join("slow.png");
        let dest_path = dest.to_string_lossy().into_owned();
        let encode = encode_for(Path::new(&dest_path));
        let worker = std::thread::spawn(move || {
            execute_async_request_staged(publishing, dest_path, mock_image, encode, move |staged, target| {
                entered_tx.send(()).expect("test is listening");
                release_rx
                    .recv_timeout(std::time::Duration::from_secs(10))
                    .map_err(|_| io::Error::other("test never released the rename"))?;
                fs::rename(staged, target)
            });
        });
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("worker reaches the rename");

        // The publishing request is mid-rename; everything else must still answer.
        assert_eq!(
            answers_within_100ms("poll other", move || poll_status(other_running)),
            1
        );
        assert_eq!(
            answers_within_100ms("poll queued", move || poll_status(other_queued)),
            0
        );
        assert_eq!(
            answers_within_100ms("cancel queued", move || pcai_media_cancel(other_queued)),
            0
        );
        assert_eq!(
            answers_within_100ms("poll cancelled", move || poll_status(other_queued)),
            4
        );
        assert_eq!(
            answers_within_100ms("cancel unknown", || pcai_media_cancel(999_996)),
            -1
        );
        assert_eq!(
            answers_within_100ms("cancel other", move || pcai_media_cancel(other_running)),
            0
        );
        assert_eq!(
            answers_within_100ms("poll other", move || poll_status(other_running)),
            4
        );

        // Same request: it reports running while publishing, and a cancel loses.
        assert_eq!(
            answers_within_100ms("poll publishing", move || poll_status(publishing)),
            1
        );
        assert_eq!(
            answers_within_100ms("cancel publishing", move || pcai_media_cancel(publishing)),
            -1
        );

        release_tx.send(()).unwrap();
        worker.join().expect("worker must not panic");
        let (status, text) = poll_full(publishing);
        assert_eq!(status, 2, "the publish that already started completes: {text:?}");
        assert_eq!(image::open(&dest).expect("published file is a complete PNG").width(), 1);
        assert_eq!(dir_listing(tmp.path()), ["slow.png"]);
    }

    // Protects: a failing rename while publishing reports Failed, removes the staging
    //   file and leaves the destination alone, again without the global lock.
    // Detects: an error path that leaves the request Publishing or leaks the partial file.
    // Needs: a temp directory; the publish step is a mock that fails.
    #[test]
    fn test_failed_publish_step_reports_failed_and_cleans_up() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let existing = tmp.path().join("existing.png");
        fs::write(&existing, ORIGINAL).unwrap();
        let id = register_request(MediaRequestStatus::Pending);
        let dest = existing.to_string_lossy().into_owned();
        let encode = encode_for(Path::new(&dest));
        execute_async_request_staged(id, dest, mock_image, encode, |_, _| {
            Err(io::Error::other("share went away"))
        });
        let (status, text) = poll_full(id);
        assert_eq!(status, 3);
        assert!(text.expect("message").contains("share went away"));
        assert_eq!(fs::read(&existing).unwrap(), ORIGINAL);
        assert_eq!(dir_listing(tmp.path()), ["existing.png"]);
    }
}
