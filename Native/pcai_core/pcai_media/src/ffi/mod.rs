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
use std::os::raw::c_char;
use std::sync::atomic::{AtomicI64, Ordering as AtomicOrdering};
use std::sync::{Mutex, OnceLock};

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
    /// Loaded generation pipeline (None until `pcai_media_load_model` succeeds).
    pipeline: Option<GenerationPipeline>,
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
            pipeline: None,
            device: "cuda:0".to_string(),
            requests: HashMap::new(),
        })
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

    let state = get_state();
    let mut guard = match state.lock() {
        Ok(g) => g,
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
        device: guard.device.clone(),
        ..PipelineConfig::default()
    };

    // `GenerationPipeline::load` is synchronous (uses blocking I/O internally
    // through hf-hub and safetensors).  Call it directly — no async executor
    // is required for the load path.
    match GenerationPipeline::load(config) {
        Ok(pipeline) => {
            guard.pipeline = Some(pipeline);
            tracing::info!(model = %path_str, gpu_layers, "model loaded");
            PcaiMediaErrorCode::Success as i32
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

    let state = get_state();
    let guard = match state.lock() {
        Ok(g) => g,
        Err(e) => {
            set_error(
                format!("Failed to acquire state lock: {e}"),
                PcaiMediaErrorCode::GenerationError,
            );
            return PcaiMediaErrorCode::GenerationError as i32;
        }
    };

    // Apply caller-supplied hyper-parameters to the loaded pipeline's config
    // by overwriting the fields in a cloned config.  The pipeline is rebuilt
    // in-place only when cfg_scale or temperature differ from the defaults.
    let pipeline = match guard.pipeline.as_ref() {
        Some(p) => p,
        None => {
            set_error(
                "No model loaded — call pcai_media_load_model first",
                PcaiMediaErrorCode::ModelNotLoaded,
            );
            return PcaiMediaErrorCode::ModelNotLoaded as i32;
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
            set_error(
                format!("Image generation failed: {e}"),
                PcaiMediaErrorCode::GenerationError,
            );
            return PcaiMediaErrorCode::GenerationError as i32;
        }
    };

    // Save to PNG.
    if let Err(e) = image.save(&out_str) {
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

    let state = get_state();
    let guard = match state.lock() {
        Ok(g) => g,
        Err(e) => {
            set_error(
                format!("Failed to acquire state lock: {e}"),
                PcaiMediaErrorCode::GenerationError,
            );
            return std::ptr::null_mut();
        }
    };

    let pipeline = match guard.pipeline.as_ref() {
        Some(p) => p,
        None => {
            set_error(
                "No model loaded — call pcai_media_load_model first",
                PcaiMediaErrorCode::ModelNotLoaded,
            );
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
    let result = UnderstandingPipeline::understand(
        pipeline.model(),
        pipeline.tokenizer(),
        &dyn_image,
        &prompt_str,
        max_tokens,
        temperature,
        pipeline.device(),
        pipeline.dtype(),
        pipeline.siglip(),
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

    let state = get_state();
    let guard = match state.lock() {
        Ok(g) => g,
        Err(e) => {
            set_error(
                format!("Failed to acquire state lock: {e}"),
                PcaiMediaErrorCode::GenerationError,
            );
            return PcaiMediaErrorCode::GenerationError as i32;
        }
    };

    let pipeline = match guard.pipeline.as_ref() {
        Some(p) => p,
        None => {
            set_error(
                "No model loaded — call pcai_media_load_model first",
                PcaiMediaErrorCode::ModelNotLoaded,
            );
            return PcaiMediaErrorCode::ModelNotLoaded as i32;
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
            set_error(
                format!("Image generation failed: {e}"),
                PcaiMediaErrorCode::GenerationError,
            );
            return PcaiMediaErrorCode::GenerationError as i32;
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
    {
        let state = get_state();
        let guard = match state.lock() {
            Ok(g) => g,
            Err(e) => {
                set_error(
                    format!("Failed to lock state: {e}"),
                    PcaiMediaErrorCode::GenerationError,
                );
                return -1;
            }
        };
        if guard.pipeline.is_none() {
            set_error(
                "No model loaded — call pcai_media_load_model first",
                PcaiMediaErrorCode::ModelNotLoaded,
            );
            return -1;
        }
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
    let cfg_scale_val = cfg_scale;
    let temperature_val = temperature;
    std::thread::spawn(move || {
        // Transition: Pending → Running (unless already cancelled).
        {
            let state = get_state();
            if let Ok(mut g) = state.lock() {
                if matches!(g.requests.get(&id), Some(MediaRequestStatus::Cancelled)) {
                    return;
                }
                g.requests.insert(id, MediaRequestStatus::Running);
            }
        }

        // Run the synchronous generation under the lock.
        let result = {
            let state = get_state();
            let guard = match state.lock() {
                Ok(g) => g,
                Err(e) => {
                    if let Ok(mut g2) = get_state().lock() {
                        g2.requests
                            .insert(id, MediaRequestStatus::Failed(format!("Lock failed: {e}")));
                    }
                    return;
                }
            };

            // Check for cancellation before expensive work.
            if matches!(guard.requests.get(&id), Some(MediaRequestStatus::Cancelled)) {
                return;
            }

            let pipeline = match guard.pipeline.as_ref() {
                Some(p) => p,
                None => {
                    drop(guard);
                    if let Ok(mut g2) = get_state().lock() {
                        g2.requests
                            .insert(id, MediaRequestStatus::Failed("Model not loaded".to_string()));
                    }
                    return;
                }
            };

            let override_cfg = if cfg_scale_val > 0.0 {
                Some(cfg_scale_val as f64)
            } else {
                None
            };
            let override_temp = if temperature_val > 0.0 {
                Some(temperature_val as f64)
            } else {
                None
            };

            pipeline.generate_with_overrides(&prompt_str, override_cfg, override_temp)
        };

        // Store the terminal status.
        if let Ok(mut g) = get_state().lock() {
            if matches!(g.requests.get(&id), Some(MediaRequestStatus::Cancelled)) {
                return;
            }
            match result {
                Ok(image) => {
                    if let Err(e) = image.save(&out_str) {
                        g.requests
                            .insert(id, MediaRequestStatus::Failed(format!("Save failed: {e}")));
                    } else {
                        g.requests.insert(id, MediaRequestStatus::Complete(out_str));
                    }
                }
                Err(e) => {
                    g.requests.insert(id, MediaRequestStatus::Failed(format!("{e}")));
                }
            }
        }
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
        Some(MediaRequestStatus::Running) => PcaiMediaAsyncResult {
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
/// found or the request had already finished.
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
                &format!("failed to load upscale model: {e:#}"),
                PcaiMediaErrorCode::IoError,
            );
            return PcaiMediaErrorCode::IoError as i32;
        }
    };

    let img = match image::open(input) {
        Ok(i) => i,
        Err(e) => {
            set_error(&format!("failed to open input image: {e}"), PcaiMediaErrorCode::IoError);
            return PcaiMediaErrorCode::IoError as i32;
        }
    };

    let upscaled = match pipeline.upscale(&img) {
        Ok(u) => u,
        Err(e) => {
            set_error(&format!("upscale failed: {e:#}"), PcaiMediaErrorCode::GenerationError);
            return PcaiMediaErrorCode::GenerationError as i32;
        }
    };

    if let Err(e) = upscaled.save(output) {
        set_error(
            &format!("failed to save upscaled image: {e}"),
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
/// [`pcai_media_load_model`] is called again.  Calling this function before
/// [`pcai_media_init`] is safe and is a no-op.
#[no_mangle]
pub extern "C" fn pcai_media_shutdown() {
    clear_error();

    // If the global state has never been initialised there is nothing to do.
    if let Some(state) = MEDIA_STATE.get() {
        if let Ok(mut guard) = state.lock() {
            guard.pipeline = None;
            tracing::info!("pcai_media_shutdown: pipeline released");
        }
    }
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
}
