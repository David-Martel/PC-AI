//! GPU preflight readiness check.
//!
//! Combines GGUF model memory estimation, NVML VRAM audit, and verdict
//! logic into a single [`check_readiness`] call that returns a compact
//! structured result.
//!
//! # Examples
//!
//! ```ignore
//! use pcai_core_lib::preflight;
//!
//! let result = preflight::check_readiness("path/to/model.gguf", 8192)?;
//! println!("{}", serde_json::to_string(&result)?);
//! // {"verdict":"go","reason":"5800MB needed, GPU1 has 14884MB free",...}
//! ```

pub mod gguf;
pub mod vram_audit;

use std::path::Path;

use anyhow::Result;
use serde::Serialize;
use thiserror::Error;

pub use vram_audit::{GpuVramSnapshot, VramProcess};

/// Re-export the NVML singleton getter for use by [`vram_audit`].
pub(crate) use crate::gpu::get_nvml;

// ── Public types ───────────────────────────────────────────────────────────

/// Readiness verdict for a GPU preflight check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// Model fits comfortably (>20% headroom on best GPU).
    Go,
    /// Model fits but tight (<20% headroom) -- may OOM under load.
    Warn,
    /// Model does not fit on any available GPU.
    Fail,
}

/// Compact preflight result returned by [`check_readiness`] and
/// [`check_vram_state`].
#[derive(Debug, Clone, Serialize)]
pub struct PreflightResult {
    /// Go / Warn / Fail verdict.
    pub verdict: Verdict,
    /// Human-readable explanation of the verdict.
    pub reason: String,
    /// Estimated model memory requirement in mebibytes.
    pub model_estimate_mb: u64,
    /// Zero-based index of the best GPU (most free VRAM), if any.
    pub best_gpu_index: Option<u32>,
    /// Per-GPU VRAM snapshots at the time of the check.
    pub gpus: Vec<GpuVramSnapshot>,
}

/// A model that the readiness check declines to assess.
///
/// Returned (wrapped in [`anyhow::Error`]) by [`check_readiness`]; recover it
/// with `err.downcast_ref::<PreflightError>()`. Through the
/// `pcai_gpu_preflight_json` export it surfaces as a `fail` verdict whose reason
/// starts with `Preflight error:`, never as `go`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum PreflightError {
    /// The model uses NVFP4, and NVFP4 readiness is deferred.
    ///
    /// The GGUF reader sizes NVFP4 tensors from the upstream block layout, but
    /// does not yet validate the declared tensor offsets, alignment or the
    /// file extent. A file written with a different NVFP4 block layout, which
    /// llama.cpp refuses to load, sums to a plausible size, so an estimate
    /// could report `go` for a model that cannot load.
    #[error(
        "NVFP4 model not assessed: readiness for NVFP4 is deferred until tensor offsets, alignment and file extent \
         are validated, because a plausible size estimate can describe a file that llama.cpp refuses to load"
    )]
    Nvfp4NotAssessed,
}

// ── Public API ─────────────────────────────────────────────────────────────

/// Run a full preflight check for loading a GGUF model.
///
/// 1. Parses the GGUF header and tensor infos to estimate memory requirements
///    from the exact tensor sizes (never reading tensor data).
/// 2. Queries NVML for live VRAM state on all GPUs.
/// 3. Returns a verdict with the best GPU choice (or why it will not fit).
///
/// # What a [`Verdict::Go`] proves, and what it does not
///
/// A `go` proves one thing: the sum of the tensor sizes the header declares,
/// plus a KV cache at fp16 for `context_length` and 10% overhead, fits in the
/// free VRAM of one GPU with at least 20% to spare, at the moment of the check.
///
/// It does **not** prove that the file loads. The check reads the header only:
/// it does not validate the declared tensor offsets, their alignment or the
/// file extent, and it does not look at tensor data, so a truncated or
/// corrupted file can still say `go`. It does not prove the inference backend
/// supports every tensor type, and it says nothing about VRAM that another
/// process takes afterwards.
///
/// NVFP4 models are not assessed at all: see [`PreflightError::Nvfp4NotAssessed`].
///
/// `context_length` overrides the model's default context length for
/// KV-cache estimation.  Pass `0` to use the model's built-in default.
///
/// # Errors
///
/// Returns an error if the GGUF file cannot be read or NVML queries fail
/// after successful initialisation. A model with a tensor of an unknown ggml
/// type or an invalid shape is an error too ([`gguf::GgufTensorError`]), not a
/// zero-size estimate, and an NVFP4 model is [`PreflightError::Nvfp4NotAssessed`].
pub fn check_readiness(model_path: &str, context_length: u64) -> Result<PreflightResult> {
    readiness(Path::new(model_path), context_length, vram_audit::vram_snapshot_all)
}

/// Run a preflight check without a model file -- just report GPU VRAM state.
///
/// Useful for answering "what is consuming my VRAM?" without targeting a
/// specific model.  Pass `required_mb` as the VRAM you need; `0` for
/// inventory-only mode (always returns [`Verdict::Go`]).
///
/// # Errors
///
/// Returns an error if NVML queries fail after successful initialisation.
pub fn check_vram_state(required_mb: u64) -> Result<PreflightResult> {
    let gpus = vram_audit::vram_snapshot_all()?;

    if required_mb == 0 {
        return Ok(PreflightResult {
            verdict: Verdict::Go,
            reason: "inventory-only mode".to_owned(),
            model_estimate_mb: 0,
            best_gpu_index: gpus.first().map(|g| g.index),
            gpus,
        });
    }

    let mut result = compute_verdict(&gpus, required_mb);
    result.model_estimate_mb = required_mb;
    result.gpus = gpus;

    Ok(result)
}

// ── Private helpers ────────────────────────────────────────────────────────

/// [`check_readiness`] with the GPU snapshot supplied by `snapshot`, so a test can give it any VRAM state.
///
/// The model is read and sized first: a model that cannot be assessed fails
/// before NVML is queried, as it always has.
fn readiness(
    path: &Path,
    context_length: u64,
    snapshot: impl FnOnce() -> Result<Vec<GpuVramSnapshot>>,
) -> Result<PreflightResult> {
    let model_estimate_mb = estimate_model_mb(path, context_length)?;
    let gpus = snapshot()?;

    let mut result = compute_verdict(&gpus, model_estimate_mb);
    result.model_estimate_mb = model_estimate_mb;
    result.gpus = gpus;

    Ok(result)
}

/// Estimate the megabytes of VRAM that loading `path` at `context_length` needs.
///
/// Reads the GGUF header and tensor infos (never tensor data) and sums the exact
/// tensor sizes, so a quantization the reader knows (NVFP4 included) is sized by
/// its real block layout. A model with a tensor of a ggml type the reader does
/// not know is an error, not a size of zero: [`check_readiness`] then reports a
/// `fail` verdict through its caller and never a `go`.
///
/// An NVFP4 model, by file type or by any tensor, is not sized at all
/// ([`PreflightError::Nvfp4NotAssessed`]) until offset, alignment and
/// file-extent validation lands.
///
/// `context_length` of `0` means the model's own default.
fn estimate_model_mb(path: &Path, context_length: u64) -> Result<u64> {
    let layout = gguf::read_gguf_layout(path)?;
    if layout.nvfp4_tensors > 0 || matches!(layout.meta.file_type, gguf::GgufFileType::Nvfp4) {
        return Err(PreflightError::Nvfp4NotAssessed.into());
    }
    let ctx = if context_length > 0 {
        context_length
    } else {
        layout.meta.context_length
    };
    Ok(gguf::estimate_layout_memory_mb(&layout, ctx))
}

/// Compute a readiness verdict from GPU snapshots and required VRAM.
///
/// Rules:
/// - No GPUs detected -> [`Verdict::Fail`].
/// - Find the GPU with the most free VRAM.
/// - If `free >= needed` AND headroom >= 20% -> [`Verdict::Go`].
/// - If `free >= needed` AND headroom <  20% -> [`Verdict::Warn`].
/// - If `free <  needed` -> [`Verdict::Fail`] (includes top 3 consumers).
fn compute_verdict(gpus: &[GpuVramSnapshot], model_estimate_mb: u64) -> PreflightResult {
    if gpus.is_empty() {
        return PreflightResult {
            verdict: Verdict::Fail,
            reason: "No NVIDIA GPUs detected".to_owned(),
            model_estimate_mb,
            best_gpu_index: None,
            gpus: Vec::new(),
        };
    }

    // Find GPU with the most free VRAM.
    #[expect(clippy::expect_used, reason = "gpus is checked non-empty immediately above")]
    let best = gpus.iter().max_by_key(|g| g.free_mb).expect("gpus is non-empty");

    if best.free_mb >= model_estimate_mb {
        // Headroom as a fraction of free VRAM.
        #[expect(
            clippy::cast_precision_loss,
            reason = "VRAM values are small enough that f64 is exact"
        )]
        let headroom_pct = ((best.free_mb - model_estimate_mb) as f64 / best.free_mb as f64) * 100.0;

        if headroom_pct >= 20.0 {
            PreflightResult {
                verdict: Verdict::Go,
                reason: format!(
                    "{model_estimate_mb}MB needed, GPU{} ({}) has {}MB free ({headroom_pct:.0}% headroom)",
                    best.index, best.name, best.free_mb,
                ),
                model_estimate_mb,
                best_gpu_index: Some(best.index),
                gpus: Vec::new(),
            }
        } else {
            PreflightResult {
                verdict: Verdict::Warn,
                reason: format!(
                    "{model_estimate_mb}MB needed, GPU{} ({}) has {}MB free \
                     — only {headroom_pct:.0}% headroom, may OOM under load",
                    best.index, best.name, best.free_mb,
                ),
                model_estimate_mb,
                best_gpu_index: Some(best.index),
                gpus: Vec::new(),
            }
        }
    } else {
        // Build a concise explanation of what is consuming VRAM on the best GPU.
        let top_consumers: Vec<String> = best
            .processes
            .iter()
            .take(3)
            .map(|p| format!("{}({}MB)", p.name, p.used_mb))
            .collect();

        let consumers_str = if top_consumers.is_empty() {
            String::new()
        } else {
            format!(" — top consumers: {}", top_consumers.join(", "))
        };

        PreflightResult {
            verdict: Verdict::Fail,
            reason: format!(
                "needs {model_estimate_mb}MB, best GPU{} ({}) has only {}MB free{consumers_str}",
                best.index, best.name, best.free_mb,
            ),
            model_estimate_mb,
            best_gpu_index: None,
            gpus: Vec::new(),
        }
    }
}

// ── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A GGUF v3 header with `general.file_type` set to `file_type` and one tensor of `type_id` and `dims`.
    fn one_tensor_gguf(file_type: u32, type_id: u32, dims: &[u64]) -> Vec<u8> {
        fn put_str(b: &mut Vec<u8>, s: &str) {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        }
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3_u32.to_le_bytes());
        b.extend_from_slice(&1_u64.to_le_bytes()); // tensors
        b.extend_from_slice(&1_u64.to_le_bytes()); // metadata entries
        put_str(&mut b, "general.file_type");
        b.extend_from_slice(&4_u32.to_le_bytes()); // u32
        b.extend_from_slice(&file_type.to_le_bytes());
        put_str(&mut b, "w");
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&type_id.to_le_bytes());
        b.extend_from_slice(&0_u64.to_le_bytes());
        b
    }

    fn write_temp(bytes: &[u8]) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(bytes).expect("write fixture");
        f
    }

    /// Why: the readiness estimate must come from the tensor sizes. One Q8_0 tensor of
    /// `[4096, 40960]` is 128 blocks of 34 bytes per row, 40960 rows: 170 MiB. The default
    /// context of 2048 on the default 32 x 4096 dimensions adds a 1024 MiB KV cache, so
    /// `(170 + 1024) * 1.1` is 1313 MB with integer division. Detects the estimate falling
    /// back to the dimension-based guess.
    #[test]
    fn estimate_uses_the_exact_tensor_size() {
        let f = write_temp(&one_tensor_gguf(7, 8, &[4096, 40_960]));
        assert_eq!(estimate_model_mb(f.path(), 0).expect("estimate"), 1313);
    }

    /// Why: a model whose tensor type this build does not know must not be estimated at all,
    /// least of all as 0 MB (which would be a `go`). The FFI turns the error into a `fail`.
    #[test]
    fn estimate_fails_for_an_unknown_tensor_type() {
        let f = write_temp(&one_tensor_gguf(7, 9_999, &[4096, 4096]));
        let err = estimate_model_mb(f.path(), 0).expect_err("unknown tensor type");
        assert!(
            matches!(
                err.downcast_ref::<gguf::GgufTensorError>(),
                Some(gguf::GgufTensorError::UnknownType { index: 0, id: 9_999 })
            ),
            "got: {err:#}"
        );
    }

    fn not_assessed(err: &anyhow::Error) -> bool {
        err.downcast_ref::<PreflightError>() == Some(&PreflightError::Nvfp4NotAssessed)
    }

    /// Why: NVFP4 readiness is deferred until offsets, alignment and extent are validated, so
    /// no NVFP4 signal may reach an estimate: the file type alone, one NVFP4 tensor alone, or both.
    /// Detects: removing either half of the gate in `estimate_model_mb`.
    #[test]
    fn nvfp4_by_file_type_or_by_tensor_is_not_assessed() {
        // file_type 39 = NVFP4 with an F32 tensor; file_type Q5_0 with an NVFP4 tensor; both.
        for (file_type, type_id) in [(39, 0), (7, 40), (39, 40)] {
            let f = write_temp(&one_tensor_gguf(file_type, type_id, &[64]));
            let err = estimate_model_mb(f.path(), 0).expect_err("NVFP4 is not assessed");
            assert!(not_assessed(&err), "file_type {file_type}, tensor {type_id}: {err:#}");
        }
    }

    /// A GPU with a terabyte free, far more than any model needs.
    fn ample_gpus() -> Result<Vec<GpuVramSnapshot>> {
        Ok(vec![GpuVramSnapshot {
            index: 0,
            name: "Ample GPU".to_owned(),
            total_mb: 1_048_576,
            used_mb: 0,
            free_mb: 1_048_576,
            processes: vec![],
        }])
    }

    /// Why: the point of deferring NVFP4: with ample VRAM an NVFP4 header must not yield `go`.
    /// The positive control proves the same GPU does say `go` for a model that is assessed, so
    /// the error is the gate and not a broken fixture.
    #[test]
    fn nvfp4_header_with_ample_vram_is_not_go() {
        let nvfp4 = write_temp(&one_tensor_gguf(39, 40, &[4096, 4096]));
        let err = readiness(nvfp4.path(), 0, ample_gpus).expect_err("NVFP4 must not be assessed");
        assert!(not_assessed(&err), "got: {err:#}");

        let q8 = write_temp(&one_tensor_gguf(7, 8, &[4096, 4096]));
        let ok = readiness(q8.path(), 0, ample_gpus).expect("a Q8_0 model is assessed");
        assert_eq!(ok.verdict, Verdict::Go);
    }

    /// Why: an unassessable model fails before NVML is asked anything.
    #[test]
    fn an_unassessable_model_does_not_query_the_gpus() {
        let nvfp4 = write_temp(&one_tensor_gguf(39, 40, &[64]));
        let err =
            readiness(nvfp4.path(), 0, || panic!("the GPU snapshot must not be taken")).expect_err("not assessed");
        assert!(not_assessed(&err));
    }

    #[test]
    fn verdict_go_when_enough_vram() {
        let gpus = vec![GpuVramSnapshot {
            index: 0,
            name: "RTX 5060 Ti".to_owned(),
            total_mb: 16384,
            used_mb: 2000,
            free_mb: 14384,
            processes: vec![],
        }];
        let model_mb = 5000;

        let verdict = compute_verdict(&gpus, model_mb);
        assert_eq!(verdict.verdict, Verdict::Go);
        assert_eq!(verdict.best_gpu_index, Some(0));
    }

    #[test]
    fn verdict_fail_when_no_gpu_fits() {
        let gpus = vec![GpuVramSnapshot {
            index: 0,
            name: "RTX 2000 Ada".to_owned(),
            total_mb: 8192,
            used_mb: 7500,
            free_mb: 692,
            processes: vec![VramProcess {
                pid: 1234,
                name: "ollama.exe".to_owned(),
                used_mb: 6000,
            }],
        }];
        let model_mb = 5000;

        let verdict = compute_verdict(&gpus, model_mb);
        assert_eq!(verdict.verdict, Verdict::Fail);
        assert!(verdict.reason.contains("692"));
    }

    #[test]
    fn verdict_warn_when_tight_fit() {
        let gpus = vec![GpuVramSnapshot {
            index: 0,
            name: "Test GPU".to_owned(),
            total_mb: 8192,
            used_mb: 2500,
            free_mb: 5692,
            processes: vec![],
        }];
        // Model needs 5000, free is 5692 -- only ~12% headroom (<20% threshold).
        let model_mb = 5000;

        let verdict = compute_verdict(&gpus, model_mb);
        assert_eq!(verdict.verdict, Verdict::Warn);
    }

    #[test]
    fn verdict_fail_when_no_gpus() {
        let verdict = compute_verdict(&[], 5000);
        assert_eq!(verdict.verdict, Verdict::Fail);
        assert!(verdict.reason.contains("No NVIDIA GPU"));
    }

    #[test]
    fn verdict_picks_best_gpu() {
        let gpus = vec![
            GpuVramSnapshot {
                index: 0,
                name: "RTX 2000 Ada".to_owned(),
                total_mb: 8192,
                used_mb: 7000,
                free_mb: 1192,
                processes: vec![],
            },
            GpuVramSnapshot {
                index: 1,
                name: "RTX 5060 Ti".to_owned(),
                total_mb: 16384,
                used_mb: 1500,
                free_mb: 14884,
                processes: vec![],
            },
        ];

        let verdict = compute_verdict(&gpus, 5000);
        assert_eq!(verdict.verdict, Verdict::Go);
        assert_eq!(verdict.best_gpu_index, Some(1));
    }

    #[test]
    fn preflight_result_serializes_compact() {
        let result = PreflightResult {
            verdict: Verdict::Fail,
            reason: "needs 5000MB, GPU0 has 692MB free".to_owned(),
            model_estimate_mb: 5000,
            best_gpu_index: None,
            gpus: vec![],
        };

        let json = serde_json::to_string(&result).expect("should serialize");
        assert!(
            json.len() < 200,
            "preflight JSON should be compact, got {} bytes",
            json.len()
        );
        assert!(json.contains("\"verdict\":\"fail\""));
    }
}
