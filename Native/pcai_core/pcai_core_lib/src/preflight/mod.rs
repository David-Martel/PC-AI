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

// ── Public API ─────────────────────────────────────────────────────────────

/// Run a full preflight check for loading a GGUF model.
///
/// 1. Parses the GGUF header to estimate memory requirements.
/// 2. Queries NVML for live VRAM state on all GPUs.
/// 3. Returns a verdict with the best GPU choice (or why it will not fit).
///
/// `context_length` overrides the model's default context length for
/// KV-cache estimation.  Pass `0` to use the model's built-in default.
///
/// # Errors
///
/// Returns an error if the GGUF file cannot be read or NVML queries fail
/// after successful initialisation.
pub fn check_readiness(model_path: &str, context_length: u64) -> Result<PreflightResult> {
    let path = Path::new(model_path);

    let meta = gguf::read_gguf_meta(path)?;
    let ctx = if context_length > 0 {
        context_length
    } else {
        meta.context_length
    };
    let model_estimate_mb = gguf::estimate_model_memory_mb(&meta, ctx);
    let gpus = vram_audit::vram_snapshot_all()?;

    let mut result = compute_verdict(&gpus, model_estimate_mb);
    result.model_estimate_mb = model_estimate_mb;
    result.gpus = gpus;

    Ok(result)
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
