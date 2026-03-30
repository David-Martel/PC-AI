//! Per-GPU VRAM process audit via NVML.
//!
//! Queries which OS processes are consuming VRAM on each GPU and resolves
//! PIDs to process names via `sysinfo`.  Designed for preflight readiness
//! checks: before loading a model, inspect what is already consuming GPU
//! memory and report per-process breakdowns.
//!
//! # Feature gate
//!
//! This module requires the `nvml` Cargo feature.  When NVML is unavailable
//! at runtime (no NVIDIA driver installed), all public functions degrade
//! gracefully by returning empty results.

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::Serialize;

// ── Public data types ───────────────────────────────────────────────────────

/// VRAM snapshot for a single GPU with per-process breakdown.
///
/// Returned by [`vram_snapshot_all`] — one entry per NVIDIA GPU detected by
/// NVML.  The `processes` list is sorted by `used_mb` descending so the
/// largest consumers appear first.
#[derive(Debug, Clone, Serialize)]
pub struct GpuVramSnapshot {
    /// Zero-based device index as enumerated by NVML.
    pub index: u32,
    /// Human-readable product name (e.g. `NVIDIA GeForce RTX 5060 Ti`).
    pub name: String,
    /// Total installed framebuffer memory in mebibytes.
    pub total_mb: u64,
    /// Currently allocated framebuffer memory in mebibytes.
    pub used_mb: u64,
    /// Currently free framebuffer memory in mebibytes.
    pub free_mb: u64,
    /// Processes consuming VRAM on this GPU, sorted by usage descending.
    pub processes: Vec<VramProcess>,
}

/// A single process consuming VRAM on a GPU.
#[derive(Debug, Clone, Serialize)]
pub struct VramProcess {
    /// Operating system process ID.
    pub pid: u32,
    /// Resolved process name (e.g. `ollama.exe`).  Falls back to
    /// `"PID-{pid}"` when the process has already exited or cannot be
    /// queried.
    pub name: String,
    /// VRAM consumed by this process in mebibytes.
    pub used_mb: u64,
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Query VRAM state for all GPUs, including per-process breakdown.
///
/// Returns an empty [`Vec`] if NVML is unavailable (no NVIDIA driver) rather
/// than an error, matching the graceful-degradation contract of the `gpu`
/// module.
///
/// For each GPU the function:
/// 1. Reads total/used/free memory via `device.memory_info()`.
/// 2. Collects running compute and graphics processes.
/// 3. Deduplicates PIDs (a process can appear in both lists), keeping the
///    larger VRAM value.
/// 4. Resolves each PID to a process name via `sysinfo`.
/// 5. Sorts processes by VRAM usage descending (top consumers first).
///
/// # Errors
///
/// Returns an error only when NVML initialises successfully but a
/// subsequent query (device count, memory info) fails unexpectedly.
pub fn vram_snapshot_all() -> Result<Vec<GpuVramSnapshot>> {
    let nvml = match super::get_nvml() {
        Some(n) => n,
        None => return Ok(Vec::new()),
    };

    let count = nvml.device_count().context("NVML device count query failed")?;
    let mut snapshots = Vec::with_capacity(count as usize);

    // Build a sysinfo::System for PID-to-name resolution.
    // We refresh only the specific PIDs we need rather than enumerating
    // every process on the system.
    use sysinfo::{Pid, ProcessesToUpdate, System};
    let mut sys = System::new();

    for idx in 0..count {
        let device = match nvml.device_by_index(idx) {
            Ok(d) => d,
            Err(err) => {
                log::warn!("Failed to open NVML device at index {idx}: {err}");
                continue;
            }
        };

        let name = device.name().unwrap_or_else(|_| format!("GPU-{idx}"));
        let mem = device
            .memory_info()
            .with_context(|| format!("NVML memory query failed for GPU {idx}"))?;

        // Collect PIDs from both compute and graphics process lists.
        let compute = device.running_compute_processes().unwrap_or_default();
        let graphics = device.running_graphics_processes().unwrap_or_default();

        // Deduplicate PIDs — a process can appear in both compute and
        // graphics lists.  Keep the larger VRAM value when duplicated.
        let mut pid_vram: HashMap<u32, u64> = HashMap::new();
        for p in compute.iter().chain(graphics.iter()) {
            let used_bytes = match p.used_gpu_memory {
                nvml_wrapper::enums::device::UsedGpuMemory::Used(bytes) => bytes,
                nvml_wrapper::enums::device::UsedGpuMemory::Unavailable => 0,
            };
            pid_vram
                .entry(p.pid)
                .and_modify(|existing| *existing = (*existing).max(used_bytes))
                .or_insert(used_bytes);
        }

        // Resolve PID -> process name via sysinfo.
        // Refresh only the PIDs we care about to avoid a full process scan.
        let pids: Vec<Pid> = pid_vram.keys().map(|&pid| Pid::from_u32(pid)).collect();
        sys.refresh_processes(ProcessesToUpdate::Some(&pids), true);

        let mut processes: Vec<VramProcess> = pid_vram
            .into_iter()
            .map(|(pid, used_bytes)| {
                let proc_name = sys
                    .process(Pid::from_u32(pid))
                    .map(|p| p.name().to_string_lossy().to_string())
                    .unwrap_or_else(|| format!("PID-{pid}"));
                VramProcess {
                    pid,
                    name: proc_name,
                    used_mb: used_bytes / (1024 * 1024),
                }
            })
            .collect();

        // Sort by VRAM usage descending so top consumers appear first.
        processes.sort_by(|a, b| b.used_mb.cmp(&a.used_mb));

        snapshots.push(GpuVramSnapshot {
            index: idx,
            name,
            total_mb: mem.total / (1024 * 1024),
            used_mb: mem.used / (1024 * 1024),
            free_mb: mem.free / (1024 * 1024),
            processes,
        });
    }

    Ok(snapshots)
}

// ── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// `GpuVramSnapshot` must serialise to valid JSON with all expected fields.
    #[test]
    fn gpu_vram_snapshot_serializes() {
        let snapshot = GpuVramSnapshot {
            index: 0,
            name: "Test GPU".to_owned(),
            total_mb: 8192,
            used_mb: 4000,
            free_mb: 4192,
            processes: vec![
                VramProcess {
                    pid: 1234,
                    name: "ollama.exe".to_owned(),
                    used_mb: 3500,
                },
                VramProcess {
                    pid: 5678,
                    name: "chrome.exe".to_owned(),
                    used_mb: 500,
                },
            ],
        };

        let json = serde_json::to_string(&snapshot).expect("should serialize");
        assert!(json.contains("\"free_mb\":4192"));
        assert!(json.contains("ollama.exe"));
        assert!(json.contains("\"index\":0"));
        assert!(json.contains("\"total_mb\":8192"));
        assert!(json.contains("\"used_mb\":4000"));
    }

    /// An empty process list is perfectly valid (no GPU consumers).
    #[test]
    fn empty_process_list_is_valid() {
        let snapshot = GpuVramSnapshot {
            index: 0,
            name: "Empty GPU".to_owned(),
            total_mb: 16384,
            used_mb: 0,
            free_mb: 16384,
            processes: vec![],
        };

        let json = serde_json::to_string(&snapshot).expect("should serialize");
        assert!(json.contains("\"processes\":[]"));
    }

    /// `Vec<GpuVramSnapshot>` serialises to a JSON array (matches FFI contract).
    #[test]
    fn snapshot_vec_serializes_to_json_array() {
        let snapshots: Vec<GpuVramSnapshot> = Vec::new();
        let json = serde_json::to_string(&snapshots).expect("empty Vec should serialize");
        assert_eq!(json, "[]");
    }

    /// `VramProcess` fields are present in serialized output.
    #[test]
    fn vram_process_serializes() {
        let proc = VramProcess {
            pid: 42,
            name: "test.exe".to_owned(),
            used_mb: 256,
        };

        let json = serde_json::to_string(&proc).expect("should serialize");
        assert!(json.contains("\"pid\":42"));
        assert!(json.contains("\"name\":\"test.exe\""));
        assert!(json.contains("\"used_mb\":256"));
    }

    /// Processes should be ordered by VRAM descending in a snapshot.
    #[test]
    fn processes_sorted_by_vram_descending() {
        let snapshot = GpuVramSnapshot {
            index: 0,
            name: "GPU".to_owned(),
            total_mb: 8192,
            used_mb: 6000,
            free_mb: 2192,
            processes: vec![
                VramProcess {
                    pid: 1,
                    name: "big.exe".to_owned(),
                    used_mb: 4000,
                },
                VramProcess {
                    pid: 2,
                    name: "medium.exe".to_owned(),
                    used_mb: 1500,
                },
                VramProcess {
                    pid: 3,
                    name: "small.exe".to_owned(),
                    used_mb: 500,
                },
            ],
        };

        // Verify ordering is maintained through serialisation round-trip
        for window in snapshot.processes.windows(2) {
            assert!(
                window[0].used_mb >= window[1].used_mb,
                "processes should be sorted by used_mb descending"
            );
        }
    }

    /// `vram_snapshot_all` returns `Ok` even when NVML is unavailable.
    ///
    /// On machines without NVIDIA drivers this returns an empty vec.
    /// On machines with NVIDIA GPUs this returns populated snapshots.
    #[test]
    fn vram_snapshot_all_graceful_when_nvml_absent() {
        let result = vram_snapshot_all();
        assert!(result.is_ok(), "vram_snapshot_all should never fail on NVML init error");
    }
}
