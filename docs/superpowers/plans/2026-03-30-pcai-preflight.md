# pcai_preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Rust-native GPU preflight module to `pcai_core_lib` that diagnoses VRAM availability, estimates model memory requirements from GGUF headers, audits per-process GPU memory, and returns a structured go/warn/fail verdict — exposed via FFI, CLI, and PowerShell.

**Architecture:** New `preflight` module in `pcai_core_lib` alongside the existing `gpu` module. It composes three capabilities: (1) GGUF header parsing for memory estimation, (2) NVML process VRAM audit, (3) readiness verdict logic. Exposed as a single FFI function returning compact JSON, a `pcai-perf preflight` CLI subcommand, and a `Test-PcaiGpuReadiness` PowerShell function. Integration into `pcai_inference` and `pcai_media` backends via preflight checks before model load.

**Tech Stack:** Rust (pcai_core_lib with `nvml` feature), nvml-wrapper 0.12 (process query API), sysinfo (process name resolution), serde_json (compact output), PowerShell 7+ (P/Invoke wrapper)

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `pcai_core_lib/src/preflight/mod.rs` | Public API: `PreflightVerdict`, `check_readiness()` |
| Create | `pcai_core_lib/src/preflight/gguf.rs` | GGUF header parser: read metadata, estimate memory |
| Create | `pcai_core_lib/src/preflight/vram_audit.rs` | NVML process VRAM query, process name resolution |
| Modify | `pcai_core_lib/src/lib.rs:17-35` | Add `pub mod preflight` + FFI export `pcai_gpu_preflight_json` |
| Modify | `pcai_core_lib/Cargo.toml` | No new deps needed (memmap2, sysinfo, nvml-wrapper already present) |
| Modify | `pcai_perf_cli/src/main.rs:104-109` | Add `"preflight"` subcommand |
| Create | `Modules/PC-AI.Gpu/Public/Test-PcaiGpuReadiness.ps1` | PowerShell wrapper calling FFI |
| Modify | `Modules/PC-AI.Gpu/PC-AI.Gpu.psd1` | Export `Test-PcaiGpuReadiness` |
| Create | `pcai_core_lib/tests/preflight_tests.rs` | Integration tests for preflight module |

---

### Task 1: GGUF Header Parser — Types and Memory Estimation

**Files:**
- Create: `Native/pcai_core/pcai_core_lib/src/preflight/gguf.rs`

The GGUF format stores model metadata in a binary header. We need to read just enough to estimate total memory: parameter count, quantization type, and context length. The GGUF v3 header format is:

```
Bytes 0-3:   magic "GGUF" (4 bytes)
Bytes 4-7:   version (u32 LE, expect 3)
Bytes 8-15:  tensor_count (u64 LE)
Bytes 16-23: metadata_kv_count (u64 LE)
Then:        metadata_kv_count key-value pairs
```

Each KV pair has: key_len (u64), key_bytes, value_type (u32), value. We scan for keys: `general.architecture`, `general.file_type`, `*.context_length`, `*.embedding_length`, `*.block_count`.

- [ ] **Step 1: Write the failing test**

Create the test file with tests for GGUF memory estimation logic:

```rust
// Native/pcai_core/pcai_core_lib/src/preflight/gguf.rs
// (tests at bottom of file)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_memory_q4_k_m_7b() {
        let meta = GgufModelMeta {
            architecture: "llama".to_owned(),
            file_type: GgufFileType::Q4KM,
            parameter_count: 7_000_000_000,
            context_length: 8192,
            embedding_length: 4096,
            block_count: 32,
        };
        let est = estimate_model_memory_mb(&meta, 8192);
        // 7B Q4_K_M ~ 4.07 GB model + KV cache
        // Model weights: 7B * 4.5 bits/param / 8 = ~3937 MB
        // KV cache: 2 * 32 * 4096 * 8192 * 2 bytes / 1MB = ~4096 MB at fp16
        // Total should be roughly 4000-8000 MB depending on ctx
        assert!(est > 3500, "7B Q4_K_M should need >3500 MB, got {est}");
        assert!(est < 10000, "7B Q4_K_M at 8K ctx should need <10000 MB, got {est}");
    }

    #[test]
    fn estimate_memory_q8_0_3b() {
        let meta = GgufModelMeta {
            architecture: "llama".to_owned(),
            file_type: GgufFileType::Q8_0,
            parameter_count: 3_000_000_000,
            context_length: 4096,
            embedding_length: 3200,
            block_count: 26,
        };
        let est = estimate_model_memory_mb(&meta, 4096);
        // 3B Q8_0 ~ 3 GB model + smaller KV cache
        assert!(est > 2500, "3B Q8_0 should need >2500 MB, got {est}");
        assert!(est < 7000, "3B Q8_0 at 4K ctx should need <7000 MB, got {est}");
    }

    #[test]
    fn bits_per_param_known_types() {
        assert!((GgufFileType::Q4_0.bits_per_param() - 4.5).abs() < 0.1);
        assert!((GgufFileType::Q4KM.bits_per_param() - 4.5).abs() < 0.2);
        assert!((GgufFileType::Q8_0.bits_per_param() - 8.5).abs() < 0.1);
        assert!((GgufFileType::F16.bits_per_param() - 16.0).abs() < 0.1);
    }

    #[test]
    fn unknown_file_type_defaults_conservatively() {
        assert!(GgufFileType::Unknown(999).bits_per_param() >= 8.0);
    }
}
```

- [ ] **Step 2: Implement GGUF types and memory estimation**

```rust
// Native/pcai_core/pcai_core_lib/src/preflight/gguf.rs

//! GGUF header parser for model memory estimation.
//!
//! Reads only the metadata section of a GGUF file (first few KB) to extract
//! the fields needed for VRAM estimation. Does NOT load tensors.

use std::io::{self, Read};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Serialize;

/// Metadata extracted from a GGUF file header.
#[derive(Debug, Clone, Serialize)]
pub struct GgufModelMeta {
    pub architecture: String,
    pub file_type: GgufFileType,
    pub parameter_count: u64,
    pub context_length: u64,
    pub embedding_length: u64,
    pub block_count: u64,
}

/// GGUF quantization file types.
///
/// Integer values match the GGUF `general.file_type` metadata field.
#[derive(Debug, Clone, Copy, Serialize)]
pub enum GgufFileType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3KS,
    Q3KM,
    Q3KL,
    Q4KS,
    Q4KM,
    Q5KS,
    Q5KM,
    Q6K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    Unknown(u32),
}

impl GgufFileType {
    /// Approximate bits per parameter for VRAM estimation.
    ///
    /// These are estimates — actual sizes vary by tensor shape and quantization
    /// group size. We err on the high side so the preflight check is conservative.
    pub fn bits_per_param(self) -> f64 {
        match self {
            Self::F32 => 32.0,
            Self::F16 => 16.0,
            Self::Q8_0 => 8.5,
            Self::Q6K => 6.5,
            Self::Q5KM | Self::Q5KS => 5.5,
            Self::Q5_0 | Self::Q5_1 => 5.5,
            Self::Q4KM | Self::Q4KS => 4.5,
            Self::Q4_0 | Self::Q4_1 => 4.5,
            Self::Q3KL | Self::Q3KM | Self::Q3KS => 3.5,
            Self::Q2K => 2.5,
            Self::IQ2XXS | Self::IQ2XS => 2.5,
            Self::IQ3XXS => 3.5,
            Self::Unknown(_) => 8.0, // conservative default
        }
    }

    fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            7 => Self::Q5_0,
            8 => Self::Q5_1,
            9 => Self::Q8_0,
            10 => Self::Q2K,
            11 => Self::Q3KS,
            12 => Self::Q3KM,
            13 => Self::Q3KL,
            14 => Self::Q4KS,
            15 => Self::Q4KM,
            16 => Self::Q5KS,
            17 => Self::Q5KM,
            18 => Self::Q6K,
            19 => Self::IQ2XXS,
            20 => Self::IQ2XS,
            21 => Self::IQ3XXS,
            other => Self::Unknown(other),
        }
    }
}

/// Estimate total VRAM needed to load and run a model at the given context size.
///
/// Returns estimated megabytes. Components:
/// - Model weights: `parameter_count * bits_per_param / 8`
/// - KV cache: `2 * block_count * embedding_length * context_length * 2 bytes` (fp16)
/// - Overhead: 10% buffer for CUDA kernels, scratch space, allocator fragmentation
pub fn estimate_model_memory_mb(meta: &GgufModelMeta, context_length: u64) -> u64 {
    let weight_bytes = (meta.parameter_count as f64 * meta.file_type.bits_per_param()) / 8.0;
    let weight_mb = (weight_bytes / (1024.0 * 1024.0)) as u64;

    // KV cache at fp16: 2 (K+V) * layers * hidden_dim * ctx_len * 2 bytes
    let kv_bytes = 2u64
        .saturating_mul(meta.block_count)
        .saturating_mul(meta.embedding_length)
        .saturating_mul(context_length)
        .saturating_mul(2); // fp16
    let kv_mb = kv_bytes / (1024 * 1024);

    // 10% overhead for CUDA scratch, allocator fragmentation
    let subtotal = weight_mb + kv_mb;
    subtotal + (subtotal / 10)
}

// ── GGUF binary parser ───────────────────────────────────────────────────────

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// GGUF metadata value types (subset we need).
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

/// Read GGUF header metadata from a file path.
///
/// Only reads the metadata section (typically <64KB). Does not mmap
/// or read tensor data.
pub fn read_gguf_meta(path: &Path) -> Result<GgufModelMeta> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open GGUF file: {}", path.display()))?;
    let mut reader = io::BufReader::new(file);

    // Magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).context("Failed to read GGUF magic")?;
    if &magic != GGUF_MAGIC {
        bail!("Not a GGUF file (magic: {:?})", magic);
    }

    // Version
    let version = read_u32_le(&mut reader)?;
    if version < 2 || version > 3 {
        bail!("Unsupported GGUF version: {version} (expected 2 or 3)");
    }

    // Tensor count (used to estimate parameter count)
    let tensor_count = read_u64_le(&mut reader)?;

    // Metadata KV count
    let kv_count = read_u64_le(&mut reader)?;

    // Parse metadata key-value pairs
    let mut architecture = String::new();
    let mut file_type_raw: Option<u32> = None;
    let mut context_length: Option<u64> = None;
    let mut embedding_length: Option<u64> = None;
    let mut block_count: Option<u64> = None;

    for _ in 0..kv_count {
        let key = read_gguf_string(&mut reader)?;
        let vtype_raw = read_u32_le(&mut reader)?;
        let vtype = GgufValueType::from_u32(vtype_raw);

        match key.as_str() {
            "general.architecture" => {
                architecture = read_gguf_typed_string(&mut reader, vtype)?;
            }
            "general.file_type" => {
                file_type_raw = Some(read_gguf_typed_u32(&mut reader, vtype)?);
            }
            k if k.ends_with(".context_length") => {
                context_length = Some(read_gguf_typed_u64(&mut reader, vtype)?);
            }
            k if k.ends_with(".embedding_length") => {
                embedding_length = Some(read_gguf_typed_u64(&mut reader, vtype)?);
            }
            k if k.ends_with(".block_count") => {
                block_count = Some(read_gguf_typed_u64(&mut reader, vtype)?);
            }
            _ => {
                skip_gguf_value(&mut reader, vtype_raw)?;
            }
        }
    }

    // Estimate parameter count from tensor_count + embedding + block_count
    // A rough formula: ~(embedding^2 * block_count * 4) for transformer models
    let emb = embedding_length.unwrap_or(4096);
    let blocks = block_count.unwrap_or(32);
    // Each transformer block has ~4 * emb^2 parameters (QKV proj + FFN)
    // Plus embedding table: vocab_size * emb (estimate vocab ~32K)
    let estimated_params = (4 * emb * emb * blocks) + (32_000 * emb);

    Ok(GgufModelMeta {
        architecture,
        file_type: GgufFileType::from_u32(file_type_raw.unwrap_or(15)),
        parameter_count: estimated_params,
        context_length: context_length.unwrap_or(2048),
        embedding_length: emb,
        block_count: blocks,
    })
}

// ── Binary reader helpers ────────────────────────────────────────────────────

fn read_u32_le(r: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le(r: &mut impl Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i32_le(r: &mut impl Read) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_gguf_string(r: &mut impl Read) -> Result<String> {
    let len = read_u64_le(r)? as usize;
    if len > 1_000_000 {
        bail!("GGUF string length too large: {len}");
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).context("Invalid UTF-8 in GGUF string")
}

fn read_gguf_typed_string(r: &mut impl Read, vtype: Option<GgufValueType>) -> Result<String> {
    match vtype {
        Some(GgufValueType::String) => read_gguf_string(r),
        _ => {
            skip_gguf_value_by_type(r, vtype)?;
            Ok(String::new())
        }
    }
}

fn read_gguf_typed_u32(r: &mut impl Read, vtype: Option<GgufValueType>) -> Result<u32> {
    match vtype {
        Some(GgufValueType::Uint32) => read_u32_le(r),
        Some(GgufValueType::Int32) => Ok(read_i32_le(r)? as u32),
        Some(GgufValueType::Uint16) => {
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)?;
            Ok(u16::from_le_bytes(buf) as u32)
        }
        _ => {
            skip_gguf_value_by_type(r, vtype)?;
            Ok(0)
        }
    }
}

fn read_gguf_typed_u64(r: &mut impl Read, vtype: Option<GgufValueType>) -> Result<u64> {
    match vtype {
        Some(GgufValueType::Uint64) => read_u64_le(r),
        Some(GgufValueType::Int64) => {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
            Ok(i64::from_le_bytes(buf) as u64)
        }
        Some(GgufValueType::Uint32) => Ok(u64::from(read_u32_le(r)?)),
        Some(GgufValueType::Int32) => Ok(read_i32_le(r)? as u64),
        _ => {
            skip_gguf_value_by_type(r, vtype)?;
            Ok(0)
        }
    }
}

fn skip_gguf_value(r: &mut impl Read, vtype_raw: u32) -> Result<()> {
    skip_gguf_value_by_type(r, GgufValueType::from_u32(vtype_raw))
}

fn skip_gguf_value_by_type(r: &mut impl Read, vtype: Option<GgufValueType>) -> Result<()> {
    match vtype {
        Some(GgufValueType::Uint8 | GgufValueType::Int8 | GgufValueType::Bool) => {
            let mut buf = [0u8; 1];
            r.read_exact(&mut buf)?;
        }
        Some(GgufValueType::Uint16 | GgufValueType::Int16) => {
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)?;
        }
        Some(GgufValueType::Uint32 | GgufValueType::Int32 | GgufValueType::Float32) => {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)?;
        }
        Some(GgufValueType::Uint64 | GgufValueType::Int64 | GgufValueType::Float64) => {
            let mut buf = [0u8; 8];
            r.read_exact(&mut buf)?;
        }
        Some(GgufValueType::String) => {
            let _ = read_gguf_string(r)?;
        }
        Some(GgufValueType::Array) => {
            let elem_type_raw = read_u32_le(r)?;
            let count = read_u64_le(r)?;
            for _ in 0..count {
                skip_gguf_value(r, elem_type_raw)?;
            }
        }
        None => bail!("Unknown GGUF value type"),
    }
    Ok(())
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd Native/pcai_core && cargo test -p pcai_core_lib --lib preflight::gguf::tests -- --no-default-features`
Expected: compilation error — module `preflight` does not exist yet (wired in Task 3)

- [ ] **Step 4: Commit GGUF parser**

```bash
git add Native/pcai_core/pcai_core_lib/src/preflight/gguf.rs
git commit -m "feat(preflight): add GGUF header parser with memory estimation"
```

---

### Task 2: VRAM Process Audit via NVML

**Files:**
- Create: `Native/pcai_core/pcai_core_lib/src/preflight/vram_audit.rs`

Uses nvml-wrapper's `device.running_compute_processes()` and `device.running_graphics_processes()` to identify what is consuming VRAM on each GPU. Resolves PIDs to process names via `sysinfo`.

- [ ] **Step 1: Write the failing test**

```rust
// At bottom of vram_audit.rs

#[cfg(test)]
mod tests {
    use super::*;

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
    }

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
}
```

- [ ] **Step 2: Implement VRAM audit module**

```rust
// Native/pcai_core/pcai_core_lib/src/preflight/vram_audit.rs

//! Per-GPU VRAM process audit via NVML.
//!
//! Queries which OS processes are consuming VRAM on each GPU and resolves
//! PIDs to process names via sysinfo.

use anyhow::{Context, Result};
use serde::Serialize;

/// VRAM snapshot for a single GPU with per-process breakdown.
#[derive(Debug, Clone, Serialize)]
pub struct GpuVramSnapshot {
    pub index: u32,
    pub name: String,
    pub total_mb: u64,
    pub used_mb: u64,
    pub free_mb: u64,
    pub processes: Vec<VramProcess>,
}

/// A single process consuming VRAM.
#[derive(Debug, Clone, Serialize)]
pub struct VramProcess {
    pub pid: u32,
    pub name: String,
    pub used_mb: u64,
}

/// Query VRAM state for all GPUs, including per-process breakdown.
///
/// Returns an empty vec if NVML is unavailable. Each GPU entry includes
/// the list of compute and graphics processes and their VRAM usage.
pub fn vram_snapshot_all() -> Result<Vec<GpuVramSnapshot>> {
    let nvml = match super::get_nvml() {
        Some(n) => n,
        None => return Ok(Vec::new()),
    };

    let count = nvml.device_count().context("NVML device count query failed")?;
    let mut snapshots = Vec::with_capacity(count as usize);

    // Build a sysinfo::System for PID-to-name resolution
    use sysinfo::{ProcessesToUpdate, System};
    let mut sys = System::new();
    // We'll refresh specific PIDs after collecting them

    for idx in 0..count {
        let device = match nvml.device_by_index(idx) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let name = device.name().unwrap_or_else(|_| format!("GPU-{idx}"));
        let mem = device.memory_info().context("NVML memory query failed")?;

        // Collect PIDs from both compute and graphics processes
        let compute = device.running_compute_processes().unwrap_or_default();
        let graphics = device.running_graphics_processes().unwrap_or_default();

        // Deduplicate PIDs (a process can appear in both lists)
        let mut pid_vram: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
        for p in compute.iter().chain(graphics.iter()) {
            let used_bytes = match p.used_gpu_memory {
                nvml_wrapper::struct_wrappers::device::UsedGpuMemory::Used(bytes) => bytes,
                nvml_wrapper::struct_wrappers::device::UsedGpuMemory::Unavailable => 0,
            };
            pid_vram
                .entry(p.pid)
                .and_modify(|existing| *existing = (*existing).max(used_bytes))
                .or_insert(used_bytes);
        }

        // Resolve PID -> process name
        let pids: Vec<sysinfo::Pid> = pid_vram.keys().map(|&pid| sysinfo::Pid::from_u32(pid)).collect();
        sys.refresh_processes(ProcessesToUpdate::Some(&pids), true);

        let mut processes: Vec<VramProcess> = pid_vram
            .into_iter()
            .map(|(pid, used_bytes)| {
                let name = sys
                    .process(sysinfo::Pid::from_u32(pid))
                    .map(|p| p.name().to_string_lossy().to_string())
                    .unwrap_or_else(|| format!("PID-{pid}"));
                VramProcess {
                    pid,
                    name,
                    used_mb: used_bytes / (1024 * 1024),
                }
            })
            .collect();

        // Sort by VRAM usage descending so top consumers are first
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd Native/pcai_core && cargo test -p pcai_core_lib --lib preflight::vram_audit::tests -- --no-default-features`
Expected: compilation error — module not wired yet

- [ ] **Step 4: Commit VRAM audit**

```bash
git add Native/pcai_core/pcai_core_lib/src/preflight/vram_audit.rs
git commit -m "feat(preflight): add NVML process VRAM audit"
```

---

### Task 3: Preflight Verdict Logic and Module Root

**Files:**
- Create: `Native/pcai_core/pcai_core_lib/src/preflight/mod.rs`
- Modify: `Native/pcai_core/pcai_core_lib/src/lib.rs` (add `pub mod preflight`)

Composes GGUF estimation + VRAM audit into a single readiness check with `go`/`warn`/`fail` verdict.

- [ ] **Step 1: Write the failing test**

```rust
// At bottom of preflight/mod.rs

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
        // Model needs 5000, free is 5692 — only 12% headroom (<20% threshold)
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
        assert!(json.len() < 200, "preflight JSON should be compact, got {} bytes", json.len());
        assert!(json.contains("\"verdict\":\"fail\""));
    }
}
```

- [ ] **Step 2: Implement preflight module root**

```rust
// Native/pcai_core/pcai_core_lib/src/preflight/mod.rs

//! GPU preflight readiness check.
//!
//! Combines GGUF model memory estimation, NVML VRAM audit, and verdict
//! logic into a single `check_readiness()` call that returns a compact
//! structured result.
//!
//! # Examples
//!
//! ```no_run
//! # #[cfg(feature = "nvml")]
//! # {
//! use pcai_core_lib::preflight;
//!
//! let result = preflight::check_readiness("path/to/model.gguf", 8192)?;
//! println!("{}", serde_json::to_string(&result)?);
//! // {"verdict":"go","reason":"5800MB needed, GPU1 has 14884MB free",...}
//! # Ok::<(), anyhow::Error>(())
//! # }
//! ```

pub mod gguf;
pub mod vram_audit;

use std::path::Path;

use anyhow::Result;
use serde::Serialize;

pub use vram_audit::{GpuVramSnapshot, VramProcess};

// Re-export the NVML singleton getter for use by vram_audit
pub(crate) use crate::gpu::get_nvml;

/// Readiness verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    /// Model fits comfortably (>20% headroom on best GPU).
    Go,
    /// Model fits but tight (<20% headroom) — may OOM under load.
    Warn,
    /// Model does not fit on any available GPU.
    Fail,
}

/// Compact preflight result.
#[derive(Debug, Clone, Serialize)]
pub struct PreflightResult {
    pub verdict: Verdict,
    pub reason: String,
    pub model_estimate_mb: u64,
    pub best_gpu_index: Option<u32>,
    pub gpus: Vec<GpuVramSnapshot>,
}

/// Run a full preflight check for loading a GGUF model.
///
/// 1. Parses the GGUF header to estimate memory requirements
/// 2. Queries NVML for live VRAM state on all GPUs
/// 3. Returns a verdict with the best GPU choice (or why it won't fit)
///
/// `context_length` overrides the model's default context length for
/// KV cache estimation. Pass 0 to use the model's built-in default.
pub fn check_readiness(model_path: &str, context_length: u64) -> Result<PreflightResult> {
    let path = Path::new(model_path);

    // Step 1: Parse GGUF header
    let meta = gguf::read_gguf_meta(path)?;
    let ctx = if context_length > 0 {
        context_length
    } else {
        meta.context_length
    };
    let model_estimate_mb = gguf::estimate_model_memory_mb(&meta, ctx);

    // Step 2: Query live VRAM state
    let gpus = vram_audit::vram_snapshot_all()?;

    // Step 3: Compute verdict
    let mut result = compute_verdict(&gpus, model_estimate_mb);
    result.model_estimate_mb = model_estimate_mb;
    result.gpus = gpus;

    Ok(result)
}

/// Run a preflight check without a model file — just report GPU VRAM state.
///
/// Useful for debugging "what's consuming my VRAM?" without targeting a
/// specific model. Pass `required_mb` as the VRAM you need; 0 for
/// inventory-only mode (always returns `Go`).
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

/// Compute verdict from GPU snapshots and required VRAM.
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

    // Find GPU with the most free VRAM
    let best = gpus.iter().max_by_key(|g| g.free_mb).expect("gpus is non-empty");

    if best.free_mb >= model_estimate_mb {
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
                    "{model_estimate_mb}MB needed, GPU{} ({}) has {}MB free — only {headroom_pct:.0}% headroom, may OOM under load",
                    best.index, best.name, best.free_mb,
                ),
                model_estimate_mb,
                best_gpu_index: Some(best.index),
                gpus: Vec::new(),
            }
        }
    } else {
        // Build a concise explanation of what's consuming VRAM on the best GPU
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
```

- [ ] **Step 3: Wire module into lib.rs**

Add to `Native/pcai_core/pcai_core_lib/src/lib.rs` after the `gpu` module:

```rust
#[cfg(feature = "nvml")]
pub mod preflight;
```

Note: The `vram_audit` module calls `super::get_nvml()` which references the parent `preflight` module, which re-exports from `crate::gpu`. The `get_nvml` function in `gpu/mod.rs` is currently private (`fn get_nvml()`). It needs to be made `pub(crate)`:

In `Native/pcai_core/pcai_core_lib/src/gpu/mod.rs`, change:

```rust
fn get_nvml() -> Option<&'static nvml_wrapper::Nvml> {
```

to:

```rust
pub(crate) fn get_nvml() -> Option<&'static nvml_wrapper::Nvml> {
```

- [ ] **Step 4: Run all preflight tests**

Run: `cd Native/pcai_core && cargo test -p pcai_core_lib --features nvml --lib preflight -- -v`
Expected: All tests in `preflight::mod::tests`, `preflight::gguf::tests`, `preflight::vram_audit::tests` PASS

- [ ] **Step 5: Commit**

```bash
git add Native/pcai_core/pcai_core_lib/src/preflight/mod.rs
git add Native/pcai_core/pcai_core_lib/src/lib.rs
git add Native/pcai_core/pcai_core_lib/src/gpu/mod.rs
git commit -m "feat(preflight): add readiness verdict with GPU selection logic"
```

---

### Task 4: FFI Export — `pcai_gpu_preflight_json`

**Files:**
- Modify: `Native/pcai_core/pcai_core_lib/src/lib.rs` (add FFI function)

Expose the preflight check as a single C FFI function returning JSON, matching the existing pattern of `pcai_gpu_info_json()`.

- [ ] **Step 1: Write the failing test**

Add to the existing test module in `lib.rs`:

```rust
#[cfg(test)]
mod preflight_ffi_tests {
    use super::*;

    #[test]
    #[cfg(feature = "nvml")]
    fn preflight_json_returns_valid_json_for_vram_check() {
        // Test the VRAM-only path (no model file needed)
        let ptr = pcai_gpu_preflight_json(std::ptr::null(), 0, 0);
        if !ptr.is_null() {
            let json_str = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
            let parsed: serde_json::Value = serde_json::from_str(json_str)
                .expect("preflight FFI should return valid JSON");
            assert!(parsed.get("verdict").is_some(), "JSON must have verdict field");
            pcai_free_string(ptr as *mut c_char);
        }
        // ptr can be null on machines without NVML — that's fine
    }

    #[test]
    #[cfg(feature = "nvml")]
    fn preflight_json_with_nonexistent_model_returns_error() {
        let path = CString::new("/nonexistent/model.gguf").unwrap();
        let ptr = pcai_gpu_preflight_json(path.as_ptr(), 0, 0);
        if !ptr.is_null() {
            let json_str = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
            let parsed: serde_json::Value = serde_json::from_str(json_str)
                .expect("error case should still return valid JSON");
            // Should return a fail verdict with file error in reason
            assert_eq!(parsed["verdict"], "fail");
            pcai_free_string(ptr as *mut c_char);
        }
    }
}
```

- [ ] **Step 2: Implement FFI export**

Add to `lib.rs` after the existing GPU FFI functions (around line 408):

```rust
/// Run a GPU preflight readiness check and return the result as a JSON string.
///
/// # Parameters
/// * `model_path` — Path to a GGUF model file (null for VRAM-only inventory)
/// * `context_length` — Context length override (0 = use model default)
/// * `required_mb` — Minimum VRAM required in MB (used when model_path is null)
///
/// # Returns
/// A heap-allocated JSON string. Caller must free with `pcai_free_string()`.
/// Returns null only if JSON serialization fails (should never happen).
///
/// JSON schema:
/// ```json
/// {
///   "verdict": "go" | "warn" | "fail",
///   "reason": "human-readable explanation",
///   "model_estimate_mb": 5800,
///   "best_gpu_index": 1,
///   "gpus": [{ "index": 0, "name": "...", "total_mb": 8192, "used_mb": 4000, "free_mb": 4192, "processes": [...] }]
/// }
/// ```
///
/// # Safety
/// * `model_path` must be null or a valid null-terminated UTF-8 C string
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
```

- [ ] **Step 3: Run FFI tests**

Run: `cd Native/pcai_core && cargo test -p pcai_core_lib --features nvml preflight_ffi_tests -- -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add Native/pcai_core/pcai_core_lib/src/lib.rs
git commit -m "feat(preflight): add pcai_gpu_preflight_json FFI export"
```

---

### Task 5: CLI Subcommand — `pcai-perf preflight`

**Files:**
- Modify: `Native/pcai_core/pcai_perf_cli/Cargo.toml` (enable nvml feature)
- Modify: `Native/pcai_core/pcai_perf_cli/src/main.rs` (add preflight subcommand)

- [ ] **Step 1: Add nvml feature to pcai_perf_cli**

In `Native/pcai_core/pcai_perf_cli/Cargo.toml`, add the nvml feature passthrough:

```toml
[dependencies]
pcai_core_lib = { path = "../pcai_core_lib", features = ["nvml"] }
```

Note: The dependency already exists without features. Change the existing line to add `features = ["nvml"]`.

- [ ] **Step 2: Add preflight subcommand to main.rs**

Add `"preflight"` to the match in `run()`:

```rust
match command {
    "processes" => run_processes(&args[2..]),
    "disk" => run_disk(&args[2..]),
    "hash-list" => run_hash_list(&args[2..]),
    "preflight" => run_preflight(&args[2..]),
    "worker" => run_worker(),
    _ => bail!("usage: pcai-perf <processes|disk|hash-list|preflight|worker> [options]"),
}
```

Add the handler function:

```rust
fn run_preflight(args: &[String]) -> Result<()> {
    let model_path = parse_string_flag(args, "--model");
    let context_length = parse_usize_flag(args, "--ctx")?.unwrap_or(0) as u64;
    let required_mb = parse_usize_flag(args, "--required-mb")?.unwrap_or(0) as u64;

    let result = if let Some(path) = model_path {
        pcai_core_lib::preflight::check_readiness(&path, context_length)?
    } else {
        pcai_core_lib::preflight::check_vram_state(required_mb)?
    };

    println!("{}", serde_json::to_string(&result)?);

    // Exit code matches verdict: 0=go, 1=warn, 2=fail
    let exit_code = match result.verdict {
        pcai_core_lib::preflight::Verdict::Go => 0,
        pcai_core_lib::preflight::Verdict::Warn => 1,
        pcai_core_lib::preflight::Verdict::Fail => 2,
    };

    if exit_code != 0 {
        std::process::exit(exit_code);
    }

    Ok(())
}
```

Also add the `"preflight"` command to the worker handler:

```rust
// Inside handle_worker_request match:
"preflight" => {
    let model_path = request.get("model").and_then(Value::as_str);
    let ctx = request.get("ctx").and_then(Value::as_u64).unwrap_or(0);
    let required_mb = request.get("required_mb").and_then(Value::as_u64).unwrap_or(0);

    let result = if let Some(path) = model_path {
        pcai_core_lib::preflight::check_readiness(path, ctx)?
    } else {
        pcai_core_lib::preflight::check_vram_state(required_mb)?
    };

    Ok(serde_json::to_value(result)?)
}
```

- [ ] **Step 3: Build and test CLI**

Run: `cd Native/pcai_core && cargo build -p pcai-perf --release`
Expected: Compiles without errors

Run: `./target/release/pcai-perf preflight`
Expected: JSON output with `verdict`, `reason`, `gpus` array (or exit 2 if no GPUs available)

Run: `./target/release/pcai-perf preflight --required-mb 4000`
Expected: JSON with verdict based on whether 4000MB is available

- [ ] **Step 4: Commit**

```bash
git add Native/pcai_core/pcai_perf_cli/Cargo.toml
git add Native/pcai_core/pcai_perf_cli/src/main.rs
git commit -m "feat(preflight): add pcai-perf preflight CLI subcommand"
```

---

### Task 6: PowerShell Wrapper — `Test-PcaiGpuReadiness`

**Files:**
- Create: `Modules/PC-AI.Gpu/Public/Test-PcaiGpuReadiness.ps1`
- Modify: `Modules/PC-AI.Gpu/PC-AI.Gpu.psd1` (add export)

Follows the same FFI pattern as `Get-NvidiaGpuInventory` — calls the DLL via P/Invoke, falls back to CLI if DLL unavailable.

- [ ] **Step 1: Create PowerShell function**

```powershell
# Modules/PC-AI.Gpu/Public/Test-PcaiGpuReadiness.ps1

#Requires -Version 5.1
<#
.SYNOPSIS
    Pre-flight GPU readiness check for LLM model loading.

.DESCRIPTION
    Checks whether sufficient VRAM is available to load a GGUF model,
    including per-process VRAM audit showing what is consuming GPU memory.

    Returns a structured verdict: Go (safe), Warn (tight fit), or Fail (won't fit).

    Uses two execution paths in priority order:
      1. FFI via pcai_core_lib.dll (pcai_gpu_preflight_json) — fast, no subprocess
      2. CLI via pcai-perf.exe preflight — fallback when DLL is unavailable

.PARAMETER ModelPath
    Path to a GGUF model file. When specified, parses the GGUF header to
    estimate memory requirements. Omit for VRAM-inventory-only mode.

.PARAMETER ContextLength
    Override the model's default context length for KV cache estimation.
    Larger contexts need more VRAM. Default: 0 (use model default).

.PARAMETER RequiredMB
    Minimum VRAM required in MB. Used when -ModelPath is omitted.
    Default: 0 (inventory-only — always returns Go).

.PARAMETER AsJson
    Return raw JSON string instead of parsed PowerShell objects.

.OUTPUTS
    PSCustomObject with properties:
      Verdict          - "go", "warn", or "fail"
      Reason           - Human-readable explanation
      ModelEstimateMB  - Estimated VRAM needed for the model
      BestGpuIndex     - Index of best GPU (null if fail)
      Gpus             - Array of GPU snapshots with per-process VRAM breakdown
      Source           - "ffi" or "cli"

.EXAMPLE
    Test-PcaiGpuReadiness
    # Returns VRAM inventory for all GPUs with process breakdown.

.EXAMPLE
    Test-PcaiGpuReadiness -ModelPath "C:\Models\qwen2.5-coder-7b-q4km.gguf"
    # Checks if the 7B model fits, shows verdict and best GPU.

.EXAMPLE
    Test-PcaiGpuReadiness -RequiredMB 6000
    # Checks if 6000 MB of VRAM is available without parsing a model file.

.EXAMPLE
    if ((Test-PcaiGpuReadiness -ModelPath $model).Verdict -eq 'fail') {
        Write-Warning "Model won't fit — check VRAM usage"
    }
#>
function Test-PcaiGpuReadiness {
    [CmdletBinding()]
    [OutputType([PSCustomObject])]
    param(
        [Parameter()]
        [string]$ModelPath,

        [Parameter()]
        [ValidateRange(0, 1048576)]
        [int]$ContextLength = 0,

        [Parameter()]
        [ValidateRange(0, 1048576)]
        [int]$RequiredMB = 0,

        [Parameter()]
        [switch]$AsJson
    )

    $ErrorActionPreference = 'Stop'

    # ── Primary path: FFI via pcai_core_lib.dll ──────────────────────────────

    $script:PreflightInteropTypeName = 'PcaiPreflightInterop'

    $coreDll = Resolve-PcaiCoreLibDll
    if ($coreDll) {
        try {
            # Reuse DLL path resolution from Get-NvidiaGpuInventory
            if (-not ($script:PreflightInteropTypeName -as [type])) {
                $escapedPath = $coreDll.Replace('\', '\\')
                $typeDef = @"
using System;
using System.Runtime.InteropServices;

public static class $($script:PreflightInteropTypeName) {
    [DllImport(@"$escapedPath", CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr pcai_gpu_preflight_json(
        [MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath,
        ulong contextLength,
        ulong requiredMb);

    [DllImport(@"$escapedPath", CallingConvention = CallingConvention.Cdecl)]
    public static extern void pcai_free_string(IntPtr ptr);

    public static string MarshalAndFree(IntPtr ptr) {
        if (ptr == IntPtr.Zero) return null;
        try { return Marshal.PtrToStringAnsi(ptr); }
        finally { pcai_free_string(ptr); }
    }
}
"@
                Add-Type -TypeDefinition $typeDef -Language CSharp -ErrorAction Stop | Out-Null
            }

            $t = ($script:PreflightInteropTypeName -as [type])
            $modelArg = if ($ModelPath) { $ModelPath } else { $null }

            $jsonPtr = $t::pcai_gpu_preflight_json(
                $modelArg,
                [uint64]$ContextLength,
                [uint64]$RequiredMB
            )
            $json = $t::MarshalAndFree($jsonPtr)

            if ($json) {
                if ($AsJson) { return $json }
                $parsed = $json | ConvertFrom-Json
                $parsed | Add-Member -NotePropertyName Source -NotePropertyValue 'ffi' -PassThru
                return
            }
        }
        catch {
            Write-Verbose "Preflight FFI failed: $($_.Exception.Message) — falling back to CLI."
        }
    }

    # ── Fallback: CLI via pcai-perf.exe ──────────────────────────────────────

    $cliCandidates = @(
        (Join-Path (Split-Path $PSScriptRoot | Split-Path | Split-Path) 'Native\pcai_core\target\release\pcai-perf.exe'),
        (Join-Path $env:USERPROFILE '.local\bin\pcai-perf.exe')
    )

    $cli = $null
    foreach ($candidate in $cliCandidates) {
        if (Test-Path $candidate -ErrorAction SilentlyContinue) {
            $cli = $candidate
            break
        }
    }

    if (-not $cli) {
        Write-Error "Neither pcai_core_lib.dll nor pcai-perf.exe found. Build with: cargo build -p pcai-perf --release"
        return
    }

    $cliArgs = @('preflight')
    if ($ModelPath) { $cliArgs += @('--model', $ModelPath) }
    if ($ContextLength -gt 0) { $cliArgs += @('--ctx', $ContextLength.ToString()) }
    if ($RequiredMB -gt 0) { $cliArgs += @('--required-mb', $RequiredMB.ToString()) }

    Write-Verbose "Running: $cli $($cliArgs -join ' ')"
    $json = & $cli @cliArgs 2>&1 | Where-Object { $_ -is [string] }

    if ($AsJson) { return ($json -join '') }

    $parsed = ($json -join '') | ConvertFrom-Json
    $parsed | Add-Member -NotePropertyName Source -NotePropertyValue 'cli' -PassThru
}
```

- [ ] **Step 2: Update module manifest**

In `Modules/PC-AI.Gpu/PC-AI.Gpu.psd1`, add `'Test-PcaiGpuReadiness'` to the `FunctionsToExport` array.

- [ ] **Step 3: Test PowerShell wrapper**

Run:
```powershell
Import-Module ./Modules/PC-AI.Gpu -Force
Test-PcaiGpuReadiness -Verbose
Test-PcaiGpuReadiness -RequiredMB 6000
Test-PcaiGpuReadiness -AsJson
```

Expected: Structured output with Verdict, Reason, Gpus (including per-process breakdown), Source

- [ ] **Step 4: Commit**

```bash
git add Modules/PC-AI.Gpu/Public/Test-PcaiGpuReadiness.ps1
git add Modules/PC-AI.Gpu/PC-AI.Gpu.psd1
git commit -m "feat(preflight): add Test-PcaiGpuReadiness PowerShell wrapper"
```

---

### Task 7: Integration Tests

**Files:**
- Create: `Native/pcai_core/pcai_core_lib/tests/preflight_tests.rs`

Integration tests that exercise the full preflight path on machines with NVIDIA GPUs.

- [ ] **Step 1: Write integration tests**

```rust
// Native/pcai_core/pcai_core_lib/tests/preflight_tests.rs

//! Integration tests for the preflight module.
//!
//! These tests require NVML (an NVIDIA GPU with drivers installed).
//! They are skipped gracefully on machines without NVIDIA hardware.

#[cfg(feature = "nvml")]
mod nvml_tests {
    use pcai_core_lib::preflight;

    #[test]
    fn check_vram_state_returns_valid_result() {
        let result = preflight::check_vram_state(0).expect("check_vram_state should not error");
        // On machines with NVIDIA GPUs, verdict should be Go (inventory mode)
        // On machines without, vram_snapshot_all returns empty → verdict is Fail
        assert!(
            matches!(result.verdict, preflight::Verdict::Go | preflight::Verdict::Fail),
            "inventory-only should be Go (with GPU) or Fail (without)"
        );
    }

    #[test]
    fn check_vram_state_with_absurd_requirement_fails() {
        let result = preflight::check_vram_state(999_999)
            .expect("should not error even with huge requirement");
        // No GPU has 999 GB of VRAM
        assert_eq!(result.verdict, preflight::Verdict::Fail);
    }

    #[test]
    fn check_readiness_with_nonexistent_file_errors() {
        let result = preflight::check_readiness("/nonexistent/model.gguf", 0);
        assert!(result.is_err(), "nonexistent model file should return Err");
    }

    #[test]
    fn vram_snapshot_gpu_names_not_empty() {
        let snapshots = preflight::vram_audit::vram_snapshot_all().unwrap_or_default();
        for gpu in &snapshots {
            assert!(!gpu.name.is_empty(), "GPU name should not be empty");
            assert!(gpu.total_mb > 0, "GPU total VRAM should be >0");
        }
    }

    #[test]
    fn preflight_result_json_is_compact() {
        let result = preflight::check_vram_state(0).expect("should not error");
        let json = serde_json::to_string(&result).expect("should serialize");
        // Compact JSON should be under 2KB for typical 2-GPU system
        assert!(
            json.len() < 4096,
            "preflight JSON should be compact, got {} bytes",
            json.len()
        );
    }
}
```

- [ ] **Step 2: Run integration tests**

Run: `cd Native/pcai_core && cargo test -p pcai_core_lib --features nvml --test preflight_tests -- -v`
Expected: All tests PASS (or skip gracefully if no GPU)

- [ ] **Step 3: Commit**

```bash
git add Native/pcai_core/pcai_core_lib/tests/preflight_tests.rs
git commit -m "test(preflight): add integration tests for GPU readiness checks"
```

---

### Task 8: Pester Tests for PowerShell Wrapper

**Files:**
- Create: `Tests/Integration/FFI.Preflight.Tests.ps1`

- [ ] **Step 1: Write Pester tests**

```powershell
# Tests/Integration/FFI.Preflight.Tests.ps1

Describe 'Test-PcaiGpuReadiness' {
    BeforeAll {
        Import-Module "$PSScriptRoot/../../Modules/PC-AI.Gpu" -Force -ErrorAction Stop
    }

    Context 'Module availability' {
        It 'Should be exported from PC-AI.Gpu module' {
            Get-Command -Name Test-PcaiGpuReadiness -Module PC-AI.Gpu | Should -Not -BeNullOrEmpty
        }
    }

    Context 'Inventory mode (no model)' {
        It 'Returns a result with Verdict property' {
            $result = Test-PcaiGpuReadiness
            $result.Verdict | Should -BeIn @('go', 'warn', 'fail')
        }

        It 'Returns a result with Reason property' {
            $result = Test-PcaiGpuReadiness
            $result.Reason | Should -Not -BeNullOrEmpty
        }

        It 'Returns Gpus array' {
            $result = Test-PcaiGpuReadiness
            $result.Gpus | Should -Not -BeNullOrEmpty -Because 'at least one GPU should be present'
        }

        It 'Returns Source property (ffi or cli)' {
            $result = Test-PcaiGpuReadiness
            $result.Source | Should -BeIn @('ffi', 'cli')
        }
    }

    Context 'Required MB mode' {
        It 'Returns Go for trivial requirement (1 MB)' {
            $result = Test-PcaiGpuReadiness -RequiredMB 1
            $result.Verdict | Should -Be 'go'
        }

        It 'Returns Fail for impossible requirement (999999 MB)' {
            $result = Test-PcaiGpuReadiness -RequiredMB 999999
            $result.Verdict | Should -Be 'fail'
        }
    }

    Context 'JSON output mode' {
        It 'Returns valid JSON with -AsJson' {
            $json = Test-PcaiGpuReadiness -AsJson
            { $json | ConvertFrom-Json } | Should -Not -Throw
        }

        It 'JSON contains verdict field' {
            $json = Test-PcaiGpuReadiness -AsJson
            $parsed = $json | ConvertFrom-Json
            $parsed.verdict | Should -BeIn @('go', 'warn', 'fail')
        }
    }

    Context 'GPU process audit' {
        It 'GPU snapshots include process list' {
            $result = Test-PcaiGpuReadiness
            foreach ($gpu in $result.Gpus) {
                # processes may be empty if nothing is using the GPU
                $gpu.PSObject.Properties.Name | Should -Contain 'processes'
            }
        }
    }
}
```

- [ ] **Step 2: Run Pester tests**

Run: `pwsh -Command "Invoke-Pester Tests/Integration/FFI.Preflight.Tests.ps1 -Output Detailed"`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add Tests/Integration/FFI.Preflight.Tests.ps1
git commit -m "test(preflight): add Pester integration tests for Test-PcaiGpuReadiness"
```

---

### Task 9: Summary Commit and Verification

- [ ] **Step 1: Run full test suite**

```bash
cd Native/pcai_core
cargo test -p pcai_core_lib --features nvml --lib -- -v
cargo test -p pcai_core_lib --features nvml --test preflight_tests -- -v
cargo clippy -p pcai_core_lib --features nvml -- -D warnings
cargo fmt -p pcai_core_lib --check
```

Expected: All tests PASS, no clippy warnings, format clean.

- [ ] **Step 2: Run PowerShell tests**

```powershell
Import-Module ./Modules/PC-AI.Gpu -Force
Test-PcaiGpuReadiness -Verbose
Invoke-Pester Tests/Integration/FFI.Preflight.Tests.ps1 -Output Detailed
```

Expected: All tests PASS, verbose output shows which path (FFI or CLI) was used.

- [ ] **Step 3: Run CLI smoke test**

```bash
./Native/pcai_core/target/release/pcai-perf preflight
./Native/pcai_core/target/release/pcai-perf preflight --required-mb 4000
echo $?  # Should be 0 (go), 1 (warn), or 2 (fail)
```

Expected: Compact JSON on stdout, meaningful exit code.

- [ ] **Step 4: Verify JSON output compactness**

The preflight JSON for a typical 2-GPU system should be under 1KB. Verify:

```bash
./Native/pcai_core/target/release/pcai-perf preflight | wc -c
# Expected: <1024 bytes
```

---

## Post-Implementation Notes

**Future tasks (not in this plan):**
1. **Backend integration** — Call `preflight::check_readiness()` inside `pcai_load_model()` and `pcai_media::config::resolve_auto_cuda_device()` to produce actionable OOM errors instead of opaque crashes. This is a separate plan because it touches the inference/media hot paths and needs careful error propagation design.
2. **Error enrichment** — On CUDA OOM catch, call `preflight::check_vram_state()` to enrich the error message with GPU state at the moment of failure.
3. **CI integration** — Add preflight check as a pre-step in evaluation harness runs (`Invoke-InferenceEvaluation.ps1`).
