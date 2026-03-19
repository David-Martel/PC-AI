# PC_AI Development TODO

> **Updated:** 2026-03-19 | **Status:** Active Development
> **Coordinating agents:** Claude, Codex, Jules (53 sessions complete)

---

## LLM Inference Optimization (Priority: HIGH)

### Current Performance
- **Janus-Pro-1B**: 36 tok/s on RTX 2000 Ada SM 89 (2.0x vs baseline)
- **Janus-Pro-7B**: TBD — model available (14 GB safetensors, fits in 16 GB RTX 5060 Ti)
- **Primary GPU**: RTX 5060 Ti (SM 120, 16 GB GDDR7, 448 GB/s, Thunderbolt 4 eGPU)
- Quality: 4/5 photorealistic images, understanding with repetition penalty
- **Target: 200+ tok/s** via GGUF quantization on 7B model

### GPU Configuration
- **RTX 5060 Ti** (cuda:1): Primary inference GPU. Connected via Thunderbolt 4 (40 Gbps = 5 GB/s).
  TB4 only affects model loading (~3s for 14 GB); inference runs at full 448 GB/s VRAM bandwidth.
- **RTX 2000 Ada** (cuda:0): Secondary. Used for display, Ollama, smaller models.

### Path to 200+ tok/s (Janus-Pro-7B on RTX 5060 Ti)
| Technique | Model Size | Theoretical (448 GB/s) | At 35% eff. | Status |
|-----------|-----------|----------------------|-------------|--------|
| BF16 (no quantization) | 14 GB | 32 tok/s | **5.8 tok/s** (OOM at VQ decode) | Measured |
| **GGUF Q8_0** | **7 GB** | **64 tok/s** | **22 tok/s** | **IN PROGRESS** |
| **GGUF Q4_K_M** | **3.5 GB** | **128 tok/s** | **45 tok/s** | **IN PROGRESS** |
| Q4_K + CUDA Graphs (+30%) | 3.5 GB | 166 tok/s | 58 tok/s | TODO |
| Q4_K + improved efficiency (50%) | 3.5 GB | 128 tok/s | **64 tok/s** | TODO |
| Speculative decoding (7B draft=12/32) | 3.5 GB | +1.5x | **96 tok/s** | TODO |
| Q4_K + spec decode + CUDA Graphs | 3.5 GB | — | **~120-150 tok/s** | TODO |

**Note**: 200 tok/s on 7B may require FP8 quantization (hardware-supported on SM 120)
or NVFP4 (4-bit tensor cores on Blackwell). These need custom CUTLASS kernels.

### Thunderbolt 4 Optimization Notes
- TB4 bandwidth: 40 Gbps = ~5 GB/s (vs PCIe x16: ~32 GB/s)
- Model loading: 14 GB / 5 GB/s = ~3s (acceptable one-time cost)
- KV cache: must stay in GPU VRAM (no CPU offload — TB4 too slow)
- Activations: all computation on-GPU, only final output transfers to CPU
- Minimize host<->device transfers: use GPU-side sampling (Gumbel-max for 100K+ vocab)

### Phase 1: Code-Level Fixes — COMPLETE (1.67x achieved)
- [x] Flash attention code path (`#[cfg(feature = "flash-attn")]`)
- [x] Remove .contiguous() after KV cache cat (27,648 GPU copies eliminated)
- [x] GPU-side argmax (576 PCIe sync stalls eliminated)
- [x] from_slice instead of from_vec (cudaMalloc pressure reduced)
- [x] KV cache axis bug fixed (narrow was using wrong dimension)
- [x] FunctionGemma flash-attn enabled as default feature

### Phase 2: cuDNN Integration — RESOLVED (no-op for Janus)
- [x] Investigated cuDNN build — cudarc 0.19.3 cuDNN only has conv/softmax/pooling
- [x] Documented as forward-compat feature (no SDPA acceleration available)
- [x] flash-attn confirmed Linux-only (candle-flash-attn build.rs requires CUDA kernels)

### Phase 3: Quantization — TODO
- [ ] NVFP4 weight quantization (native on RTX 5060 Ti Blackwell)
- [ ] FP8 KV cache quantization (50% memory reduction)
- [ ] AWQ quantization (Marlin kernel, 2.6x vs GPTQ)

### Phase 4: System-Level — COMPLETE
- [x] Pre-allocated KV cache ring buffer (PreAllocKvCache in janus_llama.rs)
- [x] CacheVariant enum wired into generate.rs (default=prealloc)
- [x] GPU Gumbel-max sampling (eliminates 16KB/step CPU transfer)
- [x] True multinomial sampling restored (argmax caused solid-color images)
- [x] Repetition penalty for understanding pipeline (1.2, prevents loops)
- [ ] CUDA Graphs (eliminate 20-30% CPU launch overhead)
- [ ] Memory pinning for faster PCIe transfers

### Phase 5: Multi-Token Prediction — RESEARCHED
Priority order (no training required first):
- [ ] **Jacobi/SJD decoding** — 1.5-2x, training-free, init 576 positions & iterate
- [ ] **Pre-computed Gumbel noise** — 5-10%, trivial: batch-generate 576 noise vectors
- [ ] **Self-speculative decoding** — 1.5-2x, use first 8/24 layers as draft model
- [ ] CUDA Graphs (eliminate 20-30% CPU launch overhead)

Requires fine-tuning:
- [ ] Medusa heads — 2.2-3.6x, 3 lightweight FFN heads (~400MB VRAM)
- [ ] GSD (Grouped Speculative) — up to 3.7x, VQ codebook grouping (ICCV 2025)
- [ ] MTP (Meta/DeepSeek style) — 1.5-2x, additional generation heads

### Phase 6: Cross-Codebase Propagation — TODO
- [ ] FunctionGemma: ring buffer KV cache (uses Tensor::cat at model.rs:669)
- [ ] FunctionGemma: GPU Gumbel sampling (uses CPU to_scalar at model.rs:1130)
- [ ] FunctionGemma: remove unused streaming/block_len fields
- [ ] pcai-inference: propagate any applicable optimizations

### Understanding Pipeline — COMPLETE
- [x] Fixed `<image>` vs `<image_placeholder>` token in prompt template
- [x] Vision embedding splice matches Python reference (before|image|after)
- [x] Prefill logits used directly (no double-processing)
- [x] Repetition penalty prevents text loops
- [x] Quality: correctly describes generated images

---

## Architectural Enhancements — TODO

### Build System
- [x] Build.ps1 defaults: cuda + cudnn + flash-attn + nvml + upscale
- [x] cuda-optimized meta-feature for pcai-media-server
- [x] ring crate eliminated (hf-hub 0.5 async-only)
- [ ] Fix cuDNN build integration
- [ ] Add CUDA build caching to CI

### Driver & SDK Management
- [ ] GPU driver update (582.41 → 591.55)
- [ ] Nsight Graphics 2025.5 installation
- [ ] nvCOMP installation (GPU compression)

### Testing & Validation
- [x] 468/638 Pester tests passing (0 regressions)
- [x] 73/73 Rust NVML tests passing
- [x] 37/37 Rust media model tests passing
- [ ] Pester test mock for native exe (Windows limitation)
- [ ] Full Pester test coverage for install flows

---

# NVIDIA Software Installer Framework — Implementation Plan

> **Created:** 2026-03-18 | **Status:** Phases 1-3 COMPLETE
> **Coordinating agents:** Claude (module + config), Codex (Build.ps1 + smoke tests)

## Overview

Smart PowerShell framework to auto-detect, download, install, and configure the full NVIDIA software stack. Integrates with PC-AI's existing driver management and build systems.

**System:** RTX 5060 Ti (16GB, SM 120 Blackwell) + RTX 2000 Ada (8GB, SM 89 Ada Lovelace)

## Current State (2026-03-18)

| Component | Installed | Latest | Status |
|-----------|-----------|--------|--------|
| GPU Driver | 582.41 | ~591.55 | Outdated |
| CUDA Toolkit | 12.9 (+ 12.1, 12.6, 12.8, 13.0, 13.1, 13.2) | 13.2.0 | Present (7 versions) |
| cuDNN | 9.8 | 9.8+ | Check |
| TensorRT | 10.9.0 | 10.9+ | Check |
| Nsight Compute | 2026.1.0 | 2026.1.0 | Current |
| Nsight Systems | 2025.6.3 | 2025.6+ | Check |
| cudarc (Rust) | 0.19.3 | 0.19.3 | Current |
| candle-core (Rust) | 0.9.2 | 0.9.2 | Current |

## Architecture Decision

**Create new `PC-AI.Gpu` module** (not extend PC-AI.Drivers).

**Rationale:** PC-AI.Drivers manages PnP device drivers via hardware IDs (VID/PID). NVIDIA SDK stack (CUDA, cuDNN, TensorRT, NSight) is fundamentally different — it's a software development toolkit, not device drivers. The GPU *driver* itself gets a PnP entry in `driver-registry.json`, but the SDK stack needs its own detection, versioning, compatibility matrix, and installation logic.

## File Structure

```
PC_AI/
  Config/
    nvidia-software-registry.json    # NEW — NVIDIA software catalog
    driver-registry.json             # MODIFY — Add NVIDIA GPU driver PnP entries
  Modules/
    PC-AI.Gpu/                       # NEW — Module directory
      PC-AI.Gpu.psd1                 # Module manifest
      PC-AI.Gpu.psm1                 # Module loader (Public/Private dot-source)
      Public/
        Get-NvidiaGpuInventory.ps1           # Detect GPUs via nvidia-smi + CIM
        Get-NvidiaSoftwareRegistry.ps1       # Load nvidia-software-registry.json
        Get-NvidiaSoftwareStatus.ps1         # Installed vs latest comparison report
        Install-NvidiaSoftware.ps1           # Download + silent install
        Update-NvidiaSoftwareRegistry.ps1    # Update registry entries
        Initialize-NvidiaEnvironment.ps1     # Unified env setup (superset of Initialize-CudaEnvironment)
        Get-NvidiaCompatibilityMatrix.ps1    # GPU↔CUDA↔cuDNN↔TensorRT compat
        Get-NvidiaGpuUtilization.ps1         # Real-time GPU status (consolidates duplicated patterns)
      Private/
        Resolve-NvidiaInstallPath.ps1        # Detect installed paths per component
        Get-NvidiaDriverVersion.ps1          # Driver version from nvidia-smi or registry
        Get-CudaVersionFromPath.ps1          # Parse version.json from CUDA installs
        Get-CudnnVersionFromHeader.ps1       # Parse cudnn_version.h
        Get-TensorRtVersionFromHeader.ps1    # Parse NvInferVersion.h
        Get-NsightVersions.ps1              # Scan for Nsight Compute/Systems
        Invoke-NvidiaSilentInstall.ps1       # Silent install executor
        Test-NvidiaDownloadUrl.ps1           # Build/validate NVIDIA download URLs
        Backup-NvidiaEnvironment.ps1         # Snapshot env state for rollback
  Tools/
    Update-NvidiaSoftware.ps1        # NEW — Top-level orchestrator
  Tests/
    Unit/
      PC-AI.Gpu.Tests.ps1           # NEW — Pester unit tests
```

## Function Specifications

### Public Functions (8)

| Function | Purpose | Risk |
|----------|---------|------|
| `Get-NvidiaGpuInventory` | Detect GPUs via nvidia-smi (name, driver, compute cap, VRAM, temp) + CIM fallback | Read-only |
| `Get-NvidiaSoftwareRegistry` | Load `nvidia-software-registry.json`, filter by `-ComponentId`/`-Category` | Read-only |
| `Get-NvidiaSoftwareStatus` | Compare installed vs latest for all components. Return Current/Outdated/NotInstalled/MultipleVersions | Read-only |
| `Install-NvidiaSoftware` | Download + silent install. Supports `-WhatIf`, `-DownloadOnly`, `-Force`. Backup before changes | **High** (admin) |
| `Update-NvidiaSoftwareRegistry` | Update JSON entries (version, URL, SHA256) | Low |
| `Initialize-NvidiaEnvironment` | Unified env setup: delegates to Initialize-CudaEnvironment, adds cuDNN/TensorRT/NSight paths | Medium |
| `Get-NvidiaCompatibilityMatrix` | Given detected GPUs, return which SDK versions are compatible. Flag outdated components | Read-only |
| `Get-NvidiaGpuUtilization` | Real-time GPU utilization, VRAM, temp, power via nvidia-smi CSV | Read-only |

### Private Functions (8)

| Function | Purpose |
|----------|---------|
| `Resolve-NvidiaInstallPath` | Scan known install locations per component type |
| `Get-NvidiaDriverVersion` | nvidia-smi query or Win32_VideoController CIM fallback |
| `Get-CudaVersionFromPath` | Read `version.json` from CUDA install dir |
| `Get-CudnnVersionFromHeader` | Parse `CUDNN_MAJOR/MINOR/PATCHLEVEL` from `cudnn_version.h` |
| `Get-TensorRtVersionFromHeader` | Parse `NV_TENSORRT_MAJOR/MINOR/PATCH` from `NvInferVersion.h` |
| `Get-NsightVersions` | Scan `Nsight Compute *` / `Nsight Systems *` directories |
| `Invoke-NvidiaSilentInstall` | Low-level installer: exe `-s`, MSI `/qb`, zip extraction. Exit code 3010 = reboot |
| `Backup-NvidiaEnvironment` | Capture all NVIDIA env vars + versions to `.pcai/nvidia-backup/` JSON |

## Config Schema: nvidia-software-registry.json

```json
{
  "version": "1.0.0",
  "lastUpdated": "2026-03-18T00:00:00Z",
  "trustedSources": [
    { "id": "nvidia", "baseUrl": "https://developer.nvidia.com", "type": "vendor" },
    { "id": "nvidia-download", "baseUrl": "https://developer.download.nvidia.com", "type": "cdn" }
  ],
  "components": [
    {
      "id": "gpu-driver",
      "name": "NVIDIA GPU Driver",
      "category": "driver",
      "detectionMethod": "nvidia-smi",
      "latestVersion": "591.55",
      "manualDownloadUrl": "https://www.nvidia.com/download/index.aspx",
      "installerType": "exe-silent",
      "silentArgs": "-s -noreboot",
      "compatibleGpus": ["RTX 5060 Ti", "RTX 2000 Ada"]
    },
    {
      "id": "cuda-toolkit",
      "name": "CUDA Toolkit",
      "category": "sdk",
      "detectionMethod": "version-json",
      "installBasePath": "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
      "latestVersion": "13.2.0",
      "manualDownloadUrl": "https://developer.nvidia.com/cuda-downloads",
      "installerType": "exe-silent",
      "silentArgs": "-s",
      "envVars": ["CUDA_PATH", "CUDA_HOME"],
      "pathEntries": ["bin", "nvvm\\bin", "libnvvp"],
      "allowMultipleVersions": true
    },
    {
      "id": "cudnn",
      "name": "cuDNN",
      "category": "sdk",
      "detectionMethod": "header-parse",
      "headerFile": "cudnn_version.h",
      "versionDefines": ["CUDNN_MAJOR", "CUDNN_MINOR", "CUDNN_PATCHLEVEL"],
      "installBasePath": "C:\\Program Files\\NVIDIA\\CUDNN",
      "latestVersion": "9.8.0",
      "manualDownloadUrl": "https://developer.nvidia.com/cudnn",
      "envVars": ["CUDNN_PATH"],
      "requiresCuda": true
    },
    {
      "id": "tensorrt",
      "name": "TensorRT",
      "category": "sdk",
      "detectionMethod": "header-parse",
      "headerFile": "NvInferVersion.h",
      "versionDefines": ["NV_TENSORRT_MAJOR", "NV_TENSORRT_MINOR", "NV_TENSORRT_PATCH"],
      "installBasePath": "C:\\Program Files\\NVIDIA\\TensorRT",
      "latestVersion": "10.9.0",
      "manualDownloadUrl": "https://developer.nvidia.com/tensorrt",
      "envVars": ["TENSORRT_PATH"],
      "requiresCuda": true
    },
    {
      "id": "nsight-compute",
      "name": "Nsight Compute",
      "category": "profiling",
      "detectionMethod": "directory-scan",
      "installBasePath": "C:\\Program Files\\NVIDIA Corporation",
      "directoryPattern": "Nsight Compute *",
      "latestVersion": "2026.1.0",
      "installerType": "bundled"
    },
    {
      "id": "nsight-systems",
      "name": "Nsight Systems",
      "category": "profiling",
      "detectionMethod": "directory-scan",
      "installBasePath": "C:\\Program Files\\NVIDIA Corporation",
      "directoryPattern": "Nsight Systems *",
      "latestVersion": "2025.6.3",
      "installerType": "bundled"
    }
  ],
  "compatibilityMatrix": {
    "entries": [
      { "computeCapability": "8.9", "gpuFamily": "Ada Lovelace", "minCuda": "12.0" },
      { "computeCapability": "12.0", "gpuFamily": "Blackwell", "minCuda": "12.8" }
    ]
  }
}
```

## Integration Points

| Target | Integration | Phase |
|--------|-------------|-------|
| `Config/driver-registry.json` | Add NVIDIA GPU driver PnP entries with `sharedDriverGroup` | 1 |
| `Build.ps1` | Call `Initialize-NvidiaEnvironment` when available (fallback to `Initialize-CudaEnvironment`) | 2 |
| `Tools/Initialize-CudaEnvironment.ps1` | Preserved — `Initialize-NvidiaEnvironment` delegates to it | 2 |
| `CargoTools` | Set `CUDNN_ROOT`, `TENSORRT_ROOT` env vars for Rust build.rs | 2 |
| `Setup-DevEnvironment.ps1` | Add `Get-NvidiaSoftwareStatus -Brief` summary | 2 |
| `PcaiInference.psm1` | Refactor `Get-PcaiCudaCapability` to delegate to `Get-NvidiaGpuInventory` | 4 |
| `.github/workflows/maintenance.yml` | Weekly NVIDIA version check | 4 |

## Safety Considerations

1. **Backup before modify** — `Backup-NvidiaEnvironment` captures env vars + versions to JSON
2. **`-WhatIf` on all writes** — `Install-NvidiaSoftware` has `ConfirmImpact = 'High'`
3. **Admin elevation check** — Reuse `Test-AdminElevation` from PC-AI.Drivers
4. **Trusted host validation** — Only `nvidia.com`, `developer.nvidia.com`, `developer.download.nvidia.com`
5. **SHA256 verification** — Downloaded installers verified against registry hash
6. **CUDA coexistence** — Never removes old versions, only adds/updates CUDA_PATH pointer
7. **Multi-GPU compat** — Validates driver works for ALL installed GPUs before offering update
8. **Reboot detection** — Exit code 3010 triggers consolidated reboot notice

## Phase Breakdown

### Phase 1: Detection and Status (Read-Only) ← COMPLETE
- [x] Design plan
- [x] `Config/nvidia-software-registry.json` with real detected data (v1.2.0, 11 components, 2 GPUs)
- [x] `Modules/PC-AI.Gpu/` module skeleton — 19 files (psm1, psd1, 8 public, 9 private)
- [x] Private detection: all 9 private functions implemented (not stubs)
- [x] Public: all 8 public functions implemented (not stubs)
- [x] `Tests/Unit/PC-AI.Gpu.Tests.ps1` — 75 test cases across 23 contexts
- [x] Module manifest validated, all .ps1 files syntax-checked, module loads cleanly
- [x] `Config/driver-registry.json` v1.2.0 — NVIDIA GPU PnP entries with sharedDriverGroup

### Phase 2: Environment Management ← COMPLETE
- [x] Public: `Initialize-NvidiaEnvironment` (403 lines — delegates to Initialize-CudaEnvironment + cuDNN/TensorRT/Nsight)
- [x] Public: `Get-NvidiaCompatibilityMatrix` (341 lines — per-GPU compat check with IsBlocker flag)
- [x] Private: `Backup-NvidiaEnvironment` (308 lines — backup + restore with ParameterSets)
- [x] Build.ps1 integration (media component build with sccache)
- [ ] Setup-DevEnvironment.ps1 integration
- [ ] Additional unit tests for Phase 2 functions

### Phase 3: Download and Install ← COMPLETE
- [x] Private: `Invoke-NvidiaSilentInstall` (318 lines — exe/msi/zip, Process.Dispose, timeout)
- [x] Private: `Test-NvidiaDownloadUrl` (179 lines — trusted hosts + HTTP HEAD)
- [x] Public: `Install-NvidiaSoftware` (428 lines — download + SHA256 + backup + install + verify)
- [x] Public: `Update-NvidiaSoftwareRegistry` (369 lines — patch mode + refresh-from-system + atomic write)
- [x] `Tools/Update-NvidiaSoftware.ps1` orchestrator (Codex)
- [x] `Tools/Sync-NvidiaDriverVersion.ps1` (811 lines — auto-detect, compare, download, install)
- [ ] Full Pester test coverage for install flows

### Phase 4: Automation and CI
- [ ] GitHub workflow for NVIDIA stack validation
- [ ] Integration with `maintenance.yml` for weekly version checks
- [ ] Registry auto-update script (scrape latest from NVIDIA)
- [x] Consolidate duplicated nvidia-smi patterns — NVML module replaces subprocess calls

### Phase 5: Additional NVIDIA SDKs and Rust Integration ← PARTIALLY COMPLETE

**HIGH priority:**
- [x] **nvml-wrapper** (Rust crate v0.12.0) — DONE. gpu/mod.rs (508 lines), 3 FFI exports, OnceLock singleton, cargo check passed.
- [ ] **Nsight Graphics 2025.5** — GPU crash dump inspector + frame debugger (D3D12/Vulkan). Download from developer.nvidia.com/nsight-graphics. Extends PC_AI diagnostics for GPU driver crash analysis.
- [ ] **TensorRT for RTX** — JIT-compiled inference for consumer RTX GPUs. 50%+ speedup. Standalone library (June 2025+). Rust: `trtx` crate v0.3.1 (experimental, uses cudarc + dynamic loading).
- [ ] **Warp** (Python, `pip install warp-lang`) — GPU simulation framework, JIT compiles to CUDA. 669x CPU speedup. Useful for physics-based diagnostics and optimization pipelines.

**MEDIUM priority — install soon:**
- [ ] **RAPIDS cuDF + cuML** (Python, `pip install cudf cuml`) — GPU-accelerated pandas + scikit-learn. Useful for finance-warehouse data preprocessing.
- [ ] **NVIDIA DALI** (Python, `pip install nvidia-dali-cuda12`) — GPU data pipeline for image preprocessing. 1000+ images/sec.
- [ ] **NPP/NPP+** — GPU image/signal processing (bundled with CUDA 13.2, verify present)
- [ ] **nvCOMP** — GPU-accelerated compression (Snappy, ZSTD, LZ4). Blackwell hardware decompression at 600 GB/s.
- [ ] **Nsight Aftermath** — GPU crash dump SDK. Bundled with Nsight Graphics. Integrates crash dump analysis into diagnostic reports.

**LOW priority / as-needed:**
- [ ] CUTLASS v3 (header-only GEMM templates; v4.x has Windows build issues — wait)
- [ ] Video Codec SDK 13.0 (NVENC/NVDEC; only if adding video diagnostics)
- [ ] NVIDIA Container Toolkit (Docker GPU passthrough; WSL2 route easier)
- [ ] OptiX 9.1 (ray tracing; not relevant for inference)

**Skip (Linux-only or not applicable):**
- DCGM (Linux-only), cuOpt (Linux-only), NeMo (pivoted to audio), NCCL (Linux-only), Merlin (Linux-focused)

**Rust Crates to Evaluate:**

| Crate | Version | Purpose | Priority |
|-------|---------|---------|----------|
| `nvml-wrapper` | 0.12.0 | NVML GPU monitoring (replace nvidia-smi calls) | HIGH |
| `trtx` | 0.3.1 | TensorRT-RTX bindings (experimental) | MEDIUM |
| `gpu-allocator` | 0.28.0 | Vulkan/DX12 memory allocation | LOW |
| `wgpu` | 28.0.0 | Cross-platform GPU compute via Vulkan | LOW |
| `cufft_rust` | 0.6.0 | cuFFT bindings | LOW |
| `cublas` | 0.2.0 | cuBLAS bindings | LOW (cudarc covers this) |

**Registry integration:** Add discovered SDKs as new components in `nvidia-software-registry.json` with `"status": "recommended"` field to distinguish from installed components.

## Agent Coordination

| Agent | Ownership | Files |
|-------|-----------|-------|
| **Claude** | Module + config + tests | `Modules/PC-AI.Gpu/`, `Config/nvidia-software-registry.json`, `Tests/Unit/PC-AI.Gpu.Tests.ps1` |
| **Codex** | Build integration + smoke | `Build.ps1`, `Tools/Update-NvidiaSoftware.ps1`, `Tools/Invoke-JanusGpuSmoke.ps1` |
| **Shared** | Driver registry | `Config/driver-registry.json` (coordinate via agent-bus) |

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Separate module vs extend PC-AI.Drivers | Separate `PC-AI.Gpu` | PnP drivers vs SDK toolkits are different abstractions |
| Separate config vs extend driver-registry | Separate `nvidia-software-registry.json` | Different schema (envVars, compatibility matrix, header parsing) |
| GPU driver placement | Both registries | PnP entry in driver-registry + detailed entry in nvidia-software-registry |
| Replace or extend Initialize-CudaEnvironment | Extend by delegation | Backward compat — new function calls existing, then adds cuDNN/TensorRT |
| Version detection method | Parse version.json + C headers | More reliable than filename scraping |
| NCCL support | Deferred | Linux-only, not applicable to Windows |
| Download URL construction | Manual curation in registry | NVIDIA URLs require sessions/EULA; not programmatically constructable |
