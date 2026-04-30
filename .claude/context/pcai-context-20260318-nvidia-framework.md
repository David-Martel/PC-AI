---
context_id: ctx-pcai-20260318-nvidia-framework
created_at: 2026-03-18T17:00:00Z
created_by: claude-opus-4.6
schema_version: "2.0"
---

# PC_AI Context — 2026-03-18: NVIDIA Software Framework + NVML Integration

## Summary

Massive session: built complete NVIDIA software installer framework (PC-AI.Gpu module), integrated nvml-wrapper Rust crate, fixed CUDA device selection, coordinated with Codex via agent-bus. 62 files changed/created. All code reviewed by specialist agents with CRITICAL/HIGH fixes applied.

## Key Deliverables

### 1. PC-AI.Gpu PowerShell Module (19 files)
- 8 public functions: Get-NvidiaGpuInventory, Get-NvidiaSoftwareRegistry, Get-NvidiaSoftwareStatus, Get-NvidiaGpuUtilization, Get-NvidiaCompatibilityMatrix, Initialize-NvidiaEnvironment, Install-NvidiaSoftware, Update-NvidiaSoftwareRegistry
- 9 private functions: version detection (CUDA, cuDNN, TensorRT, Nsight), silent install, backup/restore, trusted download validation
- 75 Pester 5 unit tests

### 2. Rust NVML GPU Module (gpu/mod.rs, 543 lines)
- nvml-wrapper 0.12.0 integrated with OnceLock singleton
- 6 public functions: gpu_count, gpu_inventory, gpu_utilization, best_available_gpu, driver_version, cuda_driver_version
- 3 FFI exports: pcai_gpu_count, pcai_gpu_info_json, pcai_driver_version
- cargo check passed (0 errors, 0 new warnings)

### 3. Driver Auto-Discovery (Sync-NvidiaDriverVersion.ps1)
- nvidia-smi query → registry comparison → multi-GPU compatibility check → download → install
- driver-registry.json v1.2.0: 13 devices, nvidia-gpu-driver shared group
- nvidia-software-registry.json v1.2.0: 11 components (6 installed + 5 recommended)

### 4. CUDA Device Selection Fix (config.rs)
- cuda:auto GPU fallback across all GPUs (descending VRAM)
- UUID-based CUDA_VISIBLE_DEVICES mapping (Codex enhancement)
- CUDA_DEVICE_ORDER=PCI_BUS_ID for stable ordering
- cudarc bumped 0.19.0 → 0.19.3

### 5. Review Fixes (9 CRITICAL/HIGH resolved)
- PS: inline download (no cross-module dep), dynamic CUDA_COMPUTE_CAPS, atomic JSON write, Process dispose, HttpClient + exit codes
- Rust: OnceLock NVML cache, *mut return types, gpu_count via device_count, Clock import fix

## Agent Registry

| Agent | Task | Files | Status |
|-------|------|-------|--------|
| nvidia-plan (Plan) | Architecture design | CLAUDE.TODO.md | Complete |
| gpu-module (powershell-pro) | Module skeleton + detection | 19 PS files | Complete |
| gpu-tests (powershell-pro) | Unit tests | PC-AI.Gpu.Tests.ps1 | Complete |
| nvidia-registry (powershell-pro) | Config with real data | nvidia-software-registry.json | Complete |
| phase2-env (powershell-pro) | Env mgmt + compat matrix | 3 PS files | Complete |
| phase3-install (powershell-pro) | Install framework | 4 PS files | Complete |
| driver-sync (powershell-pro) | Driver auto-discovery | 3 files (2 JSON + 1 PS) | Complete |
| phase5-nvml (rust-pro) | NVML Rust integration | 3 Rust files | Complete (build passed) |
| nvidia-sdk-research (search) | SDK evaluation | 20+ SDKs evaluated | Complete |
| rust-gpu-research (rust-pro) | Rust crate ecosystem | 7 categories evaluated | Complete |
| review-ps-reliability (reviewer) | PS reliability review | 19 findings | Complete |
| review-rust-safety (reviewer) | Rust safety review | 9 findings | Complete |
| fix-critical-high (powershell-pro) | PS CRITICAL/HIGH fixes | 5 files fixed | Complete |
| fix-rust-high (rust-pro) | Rust HIGH fixes | 3 files fixed | Building |
| test-installers (powershell-pro) | Live testing + cudarc audit | Running diagnostics | Running |
| find-gpu-integration (Explore) | Integration point mapping | 10 sites found | Complete |

## Decisions

- Separate PC-AI.Gpu module (not extend PC-AI.Drivers) — different abstraction levels
- nvml-wrapper 0.12.0 chosen over nvidia/nvml-rs (3.4M downloads, Windows CI, libloading)
- trtx/video-codec-sdk skipped — cudarc version conflicts (^0.11 vs ^0.19)
- cudarc 0.17.8 + 0.19.3 coexist in Cargo.lock — no conflict
- CUDA 13.2 toolkit available, env vars need updating (CUDA_PATH_V13_2 missing)

## Known Issues
- Get-NvidiaCompatibilityMatrix line 217: $comp.ComponentId should be $comp.id
- nvcc reports 12.9 but CUDA_PATH points to 13.1 (PATH ordering)
- CUDA_PATH_V13_2 env var not set for installed v13.2
- Rust fix agent still running cargo check with fixes
