---
context_id: ctx-pcai-20260318-nvidia-complete
created_at: 2026-03-18T20:00:00Z
created_by: claude-opus-4.6
schema_version: "2.0"
---

# PC_AI Context — 2026-03-18: NVIDIA Framework Complete + Jules Integration

## Summary

Two-session mega-build: complete NVIDIA software installer framework (PC-AI.Gpu module, 19 files), Rust NVML GPU monitoring (nvml-wrapper 0.12), CUDA device selection with multi-GPU fallback, driver auto-discovery, 10+ Jules sessions answered and patches applied via REST API, RealESRGAN ONNX model conversion, PcaiChatTui C# test project, comprehensive code reviews with all CRITICAL/HIGH fixes applied. 15 commits pushed to origin, 50 Jules sessions completed. Local and remote fully synchronized at 77b0cc7.

## Project State

- **Branch**: `main` @ `77b0cc7`
- **Remote**: Fully synchronized with `origin/main`
- **Working tree**: Clean (no uncommitted changes)
- **Jules**: 50/50 sessions completed, 0 active

## Commits This Session (15)

| SHA | Description |
|-----|-------------|
| 34123d0 | feat(gpu): add PC-AI.Gpu NVIDIA software management module (24 files) |
| 4b20b37 | feat(rust): add NVML GPU monitoring module via nvml-wrapper 0.12 |
| dd5cd79 | fix(gpu): fix operator precedence in nvidia-smi args + format string crash |
| 5733e83 | fix(security): replace Invoke-Expression with safe netsh invocation (Jules) |
| f207c0e | refactor(evaluation): externalize safety test cases to JSON dataset (Jules) |
| e0cdac2 | feat(media): cuda:auto GPU fallback + minor Rust fixes across crates |
| 8d3b818 | fix(gpu): review fixes for PC-AI.Gpu module (Phases 2-3) |
| 0441993 | feat(media): PcaiMedia PowerShell module + media test suites |
| 291989a | feat(tools): add Janus GPU smoke test + NVIDIA update orchestrator |
| 4703cc8 | docs: update CLAUDE.md with CUDA guidance + Build.ps1 media integration |
| 738c4c3 | perf(rust): eliminate unnecessary string clones + upscale tests (Jules) |
| a29d07a | feat: Jules batch - C# test project, media tests, refactors, ONNX converter |
| c6f725c | feat(media): add RealESRGAN ONNX conversion tool + model |
| 22c7e0a | chore(deps): bump System.Text.Json 8.0.6 -> 10.0.5 (Dependabot) |
| 77b0cc7 | test(media): add Initialize-PcaiMediaFFI unit tests (Jules) |

## Key Deliverables

### 1. PC-AI.Gpu PowerShell Module (Phases 1-3 COMPLETE)
- 19 files: 8 public + 9 private functions, psd1, psm1
- Public: Get-NvidiaGpuInventory, Get-NvidiaSoftwareRegistry, Get-NvidiaSoftwareStatus, Get-NvidiaGpuUtilization, Get-NvidiaCompatibilityMatrix, Initialize-NvidiaEnvironment, Install-NvidiaSoftware, Update-NvidiaSoftwareRegistry
- 75 Pester 5 unit tests
- nvidia-software-registry.json v1.2.0 (11 components, 2 GPUs)
- driver-registry.json v1.2.0 (13 devices, nvidia-gpu-driver shared group)
- Tools/Sync-NvidiaDriverVersion.ps1 (811 lines, auto-detect + compare + download + install)

### 2. Rust NVML GPU Module
- pcai_core_lib/src/gpu/mod.rs (508 lines): OnceLock singleton, 6 public functions
- 3 FFI exports: pcai_gpu_count, pcai_gpu_info_json, pcai_driver_version
- nvml-wrapper 0.12 with feature gate, cargo check passed

### 3. CUDA Device Selection Fix
- cuda:auto iterates all GPUs descending by VRAM with graceful fallback
- UUID-based CUDA_VISIBLE_DEVICES + CUDA_DEVICE_ORDER=PCI_BUS_ID
- cudarc 0.19.0 -> 0.19.3

### 4. Jules Integration (via REST API)
- 17 sessions unblocked via sendMessage + approvePlan API calls
- 10 patches applied from completed sessions
- PcaiChatTui.Tests xUnit project created
- INativeInferenceModule interface for testable P/Invoke
- Find-DuplicateFiles refactored into helpers
- Security fix: Invoke-Expression -> safe netsh invocation

### 5. RealESRGAN ONNX
- Convert-RealESRGAN-to-ONNX.py (self-contained, inline RRDBNet, opset 17)
- RealESRGAN_x4.onnx generated (63.9 MB, dynamic H/W axes)
- Compatible with ort 2.0.0-rc.11 in Rust upscale pipeline

## Code Reviews Applied (28 findings total)
- 1 CRITICAL: cross-module Invoke-TrustedDownload dependency (fixed: inline download)
- 8 HIGH: hardcoded CUDA_COMPUTE_CAPS, non-atomic JSON write, Process handle leak, deprecated WebClient, NVML re-init per call, *const vs *mut FFI, expensive gpu_count, unsafe env var mutations
- All CRITICAL/HIGH fixed and verified

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Separate PC-AI.Gpu module | Not extend PC-AI.Drivers | PnP drivers vs SDK toolkits are different abstractions |
| nvml-wrapper 0.12.0 | Over nvidia/nvml-rs | 3.4M downloads, Windows CI, libloading dynamic loading |
| Skip trtx/video-codec-sdk | cudarc version conflicts | ^0.11 vs ^0.19 incompatible |
| Jules API over web UI | REST API with API key | Programmatic access to session questions and plan approval |
| RealESRGAN legacy ONNX export | dynamo=False | torch 2.10 dynamo exporter has Unicode crash on Windows |

## CLAUDE.TODO.md Status

| Phase | Status | Key Items |
|-------|--------|-----------|
| Phase 1: Detection | COMPLETE | 19 module files, registry, tests |
| Phase 2: Environment | COMPLETE | Initialize-NvidiaEnvironment, CompatibilityMatrix, Backup |
| Phase 3: Install | COMPLETE | Download + SHA256 + silent install + registry update |
| Phase 4: CI/Automation | ~50% | nvidia-smi consolidation done; CI workflows pending |
| Phase 5: SDKs | ~15% | nvml-wrapper done, ONNX model done; Nsight Graphics, Warp, RAPIDS pending |

## Agent Registry

| Agent | Task | Status |
|-------|------|--------|
| nvidia-plan (Plan) | Architecture design | Complete |
| gpu-module (powershell-pro) | Module skeleton + all functions | Complete |
| gpu-tests (powershell-pro) | 75 Pester tests | Complete |
| nvidia-registry (powershell-pro) | Config with real data | Complete |
| phase2-env (powershell-pro) | Env mgmt + compat matrix + backup | Complete |
| phase3-install (powershell-pro) | Install framework (4 files) | Complete |
| driver-sync (powershell-pro) | Driver auto-discovery + registry | Complete |
| phase5-nvml (rust-pro) | NVML Rust integration | Complete (build passed) |
| nvidia-sdk-research (search) | SDK evaluation (20+ SDKs) | Complete |
| rust-gpu-research (rust-pro) | Crate ecosystem (7 categories) | Complete |
| review-ps-reliability (reviewer) | PS review (19 findings) | Complete |
| review-rust-safety (reviewer) | Rust review (9 findings) | Complete |
| fix-critical-high (powershell-pro) | 5 PS CRITICAL/HIGH fixes | Complete |
| fix-rust-high (rust-pro) | 4 Rust HIGH fixes | Complete |
| test-installers (powershell-pro) | Live testing + bug discovery | Complete |
| find-gpu-integration (Explore) | 10 integration points mapped | Complete |
| todo-audit (Explore) | CLAUDE.TODO.md verification | Complete |
| onnx-model (general) | ONNX conversion research + script | Complete |
| Jules (external, 50 sessions) | Tests, refactors, security fixes | Complete |

## Roadmap

### Immediate
- Phase 4: GitHub workflow for NVIDIA stack validation
- Phase 5: Install Nsight Graphics 2025.5, Warp (pip)
- Wire NVML FFI into PowerShell GPU functions (replace nvidia-smi subprocess)

### This Week
- CUDA PATH ordering fix (v12.9 bin shadows v13.1 in PATH)
- Set CUDA_PATH_V13_2 for installed v13.2
- Integrate nvml-wrapper into pcai_media config.rs (replace all_nvidia_smi_gpus)

### Tech Debt
- 2 Jules patches skipped (merge conflicts): resolve_device tests, Initialize-PcaiMediaFFI tests
- Phase 3 Pester test coverage for install flows
- Setup-DevEnvironment.ps1 integration with Initialize-NvidiaEnvironment
