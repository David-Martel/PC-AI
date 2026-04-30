# PC_AI Context: CUDA/LLVM Toolchain Integration & Media Pipeline Hardening

**Context ID:** ctx-pcai-20260303-toolchain
**Created:** 2026-03-03
**Branch:** main @ a8c2d00
**Version:** 0.1.0.215+a8c2d00 (215 commits, no git tags)
**Supersedes:** ctx-pcai-20260222b-tdd-cleanup

---

## Session Summary

Major toolchain integration session completing three workstreams:

1. **CUDA/LLVM toolchain auto-detection** -- CargoTools module enhanced to detect GPU, CUDA, MSVC, LLVM, and Windows SDK, then generate machine-specific `.cargo/config.toml` files
2. **Media pipeline hardening** -- pcai-media Janus-Pro pipeline completed with rand-based RNG, SigLIP vision model, async FFI, and RealESRGAN upscale module
3. **Build infrastructure** -- `.cargo/config.toml` files are now gitignored (machine-specific), template files committed, LLVM lld-link is the preferred linker

All changes committed in a single 26-file commit: `a8c2d00`.

---

## Project Overview

PC_AI is a local-first LLM-powered Windows diagnostics and optimization agent. It combines PowerShell orchestration (12 modules), Rust native acceleration (5 crates), C# P/Invoke interop, and local LLM inference. Architecture: collect -> parse -> route -> reason -> recommend. Safety-first: read-only by default.

### Component Map

```
PC-AI.ps1 (CLI entry, 2066 lines)
  |
  +-- Modules/ (12 PowerShell modules)
  |     PC-AI.Hardware, PC-AI.LLM, PC-AI.Acceleration, PC-AI.Evaluation,
  |     PC-AI.Virtualization, PC-AI.Network, PC-AI.USB, PC-AI.Cleanup,
  |     PC-AI.Performance, PC-AI.CLI, PC-AI.Common
  |     PcaiInference.psm1 (805 lines), PcaiMedia.psm1 (8 exports)
  |
  +-- Native/pcai_core/ (Rust workspace, 5 crates)
  |     pcai_core_lib     -- shared: fs, search, telemetry, perf, system
  |     pcai_inference     -- dual-backend: llamacpp + mistralrs, HTTP + FFI
  |     pcai_media_model   -- Janus-Pro: VQ-VAE, GenHead, config, tensor_utils
  |     pcai_media         -- pipeline: generate, understand, upscale, hub, FFI
  |     pcai_media_server  -- Axum HTTP: /health, /v1/images/generate
  |
  +-- Native/PcaiNative/ (C# .NET 8, 18 .cs files)
  |     InferenceModule.cs, MediaModule.cs, NativeResolver.cs, Models.cs
  |
  +-- Deploy/rust-functiongemma-*/ (tool router: core/runtime/train)
  |
  +-- Build.ps1 (unified orchestrator, 2077 lines)
```

---

## What Changed This Session (a8c2d00)

### Commit: feat: CUDA 13.1 restore, media hardening, LLVM toolchain support

**26 files changed, +1616 / -381 lines**

#### Media Pipeline Hardening
| File | Change |
|------|--------|
| `pcai_media/src/generate.rs` | Replaced hash-based RNG with `rand` crate for proper stochastic sampling |
| `pcai_media/src/understand.rs` | Wired SigLIP VisionModel (candle_transformers::models::siglip) replacing placeholder |
| `pcai_media/src/ffi/mod.rs` | Added async FFI: pcai_media_generate_image_async, poll_result, cancel (+483 lines) |
| `pcai_media/src/upscale.rs` | NEW -- RealESRGAN upscale module (ort 2.0.0-rc.11, behind `upscale` feature flag) |
| `pcai_media/src/lib.rs` | Added `pub mod upscale` |
| `pcai_media/Cargo.toml` | Added rand, ort dependencies and upscale feature |
| `pcai_media_model/src/lib.rs` | Minor fixes |
| `pcai_media_server/src/main.rs` | Minor update |

#### C# Interop Layer
| File | Change |
|------|--------|
| `NativeResolver.cs` | NEW -- centralized DLL path resolution (replaces per-module logic) |
| `InferenceModule.cs` | Refactored to use NativeResolver (-118 lines) |
| `MediaModule.cs` | Updated P/Invoke for all 16 FFI exports |
| `Models.cs` | Minor update |

#### PowerShell Layer
| File | Change |
|------|--------|
| `PcaiMedia.psm1` | Added 8 exported functions (+151 lines) |
| `PcaiMedia.psd1` | Manifest: 8 functions exported |

#### Build and Config
| File | Change |
|------|--------|
| `.gitignore` | Added `.cargo/config.toml` patterns |
| `Native/pcai_core/.cargo/config.toml.template` | NEW -- workspace-level template |
| `Native/pcai_core/pcai_inference/.cargo/config.toml.template` | NEW -- inference-specific template |
| `Native/pcai_core/pcai_inference/.cargo/config.toml` | DELETED (now gitignored) |
| `Native/pcai_core/pcai_inference/Invoke-PcaiBuild.ps1` | BF16 WMMA fix for SM 7.5, linker fixes |
| `Native/pcai_core/Cargo.lock` | Updated deps (+110 lines) |
| `Build.ps1` | Minor update |
| `Config/llm-config.json` | Updated model paths |
| `Config/pcai-functiongemma.json` | Updated model paths |
| `Config/pcai-media.json` | Updated pipeline config |

#### New Tooling
| File | Change |
|------|--------|
| `Tools/Install-LlvmFromSource.ps1` | NEW -- LLVM/LLD source build script |

---

## Key Decisions

### dec-001: .cargo/config.toml is machine-specific
- **Decision:** Gitignore all `.cargo/config.toml` files; commit `.cargo/config.toml.template` for reference
- **Rationale:** These files contain machine-specific paths (CUDA_PATH, MSVC bin dir, linker path). CargoTools `Initialize-ProjectCargoConfig` regenerates them from detected toolchain.
- **Template placeholders:** `{{SCCACHE}}`, `{{LINKER}}`, `{{MSVC_VERSION}}`, `{{MSVC_BIN_DIR}}`, `{{GPU_NAME}}`, `{{COMPUTE_CAP}}`, `{{CUDA_PATH}}`

### dec-002: LLVM lld-link as preferred linker
- **Decision:** Linker preference order: LLVM lld-link.exe > MSVC link.exe > bundled rust-lld
- **Rationale:** lld-link is significantly faster for large Rust binaries. Requires LIB env var set to MSVC + SDK lib paths (3 paths: MSVC lib, SDK um, SDK ucrt).
- **Caveat:** Must set `$env:LIB` when using lld-link because it does not auto-discover MSVC/SDK library directories like MSVC link.exe does.

### dec-003: CUDA_PATH restored to v13.1
- **Decision:** Restored CUDA_PATH to v13.1 after brief testing with v12.9
- **Issue:** NVIDIA driver 576.57 supports max CUDA 12.9 at runtime. Toolkit 13.1 is fine for compilation (nvcc) but CUDA runtime calls will fail until driver is updated.
- **Next step:** Update NVIDIA driver to 13.1-compatible version

### dec-004: RealESRGAN behind feature flag
- **Decision:** RealESRGAN upscale module gated behind `upscale` Cargo feature
- **Rationale:** Depends on ort 2.0.0-rc.11 which downloads onnxruntime.dll at build time. CRT mismatch in debug builds (works in release). Keep optional until ort stabilizes.

### dec-005: NativeResolver.cs centralizes DLL resolution
- **Decision:** Extracted DLL path resolution from InferenceModule.cs into shared NativeResolver.cs
- **Rationale:** MediaModule.cs needs the same resolution logic. Single place for bin/, build output, and PATH-based DLL discovery.

---

## CargoTools Enhancements (External Module)

CargoTools is a separate PowerShell module (not in PC_AI repo) that manages Rust build environments.

### New Detection Functions
| Function | Purpose |
|----------|---------|
| `Get-GpuInfo` | NVIDIA GPU detection (name, VRAM, compute capability, driver) |
| `Get-CudaToolkitInfo` | CUDA toolkit version, nvcc path, include/lib dirs |
| `Get-MsvcInfo` | MSVC compiler version, bin dir, lib paths |
| `Get-LlvmInfo` | LLVM/Clang/LLD detection, version, bin path |
| `Get-WindowsSdkInfo` | Windows SDK version, include/lib paths |

### New Public Function
| Function | Purpose |
|----------|---------|
| `Initialize-ProjectCargoConfig` | Generates `.cargo/config.toml` from detected toolchain state |

### LIB Environment Variable
When lld-link is the active linker, CargoTools auto-sets `$env:LIB` with three paths:
1. MSVC lib directory (e.g., `C:\Program Files\Microsoft Visual Studio\2022\...\lib\x64`)
2. Windows SDK um library (e.g., `C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64`)
3. Windows SDK ucrt library (e.g., `...\ucrt\x64`)

---

## Current Build State

### Artifacts
| Component | Status | Size | Location |
|-----------|--------|------|----------|
| pcai-mistralrs.exe | BUILT | 84MB | .pcai/build/artifacts/ |
| PcaiNative.dll | BUILT | 91KB | bin/ |
| pcai_inference.dll | BUILT | 4.1MB | bin/ |
| pcai-llamacpp | INCOMPLETE | -- | artifact dir empty |
| FunctionGemma runtime | INCOMPLETE | -- | artifact dir empty |
| pcai_media.dll | NOT YET BUILT | -- | needs `cargo build -p pcai-media` |

### Rust Test Counts (195+ total)
| Crate | Tests | Notes |
|-------|-------|-------|
| pcai-inference | 44 | lib tests with server+ffi features |
| pcai-media | 59 | generate, understand, hub, FFI |
| pcai-media-model | 31 | 29 pass + 2 ignored |
| pcai-media-server | 13 | HTTP endpoint tests |
| pcai_core_lib | 50 | fs, search, telemetry, perf |
| **Total** | **197** | |

### Downloaded Models (8.5GB)
| Model | Size | Format | Purpose |
|-------|------|--------|---------|
| Janus-Pro-1B | 3.9GB | safetensors | Image gen/understand |
| Qwen2.5-3B-Instruct | 1.8GB | GGUF Q4_K_M | LLM diagnostics |
| TinyLlama 1.1B | 638MB | GGUF Q4_K_M | Fast testing |
| FunctionGemma 270M | 253MB | GGUF Q4_K_M | Tool router |
| RealESRGAN x4 | 69.5MB | ONNX | Image upscale |

---

## GPU and Toolchain State

### GPUs
| GPU | VRAM | Compute | SM | Role |
|-----|------|---------|----|------|
| Quadro RTX 4000 | 8GB | 7.5 | sm_75 (Turing) | Inference / runtime (GPU 0) |
| RTX 5060 Ti | 16GB | 12.0 | sm_120 (Blackwell) | Training / QLoRA (GPU 1) |

Note: MEMORY.md lists "RTX 2000 Ada (SM 89)" but the user specified "Quadro RTX 4000 (SM 75)" for this session. The GPU configuration may have been updated or clarified.

### Toolchain
| Component | Version | Path |
|-----------|---------|------|
| CUDA Toolkit | 13.1 | $env:CUDA_PATH |
| NVIDIA Driver | 576.57 | supports max CUDA 12.9 runtime (mismatch!) |
| MSVC | VS 2022 | Auto-detected by CargoTools |
| LLVM/LLD | Installed | lld-link.exe preferred linker |
| Rust | stable | rustup managed |
| sccache | Installed | RUSTC_WRAPPER |

### CUDA Driver Mismatch (Active Blocker)
- **Toolkit:** CUDA 13.1 (compiles fine with nvcc)
- **Driver:** 576.57 (runtime supports max CUDA 12.9)
- **Impact:** Compilation succeeds. Runtime CUDA calls will fail until driver is updated.
- **Fix:** Update NVIDIA driver to version supporting CUDA 13.1 runtime

---

## Active Blockers

| Blocker | Severity | Status |
|---------|----------|--------|
| CUDA driver 576.57 vs toolkit 13.1 mismatch | HIGH | Pending driver update |
| llamacpp backend build incomplete | MEDIUM | Artifact dir empty |
| FunctionGemma runtime build incomplete | MEDIUM | Artifact dir empty |
| bindgen_cuda panic in mistral.rs on Windows | MEDIUM | Workaround: CPU-only build |
| protobuf CVE (Dependabot #1) | HIGH | No patch available |
| sccache + ring crate incompatibility | LOW | Workaround: disable sccache for ring |

---

## Known Issues and TODOs

### Immediate
- Update NVIDIA driver for CUDA 13.1 runtime compatibility
- Build pcai_media.dll and verify FFI exports
- Run integration tests with actual Janus-Pro model weights (now downloaded)
- Complete llamacpp backend build

### Technical Debt
- Help coverage: 48.2% (57 of 110 functions missing help)
- PC-AI.Acceleration: 0 test files
- No git tags (version is commit-count based)
- RealESRGAN CRT mismatch in debug builds (release works)
- Large-context offload for pcai-inference not implemented
- Versioned C ABI contract not formalized
- Centralized error translation (Rust -> C# -> PowerShell) incomplete
- Cancellation/timeout propagation across layers incomplete

---

## Patterns and Conventions

### FFI Pattern
```
Rust extern "C" --> C# P/Invoke (PcaiNative.dll) --> PowerShell wrapper (.psm1)
```
DLL resolution: NativeResolver.cs checks bin/, build output, then PATH.

### Build Pattern
```powershell
.\Build.ps1 -Component <name> [-EnableCuda] [-Clean] [-Package]
```
Components: llamacpp, mistralrs, inference, media, tui, native, functiongemma, nukenul, servicehost, all.

### Config Generation Pattern
```powershell
# CargoTools generates machine-specific .cargo/config.toml from toolchain detection
Initialize-ProjectCargoConfig -Path 'Native/pcai_core' -Force
Initialize-ProjectCargoConfig -Path 'Native/pcai_core/pcai_inference' -Force
```
Templates committed as `.cargo/config.toml.template`. Actual configs are gitignored.

### Testing Pattern
```powershell
# Rust unit tests (no backends needed)
cd Native\pcai_core\pcai_inference
cargo test --no-default-features --features server,ffi --lib

# All crate tests
cd Native\pcai_core
cargo test --workspace

# PowerShell tests
Invoke-Pester Tests/
```

---

## Commit History (Recent)

| Hash | Date | Description |
|------|------|-------------|
| a8c2d00 | 2026-03-03 | CUDA 13.1 restore, media hardening, LLVM toolchain support |
| 65185b4 | 2026-03-02 | Complete FFI and HTTP API surface |
| aa74957 | 2026-03-02 | Build integration, config, FFI device fix, design docs |
| 81027bb | 2026-03-02 | C# P/Invoke wrapper MediaModule.cs |
| 6e6516a | 2026-03-02 | PcaiMedia.psm1 PowerShell FFI wrapper |
| 806702b | 2026-03-02 | Scaffold pcai_media_server HTTP server |
| 36e67d2 | 2026-03-02 | FFI C ABI layer for pcai_media |
| de370d1 | 2026-03-02 | UnderstandingPipeline (image-to-text) |
| 67b4483 | 2026-03-02 | GenerationPipeline (text-to-image with CFG) |
| 83183fa | 2026-03-02 | hub.rs model download and weight loading |

---

## Recommended Next Steps

1. **Update NVIDIA driver** to support CUDA 13.1 runtime
2. **Build pcai_media.dll** and run `cargo test -p pcai-media --lib` to verify 59 tests
3. **Run integration test** with Janus-Pro-1B model (weights at Models/Janus-Pro-1B/)
4. **Complete llamacpp backend build** (artifact dir currently empty)
5. **Tag first release** (v0.1.0 or v0.2.0) to enable proper semantic versioning
6. **Regenerate .cargo/config.toml** files via `Initialize-ProjectCargoConfig` after driver update
7. **Address help coverage** (48.2% -- 57 functions missing help)
