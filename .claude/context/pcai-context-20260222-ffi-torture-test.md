# PC_AI Context: FFI Torture Test & 4-Phase Completion

**Context ID:** ctx-pcai-20260222
**Created:** 2026-02-22
**Branch:** main @ b0f763f

## State Summary

All 4 phases of the FunctionGemma + FFI validation plan are complete:
- Phase A: Fixed CUDA 13.1 nvcc compatibility by using CUDA 12.6 with `.cargo/config.toml` env overrides
- Phase B: Built all 3 FunctionGemma crates (core, runtime, train), trained LoRA adapter on GPU 0 (RTX 2000 Ada), ran eval
- Phase C: Launched FunctionGemma runtime server on port 8000, built C# TUI, verified ReAct mode with tool routing loop
- Phase D: Rebuilt pcai_inference.dll with full FFI exports (15 functions), ran 49 integration + stress tests (43 pass, 0 fail, 6 skip)

## Recent Changes

| File | Change |
|------|--------|
| `.cargo/config.toml` | CUDA 12.6 env vars for nvcc compatibility |
| `Native/pcai_core/pcai_inference/src/ffi/mod.rs` | Fixed GlobalState destructuring, repr(C) test alignment |
| `Native/PcaiNative/InferenceModule.cs` | Fixed yield-in-try-catch (CS1626), refactored to List |
| `Native/PcaiNative/PowerShellHost.cs` | CreateDefault -> CreateDefault2 (snap-in fix) |
| `Tests/Integration/FFI.Inference.Tests.ps1` | Fixed error code expectations, deploy paths, skip patterns |
| `Tests/Integration/FFI.Stress.Tests.ps1` | NEW: 21-test FFI torture suite |
| `Tests/Helpers/TestHelpers.psm1` | Fixed DeployDir path |
| `Modules/PcaiInference.psm1` | Fixed string interpolation ($configPath: -> ${configPath}:) |
| `Config/pcai-functiongemma.json` | router_gpu=0, cuda_visible_devices=[0] |
| `Tools/Set-CudaBuildEnv.ps1` | NEW: CUDA build env helper |

## Key Decisions

### dec-001: CUDA Version Strategy
- **Decision:** Use CUDA 12.6 via .cargo/config.toml env overrides instead of CUDA 13.1
- **Rationale:** nvcc 13.1 has fatal preprocessor bug without -ccbin flag; Rust build crates don't support passing -ccbin; CUDA 12.6 works for SM 75-89
- **Date:** 2026-02-22

### dec-002: FFI DLL Build Strategy
- **Decision:** Build FFI-only DLL (no backend) for torture testing, separate from full backend builds
- **Rationale:** 289KB vs 22MB, faster build, tests FFI layer independently
- **Date:** 2026-02-22

### dec-003: GPU Assignment
- **Decision:** GPU 0 (RTX 2000 Ada, 8GB, sm_89) for inference/runtime, GPU 1 (RTX 5060 Ti, 16GB, sm_120) for training
- **Rationale:** RTX 5060 Ti OOMs on candle_qmatmul inference but has enough VRAM for QLoRA training
- **Date:** 2026-02-22

## Test Results

### FFI Integration Tests (28 tests)
- 23 pass, 0 fail, 5 skip (backend-specific)

### FFI Stress Tests (21 tests)
- 20 pass, 0 fail, 1 skip
- 100 init/shutdown cycles: 12ms, no crash
- 8 threads x 100 mixed ops: 0 exceptions
- 200 cycle memory: 0.02MB growth
- 1000 error hammering: consistent codes
- 500/500 error strings valid UTF-8

### Rust FFI Unit Tests
- 13/13 pass

## Build Artifacts

| Artifact | Location | Size |
|----------|----------|------|
| pcai_inference.dll (FFI-only) | bin/pcai_inference.dll | 289KB |
| pcai_inference.dll (preserved) | .pcai/build/artifacts/pcai-inference-ffi/ | 289KB |
| FunctionGemma runtime | Deploy/rust-functiongemma-runtime/target/release/ | ~50MB |
| FunctionGemma training | Deploy/rust-functiongemma-train/target/release/ | ~50MB |
| LoRA adapter | output/functiongemma-lora/ | ~1MB |

## Agent Registry

| Agent | Task | Files | Status |
|-------|------|-------|--------|
| rust-pro | Phase A: CUDA fix | .cargo/config.toml, Set-CudaBuildEnv.ps1 | Complete |
| rust-pro | Phase B: Build FunctionGemma | Deploy/rust-functiongemma-*/ | Complete |
| csharp-pro | Phase C: TUI fixes | PowerShellHost.cs, InferenceModule.cs | Complete |
| test-automator | Phase D: FFI torture tests | FFI.Stress.Tests.ps1, FFI.Inference.Tests.ps1 | Complete |

## Patterns

- **FFI error codes:** Negative integers (InvalidInput=-3, NotInitialized=-2, Unknown=-1)
- **Thread safety:** Rust Mutex<GlobalState> protects all FFI state, safe for concurrent P/Invoke
- **String ownership:** pcai_generate returns heap-allocated string, caller must pcai_free_string; pcai_last_error returns thread-local static
- **Feature gating:** `#[cfg(feature = "ffi")]` on ffi module, `#[cfg(feature = "llamacpp")]` on backend
- **DLL resolver:** C# NativeLibrary.SetDllImportResolver walks config paths then hardcoded fallbacks

## Recommended Next Agents

1. **code-reviewer**: Review the 13 modified files before committing
2. **security-auditor**: Audit FFI boundary for memory safety (null checks, string lifetimes)
3. **performance-engineer**: Profile FFI call overhead with actual model loaded
