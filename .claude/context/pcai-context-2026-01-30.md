# PC_AI Project Context - 2026-01-30

> Context ID: ctx-pcai-20260130
> Created: 2026-01-30T19:30:00Z
> Branch: main @ ec2b5e1

## Project Overview

**PC_AI** is a local LLM-powered PC diagnostics and optimization agent for Windows 10/11 with WSL2 integration. The project has transitioned from Docker/vLLM/Ollama backends to a native **Rust-based dual-backend inference engine** (pcai-inference).

## Current State Summary

The project has completed a major architectural shift to native Rust inference:

1. **pcai-inference Rust crate** (`Deploy/pcai-inference/`) provides dual-backend support:
   - **llama-cpp-2**: GGUF model inference with CUDA GPU acceleration
   - **mistral.rs**: Alternative backend (CPU-only on Windows due to bindgen_cuda issues)

2. **HTTP Server** on port 8080 with OpenAI-compatible API:
   - `/health` - Server health check
   - `/v1/models` - Model listing
   - `/v1/chat/completions` - Chat completions with streaming

3. **FFI Layer** for direct PowerShell integration via P/Invoke:
   - `Modules/PcaiInference.psm1` - PowerShell FFI wrapper
   - DLL location: `T:\RustCache\cargo-target\release\pcai_inference.dll`

4. **CMake/MSVC Toolchain** for llama.cpp native compilation:
   - `Deploy/pcai-inference/cmake/toolchain-msvc.cmake`
   - VS 2022 Community with MSVC 14.44 detected

## Recent Changes (Last 10 Commits)

```
ec2b5e1 test(e2e): add inference end-to-end tests
92fde97 feat(tui): add InferenceBackend enum and inference tests
93cdbc1 chore(config): align LLM config with pcai-inference backend
d88aa9a feat(pcai-inference): enhance HTTP server with full OpenAI compatibility
cf20039 build(cmake): enhance MSVC toolchain with auto-detection
6e84e0c feat(virtualization): update service health and host for Rust backend
023bafb feat(tui): add native inference backend selection
f9b23e8 feat(pcai-inference): add OpenAI-compatible models endpoint
e9467f7 build(pcai-inference): add CMake toolchain for MSVC
a4bc48b docs(pc-ai): add comprehensive native inference integration guide
```

## Key Files Modified

### Rust Inference Engine
- `Deploy/pcai-inference/Cargo.toml` - Crate configuration with feature flags
- `Deploy/pcai-inference/src/backends/mod.rs` - InferenceBackend trait
- `Deploy/pcai-inference/src/backends/llamacpp.rs` - llama.cpp backend
- `Deploy/pcai-inference/src/backends/mistralrs.rs` - mistral.rs backend
- `Deploy/pcai-inference/src/http/mod.rs` - Axum HTTP server
- `Deploy/pcai-inference/src/ffi/mod.rs` - C FFI exports

### Build System
- `Deploy/pcai-inference/cmake/toolchain-msvc.cmake` - MSVC toolchain
- `Deploy/pcai-inference/CMakePresets.json` - CMake presets
- `Deploy/pcai-inference/build.ps1` - PowerShell build orchestrator
- `Deploy/pcai-inference/build-config.json` - Build configuration

### PowerShell Modules
- `Modules/PcaiInference.psm1` - Native FFI wrapper
- `Modules/PC-AI.LLM/PC-AI.LLM.psm1` - LLM module (updated for pcai-inference)
- `Modules/PC-AI.Virtualization/Public/Get-PcaiServiceHealth.ps1` - Health checks
- `Modules/PC-AI.Virtualization/Public/Invoke-PcaiServiceHost.ps1` - Server launcher

### Configuration
- `Config/llm-config.json` - Points to pcai-inference on port 8080

### Tests
- `Tests/Unit/PC-AI.Inference.Tests.ps1` - Unit tests
- `Tests/Integration/FFI.Inference.Tests.ps1` - FFI boundary tests
- `Tests/E2E/Inference.E2E.Tests.ps1` - End-to-end tests

## Architecture Decisions

### Decision 1: Dual-Backend Architecture
- **Topic**: LLM inference backend selection
- **Decision**: Support both llama-cpp-2 and mistral.rs backends via feature flags
- **Rationale**: llama-cpp-2 has broadest GGUF compatibility; mistral.rs adds multimodal support
- **Trade-offs**: Build complexity vs flexibility

### Decision 2: Native Rust vs Docker/WSL
- **Topic**: Inference deployment model
- **Decision**: Native Windows Rust binaries instead of Docker containers
- **Rationale**: Eliminates WSL/Docker dependency, reduces latency, simplifies deployment
- **Trade-offs**: Requires MSVC toolchain setup

### Decision 3: HTTP + FFI Hybrid
- **Topic**: PowerShell integration method
- **Decision**: Support both HTTP API (general use) and FFI (low-latency)
- **Rationale**: HTTP for interoperability; FFI for performance-critical paths
- **Trade-offs**: Two codepaths to maintain

## Known Issues / Blockers

1. **bindgen_cuda panic in mistral.rs**: Blocks CUDA support on Windows for mistral.rs backend
   - Workaround: Use CPU-only mode or llamacpp backend for GPU
   - Documentation: `T:\projects\rust-mistral\mistral.rs\CUDA_BUILD_BLOCKING_ISSUE.md`

2. **llamacpp MSVC requirement**: llama-cpp-sys-2 requires MSVC, not MinGW/GNU
   - Solution: CMake toolchain with vcvars64.bat environment setup
   - Status: Toolchain created, needs testing

3. **Archive cleanup**: Some files in `Archive/Deprecated-LLM-Backends/` couldn't be deleted (locked)

## Environment Configuration

### Build Requirements
- Rust 1.75+ with cargo
- Visual Studio 2022 Community (MSVC 14.44)
- CMake 3.20+
- Ninja build system
- CUDA 12.x (optional, for GPU acceleration)

### Key Paths
- Rust target: `T:\RustCache\cargo-target`
- DLL output: `T:\RustCache\cargo-target\release\pcai_inference.dll`
- VS 2022: `C:\Program Files\Microsoft Visual Studio\2022\Community`
- CUDA: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`

### Feature Flags
```toml
[features]
default = ["llamacpp", "server"]
llamacpp = ["dep:llama-cpp-2", "dep:encoding_rs"]
mistralrs-backend = ["dep:mistralrs", "dep:mistralrs-core"]
cuda = []
server = ["dep:axum", "dep:tower-http"]
ffi = []
```

## Agent Work Registry

| Agent | Task | Files | Status | Handoff |
|-------|------|-------|--------|---------|
| rust-pro | Implement pcai-inference crate | Deploy/pcai-inference/ | Complete | Ready for MSVC testing |
| powershell-pro | Update Virtualization module | Modules/PC-AI.Virtualization/ | Complete | - |
| code-reviewer | Review FFI boundary | src/ffi/mod.rs | Pending | - |
| test-automator | Create inference tests | Tests/**/Inference*.ps1 | Complete | - |

## Recommended Next Agents

1. **rust-pro**: Complete llamacpp backend MSVC build testing
2. **test-automator**: Run full test suite with real model
3. **security-auditor**: Review FFI memory safety
4. **deployment-engineer**: Create CI/CD pipeline for Rust builds

## Roadmap

### Immediate
- [ ] Test llamacpp build with MSVC toolchain
- [ ] Verify GPU offloading works with RTX GPUs
- [ ] Run inference tests with real GGUF model

### This Week
- [ ] Complete TUI backend selection UI
- [ ] Add model auto-detection from Ollama/LM Studio cache
- [ ] Implement streaming in FFI layer

### Tech Debt
- [ ] Clean up Archive/Deprecated-LLM-Backends/ folder
- [ ] Remove legacy Ollama/vLLM code paths
- [ ] Consolidate duplicate config handling

## Validation

- Last validated: 2026-01-30T19:30:00Z
- Git status: Clean (2 untracked workflow files)
- Tests: FFI tests need DLL built to run
