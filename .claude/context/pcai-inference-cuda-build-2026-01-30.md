# PC_AI Context: CUDA-Enabled Inference Build

**Context ID:** ctx-pcai-inference-20260130
**Created:** 2026-01-30T22:30:00-05:00
**Branch:** main @ ec2b5e1

## Summary

Successfully built the pcai-inference Rust library with CUDA-enabled llama.cpp backend. This session resolved multiple CMake/MSVC toolchain issues, CUDA environment configuration, and Rust API compatibility issues.

## Recent Changes

### Build Infrastructure
- `Deploy/pcai-inference/build-llamacpp-fixed.ps1` - Complete MSVC/CUDA build script with all fixes
- `Tools/Initialize-CudaEnvironment.ps1` - CUDA environment setup helper
- `Deploy/pcai-inference/test-cmake-direct.ps1` - CMake configuration test script
- `Deploy/pcai-inference/clean-llama-cache.ps1` - CMake cache cleanup utility

### Code Fixes
- `Deploy/pcai-inference/src/backends/llamacpp.rs` - Fixed `token_to_piece` → `token_to_str` API
- `Deploy/pcai-inference/src/ffi/mod.rs` - Added `#![allow(clippy::not_unsafe_ptr_arg_deref)]` for FFI
- `Deploy/pcai-inference/Cargo.toml` - Made CUDA feature optional

## Key Decisions

### dec-001: CMake Pre-Configuration Strategy
- **Decision:** Pre-configure CMake before cargo build to bypass cmake-rs limitations
- **Rationale:** cmake-rs doesn't pass environment variables (LLAMA_CURL, LLAMA_BUILD_TESTS) to CMake properly
- **Alternative:** Fork llama-cpp-sys-2 crate (rejected - too invasive)

### dec-002: Disable CURL in llama.cpp
- **Decision:** Build with `-DLLAMA_CURL=OFF`
- **Rationale:** CURL is for model downloading, not needed for local inference
- **Alternative:** Configure vcpkg toolchain (attempted but FindCURL failed)

### dec-003: CUDA Architecture Targets
- **Decision:** Target sm_75 through sm_89 for broad compatibility
- **Rationale:** User's GPUs are RTX 2000 Ada (sm_89) and RTX 5060 Ti (sm_120)
- **Setting:** `CUDAARCHS=75;80;86;89`

## Technical Details

### Build Environment Requirements
```powershell
# Required environment variables (set by build-llamacpp-fixed.ps1)
CC = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64\cl.exe"
CXX = $CC
CMAKE_GENERATOR = "Ninja"
CMAKE_MAKE_PROGRAM = "$env:USERPROFILE\.local\bin\ninja.exe"
CMAKE_RC_COMPILER = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\rc.exe"
GGML_CUDA = "ON"
CMAKE_CUDA_COMPILER = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe"
```

### CMake Options for llama.cpp
```cmake
-DLLAMA_CURL=OFF           # Disable CURL (not needed)
-DLLAMA_BUILD_TESTS=OFF    # Tests not in vendored source
-DLLAMA_BUILD_EXAMPLES=OFF # Examples not in vendored source
-DLLAMA_BUILD_TOOLS=OFF    # Tools not in vendored source
-DGGML_CUDA=ON             # Enable CUDA backend
-DBUILD_SHARED_LIBS=ON     # Build DLLs
```

### Known Issues & Workarounds

1. **CL Environment Variable Pollution**
   - Issue: User-level CL env var set to `/MP`, Git Bash translates to `C:/Program Files/Git/MP`
   - Fix: Script clears CL, _CL_, LINK, _LINK_ at start

2. **cmake-rs Cache Detection**
   - Issue: CMakeCache.txt created before configuration completes, cmake-rs thinks done
   - Fix: Pre-configure CMake manually, only clean incomplete configs (CMakeCache without build.ninja)

3. **llama-cpp-2 API Mismatch**
   - Issue: `token_to_piece` doesn't exist in llama-cpp-2 v0.1.132
   - Fix: Use `token_to_str(token, Special::Tokenize)` instead

## Build Outputs

```
T:\RustCache\cargo-target\release\
├── pcai_inference.dll      (360KB) - Main FFI library
├── pcai_inference.dll.lib  (3KB)   - Import library
└── pcai_inference.pdb      (1.3MB) - Debug symbols

T:\RustCache\cargo-target\release\build\llama-cpp-sys-2-*\out\build\bin\
├── ggml-cuda.dll  (46MB) - CUDA backend
├── llama.dll      (2MB)  - llama.cpp core
├── ggml-base.dll  (543KB)
├── ggml-cpu.dll   (785KB)
└── ggml.dll       (68KB)
```

## Files Modified This Session

### Core Changes
- `Deploy/pcai-inference/src/backends/llamacpp.rs` - API fix
- `Deploy/pcai-inference/src/ffi/mod.rs` - Clippy lint allowance
- `Deploy/pcai-inference/Cargo.toml` - Optional CUDA feature

### New Build Scripts
- `Deploy/pcai-inference/build-llamacpp-fixed.ps1` - Main build script
- `Tools/Initialize-CudaEnvironment.ps1` - CUDA setup
- `Deploy/pcai-inference/test-cmake-direct.ps1` - CMake testing
- `Deploy/pcai-inference/clean-llama-cache.ps1` - Cache cleanup

## Agent Work Registry

| Agent | Task | Files | Status | Notes |
|-------|------|-------|--------|-------|
| Claude | CUDA build debugging | build-llamacpp-fixed.ps1 | Complete | Full MSVC/CUDA toolchain |
| Claude | API compatibility fix | llamacpp.rs | Complete | token_to_str migration |
| Claude | FFI clippy fixes | ffi/mod.rs | Complete | Allow raw pointer deref |

## Recommended Next Steps

1. **test-automator**: Write tests for CUDA inference path
2. **deployment-engineer**: Package DLLs for distribution
3. **security-auditor**: Review FFI boundary for memory safety

## Validation

- **Build Status:** SUCCESS
- **Last Built:** 2026-01-30T22:18:35-05:00
- **CUDA Verified:** Yes (sm_89, sm_120a targets)
- **Tests:** Pending
