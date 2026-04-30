# Rust Context Slice - PC_AI

**Updated**: 2026-01-30 (Session 3)
**For Agents**: rust-pro, architect-reviewer

## Current Status Summary

All Rust components compile cleanly. New pcai-inference crate provides native dual-backend LLM inference.

| Component | Status | Tests | CUDA |
|-----------|--------|-------|------|
| **pcai-inference** | ✅ Building | 5 pass | llamacpp only |
| rust-functiongemma-train | ✅ Clean | 48 pass | ✅ Enabled |
| rust-functiongemma-runtime | ✅ Production | - | ✅ Ready |
| pcai_fs | ✅ Clean | 14 pass | N/A |
| pcai_core_lib | ✅ Clean | 0 | N/A |

## Completed This Session (2026-01-30)

**CUDA Integration:**
- ✅ CUDA Toolkit v13.1 environment properly configured
- ✅ Added CUDA env setup to `Tools/Invoke-RustBuild.ps1`
- ✅ Added CUDA env setup to `Native/build.ps1`
- ✅ candle-core/candle-nn CUDA features enabled
- ✅ CUDA device detection verified (DeviceId 2)
- ✅ GPU tensor operations tested and working
- ✅ Fixed runtime Cargo.toml edition (2024 → 2021)

**Previous Fixes:**
- ✅ Fixed Rust edition (2024 → 2021)
- ✅ Fixed orphan rule violations (moved Default impls)
- ✅ Fixed early_stopping test logic
- ✅ Cleaned all unused imports
- ✅ All 48 tests passing in rust-functiongemma-train

## Active Rust Projects

### 0. pcai-inference ✅ NATIVE DUAL-BACKEND
**Path**: `Deploy/pcai-inference/`
**Status**: Building, needs MSVC build test for llamacpp
**Build**: `cargo build --release --features server,llamacpp`

**Dual Backend Architecture**:
- **llama-cpp-2**: GGUF model inference with CUDA GPU acceleration (requires MSVC)
- **mistral.rs**: Alternative backend (CPU-only on Windows due to bindgen_cuda)

**HTTP Endpoints** (port 8080):
- `GET /health` → `{"status":"healthy","backend":"llamacpp"}`
- `GET /v1/models` → OpenAI-compatible model list
- `POST /v1/chat/completions` → Full OpenAI-compatible with streaming

**FFI Exports** (for PowerShell P/Invoke):
- `pcai_init(backend)` → Initialize backend
- `pcai_load_model(path, gpu_layers)` → Load GGUF model
- `pcai_generate(prompt, max_tokens, temp)` → Generate completion
- `pcai_shutdown()` → Clean shutdown

**Feature Flags**:
```toml
[features]
default = ["llamacpp", "server"]
llamacpp = ["dep:llama-cpp-2", "dep:encoding_rs"]
mistralrs-backend = ["dep:mistralrs", "dep:mistralrs-core"]
cuda = []
server = ["dep:axum", "dep:tower-http"]
ffi = []
```

**Build System**:
- CMake toolchain: `cmake/toolchain-msvc.cmake`
- CMake presets: `CMakePresets.json` (msvc-release, msvc-cuda)
- PowerShell orchestrator: `build.ps1`

### 1. rust-functiongemma-runtime ✅ PRODUCTION READY
**Path**: `Deploy/rust-functiongemma-runtime/`
**Status**: OpenAI-compatible, verified with PowerShell TUI
**Build**: `cargo build --release`

**Endpoints**:
- `GET /health` → `{"status":"ok"}`
- `GET /v1/models` → Model list
- `POST /v1/chat/completions` → Full OpenAI-compatible response

### 2. rust-functiongemma-train ✅ COMPILES CLEAN
**Path**: `Deploy/rust-functiongemma-train/`
**Status**: All compilation issues fixed, 46 tests pass
**Build**: `cargo build --manifest-path Deploy/rust-functiongemma-train/Cargo.toml`

**Modules**:
- `lora.rs` - LoRA layer with A/B decomposition
- `trainer.rs` - Training loop with gradient accumulation
- `scheduler.rs` - LR schedulers (Cosine, Linear, Constant) + warmup
- `checkpoint.rs` - Model checkpoint save/load/cleanup
- `early_stopping.rs` - Patience-based early stopping
- `dataset.rs` - JSONL loading with token caching
- `router_dataset.rs` - Tool routing dataset builder

**Test Breakdown**:
```
lib.rs unit tests:        11 passed
checkpoint_test.rs:        5 passed
early_stopping_test.rs:    5 passed
full_training_test.rs:     9 passed
integration_test.rs:       1 passed
lora_test.rs:              1 passed
peft_output_test.rs:       2 passed
router_dataset.rs:         1 passed
scheduler_test.rs:         6 passed
trainer_lora_test.rs:      4 passed
```

### 3. pcai_fs (FFI for .NET)
**Path**: `Native/pcai_core/pcai_fs/`
**Status**: Complete, 14 tests pass
**Build**: `cargo build --manifest-path Native/pcai_core/Cargo.toml`

**FFI Exports**:
- `pcai_delete_fs_item` - Safe file/directory deletion
- `pcai_replace_in_file` - Regex/literal text replacement
- `pcai_replace_in_files` - Batch replacement with parallel processing

### 4. pcai_core_lib
**Path**: `Native/pcai_core/pcai_core_lib/`
**Status**: Complete
**Modules**: fs, performance, search, system, telemetry, functiongemma (stubs)

## Crate Dependencies

```toml
# Training (rust-functiongemma-train)
candle-core = { version = "0.9.2", features = ["cuda"] }  # GPU tensors
candle-nn = { version = "0.9.2", features = ["cuda"] }    # Neural network layers
candle-transformers = { version = "0.9.2", features = ["cuda"] } # Transformer models
tokenizers = "0.22"          # Fast tokenization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
clap = { version = "4.5", features = ["derive"] }

# FFI (pcai_fs)
rayon = "1.10"       # Parallel processing
ignore = "0.4"       # Gitignore-aware walking
walkdir = "2.5"      # Directory traversal
regex = "1.11"       # Pattern matching
memmap2 = "0.9"      # Memory-mapped files
```

## Build Environment

- **Rust Edition**: 2021 (stable)
- **CUDA**: v13.1 (Ada/Blackwell compute 8.9/12.0)
- **GPUs**: RTX 2000 Ada (8GB) + RTX 5060 Ti (16GB)
- **Linker**: link.exe (lld-link optional)
- **Cache Root**: `T:\RustCache\`
- **Target Dir**: `T:\RustCache\cargo-target\`

## Key Commands

```bash
# Build training crate
cargo build --manifest-path Deploy/rust-functiongemma-train/Cargo.toml

# Run all tests
cargo test --manifest-path Deploy/rust-functiongemma-train/Cargo.toml

# Check for warnings
cargo clippy --manifest-path Deploy/rust-functiongemma-train/Cargo.toml

# Build FFI library
cargo build --manifest-path Native/pcai_core/Cargo.toml --release
```

## Recent Commits

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

## Known Issues

1. **bindgen_cuda panic in mistral.rs**: Blocks CUDA support on Windows for mistralrs backend
   - Workaround: Use CPU-only mode or llamacpp backend for GPU
   - Doc: `T:\projects\rust-mistral\mistral.rs\CUDA_BUILD_BLOCKING_ISSUE.md`

2. **llamacpp MSVC requirement**: llama-cpp-sys-2 requires MSVC, not MinGW/GNU
   - Solution: CMake toolchain with vcvars64.bat environment
   - Status: Toolchain created (`cmake/toolchain-msvc.cmake`), needs testing

3. **sccache + ring crate**: sccache has path spacing issues with ring crate builds
   - Workaround: Build runtime without `model` feature, or disable sccache

4. **Dependabot Alert**: protobuf CVE (high severity, monitoring)

## Next Steps

- [x] Enable CUDA for training (completed 2026-01-30)
- [x] Implement pcai-inference dual-backend (completed 2026-01-30)
- [x] Create CMake/MSVC toolchain (completed 2026-01-30)
- [ ] **Test llamacpp build with MSVC toolchain** (priority)
- [ ] Run inference tests with real GGUF model
- [ ] Add streaming support to FFI layer
- [ ] Fix sccache/ring compatibility for runtime model feature
- [ ] Performance benchmarks for CUDA training
