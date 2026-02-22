# Integration Tests for pcai-inference

This directory contains integration tests for the pcai-inference crate.

## Test Structure

```
tests/
├── integration_test.rs  # Main integration test suite
├── common/              # Shared test utilities
│   └── mod.rs          # Model discovery, test helpers
└── README.md           # This file
```

## Running Tests

### Basic Tests (No Backends)

Run tests without any backend features enabled:

```bash
cargo test --no-default-features
```

These tests cover:
- Configuration serialization
- Error types
- Mock backend implementation
- Basic trait behavior

### Tests with llama.cpp Backend

Run tests with the llamacpp backend:

```bash
cargo test --features llamacpp
```

**Requirements:**
- C++ compiler (MSVC on Windows, GCC/Clang on Linux)
- CUDA Toolkit (optional, for GPU tests)

### Tests with mistral.rs Backend

Run tests with the mistralrs backend:

```bash
cargo test --features mistralrs-backend
```

**Requirements:**
- Rust toolchain (no C++ compiler needed)

### Tests with FFI

Run tests for the FFI interface (requires at least one backend):

```bash
cargo test --features "llamacpp,ffi"
```

### All Features

Run all tests with all features enabled:

```bash
cargo test --all-features
```

## Testing with Real Models

Some tests are skipped by default unless a real model file is available.

### Option 1: Set Environment Variable

```bash
Update Config/llm-config.json:
  providers.pcai-native.modelPath = "C:\\path\\to\\model.gguf"
cargo test --features llamacpp
```

### Option 2: Install via Ollama

```bash
# Install Ollama and pull a model
ollama pull llama3.2:1b

# Tests will automatically discover models in ~/.ollama/models/blobs
cargo test --features llamacpp
```

### Option 3: Install via LM Studio

Install LM Studio and download a model. Tests will search:
- Linux/Mac: `~/.cache/lm-studio/models`
- Windows: `%LOCALAPPDATA%\lm-studio\models`

## Test Categories

### Configuration Tests

Tests for serialization and default values:
- `test_generate_request_serialization`
- `test_generate_response_serialization`
- `test_finish_reason_serialization`
- `test_config_defaults`

### Backend Creation Tests

Tests for backend instantiation (feature-gated):
- `test_llamacpp_backend_creation` (llamacpp)
- `test_llamacpp_backend_with_config` (llamacpp)
- `test_mistralrs_backend_creation` (mistralrs-backend)
- `test_backend_type_llamacpp` (llamacpp)
- `test_backend_type_mistralrs` (mistralrs-backend)

### Backend Lifecycle Tests

Tests for model loading/unloading:
- `test_mock_backend_lifecycle`
- `test_mock_backend_generate`
- `test_backend_generate_without_model_fails`
- `test_llamacpp_load_nonexistent_file_fails` (llamacpp)
- `test_mistralrs_load_nonexistent_file_fails` (mistralrs-backend)

### Generation Tests (Require Real Models)

Tests that need a real model file (skipped without a configured modelPath):
- `test_llamacpp_generate_with_model` (llamacpp)
- `test_mistralrs_generate_with_model` (mistralrs-backend)
- `test_ffi_full_lifecycle_with_model` (ffi + llamacpp)

### FFI Tests

Tests for the C FFI interface (ffi feature):
- `test_ffi_init_null_backend`
- `test_ffi_init_unknown_backend`
- `test_ffi_init_llamacpp` (llamacpp)
- `test_ffi_init_mistralrs` (mistralrs-backend)
- `test_ffi_load_model_null_path`
- `test_ffi_load_model_before_init`
- `test_ffi_generate_null_prompt`
- `test_ffi_generate_before_init`
- `test_ffi_generate_without_model` (llamacpp)
- `test_ffi_free_string_null`
- `test_ffi_last_error_no_error`

### Stress Tests (Run with --ignored)

Long-running tests for stability:
- `stress_test_sequential_generations` (llamacpp)
- `stress_test_backend_switching` (llamacpp + mistralrs-backend)

Run with:
```bash
cargo test --features llamacpp -- --ignored
```

## Test Utilities

### `require_model!` Macro

Skips tests that require a real model if providers.pcai-native.modelPath is not set:

```rust
#[test]
fn test_with_model() {
    require_model!();
    // test code that needs a model
}
```

### `find_test_model()` Function

Automatically discovers GGUF models in common locations:

```rust
if let Some(model_path) = find_test_model() {
    // Use model_path
}
```

### `MockBackend` Struct

Fake backend for testing without real models:

```rust
let mut backend = MockBackend::new();
backend.load_model("dummy.gguf").await.unwrap();
let response = backend.generate(request).await.unwrap();
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      # Test without backends
      - run: cargo test --no-default-features

      # Test with mistralrs (no C++ required)
      - run: cargo test --features mistralrs-backend
```

### Windows CI with MSVC

```yaml
jobs:
  test-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v2
      - uses: ilammy/msvc-dev-cmd@v1  # Setup MSVC
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      # Test with llamacpp (requires MSVC)
      - run: cargo test --features llamacpp
```

## Debugging Failed Tests

### Verbose Test Output

```bash
cargo test -- --nocapture
```

### Run Specific Test

```bash
cargo test test_llamacpp_backend_creation -- --exact
```

### Show Ignored Tests

```bash
cargo test -- --ignored
```

### Check Feature Compilation

```bash
# Verify llamacpp compiles
cargo check --features llamacpp

# Verify mistralrs compiles
cargo check --features mistralrs-backend

# Verify FFI compiles
cargo check --features "llamacpp,ffi"
```

## Coverage

Generate test coverage with tarpaulin:

```bash
cargo install cargo-tarpaulin
cargo tarpaulin --features llamacpp --out Html
```

Target: 85% minimum test coverage (95% for critical paths)

## Troubleshooting

### llamacpp build fails: "stdio.h not found"

**Solution:** Install C++ build tools:
- Windows: Install Visual Studio Build Tools with MSVC
- Linux: `sudo apt install build-essential`
- Mac: `xcode-select --install`

### FFI tests fail: "Backend not initialized"

**Cause:** FFI feature requires at least one backend feature.

**Solution:** Enable a backend:
```bash
cargo test --features "ffi,llamacpp"
```

### Model tests skipped

**Cause:** No model file found.

**Solution:** Set providers.pcai-native.modelPath in Config/llm-config.json or install via Ollama/LM Studio.

### Stress tests not running

**Cause:** Stress tests are marked `#[ignore]` by default.

**Solution:** Run with `-- --ignored` flag:
```bash
cargo test --features llamacpp -- --ignored
```
