# pcai-inference

Dual-backend LLM inference engine for PC diagnostics with native Rust performance.

## Features

- **Dual Backend Support**
  - llama-rs (`llm`) via the `llamacpp` compatibility feature
  - mistral.rs (feature: `mistralrs-backend`)

- **Flexible Deployment**
  - HTTP server with OpenAI-compatible API (feature: `server`)
  - C FFI exports for PowerShell integration (feature: `ffi`)
  - Optional CUDA acceleration (features: `cuda-llamacpp`, `cuda-mistralrs`)

- **Production Ready**
  - Async/await with Tokio
  - Structured logging with tracing
  - Type-safe error handling
  - Comprehensive test coverage

## Quick Start

### HTTP Server

```bash
# Build with llama.cpp backend and server
cargo build --release --features "llamacpp,server"

# Run server with config
./target/release/pcai-inference --config config.json
```

### Configuration

```json
{
  "backend": {
    "type": "llama_cpp",
    "n_gpu_layers": 35,
    "n_ctx": 4096
  },
  "model": {
    "path": "/path/to/model.gguf",
    "generation": {
      "max_tokens": 512,
      "temperature": 0.7,
      "top_p": 0.95
    }
  },
  "server": {
    "host": "127.0.0.1",
    "port": 8080,
    "cors": true
  }
}
```

### API Usage

```bash
# Health check
curl http://localhost:8080/health

# Generate completion
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Diagnose: disk errors",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `llamacpp` | llama-rs (`llm`) compatibility backend | Yes |
| `mistralrs-backend` | mistral.rs backend | No |
| `cuda-llamacpp` | CUDA (cuBLAS) for llama-rs backend | No |
| `cuda-mistralrs` | CUDA for mistral.rs backend | No |
| `cuda` | CUDA GPU acceleration (umbrella) | No |
| `server` | HTTP server with Axum | Yes |
| `ffi` | C FFI exports for PowerShell | No |

## CUDA Target Selection

When building with `cuda-llamacpp`, CUDA arch targets are selected by the vendored
`ggml-sys` build script in this order:

1. `GGML_CUDA_ARCH_LIST` (comma/semicolon/space-separated values such as `75,86,89`)
2. `PCAI_CUDA_ARCH_LIST` (same format)
3. Auto-detection via `nvidia-smi --query-gpu=compute_cap`
4. Fallback list: `61,70,75,80,86`

Examples:

```bash
# Build for a single target (RTX 4000 / Turing)
set GGML_CUDA_ARCH_LIST=75
cargo build --release --features "llamacpp,ffi,cuda-llamacpp"

# Build a multi-arch binary
set GGML_CUDA_ARCH_LIST=75,86,89,120
cargo build --release --features "llamacpp,ffi,cuda-llamacpp"
```

## Development

```bash
# Check compilation with no features
cargo check --no-default-features

# Run tests
cargo test

# Build with all features
cargo build --all-features

# Run with logging
RUST_LOG=pcai_inference=debug cargo run
```

## FFI Integration

When built with `ffi` feature, exports C-compatible functions:

```rust
pcai_init(config_json: *const c_char) -> *mut c_void
pcai_generate(handle: *mut c_void, prompt: *const c_char) -> *mut c_char
pcai_free_string(s: *mut c_char)
pcai_shutdown(handle: *mut c_void)
```

## Architecture

```
pcai-inference/
├── backends/       # Backend implementations
├── config/         # Configuration types
├── http/           # HTTP server (feature: server)
├── ffi/            # FFI exports (feature: ffi)
└── tests/          # Integration tests
```

## License

MIT
