# mistral.rs Backend Implementation

## Overview

The mistral.rs backend provides high-performance LLM inference with support for:
- GGUF quantized models (local files and HuggingFace repos)
- SafeTensors models (HuggingFace format)
- Multimodal models (vision + text)
- CPU inference (Windows, Linux, macOS)
- CUDA acceleration (Linux/WSL2 only)

## Windows CUDA Limitation

**Important**: CUDA builds are currently blocked on Windows due to bindgen_cuda library issues in the upstream mistral.rs project. See [CUDA_BUILD_BLOCKING_ISSUE.md](https://github.com/EricLBuehler/mistral.rs/blob/master/CUDA_BUILD_BLOCKING_ISSUE.md) in the mistral.rs repository.

The implementation automatically forces CPU mode on Windows. For GPU acceleration, use WSL2 or Linux.

## Current Limitations

### 1. Deterministic Sampling Only

The current implementation uses `TextMessages` which provides deterministic sampling by default. The following `GenerateRequest` parameters are currently **ignored**:

- `max_tokens` - No token limit (generates until stop condition)
- `temperature` - Always uses deterministic (greedy) sampling
- `top_p` - Not applied
- `stop` - Stop sequences not yet supported

**Why**: mistral.rs uses a `RequestLike` trait where sampling parameters are bundled with the message type. `TextMessages` returns `SamplingParams::deterministic()` and doesn't allow customization.

**Workaround**: To support custom sampling, we need to:
1. Create a custom struct implementing `RequestLike`
2. Override `take_sampling_params()` to return custom `SamplingParams`
3. This is planned for a future update

### 2. GGUF Path Parsing

For GGUF models, the current implementation expects:
- Local file paths (e.g., `/path/to/model.gguf`)
- The filename will be extracted automatically
- The parent directory is used as the model ID

HuggingFace repo format (e.g., `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`) is not yet supported. You must download the GGUF file locally first.

## Usage Examples

### Loading a GGUF Model

```rust
use pcai_inference::backends::{BackendType, InferenceBackend};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = BackendType::MistralRs.create()?;

    // Load local GGUF model
    backend.load_model("C:/models/llama-3.1-8b-instruct-q4_k_m.gguf").await?;

    // Generate (uses deterministic sampling)
    let request = GenerateRequest {
        prompt: "Explain quantum computing".to_string(),
        max_tokens: None,
        temperature: None,
        top_p: None,
        stop: vec![],
    };

    let response = backend.generate(request).await?;
    println!("Response: {}", response.text);

    Ok(())
}
```

### Loading a SafeTensors Model

```rust
// Load from HuggingFace
backend.load_model("microsoft/Phi-3.5-mini-instruct").await?;
```

## Building

### CPU-only (Windows, Linux, macOS)

```bash
cd Deploy/pcai-inference
cargo build --features mistralrs-backend --no-default-features
```

### With CUDA (Linux/WSL2 only)

```bash
cargo build --features mistralrs-backend,cuda-mistralrs --no-default-features
```

Note: CUDA requires appropriate CUDA toolkit and drivers installed.

## Testing

```bash
# Run unit tests
cargo test --features mistralrs-backend --no-default-features

# Check compilation
cargo check --features mistralrs-backend --no-default-features
```

## Architecture Details

### Backend Structure

```rust
pub struct MistralRsBackend {
    model: Option<Arc<Model>>,
    model_path: Option<String>,
    is_gguf: bool,
}
```

- `model`: The mistral.rs `Model` instance (wrapped in Arc for sharing)
- `model_path`: Path to the currently loaded model
- `is_gguf`: Flag indicating if the model is GGUF format

### Device Selection

`backend.device` in the configuration selects where the model runs. The accepted values are defined once,
by `DeviceSpec` in `src/config.rs`, and shared by the `pcai-mistralrs` binary and the backend:

| Value | Behaviour |
|---|---|
| unset, `""`, `auto` | CUDA when the build has it (`cuda-mistralrs`) and device 0 initializes; otherwise CPU. A Windows build without CUDA uses CPU. |
| `cpu` | CPU, even when CUDA is available. |
| `cuda:0` | CUDA device 0. Never falls back: a build without CUDA, or a device that fails to initialize, stops the load with an error naming the cause. |
| `cuda:N`, N > 0 | Refused at load. mistral.rs 0.8.1 resolves the device for its GGUF and text builders without looking at `with_device`, so it can only load on CUDA device 0. Start the process with `CUDA_VISIBLE_DEVICES=N` and use `cuda:0`. |

Any other string is a configuration error that lists the accepted forms.

### Layer placement

`backend.device_layers` controls how layers are placed on the device (default `auto`):

| Value | Behaviour |
|---|---|
| unset, `auto` | mistral.rs chooses. A model that does not fit the GPU is split, with the last layers on the CPU. For GGUF models that split loads and then fails the first forward pass with `device mismatch in rms-norm, lhs: Cpu, rhs: Cuda` (seen with Qwen2.5-7B Q4_K_M on an 8 GB GPU); the readiness probe below turns that into a not-ready server. |
| `all` | Every layer on the selected device. A model that does not fit fails at load with the loader's out-of-memory error instead of loading as a split. It has no effect with `device: "cpu"`, where every layer is already on the CPU. |

```json
{ "backend": { "type": "mistral_rs", "device": "cuda:0", "device_layers": "all" } }
```

A partial split (a number of GPU layers) is not offered: the mixed placement is the one that fails.

### Readiness

`pcai-mistralrs` runs one short deterministic generation (a few tokens, temperature 0) after the model loads,
before it binds the socket. mistral.rs 0.8.1 does run a one-token dummy run at load, but it logs
"Dummy run completed" even when the run returned an error and exposes no result, and loading the weights
proves nothing about the first forward pass: a model whose layers are split between the CPU and a GPU loads,
then fails every request with `device mismatch in rms-norm`. If the probe fails the server logs the full
error (target `pcai_inference_lib`, enabled by the default log filter), unloads the model and still starts,
so a supervisor can see that it is not usable:

- `GET /health` answers `503` with `{"status":"not_ready","reason":"generation_probe_failed","message":"..."}`.
  `generation_probe_timeout` is the code when the probe does not finish in 300 s. The route is
  unauthenticated, so the reason is a stable code with a fixed message and never text from the error, which
  can hold file paths; read the log for the cause.
- `GET /v1/models` and every inference route answer `503` with the same code (after the bearer-token check,
  which still runs first). Chat requests that the FunctionGemma router answers do not use the local model,
  so only their fallback to local inference is gated.

`GET /health` answers `200` with `"status":"ready"` only after the probe passes. It answered `200` with
`not_ready` before, which clients that only check the HTTP status (the PowerShell callers) read as healthy.

### Response Mapping

mistral.rs returns `ChatCompletionResponse` which is mapped to our `GenerateResponse`:

```rust
struct GenerateResponse {
    text: String,              // From choices[0].message.content
    tokens_generated: usize,   // From usage.completion_tokens
    finish_reason: FinishReason, // Mapped from choices[0].finish_reason
}
```

## Future Improvements

### Priority 1: Custom Sampling Parameters

Implement a custom `RequestLike` struct to support:
- Temperature control
- Top-p/top-k sampling
- Token limits
- Stop sequences

### Priority 2: Streaming Support

Add streaming generation using mistral.rs's `stream_chat_request`:

```rust
async fn generate_stream(
    &self,
    request: GenerateRequest,
    callback: impl Fn(String) + Send + 'static,
) -> Result<GenerateResponse>
```

### Priority 3: HuggingFace GGUF Support

Allow direct loading from HuggingFace repos:

```rust
backend.load_gguf_hf(
    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
).await?;
```

### Priority 4: Multimodal Support

Extend `GenerateRequest` to support images and audio:

```rust
struct MultimodalRequest {
    messages: Vec<Message>,
    images: Vec<DynamicImage>,
    audio: Vec<AudioInput>,
    // ...
}
```

## Dependencies

The implementation requires:

```toml
mistralrs = { path = "T:/projects/rust-mistral/mistral.rs/mistralrs" }
mistralrs-core = { path = "T:/projects/rust-mistral/mistral.rs/mistralrs-core" }
```

Using the user's local fork which includes:
- CUDA 13.0 support (Linux/WSL2 only)
- Qwen 2.5/2.5 VL improvements
- CPU flash attention
- Latest upstream improvements

## Troubleshooting

### Build fails with "bindgen_cuda" error on Windows

This is expected. Use CPU-only build:

```bash
cargo build --features mistralrs-backend --no-default-features
```

### "Failed to get device" error

Check that you're using CPU mode on Windows. On Linux, verify CUDA is properly installed.

### Model loading fails

1. Verify the model path is correct
2. For GGUF: Ensure the file has `.gguf` extension
3. For SafeTensors: Verify the HuggingFace model ID is correct
4. Check available disk space and memory

### Generation produces no output

This is a known issue with deterministic sampling on some models. Future updates will add temperature control to address this.

## References

- [mistral.rs Repository](https://github.com/EricLBuehler/mistral.rs)
- [User's Fork](T:/projects/rust-mistral/mistral.rs)
- [CUDA Build Issue](https://github.com/EricLBuehler/mistral.rs/blob/master/CUDA_BUILD_BLOCKING_ISSUE.md)
- [mistralrs Rust Docs](https://ericlbuehler.github.io/mistral.rs/mistralrs/)
