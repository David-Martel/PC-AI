# pcai-media: Rust-Based Janus-Pro Media LLM Agent

**Date:** 2026-03-02
**Status:** Approved
**Scope:** Full multimodal agent (image generation + understanding + upscaling)

## Summary

Build a pure Rust media LLM agent around DeepSeek Janus-Pro models using the candle framework. Supports text-to-image generation, image understanding (VQA), and RealESRGAN upscaling. Exposes both C FFI (for P/Invoke → PowerShell pipeline) and HTTP API (for TUI/external tools). Targets Janus-Pro-1B (dev) and Janus-Pro-7B (prod).

## Architecture Decision: Pure Rust via Candle

Janus-Pro cannot run through llama.cpp or mistral.rs — the VQ decoder and dual encoders aren't supported in GGUF format. Only the text LLM backbone converts to GGUF; vision understanding and image generation require the full model. We implement the complete pipeline natively in Rust using candle (0.9.x) with CUDA support.

## Crate Structure (Multi-Crate Split)

```
Native/pcai_core/                        # Existing workspace — add 3 members
├── pcai_media_model/                    # Pure model architecture (no I/O)
│   ├── src/
│   │   ├── lib.rs                       # JanusModel facade
│   │   ├── config.rs                    # JanusConfig (from HF config.json)
│   │   ├── vq_vae.rs                    # VQ-VAE encoder + decoder
│   │   ├── generation_head.rs           # Linear(4096→16384) + generation aligner MLP
│   │   └── tensor_utils.rs              # Normalize/denormalize
│   └── Cargo.toml
│
├── pcai_media/                          # Pipeline orchestration + FFI DLL
│   ├── src/
│   │   ├── lib.rs                       # Public pipeline API
│   │   ├── config.rs                    # PipelineConfig
│   │   ├── generate.rs                  # Text→Image (576-token autoregressive + CFG)
│   │   ├── understand.rs                # Image→Text (SigLIP → LLM decode)
│   │   ├── upscale.rs                   # RealESRGAN post-processing
│   │   ├── hub.rs                       # HF Hub download + safetensors loading
│   │   └── ffi/
│   │       ├── mod.rs                   # C ABI exports
│   │       └── types.rs                 # #[repr(C)] structs
│   └── Cargo.toml
│
└── pcai_media_server/                   # HTTP server binary
    ├── src/
    │   ├── main.rs                      # Entry + CLI (clap)
    │   ├── routes.rs                    # Axum routes
    │   └── handlers.rs                  # Request→Pipeline→Response
    └── Cargo.toml
```

**Dependency graph:**
```
pcai_media_server → pcai_media → pcai_media_model → candle-core/nn/transformers
                               → pcai_core_lib (telemetry, config)
```

## Component Reuse Map

### Direct Reuse (~65-70% of total code)

| Component | Source | Notes |
|-----------|--------|-------|
| Llama LLM backbone | `candle-transformers::models::llama` | Config: 30 layers, 4096 hidden, 102400 vocab |
| SigLIP vision encoder | `candle-transformers::models::siglip` | Adjust for siglip_large_patch16_384 |
| MLP projector | `candle-transformers::models::llava::MMProjector` | Maps vision features → LLM space |
| Multimodal fusion | `candle-transformers::models::llava` | Adapt interleaving logic |
| VQ codebook | `candle-transformers::models::encodec::EuclideanCodebook` | Extract for VQ tokenizer |
| Safetensors loading | `Deploy/rust-functiongemma-core/safetensors_utils.rs` | Production-ready |
| Tensor utils | `AI-Media/src/tensor_utils.rs` | Normalize/denormalize |
| Pipeline config | `AI-Media/src/main.rs` (PipelineConfig) | HF Hub, device, dtype |
| HTTP server | `pcai_inference/src/http/` | Axum routes, state, CORS |
| FFI pattern | `pcai_inference/src/ffi/mod.rs` | OnceLock+Mutex, thread-local errors |

### Adapt (existing architecture matches)

| Component | Source | Work Needed |
|-----------|--------|-------------|
| Janus model v2 | `AI-Media/src/janus_model_v2.rs` | Complete up_blocks, wire weight loading |
| Generation loop | `AI-Media/src/main.rs` | Add KV cache, integrate with candle Llama |
| RealESRGAN | ESRGAN-candle-rs (GitHub) | Extract model code, convert weights |
| C# P/Invoke | `PcaiNative/InferenceModule.cs` | Add media-specific bindings |

### Must Build (unique to Janus)

| Component | Complexity | Notes |
|-----------|------------|-------|
| VQ-VAE CNN decoder | Medium | Port from LlamaGen (skeleton in janus_model.rs) |
| VQ-VAE CNN encoder | Medium | Same architecture reversed |
| Generation head | Low | `Linear(4096 → 16384)` |
| Generation aligner MLP | Low | 2-layer MLP projector |
| Orchestration | Medium | Wire understanding + generation pathways |

## Janus-Pro Model Architecture

```
                ┌─────────────────────────────────────────┐
                │          MultiModalityCausalLM           │
                │                                         │
  Understanding │   ┌──────────┐    ┌─────────┐          │
  Pathway:      │   │ SigLIP-L │───▶│ MLP     │──┐       │
  (image→text)  │   │ 384×384  │    │ Aligner │  │       │
                │   └──────────┘    └─────────┘  │       │
                │                                ▼       │
                │                         ┌──────────┐   │
                │                         │  Llama   │   │
                │                         │ LLM 7B   │──▶ text output
                │                         │ 30 layers│   │
                │                         └──────────┘   │
                │                                ▲       │
  Generation    │   ┌──────────┐    ┌─────────┐  │       │
  Pathway:      │   │ VQ Token │◀──▶│ MLP     │──┘       │
  (text→image)  │   │ Codebook │    │ Aligner │          │
                │   │ (16384)  │    └─────────┘          │
                │   └──────────┘                         │
                │        │                               │
                │        ▼                               │
                │   ┌──────────┐                         │
                │   │ VQ-VAE   │                         │
                │   │ Decoder  │──▶ 384×384 image        │
                │   └──────────┘                         │
                └─────────────────────────────────────────┘
```

### Generation Pipeline (576-token autoregressive)

1. Tokenize prompt with template: `<|User|>{prompt}<|Assistant|>`
2. Create batched tensor (positive + negative for CFG)
3. Get input embeddings from LLM
4. Loop 576 iterations (24×24 patches at patch_size=16 for 384×384):
   - Forward through Llama → logits
   - Project logits to image vocab (16384 classes)
   - Apply CFG: `guided = uncond + scale * (cond - uncond)`
   - Sample token via temperature-controlled multinomial
   - Prepare next embedding from VQ codebook
5. Reshape tokens to (batch, 256, 24, 24) latent grid
6. Decode via VQ-VAE decoder → 384×384 RGB image
7. Denormalize [-1,1] → [0,255] → PNG

### Understanding Pipeline

1. Encode image via SigLIP-L → feature tensor
2. Project via MLP aligner → LLM-compatible embeddings
3. Interleave image tokens with text prompt tokens
4. Forward through Llama → autoregressive text generation
5. Decode tokens → response text

## FFI Contract

```rust
// Lifecycle
pub extern "C" fn pcai_media_init(device: *const c_char) -> i32
pub extern "C" fn pcai_media_load_model(model_path: *const c_char, gpu_layers: i32) -> i32
pub extern "C" fn pcai_media_shutdown()

// Image Generation
pub extern "C" fn pcai_media_generate_image(
    prompt: *const c_char,
    cfg_scale: f32,
    temperature: f32,
    output_path: *const c_char,
) -> i32

pub extern "C" fn pcai_media_generate_image_bytes(
    prompt: *const c_char,
    cfg_scale: f32,
    temperature: f32,
    out_data: *mut *mut u8,
    out_len: *mut usize,
) -> i32

// Image Understanding
pub extern "C" fn pcai_media_understand_image(
    image_path: *const c_char,
    prompt: *const c_char,
    max_tokens: u32,
    temperature: f32,
) -> *mut c_char

// Upscaling
pub extern "C" fn pcai_media_upscale(
    input_path: *const c_char,
    output_path: *const c_char,
    scale: u32,
) -> i32

// Async (same pattern as pcai-inference)
pub extern "C" fn pcai_media_generate_image_async(...) -> i64
pub extern "C" fn pcai_media_understand_image_async(...) -> i64
pub extern "C" fn pcai_media_poll_result(request_id: i64) -> PcaiMediaAsyncResult
pub extern "C" fn pcai_media_cancel(request_id: i64) -> i32

// Memory
pub extern "C" fn pcai_media_free_string(s: *mut c_char)
pub extern "C" fn pcai_media_free_bytes(data: *mut u8, len: usize)
pub extern "C" fn pcai_media_last_error() -> *const c_char
pub extern "C" fn pcai_media_last_error_code() -> i32
```

## HTTP API

```
POST /v1/images/generate      # Text→Image (returns PNG or base64 JSON)
POST /v1/images/understand     # Image→Text (returns JSON analysis)
POST /v1/images/upscale        # Upscale (returns PNG or base64 JSON)
GET  /v1/models                # List loaded models
GET  /health                   # Health + GPU status
```

## C# + PowerShell Integration

**C# wrapper (`Native/PcaiNative/MediaModule.cs`):**
- Same DLL resolution pattern as InferenceModule.cs
- High-level wrappers with automatic memory management
- Struct marshaling for async results

**PowerShell module (`Modules/PcaiMedia.psm1`):**
- `Initialize-PcaiMedia [-Device cuda|cpu]`
- `Import-PcaiMediaModel [-ModelPath ...] [-GpuLayers ...]`
- `New-PcaiImage -Prompt "..." [-CfgScale 5.0] [-Temperature 1.0]`
- `Get-PcaiImageAnalysis -ImagePath "..." [-Question "..."]`
- `Invoke-PcaiUpscale -InputPath "..." [-Scale 4]`
- `Stop-PcaiMedia`

## Model Sizes & GPU Targets

| Model | Params | VRAM (BF16) | VRAM (Q4) | Target GPU |
|-------|--------|-------------|-----------|------------|
| Janus-Pro-1B | 1.5B | ~3GB | ~1.5GB | RTX 2000 Ada (8GB) — dev |
| Janus-Pro-7B | 7B | ~14GB | ~4.5GB | RTX 5060 Ti (16GB) — prod |

Dev workflow uses 1B for fast iteration; prod uses 7B for quality.

## Build Integration

```powershell
# Add to Build.ps1
.\Build.ps1 -Component media -EnableCuda

# Produces:
# .pcai/build/artifacts/pcai-media/pcai_media.dll    (FFI library)
# .pcai/build/artifacts/pcai-media/pcai-media.exe     (HTTP server)
```

## Key Dependencies

```toml
[dependencies]
candle-core = { version = "0.9", features = ["cuda", "cudnn"] }
candle-nn = "0.9"
candle-transformers = "0.9"
candle-flash-attn = { version = "0.9", optional = true }
tokenizers = "0.20"
safetensors = "0.4"
image = "0.25"
hf-hub = "0.4"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
tokio = { version = "1", features = ["full"] }
axum = "0.7"      # server only
clap = "4"         # server only
```
