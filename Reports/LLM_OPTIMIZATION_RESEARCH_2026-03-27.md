# LLM Inference Optimization Research — March 2026

> Generated from HuggingFace, web, and paper research. Prioritized for the PC_AI Janus-Pro pipeline.

## Priority Actions

| Priority | Optimization | Impact | Effort | Status |
|----------|-------------|--------|--------|--------|
| **P0** | Fix VQ decode dtype chain (F32 for non-LLM on GGUF) | Fixes GGUF image gen | Low | **DONE** (f630133) |
| **P1** | Enable `cudnn` feature for VQ-VAE conv2d | ~2x VQ decode speedup | Low | TODO |
| **P1** | IQ4_XS with imatrix for Janus-Pro GGUF | Better quality than Q4_K_M | Low | TODO |
| **P2** | SWIFT-style self-speculative layer skipping | 1.3-1.6x tok/s | Medium | TODO |
| **P2** | INT8 KV cache quantization | Halve KV cache VRAM | Medium | TODO |
| **P3** | Tensor reuse / memory pooling in gen loop | Reduce alloc overhead | Medium | TODO |
| **P3** | CUDA Graphs for candle (576-step fixed loop) | ~40% kernel speedup | High | Research |
| **P4** | EAGLE-3 draft head for Janus-Pro | 2-6x but needs training | High | Research |

## Key Findings

### GGUF: IQ4_XS > Q4_K_M
IQ4_XS (~4.25 bits) outperforms Q4_K_M (~4.58 bits) with imatrix calibration. Generate imatrix from Janus-Pro prompts before quantizing. Command: `llama-quantize --imatrix <cal.dat> model.gguf model-iq4xs.gguf IQ4_XS`

### Speculative Decoding: SWIFT best for small models
SWIFT (on-the-fly layer skipping) is plug-and-play, no training, 1.3-1.6x. PC_AI already has speculative infra in generate.rs. For the fixed 576-token image gen, high acceptance rate expected since VQ tokens are structured.

### CUDA Graphs: Not in candle, available in llama.cpp
14% overall speedup in llama.cpp (batch=1). Would benefit the 576-step loop but requires framework changes. Use llama.cpp backend for text-only tasks.

### Flash Attention: Not viable on Windows
candle-flash-attn requires CUTLASS which doesn't build on MSVC. cuDNN fused attention is the Windows alternative but not yet in candle. For 576-token sequences, flash attention impact is minimal.

### KV Cache: Current PreAllocKvCache is near-optimal
The fixed 576-token generation with pre-allocated BF16 cache is already good. INT8 cache quantization would halve memory. Paged attention not needed for fixed-length gen.

### VQ Decode: cuDNN conv acceleration is the win
Enable `cudnn` cargo feature for the ~20 Conv2d layers in the VQ decoder. BF16 conv is 2x faster than F32 on Ada/Blackwell GPUs, but GroupNorm must stay F32.
