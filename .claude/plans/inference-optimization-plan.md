# LLM Inference Optimization Plan: 25 → 250+ tok/s

> Created: 2026-03-19 | Status: Phase 1 In Progress
> Baseline: 25 tok/s on RTX 2000 Ada 8GB (Janus-Pro-1B, BF16, 576 VQ tokens)

## Phase 1: Code-Level Fixes (4.1x → ~100 tok/s) — IN PROGRESS

| Fix | Bottleneck | Location | Est. Speedup | Status |
|-----|-----------|----------|-------------|--------|
| 1 | Flash Attention v2 (O(S²) → O(S)) | `janus_llama.rs:186-219` | 2-3x | IMPLEMENTED |
| 2 | Remove .contiguous() after KV cat | `janus_llama.rs:153-163` | 1.2-1.4x | IMPLEMENTED |
| 3 | GPU-side argmax (eliminate PCIe sync) | `generate.rs:629-651`, `understand.rs:433-448` | 1.1-1.15x | IMPLEMENTED |
| 4 | Pre-allocate token tensor | `generate.rs:517-518` | 1.05-1.08x | IMPLEMENTED |
| 5 | Wire flash-attn feature flag | `lib.rs:136`, `janus_llama.rs` | prerequisite | IMPLEMENTED |

## Phase 2: Quantization (1.6x → ~160 tok/s)

| Technique | Expected Gain | Feasibility |
|-----------|--------------|-------------|
| NVFP4 weight quantization (Blackwell native) | 1.6x throughput | Easy — quantize model offline |
| FP8 KV cache quantization | 1.5x | Medium — via cuDNN v9 |
| AWQ quantization (Marlin kernel) | 2.6x vs GPTQ | Easy — tools available |

## Phase 3: System-Level (1.5-2x → ~250+ tok/s)

| Technique | Expected Gain | Feasibility |
|-----------|--------------|-------------|
| CUDA Graphs | 1.2-1.5x (eliminate CPU launch overhead) | Medium |
| Pre-allocated KV cache ring buffer | 1.2-1.4x (eliminate 95GB bandwidth waste) | Medium-Hard |
| Memory pinning | 1.1x | Easy |

## Phase 4: Advanced (batch throughput → 600+ tok/s)

| Technique | Expected Gain | Feasibility |
|-----------|--------------|-------------|
| Continuous batching (candle-vllm) | 2-4x throughput | Hard |
| Speculative decoding (draft model) | 2-3x latency | Hard |
| Tensor parallelism (2 GPUs) | 1.2-1.5x | Hard |
| Prefix caching | 2-57x for cached prompts | Medium |

## Performance Stacking (Realistic)

```
Baseline:                    25 tok/s
+ Flash Attention:           × 2.5 = 62.5 tok/s
+ Remove .contiguous():      × 1.3 = 81 tok/s
+ GPU sampling:              × 1.12 = 91 tok/s
+ Pre-alloc tokens:          × 1.06 = 96 tok/s
+ NVFP4 quantization:        × 1.6 = 154 tok/s
+ FP8 KV cache:              × 1.5 = 231 tok/s
+ CUDA Graphs:               × 1.2 = 277 tok/s
= Single request target:     ~250-280 tok/s
```

## Key Research Sources

- Flash Attention v3: github.com/michaelfeil/candle-flash-attn-v3
- NVFP4: developer.nvidia.com/blog/introducing-nvfp4
- candle-vllm: github.com/EricLBuehler/candle-vllm
- Blackwell SM 120 TMEM: 16 TB/s read, 8 TB/s write bandwidth
- cuDNN v9: Stream-K Flash Attention (200% decode speedup)
