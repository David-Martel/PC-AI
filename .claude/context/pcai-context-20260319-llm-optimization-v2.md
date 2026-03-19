---
context_id: ctx-pcai-20260319-llm-optimization-v2
created_at: 2026-03-19T07:40:00Z
created_by: claude-opus-4.6
schema_version: "2.0"
---

# PC_AI Context — 2026-03-19: LLM Optimization Phase 2 (Quality + Multi-Token Research)

## Summary

Continued optimization session: 9 commits pushed (815d664..98f2009). Critical quality fix — argmax sampling was causing VQ codebook collapse (solid-color images); restored true multinomial sampling. Implemented PreAllocKvCache ring buffer (95GB bandwidth elimination), GPU Gumbel-max sampling, repetition penalty for understanding, image preprocessing via candle permute. Performance: 36 tok/s (2.0x vs 17.9 baseline). Quality: 4/5 prompts produce photorealistic images, understanding correctly describes them. Multi-token prediction research complete: Jacobi decoding (1.5-2x, no training) is top priority. CUDA binaries deployed (24MB DLL + 20MB server). 235/236 tests pass.

## Performance

| Config | tok/s | Quality | Notes |
|--------|-------|---------|-------|
| Baseline | 17.9 | N/A | No optimizations |
| Phase 1 (KV cache + GPU argmax) | 30.0 | N/A | 1.67x |
| Phase 4 (all, correct sampling) | **36.0** | 4/5 excellent | 2.0x, CPU multinomial |
| Phase 4 (GPU Gumbel-max) | 32.7 | 4/5 excellent | Slower for 16K vocab |
| Phase 4 (broken argmax) | 43.8 | 0/5 (solid colors) | INVALID |

## Key Decisions

1. **True multinomial sampling required** — greedy argmax causes VQ token collapse. CPU CDF walk is faster than GPU Gumbel for 16K vocab (64KB transfer < 7 kernel launches).
2. **cuDNN is a no-op** — cudarc 0.19.3 only has conv/softmax/pooling, not fused SDPA. Feature kept for forward compatibility.
3. **PreAllocKvCache default** — CacheVariant enum dispatches PreAlloc vs Dynamic via config flag.
4. **Jacobi decoding next** — training-free, 1.5-2x speedup, fixed 576-token output is ideal for Jacobi.

## Optimizations Applied (Phases 1-4)

### Code Changes
1. PreAllocKvCache ring buffer — scatter_set + narrow, eliminates ~95GB bandwidth
2. CacheVariant enum — dispatches prealloc/dynamic in generate.rs
3. GPU Gumbel-max sampling — available but CPU CDF is default for small vocab
4. True multinomial sampling — critical fix for image quality
5. Repetition penalty (1.2) — prevents understanding text loops
6. Image preprocessing via candle permute — replaces scalar loop
7. prefill_hidden clone eliminated — Option::take
8. Understanding pipeline fix — image_placeholder token, embedding splice
9. Inference scoping bug fix — prompt_tokens_est outside lock block
10. PcaiAsyncResult FFI size test — updated for 24-byte struct

### Build Artifacts
- `bin/pcai_media.dll` — 24 MB (CUDA SM 89)
- `pcai-media-server.exe` — 20 MB (CUDA SM 89)

## Agent Registry

| Agent | Task | Status |
|-------|------|--------|
| wire-prealloc (rust-pro) | Wire PreAllocKvCache into generate.rs | Complete |
| cuda-build (rust-pro) | Build CUDA binaries (SM 89) | Complete |
| perf-research (Explore) | 10 optimization opportunities identified | Complete |
| fix-cudnn (rust-pro) | Confirmed cuDNN is no-op for Janus | Complete |
| fix-understand (rust-pro) | 3 bugs fixed in understanding pipeline | Complete |
| kv-ring-buffer (rust-pro) | PreAllocKvCache implementation (8 tests) | Complete |
| fg-optimize (Explore) | FunctionGemma optimization scan | Complete |
| multi-token-research (general) | Multi-token prediction research | Complete |
| gpu-sampling (rust-pro) | GPU Gumbel-max implementation | Complete |

## Multi-Token Research Results

### No Training Required (Priority)
1. **Jacobi/SJD decoding** — 1.5-2x, init all 576 positions, iterate to convergence
2. **Pre-computed Gumbel noise** — 5-10%, batch 576 noise vectors at start
3. **Self-speculative** — 1.5-2x, use first 8/24 layers as draft model

### Requires Fine-Tuning
4. **Medusa heads** — 2.2-3.6x, 3 FFN heads (~400MB VRAM)
5. **GSD (Grouped Speculative)** — up to 3.7x, VQ codebook grouping
6. **MTP (Meta/DeepSeek)** — 1.5-2x, additional generation heads

## FunctionGemma Propagation Targets
- Ring buffer KV cache (model.rs:669-670, uses Tensor::cat)
- GPU sampling (model.rs:1130, uses CPU to_scalar after argmax)
- Remove unused streaming/block_len fields (model.rs:950-951)

## Test Results
| Suite | Pass/Total |
|-------|-----------|
| pcai-media | 75/75 |
| pcai-media-model (janus_llama) | 13/13 |
| pcai-inference | 61/61 |
| PC-AI.Gpu (Pester) | 14/14 |
| PC-AI.Media (Pester) | 72/73 |

## Roadmap

### Immediate
- Implement Jacobi decoding for image generation (1.5-2x, no training)
- Pre-compute Gumbel noise vectors (5-10%, trivial)
- CUDA Graphs for kernel launch elimination

### This Week
- Self-speculative decoding (first 8/24 layers as draft)
- FunctionGemma ring buffer + GPU sampling propagation
- Benchmark all techniques

### Tech Debt
- Understanding repetition penalty should be configurable
- Gumbel sampling decision should be vocab-size aware (>64K → GPU, <64K → CPU)
- Vision tower loading should be optional (generation-only mode)
