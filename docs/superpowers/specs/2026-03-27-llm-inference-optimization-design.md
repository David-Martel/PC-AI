# LLM Inference Optimization Design (Sub-project C)

> **Date:** 2026-03-27
> **Status:** In Progress
> **Author:** Claude Opus 4.6 + David Martel
> **Repo:** David-Martel/PC-AI
> **Part of:** 4-part testing/benchmarking/LLM optimization initiative (A → B → **C** → D)

## Completed

### VQ Decode Fix (GGUF Quantized Path)
**Root cause identified and fixed:** Non-LLM weights (VQ decoder, gen_head, post_quant_conv) were loaded with pipeline dtype (F16) in the GGUF path, but candle's GroupNorm/Conv2d don't support F16. The quantized backbone returns F32 from forward_hidden, creating a dtype chain break.

**Fix applied:**
- `generate.rs`: Compute `non_llm_dtype` separately (BF16 on CUDA, F32 on CPU) for GGUF path
- `lib.rs`: `project_to_image_vocab` cascades BF16→F32 dtype fallbacks

**Commit:** f630133

## Remaining (next session, informed by HF research)

### CUDA Graphs
- Eliminate 20-30% kernel launch overhead
- Requires static tensor shapes (no dynamic batch)
- candle support: experimental, may need custom implementation

### Speculative Decoding
- Self-speculative: use early exit from 12/24 layers as draft
- Measured at 24 tok/s for 1B (slower than full 36 tok/s — overhead too high for small models)
- Better candidate for 7B model where draft acceptance rate is higher

### FP8/NVFP4 Quantization
- Native Blackwell SM 120 support on RTX 5060 Ti
- Requires custom CUTLASS kernels or cuDNN 9.8+ support
- Potential 2x over Q4_K_M for 7B model

### KV Cache Optimization
- FunctionGemma: ring buffer KV cache (currently uses Tensor::cat)
- FunctionGemma: GPU Gumbel sampling (currently CPU to_scalar)
- pcai-inference: propagate applicable optimizations

## Success Criteria
1. GGUF Q4_K_M generates complete images end-to-end (VQ decode works)
2. Q4_K_M benchmark: target 45+ tok/s on RTX 5060 Ti
3. Research findings documented for CUDA Graphs and speculative decoding viability
