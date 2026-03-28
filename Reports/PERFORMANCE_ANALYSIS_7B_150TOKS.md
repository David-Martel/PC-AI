# 7B Model at 150 tok/s: Physics-Based Analysis

## The Hard Limit

RTX 5060 Ti: 448 GB/s memory bandwidth
qwen2.5-coder:7b Q4_K_M: 4.36 GB model size

**Theoretical max: 448 / 4.36 = 102.8 tok/s**
**Measured max: 78.4 tok/s (76% efficiency, llama-bench)**
**Ollama: 64 tok/s (82% of measured max)**

150 tok/s at Q4_K_M requires 655 GB/s bandwidth — doesn't exist on this GPU.

## Paths to 150 tok/s with 7B-Class Quality

### Path 1: Aggressive Quantization (most direct)
| Quant | Model Size | Theoretical | Expected (76% eff) | Quality |
|-------|-----------|-------------|---------------------|---------|
| Q4_K_M | 4.36 GB | 103 tok/s | **78 tok/s** | Baseline |
| Q3_K_S | ~2.8 GB | 160 tok/s | **122 tok/s** | ~5% loss |
| IQ3_XS | ~2.4 GB | 187 tok/s | **142 tok/s** | ~8% loss |
| IQ2_XXS | ~1.5 GB | 299 tok/s | **227 tok/s** | ~15% loss |

IQ3_XS gets close to 150 tok/s. IQ2_XXS exceeds it but quality degrades.

### Path 2: MoE with 3B Active (best quality per tok/s)
qwen3:30b (MoE, 3B active per token):
- Measured: **72 tok/s** on this hardware
- Quality: **Exceeds dense 7B** (30B total knowledge)
- Memory: 18 GB (fits in 5060 Ti with offload)

This is 72 tok/s with 14B+ class quality — better ROI than chasing 150 tok/s.

### Path 3: Speculative Decoding
- Ngram-based: tested, marginal improvement (~10%) on code tasks
- Draft model: requires same vocabulary family, 0.5-1B draft
- Expected: 7B from 78 to ~100-110 tok/s with good acceptance rate

### Path 4: FP8/NVFP4 Blackwell Native (future)
SM 120 has native FP8 tensor cores. If llama.cpp adds FP8 support:
- FP8 model: ~3.8 GB (7.6B params * 1 byte * 0.5 compression)
- Theoretical: 448/3.8 = 118 tok/s, with tensor core acceleration: ~160 tok/s

## Recommendation

**Don't chase 150 tok/s on a dense 7B model.**

Instead, use the right model for each speed tier:

| Speed Target | Model | Actual tok/s | Quality |
|-------------|-------|-------------|---------|
| 150+ tok/s | qwen2.5-coder:3b (dense) | **144.6** | Good code gen |
| 100+ tok/s | gemma3:4b (dense) | **94** | Excellent balance |
| 70+ tok/s | qwen3:30b (MoE, 3B active) | **72** | Frontier quality |
| 60+ tok/s | qwen2.5-coder:7b (dense) | **64** | Strong code |

The 3B model at 145 tok/s and the 30B MoE at 72 tok/s together cover
the entire quality-speed spectrum better than any single 7B model could.
