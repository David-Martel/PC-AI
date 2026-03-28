# MoE Model Research for PC_AI — March 28, 2026

## Hardware: RTX 2000 Ada (8GB) + RTX 5060 Ti (16GB) = 24GB combined

## Tier 1: Best Fit (tested, works well)

| Model | Type | Total/Active | VRAM (Q4) | tok/s | Quality |
|-------|------|-------------|-----------|-------|---------|
| **qwen3:30b** | MoE | 30B/3B | 18GB | **72 tok/s** | Best quality/speed on this hardware |
| nemotron-3-nano | MoE (Mamba2) | 30B/3.5B | 24GB | 15.9 tok/s | Math/reasoning strong, TB4 bottleneck |

## Tier 2: Feasible with tuning

| Model | Approach | VRAM | Notes |
|-------|----------|------|-------|
| Qwen3.5-35B-A3B | Q3_K_M or IQ4_XS | ~18-21GB | Vision+text, newest. Ollama mmproj issue may block vision |
| DeepSeek-V2-Lite | Q4_K_M | ~10-12GB | Fits in 5060 Ti alone. Decent but outclassed |
| Mixtral 8x7B | Q3_K_S | ~20GB | Outdated, outperformed by Qwen3/Nemotron |

## Not Feasible on 24GB

Nemotron-3-Super-120B, DeepSeek-V3 671B, Mixtral 8x22B, Llama 4 Maverick, Qwen3-235B-A22B, DBRX 132B

## Nemotron Family Details

| Model | Type | Params | MoE? | Notes |
|-------|------|--------|------|-------|
| Nemotron-Mini-4B | Dense | 4.2B | No | Fast but poor output structure |
| Nemotron-3-8B | Dense | 8B | No | Legacy |
| Nemotron-Nano-9B-v2 | Mamba2 hybrid | 8.9B | No | Novel architecture, not MoE |
| **Nemotron-3-Nano-30B** | **Mamba2+MoE** | **30B/3.5B** | **Yes** | 128 experts/layer, 6 active |
| Nemotron-3-Super-120B | Mamba2+MoE | 120B/12B | Yes | Too large for consumer |

Nemotron-3-Nano has 3.3x higher throughput than Qwen3-30B-A3B on identical hardware due to Mamba2 — but on our setup the 24GB model spans both GPUs via TB4 (40 Gbps), killing the throughput advantage. If it could fit in the 5060 Ti alone (with Q3 quantization at ~17GB), it would likely outperform Qwen3:30b.

## Backend Compatibility

| Backend | MoE Support | Speed | Notes |
|---------|------------|-------|-------|
| **Ollama** | Yes (via llama.cpp) | 72 tok/s (qwen3:30b) | Best ease of use |
| **pcai-ollama-rs** | Yes (Ollama SDK) | ~137 tok/s (dense 3B) | Fastest path, untested with MoE |
| **llama-server** | Yes | ~131 tok/s (dense 3B) | Fine-grained GPU control |
| **candle (pcai-inference)** | **No MoE support** | N/A | Dense models only |
| **mistral.rs** | Partial | Not built | Would need CUDA build |

## Key Insight

**qwen3:30b is the optimal MoE model** for this hardware:
- Fits in 5060 Ti alone (18GB at Q4_K_M in 16GB with dynamic offload)
- 72 tok/s with frontier-class quality
- Apache 2.0 licensed
- Full Ollama/llama.cpp/GGUF ecosystem support
- SWE-bench 69.2% (vs Nemotron 38.8%)

For throughput-critical loops, nemotron-3-nano would win IF it could fit in one GPU.
