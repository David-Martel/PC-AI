# Model Benchmark Results — March 28, 2026

## Hardware: RTX 2000 Ada (8GB) + RTX 5060 Ti (16GB), Ollama dual-GPU spread

## Performance by Model (diagnostic prompt, num_predict=2048)

| Model | Type | Total/Active Params | VRAM | tok/s | Quality |
|-------|------|-------------------|------|-------|---------|
| qwen2.5-coder:3b | Dense | 3B | 1.8GB | **115.4** | Terse, code-focused |
| nemotron-mini | Dense | 4B | 2.7GB | **99.8** | Fast but poorly structured |
| gemma3:4b | Dense | 4B | 3.3GB | **94.3** | Excellent balance |
| **qwen3:30b** | **MoE** | **30B/3B active** | **18GB** | **72.0** | **Best quality at speed** |
| qwen2.5-coder:7b | Dense | 7B | 4.0GB | **59.4** | Good code quality |
| deepseek-r1:7b | Dense | 7B | 4.7GB | **58.8** | Detailed reasoning |
| qwen3:14b | Dense | 14B | 9.3GB | **30.3** | Very thorough |
| nemotron-3-nano | MoE | 30B/3B active | 24GB | **15.9** | CoT reasoning, slow |

## Backend Comparison (qwen2.5-coder:3b)

| Backend | tok/s | Notes |
|---------|-------|-------|
| llama-bench (raw) | **144.6** | Hardware ceiling |
| pcai-ollama-rs (Rust) | **137.3** | Fastest framework path |
| llama-server (direct) | **131.7** | 9% overhead |
| Ollama HTTP | **115-121** | 16-19% overhead |

## Key Findings

1. **qwen3:30b MoE is the quality champion** — 30B total params, 3B active per token,
   72 tok/s. Frontier-class quality at 7B-class speed. Hits token limits (wants to generate more).

2. **pcai-ollama-rs at 137 tok/s** is the fastest framework path, 18% faster than Ollama HTTP.

3. **nemotron-3-nano (24GB)** doesn't fit efficiently — 15.9 tok/s due to cross-GPU overhead.
   The Thunderbolt 4 link (40 Gbps) is the bottleneck for models that span both GPUs.

4. **gemma3:4b** is the best balance for single-GPU use — 94 tok/s with excellent quality.

5. **nemotron-mini** is fast but output quality is poor for diagnostic use cases.

## Recommendations

- **Primary model**: qwen3:30b (MoE) for quality-critical tasks (diagnostics, analysis)
- **Speed model**: qwen2.5-coder:3b for quick code generation (115+ tok/s)
- **Balanced model**: gemma3:4b for general tasks (94 tok/s, good quality)
- **Backend**: pcai-ollama-rs for maximum throughput (137 tok/s)
