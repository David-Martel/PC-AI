# llama-bench Results — March 28, 2026

## Hardware
- GPU: RTX 5060 Ti (16GB GDDR7, SM 120 Blackwell)
- CUDA: 13.1 runtime (13.2 toolkit)
- llama.cpp: b8565 (2026-03-28)

## qwen2.5-coder:3b (Q4_K_M, 1.79 GiB)

| Test | Tokens/s | Notes |
|------|----------|-------|
| Prompt processing (pp512) | **7,490 tok/s** | Flash attention ON |
| Token generation (tg512) | **144.6 tok/s** | Raw generation speed |

## Comparison

| Backend | Generation tok/s | Prompt tok/s | Overhead |
|---------|-----------------|--------------|----------|
| llama-bench (raw) | **144.6** | 7,490 | 0% (baseline) |
| llama-server (direct) | **131.7** | 197 | 9% |
| Ollama | **116-121** | 1,325 | 16-19% |

## Key Findings
- Raw hardware ceiling: 144.6 tok/s on 3B Q4_K_M
- llama-server adds ~9% overhead vs raw (HTTP + scheduling)
- Ollama adds ~16-19% overhead vs raw (HTTP + process isolation + scheduling)
- Flash attention provides ~7,490 tok/s prompt processing
- CUDA Graphs not yet tested (requires specific build flags)
