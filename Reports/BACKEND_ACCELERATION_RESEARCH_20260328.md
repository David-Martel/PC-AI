# Backend Acceleration Research — March 28, 2026

## Validated Results

| Backend | Generation | Prompt Eval | Overhead vs Raw |
|---------|-----------|-------------|-----------------|
| llama-bench (raw ceiling) | **144.6 tok/s** | 7,490 tok/s | 0% |
| llama-server (direct) | **131.7 tok/s** | 197 tok/s | 9% |
| Ollama (NEW_ENGINE + dual GPU) | **112-121 tok/s** | 3,600 tok/s | 16-19% |
| Ollama (5060 Ti only) | **109-111 tok/s** | N/A | 23% |

## Key Findings

### OLLAMA_NEW_ENGINE=1
- Available in Ollama 0.17+, significant prompt processing improvement
- Generation speed unchanged (already at memory bandwidth limit)
- Prompt eval improved from 1,325 to 3,600 tok/s (2.7x)

### Dual GPU vs Single GPU
- Dual GPU spread: 116-121 tok/s (better)
- 5060 Ti only: 109-111 tok/s (8% slower)
- Inter-GPU transfers are offset by additional memory bandwidth

### Ollama Overhead Analysis
- Total overhead vs raw: 16-19%
- Sources: HTTP serialization, Go runtime, process isolation
- llama-server direct: only 9% overhead
- Not worth switching for 7% — Ollama's model management + multi-GPU handling saves operational complexity

## Recommendations (from research)

### Do Now
- OLLAMA_NUM_PARALLEL=1, MAX_LOADED_MODELS=1 (single-user optimization)
- Keep SCHED_SPREAD=true (dual GPU IS faster)

### Next Steps
- Test Qwen3.5-35B-A3B MoE (30B total, 3B active params, fits in 24GB combined)
- Test speculative decoding via llama-server with 0.5B draft model
- Skip: vLLM, TensorRT-LLM, SGLang (all Linux-only or wrong use case)

### Frameworks Worth Evaluating
- ExLlamaV3 + TabbyAPI: best quantization, 52% faster than vLLM for local
- Only worthwhile for 14B+ models that don't fit in GGUF efficiently
