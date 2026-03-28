# High-Performance Local LLM Inference Proposal

> **Date:** 2026-03-28
> **Author:** Claude Opus 4.6 + David Martel
> **Target:** 200+ tok/s sustained generation on local hardware

## Current State: Why We're at 13 tok/s

### Hardware
| GPU | VRAM | Bandwidth | Compute | Status |
|-----|------|-----------|---------|--------|
| RTX 2000 Ada | 8 GB GDDR6 | 192 GB/s | SM 89 | **Active** (GPU 0, display + Ollama) |
| RTX 5060 Ti | 16 GB GDDR7 | 448 GB/s | SM 120 | **Disconnected** (Thunderbolt eGPU, not detected) |

### Critical Finding
**The RTX 5060 Ti (2.3x faster bandwidth) is not connected.** All inference runs on the RTX 2000 Ada at 192 GB/s. This alone explains most of the performance gap.

### Bottleneck Analysis

LLM token generation is **memory-bandwidth-bound**. Each token requires reading the entire model from VRAM:

| Model | Size (Q4_K_M) | RTX 2000 Ada (192 GB/s) | RTX 5060 Ti (448 GB/s) |
|-------|---------------|-------------------------|------------------------|
| TinyLlama 1.1B | 0.6 GB | 320 tok/s theoretical | 747 tok/s theoretical |
| Qwen2.5-Coder 3B | 1.7 GB | 113 tok/s theoretical | 263 tok/s theoretical |
| Qwen2.5-Coder 7B | 4.0 GB | 48 tok/s theoretical | 112 tok/s theoretical |
| Qwen3 14B | 8.0 GB | 24 tok/s theoretical | 56 tok/s theoretical |

**At 35% efficiency** (typical for Ollama without CUDA Graphs):
- 3B on Ada: ~40 tok/s (we get 13 — Ollama HTTP overhead + Ada display load)
- 3B on 5060 Ti: ~92 tok/s
- 3B on 5060 Ti + CUDA Graphs (70% eff): **184 tok/s**

**To hit 200+ tok/s we need ALL of:**
1. The RTX 5060 Ti connected and active
2. CUDA Graphs enabled (eliminates kernel launch overhead)
3. Direct inference (not through Ollama HTTP)
4. Smaller or more efficient quantization (IQ4_XS or FP8)

## Proposal: Three-Phase Path to 200+ tok/s

### Phase 1: Infrastructure (immediate, 3-5x improvement)

#### 1A. Connect and configure RTX 5060 Ti
- Reconnect Thunderbolt 4 eGPU
- Verify with `nvidia-smi -L` (should show 2 GPUs)
- Set `CUDA_VISIBLE_DEVICES=1` for inference workloads (keep Ada for display)
- Expected improvement: **2.3x** (192→448 GB/s bandwidth)

#### 1B. Switch from Ollama to direct llama.cpp server
Ollama adds significant overhead:
- HTTP serialization/deserialization per request
- Process isolation (separate ollama_runner process per model)
- No CUDA Graphs support
- Generic scheduling (not optimized for single-user)

**Replace with:** Direct `llama-server` (llama.cpp's built-in server):
```powershell
# Download latest llama.cpp release with CUDA support
# Run server directly on GPU 1 (RTX 5060 Ti)
$env:CUDA_VISIBLE_DEVICES = "1"
llama-server -m Models/qwen2.5-coder-3b-q4_k_m.gguf `
  -ngl 99 -c 32768 -n 16384 `
  --flash-attn --cont-batching `
  -t 8 --host 127.0.0.1 --port 8080
```

Key flags:
- `-ngl 99`: All layers on GPU
- `--flash-attn`: Enable flash attention (supported on SM 89+)
- `--cont-batching`: Continuous batching for better GPU utilization
- No CUDA Graphs flag yet but llama.cpp auto-enables when beneficial

Expected improvement on RTX 2000 Ada alone: **2-3x** (eliminate Ollama overhead)
Combined with 5060 Ti: **5-7x** → **65-91 tok/s**

#### 1C. Download better quantized models
```powershell
# IQ4_XS with importance matrix — better quality, smaller
ollama pull qwen2.5-coder:3b-instruct-q4_K_M  # Already have
# Download GGUF directly for llama-server:
# bartowski/Qwen2.5-Coder-3B-Instruct-GGUF (HuggingFace)
# Use IQ4_XS variant if available
```

### Phase 2: CUDA Graphs + Kernel Optimization (2-4 weeks, 1.3-1.5x additional)

#### 2A. Build llama.cpp with CUDA Graphs
llama.cpp supports CUDA Graphs since late 2025 (NVIDIA contribution):
```bash
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89;120"
cmake --build build --config Release -j
```

Impact: **14-40% faster kernel execution** for the decode loop. On the 5060 Ti:
- Without CUDA Graphs: ~92 tok/s (35% efficiency)
- With CUDA Graphs: ~120-150 tok/s (50-65% efficiency)

#### 2B. FP8 quantization for Blackwell (SM 120)
The RTX 5060 Ti (Blackwell) has native FP8 tensor cores:
- FP8 E4M3 for weights: ~2x compute throughput vs Q4_K_M
- FP8 KV cache: halves cache memory
- Requires llama.cpp FP8 support or custom CUTLASS kernels

Impact: **1.5-2x** on the 5060 Ti specifically

#### 2C. Update pcai-inference to use latest llama.cpp
The current pcai-inference binary is from Feb 1, 2026. llama.cpp has added:
- CUDA Graphs support
- Flash attention improvements
- Better memory management
- Blackwell SM 120 optimizations

Rebuild: `.\Build.ps1 -Component llamacpp -EnableCuda -Clean`

### Phase 3: Speculative Decoding + Advanced (1-2 months, 1.5-2x additional)

#### 3A. Self-speculative decoding (SWIFT-style)
For the 3B model with 36 layers:
- Draft: use first 12 layers (early exit)
- Verify: run remaining 24 layers on draft tokens
- Accept rate for code: ~80% (structured, predictable)
- Net speedup: 1.3-1.6x

#### 3B. Multi-token prediction
Predict 2-4 tokens per forward pass using Medusa-style heads.
Requires training but could double throughput.

#### 3C. Tensor parallelism across both GPUs
Use both RTX 2000 Ada (8GB) and RTX 5060 Ti (16GB) simultaneously:
- Split model layers across GPUs
- 7B Q4_K_M fits in combined 24GB VRAM
- Inter-GPU communication via Thunderbolt 4 (40 Gbps)
- Expected: 7B at 80-100 tok/s (vs 6.4 tok/s current)

## Projected Performance Trajectory

| Configuration | 3B tok/s | 7B tok/s | Status |
|--------------|----------|----------|--------|
| Current (Ollama, Ada) | 13 | 6.4 | Baseline |
| + Direct llama-server | 30-40 | 15-20 | Phase 1B |
| + RTX 5060 Ti | 65-92 | 30-48 | Phase 1A |
| + CUDA Graphs | 120-150 | 50-70 | Phase 2A |
| + FP8 (Blackwell) | 150-200 | 70-100 | Phase 2B |
| + Speculative decode | **200-300** | **100-150** | Phase 3A |

## Better Models

The model choice matters as much as the framework:

| Model | Params | VRAM (Q4) | Strengths | HF Link |
|-------|--------|-----------|-----------|---------|
| Qwen2.5-Coder-3B | 3B | 1.7 GB | Best 3B coder | qwen/Qwen2.5-Coder-3B-Instruct |
| Qwen2.5-Coder-7B | 7B | 4.0 GB | Strong 7B coder | qwen/Qwen2.5-Coder-7B-Instruct |
| DeepSeek-Coder-V2-Lite | 2.4B (active) | 1.5 GB | MoE, fast | deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct |
| Qwen3-4B | 4B | 2.2 GB | Latest Qwen3, thinking mode | Qwen/Qwen3-4B |
| Gemma3-4B | 4B | 2.2 GB | Google, strong reasoning | google/gemma-3-4b-it |
| Phi-4-Mini | 3.8B | 2.1 GB | Microsoft, compact | microsoft/phi-4-mini-instruct |
| StarCoder2-7B | 7B | 4.0 GB | Code-specialized | bigcode/starcoder2-7b |

**Recommendation:** Qwen3-4B or Gemma3-4B — latest architecture, 4B sweet spot for quality vs speed on 16GB VRAM.

## Framework Architecture Change

### Current: Ollama → HTTP → pcai-ollama-rs → parse → PowerShell
**4 layers of overhead** for every token.

### Proposed: Direct llama.cpp server OR embedded inference
```
Option A (simpler): llama-server → OpenAI API → PowerShell
  - Single process, no Ollama overhead
  - CUDA Graphs, flash attention, continuous batching built-in
  - pcai-inference already supports this mode (llamacpp backend)

Option B (fastest): Embedded candle inference in pcai_inference
  - Zero HTTP overhead for local single-user
  - FFI direct to PowerShell via C#
  - Already exists but needs CUDA Graphs + flash-attn

Option C (production): llama.cpp library linked into pcai_inference
  - Best of both: llama.cpp's optimized kernels + Rust control
  - CUDA Graphs, flash-attn, FP8 all inherited from llama.cpp
  - Single binary, OpenAI-compatible API
```

**Recommendation:** Option A immediately (rebuild llama.cpp server with CUDA+Graphs), then Option C for the production path.

## Immediate Action Items

1. **Reconnect RTX 5060 Ti** (physical Thunderbolt cable)
2. **Download llama.cpp b5120+** (latest release with CUDA Graphs)
3. **Build with SM 89+120 CUDA support** and CUDA Graphs enabled
4. **Download Qwen3-4B GGUF** (IQ4_XS or Q4_K_M)
5. **Run llama-server directly** on GPU 1 with all layers offloaded
6. **Benchmark** with the same prompts used in this session
7. **Update pcai-inference** to wrap the new llama-server

Expected result: **100-150 tok/s** on Phase 1+2 alone, **200+** with Phase 3.
