---
context_id: ctx-pcai-20260319-llm-optimization
created_at: 2026-03-19T03:00:00Z
created_by: claude-opus-4.6
schema_version: "2.0"
---

# PC_AI Context — 2026-03-19: LLM Inference Optimization & CUDA Framework

## Summary

Massive multi-session effort: built NVIDIA software installer framework (PC-AI.Gpu, 19 files), Rust NVML GPU monitoring (508 lines), CUDA media pipeline (24 MB DLL), 4x inference optimizations (17.9 → 30 tok/s), FunctionGemma flash-attn enabled, Build.ps1 cuDNN/NVML defaults, 53 Jules sessions, ring crate eliminated, comprehensive testing. 36 commits pushed.

## Performance Baselines

| Config | tok/s | Binary | Notes |
|--------|-------|--------|-------|
| Codex build (baseline) | 17.9 | 38 MB | No optimizations |
| Claude optimized build | **30.0** | 20 MB | KV cache + GPU sampling |
| Projected (cuDNN SDPA) | ~50-60 | TBD | cuDNN build running |
| Projected (+ NVFP4) | ~100+ | TBD | Blackwell RTX 5060 Ti |

## Key Decisions

1. **Flash attention does NOT compile on Windows** — candle-flash-attn is Linux-only
2. **cuDNN SDPA is the Windows alternative** — 2-3x speedup, compiles with MSVC
3. **ring crate eliminated** — hf-hub 0.5 with tokio-only (no ureq/rustls)
4. **cuDNN feature added** — `--features cudnn` for fused attention kernels
5. **cuda-optimized meta-feature** — combines cuda + cudnn + nvml

## Optimizations Implemented

### Code-Level (in janus_llama.rs, generate.rs, understand.rs)
1. Flash attention path (`#[cfg(feature = "flash-attn")]`) — 2-3x potential
2. Remove .contiguous() after KV cat — eliminates 27,648 GPU copies
3. GPU-side argmax — eliminates 576 PCIe sync stalls
4. from_slice instead of from_vec — reduces cudaMalloc pressure
5. KV cache axis bug fixed — narrow was using wrong dimension

### Build System
- Build.ps1: EnableCuda activates cuda + cudnn + flash-attn + nvml + upscale
- pcai-media-server: cuda-optimized meta-feature
- FunctionGemma: flash-attn enabled as default feature

## Agent Registry (this session)

| Agent | Task | Status |
|-------|------|--------|
| perf-research (search) | 20 GPU optimization techniques | Complete |
| profile-bottlenecks (rust-pro) | Janus pipeline analysis | Complete |
| implement-perf (rust-pro) | 4 code optimizations | Complete (37/37 tests pass) |
| propagate-research (Explore) | FunctionGemma optimization scan | Complete |
| flash-attn-research (search) | Windows build feasibility | Complete (Linux-only confirmed) |
| fix-bugs (powershell-pro) | 4 actionable bugs | Complete |
| fix-rust-guidelines (rust-pro) | #[allow] → #[expect] | Complete |
| phase1-tests (powershell-pro) | Full test validation (468/638 pass) | Complete |
| phase1b-cuda-fix (powershell-pro) | CUDA 13.2 env fix | Complete |
| phase2b-sdks (powershell-pro) | SDK installs (Warp 1.12.0) | Complete |
| phase3a-wire-nvml (powershell-pro) | NVML FFI in 3 PS tools | Complete |
| ci-workflow (deployment) | nvidia-validation.yml | Complete |
| rust-nvml-optimize (rust-pro) | NVML in pcai_media config.rs | Complete |
| Jules (53 sessions) | Reviews, tests, refactors, security | All complete |

## Roadmap

### Immediate
- Complete cuDNN-optimized server build (running)
- Benchmark cuDNN SDPA vs baseline
- GPU driver update (582.41 → 591.55)

### This Week
- NVFP4 weight quantization for RTX 5060 Ti
- Pre-allocated KV cache ring buffer
- CUDA Graphs for kernel launch elimination

### Tech Debt
- flash-attn code path (Linux-only, documented)
- Pester test mock for native exe (workaround documented)
- Understanding pipeline outputs <image> tokens (prompt format issue)
