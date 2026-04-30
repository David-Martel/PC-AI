# PC_AI Context: LLM Optimization Session — 2026-03-28

> **Context ID:** ctx-pcai-20260328-llm-opt
> **Created:** 2026-03-28T17:30:00Z
> **Branch:** main @ ed3b127
> **GPUs:** RTX 2000 Ada (8GB, SM 89) + RTX 5060 Ti (16GB, SM 120)

## Summary

Comprehensive LLM inference optimization session. Benchmarked 8 models, 4 backends, tested speculative decoding, validated hardware ceilings, deployed 3 new tools (ast-grep auto-fix, local LLM review, code quality benchmark), fixed SafeHandle P0, and produced 5 research reports.

## Performance Validated

| Model | Ollama | llama-server | pcai-ollama-rs | llama-bench |
|-------|--------|-------------|----------------|-------------|
| qwen2.5-coder:3b | 121 tok/s | 131.7 | **137.3** | **144.6** |
| qwen2.5-coder:7b | 64 | 83 | — | **78.4** |
| qwen3:30b (MoE) | **72** | — | — | — |
| gemma3:4b | 94 | — | — | — |

## Key Decisions

1. **7B at 150 tok/s not physically possible at Q4_K_M** — 448 GB/s / 4.36 GB = 103 ceiling
2. **qwen3:30b MoE is optimal** — 72 tok/s with frontier quality, beats 7B
3. **pcai-ollama-rs (137 tok/s) is fastest backend** — Rust SDK direct, no HTTP
4. **Speculative decoding blocked** by Ollama GGUF vocab format incompatibility
5. **PR #11 closed** — Jules didn't fix P0 (static string double-free), fixed correctly ourselves
6. **Dual GPU spread IS faster** than single GPU for 3B (121 vs 111 tok/s)
7. **Skip vLLM, TensorRT-LLM, SGLang** — Linux-only, wrong use case

## New Tools Deployed

| Tool | Purpose | Speed |
|------|---------|-------|
| `Tools/Invoke-AstGrepAutoFix.ps1` | ast-grep → local LLM → auto-fix | 20 findings in 60s |
| `Tools/Invoke-LocalLLMReview.ps1` | Multi-task code review (5 modes) | 10 files in 7s (triage) |
| `Tests/Benchmarks/Measure-CodeQuality.ps1` | Composite quality scoring | 51.2/100 baseline |

## Reports Generated

| Report | Key Finding |
|--------|------------|
| `llama-bench-20260328.md` | 144.6 tok/s raw ceiling on 5060 Ti |
| `MODEL_BENCHMARK_20260328.md` | 8 models compared, MoE wins |
| `MOE_MODEL_RESEARCH_20260328.md` | Nemotron/Qwen3/Mixtral analysis |
| `BACKEND_ACCELERATION_RESEARCH.md` | Ollama vs llama-server vs alternatives |
| `LOCAL_LLM_QUALITY_ASSESSMENT.md` | 3B/7B/30B quality-speed matrix |
| `PERFORMANCE_ANALYSIS_7B_150TOKS.md` | Physics-based 150 tok/s analysis |

## Next Session Priorities

1. **Download compatible 0.5B GGUF** from HuggingFace (not Ollama blob) for speculative decoding
2. **Test IQ3_XS quantization** of 7B model (~85 tok/s, 8% quality loss)
3. **Deploy llama-server** as alternative backend for speculative decode workloads
4. **Run Invoke-AstGrepAutoFix** on full codebase and apply safe fixes
5. **Review Jules PRs** — 2 sessions dispatched (pcai_core_lib dead code, Cleanup/CLI tests)
6. **Quality score improvement** — address 937 PSScriptAnalyzer violations (main drag on 51.2/100)
7. **Test VQ decode** end-to-end with GGUF quantized Janus model (dtype fix applied)
