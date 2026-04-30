# PC_AI Session Context — 2026-03-27/28

> **Context ID:** ctx-pcai-20260328-session
> **Created:** 2026-03-28T03:00:00Z
> **Branch:** main @ 866be10
> **Duration:** ~8 hours

## Summary

Massive session: built Jules integration stack, applied 30+ patches, created 4-part testing/benchmarking/LLM initiative, fixed critical VQ decode bug, diagnosed LLM performance bottlenecks, and produced a 200+ tok/s inference proposal.

## Commits This Session (PC-AI: 25+)

Key commits:
- `b91f406` Jules integration stack (3681 lines, 12 files)
- `7bb6379` SHA256 download validation security fix
- `15f757a` 5 Jules test + perf patches applied
- `b7c17a7` 10 hardcoded paths → env var lookups
- `2378660` O(N^2) fix, WMI hoist, clone removal, function extraction
- `d69b18b` 60 new unit tests + training config refactor
- `a989e4d` String clone fix + mistralrs RequestLike
- `a164607` Cross-platform test harness (Sub-project A)
- `624f93e` Benchmark gate + Rust profiler (Sub-project B)
- `f630133` VQ decode fix — dtype mismatch (Sub-project C, #1 blocker)
- `c7c6a05` 66 eval cases + regression runner (Sub-project D)
- `3d242e8` NumCtx=0 crash fix
- `a83dc8b` Auto-EnableTools loading 20B model fix
- `409a589` Context 32K, output 16K defaults
- `f2118b7` 200+ tok/s inference proposal

## Wezterm Commits (3)
- `d059c5bc0` All clippy/ast-grep fixes (29 files, +571/-252)
- `03b36d3a2` Lua literal sanitizer (security)
- PRs merged: #1 (TextInputDataInner), #3 (GCD toast), #5 (probe timeout), #6 (SSH security)

## Critical Findings

### RTX 5060 Ti Disconnected
The primary inference GPU (448 GB/s, SM 120 Blackwell, 16GB) is not connected. All inference runs on RTX 2000 Ada (192 GB/s, 8GB). This is the #1 bottleneck — 2.3x bandwidth penalty.

### LLM Configuration Bugs Fixed
1. pcai-ollama-rs defaulted to 128K context → OOM
2. Invoke-OllamaChat auto-enabled tools → loaded 20B model → OOM
3. NumCtx=0 passed through ValidateRange → crash
4. num_predict was 1024 (truncating output)

### VQ Decode Root Cause
Non-LLM weights (VQ decoder, gen_head, post_quant_conv) loaded as F16 for GGUF path. Candle GroupNorm/Conv2d don't support F16. Fixed: BF16 on CUDA, F32 on CPU.

## Next Session Priorities

1. **Reconnect RTX 5060 Ti** — immediate 2.3x improvement
2. **Build llama.cpp with CUDA Graphs** — 40% kernel speedup
3. **Download Qwen3-4B GGUF** — better model for the hardware
4. **Test VQ decode end-to-end** — verify GGUF image generation works
5. **Run Invoke-EvalRegression** with live backend to establish quality baseline
6. **Review any new Jules PRs** from the 6 active sessions
