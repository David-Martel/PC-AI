# pcai_preflight — LLM Evaluation Tooling TODO

> GPU preflight readiness module for PC-AI. Tracks implementation progress.
> Plan: `docs/superpowers/plans/2026-03-30-pcai-preflight.md`
> Agent Bus Channel: `preflight-impl`

## Tasks

- [x] **Task 1**: GGUF Header Parser — types, binary reader, memory estimation (`preflight/gguf.rs`)
- [x] **Task 2**: VRAM Process Audit — NVML process query, PID resolution (`preflight/vram_audit.rs`)
- [x] **Task 3**: Preflight Verdict Logic — module root, go/warn/fail, wire into lib.rs (`preflight/mod.rs`)
- [x] **Task 4**: FFI Export — `pcai_gpu_preflight_json` C function (`lib.rs`)
- [x] **Task 5**: CLI Subcommand — `pcai-perf preflight` with exit codes (`main.rs`)
- [x] **Task 6**: PowerShell Wrapper — `Test-PcaiGpuReadiness` P/Invoke (`PC-AI.Gpu`)
- [x] **Task 7**: Rust Integration Tests (`preflight_tests.rs`)
- [x] **Task 8**: Pester Integration Tests (`FFI.Preflight.Tests.ps1`)
- [x] **Task 9**: Final Verification — full test suite, clippy, fmt, smoke test

## Completed

- **Task 1** (2026-03-30): `gguf.rs` — 582 lines, GgufModelMeta + GgufFileType (20 variants), binary parser, memory estimation, 9 unit tests. Clippy-clean with `#[expect]` annotations.
- **Task 2** (2026-03-30): `vram_audit.rs` — 285 lines, 2 types (GpuVramSnapshot, VramProcess), 6 unit tests. NVML process query with PID dedup + sysinfo name resolution.
- **Task 3+4** (2026-03-30): `mod.rs` + FFI export — Verdict logic (Go/Warn/Fail with 20% headroom), `check_readiness()`, `check_vram_state()`, `pcai_gpu_preflight_json` FFI. 21/21 tests pass. Fixed UsedGpuMemory import path in vram_audit.rs.
- **Task 5** (2026-03-30): `pcai-perf preflight` CLI — `--model`, `--ctx`, `--required-mb` flags. Exit codes 0/1/2. Worker mode support. Verified: both GPUs visible (RTX 2000 Ada 8GB + RTX 5060 Ti 16GB).
- **Task 6** (2026-03-30): `Test-PcaiGpuReadiness` PowerShell — FFI primary + CLI fallback. Added to PC-AI.Gpu module exports.
- **Task 7** (2026-03-30): `preflight_tests.rs` — 5/5 integration tests passing (0.32s). Graceful degradation without NVIDIA.
- **Task 8** (2026-03-30): `FFI.Preflight.Tests.ps1` — 8 passed, 4 skipped (need compiled nvml DLL). Auto-activate when DLL available.
- **Task 9** (2026-03-30): Final verification — 26/26 Rust tests pass. Format clean. Zero clippy errors in preflight (2 pre-existing in telemetry/). CLI smoke: both GPUs visible, verdicts correct, exit codes working.

## FunctionGemma Integration (next phase)

- [x] **FG-1**: Replace nvidia-smi subprocess in `Deploy/rust-functiongemma-core/src/gpu.rs` with NVML via pcai_core_lib
- [x] **FG-2**: Add preflight VRAM check in `Deploy/rust-functiongemma-runtime/src/inference.rs` before model load
- [x] **FG-3**: Test preflight CLI against FunctionGemma model paths
- [x] **FG-4**: Full CUDA 13.2 build validated — cudarc 0.19.4, patched candle-kernels/flash-attn with /Zc:preprocessor, SM 89+120

## Infrastructure Quick Wins (done this session)

- [x] Fix clippy errors in telemetry/process.rs and event_log.rs (needless_range_loop)
- [x] Pre-commit hook: scope ast-grep to staged files only (was scanning entire repo)
- [x] Activate sccache + lld-link via CargoTools Initialize-ProjectCargoConfig
- [x] Restart sccache (was unhealthy, 78% cache hit rate)
- [x] Update Cargo.lock: cudarc 0.19.3 → 0.19.4

## Performance Analytics (in progress)

- [x] **PERF-1**: GPU roofline model — theoretical decode/prefill ceilings per GPU+model combo
- [x] **PERF-2**: `pcai-perf roofline` CLI subcommand + `pcai_gpu_roofline_json` FFI
- [x] **PERF-3**: Bandwidth efficiency metrics (actual tok/s ÷ theoretical ceiling) — 51-55% measured
- [x] **PERF-4**: CI performance regression detection — `Invoke-PerfRegression.ps1` + 5-model baseline
- [x] **PERF-5**: Wire preflight into eval harnesses (Invoke-InferenceEvaluation + Invoke-OllamaBenchmarkSweep)
- [x] **PERF-6**: FunctionGemma training metrics — per-step fwd/bwd/opt timing, tok/s, GPU memory, convergence rate, training_metrics.json output

## FunctionGemma Training Remaining (from Deploy/rust-functiongemma-train/TODO.md)

- [ ] QLoRA NF4 quantization evaluation (qlora-rs path)
- [ ] Match FunctionGemma chat template behavior in runtime
- [ ] Deterministic generation settings for routing
- [ ] Port Python unit tests for dataset/schema handling
- [ ] Router eval harness against local runtime
- [x] Pre-allocated KV cache ring buffer (O(seq_len) → O(1) per decode token, auto-selected for non-int8)
- [ ] KV cache offload to CPU/disk
- [ ] Chunked softmax attention for large-context prefill
- [ ] CUDA memory pool evaluation (candle-cuda-vmm)
- [ ] GPUDirect Storage via cudarc for direct GPU<->disk transfer

## Notes

- No new Cargo dependencies needed (nvml-wrapper, sysinfo, memmap2 already in workspace)
- Exit codes: 0=go, 1=warn, 2=fail
- JSON output target: <1KB for typical 2-GPU system
- Tasks 1+2 are independent and can run in parallel
- Task 3 depends on 1+2; Task 4 depends on 3; Tasks 5+6 depend on 4; Tasks 7+8 depend on 5+6
