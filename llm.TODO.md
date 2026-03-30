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
- [ ] **FG-4**: Validate both preflight + FunctionGemma under memory pressure (requires VS Developer Shell for full CUDA build)

## Notes

- No new Cargo dependencies needed (nvml-wrapper, sysinfo, memmap2 already in workspace)
- Exit codes: 0=go, 1=warn, 2=fail
- JSON output target: <1KB for typical 2-GPU system
- Tasks 1+2 are independent and can run in parallel
- Task 3 depends on 1+2; Task 4 depends on 3; Tasks 5+6 depend on 4; Tasks 7+8 depend on 5+6
