---
context_id: ctx-pcai-20260318-media-pipeline
created_at: 2026-03-18T00:00:00Z
created_by: claude-opus-4.6
schema_version: "2.0"
---

# PC_AI Context — 2026-03-18: Media Pipeline & Rust Guidelines

## Project State

- **Branch**: `main` @ `b5b76a9`
- **Project type**: Mixed (Rust + C# + PowerShell)
- **23 files modified** (uncommitted), 9 untracked files

## Summary

Active development on two fronts: (1) Janus-Pro media pipeline — Rust GPU generation/understanding with FFI DLL + PowerShell wrapper + axum HTTP server, and (2) Microsoft Pragmatic Rust Guidelines enforcement across all crates with lefthook, ast-grep, and clippy lints.

## Recent Commits (last 5)

| SHA | Description |
|-----|-------------|
| b5b76a9 | feat(inference): add token counting to FFI + rustfmt across all crates |
| 5d054f8 | chore: add lefthook.yml for declarative git hook management |
| 91e9398 | feat(rust): enforce Microsoft Pragmatic Rust Guidelines |
| 12ee5c6 | refactor(functiongemma): improve error handling with map_err |
| c5f1a40 | refactor(rust): replace 288 expect("TODO") with descriptive messages |

## Uncommitted Work In Progress

### Modified Files (23)

**Rust — pcai_media crate** (image generation/understanding via Janus-Pro):
- `pcai_media/src/config.rs` — media config updates
- `pcai_media/src/ffi/mod.rs` — FFI bindings for media operations
- `pcai_media/src/generate.rs` — image generation pipeline
- `pcai_media/src/lib.rs` — crate root
- `pcai_media/src/understand.rs` — image understanding pipeline

**Rust — pcai_media_model crate** (Janus-Pro model layer):
- `pcai_media_model/src/config.rs` — model configuration
- `pcai_media_model/src/generation_head.rs` — VQ generation head
- `pcai_media_model/src/janus_llama.rs` — LLaMA backbone for Janus
- `pcai_media_model/src/lib.rs` — model crate root

**Rust — pcai_media_server** (axum HTTP server):
- `pcai_media_server/src/main.rs` — media HTTP server

**Rust — other crates**:
- `pcai_core_lib/src/performance/optimizer.rs` — performance optimizer
- `pcai_core_lib/src/telemetry/event_log.rs` — telemetry
- `pcai_inference/src/http/mod.rs` — inference HTTP module
- `pcai_perf_cli/src/main.rs` — perf CLI

**PowerShell — media module**:
- `Modules/PcaiMedia.psd1` — media module manifest
- `Modules/PcaiMedia.psm1` — media module implementation
- `Config/pcai-media.json` — media configuration

**Build**:
- `Build.ps1` — unified build orchestrator updates
- `CLAUDE.md` — project documentation

**Tests (4 files)**:
- `Tests/Unit/PC-AI.Media.Tests.ps1` — unit tests
- `Tests/Functional/Media.Functional.Tests.ps1` — functional tests
- `Tests/E2E/Media.E2E.Tests.ps1` — end-to-end tests
- `Tests/Benchmarks/Benchmarks.Media.ps1` — performance benchmarks

### Untracked Files (9)

- `pcai_media/src/python_fallback.rs` — Python fallback for media ops
- `pcai_media_model/src/vision.rs` — vision tower implementation
- `pcai_media_model/tests/vision_tower_contract.rs` — vision contract tests
- `Reports/media/understand-red.b64` — test output (base64 image)
- `Reports/media/understand-red.png` — test output (PNG)
- `Tools/Invoke-JanusGpuSmoke.ps1` — GPU smoke test script
- `Tools/janus-understand.py` — Python understanding script
- `.codex-tmp/Janus-upstream/` — Codex temp (Janus reference)
- `.codex_tmp/Janus/` — Codex temp (Janus reference)

## Decisions

### DEC-001: Janus-Pro Rust GPU Pipeline
- **Decision**: Native Rust implementation of Janus-Pro using candle for GPU inference
- **Rationale**: 30x speedup over Python CPU (~17s GPU vs ~512s CPU for 576-step generation)
- **Key fixes**: `<begin_of_image>` token injection, KV cache detach, wte/lm_head CPU offload, VarMap early drop
- **VRAM**: Stable at ~4.2 GB on RTX 2000 Ada 8GB

### DEC-002: Microsoft Pragmatic Rust Guidelines Enforcement
- **Decision**: Enforce M-* guidelines via lefthook + ast-grep + clippy lints
- **Rationale**: Consistent quality across all David-Martel Rust repos
- **Scope**: 76 ast-grep rules (32 Rust), 7 clippy categories + 18 restriction lints
- **CI**: `rust-guidelines.yml` workflow (format, clippy, test, audit, feature-powerset)

### DEC-003: Multi-Crate Media Architecture
- **Decision**: Split media into 3 crates: pcai_media (FFI/API), pcai_media_model (model layer), pcai_media_server (HTTP)
- **Rationale**: Clean separation of concerns; model layer reusable, server optional

## Patterns

### Coding Conventions
- Rust: Microsoft Pragmatic Guidelines, `#[expect(lint)]` over `#[allow(lint)]`, mimalloc for binaries
- PowerShell: Verb-Noun cmdlets, Pester tests, `.psd1` manifests
- FFI: C ABI (`extern "C"`), `*const c_char` / `*mut c_char`, null-check all pointers

### Testing Strategy
- Rust: `cargo test --no-default-features --features server,ffi --lib` (unit)
- PowerShell: Pester across Unit/Functional/E2E/Benchmarks directories
- Integration: FFI tests via PowerShell P/Invoke (`Tests/Integration/FFI.*.Tests.ps1`)

### Error Handling
- Rust: `anyhow::Result` for applications, `thiserror` for libraries
- FFI boundary: Error codes (negative i32) + `pcai_last_error()` string buffer
- PowerShell: `try/catch` with structured error formatting

## Agent Registry

| Agent | Task | Status | Notes |
|-------|------|--------|-------|
| rust-pro | Media pipeline GPU optimization | Active (WIP) | 23 modified files |
| powershell-pro | PcaiMedia module wrapper | Active (WIP) | psm1/psd1 updates |
| test-automator | Media test suites | Active (WIP) | 4 test files updated |

## Recommended Next Agents

1. **rust-pro**: Continue media pipeline — commit WIP, resolve vision tower
2. **code-reviewer**: Review 23 modified files before commit
3. **test-automator**: Expand media test coverage (unit + integration)
4. **security-auditor**: Review FFI boundary for memory safety

## Roadmap

### Immediate
- Commit media pipeline WIP (23 modified + 9 untracked)
- Clean up `.codex-tmp/` and `.codex_tmp/` temp directories
- Validate Rust compilation across all modified crates

### This Week
- GPU smoke test (`Invoke-JanusGpuSmoke.ps1`) validation
- Image understanding pipeline completion
- Media server endpoint testing

### Tech Debt
- Python fallback (`python_fallback.rs`) — decide: keep or remove
- Vision tower contract test coverage
- `.codex-tmp/` directories should be gitignored

## Performance Baselines

| Pipeline | Device | Performance | Notes |
|----------|--------|-------------|-------|
| Rust pcai-media generate | GPU (bf16) | ~34 tok/s, ~17s | RTX 2000 Ada 8GB |
| Rust pcai-media generate | CPU (f32) | ~1.1 tok/s, ~512s | Baseline |
| Python batched CFG | GPU | 7.9 tok/s, 73s | Reference implementation |
| Content search (Rust FFI) | CPU | ~13ms | 143x vs PowerShell |
| File search (Rust FFI) | CPU | ~22ms | 68x vs PowerShell |
