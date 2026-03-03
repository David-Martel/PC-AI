# PC_AI Quick Context

> For rapid session restoration - read this first
> Updated: 2026-03-03 | Branch: main @ a8c2d00 | 215 commits, no tags

## What Is This Project?

**PC_AI** is a local-first LLM-powered Windows diagnostics and optimization agent:
- 12 PowerShell modules + CLI entry point (PC-AI.ps1, 2066 lines)
- 5 Rust crates in pcai_core workspace (inference, media model, media pipeline, media server, core lib)
- C# P/Invoke bridge (PcaiNative, 18 .cs files)
- FunctionGemma tool router (Deploy/rust-functiongemma-*/*)
- Dual inference backends: llama.cpp + mistral.rs
- Media pipeline: Janus-Pro (image gen/understand) + RealESRGAN (upscale)

## Current State (2026-03-03)

| Component | Status |
|-----------|--------|
| 12 PowerShell Modules | All functional |
| pcai-mistralrs.exe | BUILT (84MB) |
| pcai_inference.dll | BUILT (4.1MB, in bin/) |
| PcaiNative.dll | BUILT (91KB, in bin/) |
| pcai-llamacpp | INCOMPLETE (artifact dir empty) |
| FunctionGemma runtime | INCOMPLETE (artifact dir empty) |
| pcai_media.dll | Code complete, needs build |
| Rust tests | 197 total across 5 crates |
| Git status | Clean (all committed) |
| Latest commit | a8c2d00 |

## Latest Session: CUDA/LLVM Toolchain Integration

26-file commit `a8c2d00` covering three workstreams:

1. **CargoTools enhanced** -- 5 new detection functions (GPU, CUDA, MSVC, LLVM, SDK), `Initialize-ProjectCargoConfig` generates machine-specific `.cargo/config.toml`
2. **Media pipeline hardened** -- rand-based RNG, SigLIP vision model wired, async FFI (generate/poll/cancel), RealESRGAN upscale module (ort, behind feature flag)
3. **Build infra** -- `.cargo/config.toml` gitignored (machine-specific), templates committed, lld-link preferred linker, NativeResolver.cs centralizes DLL resolution

## Active Blockers

| Blocker | Severity |
|---------|----------|
| CUDA driver 576.57 vs toolkit 13.1 (compile OK, runtime fails) | HIGH |
| Dependabot #1 protobuf CVE | HIGH |
| llamacpp backend build incomplete | MEDIUM |
| FunctionGemma runtime build incomplete | MEDIUM |

## GPU and Toolchain

| Component | Detail |
|-----------|--------|
| GPU 0 | Quadro RTX 4000, 8GB, SM 75 (Turing) |
| GPU 1 | RTX 5060 Ti, 16GB, SM 120 (Blackwell) |
| CUDA Toolkit | 13.1 |
| NVIDIA Driver | 576.57 (supports max CUDA 12.9 -- MISMATCH) |
| Linker | lld-link.exe (LLVM, preferred) |
| Rust | stable, sccache wrapper |

## Key Files

| Category | Path |
|----------|------|
| Rust Workspace | `Native/pcai_core/Cargo.toml` |
| Inference Crate | `Native/pcai_core/pcai_inference/` |
| Media Pipeline | `Native/pcai_core/pcai_media/` |
| Media Model | `Native/pcai_core/pcai_media_model/` |
| C# Interop | `Native/PcaiNative/` (InferenceModule, MediaModule, NativeResolver) |
| Build System | `Build.ps1` (2077 lines) |
| Cargo Templates | `Native/pcai_core/.cargo/config.toml.template` |
| Config | `Config/llm-config.json`, `pcai-media.json`, `pcai-functiongemma.json` |

## Quick Commands

```powershell
# Build everything
.\Build.ps1

# Build with CUDA
.\Build.ps1 -Component inference -EnableCuda

# Rust unit tests (no backend needed)
cd Native\pcai_core\pcai_inference
cargo test --no-default-features --features server,ffi --lib

# All workspace tests
cd Native\pcai_core
cargo test --workspace

# Regenerate .cargo/config.toml from toolchain
Initialize-ProjectCargoConfig -Path 'Native/pcai_core' -Force

# PowerShell tests
Invoke-Pester Tests/
```

## For Full Context

- **Latest Context**: `.claude/context/pcai-context-20260303-toolchain-integration.md`
- **Context Index**: `.claude/context/CONTEXT_INDEX.json`
- **Memory (master)**: `~/.claude/projects/C--codedev-pc-ai/memory/MEMORY.md`
- **Native Details**: `.claude/context/native-acceleration-context.md`
