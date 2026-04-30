# PC_AI Quick Context

> For rapid session restoration - read this first
> Updated: 2026-03-11 | Branch: main @ 5dc05d8 | Uncommitted: driver framework

## What Is This Project?

**PC_AI** is a local-first LLM-powered Windows diagnostics and optimization agent:
- 12 PowerShell modules + CLI entry point (PC-AI.ps1, 2066 lines)
- 5 Rust crates in pcai_core workspace (inference, media model, media pipeline, media server, core lib)
- C# P/Invoke bridge (PcaiNative, 18 .cs files)
- FunctionGemma tool router (Deploy/rust-functiongemma-*/*)
- Dual inference backends: llama.cpp + mistral.rs
- Media pipeline: Janus-Pro (image gen/understand) + RealESRGAN (upscale)

## Current State (2026-03-11)

| Component | Status |
|-----------|--------|
| 13 PowerShell Modules | All functional (NEW: PC-AI.Drivers) |
| PC-AI.Drivers module | NEW — scan/compare/install driver pipeline |
| driver-registry.json | NEW — 8 devices, vid_pid/friendly_name match rules |
| Install-InfDriver.ps1 | NEW — pnputil INF installer (bypasses broken setup.exe) |
| pcai_core_lib (pnp.rs) | UPDATED — driver_version/date/provider metadata |
| PcaiNative.dll | UPDATED — driver metadata P/Invoke |
| Realtek RTL8156/8157 | UPDATED to 1156.21.20.1110 / 1157.21.20.1110 |
| Rust tests | 197 total across 5 crates |
| Git status | Uncommitted: ~55 files (driver framework + prior changes) |
| Latest commit | 5dc05d8 |

## Latest Session: Driver Update Framework

Multi-session effort building driver management into PC_AI:

1. **PC-AI.Drivers module** -- Full scan/compare/install pipeline with registry-driven device matching
2. **Rust PnP enhancement** -- SetupDi registry API for driver_version/date/provider fields
3. **5-agent code review** -- 13 critical/high fixes (PS, Rust, C#, data, architecture)
4. **Realtek drivers installed** -- Extracted WinZip SFX with 7-Zip, installed INFs via pnputil
5. **Install-InfDriver.ps1** -- Reusable tool for future INF-based driver installations

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
| Driver Module | `Modules/PC-AI.Drivers/` (11 files) |
| Driver Registry | `Config/driver-registry.json` |
| Driver Orchestrator | `Tools/Update-Drivers.ps1` |
| INF Installer Tool | `Tools/Install-InfDriver.ps1` |
| PnP Rust Code | `Native/pcai_core/pcai_core_lib/src/telemetry/pnp.rs` |
| Rust Workspace | `Native/pcai_core/Cargo.toml` |
| Inference Crate | `Native/pcai_core/pcai_inference/` |
| Media Pipeline | `Native/pcai_core/pcai_media/` |
| C# Interop | `Native/PcaiNative/` (HardwareModule, InferenceModule, NativeResolver) |
| Build System | `Build.ps1` (2077 lines) |
| Config | `Config/llm-config.json`, `pcai-media.json`, `driver-registry.json` |

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

- **Latest Context**: `.claude/context/pcai-context-20260311-driver-framework.md`
- **Context Index**: `.claude/context/CONTEXT_INDEX.json`
- **Memory (master)**: `~/.claude/projects/C--codedev-pc-ai/memory/MEMORY.md`
- **Native Details**: `.claude/context/native-acceleration-context.md`
