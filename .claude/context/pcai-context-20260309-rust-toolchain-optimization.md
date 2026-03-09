# PC_AI Context: Rust Toolchain & CargoTools Optimization

**Context ID:** ctx-pcai-20260309-rust-toolchain
**Created:** 2026-03-09
**Branch:** main @ fa2755f
**Created by:** claude-opus-4.6

## State Summary

Multi-session optimization of Rust toolchain, CargoTools module (v0.9.0→v0.10.0), and lspmux LSP multiplexer integration. Added maturin/PyO3 binding support, cleaned stale toolchains, updated nightly to 1.96.0, and fixed pre-existing bugs in config generation.

### Recent Changes (this session)
- CargoTools v0.10.0: 5 new lspmux management functions (Start/Stop/Restart-LspmuxServer, Test-LspmuxHealth, Invoke-LspmuxMemoryReclaim)
- New `Private\Lspmux.ps1` module with TCP health checks, memory reclaim strategies (Soft/Medium/Hard)
- lspmux config.toml optimized: instance_timeout 600→300, gc_interval 15→10, pass_environment expanded 8→25+ vars
- Maturin wrapper enhanced: PyO3 ABI3 forward compat, venv Python auto-detect, CARGO_INCREMENTAL=0 for sccache
- Cargo config aliases: maturin-dev, maturin-build, maturin-publish, wasm-build, wasm-test
- Fixed `Get-DefaultRustAnalyzerConfig` bare-value bug (numThreads as integer instead of section)
- ConfigFiles tests updated: edition 2024, lld-link path matching, cachePriming threads
- Stale toolchain cleanup: 8 removed (4 dated nightlies + 4 old stables)
- All CargoTools tests passing: 52/52 ConfigFiles, 24/24 RA wrapper

### PC_AI Pending Changes (141 uncommitted)
- **Acceleration module**: Native FFI search/find enhancements, capabilities detection
- **Common module**: Shared cache, path resolution updates
- **Performance module**: Memory stats, new optimizer cmdlets (Get-PcaiMemoryPressure, Get-PcaiOptimizationPlan, Get-PcaiProcessCategories)
- **Native C#**: NativeCore/SearchModule/Models updates, new OptimizerModule.cs
- **Native Rust**: performance/optimizer.rs, search improvements (content, files, string)
- **Tests/Benchmarks**: Tooling benchmarks, data fabric, native performance
- **Reports**: Tool schema, backend coverage, benchmark results (20+ timestamped runs)
- **Docs**: AGENTS.md, ARCHITECTURE.md, README.md, optimization.TODO.md

## Decisions

| ID | Topic | Decision | Rationale |
|----|-------|----------|-----------|
| dec-001 | lspmux integration | Integrated into CargoTools as Private module | Centralizes Rust tooling management; CargoTools already manages sccache, RA, cargo config |
| dec-002 | lspmux GC tuning | instance_timeout=300, gc_interval=10 | More aggressive cleanup for memory efficiency; 5-min idle is sufficient for workspace switches |
| dec-003 | Memory reclaim | 3-tier strategy (Soft/Medium/Hard) | Soft for monitoring, Medium for targeted kills above threshold, Hard for full restart |
| dec-004 | Maturin env vars | Auto-set ABI3 compat, PEP517 release, auto-detect PYO3_PYTHON | Prevents common build failures across 26+ PyO3/maturin projects |
| dec-005 | ConfigFiles bug fix | numThreads moved into root section | Bare values at section level crash Merge-TomlConfig which calls .Keys on values |

## Agent Registry

| Agent | Task | Files Touched | Status |
|-------|------|---------------|--------|
| powershell-pro | Lspmux.ps1 creation | CargoTools/Private/Lspmux.ps1, CargoTools.psd1, CargoTools.psm1 | Complete |
| powershell-pro | ConfigFiles test fixes | Tests/ConfigFiles.Tests.ps1 | Complete |
| powershell-pro | Maturin optimization | wrappers/maturin.ps1, Private/Environment.ps1 | Complete |
| rust-pro | Cargo config aliases | T:\RustCache\cargo-home\config.toml, ~/.cargo/config.toml | Complete |

## Patterns

- **TOML config model**: All configs use `[ordered]@{ 'sectionName' = [ordered]@{ key = value } }` — root keys use empty string `''` section
- **Config merge strategy**: `Merge-TomlConfig` adds missing keys without overwriting — use `-Force` to override
- **Environment setup**: `Initialize-CargoEnv` sets env vars idempotently (only if not already set)
- **lspmux lifecycle**: Start → verify TCP listening → run with BelowNormal priority → monitor memory → reclaim if needed

## Roadmap

### Immediate
- Commit-cluster 141 PC_AI changes into semantic groups
- Push committed changes

### This Week
- Re-try `rustup component add rust-docs` (failed with rustup crash)
- Run full benchmark suite with optimized toolchain

### Tech Debt
- 3 flaky sccache mutex tests in CargoTools (pre-existing, not from our changes)
- PC_AI benchmark report cleanup (20+ timestamped dirs in Reports/tooling-benchmarks/)
