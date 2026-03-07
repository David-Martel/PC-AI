# PC_AI Context: CLAUDE.md Audit & Update Session
**Date:** 2026-03-07
**Branch:** main @ 4e6a209
**Agent:** claude-md-improver

## Summary

Audited and updated the project root CLAUDE.md to reflect current project state. The file had drifted significantly — 274 modified files and 18 untracked files accumulated since last commit. The CLAUDE.md architecture tree was missing 5 Rust crates, the AI-Media directory, PcaiMedia module, and several new directories/configs.

## Changes Made

### CLAUDE.md Updates (6 targeted edits)
1. **Architecture tree**: Added 4 missing Rust workspace crates (`pcai_ollama_rs`, `pcai_media`, `pcai_media_model`, `pcai_media_server`), `AI-Media/` Python dir, `PcaiMedia.psm1` module, `Deploy/docker/` + `Deploy/rag-database/`, `PC-AI.ps1` + `Build.ps1` entry points, `Tests/`, `Scripts/`, `Notebooks/`, `Reports/` dirs, 3 new Config files
2. **Build.ps1 components**: Added `media`, `native`, `lint`, `fix` examples + complete 20-value `-Component` reference
3. **Rust test counts**: Updated 61+ to 65+, added `backends/llamacpp.rs` (2), `backends/mistralrs.rs` (2), renamed `Version` to `version.rs`
4. **Integration tests**: Expanded from 2 files to documenting 22 test files with category summary
5. **CI/CD workflows**: Expanded single file reference to table of all 9 workflow files
6. **CUDA targets**: Added SM 120 Blackwell (RTX 50 series)

## Current Architecture (Verified)

### Rust Workspace (6 crates)
- `pcai_inference` — LLM inference (HTTP + FFI, llama.cpp/mistral.rs backends)
- `pcai_core_lib` — Shared library (telemetry, fs, search, Windows APIs)
- `pcai_ollama_rs` — Ollama benchmark/integration tool (NEW, untracked)
- `pcai_media` — Media processing FFI DLL (Janus-Pro)
- `pcai_media_model` — Media model definitions
- `pcai_media_server` — Media HTTP server (axum)

### PowerShell Modules (13)
PC-AI.Acceleration, PC-AI.CLI, PC-AI.Cleanup, PC-AI.Common, PC-AI.Evaluation, PC-AI.Hardware, PC-AI.LLM, PC-AI.Network, PC-AI.Performance, PC-AI.USB, PC-AI.Virtualization, PcaiInference, PcaiMedia

### Test Coverage
- Rust unit tests: 65 (across 8 source files in pcai_inference)
- Integration/functional test files: 22 (Pester)
- CI workflows: 9

## Uncommitted State (274 modified + 18 untracked)

### Logical Clusters Identified
1. **Release/ deletion** — Entire `Release/PowerShell/PC-AI/` tree removed (~160 files)
2. **Module updates** — Changes across 6 PS modules (Acceleration, CLI, Common, Evaluation, LLM, Virtualization)
3. **Native/Rust** — Cargo workspace + pcai_inference config/http changes + C# programs
4. **Config** — hvsock-proxy, llm-config, settings
5. **Build/Tools** — Build.ps1, Invoke-RustBuild, New-PcaiPowerShellRelease
6. **Tests/Scripts** — Evaluation harness, unit tests, notebook generation
7. **Ollama benchmarks** — New untracked benchmark tool + reports
8. **Dev tooling** — New module status/bootstrap/mapped-tool scripts
9. **Docs** — CLAUDE.md, Evaluation README, optimization TODO
10. **Ignore patterns** — .gitignore, .rgignore

## Decisions
- **dec-001**: CLAUDE.md updates are factual accuracy fixes only — no style changes or content bloat added
- **dec-002**: `.tmp/peft-rs/` CLAUDE.md left untouched (vendored third-party, score A)

## Next Agent Recommendations
- `commit-cluster`: Batch the 274+18 files into semantic commit groups (NEXT)
- `code-reviewer`: Review module changes for quality before commit
- `test-runner`: Run Pester integration suite to verify nothing broke
