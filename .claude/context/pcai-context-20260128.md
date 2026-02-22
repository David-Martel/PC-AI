# PC_AI Project Context - 2026-01-28

## Summary

Major Rust workspace consolidation completed. Standalone crates (pcai_performance, pcai_search, pcai_system) have been merged into pcai_core_lib as submodules. FunctionGemma training infrastructure enhanced with Docker support and a new Rust-based trainer. Gitignore updated to exclude 570+ virtual environment files.

## Current State

| Aspect | Status |
|--------|--------|
| Git Branch | main @ 06092bb |
| Build Status | Requires validation after refactor |
| Native (Rust) | Consolidated into pcai_core_lib |
| Native (C#) | Updated for new Rust exports |
| PowerShell | New PC-AI.Common module added |
| Training | Docker + Rust trainer added |
| Tests | Updated for new structure |

## Session Commits (2026-01-28)

| SHA | Type | Description |
|-----|------|-------------|
| c561a13 | chore | Update gitignore and rgignore patterns |
| 8d8a38c | refactor | Consolidate pcai crates into pcai_core_lib modules |
| 939dfcc | refactor | Update C# native modules for Rust consolidation |
| dc69c1b | feat | Add PC-AI.Common module and tool parameter validation |
| 989ba0e | feat | Enhance FunctionGemma training with Docker and profiling |
| 0f619a7 | feat | Add Rust-based FunctionGemma trainer |
| 1200f97 | test | Update FFI tests and add new integration tests |
| 06092bb | chore | Update context and configuration files |

## Architecture Changes

### Rust Workspace (Before)
```
Native/pcai_core/
├── pcai_core_lib/       # Core library
├── pcai_performance/    # Standalone crate (REMOVED)
├── pcai_search/         # Standalone crate (REMOVED)
└── pcai_system/         # Standalone crate (REMOVED)
```

### Rust Workspace (After)
```
Native/pcai_core/
└── pcai_core_lib/
    └── src/
        ├── lib.rs           # Unified exports
        ├── hash.rs          # Hashing utilities
        ├── fs/              # File system ops
        │   ├── mod.rs
        │   └── ops.rs
        ├── performance/     # Disk, memory, process
        │   ├── mod.rs
        │   ├── disk.rs
        │   ├── memory.rs
        │   └── process.rs
        ├── search/          # File/content search
        │   ├── mod.rs
        │   ├── content.rs
        │   ├── duplicates.rs
        │   ├── files.rs
        │   └── walker.rs
        ├── system/          # Logs, paths
        │   ├── mod.rs
        │   ├── logs.rs
        │   └── path.rs
        └── telemetry/       # USB, hardware
            ├── mod.rs
            └── usb.rs
```

### New Projects Added

1. **Deploy/rust-functiongemma-train/** - Rust-based FunctionGemma training
   - High-performance training data generation
   - JSONL dataset handling
   - Evaluation metrics

2. **Modules/PC-AI.Common/** - Shared PowerShell utilities
   - Module-Common.ps1
   - Cross-module helper functions

## Gitignore Additions

```gitignore
# Virtual environments (WSL-style)
.venv_wsl/
**/.venv*/

# FunctionGemma training outputs
Deploy/functiongemma-finetune/out_model/
Deploy/functiongemma-finetune/unsloth_compiled_cache/
Deploy/functiongemma-finetune/*_data.jsonl
Deploy/functiongemma-finetune/test_vectors.json
Deploy/functiongemma-finetune/dummy_tools.json

# TensorBoard events
**/*.tfevents.*
**/runs/

# Deprecated code archives
.deprecated/

# Build error logs
build_errors*.txt
release_build_errors.txt

# Root coverage reports
/coverage.xml
```

## Agent Work Registry

| Agent | Task | Files | Status | Notes |
|-------|------|-------|--------|-------|
| commit-cluster | Semantic commit clustering | 84 files | Complete | 8 commits pushed |
| - | .gitignore/.rgignore audit | 2 files | Complete | Reduced 570+ untracked files |
| - | rust-analyzer memory analysis | - | Complete | Configuration recommendations provided |

## Rust-Analyzer Memory Analysis

### System State (Investigated)
- **Version**: rust-analyzer 1.93.0
- **Installations**: 3 copies found (VS Code extension, cargo bin, rustup)
- **VS Code Extensions**: rust-lang.rust-analyzer, AltinUludagi.cross-rust-analyzer, ArunMani.rust-syntax, ArkM.anycode-rust
- **Cargo.toml Files**: 8 workspaces contributing to memory usage

### Memory Optimization Tiers

**Minimal (~500MB)** - For constrained systems:
```json
{
  "rust-analyzer.lru.capacity": 32,
  "rust-analyzer.procMacro.enable": false,
  "rust-analyzer.cargo.buildScripts.enable": false,
  "rust-analyzer.checkOnSave": false,
  "rust-analyzer.server.extraEnv": {"RA_LRU_CAP": "32"}
}
```

**Balanced (~1-2GB)** - Recommended for this project:
```json
{
  "rust-analyzer.lru.capacity": 64,
  "rust-analyzer.procMacro.attributes.enable": true,
  "rust-analyzer.procMacro.server": null,
  "rust-analyzer.numThreads": 4,
  "rust-analyzer.cargo.buildScripts.rebuildOnSave": false,
  "rust-analyzer.diagnostics.disabled": ["unresolved-proc-macro"]
}
```

**Full (~2-4GB+)** - For comprehensive analysis:
```json
{
  "rust-analyzer.cargo.buildScripts.enable": true,
  "rust-analyzer.procMacro.enable": true,
  "rust-analyzer.check.command": "clippy"
}
```

### MCP Integration Options

| Option | Install | Notes |
|--------|---------|-------|
| rust-analyzer-mcp | `cargo install rust-analyzer-mcp` | Native Rust MCP wrapper |
| cclsp | `npx @anthropic/cclsp setup` | Generic LSP-to-MCP bridge |
| Claude Code LSP Plugin | Built-in | Requires configuration |

### Recommended Actions
1. Apply "Balanced" config to `.vscode/settings.json`
2. Disable unused Rust VS Code extensions
3. Consider rust-analyzer-mcp for Claude Code integration
4. Monitor with `cargo metadata --format-version=1` for workspace complexity

## Key File References

### Rust Native
- Workspace: `Native/pcai_core/Cargo.toml`
- Core lib: `Native/pcai_core/pcai_core_lib/src/lib.rs`
- Performance: `Native/pcai_core/pcai_core_lib/src/performance/mod.rs`
- Search: `Native/pcai_core/pcai_core_lib/src/search/mod.rs`
- System: `Native/pcai_core/pcai_core_lib/src/system/mod.rs`

### C# Native
- Core: `Native/PcaiNative/PcaiCore.cs`
- Performance: `Native/PcaiNative/PerformanceModule.cs`
- Search: `Native/PcaiNative/SearchModule.cs`
- System: `Native/PcaiNative/SystemModule.cs`

### PowerShell Modules
- LLM: `Modules/PC-AI.LLM/PC-AI.LLM.psm1`
- Common: `Modules/PC-AI.Common/PC-AI.Common.psm1`
- Validation: `Modules/PC-AI.LLM/Private/Validate-ToolParameters.ps1`

### Training
- Python: `Deploy/functiongemma-finetune/train_functiongemma.py`
- Docker: `Deploy/functiongemma-finetune/Dockerfile`
- Rust: `Deploy/rust-functiongemma-train/src/main.rs`

### Tests
- FFI Performance: `Tests/Integration/FFI.Performance.Tests.ps1`
- FFI Search: `Tests/Integration/FFI.Search.Tests.ps1`
- FFI System: `Tests/Integration/FFI.System.Tests.ps1`
- vLLM Health: `Tests/Integration/Test-vLLMHealth.ps1`
- E2E: `Tests/Integration/Verification.E2E.ps1`

## Immediate Next Steps

1. **Build Validation**
   - Run `cargo build --release` in Native/pcai_core
   - Verify DLL exports match C# P/Invoke declarations
   - Run `dotnet build` for PcaiNative

2. **Test Execution**
   - Run updated FFI tests against consolidated library
   - Validate PowerShell module loading

3. **Dependabot Alert**
   - Review: https://github.com/David-Martel/PC-AI/security/dependabot/1
   - High severity vulnerability reported

## Recommended Agents

| Agent | Reason |
|-------|--------|
| rust-pro | Validate consolidated workspace builds |
| test-runner | Execute FFI test suite |
| security-auditor | Review Dependabot high-severity alert |
| csharp-pro | Verify P/Invoke bindings |

## Tech Debt

- [ ] Remove deprecated code from `.deprecated/` after validation
- [ ] Add module documentation for new pcai_core_lib structure
- [ ] Create integration test for Rust FunctionGemma trainer
- [ ] Document Docker training workflow

## Dependencies

### Rust (pcai_core_lib)
- sysinfo: 0.38
- windows-sys: 0.61
- rayon: 1.10
- serde/serde_json: 1.0

### Python (FunctionGemma)
- torch: >=2.5.0
- transformers: >=4.47.0
- peft: >=0.14.0
- accelerate: >=1.2.0

### C# (PcaiNative)
- Microsoft.PowerShell.SDK: 7.5.4
- System.Text.Json: 8.0.5
