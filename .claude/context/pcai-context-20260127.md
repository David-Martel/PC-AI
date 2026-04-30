# PC_AI Project Context
**Context ID:** ctx-pcai-20260127-133151
**Created:** 2026-01-27T13:31:51Z
**Branch:** main @ 2bbbbbc
**Schema Version:** 2.0

## Project Summary

PC_AI is a local LLM-powered PC diagnostics and optimization agent for Windows 10/11 with WSL2 integration. The system uses a hybrid architecture combining PowerShell modules, Rust native DLLs, and C# interop for performance-critical operations.

### Architecture

```
PC_AI/
├── Modules/           # PowerShell modules (8 modules)
│   ├── PC-AI.LLM/     # LLM integration, FunctionGemma routing
│   ├── PC-AI.USB/     # USB diagnostics
│   ├── PC-AI.Virtualization/  # WSL2, Docker, HVSock
│   ├── PC-AI.Hardware/        # Hardware diagnostics
│   ├── PC-AI.Network/         # Network diagnostics
│   ├── PC-AI.Performance/     # Performance monitoring
│   ├── PC-AI.Cleanup/         # System cleanup
│   ├── PC-AI.Acceleration/    # Native DLL integration
│   └── PC-AI.CLI/             # NEW: Unified CLI module
├── Native/            # Native code
│   ├── pcai_core/     # Rust workspace (DLLs)
│   ├── PcaiNative/    # C# interop layer (refactored)
│   ├── PcaiChatTui/   # Terminal UI
│   └── NukeNul/       # Cleanup utility
├── Deploy/            # Deployment configs
│   └── functiongemma-finetune/  # FunctionGemma training
├── Config/            # Configuration files
├── Tests/             # Pester test suites
├── Tools/             # Development tooling
└── Reports/           # Generated documentation
```

## Current State

### Recent Session Work (2026-01-27)

**Commit Cluster Execution:**
- 13 atomic commits pushed to main
- 130+ files organized into semantic groups
- All changes now on origin/main

**Security Fixes Applied:**
| CVE | Package | Status |
|-----|---------|--------|
| CVE-2024-30105 | System.Text.Json | ✓ Fixed (8.0.5) |
| CVE-2024-43485 | System.Text.Json | ✓ Fixed (8.0.5) |
| CVE-2026-0994 | protobuf | ⚠️ No patch available |

**Dependency Modernization:**
| Ecosystem | Package | Old → New |
|-----------|---------|-----------|
| Rust | sysinfo | 0.32 → 0.38 |
| Rust | windows-sys | 0.59 → 0.61 |
| Rust | winreg | 0.52 → 0.55 |
| NuGet | Microsoft.PowerShell.SDK | 7.4.1 → 7.5.4 |
| pip | transformers | 4.41.0 → 4.47.0 |
| pip | torch | 2.3.0 → 2.5.0 |
| pip | protobuf | 4.25.0 → 6.31.1 |

### Work In Progress
- None currently active

### Blockers
- protobuf CVE-2026-0994 has no upstream fix yet (affects pure-Python only)

## Decisions

### DEC-001: Native Code Refactoring
- **Topic:** PcaiNative.cs monolithic file structure
- **Decision:** Split into 9 modular files (HelpExtractor, LlmClients, Models, NativeCore, PcaiCore, PcaiDiagnostics, PowerShellHost, SafetyInterlock, ToolExecutor)
- **Rationale:** Improved maintainability, separation of concerns
- **Date:** 2026-01-27

### DEC-002: Rust Telemetry Modularization
- **Topic:** telemetry.rs single file
- **Decision:** Split into telemetry/ module with submodules (mod, network, process, usb, usb_codes)
- **Rationale:** Better code organization, easier testing
- **Date:** 2026-01-27

### DEC-003: Sysinfo API Compatibility
- **Topic:** Breaking change in sysinfo 0.38 (temperature() returns Option<f32>)
- **Decision:** Use `.unwrap_or(0.0)` to maintain backward compatibility
- **Rationale:** Preserve existing API contract, 0.0 indicates "no reading"
- **Date:** 2026-01-27

## Patterns

### Coding Conventions
- PowerShell: Verb-Noun naming, comment-based help
- Rust: Standard Rust conventions, workspace dependencies
- C#: .NET 8 patterns, nullable reference types enabled

### Testing Strategy
- Pester for PowerShell (Tests/Unit/, Tests/Integration/)
- cargo test for Rust
- 85% minimum coverage target

### Error Handling
- PowerShell: try/catch with Write-Error
- Rust: Result<T, E> with proper propagation
- C#: Exceptions with proper cleanup

## Agent Work Registry

| Agent | Task | Files | Status | Handoff |
|-------|------|-------|--------|---------|
| commit-cluster | Cluster and commit 130+ files | Multiple | ✓ Complete | 13 commits pushed |
| security-auditor | CVE analysis via gh CLI | - | ✓ Complete | 2 CVEs fixed, 1 pending |
| rust-pro | Sysinfo 0.38 API fix | system.rs | ✓ Complete | - |

## Recommended Next Agents

1. **test-runner**: Verify all tests pass after major refactoring
2. **security-auditor**: Monitor protobuf CVE-2026-0994 for patches
3. **python-pro**: Update FunctionGemma training scripts for new deps

## Roadmap

### Immediate
- [ ] Run full test suite to verify refactoring
- [ ] Validate Rust DLL builds on clean checkout

### This Week
- [ ] Monitor protobuf for security patch
- [ ] Test FunctionGemma training with updated dependencies
- [ ] Update documentation for new module structure

### Tech Debt
- [ ] Remove unused GUID constants in vmm_health.rs (warnings)
- [ ] Add tests for new CLI module
- [ ] Document Native/PcaiNative refactored structure

## Key Files

### Entry Points
- `PC-AI.ps1` - Main CLI entry point
- `Native/pcai_core/pcai_core_lib/src/lib.rs` - Rust FFI exports

### Configuration
- `Config/pcai-tools.json` - Tool definitions for FunctionGemma
- `Config/llm-config.json` - LLM provider configuration
- `Config/hvsock-proxy.conf` - HVSocket aliases

### Documentation
- `DIAGNOSE.md` - Diagnostics system prompt
- `DIAGNOSE_LOGIC.md` - Branched reasoning decision tree
- `AGENTS.md` - Agent guidance (NEW)
- `ARCHITECTURE.md` - System architecture (NEW)

## Validation

- **Last Validated:** 2026-01-27T13:31:51Z
- **Git State:** Clean (after commit 2bbbbbc)
- **Build Status:** Rust ✓ (warnings only), .NET ✓
