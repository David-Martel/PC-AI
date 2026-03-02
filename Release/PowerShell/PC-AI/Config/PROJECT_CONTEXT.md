# PC_AI Project Context Summary

**Last Updated:** 2026-01-31
**Location:** `C:\Users\david\PC_AI`
**Status:** GitHub Ready - All Tests Passing

---

## Quick Overview

**PC_AI** is a local LLM-powered PC diagnostics and optimization framework for Windows 10/11 with WSL2 integration. It targets development workstations with Docker, Hyper-V, and cross-platform tooling.

### Core Workflow
```
collect -> parse -> reason -> recommend
```

### Key Stats
- **8 PowerShell Modules**
- **199 Tests** (100% pass rate, 19 skipped)
- **Pester 5.x** testing framework
- **pcai-inference** OpenAI-compatible local LLM
- **FunctionGemma** router for tool-calling
- **40x+ speedup** with Rust tools

---

## Module Summary

| Module | Description | Admin Required |
|--------|-------------|:--------------:|
| **PC-AI.Hardware** | Device manager, disk health, USB, network diagnostics | Yes |
| **PC-AI.Virtualization** | WSL2, Hyper-V, Docker status and optimization | Yes |
| **PC-AI.USB** | USB device management and WSL passthrough | Yes |
| **PC-AI.Network** | Network diagnostics, WSL connectivity, VSock | Yes |
| **PC-AI.Performance** | Disk optimization, resource monitoring | No |
| **PC-AI.Cleanup** | PATH cleanup, duplicate detection | Yes |
| **PC-AI.LLM** | pcai-inference + FunctionGemma integration | No |
| **PC-AI.Acceleration** | Rust tools + PS7+ parallelism | No |

---

## Essential Commands

### Testing
```powershell
# Run all tests
.\Tests\.pester.ps1 -Type All

# Run with coverage
.\Tests\.pester.ps1 -Type All -Coverage

# CI mode (exit codes + XML)
.\Tests\.pester.ps1 -CI
```

### Diagnostics
```powershell
# Core diagnostics (requires Admin)
.\Get-PcDiagnostics.ps1

# Unified CLI
.\PC-AI.ps1 diagnose all
.\PC-AI.ps1 diagnose wsl
.\PC-AI.ps1 diagnose hardware
```

### LLM Analysis
```powershell
# Check LLM status
Get-LLMStatus

# Run diagnostic analysis
Invoke-PCDiagnosis -ReportPath ".\report.txt"

# Set LLM configuration
Set-LLMConfig -DefaultModel "pcai-inference"
```

---

## Recent Work (2026-01-23)

### Completed
- Initial framework with 8 modules
- 199 Pester tests (100% pass)
- pcai-inference integration (HTTP + native FFI)
- USB module PS version fix (7.0 -> 5.1)
- Network test assertion fixes
- Legacy scripts archived
- CI/CD with GitHub Actions
- Rust tools acceleration layer

### Recent Fixes
1. **PC-AI.USB**: Changed PS requirement from 7.0 to 5.1
2. **PC-AI.Network**: Fixed WSL connectivity test assertions

---

## Architecture

### Directory Structure
```
PC_AI/
├── PC-AI.ps1                 # Unified CLI entry point
├── Get-PcDiagnostics.ps1     # Core diagnostics script
├── DIAGNOSE.md               # LLM system prompt
├── DIAGNOSE_LOGIC.md         # Decision tree
├── Modules/
│   ├── PC-AI.Hardware/
│   ├── PC-AI.Virtualization/
│   ├── PC-AI.USB/
│   ├── PC-AI.Network/
│   ├── PC-AI.Performance/
│   ├── PC-AI.Cleanup/
│   ├── PC-AI.LLM/
│   └── PC-AI.Acceleration/
├── Tests/
│   ├── Unit/                 # 7 test suites
│   ├── Integration/          # Module loading, reports
│   ├── Fixtures/             # MockData.psm1
│   └── .pester.ps1           # Test runner
├── Config/
│   ├── settings.json
│   ├── llm-config.json
│   └── diagnostic-thresholds.json
├── Legacy/                   # Archived scripts
└── .github/workflows/        # CI/CD
```

### Design Patterns
- **Module Structure**: Public/Private folders with dot-sourced functions
- **Output Format**: PSCustomObject with consistent properties
- **Safety**: Read-only by default, explicit consent for modifications
- **Testing**: BeforeAll mocks with Context-based organization

---

## LLM Configuration

### Default Provider: pcai-inference
```json
{
  "baseUrl": "http://127.0.0.1:8080",
  "defaultModel": "pcai-inference"
}
```

### Recommended Models
| Model | Size | Best For |
|-------|------|----------|
| GGUF: Llama 3 | Varies | General analysis |
| GGUF: Mistral | Varies | Fast general analysis |
| GGUF: Phi | Varies | Lightweight local analysis |
| GGUF: Gemma | Varies | High-quality responses |

---

## Agent Coordination

### For Future Sessions
- **Context File**: `Config/project-context.json`
- **Test Suite**: `Tests/.pester.ps1 -Type All`
- **Key Entry Point**: `PC-AI.ps1`

### Successful Patterns
- Use `search-specialist` for file discovery
- Use `architect-reviewer` for module structure reviews
- Check `PSScriptAnalyzerSettings.psd1` for linting rules

### Important Files for Agents
| Purpose | File |
|---------|------|
| Project guide | `CLAUDE.md` |
| LLM prompt | `DIAGNOSE.md` |
| Decision tree | `DIAGNOSE_LOGIC.md` |
| Test runner | `Tests/.pester.ps1` |
| Mock data | `Tests/Fixtures/MockData.psm1` |

---

## Technical Debt

### Identified
- Coverage tracking not fully in CI
- Some integration tests need real hardware
- Router availability not fully validated in all environments

### Planned
- Add codecov integration
- Mock hardware dependencies
- Complete pcai-inference + router parity across environments

---

## Future Roadmap

### v1.1
- GUI dashboard
- Scheduled diagnostics
- HTML/PDF reports

### v1.2
- Remote diagnostics
- Multi-machine aggregation
- Historical trends

### v2.0
- Cross-platform support
- Cloud LLM option
- Autonomous remediation

---

## ConfigManagerErrorCode Reference

| Code | Meaning |
|:----:|---------|
| 0 | Device working properly |
| 1 | Device not configured correctly |
| 10 | Device cannot start |
| 12 | Cannot find free resources |
| 22 | Device is disabled |
| 28 | Drivers not installed |
| 31 | Device not working properly |
| 43 | Device stopped responding |

---

## Safety Constraints

- **Read-only by default** - Diagnostics collect without modifications
- **Explicit consent** - Destructive operations require confirmation
- **Backup warnings** - BIOS/disk operations prompt for backup
- **Dry-run support** - Preview changes before execution

---

*This context was generated by the context-manager agent on 2026-01-23*
*Updated for pcai-inference architecture on 2026-01-31*

