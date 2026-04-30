# PC_AI Project Context Summary

**Last Updated:** 2026-04-30
**Location:** `C:\codedev\PC_AI`
**Status:** Active native-first diagnostics and workstation reliability platform

---

## Quick Overview

**PC_AI** is a local LLM-powered PC diagnostics and optimization framework for
Windows 10/11. It now covers local LLM routing, native Rust/C# acceleration,
benchmark-driven tooling, media/Janus work, and workstation boot/sync
reliability.

### Core Workflow
```
collect -> parse -> route -> reason -> recommend
```

### Key Stats
- **PowerShell, Rust, and C# modules**
- **Native-first acceleration and inference surfaces**
- **Pester 5.x** testing framework
- **pcai-inference** OpenAI-compatible local LLM
- **FunctionGemma** router for tool-calling
- **40x+ speedup** with Rust tools
- **Boot/sync reliability tooling** for VHD startup, OneDrive, Process Lasso,
  and Filter Manager evidence capture

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
| **PC-AI.Gpu** | NVIDIA GPU inventory, preflight, and software status | No |
| **PC-AI.Drivers** | Driver inventory, Thunderbolt, and USB4 diagnostics | Yes |

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

# Boot/session tooling contracts
Invoke-Pester -Path .\Tests\Boot\PersistentVHDX.Tests.ps1,.\Tests\Boot\BootValidationTools.Tests.ps1
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

## Recent Work (2026-04-30)

### Completed
- Boot/session tools now expose `-h`, `--help`, and `-DryRun` contracts with
  Pester coverage.
- VHD startup one-liners were replaced with maintained wrappers, structured
  result JSON, transcripts, event-log integration, retries, and staggered
  Task Scheduler registration.
- Process Lasso policy now protects input/shell/vendor helper processes and
  de-elevates sync/build/GPU-overlay background competitors.
- OneDrive repair tooling captures Microsoft reference material, installer
  repair, reset/start attempts, WER evidence, and post-repair health checks.
- Risky registry and `~\bin` performance scripts were reviewed; known risky
  registry scripts now default to report-only behavior with explicit apply,
  dry-run, snapshot, restore, and cloud-sync preflight controls.
- Dependency-security updates were merged and stale remote branches/worktrees
  were cleaned up.

### Active Follow-Up

1. **OneDrive**: wait for a clean 60 minute sync-provider health window after
   the reset repair.
2. **Registry rollback**: validate persisted filesystem/cache rollback after a
   clean reboot sample.
3. **Cloud roots on VHDs**: decide whether Dropbox/Proton should be delayed
   until VHD mount health is proven.
4. **UDM startup**: keep SMB/rclone auto-launch disabled until OneDrive is
   stable or a rclone-only mode exists.
5. **LLM large context**: continue FunctionGemma and `pcai_inference`
   large-context/offload work from `llm.TODO.md`.

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

*Originally generated by the context-manager agent on 2026-01-23.*
*Reconciled for native-first PC-AI plus boot/sync workstation tooling on 2026-04-30.*

