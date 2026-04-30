# PC_AI Project Context

> **Last Updated**: 2026-01-23
> **Context Version**: 1.0
> **Purpose**: Local LLM-powered PC diagnostics and optimization agent for Windows 10/11 with WSL2 integration

---

## 1. Project Overview

### Goals and Objectives

- **Primary Goal**: Diagnose hardware issues, device errors, and system problems using local LLM inference
- **Target Platform**: Windows 10/11 development workstations with Docker, Hyper-V, and WSL2
- **LLM Integration**: Ollama and LM Studio for local model inference (qwen2.5-coder:7b default)
- **Analysis Workflow**: collect -> parse -> reason -> recommend

### Key Architectural Decisions

1. **7 PowerShell Modules** organized by domain responsibility:
   - `PC-AI.Hardware` - Device errors, disk health, USB status, network adapters, system events
   - `PC-AI.Network` - Network diagnostics, VSock performance, WSL connectivity
   - `PC-AI.Virtualization` - WSL status, Hyper-V, Docker, Defender exclusions
   - `PC-AI.USB` - USB device passthrough to WSL via usbipd-win
   - `PC-AI.Performance` - Disk space, process monitoring, disk optimization
   - `PC-AI.Cleanup` - PATH duplicate removal, temp files, duplicate file detection
   - `PC-AI.LLM` - Ollama/LM Studio integration, diagnostic analysis, chat interface
   - `PC-AI.Acceleration` - (Optional) Rust-accelerated tools (rg, fd, dust, procs)

2. **LLM System Prompts**:
   - `DIAGNOSE.md` - Defines assistant role, safety constraints, and workflow
   - `DIAGNOSE_LOGIC.md` - Branched reasoning decision tree for analysis

3. **Module Structure Pattern**:
   ```
   Modules/PC-AI.<Domain>/
   ├── PC-AI.<Domain>.psd1      # Module manifest
   ├── PC-AI.<Domain>.psm1      # Module loader
   ├── Public/                   # Exported functions
   │   └── *.ps1
   └── Private/                  # Internal helpers
       └── *-Helpers.ps1
   ```

### Technology Stack

| Component | Technology |
|-----------|------------|
| Language | PowerShell 5.1/7+ |
| Testing | Pester 5.x |
| LLM Backend | Ollama, LM Studio |
| Default Model | qwen2.5-coder:7b |
| USB Passthrough | usbipd-win |
| Virtualization | WSL2, Hyper-V, Docker Desktop |

### Team Conventions

1. **Return Types**: All public functions return `PSCustomObject` with consistent properties:
   - `Success` (bool)
   - `Message` (string)
   - `Severity` (string: OK/Warning/Error/Critical)
   - Domain-specific properties

2. **Encoding Compatibility**: Use `[System.IO.File]::WriteAllText($path, $content, [System.Text.Encoding]::UTF8)` instead of `Out-File -Encoding utf8` for PS5.1/PS7+ compatibility

3. **Admin Checks**: Implement inside function body, not via `#Requires -RunAsAdministrator`

4. **Parameter Validation**: Use `[ValidatePattern()]` for structured inputs (e.g., `^\d+-\d+$` for BusId)

---

## 2. Current State

### Recently Implemented Features

- All 7 modules created with public/private function structure
- Pester 5.x test framework with `.pester.ps1` runner
- 28 comprehensive tests for PC-AI.LLM module (100% passing)
- Module-scoped mocking with `-ModuleName` parameter
- MockData.psm1 fixture library for consistent test data

### Test Status (as of 2026-01-23)

| Module | Tests | Passed | Failed | Skipped |
|--------|-------|--------|--------|---------|
| PC-AI.Network | 24 | 24 | 0 | 0 |
| PC-AI.USB | 24 | 24 | 0 | 0 |
| PC-AI.Hardware | 33 | 33 | 0 | 0 |
| PC-AI.LLM | 28 | 28 | 0 | 0 |
| PC-AI.Cleanup | 26 | 26 | 0 | 0 |
| PC-AI.Performance | 22 | 19 | 3 | 0 |
| PC-AI.Virtualization | TBD | TBD | TBD | TBD |
| **Total** | ~175 | ~165 | ~10 | ~19 (admin) |

**Pass Rate**: ~79% (improved from 23% initial)

### Work in Progress

1. **Pester Test Fixes**:
   - TestDrive path concatenation issues in some tests
   - `CursorPosition` console errors in `Watch-VSockPerformance`
   - `Optimize-Disks` context failures

2. **Known Issues**:
   - Admin tests skipped on non-elevated runs
   - Some integration tests require live Ollama instance

### Performance Baselines

- Test execution time: ~60-90 seconds for full suite
- Module load time: <500ms per module
- Ollama API response: 2-30 seconds depending on model and prompt length

---

## 3. Design Decisions

### Architectural Choices

| Decision | Rationale |
|----------|-----------|
| Separate modules per domain | Single Responsibility Principle; easy maintenance |
| PSCustomObject returns | Consistent API for LLM parsing; scriptable output |
| Local LLM only | Privacy; no data leaves machine; works offline |
| Ollama as primary backend | Open source; good model library; easy to install |
| usbipd-win for USB | Official Microsoft-supported solution for WSL USB |

### API Design Patterns

1. **Diagnostic Functions** return structured objects:
   ```powershell
   [PSCustomObject]@{
       Name       = "Device Name"
       Status     = "OK"
       ErrorCode  = 0
       Severity   = "OK"
       Details    = @{}
   }
   ```

2. **Operation Functions** return result objects:
   ```powershell
   [PSCustomObject]@{
       Success = $true
       Message = "Operation completed"
       Data    = $resultData
   }
   ```

3. **LLM Functions** wrap API responses:
   ```powershell
   [PSCustomObject]@{
       Response = "LLM analysis text..."
       Model    = "qwen2.5-coder:7b"
       TokenCount = 500
       Duration = "2.5s"
   }
   ```

### Testing Strategies

1. **Module-Scoped Mocking**:
   ```powershell
   Mock Get-CimInstance {
       return Get-MockDeviceData
   } -ModuleName PC-AI.Hardware
   ```

2. **Admin Skip Pattern**:
   ```powershell
   BeforeAll {
       $script:IsAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
   }

   It "Should do admin thing" -Skip:(-not $script:IsAdmin) {
       # Test code
   }
   ```

3. **API Mocking Pattern**:
   ```powershell
   Mock Invoke-RestMethod {
       param($Uri)
       if ($Uri -match "/api/tags") {
           return Get-MockOllamaResponse -Type ModelList
       } elseif ($Uri -match "/api/chat") {
           return Get-MockOllamaResponse -Type Chat
       }
   } -ModuleName PC-AI.LLM
   ```

### Error Handling

1. **Graceful Degradation**: Functions return error objects rather than throwing
2. **Admin Check Inside Body**: Allows function to be called but return helpful error
3. **Verbose Logging**: Use `Write-Verbose` for debugging, `Write-Host` for user feedback
4. **Error Objects**: Include `Success = $false` and descriptive `Message`

---

## 4. Code Patterns

### Encoding Fix Pattern (CRITICAL)

```powershell
# WRONG - BOM issues in PS5.1
$content | Out-File -Path $path -Encoding utf8

# CORRECT - Consistent cross-version
[System.IO.File]::WriteAllText($path, $content, [System.Text.Encoding]::UTF8)
```

### Admin Skip Pattern for Tests

```powershell
BeforeAll {
    $script:IsAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

Describe "Admin-Required Tests" {
    It "Requires elevation" -Skip:(-not $script:IsAdmin) {
        # Test only runs when elevated
    }
}
```

### Mock Pattern for CIM Cmdlets

```powershell
# Mock actual cmdlets, not wrapper functions
Mock Get-CimInstance {
    param($ClassName)
    switch ($ClassName) {
        'Win32_PnPEntity' { return Get-MockDeviceData }
        'Win32_DiskDrive' { return Get-MockDiskData }
        'Win32_NetworkAdapter' { return Get-MockNetworkData }
    }
} -ModuleName PC-AI.Hardware
```

### Parameter Validation Pattern

```powershell
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^\d+-\d+$')]  # USB BusId format: 1-3, 18-4
    [string]$BusId,

    [Parameter()]
    [ValidateRange(1, 600)]
    [int]$Timeout = 120,

    [Parameter()]
    [ValidatePattern('^https?://')]
    [string]$ApiUrl = 'http://localhost:11434'
)
```

### PSCustomObject Return Pattern

```powershell
function Get-Something {
    [OutputType([PSCustomObject])]
    param()

    $result = [PSCustomObject]@{
        Success  = $false
        Message  = $null
        Data     = $null
        Severity = 'Unknown'
    }

    try {
        # Do work
        $result.Success = $true
        $result.Message = "Operation completed"
        $result.Severity = 'OK'
    }
    catch {
        $result.Message = "Operation failed: $_"
        $result.Severity = 'Error'
    }

    return $result
}
```

---

## 5. Agent Coordination History

### Session Timeline

| Round | Agents | Focus | Outcome |
|-------|--------|-------|---------|
| 1 | test-automator, explore | PC-AI.Network, PC-AI.USB | 24/24 tests each |
| 2 | Parallel agents | Hardware, LLM, Cleanup, Performance | Fixed 106+ tests |
| 3 | Admin-skip, encoding-fix | Cross-cutting concerns | PS5.1/PS7+ compat |

### Key Fixes Applied

1. **PC-AI.LLM.Tests.ps1** (28/28 passing):
   - Changed string matching to PSCustomObject property access
   - Fixed parameter names (`-System` not `-SystemMessage`)
   - Added comprehensive Invoke-RestMethod mocks
   - Exception testing with `Should -Throw`

2. **PC-AI.Hardware.Tests.ps1** (33/33 passing):
   - Module-scoped mocking
   - CIM cmdlet mocks for device data

3. **PC-AI.Cleanup.Tests.ps1** (26/26 passing):
   - TestDrive path handling
   - File operation mocks

4. **PC-AI.Performance.Tests.ps1** (19/22 passing):
   - Remaining issues: `Optimize-Disks` context failures

### Lessons Learned

1. **Mock at the right level** - Mock `Invoke-RestMethod` for API calls, not wrapper functions
2. **Verify return types** - Don't assume string when functions return objects
3. **Use `-ModuleName`** - Required for mocking inside module scope
4. **Test exceptions properly** - Use `Should -Throw` with `-ErrorAction Stop`

---

## 6. Future Roadmap

### Immediate (Next Session)

- [ ] Fix remaining 10-26 test failures
- [ ] Address `Optimize-Disks` context issues
- [ ] Fix `CursorPosition` console errors
- [ ] Run full test suite with coverage

### Short-Term

- [ ] Push to GitHub as public repository
- [ ] Add GitHub Actions CI workflow
- [ ] Create installation script for dependencies
- [ ] Document API for LLM integration

### Medium-Term

- [ ] Test with LM Studio agents using PC_AI API
- [ ] Add integration tests with live Ollama
- [ ] Build comprehensive diagnostic report generator
- [ ] Add scheduled task for periodic health checks

### Long-Term

- [ ] Add GPU diagnostics (NVIDIA, AMD)
- [ ] Implement fix/repair functions with safety checks
- [ ] Create TUI dashboard for monitoring
- [ ] Add support for additional LLM backends (llamafile, etc.)

---

## 7. Key Files Reference

### Core Files

| File | Purpose |
|------|---------|
| `DIAGNOSE.md` | LLM system prompt defining assistant behavior |
| `DIAGNOSE_LOGIC.md` | Branched reasoning decision tree |
| `Get-PcDiagnostics.ps1` | Core hardware diagnostics script |
| `PC-AI.ps1` | Main entry point script |
| `Tests/.pester.ps1` | Pester 5.x test runner |

### Module Paths

```
C:\Users\david\PC_AI\Modules\
├── PC-AI.Hardware\
├── PC-AI.Network\
├── PC-AI.Virtualization\
├── PC-AI.USB\
├── PC-AI.Performance\
├── PC-AI.Cleanup\
├── PC-AI.LLM\
└── PC-AI.Acceleration\
```

### Test Paths

```
C:\Users\david\PC_AI\Tests\
├── Unit\
│   ├── PC-AI.Hardware.Tests.ps1
│   ├── PC-AI.Network.Tests.ps1
│   ├── PC-AI.Virtualization.Tests.ps1
│   ├── PC-AI.USB.Tests.ps1
│   ├── PC-AI.Performance.Tests.ps1
│   ├── PC-AI.Cleanup.Tests.ps1
│   └── PC-AI.LLM.Tests.ps1
├── Integration\
│   ├── ModuleLoading.Tests.ps1
│   └── ReportGeneration.Tests.ps1
├── Fixtures\
│   └── MockData.psm1
└── PesterConfiguration.psd1
```

---

## 8. Quick Commands

### Run Tests

```powershell
# All tests
.\Tests\.pester.ps1 -Type All

# Unit tests only
.\Tests\.pester.ps1 -Type Unit

# With coverage (85% target)
.\Tests\.pester.ps1 -Type All -Coverage

# CI mode (exit codes, XML output)
.\Tests\.pester.ps1 -CI
```

### Import Modules

```powershell
Import-Module .\Modules\PC-AI.Hardware -Force
Import-Module .\Modules\PC-AI.LLM -Force
```

### Test LLM Connection

```powershell
Get-LLMStatus -TestConnection
```

### Run Diagnostics with LLM Analysis

```powershell
# Generate diagnostic report
.\Get-PcDiagnostics.ps1

# Analyze with LLM
Invoke-PCDiagnosis -DiagnosticReportPath "$env:USERPROFILE\Desktop\Hardware-Diagnostics-Report.txt"
```

---

## 9. Context Checkpoints

### Checkpoint: 2026-01-23 (Initial Context)

- 7 modules implemented
- 165+ tests passing (79% pass rate)
- LLM integration functional with Ollama
- Test framework operational with Pester 5.x

### Next Checkpoint Trigger

- When pass rate reaches 95%+
- When GitHub repository is created
- When significant architectural changes occur

---

*This context document should be updated after significant milestones or architectural changes.*

