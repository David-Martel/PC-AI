# PC_AI Project Context

> Last Updated: 2026-01-23
> Context Version: 3.0.0
> Session Type: Test Fixing Campaign

## Quick Context (< 500 tokens)

**Current Task**: Fix Pester test failures - targeting 95%+ pass rate
**Immediate Goals**: Fix remaining ~26 test failures, push to GitHub as public repository
**Recent Decisions**: Module-scoped mocking with -ModuleName, admin-skip pattern, encoding fix for PS5.1/PS7+
**Active Blockers**: Optimize-Disks context failures, CursorPosition console errors
**Key Achievement**: Test pass rate improved from 23% to 79% (173/218 tests passing)

---

## 1. Project Overview

**Name**: PC_AI
**Type**: Local LLM-powered PC diagnostics and optimization framework
**Platform**: Windows 10/11 with WSL2 integration
**Version**: 1.0.0
**Primary Language**: PowerShell (5.1+ / 7.0+ for acceleration)

### Core Purpose
1. Diagnose hardware issues, device errors, and system problems
2. Analyze event logs, SMART status, and device configurations
3. Propose optimizations for disk, network, WSL2, virtualization, and system performance
4. Clean up duplicates, PATH entries, and unnecessary system artifacts
5. Local LLM analysis via Ollama with 11+ models (qwen2.5-coder:7b primary)

### Project Location
```
C:\Users\david\PC_AI\
```

---

## 2. Current State

### Module Status (All 8 Functional)

| Module | Purpose | Status | Functions |
|--------|---------|--------|-----------|
| PC-AI.Hardware | Device errors, disk health, USB, network adapters | Complete | 6 public |
| PC-AI.Virtualization | WSL2, Hyper-V, Docker management | Complete | 7 public |
| PC-AI.USB | USB/WSL passthrough via usbipd | Complete | 5 public |
| PC-AI.Network | Network diagnostics, VSock optimization | Complete | 5 public |
| PC-AI.Performance | Disk space, process monitoring, optimization | Complete | 4 public |
| PC-AI.Cleanup | PATH cleanup, temp files, duplicate detection | Complete | 4 public |
| PC-AI.LLM | Ollama integration, LLM chat, PC diagnosis | Complete | 6 public |
| PC-AI.Acceleration | Rust tool integration, parallel processing | **NEW** | 9 public |

### Unified CLI

Entry point: `C:\Users\david\PC_AI\PC-AI.ps1`

Commands:
- `diagnose` - Hardware and system diagnostics
- `optimize` - System optimization operations
- `usb` - USB device and WSL passthrough management
- `analyze` - LLM-powered diagnostic analysis
- `chat` - Interactive LLM chat interface
- `llm` - LLM configuration and status
- `cleanup` - System cleanup operations
- `perf` - Performance monitoring and analysis
- `status` - Overall system status summary

### Testing Framework

- **Pester 5.x** with configuration in `Tests/PesterConfiguration.psd1`
- **Coverage target**: 85%
- **CI/CD**: GitHub Actions with matrix testing (PS 5.1, PS 7.4)
- **Test files**: 7 unit test suites + 2 integration tests

---

## 3. Design Decisions

### Module Architecture

```
Modules/
  PC-AI.<Name>/
    PC-AI.<Name>.psd1     # Module manifest
    PC-AI.<Name>.psm1     # Module loader
    Public/               # Exported functions
    Private/              # Internal helpers
```

### Key Design Patterns

1. **CmdletBinding and OutputType** on all public functions
2. **Module manifests export only public functions** via FunctionsToExport
3. **Private helpers in Private/ folder** for internal use only
4. **Lazy module loading** in CLI for fast startup
5. **Graceful fallback** when tools unavailable

### Acceleration Strategy (3-Tier Fallback)

```
Rust Tool (fastest) -> PS7+ Parallel -> Sequential PowerShell
```

Example from `Get-ProcessesFast`:
```powershell
if ($useProcs -and $RawOutput) {
    return Get-ProcessesWithProcs @PSBoundParameters -ProcsPath $procsPath
}
else {
    return Get-ProcessesParallel @PSBoundParameters  # PS7+ ForEach-Object -Parallel
}
```

### Parallel Processing Pattern (PS7+)

```powershell
$results = $items | ForEach-Object -Parallel {
    # Parallel block
} -ThrottleLimit $throttleLimit
```

**Why PS7+ native parallelism?**
- Replaces broken .NET Parallel.ForEach pattern
- Better integration with PowerShell pipeline
- Automatic throttling based on processor count
- Cleaner error handling

---

## 4. Code Patterns

### Standard Function Template

```powershell
#Requires -Version 5.1
function Verb-Noun {
    [CmdletBinding()]
    [OutputType([PSCustomObject[]])]
    param(
        [Parameter(Position = 0)]
        [ValidateSet('value1', 'value2')]
        [string]$Parameter
    )

    # Implementation
}
```

### Rust Tool Detection with Caching

```powershell
$rustPath = Get-RustToolPath -ToolName 'rg'
$useRust = $null -ne $rustPath -and (Test-Path $rustPath)
```

### Error Handling

```powershell
try {
    $result = & $toolPath @args 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Tool failed with exit code $LASTEXITCODE"
        return Invoke-Fallback @PSBoundParameters
    }
    return $result
}
catch {
    Write-Warning "Tool execution failed: $_"
    return Invoke-Fallback @PSBoundParameters
}
```

---

## 5. Recent Bug Fixes

### 5.1 Get-DiskUsageFast.ps1:90 - Switch Statement Syntax

**Problem**: Return statements inside switch case blocks caused unexpected behavior

**Fix**: Moved return outside switch, used variable assignment
```powershell
# Before (broken)
switch ($sortBy) {
    'size' { return $results | Sort-Object Size -Descending }
}

# After (fixed)
$results = switch ($sortBy) {
    'size' { $results | Sort-Object Size -Descending }
}
return $results
```

### 5.2 Find-FilesFast.ps1 - fd Argument Ordering

**Problem**: fd CLI expects pattern first, then options, then path last

**Fix**: Reordered arguments
```powershell
# Before (broken)
$args = @('--type', 'f', $path, $pattern)

# After (fixed)
$args = @($pattern, '--type', 'f', $path)
```

### 5.3 Get-ProcessesFast.ps1 - .NET Parallel.ForEach Replacement

**Problem**: .NET Parallel.ForEach didn't work correctly in PowerShell context

**Fix**: Replaced with PS7+ ForEach-Object -Parallel
```powershell
# Before (broken)
[System.Threading.Tasks.Parallel]::ForEach($processes, [Action[object]]{ ... })

# After (fixed)
$processes | ForEach-Object -Parallel {
    $proc = $_
    # Process in parallel
} -ThrottleLimit $throttleLimit
```

### 5.4 Get-FileHashParallel.ps1 - Include Parameter

**Problem**: -Include parameter requires -Recurse to work

**Fix**: Added -Recurse when using -Include
```powershell
# Before (broken)
Get-ChildItem -Path $Path -Include $Extensions

# After (fixed)
Get-ChildItem -Path $Path -Include $Extensions -Recurse
```

---

## 6. Key Files

### Core Framework
| File | Purpose |
|------|---------|
| `C:\Users\david\PC_AI\PC-AI.ps1` | Unified CLI entry point |
| `C:\Users\david\PC_AI\DIAGNOSE.md` | LLM system prompt with safety constraints |
| `C:\Users\david\PC_AI\DIAGNOSE_LOGIC.md` | Decision tree for diagnostic analysis |
| `C:\Users\david\PC_AI\Get-PcDiagnostics.ps1` | Legacy standalone diagnostics |
| `C:\Users\david\PC_AI\CLAUDE.md` | Project documentation |

### Acceleration Module
| File | Purpose |
|------|---------|
| `Modules/PC-AI.Acceleration/PC-AI.Acceleration.psd1` | Module manifest |
| `Modules/PC-AI.Acceleration/Private/Initialize-RustTools.ps1` | Tool detection and caching |
| `Modules/PC-AI.Acceleration/Public/Get-RustToolStatus.ps1` | Tool status reporting |
| `Modules/PC-AI.Acceleration/Public/Search-ContentFast.ps1` | ripgrep wrapper |
| `Modules/PC-AI.Acceleration/Public/Find-FilesFast.ps1` | fd wrapper |
| `Modules/PC-AI.Acceleration/Public/Get-ProcessesFast.ps1` | procs wrapper + PS7 parallel |
| `Modules/PC-AI.Acceleration/Public/Get-FileHashParallel.ps1` | Parallel hashing |
| `Modules/PC-AI.Acceleration/Public/Find-DuplicatesFast.ps1` | Fast duplicate detection |
| `Modules/PC-AI.Acceleration/Public/Get-DiskUsageFast.ps1` | dust wrapper |
| `Modules/PC-AI.Acceleration/Public/Measure-CommandPerformance.ps1` | hyperfine wrapper |
| `Modules/PC-AI.Acceleration/Public/Compare-ToolPerformance.ps1` | Benchmarking utilities |

### Testing
| File | Purpose |
|------|---------|
| `Tests/.pester.ps1` | Pester test runner |
| `Tests/PesterConfiguration.psd1` | Pester configuration (85% coverage target) |
| `Tests/Unit/PC-AI.*.Tests.ps1` | Unit tests for each module |
| `Tests/Integration/ModuleLoading.Tests.ps1` | Module import tests |

### CI/CD
| File | Purpose |
|------|---------|
| `.github/workflows/powershell-tests.yml` | Main CI pipeline (lint, test, integration) |
| `.github/workflows/security.yml` | Security scanning |
| `.github/workflows/release.yml` | Release automation |

---

## 7. Rust Tool Integration

### Available Tools (8 installed)

| Tool | Name | Use | Accelerates |
|------|------|-----|-------------|
| rg | ripgrep | Fast text search | Search-LogsFast, Search-ContentFast |
| fd | fd | Fast file finder | Find-FilesFast, Find-DuplicatesFast |
| procs | procs | Modern process viewer | Get-ProcessesFast |
| bat | bat | Syntax highlighting | Show-FileWithHighlighting |
| hyperfine | hyperfine | Command benchmarking | Measure-CommandPerformance |
| tokei | tokei | Code statistics | Get-CodeStatistics |
| eza | eza | Modern ls | Get-DirectoryListingFast |
| sd | sd | Find & replace | Text replacement |

### Missing Tools (2 not installed)

| Tool | Name | Use | Install Command |
|------|------|-----|-----------------|
| dust | dust | Disk usage analyzer | `cargo install du-dust` or `winget install dust` |
| btm | bottom | System monitor | `cargo install bottom` or `winget install bottom` |

### Performance Benchmarks

From actual testing:
- **ripgrep vs Select-String**: 44.6x faster (0.892s vs 0.020s for 1000+ files)
- **fd vs Get-ChildItem -Recurse**: ~10x faster for large directories
- **procs vs Get-Process + CIM**: ~3x faster with better formatting

---

## 8. Future Roadmap

### Immediate (Next Session)
- [ ] Install dust and btm for complete Rust tool coverage
- [ ] Add acceleration tests to Pester suite
- [ ] Create benchmark documentation

### Short-Term
- [ ] Integrate acceleration functions into PC-AI.Performance
- [ ] Add Search-LogsFast to PC-AI.Hardware for event analysis
- [ ] Benchmark all accelerated functions systematically

### Medium-Term
- [ ] Consider tree-sitter integration for code analysis
- [ ] Add GPU diagnostics module
- [ ] Streaming LLM analysis with progress indicators

### Long-Term
- [ ] Python wrapper for faster local LLM inference
- [ ] Automated optimization workflows
- [ ] Report archival and trending

---

## 9. Directory Structure

```
C:\Users\david\PC_AI\
+-- .claude/
|   +-- context/
|       +-- project-context.md    # This file
|       +-- quick-context.md      # Rapid restoration
+-- .github/
|   +-- workflows/
|       +-- powershell-tests.yml  # CI pipeline
|       +-- security.yml
|       +-- release.yml
+-- Config/
|   +-- settings.json
|   +-- llm-config.json
+-- Modules/
|   +-- PC-AI.Hardware/
|   +-- PC-AI.Virtualization/
|   +-- PC-AI.USB/
|   +-- PC-AI.Network/
|   +-- PC-AI.Performance/
|   +-- PC-AI.Cleanup/
|   +-- PC-AI.LLM/
|   +-- PC-AI.Acceleration/       # NEW
+-- Reports/
+-- Tests/
|   +-- Unit/
|   +-- Integration/
|   +-- PesterConfiguration.psd1
+-- CLAUDE.md
+-- DIAGNOSE.md
+-- DIAGNOSE_LOGIC.md
+-- Get-PcDiagnostics.ps1
+-- PC-AI.ps1                     # Unified CLI
+-- PSScriptAnalyzerSettings.psd1
```

---

## 10. Session Restoration Instructions

### Quick Restoration
1. Read quick context: `C:\Users\david\PC_AI\.claude\context\quick-context.md`
2. Check module status: `.\PC-AI.ps1 status`

### Full Restoration
1. Read this file: `C:\Users\david\PC_AI\.claude\context\project-context.md`
2. Review CLAUDE.md: `C:\Users\david\PC_AI\CLAUDE.md`
3. Check Rust tools: Import-Module .\Modules\PC-AI.Acceleration; Get-RustToolStatus
4. Run tests: `.\Tests\.pester.ps1`

### Verify Project Health
```powershell
# Check CLI works
.\PC-AI.ps1 status

# Check all modules load
.\Tests\.pester.ps1 -Path .\Tests\Integration\ModuleLoading.Tests.ps1

# Check acceleration module
Import-Module .\Modules\PC-AI.Acceleration -Force
Get-RustToolStatus | Format-Table Tool, Available, Version -AutoSize
```

---

## 11. Context Metadata

```yaml
project: PC_AI
location: C:\Users\david\PC_AI
context_version: 2.0.0
last_updated: 2025-01-23
session_type: full_documentation
primary_language: PowerShell
secondary_languages: [Markdown]
powershell_versions: [5.1, 7.0+]
platform: Windows 10/11
integrations: [WSL2, Docker, Hyper-V, Ollama]
modules_count: 8
rust_tools_available: 8
rust_tools_missing: [dust, btm]
test_framework: Pester 5.x
coverage_target: 85%
ci_cd: GitHub Actions
safety_mode: read_only_default
native_acceleration: true
native_phase: 3 (Performance Module Complete)
ffi_tests: 62
rust_unit_tests: 59
```

---

## 12. Native Acceleration Framework

> **Full Details**: See `native-acceleration-context.md`

### Architecture: Rust DLL + C# Hybrid Framework

```
Rust DLL (#[no_mangle] extern "C")
    |
    v
C# P/Invoke ([DllImport])
    |
    v
PowerShell Cmdlets (with fallback)
```

### Phase Status

| Phase | Module | Status | FFI Tests | Unit Tests |
|-------|--------|--------|-----------|------------|
| 1 | pcai_core_lib | COMPLETE | 16 | 26 |
| 2 | pcai_search | COMPLETE | 21 | 14 |
| 3 | pcai_performance | COMPLETE | 25 | 19 |
| 4 | pcai_system | PLANNED | - | - |

**Latest Commit**: 9404eb9 "feat(native): add Phase 3 Performance Module with FFI integration"

### Key Native Files

| Category | Path |
|----------|------|
| Rust Workspace | `Native/pcai_core/Cargo.toml` |
| Core Library | `Native/pcai_core/pcai_core_lib/src/lib.rs` |
| Search Module | `Native/pcai_core/pcai_search/src/*.rs` |
| Performance Module | `Native/pcai_core/pcai_performance/src/*.rs` |
| C# Wrappers | `Native/PcaiNative/*.cs` |
| FFI Tests | `Tests/Integration/FFI.*.Tests.ps1` |
| Built DLLs | `bin/*.dll` |

### Key Code Patterns

**Rust FFI**:
```rust
#[repr(C)]
pub struct PcaiStringBuffer {
    pub status: PcaiStatus,
    pub data: *mut c_char,
    pub length: usize,
}

#[no_mangle]
pub extern "C" fn pcai_xxx(path: *const c_char) -> PcaiStringBuffer { ... }
```

**C# P/Invoke**:
```csharp
[DllImport("pcai_xxx.dll", CallingConvention = CallingConvention.Cdecl)]
internal static extern PcaiStringBuffer pcai_xxx(
    [MarshalAs(UnmanagedType.LPUTF8Str)] string path);

// Always free string buffers in try/finally
var buffer = NativeXxx.pcai_xxx(path);
try { return buffer.ToManagedString(); }
finally { NativeCore.pcai_free_string_buffer(ref buffer); }
```

### Build Commands

```powershell
# Rust build
cd Native\pcai_core && cargo build --release

# C# build
cd Native\PcaiNative && dotnet build -c Release

# Run FFI tests
Invoke-Pester -Path 'Tests\Integration\FFI.*.Tests.ps1'
```

---

## 13. Session History

### 2026-01-23: Pester Test Fixing Campaign

**Objective**: Fix failing Pester tests and achieve 95%+ pass rate

**Starting State**: 23% pass rate (50/218 tests)
**Current State**: ~79% pass rate (173/218 tests)

#### Round 1: Network and USB Tests
- **Agents**: test-automator, explore
- **Modules Fixed**: PC-AI.Network (24/24), PC-AI.USB (24/24)
- **Key Fixes**: Module-scoped mocking, ValidatePattern for BusId

#### Round 2: Core Module Tests (Parallel)
- **Agents**: 4 parallel fix agents
- **Modules Fixed**:
  - PC-AI.Hardware (33/33) - CIM cmdlet mocks
  - PC-AI.LLM (28/28) - API endpoint mocking, parameter names
  - PC-AI.Cleanup (26/26) - TestDrive paths, file operation mocks
  - PC-AI.Performance (19/22) - Partial; Optimize-Disks context issues

#### Round 3: Cross-Cutting Fixes
- **Agents**: admin-skip, encoding-fix
- **Patterns Applied**:
  - `[System.IO.File]::WriteAllText()` for PS5.1/PS7+ compatibility
  - `-Skip:(-not $script:IsAdmin)` for elevation-dependent tests

#### Remaining Issues
1. Optimize-Disks context failures (3 tests)
2. CursorPosition console errors in Watch-VSockPerformance
3. PC-AI.Virtualization tests (TBD)
4. Admin tests skipped (~19 tests)

### 2026-01-23 (Later): Native Acceleration Framework - Phase 3 Complete

**Objective**: Build Rust DLL + C# Hybrid Framework for high-performance native operations

**Phases Completed**:
- **Phase 1 (Foundation)**: pcai_core_lib with FFI utilities (16 FFI tests, 26 unit tests)
- **Phase 2 (Search)**: pcai_search with duplicates, files, content search (21 FFI tests, 14 unit tests)
- **Phase 3 (Performance)**: pcai_performance with disk, process, memory (25 FFI tests, 19 unit tests)

**Architecture Pattern**: Follows NukeNul reference implementation
- Rust crates export `#[repr(C)]` structs and `extern "C"` functions
- C# uses P/Invoke with `[DllImport]` and `[StructLayout]`
- PcaiStringBuffer pattern for cross-FFI string handling
- JSON output optimized for LLM consumption

**Key Achievements**:
- 62 FFI integration tests passing (Pester 5.x)
- 59 Rust unit tests passing
- Clean memory management with try/finally pattern
- Full test coverage for all three modules

**Latest Commit**: 9404eb9 "feat(native): add Phase 3 Performance Module with FFI integration"

### 2025-01-23 (Earlier): PowerShell Acceleration Module Complete
- Created PC-AI.Acceleration module with 9 public functions
- Integrated 8 Rust CLI tools (rg, fd, procs, bat, hyperfine, tokei, eza, sd)
- Identified 2 missing tools (dust, btm)
- Fixed 4 bugs in acceleration functions
- Documented PS7+ ForEach-Object -Parallel pattern
- Performance benchmark: ripgrep 44.6x faster than Select-String

### 2025-01-23 (Initial): Project Setup
- Created comprehensive project context
- Documented 3-file architecture pattern
- Identified 12 migration candidates from home directory
- Established safety constraints and design decisions
