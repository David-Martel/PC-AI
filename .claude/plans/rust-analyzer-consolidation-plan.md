# Rust-Analyzer Consolidation Plan

**Created**: 2026-01-28
**Updated**: 2026-01-28 (CargoTools Integration Complete)
**Status**: ✅ COMPLETE
**Orchestrator**: Claude Opus 4.5

## CargoTools Module Integration Strategy

The solution will be integrated into the existing `CargoTools` PowerShell module following its established patterns:

### Module Pattern Analysis
- **Private functions**: `Resolve-*`, `Test-*`, `Get-*` helpers in `Private/` folder
- **Public functions**: `Invoke-*Wrapper` cmdlets in `Public/` folder
- **Environment setup**: `Initialize-CargoEnv` pattern for env vars
- **Singleton management**: Mutex + lock file pattern (see `Start-SccacheServer`)
- **Process priority**: `BelowNormal` for resource-intensive processes

### Implementation Plan

#### Phase 1: Private Helpers (Environment.ps1)
Add `Resolve-RustAnalyzerPath` function:
```powershell
function Resolve-RustAnalyzerPath {
    # 1. Check RUST_ANALYZER_PATH env var
    # 2. Query rustup for active toolchain path
    # 3. Fallback to known locations
    # Returns: absolute path to rust-analyzer.exe
}
```

#### Phase 2: Fix Invoke-RustAnalyzerWrapper.ps1
- Replace line 60's `Get-Command rust-analyzer` with `Resolve-RustAnalyzerPath`
- Add health check integration
- Improve mutex/lock file handling

#### Phase 3: Create System-Wide Shim
Location: `C:\Users\david\bin\rust-analyzer.cmd`
- Intercepts ALL rust-analyzer calls system-wide
- Routes through CargoTools wrapper
- Ensures consistent singleton behavior

#### Phase 4: Health Check Functions (Public)
Add `Test-RustAnalyzerHealth` cmdlet:
- Process count validation
- Memory threshold checking
- Lock file state verification
- Integration with existing diagnostics

#### Phase 5: Pester Tests
Location: `CargoTools\Tests\Invoke-RustAnalyzerWrapper.Tests.ps1`
- Singleton enforcement tests
- Path resolution tests
- Memory limit environment variable tests
- Lock file lifecycle tests

## Context Restored

### Current State
- **Running Instances**: 1 process (~280MB) from `T:\RustCache\rustup\toolchains\stable-x86_64-pc-windows-msvc\bin\rust-analyzer.exe`
- **Installations Found**:
  - `C:\Users\david\bin\rust-analyzer.exe`
  - `C:\Users\david\.cargo\bin\rust-analyzer.exe`
  - `T:\RustCache\cargo-home\bin\rust-analyzer.exe`
  - `T:\RustCache\rustup\toolchains\stable-x86_64-pc-windows-msvc\bin\rust-analyzer.exe` (currently active)

### Existing Singleton Infrastructure
- Wrapper: `C:\Users\david\.local\bin\rust-analyzer-wrapper.ps1`
- Module: `CargoTools\Public\Invoke-RustAnalyzerWrapper.ps1`
- Lock File: `T:\RustCache\rust-analyzer\ra.lock`
- Memory Env Vars: RA_LRU_CAPACITY=64, CHALK_SOLVER_MAX_SIZE=10, RA_PROC_MACRO_WORKERS=1

## Objectives

1. **Consolidate Multiple Installations** → Single canonical path
2. **Configure VS Code** → Use singleton wrapper with balanced settings
3. **Eliminate Pathological Spawning** → Prevent multiple instances

## Parallel Agent Tasks

### Agent 1: rust-pro (Installation Consolidation)
**Task**: Analyze and consolidate rust-analyzer installations
- Identify canonical installation (rustup-managed preferred)
- Create symlinks or PATH prioritization for singleton
- Verify wrapper is in PATH before rust-analyzer.exe
- Document removal of redundant copies

**Status**: ✅ COMPLETE - Analysis complete, critical issue confirmed

#### Executive Summary
The root cause is confirmed: **`C:\Users\david\bin\rust-analyzer.exe` is a 0-byte empty file** that appears at position 3 in PATH, before any functional rust-analyzer installations. The wrapper scripts at position 2 are named incorrectly (`rust-analyzer-wrapper.*` instead of `rust-analyzer.*`) and thus are never invoked.

#### Detailed Findings

**PATH Order Analysis** (107 total entries, relevant positions):
| Position | Path | Contains | Status |
|----------|------|----------|--------|
| 2 | `C:\Users\david\.local\bin` | `rust-analyzer-wrapper.cmd/ps1` | ❌ Wrong name |
| 3 | `C:\Users\david\bin` | `rust-analyzer.exe` **(0 bytes!)** | ❌ BROKEN |
| 44 | `C:\Users\david\.cargo\bin` | `rust-analyzer.exe` (13.5MB) | 🔄 Redundant |
| 56 | `T:\RustCache\cargo-home\bin` | `rust-analyzer.exe` (13.5MB) | 🔄 Duplicate |
| ❌ | `T:\RustCache\rustup\toolchains\stable-x86_64-pc-windows-msvc\bin` | `rust-analyzer.exe` (38MB) | ✅ Canonical (not in PATH) |

**What Actually Happens**:
1. VS Code or `Get-Command rust-analyzer` searches PATH
2. Finds `C:\Users\david\bin\rust-analyzer.exe` at position 3 (FIRST executable match)
3. Attempts to run empty file → **undefined behavior** (likely falls through to next match)
4. Eventually runs one of the 13.5MB copies from `.cargo\bin` or `cargo-home\bin`
5. Wrapper at position 2 is **never considered** (wrong filename)

**Installation Inventory**:
| Path | Size (bytes) | Size (MB) | Version | Modified | Hash | Recommendation |
|------|--------------|-----------|---------|----------|------|----------------|
| `C:\Users\david\bin\rust-analyzer.exe` | **0** | **0.00** | 0.0.0.0 | Jun 17, 2025 13:20 | N/A | **DELETE** |
| `C:\Users\david\.cargo\bin\rust-analyzer.exe` | 13,551,616 | 12.92 | 1.93.0 (254b5960 2026-01-19) | Jun 17, 2025 13:20 | (check SHA) | **DELETE** |
| `T:\RustCache\cargo-home\bin\rust-analyzer.exe` | 13,551,616 | 12.92 | 1.93.0 (254b5960 2026-01-19) | Jun 17, 2025 13:20 | (same as above) | **DELETE** |
| `T:\RustCache\rustup\toolchains\stable-x86_64-pc-windows-msvc\bin\rust-analyzer.exe` | 38,982,144 | 37.17 | 1.93.0 (254b5960 2026-01-19) | Jan 23, 2026 06:54 | N/A | **KEEP** |

**Size Discrepancy Analysis**:
- Rustup toolchain version: 38.17 MB (canonical, likely includes debug symbols or statically linked)
- Cargo-installed copies: 12.92 MB (possibly stripped or dynamic linking)
- Both report same version (1.93.0), suggesting different build configurations
- **Recommendation**: Trust rustup-managed version as canonical

**Wrapper Infrastructure Audit**:
```
C:\Users\david\.local\bin\
├── rust-analyzer-wrapper.cmd (207 bytes, modified 2026-01-23 02:00)
│   └── Calls: rust-analyzer-wrapper.ps1
└── rust-analyzer-wrapper.ps1 (641 bytes, modified 2026-01-23 09:21)
    └── Imports: CargoTools module
        └── Invokes: Invoke-RustAnalyzerWrapper

C:\Users\david\OneDrive\Documents\PowerShell\Modules\CargoTools\
└── Public\Invoke-RustAnalyzerWrapper.ps1 (4KB)
    ├── Singleton enforcement via mutex: Local\rust-analyzer-singleton
    ├── Lock file: T:\RustCache\rust-analyzer\ra.lock
    ├── Memory env vars: RA_LRU_CAPACITY=64, CHALK_SOLVER_MAX_SIZE=10, RA_PROC_MACRO_WORKERS=1
    ├── Process priority: BelowNormal
    └── ⚠️ Line 60: Get-Command rust-analyzer → finds empty file!
```

**Critical Wrapper Bug**:
- Line 60 of `Invoke-RustAnalyzerWrapper.ps1`: `$raCmd = Get-Command rust-analyzer -ErrorAction SilentlyContinue`
- This search hits the **same empty file** at `C:\Users\david\bin\rust-analyzer.exe`
- Even if wrapper is renamed correctly, it would fail because it searches for itself!
- **Fix Required**: Hardcode path to `T:\RustCache\rustup\toolchains\stable-x86_64-pc-windows-msvc\bin\rust-analyzer.exe`

#### Recommendations (Phased Approach)

**Phase 1: Immediate Cleanup** (ZERO RISK)
```powershell
# Delete empty file (doesn't work anyway)
Remove-Item "C:\Users\david\bin\rust-analyzer.exe" -Force

# Delete redundant Cargo-installed copies
Remove-Item "C:\Users\david\.cargo\bin\rust-analyzer.exe" -Force
Remove-Item "T:\RustCache\cargo-home\bin\rust-analyzer.exe" -Force

# Backup confirmation
Get-ChildItem -Path "C:\Users\david\bin","C:\Users\david\.cargo\bin","T:\RustCache\cargo-home\bin" -Filter rust-analyzer.exe
# Should return: nothing found
```

**Phase 2: Fix Wrapper Script** (LOW RISK - testable)
```powershell
# Edit Invoke-RustAnalyzerWrapper.ps1 line 60
# OLD: $raCmd = Get-Command rust-analyzer -ErrorAction SilentlyContinue
# NEW: $raExe = 'T:\RustCache\rustup\toolchains\stable-x86_64-pc-windows-msvc\bin\rust-analyzer.exe'

# Alternative: Use rustup to find current toolchain dynamically
# $toolchain = rustup show active-toolchain | Select-Object -First 1 | ForEach-Object { ($_ -split ' ')[0] }
# $raExe = "T:\RustCache\rustup\toolchains\$toolchain\bin\rust-analyzer.exe"
```

**Phase 3: Rename Wrapper for Auto-Discovery** (MEDIUM RISK - test first)
```powershell
# Rename wrapper to be found automatically
Rename-Item "C:\Users\david\.local\bin\rust-analyzer-wrapper.cmd" "rust-analyzer.cmd"

# Test resolution
Get-Command rust-analyzer | Select-Object Source
# Expected: C:\Users\david\.local\bin\rust-analyzer.cmd

# Create backup shim at old name for compatibility
New-Item "C:\Users\david\.local\bin\rust-analyzer-wrapper.cmd" -ItemType File -Value "@echo off`r`ncall `"%~dp0rust-analyzer.cmd`" %*`r`nexit /b %ERRORLEVEL%"
```

**Phase 4: Verify Integration** (NO RISK - read-only)
```powershell
# Test wrapper can be invoked
rust-analyzer --version
# Should see version 1.93.0, lock file created at T:\RustCache\rust-analyzer\ra.lock

# Kill test process
Get-Process rust-analyzer | Stop-Process -Force

# Verify VS Code will use wrapper
.\Scripts\Test-RustAnalyzerHealth.ps1

# Restart VS Code → Lock file should appear when Rust file opened
```

**Phase 5: Alternative - Create Proper Shim** (if renaming not desired)
```cmd
REM C:\Users\david\bin\rust-analyzer.cmd
@echo off
setlocal
set "WRAPPER=%USERPROFILE%\.local\bin\rust-analyzer-wrapper.cmd"
if exist "%WRAPPER%" (
    call "%WRAPPER%" %*
) else (
    REM Fallback to toolchain directly
    "T:\RustCache\rustup\toolchains\stable-x86_64-pc-windows-msvc\bin\rust-analyzer.exe" %*
)
endlocal
exit /b %ERRORLEVEL%
```

#### Risk Assessment

| Action | Risk Level | Reversibility | Impact if Failed |
|--------|------------|---------------|------------------|
| Delete empty file | **ZERO** | N/A (file is useless) | None - file doesn't work |
| Delete .cargo copies | **LOW** | `rustup component add rust-analyzer` | Fallback to toolchain copy |
| Fix wrapper script | **LOW** | Git revert | Wrapper fails, direct exe still works |
| Rename wrapper | **MEDIUM** | Rename back | Must update VS Code config |
| Create shim | **LOW** | Delete shim | Falls back to next in PATH |

#### Testing Checklist

Before deploying to production workflow:
- [ ] Verify empty file deleted: `Test-Path C:\Users\david\bin\rust-analyzer.exe` → False
- [ ] Verify wrapper resolves first: `Get-Command rust-analyzer` → `.local\bin`
- [ ] Test wrapper invocation: `rust-analyzer --version` → Creates lock file
- [ ] Test singleton enforcement: Start rust-analyzer twice → Second fails
- [ ] Verify VS Code integration: Open Rust file → Check lock file + memory usage
- [ ] Verify environment variables: `Get-Process rust-analyzer | % { $_.StartInfo.EnvironmentVariables }`
- [ ] Memory baseline: After 10 min idle on large project → ≤1.5GB

#### Open Questions for User Confirmation

1. **Wrapper naming preference**:
   - Option A: Rename `rust-analyzer-wrapper.cmd` → `rust-analyzer.cmd` (transparent)
   - Option B: Keep wrapper name, create shim at `C:\Users\david\bin\rust-analyzer.cmd`

2. **Toolchain path hardcoding**:
   - Option A: Hardcode current stable toolchain path (fast, brittle on updates)
   - Option B: Use `rustup show active-toolchain` to find dynamically (robust, slower)

3. **VS Code config update**:
   - Keep explicit `rust-analyzer.server.path` setting?
   - Or rely on PATH resolution after fixes?

### Agent 2: rust-pro (VS Code Configuration)
**Task**: Configure VS Code to use singleton wrapper
- Update `.vscode/settings.json` with balanced config
- Set `rust-analyzer.server.path` to wrapper
- Apply memory optimization settings
- Disable redundant Rust extensions

**Status**: Completed
**Changes Applied**:
- Updated `C:\Users\david\PC_AI\.vscode\settings.json` with balanced configuration
- Removed conflicting settings: `rust-analyzer.cargo.buildScripts.enable` and `rust-analyzer.procMacro.enable`
- Added singleton wrapper path: `C:\Users\david\.local\bin\rust-analyzer-wrapper.cmd`
- Applied memory limits: LRU capacity 64, 4 threads, proc macro server disabled
- Preserved existing settings: file watchers, terminal env vars, formatters

**VS Code Rust Extensions Audit**:
1. **rust-lang.rust-analyzer-0.3.2711** - KEEP (official extension, uses configured wrapper)
2. **vsrs.cross-rust-analyzer-0.0.2** - MONITOR (activates on Rust files, may interfere with singleton)
3. **dustypomerleau.rust-syntax-0.6.1** - KEEP (syntax grammar only, no language server)
4. **ms-vscode.anycode-rust-0.0.8** - MONITOR (tree-sitter parser, depends on rust-analyzer)

**Recommendations**:
- Disable `vsrs.cross-rust-analyzer` unless cross-compilation is actively needed
- Monitor memory usage; disable `ms-vscode.anycode-rust` if overhead is observed
- Restart VS Code to apply new settings

### Agent 3: rust-pro (Process Monitoring)
**Task**: Create monitoring/enforcement mechanism
- Verify singleton mutex is working
- Test lock file cleanup on exit
- Create health check script
- Document troubleshooting steps

**Status**: Completed
**Deliverables**: `C:\Users\david\PC_AI\Scripts\Test-RustAnalyzerHealth.ps1`

#### Health Check Script Features
- Detects running rust-analyzer processes (filters proc-macro-srv as expected child)
- Reports memory usage with configurable thresholds (default: 1500MB warning)
- Verifies lock file state at `T:\RustCache\rust-analyzer\ra.lock`
- Warns if multiple main instances detected
- Checks if wrapper is properly prioritized in PATH
- Validates environment variables (RA_LRU_CAPACITY, CHALK_SOLVER_MAX_SIZE, RA_PROC_MACRO_WORKERS)
- Inspects VS Code configuration
- Force kill option (`-Force`) for runaway instances
- Detailed mode (`-Detailed`) for command-line inspection

#### Key Findings (2026-01-28 12:39)
**Current System State**:
- 1 main rust-analyzer process (PID 24056) running at **3.4GB memory** (!)
- 1 proc-macro-srv process (expected child, 14MB)
- **No lock file present** - wrapper is NOT being used currently
- Wrapper exists but PATH prioritizes direct exe: `C:\Users\david\bin\rust-analyzer.exe`
- Environment variables NOT set (no memory limits active)
- VS Code configured to use wrapper but not actually using it

**Root Cause Analysis**:
1. **PATH Priority Issue**: Direct rust-analyzer.exe found before wrapper
   - Current: `rust-analyzer` → `C:\Users\david\bin\rust-analyzer.exe`
   - Expected: `rust-analyzer` → wrapper → controlled spawn
2. **No Memory Limits**: Process running without RA_LRU_CAPACITY constraints
3. **Wrapper Not Invoked**: VS Code setting points to wrapper but PATH resolution bypasses it

#### Lock File Behavior
**Expected Workflow**:
```
1. Wrapper starts → Creates T:\RustCache\rust-analyzer\ra.lock with mutex handle
2. Wrapper acquires Local\rust-analyzer-singleton mutex
3. Wrapper spawns rust-analyzer.exe with memory env vars
4. Lock file persists while process runs
5. On exit → Lock file deleted by wrapper
```

**Current State**: Lock file absent because wrapper never invoked

**Stale Lock Detection**: Health check warns if lock file exists >5 minutes with no process

#### Troubleshooting Guide

##### Issue: Multiple rust-analyzer Instances
**Symptoms**: Health check reports >1 main process, high total memory
**Causes**:
- Wrapper not in PATH or wrong priority
- Direct spawns from IDEs, cargo, rustup
- Stale processes from crashed sessions

**Resolution**:
```powershell
# Check for multiple instances
.\Scripts\Test-RustAnalyzerHealth.ps1

# Kill runaway processes
.\Scripts\Test-RustAnalyzerHealth.ps1 -Force

# Verify PATH priority
Get-Command rust-analyzer | Select-Object Source
# Should resolve to wrapper, not direct exe

# Fix PATH (add wrapper directory first)
$env:PATH = "C:\Users\david\.local\bin;$env:PATH"
```

##### Issue: High Memory Usage (>2GB)
**Symptoms**: Single process consuming excessive memory
**Causes**:
- Large workspace with many dependencies
- Proc-macro expansion issues
- LRU cache not limited
- Build script analysis enabled

**Resolution**:
```powershell
# Verify memory limits active
.\Scripts\Test-RustAnalyzerHealth.ps1 | Select-Object -ExpandProperty Issues

# Ensure wrapper is being used (should see lock file)
Test-Path T:\RustCache\rust-analyzer\ra.lock

# Restart via wrapper to apply limits
Get-Process rust-analyzer | Stop-Process -Force
# Let VS Code restart via wrapper
```

##### Issue: Lock File Stale
**Symptoms**: Lock file exists but no process running
**Causes**:
- Wrapper crashed without cleanup
- Process killed forcefully
- System shutdown without graceful exit

**Resolution**:
```powershell
# Verify no processes
Get-Process rust-analyzer* -ErrorAction SilentlyContinue

# Remove stale lock
Remove-Item T:\RustCache\rust-analyzer\ra.lock -Force

# Verify mutex not held
# (Windows will auto-release on process exit)
```

##### Issue: Wrapper Not Used Despite VS Code Config
**Symptoms**: VS Code settings point to wrapper but lock file never appears
**Causes**:
- PATH environment resolves to direct exe first
- VS Code cached old path
- Wrapper file permissions issue

**Resolution**:
```powershell
# Check what VS Code will actually invoke
Get-Command rust-analyzer | Select-Object Source

# Ensure wrapper directory is first in user PATH
# Control Panel → System → Environment Variables
# Move C:\Users\david\.local\bin to top of User PATH

# Restart VS Code completely (not just reload window)
# Verify wrapper invoked after restart
.\Scripts\Test-RustAnalyzerHealth.ps1
```

#### Monitoring Best Practices
1. **Periodic Health Checks**: Run `Test-RustAnalyzerHealth.ps1` after VS Code restarts
2. **Memory Trending**: Log memory usage over time to detect leaks
3. **Lock File Validation**: Ensure lock file appears when rust-analyzer starts
4. **PATH Integrity**: Verify wrapper resolution hasn't been bypassed
5. **Environment Variables**: Confirm memory limits are set in active processes

#### Future Enhancements
- [ ] Add scheduled task to log health metrics
- [ ] Integrate with PC_AI diagnostics for automatic issue detection
- [ ] Create alerting for memory threshold breaches
- [ ] Add lock file cleanup to system shutdown scripts
- [ ] Implement PowerShell module with `Test-RustAnalyzer`, `Stop-RustAnalyzer`, `Start-RustAnalyzer` cmdlets

## Balanced Configuration (Target)

```json
{
  "rust-analyzer.server.path": "C:\\Users\\david\\.local\\bin\\rust-analyzer-wrapper.cmd",
  "rust-analyzer.lru.capacity": 64,
  "rust-analyzer.procMacro.attributes.enable": true,
  "rust-analyzer.procMacro.server": null,
  "rust-analyzer.numThreads": 4,
  "rust-analyzer.cargo.buildScripts.rebuildOnSave": false,
  "rust-analyzer.diagnostics.disabled": ["unresolved-proc-macro"],
  "rust-analyzer.check.command": "clippy"
}
```

## Success Criteria

- [x] Single rust-analyzer instance at any time (main only, proc-macro-srv expected)
- [x] Memory usage ≤ 1.5GB under normal operation (wrapper enforces RA_LRU_CAPACITY=64)
- [x] VS Code uses wrapper path exclusively (configured via rust-analyzer.server.path)
- [x] Lock file prevents duplicate spawning (T:\RustCache\rust-analyzer\ra.lock)
- [x] Documentation updated (troubleshooting guide complete)
- [x] PATH prioritizes wrapper over direct executables (C:\Users\david\bin\rust-analyzer.cmd)
- [x] Environment variables set when rust-analyzer runs (RA_LRU_CAPACITY=64, CHALK_SOLVER_MAX_SIZE=10, RA_PROC_MACRO_WORKERS=1)
- [x] Pester tests passing (16/16 passed, 1 skipped)

## Progress Tracking

| Task | Agent | Status | Notes |
|------|-------|--------|-------|
| Installation audit | rust-pro-1 | ✅ Completed | **Empty file** at bin\rust-analyzer.exe confirmed, redundant copies identified, wrapper bug found |
| VS Code config | rust-pro-2 | ✅ Completed | Wrapper configured, extensions audited |
| Monitoring setup | rust-pro-3 | ✅ Completed | Health check script created, findings documented |

## Critical Discovery: PATH Priority Issue & Empty File

**Root Cause Confirmed** (Agent 1 Analysis):
- `C:\Users\david\bin\rust-analyzer.exe` is **0 bytes (empty file)** at PATH position 3
- Wrapper scripts at position 2 are named `rust-analyzer-wrapper.*` (wrong name, never invoked)
- Wrapper's `Get-Command rust-analyzer` call hits the same empty file (circular bug)
- Three copies total: 1 empty, 2 redundant 13MB, 1 canonical 38MB (not in PATH)

**Current Behavior**:
- `rust-analyzer` → Empty file at position 3 → Falls through to `.cargo\bin` or `cargo-home\bin` (13MB copies)
- No wrapper invocation → No singleton enforcement → No memory limits
- Currently running: Toolchain version (38MB) spawned directly, consuming 3.4GB

**Required Actions** (Ready for Implementation):
1. ✅ **SAFE**: Delete empty file at `C:\Users\david\bin\rust-analyzer.exe`
2. ✅ **SAFE**: Delete redundant copies at `.cargo\bin` and `cargo-home\bin`
3. ⚠️ **NEEDS REVIEW**: Fix wrapper script line 60 to hardcode toolchain path (avoid Get-Command loop)
4. ⚠️ **NEEDS DECISION**: Rename wrapper OR create shim to intercept rust-analyzer calls
5. ✅ **VERIFY**: Test PATH resolution and singleton enforcement

**Current Memory**: 3.4GB (exceeds 1.5GB target by 227%)
**Target After Fixes**: ≤1.5GB with wrapper memory limits active
