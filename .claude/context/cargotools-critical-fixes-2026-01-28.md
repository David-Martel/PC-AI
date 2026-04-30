# CargoTools Critical Fixes - 2026-01-28

## Summary

Applied critical fixes to CargoTools module based on agent review findings.

## Fixes Applied

### 1. Mutex Acquisition Bug (Critical)
**File**: `Public/Invoke-RustAnalyzerWrapper.ps1`

**Problem**: Mutex was created but never actually acquired - `New-Object Mutex($false, ...)` creates the mutex without taking ownership, and `WaitOne()` was never called.

**Fix**: Added proper mutex acquisition with timeout:
```powershell
$mutexAcquired = $mutex.WaitOne(100)
if (-not $mutexAcquired) {
    # Handle contention or force
}
```

Also added handling for `AbandonedMutexException` when previous holder crashes.

### 2. TOCTOU Race Condition (Critical)
**File**: `Public/Invoke-RustAnalyzerWrapper.ps1`

**Problem**: Lock file check (`Test-Path`) and creation (`Set-Content`) were not atomic, allowing race conditions.

**Fix**: Replaced with atomic file creation using `[System.IO.File]::Open()` with `FileMode::CreateNew` and exclusive access:
```powershell
$lockStream = [System.IO.File]::Open(
    $LockFile,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::None
)
```

### 3. Hardcoded Paths Removed (High)
**Files**: Multiple

**Problem**: Hardcoded paths like `T:\RustCache\...` and `C:\Users\david\...` reduced portability.

**Fix**: All paths now use dynamic resolution via `Resolve-CacheRoot` and `$env:USERPROFILE`:
- `Invoke-RustAnalyzerWrapper.ps1`: Lock file, cache directories
- `Environment.ps1`: rustup paths, known locations
- `Test-RustAnalyzerHealth.ps1`: Lock file, shim path, recommendations

## Test Results

All 16 passing tests continue to pass after fixes:
- Resolve-RustAnalyzerPath: 4 tests
- Test-RustAnalyzerSingleton: 3 tests
- Get-RustAnalyzerMemoryMB: 2 tests
- Invoke-RustAnalyzerWrapper: 4 tests
- System-wide shim: 2 tests
- Integration Tests: 1 test (1 skipped)

## Files Modified (in OneDrive)

1. `C:\Users\david\OneDrive\Documents\PowerShell\Modules\CargoTools\Public\Invoke-RustAnalyzerWrapper.ps1`
2. `C:\Users\david\OneDrive\Documents\PowerShell\Modules\CargoTools\Private\Environment.ps1`
3. `C:\Users\david\OneDrive\Documents\PowerShell\Modules\CargoTools\Public\Test-RustAnalyzerHealth.ps1`

## Remaining Work

- Increase test coverage from ~20% to 85%
- Add ShouldProcess support for -Force operations
- Consider splitting Environment.ps1 (currently has 7 concerns)
- Add integration tests for full singleton workflow
