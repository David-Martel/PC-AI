<#
.SYNOPSIS
    Integration test runner for PC_AI native modules.

.DESCRIPTION
    Runs Rust unit tests and Pester FFI integration tests.

.PARAMETER RustOnly
    Only run Rust tests

.PARAMETER PesterOnly
    Only run Pester tests

.PARAMETER Verbose
    Show detailed test output

.EXAMPLE
    .\test.ps1
    Run all tests

.EXAMPLE
    .\test.ps1 -RustOnly
    Run only Rust unit tests
#>

[CmdletBinding()]
param(
    [switch]$RustOnly,
    [switch]$PesterOnly
)

$ErrorActionPreference = 'Stop'

$RootDir = $PSScriptRoot
$RustWorkspace = Join-Path $RootDir "pcai_core"
$TestsDir = Join-Path (Split-Path $RootDir) "Tests\Integration"
$BinDir = Join-Path (Split-Path $RootDir) "bin"

$Results = @{
    RustPassed = $false
    RustSkipped = $false
    PesterPassed = $false
    PesterSkipped = $false
}

Write-Host "`n=== PC_AI Native Test Runner ===" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# RUST TESTS
# ============================================================================

if (-not $PesterOnly) {
    Write-Host "[1/2] Running Rust Tests..." -ForegroundColor Magenta

    $buildScript = Join-Path (Split-Path $RootDir) 'Build.ps1'
    if (-not (Test-Path $buildScript)) {
        Write-Host "  [SKIP] Build.ps1 not found at expected path: $buildScript" -ForegroundColor Yellow
        $Results.RustSkipped = $true
    }
    else {
        try {
            $output = & $buildScript -Component inference -RunTests 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  [PASS] Rust tests passed" -ForegroundColor Green
                $Results.RustPassed = $true
            }
            else {
                Write-Host "  [FAIL] Rust tests failed" -ForegroundColor Red
                $output | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
            }
        }
        catch {
            Write-Host "  [FAIL] Error running Rust tests: $_" -ForegroundColor Red
        }
    }
}
else {
    Write-Host "[1/2] Rust Tests [SKIPPED]" -ForegroundColor DarkGray
    $Results.RustSkipped = $true
}

# ============================================================================
# PESTER TESTS
# ============================================================================

if (-not $RustOnly) {
    Write-Host "`n[2/2] Running Pester FFI Tests..." -ForegroundColor Magenta

    $PesterTest = Join-Path $TestsDir "FFI.Core.Tests.ps1"

    if (-not (Test-Path $PesterTest)) {
        Write-Host "  [SKIP] Pester tests not found: $PesterTest" -ForegroundColor Yellow
        $Results.PesterSkipped = $true
    }
    else {
        try {
            # Check if Pester is available
            $pester = Get-Module -ListAvailable -Name Pester | Sort-Object Version -Descending | Select-Object -First 1
            if (-not $pester -or $pester.Version.Major -lt 5) {
                Write-Host "  [SKIP] Pester 5.x required. Install with: Install-Module Pester -Force" -ForegroundColor Yellow
                $Results.PesterSkipped = $true
            }
            else {
                Import-Module Pester -MinimumVersion 5.0 -Force

                $config = New-PesterConfiguration
                $config.Run.Path = $PesterTest
                $config.Output.Verbosity = 'Detailed'
                $config.Run.PassThru = $true

                $result = Invoke-Pester -Configuration $config

                if ($result.FailedCount -eq 0) {
                    Write-Host "  [PASS] Pester tests passed ($($result.PassedCount) tests)" -ForegroundColor Green
                    $Results.PesterPassed = $true
                }
                else {
                    Write-Host "  [FAIL] Pester tests failed ($($result.FailedCount) failures)" -ForegroundColor Red
                }
            }
        }
        catch {
            Write-Host "  [FAIL] Error running Pester tests: $_" -ForegroundColor Red
        }
    }
}
else {
    Write-Host "`n[2/2] Pester Tests [SKIPPED]" -ForegroundColor DarkGray
    $Results.PesterSkipped = $true
}

# ============================================================================
# SUMMARY
# ============================================================================

Write-Host "`n=== Test Summary ===" -ForegroundColor Cyan

$passCount = 0
$failCount = 0
$skipCount = 0

if ($Results.RustSkipped) { $skipCount++ } elseif ($Results.RustPassed) { $passCount++ } else { $failCount++ }
if ($Results.PesterSkipped) { $skipCount++ } elseif ($Results.PesterPassed) { $passCount++ } else { $failCount++ }

Write-Host "  Passed:  $passCount" -ForegroundColor Green
Write-Host "  Failed:  $failCount" -ForegroundColor $(if ($failCount -gt 0) { 'Red' } else { 'DarkGray' })
Write-Host "  Skipped: $skipCount" -ForegroundColor $(if ($skipCount -gt 0) { 'Yellow' } else { 'DarkGray' })
Write-Host ""

if ($failCount -gt 0) {
    exit 1
}
exit 0
