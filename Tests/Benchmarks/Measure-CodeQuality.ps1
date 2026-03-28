#Requires -Version 7.0
<#
.SYNOPSIS
    Measures code quality metrics across the PC_AI codebase.

.DESCRIPTION
    Collects lint violations, test coverage, code complexity, and structural
    metrics across Rust, C#, and PowerShell codebases. Outputs a structured
    report for tracking quality over time and detecting regressions.

.PARAMETER Format
    Output format: Table, Json, or Markdown.

.PARAMETER SaveBaseline
    Save results as the code quality baseline for future comparisons.

.PARAMETER BaselinePath
    Path to the baseline file.

.PARAMETER Threshold
    Regression threshold percentage for quality score.

.EXAMPLE
    .\Measure-CodeQuality.ps1 -Format Table
    .\Measure-CodeQuality.ps1 -SaveBaseline
#>
[CmdletBinding()]
param(
    [ValidateSet('Table', 'Json', 'Markdown')][string]$Format = 'Table',
    [switch]$SaveBaseline,
    [string]$BaselinePath = '.pcai/benchmarks/quality-baseline.json',
    [int]$Threshold = 10
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $RepoRoot

try {
    $timestamp = [DateTime]::UtcNow.ToString('o')
    $gitHash = (git rev-parse --short HEAD 2>$null) ?? 'unknown'

    # ── Rust metrics ──────────────────────────────────────────────────────
    Write-Host 'Collecting Rust metrics...' -ForegroundColor Cyan
    $rustClippyWarnings = 0
    $rustTestCount = 0
    $rustLoc = 0
    $rustFmtClean = $false

    $cargoRoot = Join-Path $RepoRoot 'Native' 'pcai_core'
    if (Test-Path $cargoRoot) {
        Push-Location $cargoRoot
        try {
            $env:RUSTC_WRAPPER = ''

            # Clippy warnings
            $clippyOut = & cargo clippy --workspace --all-targets --no-deps --message-format=json `
                -- -D warnings -A clippy::type_complexity 2>&1
            $rustClippyWarnings = @($clippyOut | Where-Object { $_ -match '"level":"warning"' }).Count

            # Format check
            $fmtOut = & cargo fmt --all --check 2>&1
            $rustFmtClean = ($LASTEXITCODE -eq 0)

            # Test count (exclude target/ and vendor/)
            $rustTestCount = @(Get-ChildItem -Path . -Recurse -Filter '*.rs' -File |
                Where-Object { $_.FullName -notmatch '\\target\\|\\vendor\\' } |
                Select-String -Pattern '#\[test\]').Count

            # LOC (exclude target/ and vendor/)
            $rustLoc = @(Get-ChildItem -Path . -Recurse -Filter '*.rs' -File |
                Where-Object { $_.FullName -notmatch 'target|vendor' } |
                ForEach-Object { (Get-Content $_.FullName).Count } |
                Measure-Object -Sum).Sum
        } finally { Pop-Location }
    }

    # ── PowerShell metrics ────────────────────────────────────────────────
    Write-Host 'Collecting PowerShell metrics...' -ForegroundColor Cyan
    $psViolations = 0
    $psTestFiles = 0
    $psLoc = 0
    $psDescribeBlocks = 0

    $modulesPath = Join-Path $RepoRoot 'Modules'
    if (Test-Path $modulesPath) {
        $psFindings = @(Invoke-ScriptAnalyzer -Path $modulesPath -Recurse `
            -Severity Warning, Error -ErrorAction SilentlyContinue)
        $psViolations = $psFindings.Count
    }

    $testPath = Join-Path $RepoRoot 'Tests' 'Unit'
    if (Test-Path $testPath) {
        $psTestFiles = @(Get-ChildItem -Path $testPath -Filter '*.Tests.ps1' -File).Count
        $psDescribeBlocks = @(Get-ChildItem -Path $testPath -Filter '*.Tests.ps1' -File |
            Select-String -Pattern '^\s*Describe\s' -ErrorAction SilentlyContinue).Count
    }

    $psLoc = @(Get-ChildItem -Path $modulesPath -Recurse -Include '*.ps1', '*.psm1' -File -ErrorAction SilentlyContinue |
        ForEach-Object { (Get-Content $_.FullName -ErrorAction SilentlyContinue).Count } |
        Measure-Object -Sum).Sum

    # ── C# metrics ────────────────────────────────────────────────────────
    Write-Host 'Collecting C# metrics...' -ForegroundColor Cyan
    $csharpLoc = 0
    $csharpFiles = 0

    $csharpPath = Join-Path $RepoRoot 'Native' 'PcaiNative'
    if (Test-Path $csharpPath) {
        $csFiles = @(Get-ChildItem -Path $csharpPath -Filter '*.cs' -File)
        $csharpFiles = $csFiles.Count
        $csharpLoc = @($csFiles |
            ForEach-Object { (Get-Content $_.FullName).Count } |
            Measure-Object -Sum).Sum
    }

    # ── Structural metrics ────────────────────────────────────────────────
    Write-Host 'Collecting structural metrics...' -ForegroundColor Cyan
    $todoCount = @(Get-ChildItem -Path $RepoRoot -Recurse -Include '*.rs', '*.ps1', '*.cs' -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch 'target|node_modules|\.pcai|vendor' } |
        Select-String -Pattern 'TODO|FIXME|HACK' -ErrorAction SilentlyContinue).Count

    $unsafeCount = @(Get-ChildItem -Path (Join-Path $RepoRoot 'Native' 'pcai_core') -Recurse -Filter '*.rs' -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch 'target|vendor' } |
        Select-String -Pattern '^\s*unsafe\s' -ErrorAction SilentlyContinue).Count

    $unwrapCount = @(Get-ChildItem -Path (Join-Path $RepoRoot 'Native' 'pcai_core') -Recurse -Filter '*.rs' -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch 'target|vendor|test' } |
        Select-String -Pattern '\.unwrap\(\)' -ErrorAction SilentlyContinue).Count

    # ── Quality score ─────────────────────────────────────────────────────
    # Higher is better. Penalize warnings, violations, unwraps. Reward tests.
    $totalLoc = $rustLoc + $psLoc + $csharpLoc
    $totalTests = $rustTestCount + $psDescribeBlocks
    $totalIssues = $rustClippyWarnings + $psViolations + $unwrapCount + $todoCount

    $testDensity = if ($totalLoc -gt 0) { [math]::Round($totalTests / ($totalLoc / 1000), 2) } else { 0 }
    $issueDensity = if ($totalLoc -gt 0) { [math]::Round($totalIssues / ($totalLoc / 1000), 2) } else { 0 }

    # Quality score: weighted composite (0-100)
    # Components: lint cleanliness (40%), test coverage density (30%), code hygiene (30%)
    $lintScore = [math]::Max(0, 100 - ($rustClippyWarnings * 5) - [math]::Min(50, $psViolations * 0.05))
    $testScore = [math]::Min(100, $testDensity * 20)  # 5 tests/KLOC = 100
    $hygieneScore = [math]::Max(0, 100 - ($unwrapCount * 0.3) - ($todoCount * 0.5) - [math]::Min(30, $unsafeCount * 0.02))
    $qualityScore = [math]::Round($lintScore * 0.4 + $testScore * 0.3 + $hygieneScore * 0.3, 1)

    # ── Build result ──────────────────────────────────────────────────────
    $result = [ordered]@{
        timestamp        = $timestamp
        git_hash         = $gitHash
        quality_score    = [math]::Round($qualityScore, 1)
        test_density     = $testDensity
        issue_density    = $issueDensity
        rust             = [ordered]@{
            clippy_warnings = $rustClippyWarnings
            fmt_clean       = $rustFmtClean
            test_count      = $rustTestCount
            loc             = $rustLoc
            unsafe_blocks   = $unsafeCount
            unwrap_calls    = $unwrapCount
        }
        powershell       = [ordered]@{
            analyzer_violations = $psViolations
            test_files          = $psTestFiles
            describe_blocks     = $psDescribeBlocks
            loc                 = $psLoc
        }
        csharp           = [ordered]@{
            files = $csharpFiles
            loc   = $csharpLoc
        }
        structural       = [ordered]@{
            todo_fixme_hack = $todoCount
            total_loc       = $totalLoc
            total_tests     = $totalTests
            total_issues    = $totalIssues
        }
    }

    # ── Save baseline ─────────────────────────────────────────────────────
    if ($SaveBaseline) {
        $baseDir = Split-Path $BaselinePath -Parent
        if (-not (Test-Path $baseDir)) { New-Item -ItemType Directory -Path $baseDir -Force | Out-Null }
        $result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $BaselinePath -Encoding utf8
        Write-Host "`nQuality baseline saved: $BaselinePath" -ForegroundColor Green
    }

    # ── Compare with baseline ─────────────────────────────────────────────
    $comparison = $null
    if (-not $SaveBaseline -and (Test-Path $BaselinePath)) {
        $baseline = Get-Content -LiteralPath $BaselinePath -Raw | ConvertFrom-Json
        $comparison = [ordered]@{
            score_delta         = [math]::Round($qualityScore - $baseline.quality_score, 1)
            clippy_delta        = $rustClippyWarnings - $baseline.rust.clippy_warnings
            violations_delta    = $psViolations - $baseline.powershell.analyzer_violations
            tests_delta         = $totalTests - $baseline.structural.total_tests
            issues_delta        = $totalIssues - $baseline.structural.total_issues
            unwrap_delta        = $unwrapCount - $baseline.rust.unwrap_calls
        }
    }

    # ── Output ────────────────────────────────────────────────────────────
    switch ($Format) {
        'Json' {
            $output = [ordered]@{ metrics = $result }
            if ($comparison) { $output.comparison = $comparison }
            $output | ConvertTo-Json -Depth 5
        }
        'Markdown' {
            "## Code Quality Report ($gitHash)`n"
            "| Metric | Value |"
            "|--------|-------|"
            "| Quality Score | $($result.quality_score)/100 |"
            "| Test Density | $testDensity tests/KLOC |"
            "| Issue Density | $issueDensity issues/KLOC |"
            "| Rust Clippy Warnings | $rustClippyWarnings |"
            "| Rust Tests | $rustTestCount |"
            "| Rust LOC | $rustLoc |"
            "| PS Violations | $psViolations |"
            "| PS Test Files | $psTestFiles |"
            "| PS Describe Blocks | $psDescribeBlocks |"
            "| PS LOC | $psLoc |"
            "| C# Files | $csharpFiles |"
            "| C# LOC | $csharpLoc |"
            "| TODO/FIXME/HACK | $todoCount |"
            "| Unsafe Blocks | $unsafeCount |"
            "| Unwrap Calls | $unwrapCount |"
            if ($comparison) {
                "`n### Comparison vs Baseline"
                "| Metric | Delta | Status |"
                "|--------|-------|--------|"
                $comparison.GetEnumerator() | ForEach-Object {
                    $status = if ($_.Value -lt 0 -and $_.Key -match 'issues|clippy|violations|unwrap') { 'improved' }
                              elseif ($_.Value -gt 0 -and $_.Key -match 'tests|score') { 'improved' }
                              elseif ($_.Value -eq 0) { 'unchanged' }
                              else { 'regressed' }
                    "| $($_.Key) | $($_.Value) | $status |"
                }
            }
        }
        default {
            Write-Host "`n=== Code Quality Report ($gitHash) ===" -ForegroundColor Cyan
            Write-Host "Quality Score:    $($result.quality_score)/100" -ForegroundColor $(if ($qualityScore -ge 80) { 'Green' } elseif ($qualityScore -ge 60) { 'Yellow' } else { 'Red' })
            Write-Host "Test Density:     $testDensity tests/KLOC"
            Write-Host "Issue Density:    $issueDensity issues/KLOC"
            Write-Host ''
            Write-Host 'Rust:'  -ForegroundColor White
            Write-Host "  Clippy warnings:  $rustClippyWarnings"
            Write-Host "  Format clean:     $rustFmtClean"
            Write-Host "  Tests (#[test]):  $rustTestCount"
            Write-Host "  LOC:              $rustLoc"
            Write-Host "  Unsafe blocks:    $unsafeCount"
            Write-Host "  Unwrap calls:     $unwrapCount"
            Write-Host ''
            Write-Host 'PowerShell:' -ForegroundColor White
            Write-Host "  PSA violations:   $psViolations"
            Write-Host "  Test files:       $psTestFiles"
            Write-Host "  Describe blocks:  $psDescribeBlocks"
            Write-Host "  LOC:              $psLoc"
            Write-Host ''
            Write-Host 'C#:' -ForegroundColor White
            Write-Host "  Files:            $csharpFiles"
            Write-Host "  LOC:              $csharpLoc"
            Write-Host ''
            Write-Host 'Structural:' -ForegroundColor White
            Write-Host "  TODO/FIXME/HACK:  $todoCount"
            Write-Host "  Total LOC:        $totalLoc"
            Write-Host "  Total Tests:      $totalTests"

            if ($comparison) {
                Write-Host "`n--- Comparison vs Baseline ---" -ForegroundColor Yellow
                $comparison.GetEnumerator() | ForEach-Object {
                    $color = if ($_.Value -eq 0) { 'Gray' }
                             elseif (($_.Value -lt 0 -and $_.Key -match 'issues|clippy|violations|unwrap') -or
                                     ($_.Value -gt 0 -and $_.Key -match 'tests|score')) { 'Green' }
                             else { 'Red' }
                    Write-Host "  $($_.Key): $($_.Value)" -ForegroundColor $color
                }
            }
        }
    }
} finally {
    Pop-Location
}
