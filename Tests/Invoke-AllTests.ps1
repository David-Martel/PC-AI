#Requires -Version 7.0
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0.0' }

<#
.SYNOPSIS
    Comprehensive test runner for PCAI Inference tests.

.DESCRIPTION
    Discovers and runs all test files in the Tests directory with proper filtering
    and reporting. Supports running specific test suites or tags.

.PARAMETER Suite
    Test suite to run: Unit, Integration, E2E, Functional, or All (default).

.PARAMETER Tag
    Specific Pester tags to filter tests by (e.g., 'FFI', 'Backend', 'Model').

.PARAMETER ExcludeTag
    Pester tags to exclude from test runs.

.PARAMETER SkipSlow
    Skip slow tests (excludes Performance and Model tags).

.PARAMETER CodeCoverage
    Enable code coverage analysis for PowerShell modules.

.PARAMETER OutputFormat
    Output format: NUnitXml, JUnitXml, or None (console only).

.PARAMETER OutputFile
    Path to save test results (default: test-results.xml).

.EXAMPLE
    .\Tests\Invoke-AllTests.ps1
    Run all tests

.EXAMPLE
    .\Tests\Invoke-AllTests.ps1 -Suite Integration
    Run only integration tests

.EXAMPLE
    .\Tests\Invoke-AllTests.ps1 -Tag FFI,Backend
    Run tests with FFI or Backend tags

.EXAMPLE
    .\Tests\Invoke-AllTests.ps1 -SkipSlow
    Run all tests except slow performance tests

.EXAMPLE
    .\Tests\Invoke-AllTests.ps1 -CodeCoverage -OutputFormat NUnitXml
    Run all tests with code coverage and NUnit output
#>

param(
    [Parameter(Mandatory = $false)]
    [ValidateSet('All', 'Unit', 'Integration', 'E2E', 'Functional', 'Benchmarks')]
    [string]$Suite = 'All',

    [Parameter(Mandatory = $false)]
    [string[]]$Tag,

    [Parameter(Mandatory = $false)]
    [string[]]$ExcludeTag,

    [Parameter(Mandatory = $false)]
    [switch]$SkipSlow,

    [Parameter(Mandatory = $false)]
    [switch]$CodeCoverage,

    [Parameter(Mandatory = $false)]
    [ValidateSet('None', 'NUnitXml', 'JUnitXml')]
    [string]$OutputFormat = 'None',

    [Parameter(Mandatory = $false)]
    [string]$OutputFile = 'test-results.xml'
)

$ErrorActionPreference = 'Stop'

# Get project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TestsRoot = Join-Path $ProjectRoot "Tests"

if ($Suite -eq 'Benchmarks') {
    $benchmarkScript = Join-Path $TestsRoot 'Benchmarks\Invoke-PcaiToolingBenchmarks.ps1'
    if (-not (Test-Path $benchmarkScript)) {
        Write-Warning "Benchmark runner not found: $benchmarkScript"
        exit 1
    }

    Write-Host "=== PCAI Tooling Benchmark Runner ===" -ForegroundColor Cyan
    $benchmarkResult = & $benchmarkScript -PassThru
    Write-Host "Benchmark report: $($benchmarkResult.MarkdownPath)" -ForegroundColor Green
    exit 0
}

Write-Host "=== PCAI Inference Test Runner ===" -ForegroundColor Cyan
Write-Host "Project Root: $ProjectRoot" -ForegroundColor Gray
Write-Host "Test Suite: $Suite" -ForegroundColor Gray
Write-Host ""

# Build test path based on suite
$testPaths = @()
switch ($Suite) {
    'Unit' { $testPaths += Join-Path $TestsRoot "Unit\*.Tests.ps1" }
    'Integration' { $testPaths += Join-Path $TestsRoot "Integration\*.Tests.ps1" }
    'E2E' { $testPaths += Join-Path $TestsRoot "E2E\*.Tests.ps1" }
    'Functional' { $testPaths += Join-Path $TestsRoot "Functional\*.Tests.ps1" }
    'Benchmarks' { $testPaths += Join-Path $TestsRoot "Benchmarks\*.Tests.ps1" }
    'All' {
        $testPaths += Join-Path $TestsRoot "Unit\*.Tests.ps1"
        $testPaths += Join-Path $TestsRoot "Integration\*.Tests.ps1"
        $testPaths += Join-Path $TestsRoot "E2E\*.Tests.ps1"
        $testPaths += Join-Path $TestsRoot "Functional\*.Tests.ps1"
    }
}

# Discover test files
$testFiles = @()
foreach ($path in $testPaths) {
    $files = Get-ChildItem -Path $path -ErrorAction SilentlyContinue
    if ($files) {
        $testFiles += $files
    }
}

if ($testFiles.Count -eq 0) {
    Write-Warning "No test files found for suite: $Suite"
    exit 0
}

Write-Host "Found $($testFiles.Count) test file(s):" -ForegroundColor Green
foreach ($file in $testFiles) {
    Write-Host "  - $($file.Name)" -ForegroundColor Gray
}
Write-Host ""

# Build Pester configuration
$pesterConfig = New-PesterConfiguration

# Set test paths
$pesterConfig.Run.Path = $testFiles.FullName

# Set output options
$pesterConfig.Output.Verbosity = 'Detailed'

# Apply tag filters
if ($Tag) {
    $pesterConfig.Filter.Tag = $Tag
    Write-Host "Filtering by tags: $($Tag -join ', ')" -ForegroundColor Yellow
}

if ($ExcludeTag) {
    $pesterConfig.Filter.ExcludeTag = $ExcludeTag
    Write-Host "Excluding tags: $($ExcludeTag -join ', ')" -ForegroundColor Yellow
}

if ($SkipSlow) {
    $excludeTags = @('Performance', 'Model', 'Slow')
    if ($ExcludeTag) {
        $excludeTags += $ExcludeTag
    }
    $pesterConfig.Filter.ExcludeTag = $excludeTags
    Write-Host "Skipping slow tests (excluding: $($excludeTags -join ', '))" -ForegroundColor Yellow
}

# Set test results output
if ($OutputFormat -ne 'None') {
    $outputPath = if ([System.IO.Path]::IsPathRooted($OutputFile)) {
        $OutputFile
    } else {
        Join-Path $TestsRoot $OutputFile
    }

    switch ($OutputFormat) {
        'NUnitXml' {
            $pesterConfig.TestResult.Enabled = $true
            $pesterConfig.TestResult.OutputFormat = 'NUnitXml'
            $pesterConfig.TestResult.OutputPath = $outputPath
        }
        'JUnitXml' {
            $pesterConfig.TestResult.Enabled = $true
            $pesterConfig.TestResult.OutputFormat = 'JUnitXml'
            $pesterConfig.TestResult.OutputPath = $outputPath
        }
    }
    Write-Host "Test results will be saved to: $outputPath" -ForegroundColor Yellow
    Write-Host ""
}

# Configure code coverage
if ($CodeCoverage) {
    $modulesToCover = @(
        (Join-Path $ProjectRoot "Modules\PcaiInference.psm1")
    )

    $existingModules = $modulesToCover | Where-Object { Test-Path $_ }
    if ($existingModules.Count -gt 0) {
        $pesterConfig.CodeCoverage.Enabled = $true
        $pesterConfig.CodeCoverage.Path = $existingModules
        $pesterConfig.CodeCoverage.OutputPath = Join-Path $TestsRoot "coverage.xml"

        Write-Host "Code coverage enabled for:" -ForegroundColor Yellow
        foreach ($module in $existingModules) {
            Write-Host "  - $(Split-Path -Leaf $module)" -ForegroundColor Gray
        }
        Write-Host ""
    } else {
        Write-Warning "No modules found for code coverage analysis"
    }
}

# Run tests
Write-Host "=== Running Tests ===" -ForegroundColor Cyan
Write-Host ""

$result = Invoke-Pester -Configuration $pesterConfig

# Print summary
Write-Host ""
Write-Host "=== Test Summary ===" -ForegroundColor Cyan
Write-Host "Total Tests: $($result.TotalCount)" -ForegroundColor Gray
Write-Host "Passed:      $($result.PassedCount)" -ForegroundColor Green
Write-Host "Failed:      $($result.FailedCount)" -ForegroundColor $(if ($result.FailedCount -gt 0) { 'Red' } else { 'Gray' })
Write-Host "Skipped:     $($result.SkippedCount)" -ForegroundColor Yellow
Write-Host "Inconclusive: $($result.NotRunCount)" -ForegroundColor Yellow

if ($CodeCoverage -and $result.CodeCoverage) {
    $coverage = $result.CodeCoverage
    $coveredCommands = $coverage.CommandsExecutedCount
    $totalCommands = $coverage.CommandsAnalyzedCount
    $coveragePercent = if ($totalCommands -gt 0) {
        ($coveredCommands / $totalCommands * 100).ToString('F2')
    } else { '0.00' }

    Write-Host ""
    Write-Host "Code Coverage: $coveragePercent% ($coveredCommands/$totalCommands commands)" -ForegroundColor Cyan
}

# Exit code based on test results
if ($result.FailedCount -gt 0) {
    Write-Host ""
    Write-Host "Tests FAILED" -ForegroundColor Red
    exit 1
} else {
    Write-Host ""
    Write-Host "Tests PASSED" -ForegroundColor Green
    exit 0
}
