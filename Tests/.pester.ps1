<#
.SYNOPSIS
    Pester 5.x test runner for PC_AI testing framework

.DESCRIPTION
    Executes unit and integration tests with flexible configuration options.
    Supports coverage analysis, CI mode, and selective test execution.

.PARAMETER Type
    Test type to run: Unit, Integration, or All (default: All)

.PARAMETER Coverage
    Enable code coverage analysis (85% target)

.PARAMETER CI
    Enable CI mode: exit codes, XML output, Normal verbosity

.PARAMETER Tag
    Run only tests with specific tags

.PARAMETER ExcludeTag
    Exclude tests with specific tags

.EXAMPLE
    .\Tests\.pester.ps1 -Type Unit
    Run fast unit tests only

.EXAMPLE
    .\Tests\.pester.ps1 -Type All -Coverage
    Run full suite with coverage analysis

.EXAMPLE
    .\Tests\.pester.ps1 -CI
    Run in CI mode with XML output
#>

[CmdletBinding()]
param(
    [Parameter()]
    [ValidateSet('Unit', 'Integration', 'All')]
    [string]$Type = 'All',

    [Parameter()]
    [switch]$Coverage,

    [Parameter()]
    [switch]$CI,

    [Parameter()]
    [string[]]$Tag,

    [Parameter()]
    [string[]]$ExcludeTag
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Tests/PesterConfiguration.psd1 declares every path relative to the REPO ROOT
# (Run.Path = 'Tests/Unit', CodeCoverage.Path = './Modules/**'), because that is
# the working directory ci.yml invokes Invoke-Pester from. This runner must use
# the same cwd or coverage resolution dies with
# "Cannot find path '<repo>\Tests\Modules' because it does not exist".
$TestRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $TestRoot
Push-Location $RepoRoot

try {
    # Check Pester version
    $pesterModule = Get-Module -Name Pester -ListAvailable | Sort-Object Version -Descending | Select-Object -First 1

    if (-not $pesterModule) {
        Write-Error "Pester module not found. Install with: Install-Module -Name Pester -MinimumVersion 5.0.0 -Force"
        exit 1
    }

    if ($pesterModule.Version.Major -lt 5) {
        Write-Error "Pester 5.x required. Current version: $($pesterModule.Version). Update with: Install-Module -Name Pester -Force"
        exit 1
    }

    Import-Module Pester -MinimumVersion 5.0.0 -ErrorAction Stop
    Write-Host "Using Pester version: $($pesterModule.Version)" -ForegroundColor Cyan

    # Load base configuration
    $configPath = Join-Path $TestRoot 'PesterConfiguration.psd1'
    $config = New-PesterConfiguration -Hashtable (Import-PowerShellDataFile $configPath)

    # Configure paths based on test type
    switch ($Type) {
        'Unit' {
            $config.Run.Path = @('Tests/Unit')
            Write-Host "Running unit tests only" -ForegroundColor Yellow
        }
        'Integration' {
            $config.Run.Path = @('Tests/Integration')
            Write-Host "Running integration tests only" -ForegroundColor Yellow
        }
        'All' {
            $config.Run.Path = @('Tests/Unit', 'Tests/Integration')
            Write-Host "Running all tests" -ForegroundColor Yellow
        }
    }

    # Configure coverage. The base configuration enables it, so -Coverage is an
    # explicit opt-in here and its absence must actually turn coverage off --
    # otherwise every plain run pays the instrumentation cost and the -Coverage
    # switch is decorative.
    $config.CodeCoverage.Enabled = [bool]$Coverage
    if ($Coverage) {
        # OutputPath is repo-root relative; Pester will not create the parent.
        $coverageDir = Split-Path -Parent $config.CodeCoverage.OutputPath.Value
        if ($coverageDir -and -not (Test-Path $coverageDir)) {
            New-Item -ItemType Directory -Path $coverageDir -Force | Out-Null
        }
        Write-Host "Code coverage enabled (target: 85%)" -ForegroundColor Green
    }

    # Configure CI mode
    if ($CI) {
        $config.Run.Exit = $true
        $config.TestResult.Enabled = $true
        $config.Output.Verbosity = 'Normal'
        Write-Host "CI mode enabled: exit codes + XML output" -ForegroundColor Cyan
    }

    # Configure tags
    if ($Tag) {
        $config.Filter.Tag = $Tag
        Write-Host "Filtering by tags: $($Tag -join ', ')" -ForegroundColor Magenta
    }

    if ($ExcludeTag) {
        $config.Filter.ExcludeTag = $ExcludeTag
        Write-Host "Excluding tags: $($ExcludeTag -join ', ')" -ForegroundColor Magenta
    }

    # Run tests
    Write-Host "`nStarting test execution..." -ForegroundColor Cyan
    Write-Host ("=" * 80) -ForegroundColor Gray

    $result = Invoke-Pester -Configuration $config

    # Report results
    Write-Host "`n" -NoNewline
    Write-Host ("=" * 80) -ForegroundColor Gray
    Write-Host "`nTest Results:" -ForegroundColor Cyan
    Write-Host "  Total:    $($result.TotalCount)" -ForegroundColor White
    Write-Host "  Passed:   $($result.PassedCount)" -ForegroundColor Green
    Write-Host "  Failed:   $($result.FailedCount)" -ForegroundColor $(if ($result.FailedCount -eq 0) { 'Green' } else { 'Red' })
    Write-Host "  Skipped:  $($result.SkippedCount)" -ForegroundColor Yellow
    Write-Host "  Duration: $($result.Duration.TotalSeconds)s" -ForegroundColor White

    # Coverage report
    if ($Coverage -and $result.CodeCoverage) {
        $coveredPercent = [math]::Round(($result.CodeCoverage.CommandsExecutedCount / $result.CodeCoverage.CommandsAnalyzedCount) * 100, 2)
        $coverageColor = if ($coveredPercent -ge 85) { 'Green' } else { 'Yellow' }

        Write-Host "`nCode Coverage:" -ForegroundColor Cyan
        Write-Host "  Commands Analyzed: $($result.CodeCoverage.CommandsAnalyzedCount)" -ForegroundColor White
        Write-Host "  Commands Executed: $($result.CodeCoverage.CommandsExecutedCount)" -ForegroundColor White
        Write-Host "  Coverage:          $coveredPercent%" -ForegroundColor $coverageColor
        Write-Host "  Target:            85%" -ForegroundColor White

        if ($coveredPercent -lt 85) {
            Write-Warning "Coverage below target (85%). Current: $coveredPercent%"
        }

        # Show missed commands
        if ($result.CodeCoverage.MissedCommands.Count -gt 0) {
            Write-Host "`nTop Missed Commands:" -ForegroundColor Yellow
            $result.CodeCoverage.MissedCommands |
                Group-Object File |
                Sort-Object Count -Descending |
                Select-Object -First 5 |
                ForEach-Object {
                    $fileName = Split-Path $_.Name -Leaf
                    Write-Host "  ${fileName}: $($_.Count) missed" -ForegroundColor Gray
                }
        }

        Write-Host "`nCoverage report: $($config.CodeCoverage.OutputPath.Value)" -ForegroundColor Cyan
    }

    # Output file locations
    if ($config.TestResult.Enabled.Value) {
        Write-Host "`nTest results: $($config.TestResult.OutputPath.Value)" -ForegroundColor Cyan
    }

    # Exit with appropriate code
    if ($result.FailedCount -gt 0) {
        Write-Host "`nTests FAILED" -ForegroundColor Red
        if ($CI) {
            exit 1
        }
    } else {
        Write-Host "`nAll tests PASSED" -ForegroundColor Green
        if ($CI) {
            exit 0
        }
    }

} catch {
    Write-Error "Test execution failed: $_"
    if ($CI) {
        exit 1
    }
} finally {
    Pop-Location
}
